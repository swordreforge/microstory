// Standalone C++ inference for MacroStories (Ouro 20k architecture).
// Reads model.safetensors + tokenizer.json directly, no ML framework.
//
// Generation uses a per-recurrent-step KV cache by default (token-at-a-time,
// O(pos) per step). Pass --no-kv to fall back to full-sequence recompute.
// `--selftest-kv` compares cached vs full logits on a 257-token sequence.
//
// SIMD: scalar reference + AVX2 + AVX-512/FMA kernels, selectable with
// `--simd avx512|avx2|scalar` (or `--no-simd`). `--selftest-simd` checks
// each compiled level against the scalar reference.
// Batch: `--threads N` runs N independent sequences concurrently;
// `--batch-gemm B` decodes B sequences in one lockstep forward (weight-row
// reuse), combinable as `--threads T --batch-gemm B`. `--selftest-batch`
// compares batched output against per-sequence output.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

static constexpr int VOCAB = 378;
static constexpr int HID = 32;
static constexpr int INTER = 48;
static constexpr int NH = 2;
static constexpr int NKV = 1;
static constexpr int HD = 16;
static constexpr int STEPS = 4;
static constexpr float MIX = 0.5f;
static constexpr float EPS = 1e-6f;
static constexpr float THETA = 10000.0f;
static constexpr int BOS = 1;
static constexpr int EOS = 2;

static std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    f.seekg(0, std::ios::end);
    std::streamoff n = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(static_cast<size_t>(n));
    f.read(buf.data(), n);
    return buf;
}

struct Tensor { const float* data; int numel; };

struct Weights {
    Tensor embed, q, k, v, o, gate, up, down;
    Tensor ln1, ln1b, ln2, ln2b, norm;
};

static std::vector<float> g_storage;

static Tensor find_tensor(const std::vector<char>& file, const std::string& header,
                          const std::string& name) {
    std::string key = "\"" + name + "\"";
    size_t p = header.find(key);
    if (p == std::string::npos) { std::fprintf(stderr, "missing tensor %s\n", name.c_str()); std::exit(1); }
    size_t d = header.find("\"data_offsets\"", p);
    size_t lb = header.find('[', d);
    size_t c1 = header.find(',', lb);
    size_t c2 = header.find(']', c1);
    long start = std::stol(header.substr(lb + 1, c1 - lb - 1));
    long end = std::stol(header.substr(c1 + 1, c2 - c1 - 1));
    int numel = static_cast<int>((end - start) / 4);
    size_t data_base = 8 + header.size();
    const float* src = reinterpret_cast<const float*>(file.data() + data_base + start);
    size_t off = g_storage.size();
    g_storage.resize(off + numel);
    std::memcpy(g_storage.data() + off, src, sizeof(float) * numel);
    return Tensor{g_storage.data() + off, numel};
}

static Weights load_weights(const std::string& path) {
    std::vector<char> file = read_file(path);
    uint64_t hlen;
    std::memcpy(&hlen, file.data(), 8);
    std::string header(file.data() + 8, hlen);
    g_storage.reserve(file.size() / sizeof(float) + 64);
    Weights w;
    w.embed = find_tensor(file, header, "model.embed_tokens.weight");
    const std::string p = "model.layers.0.";
    w.q = find_tensor(file, header, p + "self_attn.q_proj.weight");
    w.k = find_tensor(file, header, p + "self_attn.k_proj.weight");
    w.v = find_tensor(file, header, p + "self_attn.v_proj.weight");
    w.o = find_tensor(file, header, p + "self_attn.o_proj.weight");
    w.gate = find_tensor(file, header, p + "mlp.gate_proj.weight");
    w.up = find_tensor(file, header, p + "mlp.up_proj.weight");
    w.down = find_tensor(file, header, p + "mlp.down_proj.weight");
    w.ln1 = find_tensor(file, header, p + "input_layernorm.weight");
    w.ln1b = find_tensor(file, header, p + "input_layernorm_2.weight");
    w.ln2 = find_tensor(file, header, p + "post_attention_layernorm.weight");
    w.ln2b = find_tensor(file, header, p + "post_attention_layernorm_2.weight");
    w.norm = find_tensor(file, header, "model.norm.weight");
    return w;
}

// ---- kernels: scalar reference + AVX2/AVX-512 FMA, runtime selectable ------
// g_simd: 0 = scalar, 2 = AVX2, 5 = AVX-512, 6 = hybrid (AVX2 for n<=16,
// AVX-512 otherwise). Defaults to AVX2 even when AVX-512 is compiled:
// on the tested Xeon Platinum, sustained AVX-512 downclocks the core and
// measured ~17% slower than AVX2 for this model. Opt in with
// `--simd avx512` or `--simd hybrid` on hardware where it actually wins.
#if defined(__AVX2__)
static int g_simd = 2;
#else
static int g_simd = 0;
#endif

static const char* simd_name(int s) {
    if (s >= 6) return "hybrid";
    if (s >= 5) return "avx512";
    return s >= 2 ? "avx2" : "scalar";
}

#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize")
static void matvec_scalar(const Tensor& t, int out_dim, int in_dim, const float* x, float* y) {
    for (int i = 0; i < out_dim; ++i) {
        const float* row = t.data + static_cast<size_t>(i) * in_dim;
        float acc = 0.0f;
        for (int j = 0; j < in_dim; ++j) acc += row[j] * x[j];
        y[i] = acc;
    }
}
static void rmsnorm_scalar(const Tensor& w, const float* x, float* y) {
    float ss = 0.0f;
    for (int i = 0; i < HID; ++i) ss += x[i] * x[i];
    float inv = 1.0f / std::sqrt(ss / HID + EPS);
    for (int i = 0; i < HID; ++i) y[i] = w.data[i] * x[i] * inv;
}
static void silu_mul_scalar(const float* g, const float* u, float* out) {
    for (int i = 0; i < INTER; ++i) {
        float x = g[i];
        out[i] = (x / (1.0f + std::exp(-x))) * u[i];
    }
}
static float dot_scalar(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}
static void axpy_scalar(float* y, const float* x, float a, int n) {
    for (int i = 0; i < n; ++i) y[i] += a * x[i];
}
#pragma GCC pop_options

#if defined(__AVX2__)
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}
static float dot_avx2(const float* a, const float* b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    float s = hsum256(acc);
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}
static void axpy_avx2(float* y, const float* x, float a, int n) {
    __m256 va = _mm256_set1_ps(a);
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
    for (; i < n; ++i) y[i] += a * x[i];
}
static void matvec_avx2(const Tensor& t, int out_dim, int in_dim, const float* x, float* y) {
    for (int i = 0; i < out_dim; ++i)
        y[i] = dot_avx2(t.data + static_cast<size_t>(i) * in_dim, x, in_dim);
}
static void rmsnorm_avx2(const Tensor& w, const float* x, float* y) {
    float ss = dot_avx2(x, x, HID);
    __m256 vinv = _mm256_set1_ps(1.0f / std::sqrt(ss / HID + EPS));
    for (int i = 0; i + 8 <= HID; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 wt = _mm256_loadu_ps(w.data + i);
        _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_mul_ps(v, wt), vinv));
    }
}
#endif

#if defined(__AVX512F__)
static inline float hsum512(__m512 v) {
#if defined(__AVX512DQ__)
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    return hsum256(_mm256_add_ps(lo, hi));
#else
    alignas(64) float t[16];
    _mm512_storeu_ps(t, v);
    float s = 0.0f;
    for (int i = 0; i < 16; ++i) s += t[i];
    return s;
#endif
}
static float dot_avx512(const float* a, const float* b, int n) {
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16)
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    float s = hsum512(acc);
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}
static void axpy_avx512(float* y, const float* x, float a, int n) {
    __m512 va = _mm512_set1_ps(a);
    int i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(y + i, _mm512_fmadd_ps(va, _mm512_loadu_ps(x + i), _mm512_loadu_ps(y + i)));
    for (; i < n; ++i) y[i] += a * x[i];
}
static void matvec_avx512(const Tensor& t, int out_dim, int in_dim, const float* x, float* y) {
    for (int i = 0; i < out_dim; ++i)
        y[i] = dot_avx512(t.data + static_cast<size_t>(i) * in_dim, x, in_dim);
}
static void rmsnorm_avx512(const Tensor& w, const float* x, float* y) {
    float ss = dot_avx512(x, x, HID);
    __m512 vinv = _mm512_set1_ps(1.0f / std::sqrt(ss / HID + EPS));
    for (int i = 0; i + 16 <= HID; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        __m512 wt = _mm512_loadu_ps(w.data + i);
        _mm512_storeu_ps(y + i, _mm512_mul_ps(_mm512_mul_ps(v, wt), vinv));
    }
}
#endif

static inline float dotv(const float* a, const float* b, int n) {
#if defined(__AVX512F__)
    if (g_simd >= 5) {
#if defined(__AVX2__)
        if (g_simd >= 6 && n <= 16) return dot_avx2(a, b, n);
#endif
        return dot_avx512(a, b, n);
    }
#endif
#if defined(__AVX2__)
    if (g_simd >= 2) return dot_avx2(a, b, n);
#endif
    return dot_scalar(a, b, n);
}
static inline void axpyv(float* y, const float* x, float a, int n) {
#if defined(__AVX512F__)
    if (g_simd >= 5) {
#if defined(__AVX2__)
        if (g_simd >= 6 && n <= 16) { axpy_avx2(y, x, a, n); return; }
#endif
        axpy_avx512(y, x, a, n);
        return;
    }
#endif
#if defined(__AVX2__)
    if (g_simd >= 2) { axpy_avx2(y, x, a, n); return; }
#endif
    axpy_scalar(y, x, a, n);
}
static inline void matvec(const Tensor& t, int out_dim, int in_dim, const float* x, float* y) {
#if defined(__AVX512F__)
    if (g_simd >= 5) { matvec_avx512(t, out_dim, in_dim, x, y); return; }
#endif
#if defined(__AVX2__)
    if (g_simd >= 2) { matvec_avx2(t, out_dim, in_dim, x, y); return; }
#endif
    matvec_scalar(t, out_dim, in_dim, x, y);
}
static inline void rmsnorm(const Tensor& w, const float* x, float* y) {
#if defined(__AVX512F__)
    if (g_simd >= 5) { rmsnorm_avx512(w, x, y); return; }
#endif
#if defined(__AVX2__)
    if (g_simd >= 2) { rmsnorm_avx2(w, x, y); return; }
#endif
    rmsnorm_scalar(w, x, y);
}
static void silu_mul(const float* g, const float* u, float* out) {
    silu_mul_scalar(g, u, out);
}

// Forward over a token sequence; writes logits of the last position into logits[VOCAB].
static void forward(const Weights& w, const int* ids, int S, float* logits) {
    std::vector<float> x(static_cast<size_t>(S) * HID);
    std::vector<float> cosv(static_cast<size_t>(S) * HD), sinv(static_cast<size_t>(S) * HD);
    std::vector<float> q(static_cast<size_t>(S) * NH * HD), k(static_cast<size_t>(S) * NKV * HD);
    std::vector<float> v(static_cast<size_t>(S) * NKV * HD), first_v(static_cast<size_t>(S) * NKV * HD);
    std::vector<float> attn(static_cast<size_t>(S) * NH * HD);
    std::vector<float> tmp(static_cast<size_t>(S) * HID), z(static_cast<size_t>(S) * HID);
    std::vector<float> h(static_cast<size_t>(S) * INTER), cur(HID), out(HID);

    // input embeddings
    for (int s = 0; s < S; ++s)
        for (int i = 0; i < HID; ++i) x[static_cast<size_t>(s) * HID + i] = w.embed.data[static_cast<size_t>(ids[s]) * HID + i];

    // RoPE cos/sin tables
    for (int j = 0; j < HD / 2; ++j) {
        float invfreq = std::pow(THETA, -(2.0f * j) / HD);
        for (int s = 0; s < S; ++s) {
            float a = s * invfreq;
            float c = std::cos(a), sn = std::sin(a);
            cosv[static_cast<size_t>(s) * HD + j] = c;
            cosv[static_cast<size_t>(s) * HD + HD / 2 + j] = c;
            sinv[static_cast<size_t>(s) * HD + j] = sn;
            sinv[static_cast<size_t>(s) * HD + HD / 2 + j] = sn;
        }
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));

    for (int step = 0; step < STEPS; ++step) {
        // attention projections
        for (int s = 0; s < S; ++s) {
            rmsnorm(w.ln1, &x[static_cast<size_t>(s) * HID], cur.data());
            matvec(w.q, NH * HD, HID, cur.data(), &q[static_cast<size_t>(s) * NH * HD]);
            matvec(w.k, NKV * HD, HID, cur.data(), &k[static_cast<size_t>(s) * NKV * HD]);
            matvec(w.v, NKV * HD, HID, cur.data(), &v[static_cast<size_t>(s) * NKV * HD]);
        }
        // value-residual mixing
        if (step == 0) {
            std::memcpy(first_v.data(), v.data(), sizeof(float) * v.size());
        } else {
            for (size_t i = 0; i < v.size(); ++i) v[i] = (1.0f - MIX) * v[i] + MIX * first_v[i];
        }
        // RoPE on q and k (compute into temp to avoid in-place corruption)
        float rope[HD];
        auto apply_rope = [&](float* vec, int s) {
            for (int d = 0; d < HD; ++d) {
                float rot = (d < HD / 2) ? -vec[d + HD / 2] : vec[d - HD / 2];
                rope[d] = vec[d] * cosv[static_cast<size_t>(s) * HD + d] + rot * sinv[static_cast<size_t>(s) * HD + d];
            }
            std::memcpy(vec, rope, sizeof(float) * HD);
        };
        for (int s = 0; s < S; ++s) {
            for (int head = 0; head < NH; ++head)
                apply_rope(&q[static_cast<size_t>(s) * NH * HD + head * HD], s);
            apply_rope(&k[static_cast<size_t>(s) * NKV * HD], s);
        }
        // causal attention
        std::vector<float> scores(static_cast<size_t>(S));
        for (int t = 0; t < S; ++t) {
            for (int head = 0; head < NH; ++head) {
                const float* qt = &q[static_cast<size_t>(t) * NH * HD + head * HD];
                float mx = -1e30f;
                for (int s2 = 0; s2 <= t; ++s2) {
                    const float* ks = &k[static_cast<size_t>(s2) * NKV * HD];
                    float acc = dotv(qt, ks, HD) * scale;
                    scores[s2] = acc;
                    if (acc > mx) mx = acc;
                }
                float sum = 0.0f;
                for (int s2 = 0; s2 <= t; ++s2) { scores[s2] = std::exp(scores[s2] - mx); sum += scores[s2]; }
                float inv = 1.0f / sum;
                float* ao = &attn[static_cast<size_t>(t) * NH * HD + head * HD];
                for (int d = 0; d < HD; ++d) ao[d] = 0.0f;
                for (int s2 = 0; s2 <= t; ++s2) {
                    float p = scores[s2] * inv;
                    const float* vs = &v[static_cast<size_t>(s2) * NKV * HD];
                    axpyv(ao, vs, p, HD);
                }
            }
        }
        // o_proj + residual + norm
        for (int s = 0; s < S; ++s) {
            float* xs = &x[static_cast<size_t>(s) * HID];
            matvec(w.o, HID, NH * HD, &attn[static_cast<size_t>(s) * NH * HD], out.data());
            rmsnorm(w.ln1b, out.data(), tmp.data());
            for (int i = 0; i < HID; ++i) xs[i] += tmp[i];
        }
        // MLP + residual + norm
        for (int s = 0; s < S; ++s) {
            float* xs = &x[static_cast<size_t>(s) * HID];
            rmsnorm(w.ln2, xs, z.data());
            float g[INTER], u[INTER], m[INTER];
            matvec(w.gate, INTER, HID, z.data(), g);
            matvec(w.up, INTER, HID, z.data(), u);
            silu_mul(g, u, m);
            matvec(w.down, HID, INTER, m, out.data());
            rmsnorm(w.ln2b, out.data(), tmp.data());
            for (int i = 0; i < HID; ++i) xs[i] += tmp[i];
        }
        // final norm (feeds next loop)
        for (int s = 0; s < S; ++s) {
            float* xs = &x[static_cast<size_t>(s) * HID];
            rmsnorm(w.norm, xs, tmp.data());
            std::memcpy(xs, tmp.data(), sizeof(float) * HID);
        }
    }
    // lm_head (tied to embedding) on last position
    const float* xl = &x[static_cast<size_t>(S - 1) * HID];
    matvec(w.embed, VOCAB, HID, xl, logits);
}

// ---- KV cache -------------------------------------------------------------
// The Ouro layer is applied STEPS times with shared weights; the reference
// implementation keeps one cache slot per recurrent step (index =
// current_ut * num_layers + layer). We mirror that with STEPS independent
// K/V buffers. Cached K is post-RoPE, cached V is post-value-residual.
struct KVCache {
    int max_seq;
    std::vector<float> k, v;
    explicit KVCache(int max_seq_)
        : max_seq(max_seq_),
          k(static_cast<size_t>(STEPS) * max_seq_ * NKV * HD),
          v(static_cast<size_t>(STEPS) * max_seq_ * NKV * HD) {}
    float* k_step(int step) { return k.data() + static_cast<size_t>(step) * max_seq * NKV * HD; }
    float* v_step(int step) { return v.data() + static_cast<size_t>(step) * max_seq * NKV * HD; }
};

static void rope_apply(float* vec, const float* cosv, const float* sinv) {
    float tmp[HD];
    for (int d = 0; d < HD; ++d) {
        float rot = (d < HD / 2) ? -vec[d + HD / 2] : vec[d - HD / 2];
        tmp[d] = vec[d] * cosv[d] + rot * sinv[d];
    }
    std::memcpy(vec, tmp, sizeof(float) * HD);
}

// Forward a single token at position pos, appending to the KV cache.
// Attends over cached positions [0, pos]; writes logits[VOCAB].
static void forward_token(const Weights& w, int token, int pos, KVCache& cache, float* logits) {
    float x[HID], r[HID], cur[HID], qv[NH * HD], kv[NKV * HD], vv[NKV * HD], first_v[NKV * HD];
    float attn[NH * HD], out[HID], tmp[HID], z[HID], g[INTER], u[INTER], m[INTER];
    static thread_local std::vector<float> scores;
    scores.resize(static_cast<size_t>(pos) + 1);

    for (int i = 0; i < HID; ++i) x[i] = w.embed.data[static_cast<size_t>(token) * HID + i];

    float cosv[HD], sinv[HD];
    for (int j = 0; j < HD / 2; ++j) {
        float invfreq = std::pow(THETA, -(2.0f * j) / HD);
        float a = pos * invfreq;
        cosv[j] = cosv[j + HD / 2] = std::cos(a);
        sinv[j] = sinv[j + HD / 2] = std::sin(a);
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));

    for (int step = 0; step < STEPS; ++step) {
        std::memcpy(r, x, sizeof(float) * HID);
        rmsnorm(w.ln1, x, cur);
        matvec(w.q, NH * HD, HID, cur, qv);
        matvec(w.k, NKV * HD, HID, cur, kv);
        matvec(w.v, NKV * HD, HID, cur, vv);
        if (step == 0) {
            std::memcpy(first_v, vv, sizeof(float) * NKV * HD);
        } else {
            for (int i = 0; i < NKV * HD; ++i) vv[i] = (1.0f - MIX) * vv[i] + MIX * first_v[i];
        }
        for (int head = 0; head < NH; ++head) rope_apply(&qv[head * HD], cosv, sinv);
        rope_apply(kv, cosv, sinv);

        float* K = cache.k_step(step);
        float* V = cache.v_step(step);
        std::memcpy(K + static_cast<size_t>(pos) * NKV * HD, kv, sizeof(float) * NKV * HD);
        std::memcpy(V + static_cast<size_t>(pos) * NKV * HD, vv, sizeof(float) * NKV * HD);

        for (int head = 0; head < NH; ++head) {
            const float* qh = &qv[head * HD];
            float mx = -1e30f;
            for (int s2 = 0; s2 <= pos; ++s2) {
                const float* ks = K + static_cast<size_t>(s2) * NKV * HD;
                float acc = dotv(qh, ks, HD) * scale;
                scores[s2] = acc;
                if (acc > mx) mx = acc;
            }
            float sum = 0.0f;
            for (int s2 = 0; s2 <= pos; ++s2) { scores[s2] = std::exp(scores[s2] - mx); sum += scores[s2]; }
            float inv = 1.0f / sum;
            float* ao = &attn[head * HD];
            for (int d = 0; d < HD; ++d) ao[d] = 0.0f;
            for (int s2 = 0; s2 <= pos; ++s2) {
                float p = scores[s2] * inv;
                const float* vs = V + static_cast<size_t>(s2) * NKV * HD;
                axpyv(ao, vs, p, HD);
            }
        }

        matvec(w.o, HID, NH * HD, attn, out);
        rmsnorm(w.ln1b, out, tmp);
        for (int i = 0; i < HID; ++i) x[i] = r[i] + tmp[i];

        std::memcpy(r, x, sizeof(float) * HID);
        rmsnorm(w.ln2, x, z);
        matvec(w.gate, INTER, HID, z, g);
        matvec(w.up, INTER, HID, z, u);
        silu_mul(g, u, m);
        matvec(w.down, HID, INTER, m, out);
        rmsnorm(w.ln2b, out, tmp);
        for (int i = 0; i < HID; ++i) x[i] = r[i] + tmp[i];

        rmsnorm(w.norm, x, tmp);
        std::memcpy(x, tmp, sizeof(float) * HID);
    }

    matvec(w.embed, VOCAB, HID, x, logits);
}

static int sample(const float* logits, float temp, float top_p, std::mt19937& rng);

// ---- batched (lockstep) decode -------------------------------------------
// Process B sequences in one forward pass at a common position. Weight rows
// are loaded once and reused across the batch (M rows). Every active sequence
// has been advanced the same number of steps, so cache index == pos for all.
static void matmul_batch(const Tensor& W, int M, int N, int K, const float* x, float* y) {
    for (int o = 0; o < N; ++o) {
        const float* row = W.data + static_cast<size_t>(o) * K;
        for (int b = 0; b < M; ++b)
            y[static_cast<size_t>(b) * N + o] = dotv(row, x + static_cast<size_t>(b) * K, K);
    }
}

static void forward_batch_token(const Weights& w, const int* tokens, int M, int pos,
                                KVCache** caches, float* logits) {
    static thread_local std::vector<float> x, cur, r, z, q, k, v, first_v, attn, out, tmp, g, u, m, scores;
    x.resize(static_cast<size_t>(M) * HID);
    cur.resize(static_cast<size_t>(M) * HID);
    r.resize(static_cast<size_t>(M) * HID);
    z.resize(static_cast<size_t>(M) * HID);
    q.resize(static_cast<size_t>(M) * NH * HD);
    k.resize(static_cast<size_t>(M) * NKV * HD);
    v.resize(static_cast<size_t>(M) * NKV * HD);
    first_v.resize(static_cast<size_t>(M) * NKV * HD);
    attn.resize(static_cast<size_t>(M) * NH * HD);
    out.resize(static_cast<size_t>(M) * HID);
    tmp.resize(static_cast<size_t>(M) * HID);
    g.resize(static_cast<size_t>(M) * INTER);
    u.resize(static_cast<size_t>(M) * INTER);
    m.resize(static_cast<size_t>(M) * INTER);
    scores.resize(static_cast<size_t>(pos) + 1);

    for (int b = 0; b < M; ++b)
        for (int i = 0; i < HID; ++i)
            x[static_cast<size_t>(b) * HID + i] = w.embed.data[static_cast<size_t>(tokens[b]) * HID + i];

    float cosv[HD], sinv[HD];
    for (int j = 0; j < HD / 2; ++j) {
        float invfreq = std::pow(THETA, -(2.0f * j) / HD);
        float a = pos * invfreq;
        cosv[j] = cosv[j + HD / 2] = std::cos(a);
        sinv[j] = sinv[j + HD / 2] = std::sin(a);
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));

    for (int step = 0; step < STEPS; ++step) {
        for (int b = 0; b < M; ++b)
            rmsnorm(w.ln1, &x[static_cast<size_t>(b) * HID], &cur[static_cast<size_t>(b) * HID]);
        matmul_batch(w.q, M, NH * HD, HID, cur.data(), q.data());
        matmul_batch(w.k, M, NKV * HD, HID, cur.data(), k.data());
        matmul_batch(w.v, M, NKV * HD, HID, cur.data(), v.data());
        if (step == 0) {
            std::memcpy(first_v.data(), v.data(), sizeof(float) * v.size());
        } else {
            for (size_t i = 0; i < v.size(); ++i) v[i] = (1.0f - MIX) * v[i] + MIX * first_v[i];
        }
        for (int b = 0; b < M; ++b) {
            for (int head = 0; head < NH; ++head)
                rope_apply(&q[static_cast<size_t>(b) * NH * HD + head * HD], cosv, sinv);
            rope_apply(&k[static_cast<size_t>(b) * NKV * HD], cosv, sinv);
            std::memcpy(caches[b]->k_step(step) + static_cast<size_t>(pos) * NKV * HD,
                        &k[static_cast<size_t>(b) * NKV * HD], sizeof(float) * NKV * HD);
            std::memcpy(caches[b]->v_step(step) + static_cast<size_t>(pos) * NKV * HD,
                        &v[static_cast<size_t>(b) * NKV * HD], sizeof(float) * NKV * HD);

            float* K = caches[b]->k_step(step);
            float* V = caches[b]->v_step(step);
            for (int head = 0; head < NH; ++head) {
                const float* qh = &q[static_cast<size_t>(b) * NH * HD + head * HD];
                float mx = -1e30f;
                for (int s2 = 0; s2 <= pos; ++s2) {
                    float acc = dotv(qh, K + static_cast<size_t>(s2) * NKV * HD, HD) * scale;
                    scores[s2] = acc;
                    if (acc > mx) mx = acc;
                }
                float sum = 0.0f;
                for (int s2 = 0; s2 <= pos; ++s2) { scores[s2] = std::exp(scores[s2] - mx); sum += scores[s2]; }
                float inv = 1.0f / sum;
                float* ao = &attn[static_cast<size_t>(b) * NH * HD + head * HD];
                for (int d = 0; d < HD; ++d) ao[d] = 0.0f;
                for (int s2 = 0; s2 <= pos; ++s2) {
                    float p = scores[s2] * inv;
                    axpyv(ao, V + static_cast<size_t>(s2) * NKV * HD, p, HD);
                }
            }
        }

        matmul_batch(w.o, M, HID, NH * HD, attn.data(), out.data());
        for (int b = 0; b < M; ++b) {
            std::memcpy(&r[static_cast<size_t>(b) * HID], &x[static_cast<size_t>(b) * HID], sizeof(float) * HID);
            rmsnorm(w.ln1b, &out[static_cast<size_t>(b) * HID], &tmp[static_cast<size_t>(b) * HID]);
            for (int i = 0; i < HID; ++i)
                x[static_cast<size_t>(b) * HID + i] = r[static_cast<size_t>(b) * HID + i] + tmp[static_cast<size_t>(b) * HID + i];
            std::memcpy(&r[static_cast<size_t>(b) * HID], &x[static_cast<size_t>(b) * HID], sizeof(float) * HID);
            rmsnorm(w.ln2, &x[static_cast<size_t>(b) * HID], &z[static_cast<size_t>(b) * HID]);
        }
        matmul_batch(w.gate, M, INTER, HID, z.data(), g.data());
        matmul_batch(w.up, M, INTER, HID, z.data(), u.data());
        for (int b = 0; b < M; ++b)
            silu_mul(&g[static_cast<size_t>(b) * INTER], &u[static_cast<size_t>(b) * INTER], &m[static_cast<size_t>(b) * INTER]);
        matmul_batch(w.down, M, HID, INTER, m.data(), out.data());
        for (int b = 0; b < M; ++b) {
            rmsnorm(w.ln2b, &out[static_cast<size_t>(b) * HID], &tmp[static_cast<size_t>(b) * HID]);
            for (int i = 0; i < HID; ++i)
                x[static_cast<size_t>(b) * HID + i] = r[static_cast<size_t>(b) * HID + i] + tmp[static_cast<size_t>(b) * HID + i];
            rmsnorm(w.norm, &x[static_cast<size_t>(b) * HID], &tmp[static_cast<size_t>(b) * HID]);
            std::memcpy(&x[static_cast<size_t>(b) * HID], &tmp[static_cast<size_t>(b) * HID], sizeof(float) * HID);
        }
    }

    matmul_batch(w.embed, M, VOCAB, HID, x.data(), logits);
}

// Decode B sequences together, shrinking the batch as sequences hit EOS.
static std::vector<std::vector<int>> generate_batch_gemm(const Weights& w, int seed, int nseq,
                                                        int max_new, float temp, float top_p) {
    std::vector<std::vector<int>> out(nseq);
    std::vector<std::mt19937> rngs;
    rngs.reserve(nseq);
    for (int i = 0; i < nseq; ++i) rngs.emplace_back(static_cast<uint32_t>(seed + i));

    std::vector<KVCache> caches;
    caches.reserve(nseq);
    for (int i = 0; i < nseq; ++i) caches.emplace_back(max_new + 2);

    std::vector<int> last(nseq, BOS);
    std::vector<char> done(nseq, 0);
    std::vector<float> logits(static_cast<size_t>(nseq) * VOCAB);

    std::vector<int> active;
    std::vector<int> toks;
    std::vector<KVCache*> cps;
    for (int pos = 0; pos < max_new; ++pos) {
        active.clear();
        for (int i = 0; i < nseq; ++i) if (!done[i]) active.push_back(i);
        if (active.empty()) break;
        int M = static_cast<int>(active.size());
        toks.resize(M);
        cps.resize(M);
        for (int j = 0; j < M; ++j) { toks[j] = last[active[j]]; cps[j] = &caches[active[j]]; }
        forward_batch_token(w, toks.data(), M, pos, cps.data(), logits.data());
        for (int j = 0; j < M; ++j) {
            int id = active[j];
            int nxt = sample(&logits[static_cast<size_t>(j) * VOCAB], temp, top_p, rngs[id]);
            out[id].push_back(nxt);
            last[id] = nxt;
            if (nxt == EOS) done[id] = 1;
        }
    }
    return out;
}

// ---- tokenizer ------------------------------------------------------------
struct Tokenizer {
    std::vector<std::string> id2tok;
    std::unordered_map<int, unsigned char> cp2byte;
};

static void build_byte_decoder(Tokenizer& tk) {
    std::vector<int> bs, cs;
    for (int b = 33; b <= 126; ++b) bs.push_back(b);
    for (int b = 161; b <= 172; ++b) bs.push_back(b);
    for (int b = 174; b <= 255; ++b) bs.push_back(b);
    cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        bool present = false;
        for (int x : bs) if (x == b) { present = true; break; }
        if (!present) { bs.push_back(b); cs.push_back(256 + n); ++n; }
    }
    for (size_t i = 0; i < bs.size(); ++i) tk.cp2byte[cs[i]] = static_cast<unsigned char>(bs[i]);
}

static void parse_vocab(const std::string& json, Tokenizer& tk) {
    size_t vp = json.find("\"vocab\"");
    size_t i = json.find('{', vp);
    ++i;
    tk.id2tok.assign(VOCAB, std::string());
    while (i < json.size()) {
        while (i < json.size() && (json[i] == ' ' || json[i] == '\n' || json[i] == '\r' || json[i] == '\t' || json[i] == ',')) ++i;
        if (i >= json.size() || json[i] == '}') break;
        if (json[i] != '"') break;
        ++i;
        std::string key;
        while (i < json.size() && json[i] != '"') {
            if (json[i] == '\\' && i + 1 < json.size()) { key.push_back(json[i + 1]); i += 2; }
            else key.push_back(json[i++]);
        }
        ++i; // closing quote
        while (i < json.size() && json[i] != ':') ++i;
        ++i;
        while (i < json.size() && (json[i] == ' ' || json[i] == '\n')) ++i;
        int id = 0;
        while (i < json.size() && json[i] >= '0' && json[i] <= '9') id = id * 10 + (json[i++] - '0');
        if (id >= 0 && id < VOCAB) tk.id2tok[id] = key;
    }
}

static Tokenizer load_tokenizer(const std::string& path) {
    std::vector<char> buf = read_file(path);
    Tokenizer tk;
    build_byte_decoder(tk);
    parse_vocab(std::string(buf.begin(), buf.end()), tk);
    return tk;
}

// Decode a UTF-8 codepoint sequence through the GPT-2 byte map.
static void decode_token_bytes(const std::string& tok, const Tokenizer& tk, std::string& out) {
    size_t i = 0;
    while (i < tok.size()) {
        unsigned char c = static_cast<unsigned char>(tok[i]);
        unsigned int cp = 0;
        int len = 1;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c >> 5) == 0x6) { cp = c & 0x1F; len = 2; }
        else if ((c >> 4) == 0xE) { cp = c & 0x0F; len = 3; }
        else { cp = c & 0x07; len = 4; }
        for (int j = 1; j < len && i + j < tok.size(); ++j) cp = (cp << 6) | (static_cast<unsigned char>(tok[i + j]) & 0x3F);
        auto it = tk.cp2byte.find(static_cast<int>(cp));
        if (it != tk.cp2byte.end()) out.push_back(static_cast<char>(it->second));
        i += len;
    }
}

static std::string detokenize(const std::vector<int>& ids, const Tokenizer& tk) {
    std::string out;
    for (int id : ids) {
        if (id < 0 || id >= VOCAB) continue;
        if (id == 4) { out += "Alex"; continue; }
        if (id == 5) { out += "Robin"; continue; }
        if (id == 6) { out += "Casey"; continue; }
        if (id == BOS || id == EOS || id == 0 || id == 3) continue;
        decode_token_bytes(tk.id2tok[id], tk, out);
    }
    size_t a = out.find_first_not_of(" \t\n\r");
    size_t b = out.find_last_not_of(" \t\n\r");
    if (a == std::string::npos) return std::string();
    return out.substr(a, b - a + 1);
}

// ---- sampling -------------------------------------------------------------
static int sample(const float* logits, float temp, float top_p, std::mt19937& rng) {
    float p[VOCAB];
    int order[VOCAB];
    float mx = -1e30f;
    for (int i = 0; i < VOCAB; ++i) { p[i] = logits[i] / temp; if (p[i] > mx) mx = p[i]; }
    float sum = 0.0f;
    for (int i = 0; i < VOCAB; ++i) { p[i] = std::exp(p[i] - mx); sum += p[i]; }
    for (int i = 0; i < VOCAB; ++i) p[i] /= sum;

    for (int i = 0; i < VOCAB; ++i) order[i] = i;
    std::sort(order, order + VOCAB, [&](int a, int b) { return p[a] > p[b]; });
    float cum = 0.0f;
    int keep = 0;
    for (int i = 0; i < VOCAB; ++i) { cum += p[order[i]]; ++keep; if (cum >= top_p) break; }
    float norm = 0.0f;
    for (int i = 0; i < keep; ++i) norm += p[order[i]];
    std::uniform_real_distribution<float> dist(0.0f, norm);
    float r = dist(rng);
    float c = 0.0f;
    for (int i = 0; i < keep; ++i) { c += p[order[i]]; if (r <= c) return order[i]; }
    return order[keep - 1];
}

static std::vector<int> generate_ids(const Weights& w, int seed, int max_new,
                                     float temp, float top_p, bool use_kv) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::vector<float> logits(VOCAB);
    std::vector<int> body;
    if (use_kv) {
        KVCache cache(max_new + 2);
        int cur = BOS;
        for (int pos = 0; pos < max_new; ++pos) {
            forward_token(w, cur, pos, cache, logits.data());
            int nxt = sample(logits.data(), temp, top_p, rng);
            body.push_back(nxt);
            cur = nxt;
            if (nxt == EOS) break;
        }
    } else {
        std::vector<int> ids = {BOS};
        for (int step = 0; step < max_new; ++step) {
            forward(w, ids.data(), static_cast<int>(ids.size()), logits.data());
            int nxt = sample(logits.data(), temp, top_p, rng);
            ids.push_back(nxt);
            if (nxt == EOS) break;
        }
        body.assign(ids.begin() + 1, ids.end());
    }
    return body;
}

int main(int argc, char** argv) {
    std::string root = ".";
    int seed = 20261007, max_new = 512, nthreads = 1, batch_gemm = 0;
    float temp = 0.5f, top_p = 0.9f;
    bool use_kv = true, selftest_kv = false, selftest_simd = false, selftest_batch = false, quiet = false;
    std::vector<int> dump_ids;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--root" && i + 1 < argc) root = argv[++i];
        else if (a == "--seed" && i + 1 < argc) seed = std::atoi(argv[++i]);
        else if (a == "--max" && i + 1 < argc) max_new = std::atoi(argv[++i]);
        else if (a == "--temp" && i + 1 < argc) temp = std::atof(argv[++i]);
        else if (a == "--top-p" && i + 1 < argc) top_p = std::atof(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) nthreads = std::max(1, std::atoi(argv[++i]));
        else if (a == "--batch" && i + 1 < argc) nthreads = std::max(1, std::atoi(argv[++i]));
        else if (a == "--batch-gemm" && i + 1 < argc) batch_gemm = std::max(1, std::atoi(argv[++i]));
        else if (a == "--no-kv") use_kv = false;
        else if (a == "--no-simd") g_simd = 0;
        else if (a == "--simd" && i + 1 < argc) {
            std::string s = argv[++i];
            if (s == "scalar" || s == "0") g_simd = 0;
            else if (s == "avx2" || s == "2") g_simd = 2;
            else if (s == "avx512" || s == "512" || s == "5") g_simd = 5;
            else if (s == "hybrid" || s == "6") g_simd = 6;
        }
        else if (a == "--selftest-kv") selftest_kv = true;
        else if (a == "--selftest-simd") selftest_simd = true;
        else if (a == "--selftest-batch") selftest_batch = true;
        else if (a == "--quiet") quiet = true;
        else if (a == "--logits" && i + 1 < argc) {
            std::string s = argv[++i];
            size_t st = 0;
            while (st <= s.size()) {
                size_t c = s.find(',', st);
                if (c == std::string::npos) c = s.size();
                dump_ids.push_back(std::atoi(s.substr(st, c - st).c_str()));
                st = c + 1;
            }
        }
    }

    Weights w = load_weights(root + "/model.safetensors");
    Tokenizer tk = load_tokenizer(root + "/tokenizer.json");

    if (selftest_kv) {
        const int L = 256;
        std::mt19937 r(0);
        std::uniform_int_distribution<int> d(0, VOCAB - 1);
        std::vector<int> seq = {BOS};
        for (int i = 0; i < L; ++i) seq.push_back(d(r));
        KVCache cache(static_cast<int>(seq.size()) + 2);
        std::vector<float> inc(VOCAB), full(VOCAB);
        for (int pos = 0; pos < static_cast<int>(seq.size()); ++pos)
            forward_token(w, seq[pos], pos, cache, inc.data());
        forward(w, seq.data(), static_cast<int>(seq.size()), full.data());
        float md = 0.0f;
        for (int i = 0; i < VOCAB; ++i) md = std::max(md, std::fabs(inc[i] - full[i]));
        std::printf("selftest L=%d max_abs_diff=%.3e\n", L, md);
        return 0;
    }

    if (selftest_simd) {
        std::vector<int> levels = {0};
#if defined(__AVX2__)
        levels.push_back(2);
#endif
#if defined(__AVX512F__)
        levels.push_back(5);
        levels.push_back(6);
#endif
        std::printf("compiled levels:");
        for (int l : levels) std::printf(" %s", simd_name(l));
        std::printf("\n");

        std::mt19937 r(123);
        std::uniform_int_distribution<int> d(0, VOCAB - 1);
        float worst = 0.0f;
        int worstL = 0;
        for (int L : {1, 7, 33, 128, 257}) {
            std::vector<int> seq(L);
            for (auto& x : seq) x = d(r);
            std::vector<std::vector<float>> outs(levels.size(), std::vector<float>(VOCAB));
            for (size_t k = 0; k < levels.size(); ++k) {
                g_simd = levels[k];
                KVCache c(L + 2);
                for (int pos = 0; pos < L; ++pos) forward_token(w, seq[pos], pos, c, outs[k].data());
            }
            for (size_t k = 1; k < levels.size(); ++k) {
                float md = 0.0f;
                for (int i = 0; i < VOCAB; ++i) md = std::max(md, std::fabs(outs[k][i] - outs[0][i]));
                std::printf("L=%4d %-7s-vs-scalar max_abs_diff=%.3e\n", L, simd_name(levels[k]), md);
                if (md > worst) { worst = md; worstL = L; }
            }
        }
        g_simd = 0;
        std::vector<int> ref = generate_ids(w, 42, 128, temp, top_p, true);
        bool same = true;
        for (size_t k = 1; k < levels.size(); ++k) {
            g_simd = levels[k];
            if (generate_ids(w, 42, 128, temp, top_p, true) != ref) same = false;
        }
        g_simd = levels.back();
        std::printf("worst L=%d diff=%.3e; generation identical across levels=%s\n",
                    worstL, worst, same ? "YES" : "NO");
        return 0;
    }

    if (selftest_batch) {
        int B = 5;
        auto bg = generate_batch_gemm(w, 700, B, max_new, temp, top_p);
        bool ok = true;
        for (int i = 0; i < B; ++i) {
            auto seq = generate_ids(w, 700 + i, max_new, temp, top_p, true);
            bool same = (seq == bg[i]);
            ok = ok && same;
            std::printf("batch seq %d: identical=%s len=%zu\n", i, same ? "YES" : "NO", bg[i].size());
        }
        std::printf("ALL IDENTICAL: %s\n", ok ? "YES" : "NO");
        return 0;
    }

    if (!dump_ids.empty()) {
        std::vector<float> logits(VOCAB);
        forward(w, dump_ids.data(), static_cast<int>(dump_ids.size()), logits.data());
        for (int i = 0; i < VOCAB; ++i) std::printf("%.6f%s", logits[i], i + 1 == VOCAB ? "\n" : " ");
        return 0;
    }

    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::vector<int>> results;
    const char* mode = "single";
    if (batch_gemm > 0) {
        mode = "batch-gemm";
        int groups = nthreads;
        results.resize(static_cast<size_t>(groups) * batch_gemm);
        if (groups == 1) {
            results = generate_batch_gemm(w, seed, batch_gemm, max_new, temp, top_p);
        } else {
            std::vector<std::vector<std::vector<int>>> grouped(groups);
            std::vector<std::thread> pool;
            pool.reserve(groups);
            for (int t = 0; t < groups; ++t)
                pool.emplace_back([&, t]() {
                    grouped[t] = generate_batch_gemm(w, seed + t * batch_gemm, batch_gemm, max_new, temp, top_p);
                });
            for (auto& th : pool) th.join();
            for (int t = 0; t < groups; ++t)
                for (int j = 0; j < batch_gemm; ++j)
                    results[static_cast<size_t>(t) * batch_gemm + j] = std::move(grouped[t][j]);
        }
    } else if (nthreads == 1) {
        results.resize(1);
        results[0] = generate_ids(w, seed, max_new, temp, top_p, use_kv);
    } else {
        mode = "threads";
        results.resize(nthreads);
        std::vector<std::thread> pool;
        pool.reserve(nthreads);
        for (int t = 0; t < nthreads; ++t)
            pool.emplace_back([&, t]() {
                results[t] = generate_ids(w, seed + t, max_new, temp, top_p, use_kv);
            });
        for (auto& th : pool) th.join();
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    size_t total = 0;
    for (size_t t = 0; t < results.size(); ++t) {
        total += results[t].size();
        if (!quiet) {
            if (results.size() > 1) std::printf("--- story %zu (seed %d) ---\n", t, seed + static_cast<int>(t));
            std::string story = detokenize(results[t], tk);
            std::printf("%s\n", story.c_str());
        }
    }
    std::fprintf(stderr, "[mode %s, seq %zu, tokens %zu, %.1f ms, %.0f tok/s, kv=%s, simd=%s]\n",
                 mode, results.size(), total, secs * 1000.0, total / secs,
                 use_kv ? "on" : "off", simd_name(g_simd));
    return 0;
}
