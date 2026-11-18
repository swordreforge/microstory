> 归档说明：本文是一次外部（LLM）代码评审的原始记录，未经删改。
> 结论已逐条对照当前实现处理，见 [REPORT.md 第 10 节](../REPORT.md#10-外部代码评审的处理)。
> 文中引用的行号、版本与数据对应评审当时的实现，可能已过时。

我把这份约 870 行的 C++ 实现从头到尾检查了一遍。先给结论：

这是一个质量不错的轻量级推理原型，而且已经不只是简单的模型移植：你们实现了 KV Cache、AVX2/FMA、独立序列并行、Batch GEMM 和多组数值一致性测试。

不过，如果目标是把 MacroStories 做成一个真正高性能、可长期维护的零依赖 C++17 推理后端，我认为还有相当大的优化空间。

我尤其注意到一个问题：你们现在的瓶颈可能不在矩阵乘法本身，而在推理循环中的内存分配、函数调用、重复计算和 SIMD 数据组织。

下面按代码质量、正确性、性能和架构四个方面分析。

# 一、整体代码质量评价

整体评价

原型质量良好

# 7.5 / 10

基于当前提交的静态代码审查，不代表实际运行性能评分。

推理算法实现

8.5/10

KV Cache、循环步缓存和残差逻辑比较完整

SIMD 实现

7.5/10

有 AVX2/FMA，但内核还有优化空间

内存管理

6.5/10

存在重复动态分配和不必要的数据复制

批量推理

7/10

有批处理机制，但还不是真正高效的矩阵乘法

工程健壮性

6/10

参数、模型格式和 tokenizer 校验不足

可维护性

7/10

功能集中，但架构参数与实现耦合较强

## 已经做得比较好的地方

### 1. KV Cache 设计值得保留

你们没有简单地把普通 Transformer 的 KV Cache 套进来，而是针对 Ouro 的循环结构：

```
struct KVCache {
    int max_seq;
    std::vector<float> k, v;

    float* k_step(int step) {
        return k.data()
            + static_cast<size_t>(step) * max_seq * NKV * HD;
    }

    float* v_step(int step) {
        return v.data()
            + static_cast<size_t>(step) * max_seq * NKV * HD;
    }
};
```

将不同 recurrent step 的 K/V 分开缓存。

这是正确的设计方向。

原因是每次循环迭代都会产生不同的隐藏状态，同一个位置在不同循环步中产生的 K/V 不能简单混用。

而且你们已经有：

- `--selftest-kv`
- `--no-kv`
- Cached K/V 与 post-RoPE K 的处理
- Value Residual Mixing 的缓存

这比仅仅实现一个普通 KV Cache 完整得多。

参考：KV Cache 实现及验证逻辑。

粘贴的文本 (1).txt



### 2. Scalar Reference + SIMD 双路径

这个设计我也建议保留：

```
static inline float dotv(
    const float* a,
    const float* b,
    int n
) {
#if defined(__AVX2__)
    if (g_use_simd)
        return dot_avx2(a, b, n);
#endif

    return dot_scalar(a, b, n);
}
```

有一个独立的标量参考实现，方便验证 SIMD 计算结果。

而且提供了：

```
--no-simd
--selftest-simd
```

这对于推理后端尤其重要，因为浮点计算的微小差异有时会影响后续采样结果。

不过，你们当前的 SIMD 还有一个值得注意的问题：它并不是真正的 CPU 运行时指令集检测，后面会讲。

粘贴的文本 (1).txt

### 3. 有独立的 Batch Decode

你们已经实现了两种并发模式：

- `--threads T`：独立序列并行。
- `--batch-gemm B`：多个序列锁步执行。

甚至可以组合：

```
./macrostories \
    --threads 2 \
    --batch-gemm 4
```

这一点很好。

但需要区分：当前 `matmul_batch()` 本质上还是一组矩阵向量乘法，并不是真正意义上的高效 Batch GEMM。

粘贴的文本 (1).txt

# 二、最值得优先优化的 6 个地方

我按照预期收益和改造成本排序。

## P0：消除每个 Token 的动态内存分配

最高优先级

这是我认为最值得先改的地方。

看 `forward_token()`：

```
static void forward_token(
    const Weights& w,
    int token,
    int pos,
    KVCache& cache,
    float* logits
) {
    float x[HID], ...;

    std::vector<float> scores(
        static_cast<size_t>(pos) + 1
    );
```

每生成一个 token，就会创建一个新的 `scores` vector。

这意味着：

- 动态内存分配。
- 动态内存释放。
- 堆分配器调用。
- 重复初始化。

虽然只有一个 vector，但这是一个每 Token 都会执行的路径。

而且 `forward_batch_token()` 也存在类似问题。

### 建议：引入 DecodeWorkspace

不要让推理函数自己分配内存。

```
struct DecodeWorkspace {
    std::vector<float> scores;

    explicit DecodeWorkspace(int max_seq)
        : scores(max_seq) {}
};
```

修改函数签名：

```
static void forward_token(
    const Weights& w,
    int token,
    int pos,
    KVCache& cache,
    DecodeWorkspace& workspace,
    float* logits
) {
    // ...

    float* scores = workspace.scores.data();

    // 后面直接使用 scores[s2]
}
```

初始化时：

```
KVCache cache(max_new + 2);

DecodeWorkspace workspace(max_new + 2);
```

这样整个生成过程不再为 attention scores 反复分配内存。

进一步建议：

将以下内容全部放进 Workspace：

- Attention scores
- Sampling probabilities
- Sampling index
- RoPE 临时数据
- 临时 logits

原则是：

> 一次推理任务分配一次内存，整个 Token Decode 生命周期复用。

对于多线程，每个线程独立拥有 Workspace，避免共享写入。

## P1：预计算 RoPE，消除重复 sin/cos/pow

高优先级

你们在 `forward_token()` 中：

```
for (int j = 0; j < HD / 2; ++j) {
    float invfreq = std::pow(
        THETA,
        -(2.0f * j) / HD
    );

    float a = pos * invfreq;

    cosv[j] = std::cos(a);
    sinv[j] = std::sin(a);
}
```

每个 Token 都重新计算：

- 8 次 `pow`
- 8 次 `cos`
- 8 次 `sin`

而且每个 Token 的 RoPE 频率是固定的，只与位置有关。

这完全可以预计算。

### 推荐实现

```
struct RopeTable {
    int max_seq;

    std::vector<float> cos_table;
    std::vector<float> sin_table;

    explicit RopeTable(int max_seq)
        : max_seq(max_seq),
          cos_table(max_seq * (HD / 2)),
          sin_table(max_seq * (HD / 2))
    {
        float invfreq[HD / 2];

        for (int j = 0; j < HD / 2; ++j) {
            invfreq[j] = std::pow(
                THETA,
                -(2.0f * j) / HD
            );
        }

        for (int pos = 0; pos < max_seq; ++pos) {
            for (int j = 0; j < HD / 2; ++j) {
                float a = pos * invfreq[j];

                cos_table[pos * (HD / 2) + j]
                    = std::cos(a);

                sin_table[pos * (HD / 2) + j]
                    = std::sin(a);
            }
        }
    }
};
```

之后每个 Token 只需要读取：

```
const float* cosv =
    rope.cos_table.data() + pos * (HD / 2);

const float* sinv =
    rope.sin_table.data() + pos * (HD / 2);
```

同时把 `rope_apply()` 改为通过 `d % (HD / 2)` 读取频率。

这样可以彻底移除 Token Decode 路径中的三角函数计算。

注意：RoPE 的位置、维度排列和旋转公式必须保持与现有模型一致，建议继续使用 KV 一致性测试验证。

## P2：优化 RMSNorm + Residual

你们目前有类似代码：

```
rmsnorm(w.ln1b, out, tmp);

for (int i = 0; i < HID; ++i)
    x[i] = r[i] + tmp[i];
```

这里可以融合成一个函数。

```
static inline void rmsnorm_add(
    const Tensor& w,
    const float* x,
    const float* residual,
    float* out
) {
    float ss = dotv(x, x, HID);

    float inv = 1.0f / std::sqrt(
        ss / HID + EPS
    );

    for (int i = 0; i < HID; ++i) {
        out[i] = residual[i]
               + w.data[i] * x[i] * inv;
    }
}
```

调用：

```
rmsnorm_add(w.ln1b, out, r, x);
```

这样可以省掉：

1. 一次临时数组写入。
2. 一次临时数组读取。
3. 一次额外的循环。

MLP 的第二次 Residual 也可以使用同样的方式。

另外，最终的：

```
rmsnorm(w.norm, x, tmp);
std::memcpy(x, tmp, sizeof(float) * HID);
```

可以改成专门的 `rmsnorm_inplace()`。

先计算平方和，再原地写入，不需要 `tmp`。

这类优化单次收益可能不大，但它们位于每个 Token 的四次循环中，累计起来比较值得。

粘贴的文本 (1).txt

## P3：优化 Sampling

当前：

```
static int sample(
    const float* logits,
    float temp,
    float top_p,
    std::mt19937& rng
) {
    std::vector<float> p(VOCAB);
    // ...

    std::vector<int> order(VOCAB);

    // ...

    std::sort(order.begin(), order.end(), ...);
}
```

这里有两个 vector，每生成一个 Token 都会重新分配。

建议使用：

```
#include <array>

struct SamplingWorkspace {
    std::array<float, VOCAB> probs;
    std::array<int, VOCAB> order;
};
```

然后：

```
static int sample(
    const float* logits,
    float temp,
    float top_p,
    std::mt19937& rng,
    SamplingWorkspace& ws
);
```

排序逻辑不必立即重写，先消除动态分配。

之后可以考虑：

- 固定大小的排序数组。
- `std::partial_sort`。
- 针对小词表的专用 Top-P 选择算法。

不过这里有个实际问题：你们的词表只有 378。

所以不要过度优化排序。相较于模型前向计算，它未必是最大的瓶颈。

粘贴的文本 (1).txt

## P4：重新设计 AVX2 MatVec

这是你们后端最有意思的优化方向之一。

当前：

```
static void matvec_avx2(
    const Tensor& t,
    int out_dim,
    int in_dim,
    const float* x,
    float* y
) {
    for (int i = 0; i < out_dim; ++i)
        y[i] = dot_avx2(
            t.data + static_cast<size_t>(i) * in_dim,
            x,
            in_dim
        );
}
```

当前 SIMD 的工作方式：

- 每次处理一行权重。
- AVX2 处理 8 个 float。
- 最后水平求和。
- 写回一个输出值。

问题是：每行都需要执行水平求和。

对于 HID=32、INTER=48 这样的小矩阵，水平求和的成本不容忽略。

### 一个可以尝试的方向：多行并行累加

例如一次处理 4 个输出神经元：

```
// 伪代码：演示内核组织方式
for (int o = 0; o < out_dim; o += 4) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    for (int k = 0; k < in_dim; k += 8) {
        __m256 xv = _mm256_loadu_ps(x + k);

        acc0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(W + (o + 0) * in_dim + k),
            xv, acc0
        );

        acc1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(W + (o + 1) * in_dim + k),
            xv, acc1
        );

        // acc2、acc3 同理
    }

    // 每个输出行最后分别水平求和
}
```

好处：

- 提高指令级并行度。
- 减少单一累加器的依赖链。
- 改善矩阵权重的连续访问。
- 降低部分内核调度开销。

但需要特别强调：这不是保证加速的优化。

因为模型很小，权重很可能已经全部处于 CPU Cache 中。必须和原始 `matvec_avx2()` 实际对比。

## P5：真正优化 Batch GEMM

目前：

```
for (int o = 0; o < N; ++o) {
    const float* row = W.data
        + static_cast<size_t>(o) * K;

    for (int b = 0; b < M; ++b) {
        y[b * N + o] = dotv(
            row,
            x + b * K,
            K
        );
    }
}
```

这是一个合理的初版，因为同一行权重被多个序列重复使用。

但是它并没有利用 SIMD 跨 Batch 计算。

我建议未来把它发展成：

```
MatMulBatch
    |
    +-- B=1: AVX2 MatVec
    |
    +-- B=2~4: 小 Batch AVX2 Kernel
    |
    +-- B=8+: 分块矩阵乘法
```

不过你们的模型权重规模大约只有 20K 参数。

这意味着：

不要照搬大型 Transformer 的 GEMM 优化策略。

对于你们这个模型，权重体积很小，计算量也不大。批处理最大的价值可能是减少重复读取权重，而不是解决显存或内存带宽瓶颈。

粘贴的文本 (1).txt

# 三、还有几个需要修复的工程问题

这些未必直接提升速度，但会影响后端可靠性。

| 问题                      | 代码位置    | 建议                      |
| ----------------------- | ------- | ----------------------- |
| 缺少模型 Tensor Shape 检查    | 64–105  | 检查每个权重的维度、元素数           |
| 未验证 safetensors dtype   | 64–80   | 明确拒绝非 F32 数据            |
| 手写 JSON 解析不完整           | 592–622 | 实现完整 JSON 转义处理          |
| Token ID 越界可能访问非法内存     | 221–224 | 验证输入 ID                 |
| `max_new` 缺少范围验证        | 711–727 | 限制最大上下文长度               |
| `temp=0` 会造成除零          | 660–665 | 验证温度大于零                 |
| `top_p` 没有合法范围检查        | 660–675 | 限制在 (0,1]               |
| AVX2 不是真正运行时检测          | 143–207 | 增加 CPU Feature Dispatch |
| `--batch` 实际等价于 threads | 724–726 | 明确参数含义                  |
| 部分临时数组空间冗余              | 211–219 | 删除未使用的 `h` 等数组          |

另外，`parse_vocab()` 目前只处理了非常简单的 JSON 转义方式，并不是通用 JSON 解析器。对于 tokenizer.json 中的 Unicode 转义、反斜杠、特殊字符，需要更加完整的处理。

还有一个架构层面的限制：

当前实现固定了：

```
VOCAB = 378
HID = 32
INTER = 48
NH = 2
NKV = 1
STEPS = 4
```

并且只加载 `model.layers.0` 的权重。

所以它本质上是一个针对特定 MacroStories/Ouro 检查点的专用推理引擎，而不是通用 Safetensors Transformer Runtime。

如果这是你们的明确目标，完全可以接受，但应该在加载阶段严格验证模型结构。

粘贴的文本 (1).txt

粘贴的文本 (1).txt



# 四、Intel Core Ultra 7 上的编译与优化建议

结合你们之前的 Linux + Intel Core Ultra 7 环境，我建议先走纯 CPU 优化路线，不需要引入任何第三方推理框架。

## 推荐编译配置

针对本机测试：

```
g++ macrostories.cpp \
    -o macrostories \
    -std=c++17 \
    -O3 \
    -DNDEBUG \
    -march=native \
    -mtune=native \
    -flto \
    -mavx2 \
    -mfma \
    -pthread
```

注意：

- `-O3`：主要优化。
- `-march=native`：针对当前 CPU 生成指令。
- `-flto`：链接期优化。
- `-mfma`：确保 FMA Intrinsics 可用。
- 不建议一开始就使用 `-Ofast` 或 `-ffast-math`，它们可能改变浮点计算语义。

另外，当前代码的：

```
#if defined(__AVX2__)
```

是编译期判断，不是运行时 CPU 检测。

如果要发布一个通用 Linux 二进制，我建议采用：

```
__attribute__((target("avx2,fma")))
static void matvec_avx2(...);
```

再使用：

```
__builtin_cpu_supports("avx2")
__builtin_cpu_supports("fma")
```

进行运行时选择。

这样可以保留一个标量版本，同时让支持 AVX2/FMA 的 CPU 自动选择优化路径。

# 五、我建议建立一套真正的性能基准

目前你们已经有正确性测试，但还缺少内核级 Benchmark。

不要只比较最终的 tok/s，因为这样很难判断到底是哪一个环节拖慢了速度。

建议拆成：

| Benchmark | 测量内容                  |
| --------- | --------------------- |
| MatVec    | AVX2 / Scalar         |
| RMSNorm   | Scalar / AVX2 / Fused |
| RoPE      | 在线计算 / 预计算            |
| Attention | KV Cache 读取与计算        |
| Sampling  | Top-P 采样              |
| Decode    | 单 Token 完整前向          |
| Batch     | B=1、2、4、8、16          |
| Threads   | T=1、2、4、8             |

### 推荐测试矩阵

MacroStories Benchmark 配置

推理模式

KV Cache

Full Recompute

Batch Size

线程数

最大生成 Token

512

&#x20;复制测试配置

实际测试建议：

```
# 纯 KV Decode
./macrostories --max 512 --quiet

# Scalar
./macrostories --max 512 --no-simd --quiet

# 多线程
./macrostories --threads 2 --max 512 --quiet

# Batch
./macrostories --batch-gemm 4 --max 512 --quiet

# Batch + Threads
./macrostories --threads 2 --batch-gemm 4 --max 512 --quiet
```

使用 Linux perf 观察 CPU：

```
perf stat \
    -e cycles,instructions,cache-misses,branches,branch-misses \
    ./macrostories --max 512 --quiet
```

每组运行至少 5～10 次，记录中位数，最好再记录 P95 延迟。

特别注意：你们现在的计时从生成开始到生成结束，不包含模型加载，这一点是合理的。

粘贴的文本 (1).txt



# 六、我认为 MacroStories 最终应该发展成什么架构

目前你们的实现基本上是一个单文件推理程序。我建议下一步不要继续往单文件里堆功能，而是进行一次轻量级重构。

保持 C++17、零第三方依赖，但分离职责。

## MacroStories Runtime

Generation API

generate / generate_batch / sampling

Decode Engine

KV Cache / Batch / Recurrent Steps

Kernel Engine

MatVec / RMSNorm / RoPE / Attention

Model Loader

Safetensors / Validation

Tokenizer

JSON / Byte Decoder

CPU Backend

Scalar Reference / AVX2 + FMA / Runtime Dispatch

建议的目录：

```
macrostories/
├── CMakeLists.txt
├── include/
│   ├── model.h
│   ├── tensor.h
│   ├── runtime.h
│   ├── cache.h
│   └── tokenizer.h
│
├── src/
│   ├── model_loader.cpp
│   ├── safetensors.cpp
│   ├── tokenizer.cpp
│   ├── runtime.cpp
│   ├── decode.cpp
│   ├── attention.cpp
│   └── sampling.cpp
│
├── kernels/
│   ├── scalar.cpp
│   ├── avx2.cpp
│   ├── rope.cpp
│   └── matvec.cpp
│
├── benchmarks/
│   ├── bench_matvec.cpp
│   ├── bench_decode.cpp
│   └── bench_batch.cpp
│
└── tests/
    ├── test_kv.cpp
    ├── test_simd.cpp
    ├── test_batch.cpp
    └── test_tokenizer.cpp
```

# 七、我的最终优化优先级

如果是我来继续开发这个后端，我会这样安排：

| 阶段   | 工作                       | 目标           |
| ---- | ------------------------ | ------------ |
| 第一阶段 | Workspace + RoPE 预计算     | 消除热路径分配和重复计算 |
| 第二阶段 | RMSNorm 融合 + Sampling 优化 | 减少内存读写       |
| 第三阶段 | AVX2 MatVec 重构           | 提高 CPU 指令利用率 |
| 第四阶段 | Batch Kernel 优化          | 提高多序列吞吐量     |
| 第五阶段 | Runtime Dispatch + 模型校验  | 工程化和跨 CPU 兼容 |
| 第六阶段 | PGO + 性能调优               | 根据实际运行数据优化   |

有一点我特别想强调：暂时不要急着做 INT8、INT4 量化。

你们这个模型只有约 20K 参数，FP32 权重本身就很小。量化减少的内存占用有限，反而会增加反量化开销。对于这种小模型，优先优化调度和内核更合理。

## 我还想确认一个关键问题

你们现在做 MacroStories 的定位是什么？这会决定后续应该采用哪种优化策略。

1\. 最主要的性能目标

最大化 tok/s（批量生成大量故事）

最低单序列延迟（实时生成）

同时兼顾吞吐量与低延迟

2\. 最终产品形态

独立 CLI 推理程序

可嵌入其他 C++ 项目的静态/动态库

本地推理 Server / HTTP API

CLI + Library + Server

3\. 目标硬件

Intel CPU（Core Ultra）

跨平台 x86-64（Intel + AMD）

未来也考虑 ARM

提交开发目标

我倾向于把它发展成一个独立、可嵌入、无第三方依赖的 MacroStories Runtime，而不是只做一个更快的命令行程序。这样你们现在已经写好的 KV Cache、SIMD 和 Batch Decode 都能成为长期可复用的基础。
