// MacroStories HTTP API server (Linux only, io_uring based).
//
// Endpoints (OpenAI-compatible subset, see openapi.yaml):
//   GET  /health
//   GET  /v1/models
//   POST /v1/completions
//   POST /v1/stories
//
// Design:
//  * io_uring carries every socket operation (accept / recv / send) and the
//    request log append, so the request path is one completion queue.
//  * The model is loaded once and is read-only afterwards. Generation is CPU
//    bound and runs inline on the event loop; a 400-token story takes ~10ms
//    at ~110k tok/s on this machine, so a single thread is adequate for the
//    synchronous HTTP clients this targets. For heavy concurrency run several
//    processes behind a front proxy (SO_REUSEPORT is set).
//  * Dependencies: liburing + libc. Model code is shared with cpp/stories.cpp.

#define MACROSTORIES_NO_MAIN
#include "../cpp/stories.cpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <liburing.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal JSON reader (objects/arrays/strings/numbers/bools/null)
// ---------------------------------------------------------------------------
namespace mjson {

struct Value {
    enum Type { Null, Bool, Num, Str, Arr, Obj } type = Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Value> arr;
    std::vector<std::pair<std::string, Value>> obj;

    const Value* get(const std::string& key) const {
        if (type != Obj) return nullptr;
        for (const auto& kv : obj) {
            if (kv.first == key) return &kv.second;
        }
        return nullptr;
    }
};

struct Parser {
    const char* p;
    const char* end;
    explicit Parser(const std::string& s) : p(s.data()), end(s.data() + s.size()) {}

    bool parse(Value& v) {
        ws();
        if (!parseValue(v)) return false;
        ws();
        return p == end;
    }

  private:
    void ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
    }

    bool parseValue(Value& v) {
        ws();
        if (p >= end) return false;
        switch (*p) {
            case '{': return parseObject(v);
            case '[': return parseArray(v);
            case '"': v.type = Value::Str; return parseString(v.str);
            case 't':
                if (end - p < 4 || std::strncmp(p, "true", 4) != 0) return false;
                p += 4;
                v.type = Value::Bool;
                v.b = true;
                return true;
            case 'f':
                if (end - p < 5 || std::strncmp(p, "false", 5) != 0) return false;
                p += 5;
                v.type = Value::Bool;
                v.b = false;
                return true;
            case 'n':
                if (end - p < 4 || std::strncmp(p, "null", 4) != 0) return false;
                p += 4;
                v.type = Value::Null;
                return true;
            default: return parseNumber(v);
        }
    }

    bool parseNumber(Value& v) {
        const char* start = p;
        if (p < end && (*p == '-' || *p == '+')) ++p;
        bool digits = false;
        while (p < end) {
            const char c = *p;
            if (c >= '0' && c <= '9') {
                digits = true;
                ++p;
            } else if (c == '.' || c == 'e' || c == 'E' || c == '-' || c == '+') {
                ++p;
            } else {
                break;
            }
        }
        if (!digits) return false;
        v.type = Value::Num;
        v.num = std::strtod(std::string(start, p).c_str(), nullptr);
        return true;
    }

    bool parseString(std::string& out) {
        if (p >= end || *p != '"') return false;
        ++p;
        out.clear();
        while (p < end) {
            const char c = *p++;
            if (c == '"') return true;
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (p >= end) return false;
            switch (*p++) {
                case 'n': out.push_back('\n'); break;
                case 't': out.push_back('\t'); break;
                case 'r': out.push_back('\r'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'u': {
                    if (end - p < 4) return false;
                    unsigned cp = 0;
                    for (int i = 0; i < 4; ++i) {
                        const char h = *p++;
                        cp <<= 4;
                        if (h >= '0' && h <= '9') cp |= static_cast<unsigned>(h - '0');
                        else if (h >= 'a' && h <= 'f') cp |= static_cast<unsigned>(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') cp |= static_cast<unsigned>(h - 'A' + 10);
                        else return false;
                    }
                    if (cp < 0x80) {
                        out.push_back(static_cast<char>(cp));
                    } else if (cp < 0x800) {
                        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                    } else {
                        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                    }
                    break;
                }
                default: return false;
            }
        }
        return false;
    }

    bool parseArray(Value& v) {
        ++p;  // '['
        v.type = Value::Arr;
        ws();
        if (p < end && *p == ']') { ++p; return true; }
        while (true) {
            Value item;
            if (!parseValue(item)) return false;
            v.arr.push_back(std::move(item));
            ws();
            if (p >= end) return false;
            if (*p == ',') { ++p; continue; }
            if (*p == ']') { ++p; return true; }
            return false;
        }
    }

    bool parseObject(Value& v) {
        ++p;  // '{'
        v.type = Value::Obj;
        ws();
        if (p < end && *p == '}') { ++p; return true; }
        while (true) {
            ws();
            std::string key;
            if (!parseString(key)) return false;
            ws();
            if (p >= end || *p != ':') return false;
            ++p;
            Value item;
            if (!parseValue(item)) return false;
            v.obj.emplace_back(std::move(key), std::move(item));
            ws();
            if (p >= end) return false;
            if (*p == ',') { ++p; continue; }
            if (*p == '}') { ++p; return true; }
            return false;
        }
    }
};

inline void escape(const std::string& in, std::string& out) {
    for (unsigned char c : in) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
}

}  // namespace mjson

// ---------------------------------------------------------------------------
// Model engine (loaded once, read-only afterwards)
// ---------------------------------------------------------------------------
static bool file_exists(const std::string& path) {
    return ::access(path.c_str(), F_OK) == 0;
}

struct Engine {
    Weights w{};
    Tokenizer tk;
    std::string openapi;
    const char* name = "macrostories-20k";

    bool load(const std::string& root) {
        if (!file_exists(root + "/model.safetensors")) {
            std::fprintf(stderr, "missing %s/model.safetensors\n", root.c_str());
            return false;
        }
        w = load_weights(root + "/model.safetensors");
        tk = load_tokenizer(root + "/tokenizer.json");
        return true;
    }

    bool load_openapi(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        std::ostringstream ss;
        ss << f.rdbuf();
        openapi = ss.str();
        return !openapi.empty();
    }

    void generate(int seed, int max_tokens, float temperature, float top_p,
                  std::string& text, int& n_tokens, bool& eos) const {
        std::vector<int> ids = generate_ids(w, seed, max_tokens, temperature, top_p, true);
        n_tokens = static_cast<int>(ids.size());
        eos = !ids.empty() && ids.back() == EOS;
        text = detokenize(ids, tk);
    }
};

// ---------------------------------------------------------------------------
// Incremental detokenizer for streaming.
//
// ByteLevel tokens can split a UTF-8 sequence across token boundaries, so we
// buffer incomplete bytes until the next token completes them. Trailing spaces
// are held back until we know whether more text follows, then flushed by
// finish(); this mirrors detokenize()'s leading/trailing trim closely enough
// for streaming clients.
// ---------------------------------------------------------------------------
static size_t valid_utf8_prefix(const std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        size_t n = 0;
        if (c < 0x80) n = 1;
        else if ((c & 0xE0) == 0xC0) n = 2;
        else if ((c & 0xF0) == 0xE0) n = 3;
        else if ((c & 0xF8) == 0xF0) n = 4;
        else break;  // invalid lead byte: keep nothing past it
        if (i + n > s.size()) break;  // incomplete: wait for more bytes
        bool ok = true;
        for (size_t j = 1; j < n; ++j) {
            if ((static_cast<unsigned char>(s[i + j]) & 0xC0) != 0x80) { ok = false; break; }
        }
        if (!ok) break;
        i += n;
    }
    return i;
}

struct IncDecoder {
    std::string pending;  // incomplete UTF-8 bytes
    std::string text;     // decoded, not yet emitted
    bool started = false;

    std::string take() {
        if (!started) {
            const size_t a = text.find_first_not_of(" \t\n\r");
            if (a == std::string::npos) { text.clear(); return {}; }
            text.erase(0, a);
            started = true;
        }
        const size_t b = text.find_last_not_of(" \t\n\r");
        if (b == std::string::npos) return {};  // all trailing spaces; hold
        std::string out = text.substr(0, b + 1);
        text.erase(0, b + 1);
        return out;
    }

    std::string feed(int id, const Tokenizer& tk) {
        if (id == 4) { text += "Alex"; return take(); }
        if (id == 5) { text += "Robin"; return take(); }
        if (id == 6) { text += "Casey"; return take(); }
        if (id == BOS || id == EOS || id == 0 || id == 3) return {};
        if (id < 0 || id >= VOCAB) return {};
        decode_token_bytes(tk.id2tok[id], tk, pending);
        const size_t n = valid_utf8_prefix(pending);
        text.append(pending.data(), n);
        pending.erase(0, n);
        return take();
    }

    std::string finish() {
        text += pending;
        pending.clear();
        if (!started) {
            const size_t a = text.find_first_not_of(" \t\n\r");
            if (a == std::string::npos) { text.clear(); return {}; }
            text.erase(0, a);
            started = true;
        }
        const size_t b = text.find_last_not_of(" \t\n\r");
        std::string out = (b == std::string::npos) ? std::string() : text.substr(0, b + 1);
        text.clear();
        return out;
    }
};

struct Stream {
    bool active = false;
    bool done = false;
    bool eos = false;
    int seed = 0;
    int max_tokens = 0;
    int pos = 0;
    int n_tokens = 0;
    int cur = BOS;
    float temp = 0.5f;
    float top_p = 0.9f;
    std::unique_ptr<KVCache> cache;
    std::mt19937 rng;
    std::vector<float> logits;
    IncDecoder dec;
    std::string id;
};

struct Conn {
    int fd = -1;
    std::string in;            // bytes received but not yet consumed
    std::string out;           // bytes pending send
    size_t out_off = 0;
    std::vector<char> rbuf;    // per-connection recv buffer
    Stream stream;
    bool closed = false;
};

// ---------------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------------
struct Response {
    int status = 200;
    std::string body;
    bool stream = false;  // response has already been written to Conn::out
};

static const char* status_text(int s) {
    switch (s) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 413: return "Payload Too Large";
        case 500: return "Internal Server Error";
        default: return "Error";
    }
}

static Response error_response(int status, const char* type, const std::string& message) {
    Response r;
    r.status = status;
    r.body = "{\"error\":{\"type\":\"";
    r.body += type;
    r.body += "\",\"message\":\"";
    mjson::escape(message, r.body);
    r.body += "\"}}";
    return r;
}

enum class ParseResult { Complete, Incomplete, Bad };

struct ParsedRequest {
    std::string method;
    std::string path;
    std::string body;
    size_t consumed = 0;
};

static ParseResult extract_request(const std::string& raw, ParsedRequest& out) {
    const size_t hdr_end = raw.find("\r\n\r\n");
    if (hdr_end == std::string::npos) {
        return raw.size() > 64 * 1024 ? ParseResult::Bad : ParseResult::Incomplete;
    }
    const std::string head = raw.substr(0, hdr_end);
    const size_t sp1 = head.find(' ');
    if (sp1 == std::string::npos) return ParseResult::Bad;
    const size_t sp2 = head.find(' ', sp1 + 1);
    if (sp2 == std::string::npos) return ParseResult::Bad;
    out.method = head.substr(0, sp1);
    out.path = head.substr(sp1 + 1, sp2 - sp1 - 1);
    const size_t q = out.path.find('?');
    if (q != std::string::npos) out.path.resize(q);

    size_t content_length = 0;
    std::string lower = head;
    for (auto& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    const size_t cl = lower.find("\r\ncontent-length:");
    if (cl != std::string::npos) {
        content_length = static_cast<size_t>(std::strtoul(head.c_str() + cl + 17, nullptr, 10));
    }
    const size_t body_start = hdr_end + 4;
    if (raw.size() < body_start + content_length) return ParseResult::Incomplete;
    out.body = raw.substr(body_start, content_length);
    out.consumed = body_start + content_length;
    return ParseResult::Complete;
}

static void serialize(const Response& r, std::string& out) {
    char head[256];
    std::snprintf(head, sizeof head,
                  "HTTP/1.1 %d %s\r\n"
                  "Content-Type: application/json\r\n"
                  "Content-Length: %zu\r\n"
                  "Connection: keep-alive\r\n"
                  "Access-Control-Allow-Origin: *\r\n"
                  "Access-Control-Allow-Headers: content-type\r\n"
                  "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                  "\r\n",
                  r.status, status_text(r.status), r.body.size());
    out = head;
    out += r.body;
}

// ---------------------------------------------------------------------------
// Streaming (SSE, OpenAI `stream: true`)
// ---------------------------------------------------------------------------
static constexpr int kStreamBatch = 16;  // tokens generated per flush

static void append_chunk(std::string& out, const std::string& data) {
    if (data.empty()) return;
    char hdr[32];
    std::snprintf(hdr, sizeof hdr, "%zx\r\n", data.size());
    out += hdr;
    out += data;
    out += "\r\n";
}

static std::string sse_delta(const std::string& id, const char* model, const std::string& text) {
    std::string e = "data: {\"id\":\"";
    e += id;
    e += "\",\"object\":\"text_completion\",\"created\":0,\"model\":\"";
    e += model;
    e += "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"";
    mjson::escape(text, e);
    e += "\"},\"finish_reason\":null}]}\n\n";
    return e;
}

static std::string sse_final(const std::string& id, const char* model, bool eos, int n_tokens) {
    std::string e = "data: {\"id\":\"";
    e += id;
    e += "\",\"object\":\"text_completion\",\"created\":0,\"model\":\"";
    e += model;
    e += "\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"";
    e += eos ? "stop" : "length";
    e += "\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":";
    e += std::to_string(n_tokens);
    e += ",\"total_tokens\":";
    e += std::to_string(n_tokens + 1);
    e += "}}\n\n";
    return e;
}

static void start_stream(Conn* c, int seed, int max_tokens, float temp, float top_p) {
    Stream& s = c->stream;
    s.active = true;
    s.done = false;
    s.eos = false;
    s.seed = seed;
    s.max_tokens = max_tokens;
    s.pos = 0;
    s.n_tokens = 0;
    s.cur = BOS;
    s.temp = temp;
    s.top_p = top_p;
    s.cache = std::make_unique<KVCache>(max_tokens + 2);
    s.logits.assign(VOCAB, 0.0f);
    s.rng.seed(static_cast<std::mt19937::result_type>(static_cast<uint32_t>(seed)));
    s.dec = IncDecoder{};
    s.id = "cmpl-" + std::to_string(seed);

    c->out += "HTTP/1.1 200 OK\r\n"
              "Content-Type: text/event-stream\r\n"
              "Transfer-Encoding: chunked\r\n"
              "Cache-Control: no-cache\r\n"
              "Connection: keep-alive\r\n"
              "Access-Control-Allow-Origin: *\r\n"
              "Access-Control-Allow-Headers: content-type\r\n"
              "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
              "\r\n";
}

// Generate one batch, append SSE chunks (and the terminal frame when done).
static void stream_batch(Engine& engine, Conn* c) {
    Stream& s = c->stream;
    if (!s.active || s.done) return;

    std::string delta;
    int produced = 0;
    while (!s.done && produced < kStreamBatch) {
        forward_token(engine.w, s.cur, s.pos, *s.cache, s.logits.data());
        const int nxt = sample(s.logits.data(), s.temp, s.top_p, s.rng);
        ++s.pos;
        ++s.n_tokens;
        if (nxt == EOS) {
            s.eos = true;
            s.done = true;
            break;
        }
        s.cur = nxt;
        delta += s.dec.feed(nxt, engine.tk);
        ++produced;
        if (s.pos >= s.max_tokens) {
            s.done = true;
            break;
        }
    }
    if (!delta.empty()) append_chunk(c->out, sse_delta(s.id, engine.name, delta));
    if (s.done) {
        const std::string tail = s.dec.finish();
        if (!tail.empty()) append_chunk(c->out, sse_delta(s.id, engine.name, tail));
        append_chunk(c->out, sse_final(s.id, engine.name, s.eos, s.n_tokens));
        append_chunk(c->out, "data: [DONE]\n\n");
        c->out += "0\r\n\r\n";
    }
}

static Response route(Engine& engine, const ParsedRequest& req, Conn* c) {
    if (req.method == "OPTIONS") {
        Response r;
        r.body = "{}";
        return r;
    }
    if (req.method == "GET" && req.path == "/health") {
        Response r;
        r.body = std::string("{\"status\":\"ok\",\"model\":\"") + engine.name + "\"}";
        return r;
    }
    if (req.method == "GET" && req.path == "/v1/models") {
        Response r;
        r.body = std::string("{\"object\":\"list\",\"data\":[{\"id\":\"") + engine.name +
                 "\",\"object\":\"model\",\"owned_by\":\"macrostories\"}]}";
        return r;
    }
    if (req.method == "GET" && req.path == "/openapi.json") {
        if (engine.openapi.empty()) {
            return error_response(404, "not_found", "openapi spec not loaded (use --openapi)");
        }
        Response r;
        r.body = engine.openapi;
        return r;
    }
    if (req.path != "/v1/completions" && req.path != "/v1/stories") {
        return error_response(404, "not_found", "unknown path: " + req.path);
    }
    if (req.method != "POST") {
        return error_response(405, "method_not_allowed", "use POST");
    }

    mjson::Value v;
    mjson::Parser parser(req.body);
    if (!parser.parse(v) || v.type != mjson::Value::Obj) {
        return error_response(400, "invalid_request_error", "body must be a JSON object");
    }

    int max_tokens = 400;
    float temperature = 0.5f;
    float top_p = 0.9f;
    int seed = 20261007;

    if (const auto* p = v.get("max_tokens")) {
        if (p->type != mjson::Value::Num) {
            return error_response(400, "invalid_request_error", "max_tokens must be a number");
        }
        max_tokens = static_cast<int>(p->num);
    }
    if (const auto* p = v.get("temperature")) {
        if (p->type != mjson::Value::Num) {
            return error_response(400, "invalid_request_error", "temperature must be a number");
        }
        temperature = static_cast<float>(p->num);
    }
    if (const auto* p = v.get("top_p")) {
        if (p->type != mjson::Value::Num) {
            return error_response(400, "invalid_request_error", "top_p must be a number");
        }
        top_p = static_cast<float>(p->num);
    }
    if (const auto* p = v.get("seed")) {
        if (p->type != mjson::Value::Num) {
            return error_response(400, "invalid_request_error", "seed must be a number");
        }
        seed = static_cast<int>(p->num);
    }
    if (const auto* p = v.get("prompt")) {
        if (p->type != mjson::Value::Str && p->type != mjson::Value::Null) {
            return error_response(400, "invalid_request_error", "prompt must be a string");
        }
    }
    bool stream = false;
    if (const auto* p = v.get("stream")) {
        if (p->type == mjson::Value::Bool) {
            stream = p->b;
        } else if (p->type == mjson::Value::Num) {
            stream = p->num != 0;
        } else {
            return error_response(400, "invalid_request_error", "stream must be a boolean");
        }
    }
    if (max_tokens < 1 || max_tokens > 2048) {
        return error_response(400, "invalid_request_error", "max_tokens must be in [1, 2048]");
    }
    if (temperature <= 0.0f) {
        return error_response(400, "invalid_request_error", "temperature must be > 0");
    }
    if (top_p <= 0.0f || top_p > 1.0f) {
        return error_response(400, "invalid_request_error", "top_p must be in (0, 1]");
    }

    if (stream) {
        if (c == nullptr) {
            return error_response(500, "internal_error", "stream requires a connection");
        }
        if (c->stream.active) {
            return error_response(400, "invalid_request_error", "a stream is already active");
        }
        start_stream(c, seed, max_tokens, temperature, top_p);
        stream_batch(engine, c);
        Response r;
        r.stream = true;
        return r;
    }

    std::string text;
    int n_tokens = 0;
    bool eos = false;
    engine.generate(seed, max_tokens, temperature, top_p, text, n_tokens, eos);

    Response r;
    r.body.reserve(text.size() + 512);
    r.body += "{\"id\":\"cmpl-";
    r.body += std::to_string(seed);
    r.body += "-";
    r.body += std::to_string(n_tokens);
    r.body += "\",\"object\":\"text_completion\",\"created\":0,\"model\":\"";
    r.body += engine.name;
    r.body += "\",\"choices\":[{\"index\":0,\"text\":\"";
    mjson::escape(text, r.body);
    r.body += "\",\"finish_reason\":\"";
    r.body += eos ? "stop" : "length";
    r.body += "\",\"logprobs\":null}],\"usage\":{\"prompt_tokens\":1,"
              "\"completion_tokens\":";
    r.body += std::to_string(n_tokens);
    r.body += ",\"total_tokens\":";
    r.body += std::to_string(n_tokens + 1);
    r.body += "}}";
    return r;
}

// ---------------------------------------------------------------------------
// io_uring event loop
// ---------------------------------------------------------------------------
static constexpr size_t kRecvBuf = 64 * 1024;   // per-read size
static constexpr size_t kMaxIn = 1 << 20;       // max buffered request
static bool g_debug = false;

#define DBG(...) do { if (g_debug) std::fprintf(stderr, __VA_ARGS__); } while (0)

struct Op {
    enum Kind { Accept, Recv, Send, Log, Close } kind = Recv;
    Conn* conn = nullptr;
    std::string* log = nullptr;
};

struct Server {
    io_uring ring{};
    int listen_fd = -1;
    int log_fd = -1;
    Engine* engine = nullptr;
    size_t n_conns = 0;

    bool init_socket(const char* addr, int port) {
        listen_fd = ::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
        if (listen_fd < 0) { std::perror("socket"); return false; }
        int one = 1;
        ::setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        ::setsockopt(listen_fd, SOL_SOCKET, SO_REUSEPORT, &one, sizeof one);
        sockaddr_in sa{};
        sa.sin_family = AF_INET;
        sa.sin_port = htons(static_cast<uint16_t>(port));
        if (::inet_pton(AF_INET, addr, &sa.sin_addr) != 1) {
            std::fprintf(stderr, "bad bind address: %s\n", addr);
            return false;
        }
        if (::bind(listen_fd, reinterpret_cast<sockaddr*>(&sa), sizeof sa) < 0) {
            std::perror("bind");
            return false;
        }
        if (::listen(listen_fd, 512) < 0) { std::perror("listen"); return false; }
        return true;
    }

    bool init_ring() {
        if (io_uring_queue_init(512, &ring, 0) < 0) {
            std::perror("io_uring_queue_init");
            return false;
        }
        return true;
    }

    void arm_accept() {
        auto* op = new Op{Op::Accept, nullptr, nullptr};
        io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        if (!sqe) { delete op; return; }
        io_uring_prep_accept(sqe, listen_fd, nullptr, nullptr, SOCK_NONBLOCK);
        io_uring_sqe_set_data(sqe, op);
    }

    void arm_recv(Conn* c) {
        auto* op = new Op{Op::Recv, c, nullptr};
        io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        if (!sqe) { delete op; return; }
        io_uring_prep_recv(sqe, c->fd, c->rbuf.data(), c->rbuf.size(), 0);
        io_uring_sqe_set_data(sqe, op);
    }

    void arm_send(Conn* c) {
        if (c->out_off >= c->out.size()) return;
        auto* op = new Op{Op::Send, c, nullptr};
        io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        if (!sqe) { delete op; return; }
        io_uring_prep_send(sqe, c->fd, c->out.data() + c->out_off,
                           c->out.size() - c->out_off, MSG_NOSIGNAL);
        io_uring_sqe_set_data(sqe, op);
    }

    void arm_log(const std::string& line) {
        if (log_fd < 0) return;
        auto* buf = new std::string(line);
        auto* op = new Op{Op::Log, nullptr, buf};
        io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        if (!sqe) { delete buf; delete op; return; }
        io_uring_prep_write(sqe, log_fd, buf->data(), buf->size(), -1);
        io_uring_sqe_set_data(sqe, op);
    }

    void close_conn(Conn* c) {
        if (c->closed) return;
        c->closed = true;
        ::close(c->fd);
        if (n_conns > 0) --n_conns;
        delete c;
    }

    // Process as many complete requests as the buffer holds. Returns true if a
    // send was queued (so the caller should not arm a recv yet).
    bool process(Conn* c) {
        while (!c->closed) {
            ParsedRequest req;
            const ParseResult pr = extract_request(c->in, req);
            // No complete request left, but there may already be a response
            // queued from an earlier iteration; the caller must flush it.
            if (pr == ParseResult::Incomplete) return c->out_off < c->out.size();
            if (pr == ParseResult::Bad) {
                Response r = error_response(400, "invalid_request_error", "malformed HTTP request");
                std::string wire;
                serialize(r, wire);
                c->out += wire;
                c->in.clear();
                return true;
            }
            c->in.erase(0, req.consumed);
            if (c->in.size() > kMaxIn) {
                Response r = error_response(413, "payload_too_large", "request too large");
                std::string wire;
                serialize(r, wire);
                c->out += wire;
                c->in.clear();
                return true;
            }
            const auto t0 = std::chrono::steady_clock::now();
            Response r = route(*engine, req, c);
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (r.stream) {
                char line[256];
                std::snprintf(line, sizeof line, "%s %s %d %.2fms stream\n",
                              req.method.c_str(), req.path.c_str(), r.status, ms);
                arm_log(line);
                return true;  // caller arms send; the stream resumes after each send
            }
            std::string wire;
            serialize(r, wire);
            c->out += wire;

            char line[256];
            std::snprintf(line, sizeof line, "%s %s %d %.2fms\n",
                          req.method.c_str(), req.path.c_str(), r.status, ms);
            arm_log(line);

            if (c->out.size() > 64 * 1024) return true;  // flush first
        }
        return c->out_off < c->out.size();
    }
};

int main(int argc, char** argv) {
    std::string root = "..";
    std::string log_path;
    std::string openapi_path = "openapi.json";
    const char* bind_addr = "127.0.0.1";
    int port = 8080;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--root" && i + 1 < argc) root = argv[++i];
        else if (a == "--bind" && i + 1 < argc) bind_addr = argv[++i];
        else if (a == "--port" && i + 1 < argc) port = std::atoi(argv[++i]);
        else if (a == "--log" && i + 1 < argc) log_path = argv[++i];
        else if (a == "--openapi" && i + 1 < argc) openapi_path = argv[++i];
        else if (a == "--debug") g_debug = true;
        else if (a == "--help") {
            std::printf("usage: %s --root DIR [--bind ADDR] [--port N] [--log FILE] "
                        "[--openapi FILE]\n", argv[0]);
            return 0;
        }
    }

    Engine engine;
    if (!engine.load(root)) return 1;
    if (!engine.load_openapi(openapi_path)) {
        std::fprintf(stderr, "warning: cannot read openapi spec at %s\n", openapi_path.c_str());
    }

    Server server;
    server.engine = &engine;
    if (!server.init_socket(bind_addr, port)) return 1;
    if (!server.init_ring()) return 1;
    if (!log_path.empty()) {
        server.log_fd = ::open(log_path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (server.log_fd < 0) { std::perror("open log"); return 1; }
    }

    std::fprintf(stderr, "macrostories server on http://%s:%d (root=%s)\n", bind_addr, port,
                 root.c_str());

    server.arm_accept();
    io_uring_submit(&server.ring);

    while (true) {
        io_uring_cqe* cqe = nullptr;
        if (io_uring_wait_cqe(&server.ring, &cqe) < 0) break;
        auto* op = static_cast<Op*>(io_uring_cqe_get_data(cqe));
        const int res = cqe->res;
        io_uring_cqe_seen(&server.ring, cqe);
        if (!op) continue;
        DBG("cqe kind=%d res=%d fd=%d\n", static_cast<int>(op->kind), res,
            op->conn ? op->conn->fd : -1);

        switch (op->kind) {
            case Op::Accept: {
                if (res >= 0) {
                    int one = 1;
                    ::setsockopt(res, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
                    auto* c = new Conn();
                    c->fd = res;
                    c->rbuf.resize(kRecvBuf);
                    ++server.n_conns;
                    server.arm_recv(c);
                }
                server.arm_accept();
                delete op;
                break;
            }
            case Op::Recv: {
                Conn* c = op->conn;
                delete op;
                if (c->closed) break;
                if (res <= 0) { server.close_conn(c); break; }
                c->in.append(c->rbuf.data(), static_cast<size_t>(res));
                if (c->in.size() > kMaxIn) { server.close_conn(c); break; }
                if (server.process(c)) {
                    server.arm_send(c);
                } else {
                    server.arm_recv(c);
                }
                break;
            }
            case Op::Send: {
                Conn* c = op->conn;
                delete op;
                if (c->closed) break;
                if (res <= 0) { server.close_conn(c); break; }
                c->out_off += static_cast<size_t>(res);
                if (c->out_off < c->out.size()) {
                    server.arm_send(c);  // partial write
                } else {
                    c->out.clear();
                    c->out_off = 0;
                    if (c->stream.active) {
                        if (!c->stream.done) {
                            do {
                                stream_batch(*server.engine, c);
                            } while (!c->stream.done && c->out.empty());
                        }
                        if (c->stream.done && c->out.empty()) {
                            // The terminal chunk has been flushed.
                            c->stream.active = false;
                            c->stream.cache.reset();
                            if (server.process(c)) server.arm_send(c);
                            else server.arm_recv(c);
                        } else if (!c->out.empty()) {
                            server.arm_send(c);
                        } else {
                            server.arm_recv(c);
                        }
                    } else {
                        // More pipelined requests may already be buffered.
                        if (server.process(c)) server.arm_send(c);
                        else server.arm_recv(c);
                    }
                }
                break;
            }
            case Op::Log: {
                delete op->log;
                delete op;
                break;
            }
            default:
                delete op;
                break;
        }
        io_uring_submit(&server.ring);
    }

    if (server.log_fd >= 0) ::close(server.log_fd);
    return 0;
}
