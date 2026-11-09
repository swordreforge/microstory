# MacroStories HTTP API (Linux + io_uring + OpenAPI)

A small, zero-third-party-dependency HTTP server for the MacroStories model.
All socket I/O (accept / recv / send) and the access-log append go through
**io_uring**. The JSON wire format follows [`openapi.yaml`](./openapi.yaml)
(machine-readable copy: [`openapi.json`](./openapi.json), also served at
`GET /openapi.json`).

## Build

```bash
make -C server          # requires liburing (headers + .so)
```

The binary links `../cpp/stories.cpp` (the model implementation is included
with `MACROSTORIES_NO_MAIN`), so keep the `cpp/` directory next to `server/`.

## Run

From the `server/` directory (default `--root ..` points at the model files):

```bash
./stories_server --root .. --port 8080 --openapi openapi.json --log access.log
```

Or from the repository root:

```bash
./server/stories_server --root . --port 8080 --openapi server/openapi.json
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--root DIR` | `..` | directory containing `model.safetensors` + `tokenizer.json` |
| `--bind ADDR` | `127.0.0.1` | listen address (use `0.0.0.0` behind a proxy) |
| `--port N` | `8080` | listen port |
| `--log FILE` | none | append access log via io_uring |
| `--openapi FILE` | `openapi.json` | spec served at `/openapi.json` |
| `--debug` | off | print io_uring completion events to stderr |

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness / model id |
| GET | `/v1/models` | OpenAI-compatible model list |
| POST | `/v1/completions` | OpenAI-compatible text completion |
| POST | `/v1/stories` | alias of `/v1/completions` |
| GET | `/openapi.json` | this API's OpenAPI document |

### Request

```json
{
  "seed": 20261007,
  "max_tokens": 400,
  "temperature": 0.5,
  "top_p": 0.9
}
```

All fields are optional. `prompt` is accepted for OpenAI-client compatibility
but is currently ignored: the model is a from-scratch story generator and
always starts from `<|start_story|>`.

### Response

```json
{
  "id": "cmpl-20261007-128",
  "object": "text_completion",
  "created": 0,
  "model": "macrostories-20k",
  "choices": [
    { "index": 0, "text": "…", "finish_reason": "stop", "logprobs": null }
  ],
  "usage": { "prompt_tokens": 1, "completion_tokens": 128, "total_tokens": 129 }
}
```

`finish_reason` is `stop` when EOS was sampled, `length` when `max_tokens`
was reached.

### curl examples

```bash
curl -s http://127.0.0.1:8080/health | jq .
curl -s http://127.0.0.1:8080/v1/models | jq .

curl -s -X POST http://127.0.0.1:8080/v1/stories \
  -H 'Content-Type: application/json' \
  -d '{"seed":20261007,"max_tokens":400,"temperature":0.5,"top_p":0.9}' | jq .

curl -s http://127.0.0.1:8080/openapi.json | jq .info
```

Errors use `{"error":{"type":"…","message":"…"}}` with status 400/404/405/413.

## Design notes

- **io_uring**: one ring per process. `accept`, `recv`, `send`, and the
  log-file `write` are all submitted to the same completion queue. Verify with
  `ls -l /proc/<pid>/fd | grep io_uring` → `anon_inode:[io_uring]`.
- **Model lifetime**: loaded once at startup; weights and tokenizer are
  read-only afterwards. Each request allocates its own KV cache and RNG.
- **Keep-alive**: connections are reused; partial recv/send are handled.
  Read buffer is 64 KiB per connection, request cap is 1 MiB.
- **Concurrency model**: generation runs inline on the event loop, so one
  process handles one generation at a time. For CPU parallelism run several
  processes on the same port; `SO_REUSEPORT` is set so the kernel balances
  connections.
- **JSON**: a small hand-written parser/serializer is used (no third-party
  JSON dependency); the format is validated against `openapi.yaml`.
- **CORS**: permissive `Access-Control-Allow-*` headers are returned so browser
  clients can call the API directly.

## Measured on this machine

| Setup | Requests | Tokens | Wall | Throughput |
|---|---:|---:|---:|---:|
| 1 process, 4 client threads | 20 | 2,504 | 16.5 ms | ~1,215 req/s, ~152k tok/s |
| 4 processes (SO_REUSEPORT), 4 client threads | 20 | 2,504 | 12.4 ms | ~1,611 req/s, ~202k tok/s |

> The model itself is ~110k tok/s single-thread; these numbers are dominated by
> short request durations and client overhead, not by the HTTP layer.

## Limitations

- No streaming/SSE, no `n > 1`, no prompt-conditioned generation.
- Single-threaded generation per process; use multiple processes for parallel
  CPU use.
- No TLS or authentication. Bind to `127.0.0.1` or put it behind a reverse
  proxy if exposed.
- `max_tokens` is capped at the model's `max_position_embeddings` (2048).

## Files

| Path | Purpose |
|---|---|
| `stories_server.cpp` | io_uring HTTP server + JSON + routes |
| `openapi.yaml` / `openapi.json` | OpenAPI 3.1 spec |
| `Makefile` | builds `stories_server` with `-luring` |
| `.gitignore` | ignores the binary |
