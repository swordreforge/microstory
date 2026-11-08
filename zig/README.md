# MacroStories — Zig 0.17 单文件移植

零依赖、单文件 `stories.zig` 的 MacroStories 推理实验，重点体验 Zig 的**显式分配/显式 I/O**。算法与 C++ 版一致（KV cache 解码），并通过 `--logits` 与 Python 参考实现对齐。

## 构建与运行

```bash
cd zig
zig build-exe -O ReleaseFast -fstrip stories.zig -femit-bin=stories
./stories --root .. --seed 20261007 --max 400
./stories --root .. --logits 1,9,10,11,12,16,17   # 与 Python 参考对齐
```

## Zig 0.17 API 变动（相对常见旧写法）

| 旧写法 | Zig 0.17 |
|---|---|
| `pub fn main() !void`，自行获取参数/allocator | `pub fn main(init: std.process.Init) !void`，从 `init.io/.gpa/.arena/.minimal.args` 取 |
| `std.io.getStdOut().writer()` | `std.Io.File.stdout().writer(io, &buf)`，再用 `&fw.interface` |
| `std.fs.cwd().readFileAlloc(...)` | `std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(n))`（File/Dir 都要显式 `io`） |
| `std.heap.GeneralPurposeAllocator` | `std.heap.DebugAllocator(.{})`（`.init`、`.allocator()`、`.deinit()`） |
| `std.ArrayList(T).init(alloc)` | `std.ArrayList(T)` 为 unmanaged（`array_list.Aligned`），方法显式传 allocator |
| `std.json.parseFromSlice` | 仍可用；`parseFromSliceLeaky(Value, alloc, s, .{})` 更适合 arena |
| `std.mem.readInt(u64, buf[0..8], .little)` | 需要 `*const [N]u8`；本移植直接按字节拼 |
| `std.rand.DefaultPrng` | `std.Random.DefaultPrng.init(seed)` → `.random()` |
| `std.sort.heap`（误用） | 用 `std.sort.pdq`（PDQ，接近 C++ `std::sort`）；`heap` 在本模型采样上实测慢约 4× |
| 无内建向量 | `@Vector(8, f32)` + `@splat` + `@mulAdd` + `@reduce(.Add, ...)` |

## 显式分配设计

```zig
var arena_state = std.heap.ArenaAllocator.init(init.gpa);
defer arena_state.deinit();
const arena = arena_state.allocator();

const model_bytes = try cwd.readFileAlloc(io, model_path, arena, .limited(1 << 24));
const w = try loadWeights(header, model_bytes, data_base, arena);   // 所有权重来自 arena
var cache = KVCache{
    .k = try arena.alloc(f32, STEPS * max_seq * NKV * HD),
    .v = try arena.alloc(f32, STEPS * max_seq * NKV * HD),
    .scores = try arena.alloc(f32, max_seq),
};
```

- 所有长生命周期内存（权重、tokenizer、KV cache、输出缓冲）一次性从 `arena` 分配，末尾 `deinit` 统一释放。
- 每 token 的临时量（HID/INTER/HD）用栈数组，**热路径无堆分配**。
- `init.gpa` 是运行时的默认分配器；`init.arena` / `init.io` 由 `std.process.Init` 提供。

## 验证与性能

- `--logits` 与 Python 参考（`export_onnx.py` 的独立前向）对齐：**max abs diff 6.5e-6**。
- 生成示例（seed 20261007）可正常产出故事。
- 单线程解码（CPU 时间，排除加载）：Zig **~67µs/token ≈ 15k tok/s**，C++ AVX2 **~36µs/token ≈ 28k tok/s**；`-fstrip` 后二进制约 **358KB**。
- 差距拆分（实测）：
  - **前向**：Zig 22.4µs vs C++ 11.5µs（约 2×）。C++ 对 matvec/rmsnorm/axpy 都手写 AVX2 并展开；Zig 只有 `dot` 用 `@Vector`。
  - **采样**：C++ `std::sort` 7.4µs vs Zig `pdq` 20.8µs（旧版误用 `std.sort.heap` 为 31.7µs）。换 `pdq` 后完整生成约快 **15%**。（该微基准为固定 logits，可能被编译器 hoist；C++ 现已在解码循环改用 top-p 选择，完整生成约 13–17µs/token；Zig 仍是全排序。）
  - **加载不是瓶颈**：Zig 1.5–1.7ms，C++ 2.2–2.4ms（Zig 反而更快）。

## 编译器 / LLVM 归因（GCC 16 vs Zig 0.17 的 LLVM 22）

版本：GCC 16.2.1、Clang 23.1.4、Zig 0.17.1（依赖 LLVM 22）。同一份 C++ 源码、同一内核结构，换编译器（forward-only，CPU 时间）：

| 配置 | forward-only |
|---|---:|
| GCC 16，full AVX2 | 11.24 µs/tok |
| Clang 23，full AVX2 | 14.89 µs/tok |
| GCC 16，dot-only AVX2 | 15.51 µs/tok |
| Clang 23，dot-only AVX2 | 15.57 µs/tok |
| Zig strict（dot-only 覆盖） | 20.39 µs/tok |
| Zig `@setFloatMode(.optimized)` | 18.25 µs/tok |

结论：

- 等价内核结构下 GCC 与 LLVM 基本打平（15.51 vs 15.57）→ **编译器不是主因**。
- GCC 仅在手写 intrinsics 的 full AVX2 代码上有内联/展开优势（11.24 vs 14.89）。
- Zig 与等价 C++ 仍有 ~18–30% 差距，来自切片/`@memcpy`/未展开等语言与运行时细节；strict FP 再损失 ~10%。
- `std.sort.heap` vs `std::sort` 是采样的大头（已换 `pdq`），属库差异而非编译器。

## 与 C++ 版的差异

- 仅实现单序列 KV 解码 + `--logits`；未实现 `--threads` / `--batch-gemm` / 多级 SIMD 选择。
- 采样 RNG 不同（`std.Random.DefaultPrng` vs `std::mt19937`），故事文本不会逐字相同，但 logits 对齐。
- 采样排序用 `std.sort.pdq`（O(V log V)），对 378 词表足够；`heap` 实测慢约 4×。

## 文件

- `stories.zig`：单文件实现。
- `.gitignore`：忽略 `stories` / `*.o` / Zig 缓存目录。
