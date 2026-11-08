# MacroStories 本地推理工程复盘：经验与教训

> 一份关于「把 2 万参数的故事模型跑到本机极限」的完整复盘：做了什么、为什么这么做、哪里做对了、哪里踩了坑、测量方法如何影响结论、以及下次应该怎么开始。

---

## 1. 项目速览

### 1.1 目标

在本机（Intel Core Ultra 7 155H + Intel Arc MTL 核显）上：
1. 跑通 `raincandy-u/MacroStories`（19,969 参数的故事生成模型）。
2. 比较 PyTorch / ONNX Runtime / ncnn / 手写 C++ 四条路线。
3. 在 C++ 上做 KV cache、SIMD、多线程、批量、采样与 exp 优化。
4. 顺带移植一个 Zig 0.17 单文件版本，体验显式分配。
5. 所有优化都必须可验证（数值对齐 + 自检 + 逐字生成对比）。

### 1.2 模型

| 项 | 值 |
|---|---|
| 参数 / 体积 | **19,969** / **81,364 B**（FP32） |
| 架构 | Ouro：1 个解码层，权重共享循环 **4** 次 |
| 隐藏维 / 中间维 | 32 / 48 |
| 注意力 | 2 Q 头、1 KV 头，head_dim=16 |
| 词表 | 378（WordLevel + GPT-2 ByteLevel 解码） |
| 位置编码 | RoPE（theta=10000） |
| 其它 | tied embedding、value_residual_mix=0.5、max_position=2048 |

### 1.3 设备与运行时

- CPU：Intel Core Ultra 7 155H，22 逻辑线程，AVX2/FMA/F16C/AVX-VNNI，**无 AVX-512**。
- iGPU：Intel Arc (Meteor Lake)，Level Zero 1.34、Vulkan 1.4（Mesa ANV）、OpenCL 均可用。
- 默认 Python 3.14 无 torch；用 `uv` 建 Python 3.12 venv，装 `torch==2.8.0+xpu`、`transformers==4.55.0`、`safetensors`、`onnxruntime`、`pnnx`、`ncnn`。
- 远程测试机：Intel Xeon Platinum（2 vCPU，AVX-512 F/BW/CD/DQ/VL），g++ 14.2。

---

## 2. 时间线与成果

| 阶段 | 工作 | 结果 |
|---|---|---|
| 1 | 环境、权重、Python 跑通 | 发现权重是 LFS 指针；拉回真实 81KB 权重；torch CPU/XPU 跑通 |
| 2 | ONNX / ncnn | ONNX fp32 精确（1.2e-5）；ncnn CPU 精确；Vulkan 慢 18× |
| 3 | 手写 C++ 前向 | 与 Python 参考 max abs diff 5.6e-6 |
| 4 | KV cache | 全量重算 860 tok/s → 13k tok/s；自检 0.0 |
| 5 | AVX2/FMA + 多线程 + 批量 | 单线程 ~25k；22 线程 ~28.5 万；22×b4 ~32.5 万 tok/s |
| 6 | 热路径分配消除 | 单条 malloc 728→344；批量 75,450→2,045 |
| 7 | AVX-512 远程实测 | 比 AVX2 慢 ~17%（降频）；默认保持 AVX2 |
| 8 | Zig 0.17 单文件移植 | 数值对齐 6.5e-6；约 15k tok/s；显式 arena 分配 |
| 9 | 编译器归因 | 等价内核结构下 GCC/LLVM 基本打平；GCC 仅在手写 AVX2 上更强 |
| 10 | 多行 matvec A/B | 无收益（GCC 已做行间 ILP） |
| 11 | top-p 选择 A/B | 完整生成 ~2.3×（36–39 → 13–17µs/token） |
| 12 | 向量化/近似 exp | 完整生成再降 20.7%（11.41 → 9.05µs/token，~110k tok/s） |

---

## 3. 关键数据总表

### 3.1 后端吞吐（同 seed / 温度 0.5 / top-p 0.9）

| 后端 | 吞吐 | 备注 |
|---|---:|---|
| torch XPU（Arc） | ~22 tok/s | 小模型 kernel 启动开销大 |
| torch CPU | ~113 tok/s | 默认多线程 |
| ONNX Runtime CPU | ~342 tok/s | 无 KV cache，全量重算 |
| ncnn CPU | 精确但慢 | 适合极轻运行时 |
| ncnn Vulkan | 比 CPU 慢 ~18× | dispatch 开销主导 |
| **C++ 单线程（最终）** | **~110k tok/s** | top-p 选择 + 向量化 exp |
| **C++ 22 线程** | **~28.5 万 tok/s** | |
| **C++ 22 线程 × batch 4** | **~32.5 万 tok/s** | |

### 3.2 资源占用

| 场景 | 峰值 RSS | CPU |
|---|---:|---:|
| C++ 单条 | **7.6 MB** | 1 核 |
| C++ 22 线程 | 8.4 MB | ~10.5 核 |
| C++ 22×b4 | 21.4 MB | ~12.6 核 |
| torch CPU | 779 MB | ~4 核 |
| torch XPU | 3.2 GB | ~1.2 核 + GPU |
| ORT | 787 MB | ~1.4 核 |

### 3.3 关键优化收益

| 优化 | 收益 | 结论 |
|---|---:|---|
| KV cache | ~15× | 必做 |
| AVX2 SIMD（全内核） | ~1.85× forward | 必做 |
| 多线程 | ~10×（22 线程） | 吞吐主力 |
| batch-gemm | +14%（与线程组合） | 有限 |
| top-p 选择代替全排序 | **~2.3× 完整生成** | 最大单项 |
| 向量化/近似 exp | **−20.7% 完整生成 / −40.6% forward** | 有效 |
| 多行 matvec | ~0 | 无收益 |
| AVX-512 | −17% | 降频，默认关 |
| libmvec exp | 比标量还慢 | 不用 |
| int8 量化 | 体积不降、误差 0.93 | 放弃 |
| Vulkan | 慢 ~18× | 放弃 |

---

## 4. 做对了什么（经验）

### 4.1 先验证环境，而不是先写代码

- 一开始就确认了 CPU 指令集、iGPU 运行时、Python 环境、包管理器，避免在错误的前提上优化。
- 默认 Python 3.14 没有 torch，直接用 `uv` 拉 Python 3.12，避免了和系统 Python 纠缠。

### 4.2 权重先验证，再谈性能

- `model.safetensors` 实际是 **Git LFS 指针（130 字节）**，不是权重。
- 从 HF 拉取真实文件后，用 `sha256` 与 LFS 指针里的 oid 对比，确认无误。
- 教训：**任何“模型跑出乱码”先怀疑权重是否正确加载**。

### 4.3 用「独立复现 + 数值对齐」代替「信任框架」

- ONNX 导出没有直接 trace `trust_remote_code` 的 HF 模型（内部有 DynamicCache、自定义 mask），而是用独立实现加载 safetensors，先与 HF 原模型对齐到 **0.0**，再导出。
- 手写 C++ 同样先与 Python 参考对齐到 **5.6e-6**。
- 好处：任何一步出错都能定位到“数学”还是“工程”。

### 4.4 把「正确性」做成可执行的自检

C++ 版内置：
- `--selftest-kv`：增量 KV 与全量重算对比（最终 0.0）。
- `--selftest-simd`：AVX2/AVX-512/hybrid 与标量对比。
- `--selftest-batch`：批量与逐条生成逐 token 对比。
- `--logits`：与 Python 参考对齐。

这些自检在每次优化后都跑一遍，保证“优化不改结果”。

### 4.5 大项优先：KV cache、SIMD、线程

- KV cache 把全量重算从 860 tok/s 提升到 13k tok/s（~15×）。
- SIMD 再 ~1.85×。
- 多线程/批量把吞吐推到 30 万 tok/s 量级。
- 这些是“结构性”收益，优先做。

### 4.6 用运行时开关做 A/B

- 同一个二进制里用 `--simd scalar|avx2|avx512|hybrid`、`--mv 1|2|4|8`、`--sample-select`、`--exp-scalar` 切换实现。
- 好处：消除“不同编译/不同进程”的噪声，A/B 更可信。

### 4.7 显式分配（Zig）的价值

- Zig 版用 `ArenaAllocator` 一次性承载权重/tokenizer/KV cache/输出，热路径只用栈数组；生命周期清晰、无隐式分配。
- 代价是性能：约 15k tok/s（C++ 最终 ~110k），差距来自内核覆盖、库实现与语言运行时，而不是“Zig 不行”。

---

## 5. 踩过的坑（教训）

### 5.1 测量类（最影响结论）

1. **墙钟时间噪声极大**
   - 同一配置不同轮次可在 12k–18k tok/s 波动（turbo/频率/调度）。
   - 解决：用 `wait4` 拿 **CPU 时间（user+sys）**；长任务用同进程连续跑（`--bench-generate`），减少进程启动和频率漂移。

2. **微基准会被编译器 hoist**
   - 固定 logits 反复调用 `sample` 时，编译器把 softmax/sort 提到循环外，测出 5.3µs；实际 in-context 约 25–30µs。
   - 解决：让输入每轮变化，或用**完整生成 A/B** 作为最终判据。

3. **消融实验会被“生成长度”污染**
   - 去掉排序后 EOS 变化，生成长度从 135 变 400，而 attention 成本随位置增长 → 平均每 token 成本不可比。
   - 解决：固定长度 forward-only 基准（`--bench N`），或在同一位置比较。

4. **加载时间 ≠ 解码时间**
   - Zig 加载反而比 C++ 快（1.5 vs 2.3ms），早期墙钟对比会误判。
   - 解决：拆分 load vs decode。

5. **libmvec 未必更快**
   - 直接调用 `_ZGVdN8v_expf` 在所有长度上都比标量 `expf` 慢（0.32–0.89×）。
   - 教训：**“向量化”不等于“更快”**，必须测。

### 5.2 数值/实现类

1. **RoPE 原地修改导致前后半段互相覆盖**
   - `rotate_half` 需要原始后半段；原地写会读到已覆盖的数据。
   - 解决：先写临时数组，再拷回。

2. **权重指针因 `std::vector` 重分配而失效**
   - 先 `resize` 后保存指针，后续扩容导致野指针。
   - 解决：提前 `reserve`，或保存 offset 而非指针。

3. **tokenizer 的 vocab 在 `model.vocab` 下**
   - 顶层 `vocab` 不存在，Zig 版一度报 `MissingVocab`。
   - 解决：按 tokenizer.json 实际结构读取。

4. **top_p=1.0 时“重复取最大值”退化为 O(V²)**
   - keep 均值 162，选择算法比排序慢一倍。
   - 解决：`top_p >= 0.999` 直接走排序；选择循环设 64 次上限回退。

5. **fp16 / int8 的精度代价**
   - ncnn fp16 存储 logits 误差 0.144；int8 动态量化误差 0.93。
   - 对 20k 模型，量化收益几乎为零。

### 5.3 硬件/平台类

1. **AVX-512 在 Xeon 上降频**
   - 纯点积微基准里 AVX-512 在 n≥32 更快，但整模型持续跑 512-bit FMA 触发降频，整体慢 ~17%。
   - 结论：默认 AVX2，AVX-512 仅作为可选。

2. **Vulkan 的 dispatch 开销**
   - 226 层、极小张量；GPU 每层启动/同步成本远超计算，S=16/512 都比 CPU 慢 ~18×。
   - 结论：小模型别碰 GPU。

3. **big.LITTLE + 超线程**
   - 22 逻辑线程实际只吃到 ~10.5 核；批量可到 ~13.5 核。
   - 线程数不是越多越好，短任务调度有上限。

### 5.4 工程类

1. **不要默认“多行 matvec 一定更快”**
   - 实测 4/8 行并行无收益（甚至更慢）；GCC 已经把行间 ILP 做出来了。
2. **不要默认“手写 JSON 解析器不够”**
   - 本 tokenizer 只有 `\"` 和 `\\`，手写解析足够；通用 JSON 解析器是过度设计。
3. **不要默认“GCC vs LLVM”是性能差距主因**
   - 同源码 dot-only：GCC 15.51µs vs Clang 15.57µs，基本打平。

---

## 6. 测量方法论（下次直接照做）

1. **明确指标**：单序列延迟（µs/token）还是批量吞吐（tok/s）？两者优化方向不同。
2. **分离阶段**：加载 / 前向 / 采样 / 反序列化，分别计时。
3. **CPU 时间优先**：`wait4` 的 `ru_utime+ru_stime`；墙钟只作参考。
4. **足够长的运行**：同进程连续生成数千 token，摊薄启动/调度噪声。
5. **同一二进制 A/B**：运行时开关，避免编译差异。
6. **固定位置/固定长度**：attention 随位置增长，比较必须在同一位置区间。
7. **防 hoist 微基准**：每轮改变输入，或直接测完整生成。
8. **报告分布**：至少 best/median；关注 p90，不只看平均值。
9. **交叉验证**：自检 + Python 参考 + 逐字生成对比。
10. **记录环境**：CPU/频率/编译器版本/负载，否则数字不可复现。

---

## 7. 优化优先级与 ROI

### 7.1 正确的顺序

1. **算法/数据结构**：KV cache、采样算法（top-p 选择）、避免全排序。
2. **库/数学**：向量化 exp、SIMD 内核、融合算子。
3. **并行**：多线程、批量。
4. **微架构**：多行 matvec、prefetch、PGO。
5. **量化/GPU**：对小模型通常负收益。

### 7.2 为什么“先 profile 再优化”

- 我们最初假设“多行 matvec / AVX-512”是最大缺口，实测都不是。
- 真正的最大缺口是 **采样全排序**（in-context 25–30µs/token）和 **exp**（forward 中约 1,000 次/token）。
- 教训：**优化前先拆解成本**；小模型里“库函数/算法选择”经常比“手写 SIMD”更值钱。

### 7.3 “理论极限”怎么看

- 纯 FLOP 下界约 **1µs/token**（60–76k MAC / AVX2 FMA），但这个数没有指导意义。
- 现实下界受 **1,274 个小点积的归约/依赖链**约束：每行 ~8–10 cycles，forward ~4–6µs 是现实目标。
- 采样从 25–30µs 降到 ~5µs；exp 再降 ~2µs。
- 结论：单线程从 ~40µs 降到 ~9µs（~4.4×），已经接近这个体量的实用极限；不存在 10× 空间。

---

## 8. 正确性与可验证性

### 8.1 三层验证

| 层级 | 方法 | 结果 |
|---|---|---|
| 数值 | 与 Python 独立前向对比 logits | 5.3–6.1e-6 |
| 自检 | KV / SIMD / batch 三种 selftest | 0.0 / ≤1.7e-5 / 逐字一致 |
| 生成 | 同 seed 与旧实现逐字对比 | top_p=0.9 4–5 seed 全一致 |

### 8.2 优化的“安全网”

- 每个新算法都保留参考路径：`--simd scalar`、`--exp-scalar`、`--sample-select` 关闭。
- 先证明“结果不变”，再接受性能收益。
- 对近似算法（exp），记录误差并确认生成逐字一致。

---

## 9. 编译器与语言（GCC / LLVM / Zig）

- 版本：GCC 16.2.1、Clang 23.1.4、Zig 0.17.1（LLVM 22）。
- 同 C++ 源码：
  - full AVX2：GCC 11.24µs vs Clang 14.89µs（GCC 内联/展开更好）。
  - dot-only：GCC 15.51µs vs Clang 15.57µs（**基本持平**）。
- 结论：**编译器不是主因**；差距主要来自内核覆盖、库实现和语言运行时。
- Zig：
  - strict FP 比 `@setFloatMode(.optimized)` 慢 ~10%（20.39→18.25µs）。
  - `std.sort.heap` 比 `pdq` 慢 ~4×（31.7→20.8µs/sample）。
  - 机械移植正确（6.5e-6），但性能约 15k tok/s；要追性能需要手写更多内核。

---

## 10. 移植经验（C++ → Zig 0.17）

### 10.1 Zig 0.17 API 变化（实际遇到的）

| 旧写法 | Zig 0.17 |
|---|---|
| `pub fn main() !void` 自行取参数/allocator | `pub fn main(init: std.process.Init) !void`，从 `init.io/.gpa/.arena/.minimal.args` 取 |
| `std.io.getStdOut().writer()` | `std.Io.File.stdout().writer(io, &buf)`，用 `&fw.interface` |
| `std.fs.cwd().readFileAlloc(...)` | `std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(n))` |
| `std.heap.GeneralPurposeAllocator` | `std.heap.DebugAllocator` |
| `std.ArrayList(T).init(alloc)` | `std.ArrayList(T)` 是 unmanaged，方法显式传 allocator |
| `std.sort.pdq` | 有 `pdq`；不要用 `heap`（本模型慢 ~4×） |
| `std.rand.DefaultPrng` | `std.Random.DefaultPrng.init(seed)` → `.random()` |
| 无内建向量 | `@Vector(8, f32)` + `@splat` + `@mulAdd` + `@reduce(.Add, ...)` |

### 10.2 显式分配

- `ArenaAllocator` 管长生命周期；每 token 临时量走栈；热路径无堆分配。
- 这是 Zig 体验的核心：分配器显式传递、生命周期可见。

### 10.3 移植的教训

- **机械移植能保证正确，但不保证性能**；排序/exp/内核覆盖比语言本身影响更大。
- Zig 的 `std.Io`/`main(init)` 变化较大，直接照旧 API 写会编译失败；先读本机 std 源码最省时间。

---

## 11. 决策矩阵

| 需求 | 选择 |
|---|---|
| 快速跑通 / 演示 | PyTorch（`run_story.py`） |
| 跨平台、动态长度 | ONNX Runtime |
| 极轻运行时（几 MB） | ncnn（CPU；Vulkan 不划算） |
| 最高性能 / 最低内存 | **手写 C++（最终 ~110k tok/s 单线程）** |
| 学习显式分配 / 内存安全 | Zig 0.17 单文件移植 |

---

## 12. 如果重来一次（Checklist）

1. 先确认硬件 ISA / GPU 运行时 / Python 环境。
2. 校验权重（LFS / sha256 / dtype / shape）。
3. 用 Python 参考实现 + `--logits` 对齐。
4. 先上 KV cache，再谈 SIMD。
5. 采样先问“keep 有多大”；小 keep 用选择，不用全排序。
6. exp 先做微基准；libmvec 不一定是答案。
7. 每个优化都做同二进制 A/B + CPU 时间 + 长跑。
8. 优化结果必须与旧实现逐字/数值对齐。
9. 对近似算法保留精确回退开关。
10. 小模型不要默认上 GPU / 量化 / 多行手写内核；先测。

---

## 13. 未解问题与后续工作

- **Zig 版追性能**：手写更多 AVX2 内核、`@setFloatMode(.optimized)`、避免全排序；目前 ~15k tok/s。
- **真正批量化 GEMM**：当前 batch 只是锁步前向，B≥4 饱和；更大 batch 的收益需要专门内核与内存布局。
- **AVX-512**：在降频不明显的 CPU（如部分客户端型号 / Sapphire Rapids）上重新测 `--simd avx512`。
- **向量化 exp 的边界**：当前多项式为 6 阶，精度 ~1e-7；若用于更大模型，需重新评估误差累积。
- **GPU**：仅当模型规模上去（百万参数以上）再考虑 Vulkan/Level Zero；20k 参数不划算。
- **文档/基准**：把 `--bench-generate` 等基准固化进仓库，便于回归。

---

## 14. 附录

### 14.1 常用命令

```bash
# Python
.venv/bin/python run_story.py --device cpu|xpu

# ONNX
.venv/bin/python export_onnx.py

# C++
make -C cpp
./cpp/stories --root . --seed 20261007 --max 400
./cpp/stories --root . --selftest-kv
./cpp/stories --root . --selftest-simd
./cpp/stories --root . --selftest-batch
./cpp/stories --root . --exp-scalar          # 精确 exp 回退
./cpp/stories --root . --simd scalar         # 标量 SIMD 回退
./cpp/stories --root . --threads 22 --batch-gemm 4 --quiet

# Zig
cd zig
zig build-exe -O ReleaseFast -fstrip stories.zig -femit-bin=stories
./stories --root .. --seed 20261007 --max 400
./stories --root .. --logits 1,9,10,11,12,16,17
```

### 14.2 文件

| 路径 | 说明 |
|---|---|
| `run_story.py` | PyTorch 运行脚本 |
| `export_onnx.py` | ONNX 导出（独立前向） |
| `converted/macro_stories.onnx` | ONNX fp32 |
| `converted/ncnn/` | ncnn fp32 |
| `cpp/stories.cpp` | 手写 C++ 推理（最终版） |
| `cpp/Makefile` | `-O3 -march=native -pthread` |
| `zig/stories.zig` | Zig 0.17 单文件移植 |
| `zig/README.md` | Zig API 变动与移植说明 |
| `REPORT.md` | 工程报告（数据与结论） |
| `RETROSPECTIVE.md` | 本文（经验与教训） |

### 14.3 Commit 记录

| Commit | 内容 |
|---|---|
| `221e44c` | 手写 C++ 推理（KV cache + SIMD + 多线程） |
| `20597aa` | 锁步批量化解码（batch-gemm） |
| `78e7968` | AVX-512/hybrid、热路径分配消除、REPORT |
| `2616206` | 输入/权重校验、死代码清理、RoPE 频率预计算 |
| `af85f8a` | Zig 0.17 单文件移植与 API 文档 |
| `81fb5db` | 编译器/LLVM 归因文档 |
| `ca45ef4` | top-p 选择采样（`sample_select` + 回退） |
| `cf73b9e` | 向量化/近似 exp（默认开启，`--exp-scalar` 回退） |

> `model.safetensors`（真实权重）与 `converted/` 产物未纳入 git；建议用 Git LFS 管理权重，产物按需生成。
