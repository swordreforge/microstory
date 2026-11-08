# MacroStories 本地推理工程报告

> 目标：在本机（Intel Arc 核显 / Core Ultra 7 155H）上跑通 20k 参数的 MacroStories 故事生成模型，评估各运行后端，并给出一份可复现、可验证、按投入产出比排序的结论。

---

## 1. TL;DR

- 模型只有 **19,969 个参数 / 81,364 字节（FP32）**，1 层权重循环 4 次。
- 本机默认只有 Python 3.14（无 torch）；用 `uv` 建了 Python 3.12 venv，装 `torch==2.8.0+xpu` + `transformers==4.55.0` 跑通。
- 四个后端全部验证通过：**PyTorch（CPU/XPU）、ONNX Runtime、ncnn、手写 C++**。
- **手写 C++ 是当前最优解**：单线程 ~60k tok/s（top-p 选择优化后），多线程 ~28.5 万 tok/s，峰值内存仅 **~7.6MB**。
- **Vulkan 不划算**：本机驱动齐全（Mesa ANV，Vulkan 1.4），但 ncnn Vulkan 比 CPU 慢约 **18×**，小模型被 kernel 启动开销拖死。
- **int8 / AVX-VNNI 可跳过**：模型太小，量化误差（logits 差 0.93）远大于收益。
- **AVX-512 已实测但默认不用**：远程 Xeon 上比 AVX2 慢约 **17%**（持续 512-bit FMA 触发降频），故默认 AVX2，仅保留 `--simd avx512|hybrid` 选项。

---

## 2. 项目与硬件

### 2.1 模型

| 项 | 值 |
|---|---|
| 参数量 | 19,969 |
| 权重体积 | 81,364 字节（FP32） |
| 架构 | Ouro（1 个解码层，权重共享循环 4 次） |
| 隐藏维 / 中间维 | 32 / 48 |
| 注意力 | 2 个 Q 头、1 个 KV 头，head_dim=16（GQA/MQA） |
| 词表 | 378（WordLevel + GPT-2 ByteLevel 解码） |
| 位置编码 | RoPE（theta=10000） |
| 其它 | tied embedding、value_residual_mix=0.5、max_position=2048 |

仓库里的 `model.safetensors` 原本是 **Git LFS 指针（130 字节）**；已从 HuggingFace 拉取真实权重（`sha256=b1bd72cc…`，81364 字节）并在本地替换。

### 2.2 设备

| 项 | 值 |
|---|---|
| OS | CachyOS（Arch 系），kernel 7.3.1 |
| CPU | Intel Core Ultra 7 155H，22 逻辑线程 |
| CPU 指令集 | AVX2、FMA、F16C、AVX-VNNI；**无 AVX-512** |
| iGPU | Intel Arc Graphics (Meteor Lake，集成) |
| GPU 运行时 | Level Zero 1.34、Intel Compute Runtime、OpenCL、Vulkan 1.4.354（Mesa ANV 26.2.6） |
| 内存 | 30GB |

### 2.3 环境

- 默认 Python 3.14 无 torch/transformers；仅 Python 3.14 可用。
- 用 `uv` 建 `.venv`（Python 3.12.15）。
- 关键依赖：`torch==2.8.0+xpu`、`transformers==4.55.0`、`safetensors`、`tokenizers`、`huggingface_hub`、`onnx`、`onnxruntime`、`pnnx`、`ncnn`。

---

## 3. 运行方式

### 3.1 PyTorch（本地目录）

```bash
python run_story.py --device xpu   # Arc 核显（oneAPI Level Zero）
python run_story.py --device cpu   # 纯 CPU
```

### 3.2 导出 ONNX

```bash
python export_onnx.py              # -> converted/macro_stories.onnx
```

> 导出用独立复现的前向（绕开 transformers 的 DynamicCache 等内部结构），与 HF 原模型 logits 误差 **5.6e-6**。

### 3.3 ncnn

```bash
# 已生成：converted/ncnn/macro_stories.ncnn.{param,bin}
# 由 TorchScript 追踪后经 pnnx 转换（fp16=0 即 fp32 权重）
pnnx model.pt inputshape=[1,16]i64 fp16=0
```

### 3.4 手写 C++（推荐）

```bash
make -C cpp
./cpp/stories --root . --seed 20261007            # KV cache + AVX2
./cpp/stories --root . --threads 22 --quiet        # 多线程批量
./cpp/stories --root . --threads 22 --batch-gemm 4 --quiet
./cpp/stories --root . --selftest-kv               # KV vs 全量自检
./cpp/stories --root . --selftest-simd             # SIMD vs 标量自检
./cpp/stories --root . --selftest-batch            # 批量 vs 逐条自检
```

常用参数：`--device`（torch）、`--seed`、`--max`、`--temp`、`--top-p`、`--no-kv`、`--simd avx512|avx2|scalar|hybrid`、`--no-simd`、`--threads`、`--batch-gemm`、`--quiet`。

---

## 4. 各后端实测

### 4.1 生成吞吐（同 prompt / seed 20261007，温度 0.5，top-p 0.9）

| 后端 | 生成量 | 耗时 | 吞吐 |
|---|---:|---:|---:|
| torch XPU（Arc） | 151 tok | 6.76 s | ~22 tok/s |
| torch CPU | 141 tok | 1.25 s | ~113 tok/s |
| ONNX Runtime CPU（无 KV cache） | 131 tok | 0.38 s | ~342 tok/s |
| **C++ 单线程（KV+SIMD+top-p 选择）** | 131 tok | ~2 ms（纯解码） | **~60k tok/s** |
| **C++ 22 线程** | 3026 tok | ~10.6 ms | **~28.5 万 tok/s** |
| **C++ 22 线程 × batch 4** | 12078 tok | ~37 ms | **~32.5 万 tok/s** |

### 4.2 转换产物尺寸与精度

| 产物 | 大小 | 误差（vs torch 参考） |
|---|---:|---:|
| ONNX fp32 | 230,923 B | 1.2e-5 |
| ONNX int8（动态量化） | 224,122 B | 0.93（放弃） |
| ncnn fp32（param+bin） | 17,091 + 225,656 B | 1.2e-5 |
| ncnn fp16 bin | 115,832 B | 0.14（fp16 存储） |
| C++（AVX2/SIMD） | 二进制 ~81KB | 5.6e-6（对 Python 参考） |

### 4.3 Vulkan（ncnn，Intel Arc）

| 工作负载 | CPU fp32 | Vulkan fp32 | Vulkan fp16 |
|---|---:|---:|---:|
| S=16 单次前向 | **0.66 ms** | 11.92 ms | 10.91 ms |
| S=512 单次前向 | **10.56 ms** | 192.12 ms | — |
| 精度 max abs diff | 1.2e-5 | 1.5e-5 | 0.144 |

结论：Vulkan 在 S=16 与 S=512 都稳定慢约 **18×**，属于每层 kernel 启动/同步开销主导（图有 226 个极小算子），不是算力问题。

---

## 5. 内存与 CPU 开销

测量方法：`fork + wait4` 读取子进程 `ru_maxrss`（峰值 RSS）与 `ru_utime+ru_stime`，每项取多次中位数。

| 场景 | Wall | CPU 时间 | CPU 占用 | 峰值 RSS |
|---|---:|---:|---:|---:|
| C++ 进程基线 | 2.56 ms | 2.16 ms | 84% | **7.64 MB** |
| C++ 加载 + 1 次前向 | 2.84 ms | 2.57 ms | 90% | 7.64 MB |
| C++ 单条（KV+SIMD） | 8.29 ms | 8.04 ms | 97%（1 核） | **7.65 MB** |
| C++ `--threads 22` | 16.4 ms | 171.7 ms | 1049%（≈10.5 核） | 8.36 MB |
| C++ `th=22 batch=4`（88 条） | 46.9 ms | 588 ms | 1255% | 21.4 MB |
| C++ `th=22 batch=8`（176 条） | 84.1 ms | 1133 ms | 1347% | 39.9 MB |
| torch CPU 生成 | — | 23.7 s | 402% | **778.7 MB** |
| torch XPU 生成 | — | 11.4 s | 119% | **3221.6 MB** |
| ONNX Runtime 生成 | — | 6.0 s | 138% | **786.8 MB** |

> torch/ORT 的 wall 含 Python 解释器与库导入，不宜直接比延迟；RSS/CPU 为进程级，可横向比较。

### 5.1 内存拆解

- **权重**：81,364 B（≈80KB）；tokenizer 词表约 10KB。
- **C++ 进程底盘**：≈7.6MB（动态库 + 可执行文件）。
- **KV cache**（每条序列）：
  `4(recurrent) × (max_new+2) × 16 × 2(k/v) × 4B`
  - `--max 400` → ≈201 KB/条
  - `--max 512` → ≈257 KB/条
- 批量时内存随 `线程数 × 批大小 × cache/条` 线性增长；减小 `--max` 可直接降低占用。

### 5.2 CPU 结论

- 单条稳定占 **1 核**。
- 22 逻辑线程实际只吃到 **≈10.5 核**（P/E 混合核 + 超线程 + 短任务调度限制）。
- 批量可继续提升到 ~13.5 核的有效利用率，但收益递减。

---

## 6. 手写 C++ 的关键实现

### 6.1 组成

- 单文件 `cpp/stories.cpp`，零第三方依赖，直接解析 `model.safetensors` 与 `tokenizer.json`。
- 采样：temperature + top-p（`mt19937`）。
- 解码：GPT-2 ByteLevel 逆映射（`Ġ`→空格）+ 特殊 token（`[PERSON_NAME_*]`→Alex/Robin/Casey）。

### 6.2 正确性验证（可验证性）

| 自检 | 结果 |
|---|---|
| 与 Python 参考实现 logits 对比 | **5.6e-6** |
| `--selftest-kv`（增量缓存 vs 全量重算，257 token） | **0.0** |
| `--selftest-simd`（AVX2 vs 标量，L=1/7/33/128/257） | ≤ **1.7e-5**，生成序列一致 |
| `--selftest-batch`（批量 vs 逐条，5 条） | **逐 token 完全一致** |

### 6.3 KV cache

- 该模型是 1 层权重循环 4 次；参考实现为每次循环维护独立缓存槽（index = `current_ut * num_layers + layer`），故使用 **4 份 K/V 缓存**。
- 缓存内容：**RoPE 后的 K** + **value-residual 混合后的 V**。
- 由于每一步都是因果注意力，前缀 hidden state 不再变化，因此**逐 token 增量解码与整段重算数学等价**（自检误差 0.0）。
- 吞吐：全量重算 ~860 tok/s → KV cache ~13k tok/s（早期单线程），叠加 SIMD 后更高。

### 6.4 SIMD（AVX2 + FMA，已验证）

- 显式内核：`matvec`、`RMSNorm`、注意力 `q·k` 与 `p·V` 累加。
- 标量参考用 `#pragma GCC optimize("no-tree-vectorize")` 保证真实标量路径。
- 单线程吞吐：标量 ~13.3k → SIMD ~24.6k tok/s（约 **1.85×**）。
- 本机支持 AVX2/FMA/F16C/AVX-VNNI，无 AVX-512；AVX2（8×fp32）是本机合适宽度。远程 AVX-512 实测见 6.5。

### 6.5 AVX-512 实测（远程 Xeon）

远程环境：Intel Xeon Platinum，2 vCPU，支持 `avx512f/bw/cd/dq/vl`（无 VNNI/BF16），g++ 14.2。

实现：新增 `dot/axpy/matvec/rmsnorm` 的 AVX-512 内核，并提供 `--simd avx512`；`hybrid` 在 n≤16 用 AVX2、其余用 AVX-512。所有级别相对标量参考的 logits 误差 ≤1.7e-5，生成 token 序列完全一致（KV / 批量自检同样通过）。

性能（best of 15）：

| 级别 | 单线程 tok/s | `threads=2 batch=4` tok/s |
|---|---:|---:|
| scalar | 13,469 | 18,304 |
| **avx2** | **28,128** | **33,900** |
| avx512 | 23,252 | 31,805 |
| hybrid | 23,269 | 32,313 |

隔离点积微基准（ns/dot）：

| n | AVX2 | AVX-512 | 比值 |
|---:|---:|---:|---:|
| 16 | 3.89 | 5.10 | 1.31×（512 更慢） |
| 32 | 4.42 | 3.80 | 0.86× |
| 48 | 5.60 | 4.31 | 0.77× |
| 128 | 13.82 | 8.14 | 0.59× |
| 512 | 104.65 | 34.12 | 0.33× |

结论：AVX-512 在较长点积上更快，但本模型注意力为 head_dim=16（512 的约简开销更大），且持续 512-bit FMA 会触发该类 Xeon 的降频，整体反而慢约 17%；hybrid 因降频是整核范围的，同样无改善。故默认 AVX2，AVX-512 仅作为可选。

### 6.6 多线程与批量化 GEMM

- `--threads N`：N 条独立序列并行，结果与逐条运行逐字一致。
- `--batch-gemm B`：一次前向锁步处理 B 条序列，权重行跨 batch 复用，遇 EOS 动态收缩批次。
- 单一后端对比：
  - 单线程批量 GEMM：B≥4 后饱和在 ~32k tok/s（相对单条 ~1.3×）。
  - 多线程扩展性远强：22 线程 ~28.5 万 tok/s。
  - 组合 `threads=22, batch=4`：~32.5 万 tok/s（相对纯多线程约 +14%）。
- 原因：权重仅 80KB，本就在 L1/L2；批量 GEMM 的“权重摊薄”优势不存在，注意力仍是逐序列成本。

### 6.7 Top-p 采样选择优化（已合入）

- 现象：默认 `top_p=0.9` 时 top-p 前缀极小（中位数 1 个 token，p90=2，最大 ~4）；对 378 个 token 全排序是浪费，实际解码循环里 `std::sort` 成本约 **25–30µs/token**。
- 做法（`sample_select`）：
  - 用未归一化 exp 权重，省掉 378 次除法；
  - 反复取最大值，直到累计质量达到 `top_p * total`（概率互异时与排序前缀等价）；
  - 设 64 次扫描上限，超过则回退到 `sample_sort`；`top_p >= 0.999` 直接走排序，避免 `top_p=1.0` 退化为 O(V²)。
- 正确性：5 个 seed（top_p=0.9）与 2 个 seed（top_p=1.0）生成文本与旧实现**逐字一致**；全套自检通过。
- 性能（CPU 时间）：top_p=0.9 从 **~36–39µs** 降到 **~13–17µs/token（约 2.3×）**；top_p=1.0 与旧版持平（±噪声）。
- keep 分布：top_p=0.9 中位数 1；0.99 中位数 2、p90=5；1.0 均值 ~162（因此必须回退）。

---

## 7. 结论与建议

### 7.1 选型

| 需求 | 建议 |
|---|---|
| 本地快速跑通 / 演示 | PyTorch（`run_story.py`） |
| 跨平台部署、保留动态长度 | **ONNX Runtime**（精确、通用） |
| 极轻运行时（几 MB） | **ncnn**（CPU 精确；Vulkan 精度/性能不佳） |
| 最高性能 / 最低内存 | **手写 C++**（推荐） |

### 7.2 明确的“不做”

- **Vulkan 加速**：本机可用但慢约 18×，对 20k 模型无意义；等模型规模上去再考虑。
- **int8 / AVX-VNNI 量化**：体积几乎不降、误差显著（0.93），投入产出比为负。
- **AVX-512 作为默认**：已实现并验证正确，但在远程 Xeon 上比 AVX2 慢约 17%（降频），故默认仍用 AVX2，仅保留 `--simd avx512` / `--simd hybrid` 供其他硬件按需启用。
- **F16C 权重压缩**：模型仅 80KB，收益可忽略。

### 7.3 资源结论

- C++ 版峰值内存 **~7.6MB**，单条用 1 核；相比 torch/ORT 省 **100–400 倍**内存。
- 多线程/批量时内存按 KV cache 线性增长，可通过 `--max` 控制。

---

## 8. 复现命令汇总

```bash
# 环境
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.8.0 \
    --index-url https://download.pytorch.org/whl/xpu
uv pip install --python .venv/bin/python \
    transformers==4.55.0 safetensors tokenizers huggingface_hub onnx onnxruntime pnnx ncnn

# PyTorch
.venv/bin/python run_story.py --device cpu|xpu

# ONNX
.venv/bin/python export_onnx.py

# C++
make -C cpp
./cpp/stories --root . --seed 20261007
./cpp/stories --root . --selftest-kv
./cpp/stories --root . --selftest-simd            # 遍历 scalar/avx2/avx512/hybrid
./cpp/stories --root . --selftest-batch
./cpp/stories --root . --threads 22 --batch-gemm 4 --quiet
./cpp/stories --root . --simd avx512 --quiet      # 仅在支持 AVX-512 的机器上生效
```

---

## 9. 文件清单

| 路径 | 说明 |
|---|---|
| `run_story.py` | PyTorch 运行脚本（CPU/XPU） |
| `export_onnx.py` | ONNX 导出（独立复现前向） |
| `converted/macro_stories.onnx` | ONNX fp32 产物 |
| `converted/ncnn/macro_stories.ncnn.{param,bin}` | ncnn fp32 产物 |
| `cpp/stories.cpp` | 手写 C++ 推理（KV cache / AVX2+AVX-512 / 多线程 / batch-gemm / 低分配） |
| `cpp/Makefile` | 构建（`-O3 -march=native -pthread`） |
| `cpp/.gitignore` | 忽略编译产物 `stories` |
| `zig/stories.zig` | Zig 0.17 单文件移植（显式分配实验） |
| `zig/README.md` | Zig 0.17 API 变动与移植说明 |
| `REPORT.md` | 本报告 |

### Git 记录

| Commit | 内容 |
|---|---|
| `221e44c` | 手写 C++ 推理（KV cache + SIMD + 多线程） |
| `20597aa` | 锁步批量化解码（batch-gemm） |
| `78e7968` | AVX-512/hybrid SIMD、热路径分配消除、本报告 |

> 说明：`model.safetensors`（真实权重）与 `converted/` 产物目前未纳入 git；如需版本化，建议用 Git LFS 管理权重、产物按需生成。
> 输入/权重校验、死代码清理与 RoPE 频率预计算随最新一次提交入库。
> Zig 0.17 单文件移植与 API 变动说明见 `zig/README.md`，随本次提交入库。
> top-p 选择优化（`sample_select` + 回退）随最新一次提交入库。

## 10. 外部代码评审的处理

对一次外部（LLM）代码评审逐条对照当前实现后的处理：

| 评审项 | 结论 | 处理 |
|---|---|---|
| 每 token 动态分配（scores / sample） | 评审基于旧版，已过时 | 已用 thread_local / 栈数组消除；单条 malloc 728→344 |
| 预计算 RoPE | 部分成立 | 已预计算 inv_freq，去掉每 token 的 8 次 `pow`；全量 cos/sin 表收益 <1%，不做 |
| RMSNorm + Residual 融合 | 成立但低收益（~1–2%） | 暂不做，需按 ISA 各写融合变体 |
| 多行并行 MatVec | 成立，需实测 | 未做，列为后续实验项 |
| 真正的 Batch GEMM | 成立，但实测收益有限 | B≥4 饱和，叠加多线程仅 +14% |
| 模型 dtype / 维度校验 | 成立 | 已加 F32 + numel 校验 |
| temp / top_p / max / token 越界校验 | 成立 | 已加 |
| `--batch` 与 `--threads` 混淆 | 成立 | 已删除 `--batch` 别名 |
| 未使用的 `h` 数组 | 成立 | 已删除 |
| “JSON 解析不完整” | 对本 tokenizer 不成立 | 仅有 `\"` / `\\`，已覆盖 |
| 运行时 CPU 指令集检测 | 对通用二进制成立 | `-march=native` 场景不适用，暂不做 |
| 多文件 + CMake 重构 | 长期合理 | 当前 ROI 低，保持单文件 |

验证：`--selftest-kv` = 0.0、`--selftest-simd` ≤ 1.7e-5、`--selftest-batch` 全一致；吞吐无回退（单线程 ~25k、22 线程 ~256k、22×batch4 ~323k tok/s）。

## 11. Zig 0.17 单文件移植（显式分配实验）

- `zig/stories.zig`：零依赖单文件，读取同一份 safetensors/tokenizer，实现 KV 解码与 `--logits`。
- Zig 0.17 关键 API 变化：
  - `pub fn main(init: std.process.Init) !void`，从 `init.io/.gpa/.arena/.minimal.args` 取资源；
  - `std.Io.Dir` / `std.Io.File` 的 I/O 需要显式传 `io`；stdout 用 `File.stdout().writer(io, &buf).interface`；
  - `std.heap.DebugAllocator` 取代 `GeneralPurposeAllocator`；
  - `std.ArrayList` 为 unmanaged（方法显式传 allocator）；
  - `std.sort` 有 `heap` / `insertion` / `pdq`（推荐 `pdq`；`heap` 在本模型采样上慢约 4×）；
  - SIMD 用 `@Vector(8, f32)` + `@mulAdd` + `@reduce(.Add, ...)`。
- 显式分配：`ArenaAllocator` 一次性承载权重、tokenizer、KV cache 与输出缓冲，末尾统一 `deinit`；每 token 临时量走栈，热路径无堆分配。
- 验证：`--logits` 与 Python 参考 max abs diff **6.5e-6**；单线程解码（CPU 时间）约 **67µs/token ≈ 15k tok/s**，C++ AVX2 约 **36µs/token ≈ 28k tok/s**（`-fstrip` 后约 358KB）。
- 注：上述 C++ 对比在 top-p 选择优化合入前测得；合入后 C++ 单线程完整生成约 **13–17µs/token（~60k tok/s）**，Zig 仍用 `std.sort.pdq` 全排序。
- 差距根因（实测拆分）：**前向** Zig 22.4µs vs C++ 11.5µs（C++ 手写 AVX2 覆盖 matvec/rmsnorm/axpy，Zig 仅 `dot` 用 `@Vector`）；**采样** Zig `pdq` 20.8µs vs C++ `std::sort` 7.4µs（误用 `std.sort.heap` 时 31.7µs，换 `pdq` 后完整生成约快 15%）；**加载不是瓶颈**（Zig 1.5–1.7ms，C++ 2.2–2.4ms）。
- 编译器归因（GCC 16 vs Zig 0.17 的 LLVM 22）：**等价内核结构下 GCC/Clang 基本打平**（dot-only 15.51 vs 15.57µs），编译器不是主因；GCC 仅在手写 AVX2 全内核上有内联/展开优势（11.24 vs 14.89µs）。Zig 与等价 C++ 仍有 ~18–30% 语言/运行时差距，strict FP 再损失 ~10%（`@setFloatMode(.optimized)` 20.39→18.25µs，精度 6.95e-6）。
- 详见 `zig/README.md`。
