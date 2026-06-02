# Triton Kernel 实现与优化指南

本文件只用于通用 Triton kernel 实现和优化决策。算子接口、功能测试和性能测试契约由 `SKILL.md` 主流程整理；具体实现仓库规则和项目约束只能写入 [custom_op_contract.md](custom_op_contract.md)。

## 目录

- [设计检查清单](#设计检查清单)
- [实现模式](#实现模式)
- [常见算子形态](#常见算子形态)
- [性能启发式](#性能启发式)
- [Triton 调试环境变量](#triton-调试环境变量)
- [PTX / SASS / NCU 深度调优](#ptx--sass--ncu-深度调优)
- [正确性陷阱](#正确性陷阱)
- [优化循环指南](#优化循环指南)
- [交付检查清单](#交付检查清单)

## 设计检查清单

写代码前先明确：

1. 每个程序实例负责计算哪些输出元素。
2. `tl.program_id(...)` 的各个维度如何映射到输出 tile。
3. 指针偏移从张量 shape 和 stride 推导；除非算子契约明确要求 contiguous，否则不要假设输入连续。
4. 哪些维度是 block 维度、归约维度和 batch 维度。
5. 根据数据搬运和归约模式选择初始 `BLOCK_SIZE`、`num_warps` 和 `num_stages`。
6. 在写 `tl.load` 或 `tl.store` 前，先规划每个边界条件的 mask。

## 实现模式

围绕 Triton JIT kernel 实现一个小的 Python 封装函数。封装函数应该：

- 校验或规范化算子契约要求的 shape 和 dtype。
- 分配输出，除非算子明确是 in-place 或写入外部传入 buffer。
- 将 shape、stride 和标量元数据作为 kernel 参数传入。
- 根据输出 tile 拆分计算启动网格。
- 对控制编译期结构的元参数使用 `tl.constexpr`。

Triton kernel 应该：

- 使用 `tl.arange` 和 `tl.program_id` 构造向量化 offset。
- 对所有可能越界的 load 和 store 使用 mask。
- 让最快变化维度尽量形成 coalesced 连续访存。
- 尽可能将中间值保存在寄存器中。
- 只写入当前程序实例负责的输出 tile。
- 不隐含跨程序实例同步假设。

## 常见算子形态

逐元素算子：

- 当参与运算的张量都是 contiguous，或广播规则很简单时，可使用一维 flatten。
- 对带 stride 或广播后的输入，显式计算每个张量的 offset。
- 使用足够大的 2 的幂 block size 来摊薄启动开销，同时避免过高寄存器压力。

归约算子：

- 当归约维度能放入合理 block size 时，优先在 block 内完成归约。
- 只有 shape 很大或数值稳定性需要时，才使用多阶段归约。
- 根据语义选择安全的累加 dtype；例如低精度输入通常需要更高精度累加。

归一化算子：

- 尽可能一次加载被归一化的 row 或 tile。
- 对中小 row，在寄存器中计算均值、方差或 RMS。
- 当寄存器压力允许时，复用已加载值完成最终变换，避免重复从全局内存读取。

矩阵或 tile 类算子：

- 使用二维 program id 映射输出 tile。
- 选择能提升数据复用、但不会造成过高寄存器压力的 tile size。
- 可根据算子契约选择分组遍历、分块重排或其他 cache locality 优化。

## 性能启发式

优化实际瓶颈，而不是抽象地按算子类别套模板。

- 受内存带宽限制：改善 coalescing，减少 global load/store，融合简单操作，避免物化中间结果，使用向量化 tile size。
- 受启动开销限制：增加每个程序实例的工作量；如果接口允许，可融合相邻简单操作。
- 受归约开销限制：调节 block size、累加 dtype 和 `num_warps`，避免不必要的多阶段归约。
- 受寄存器压力限制：减少同时存活的中间值，拆分过大的 tile，简化分支，避免在寄存器中保留重复张量。
- 受 occupancy 限制：调节 `num_warps`、block size 和 stages，并在少量有针对性的配置之间比较性能测试结果。
- 受不规则访存限制：先保证正确性；如果算子契约允许 layout 约束或 dispatch，再考虑 layout-specific fast path。

只有当算子契约和项目约束允许时，才使用 autotune。候选配置要聚焦；过宽的 autotune 配置网格会拖慢测试，也会模糊主要瓶颈。

## Triton 调试环境变量

迭代调试和性能优化时，可以按需给功能测试或性能测试命令添加以下环境变量：

| 环境变量 | 作用 | 使用建议 |
|----------|------|----------|
| `TRITON_ALWAYS_COMPILE=1` | 忽略 Triton 编译缓存命中，强制重新编译 kernel。 | 适合验证 kernel 代码、meta 参数或 autotune 配置变更确实生效。不要默认用于最终性能结论；如果 benchmark 计入编译时间，它会污染耗时。 |
| `TRITON_PRINT_AUTOTUNING=1` | autotune 完成后打印最佳配置和调优耗时。 | 适合记录每轮优化选中的配置、判断 autotune 是否稳定、决定是否固化配置。性能不合格时，将输出摘要写入临时优化日志。 |
| `TRITON_CACHE_DIR=/tmp/kernel-generate/triton-cache` | 指定 Triton 编译缓存目录。 | 适合隔离本任务缓存，便于清理、复现和查看编译产物，避免不同任务共享默认缓存造成干扰。 |
| `TRITON_KERNEL_DUMP=1` | 导出 Triton 编译中间产物。 | 适合深度调优时查看 TTIR、TTGIR、LLVM IR、PTX 和 cubin。仅在需要分析编译产物时启用。 |
| `TRITON_DUMP_DIR=/tmp/kernel-generate/triton-dump` | 指定 dump 文件输出目录。 | 与 `TRITON_KERNEL_DUMP=1` 配合使用，便于按任务隔离 dump 产物。 |

调试或确认变更生效时可使用：

```bash
rm -rf /tmp/kernel-generate/triton-dump /tmp/kernel-generate/triton-cache
mkdir -p /tmp/kernel-generate/triton-dump /tmp/kernel-generate/triton-cache

TRITON_CACHE_DIR=/tmp/kernel-generate/triton-cache \
TRITON_DUMP_DIR=/tmp/kernel-generate/triton-dump \
TRITON_KERNEL_DUMP=1 \
TRITON_PRINT_AUTOTUNING=1 \
TRITON_ALWAYS_COMPILE=1 \
<功能测试或性能测试命令>
```

形成最终性能结论时，优先不要设置 `TRITON_ALWAYS_COMPILE=1` 和 `TRITON_KERNEL_DUMP=1`；如果必须设置，确认 benchmark 已经通过 warmup 排除了编译和 dump 开销。常用最终性能环境为：

```bash
TRITON_CACHE_DIR=/tmp/kernel-generate/triton-cache \
TRITON_PRINT_AUTOTUNING=1 \
<性能测试命令>
```

## PTX / SASS / NCU 深度调优

当普通参数调节、少量 autotune 配置和代码级检查无法让 `speedup >= 0.9`，或者瓶颈假设不清晰时，升级到证据驱动的深度调优。核心原则：PTX 能提示问题，但不能单独证明瓶颈；SASS 更接近 GPU 实际执行；Nsight Compute 指标用于判断真实硬件瓶颈。

仅在工具可用时执行本节：PTX/cubin dump 依赖 Triton dump；SASS 分析通常需要 NVIDIA CUDA 工具 `cuobjdump` 或 `nvdisasm`；NCU 分析需要 `ncu`。如果工具不可用，在优化日志中记录“未执行”和原因。

### Dump 编译产物

运行带 dump 的功能测试或性能测试后，查看产物：

```bash
find /tmp/kernel-generate/triton-dump -type f
find /tmp/kernel-generate/triton-dump -type f -name "*.ptx"
find /tmp/kernel-generate/triton-dump -type f -name "*.cubin"
```

常见产物含义：

| 文件 | 用途 |
|------|------|
| `*.ttir` | Triton IR，适合看高层变换后的 IR。 |
| `*.ttgir` | Triton GPU IR，适合看 GPU lowering 后结构。 |
| `*.llir` | LLVM IR，适合看进一步 lowering 后的控制流和指令形态。 |
| `*.ptx` | PTX 汇编，适合快速发现 Tensor Core、spill、访存、除法和分支迹象。 |
| `*.cubin` | CUDA binary，可用于反汇编 SASS。 |

### PTX 观察点

| 检查项 | 命令 | 如何解读 |
|--------|------|----------|
| Tensor Core | `grep -R "mma.sync" /tmp/kernel-generate/triton-dump` | 对 matmul、conv、attention 类算子，没有 `mma.sync` 往往说明没有走 Tensor Core。继续检查 dtype、`tl.dot` 用法、tile shape、layout 和 K 维对齐。 |
| 寄存器 spill | `grep -R "ld.local\|st.local" /tmp/kernel-generate/triton-dump` | 大量 local memory 访问通常表示寄存器压力过高。优先减小 tile、减少中间张量、降低 live 变量数量、降低 `num_warps` 或拆分 kernel。 |
| 全局访存 | `grep -R "ld.global\|st.global" /tmp/kernel-generate/triton-dump` | 关注是否大量标量访存、非连续访存、复杂 stride 或过多 mask load。优先改善 program id 映射、拆 fast path、减少边界 mask 和动态索引。 |
| 整数除法/取模 | `grep -R "div\|rem" /tmp/kernel-generate/triton-dump` | 热路径中大量 div/rem 通常很慢。优先用 `tl.constexpr` 固化尺寸、用 shift/mask 替代、host 侧预计算或拆特殊路径。 |
| mask/分支 | `grep -R "setp\|selp\|bra" /tmp/kernel-generate/triton-dump` | 大量 predicate、select、branch 可能表示边界处理或 general path 过重。考虑拆 contiguous、整除 shape、边界 block、tiny/wide 等路径。 |

### SASS 观察点

PTX 不是最终执行指令。若 dump 中有 cubin，可反汇编：

```bash
CUBIN=$(find /tmp/kernel-generate/triton-dump -type f -name "*.cubin" | head -n 1)
cuobjdump --dump-sass "${CUBIN}" > /tmp/kernel-generate/kernel.sass
# 或：nvdisasm "${CUBIN}" > /tmp/kernel-generate/kernel.sass

grep -n "HMMA\|MMA\|WGMMA\|LDG\|STG\|LDS\|STS\|ATOM\|BRA\|BAR" /tmp/kernel-generate/kernel.sass | head -100
```

重点确认：Tensor Core 指令是否真的出现，global/local load/store 是否异常多，是否有大量 branch、atomic、barrier，指令数量是否异常膨胀。PTX 中看到 `mma.sync` 不代表最终执行一定最优；SASS 中没有对应 HMMA/MMA/WGMMA，才说明 Tensor Core 没真正吃上。

### Nsight Compute 定位真实瓶颈

对固定性能测试用例运行 NCU：

```bash
ncu --target-processes all \
    --set full \
    -o /tmp/kernel-generate/ncu_report \
    <性能测试命令>
```

如果 kernel 数量很多，用 kernel 名过滤：

```bash
ncu --target-processes all \
    --kernel-name regex:.*<kernel_name>.* \
    --set full \
    -o /tmp/kernel-generate/ncu_report \
    <性能测试命令>
```

优先查看 `Speed Of Light`、`Memory Workload Analysis`、`Scheduler Statistics`、`Warp State Statistics`、`Launch Statistics`、`Occupancy` 和 `Source Counters`。

| 现象 | 可能瓶颈 | 优先动作 |
|------|----------|----------|
| DRAM throughput 高 | memory bandwidth bound | 改善 coalescing、减少 HBM 往返、融合操作、减少中间结果。 |
| L2 hit rate 低 | locality 差 | 调整 tile/遍历顺序、重用数据、减少不规则访问。 |
| achieved occupancy 低 | block/warps/register/shared memory 配置不佳 | 降低寄存器或 shared memory 占用，调整 block size 和 `num_warps`。 |
| long scoreboard stall 高 | global memory latency | 增强数据复用、改善访存连续性、增加并行度或隐藏延迟。 |
| barrier stall 高 | 同步或 barrier 过重 | 减少同步点，检查 shared memory 使用和分阶段计算。 |
| not selected stall 高 | warp 过多或调度竞争 | 调整 `num_warps`、tile size 或并发工作量。 |
| issue slot utilization 低 | 指令并行度不足 | 减少依赖链，增加独立计算，检查过多分支。 |
| local memory load/store 多 | register spill | 减少 live 变量、减小 tile、降低 `num_warps`、拆 kernel。 |
| Tensor Core utilization 低 | 未吃到 Tensor Core 或 tile 不合适 | 检查 dtype、`tl.dot`、tile shape、K 对齐和 SASS Tensor Core 指令。 |
| launch 数过多 | host dispatch 或 fusion 问题 | 评估 kernel fusion 或减少小 kernel 调度。 |
| atomic stall 高 | atomic 冲突严重 | 改写归约/聚合策略，减少竞争。 |

### 深度调优闭环

每轮深度优化只改一个主要点：

1. 固定一个性能测试用例和一个最小功能测试子集。
2. dump PTX/cubin，并运行性能测试。
3. 用 PTX 快速找 Tensor Core、spill、div/rem、mask、访存疑点。
4. 用 SASS 确认真实指令形态。
5. 用 NCU 判断 memory bound、compute bound、occupancy 或 stall 类型。
6. 提出一个瓶颈假设，并应用一个有针对性的修改。
7. 重新运行功能测试和性能测试。
8. 变快则保留，变慢则回滚或记录为失败假设。
9. 将 PTX/SASS/NCU 证据、修改和结果写入临时优化日志。

日志中建议追加：

```markdown
## 第 <n> 轮深度调优
- 固定用例：
- PTX 观察：
- SASS 观察：
- NCU 关键指标：
- 瓶颈假设：
- 修改：
- 功能测试结果：
- 性能测试结果：
- 决策：keep/revert
```

## 正确性陷阱

归因性能前，先检查：

- 非 2 的幂维度的边界 mask。
- size-one 维度上的广播 offset。
- 算子契约允许时的 non-contiguous stride，以及特殊 stride 情况。
- 累加 dtype 和最终 cast 行为。
- in-place 操作和输出 buffer 的别名关系规则。
- 低精度输入的数值误差容忍度。
- device 放置和 stream 行为。

## 优化循环指南

每轮性能优化：

1. 从测得的慢用例出发。
2. 明确一个瓶颈假设。
3. 只修改 kernel 或启动配置中的一个连贯部分。
4. 重新运行能捕获正确性回归的最小功能测试子集。
5. 重新运行相关性能测试用例。
6. 将本轮记录到临时优化日志。

避免在一轮中堆叠互不相关的修改。如果某个修改提升一种 shape 但伤害另一种 shape，记录该取舍，并根据算子契约判断是否值得做 shape-specific dispatch 或 autotune 配置。

## 交付检查清单

kernel 实现只有在以下条件都满足时才算完成：

- 封装函数符合算子契约。
- 功能测试覆盖必需语义并通过。
- 性能测试能够针对约定基准计算 speedup。
- 所有必测性能测试用例的 `speedup >= 0.9`。
- 如果创建过临时优化日志，日志记录了通向最终实现的瓶颈和实测结果。
