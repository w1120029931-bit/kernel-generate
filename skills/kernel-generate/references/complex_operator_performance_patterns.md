# Complex Operator Performance Patterns

本文件总结自当前目标仓库中 matmul、conv、SDPA/SDPA backward 等复杂 Triton 算子的多轮优化经验。它只记录可迁移的性能方法、判断顺序和常见反例；仓库路径、测试命令、注册方式等项目规则仍应放在 `custom_op_contract.md`。

## 目录

- [何时使用](#何时使用)
- [先归因再动算法](#先归因再动算法)
- [矩阵乘类方法](#矩阵乘类方法)
- [卷积类方法](#卷积类方法)
- [注意力类方法](#注意力类方法)
- [跨算子通用策略](#跨算子通用策略)
- [常见反例](#常见反例)
- [迭代记录要点](#迭代记录要点)

## 何时使用

在优化以下复杂算子时优先读取本文件：

- matmul、batched matmul、projection GEMM、implicit GEMM。
- conv fprop/dgrad/wgrad，尤其是 stride/padding/filter 规则固定的模型热点 shape。
- SDPA forward/backward、GQA/MQA、causal attention、online softmax、split/reduce attention。
- 已完成基础 autotune，但 speedup 仍低于目标，需要判断是否更换算法映射。
- PTX/SASS 显示 Tensor Core 已用上，但仍有明显 gap，需要从布局、分流、归约结构或 launch 结构继续推进。

## 先归因再动算法

1. 对比 graph/prepared/direct/kernel 路径。
   如果 graph 和 prepared/direct latency 接近，瓶颈基本在 kernel 或算法；如果 graph 明显慢，先优化输出分配、prepared launcher、静态参数绑定和 cached call。
2. 区分 benchmark 抖动和真实 gap。
   临界 shape 至少做 focused rerun，避免把 1%-2% 噪声当作保留依据。
3. 确认 Tensor Core 指令。
   matmul/conv/attention 类算子若没有 `mma.sync`、`wgmma`、`HGMMA`，先检查 dtype、`tl.dot` 形状、K 对齐、layout 和 cast。
4. 检查整数索引和 local memory。
   PTX/SASS 中的 hot-loop `div/rem`、大量 predicate/select、`LDL/STL`、高 register、barrier/membar 都可能比访存更关键。
5. 将失败路径快速回退并记录。
   正确但慢、快但错、编译失败、只改善一个 dtype 但伤害另一个 dtype，都应明确拒绝，不留在最终代码里。

## 矩阵乘类方法

### Shape-specific dispatch

- 对固定大 shape 建专用路径比扩大通用 autotune 更有效。
- fp32、fp16、bf16 不一定共享最佳 lowering。fp32 可以尝试 direct pointer、TF32、fp32-to-fp16 快路径；低精度可以尝试 persistent 或更大的 WGMMA tile。
- dispatch 条件要窄：只覆盖实测目标 shape 或明确同类 regime，避免污染其他 shape。

### Direct pointer 与 block pointer

- block pointer 边界处理简洁，但在某些 fp32 或投影 GEMM 上可能带来额外 lowering 成本。
- direct pointer 可消除部分 block pointer/boundary 路径开销，适合固定 contiguous、整除或可用简单 mask 的大 GEMM。
- 如果 shape 整除，显式拆无 boundary path，避免每个 tile 带 mask/boundary_check。

### Persistent kernel

- persistent 更适合 launch/CTA scheduling 占比较高、tile 数和 batch 数稳定的中等 square GEMM。
- 不要假设 persistent 能改善所有 projection GEMM；投影类 M/N/K 不均衡时，persistent 可能 underfill 或牺牲并行度。
- 对 persistent 候选要比较不同 CTA multiplier，但若 baseline 已有良好 WGMMA pipeline，收益常很窄。

### Tile/GROUP_M 调优

- 先围绕瓶颈假设扫少量配置：`BLOCK_M/N/K`、`GROUP_M`、warps、stages。
- 大 N tile 或双输出 tile 可能减少重读，但会增加 accumulator/register pressure；只有 SASS/计时同时支持时才保留。
- split-K + atomic 在 GEMM 上常被额外 zero/atomic/launch 吞掉收益；除非 K 维明显 under-parallelized 且 atomic 量可控。

## 卷积类方法

### 将卷积重写为更规则的 GEMM

- 对固定 stride/padding/filter 的热点 shape，先判断通用 implicit GEMM 是否被索引和 mask 拖慢。
- 如果输入/权重能被一次 pack 成连续 K 维，显式 pack + tuned GEMM 可能远快于复杂 direct kernel。
- pack 成本必须单独计时。只有 pack+GEMM 总时间优于 direct 和 baseline 才保留。
- fp32 GEMM meta 可能与低精度冲突；必要时 dtype-specific 固定 meta 或单独 tune key，避免全局 config 污染低精度。

### Layout pack

- 优先把热路径 K 维变连续，例如把 filter 或 activation pack 成 `[K, N]` / `[KHW, C]` / `[Cout, HW]` 等利于 `tl.dot` 的形式。
- pack 后的 GEMM tile 可以独立调优；direct path 保留为非 contiguous、非目标 shape 或 fallback-style 的 Triton 路线。
- 对 wgrad/dgrad，pack 的方向要服务于真正热的 dot：有时输出连续 store 不如 B operand 连续 load 重要。

### Parity、row grouping 与 fixed tap

- stride2/pad1/3x3 常有固定 parity 和边界结构。把 output rows/cols 分组，可以去掉 hot-loop `%`、`//` 和大量 invalid tap。
- row4、tile2w、tile4 这类映射本质是在 launch 数、accumulator 数、K padding、边界 mask 之间取舍。
- 四输出 tile 能减少 launch/重复读，但可能寄存器压力过高；两输出水平 tile 往往是折中点。

### Split/reduce 与 zero+atomic

- wgrad 常需要跨 N/HW/K 归约。split + reduce 适合正确性要求高且 atomic 误差不可接受的低精度路径。
- bf16/fp16 partial 是否可用必须用正确性验证；低精度 partial 快但容易超容差。
- fp32 或容差允许时，zero + atomic-all 有时比 store-first/atomic-rest 或 split+reduce 更快，因为 launch 更少、结构更简单。
- 对 exact shape，可用第一 split store、后续 split atomic，减少 zero launch；但如果同步/多 launch 开销超过 atomic 节省，应改回 zero+atomic-all。

### Precision-specific route

- `input_precision=tf32`、`tf32x3`、`ieee` 需要按正确性和性能一起判断。
- fp32 direct/atomic 快路径可能因为 TF32 误差不达标；`tf32x3` 可能是性能和精度折中，`ieee` 常过慢。
- 不要把低精度和 fp32 绑定到同一 conv route；它们通常偏好不同 tile、split 数、partial dtype 和 precision。

## 注意力类方法

### Descriptor/TMA 与 pointer load

- 长序列 attention forward 可尝试 tensor descriptor/TMA 路线，尤其是固定 BHSD、D=64/128、causal/GQA 形态。
- descriptor 不一定适合每个阶段。masked diagonal tail 改成 pointer load 看似减少无效列，但可能破坏统一调度并回退。
- TMA/descriptor 候选必须检查 SASS 中是否出现预期的 async/global-to-shared 指令，并用 focused timing 判断。

### Online softmax 的重排

- forward attention 的 `m/l/acc` 状态不要轻易拆到 workspace。split-KV 能增加并行度，但 partial `acc/m/l` materialization 和 combine launch 通常很贵。
- epilogue 中的除法、reciprocal、stats 计算顺序可能被 ptxas 自动优化；源码层 reciprocal 或 stats schedule 变换需要实测，不应凭 PTX 单点判断。
- 对 causal full-prefix 与 masked phase，先确认 masked work 是否真是瓶颈；只优化 diagonal tail 常常收益不足。

### GQA/MQA 分流

- GQA 的核心机会是 K/V 复用与 head grouping，但 grouped head tile 会增加 accumulator 和调度压力。
- 形如 `(BLOCK_M, BLOCK_H, BLOCK_N)` 的配置要让 group names 真正绑定到 data reuse，不要只扩大 row/head tile。
- 对 GQA backward，dQ 与 delta 可融合：同一 pass 计算 `delta=sum(o*dO)` 并立即用于 dQ，再把 delta 存给 dK/dV，可去掉单独 delta pass。

### Backward 的 dQ/dK/dV 分解

- 先判断 dQ、dK、dV 哪个输出需要跨 program 累加。直接写 dQ、atomic dK/dV 的 m-loop 路线，常比所有输出 atomic 更稳。
- 对 exact causal D128，可专门去掉边界 mask、预计算或融合 delta、缩小 zero 范围。
- diagonal-init/offdiag-tail、q-head direct-store、per-head scratch reduce 等更深路线必须用正确性和性能一起拒绝或保留；它们常因额外 launch、workspace traffic 或 rounding 失败。
- dK/dV atomic kernel 的瓶颈通常是 global reduction、barrier 和 register pressure；如果 tile/warps/stages 多次回退，应转向算法结构，而不是继续扩大 config 网格。

## 跨算子通用策略

### Compile-time specialization

- 对 graph/prepared 已知的 shape、stride、head/group 数、padding、dilation、alignment，用 `tl.constexpr` 固化，减少 hot-loop address arithmetic。
- 将 `SQ/SKV/HQ/Q_PER/stride` 等变为 constexpr 后，再复查 PTX 是否减少 div/rem、IMAD、predicate。
- 注意 constexpr 也可能增加代码体积或改变 ptxas 调度；保留必须以 focused benchmark 为准。

### Static fast path 与 generic path 并存

- exact shape fast path 用来去掉动态 mask、边界和索引；generic path 保留覆盖面。
- fast path dispatch 条件必须包含所有假设：contiguous/stride、整除、dtype、D、filter size、padding、groups、GQA ratio 等。
- 如果 fast path 只对一个 dtype 有收益，做 dtype-specific dispatch，不要强行统一。

### 减少 launch 与减少 work 的权衡

- 融合 zero、delta、pack、cast 或 reduce 可以减少 launch，但可能增加寄存器压力或重复访存。
- 多 launch 算法只有在明显降低 hot kernel work 或改善 Tensor Core 利用时才值得保留。
- small shape 对 launch 开销敏感；large shape 更看重 memory layout、Tensor Core pipeline 和 reduction structure。

### Autotune 使用方式

- 第一次正式性能结论前接入仓库级 autotune；但参数网格要围绕假设收窄。
- 当一个配置稳定胜出，把候选集收窄，避免后续完整 benchmark 被编译成本和噪声干扰。
- 如果 autotune key 缺少 dtype、layout 或 exact shape 信息，手工固定某个分支的 meta 可能比扩大全局配置更安全。

### PTX/SASS/NCU 证据

- PTX 用于快速看 Tensor Core、div/rem、local memory、atomics、mask/branch。
- SASS 更适合看真实 HGMMA/WGMMA、LDGSTS/cp.async、REDG/ATOM、LDL/STL、BAR/MEMBAR。
- NCU 不可用时，记录权限或工具错误，并用 PTX/SASS、受控 timing、baseline 算法对照替代。
- 不要仅凭 PTX 判断理论不可达；只有结合硬件指标、SASS、baseline 算法或项目约束，才能做强结论。

## 常见反例

- 盲目增大 tile：常因 accumulator/register pressure 或 barrier 增多而回退。
- 盲目减少 stages/warps：可能严重破坏 WGMMA pipeline 或 occupancy。
- source-level reciprocal 替代除法：ptxas 可能已经做了更合适 lowering，实测常无收益。
- pointer load 替代 descriptor tail：可能减少无效 compute，但破坏调度或增加地址指令。
- split-KV / split-K workspace：增加并行度的同时引入 workspace traffic 和 combine launch，常不划算。
- 低精度 partial/atomic：速度可能达标但正确性失败，不能作为保留路径。
- 全局 tune config 加入某个 shape 的最佳 meta：如果 tune key 不区分 dtype/shape regime，可能让其他 dtype 回退。
- output cache 复用：可能减少分配开销，但改变返回 tensor 复用语义；除非接口允许且收益明确，否则不要保留。

## 迭代记录要点

每轮复杂算子优化日志至少记录：

- 目标 shape、dtype、起始 latency/speedup 和当前命中 config。
- 瓶颈假设：索引、layout、Tensor Core、reduction、atomic、launch、graph wrapper、baseline algorithm gap。
- 改动类别：META、dispatch、layout pack、kernel mapping、split/reduce、zero/atomic、prepared wrapper。
- 正确性状态：是否触及数值路径、最小功能测试结果、最大误差或失败 dtype。
- 性能结果：focused timing 与是否需要 full benchmark。
- 工具证据：PTX/SASS/NCU 状态，特别是 Tensor Core、div/rem、local memory、barrier、atomic。
- 保留或回退决定：为什么保留，或为什么拒绝并完全移除。
