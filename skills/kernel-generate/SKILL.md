---
name: kernel-generate
description: >
  指导 AI 智能体开发算子 kernel、功能测试、性能测试并迭代优化高性能 Triton kernel。
  当用户要求生成或实现 GPU 算子或 kernel、编写 Triton kernel、优化 kernel 性能、调试 kernel 正确性、
  运行功能测试或性能测试，或显式调用 kernel-generate 时使用。
---

# kernel-generate

## 概述

使用这个 skill 开发完整的 Triton kernel 实现：实现必须符合当前任务的算子接口，先通过功能测试证明正确性，再通过性能测试证明性能。流程需要保持通用：主流程和通用 kernel 方法不绑定任何具体实现仓库；所有具体仓库规则只允许写入 [references/custom_op_contract.md](references/custom_op_contract.md)。

## 资源加载

按需读取以下文件：

- [references/custom_op_contract.md](references/custom_op_contract.md)：实现前必须读取。这是唯一允许记录具体实现仓库规则、路径、测试框架、性能测试框架、代码风格和项目约束的文件。
- [references/kernel_impl_optimization.md](references/kernel_impl_optimization.md)：编写或修改 Triton 代码前必须读取。用于 kernel 设计、启动结构、索引、mask、访存行为和性能优化。

## 执行规则

1. 写代码前先读取 `custom_op_contract.md`，并把其中的具体仓库规则作为当前任务的硬约束。
2. 不要在 `SKILL.md`、`kernel_impl_optimization.md` 或 README 中新增具体仓库名、仓库路径、测试目录、benchmark 目录、注册方式或项目专属代码风格；这些内容只能写入 `custom_op_contract.md`。
3. 实现前先整理算子契约：公开 API、参数签名、输出约定、shape/dtype/layout 规则、参考实现、误差容忍度、功能测试覆盖、性能测试基准和通过条件。
4. 不要跳过功能测试。功能测试失败时，必须先分析并修复正确性，再运行性能测试。
5. 使用 `speedup = baseline_time / candidate_time`。所有必测性能测试用例的 `speedup >= 0.9` 才算性能合格。
6. 性能不合格时，下一轮优化前必须创建或更新临时优化日志。
7. 性能不合格时，不能只报告“未达标”。必须用实测证据说明主要瓶颈属于 benchmark 偏差、包装层/graph 开销、kernel 本体、baseline 算法差距、工具/环境阻塞，还是仍未归因。
8. 在向用户报告性能未达标、阻塞或放弃继续优化前，必须至少对一个代表性慢用例执行 PTX dump；如果 SASS 或 NCU 工具可用，还必须完成 SASS/NCU 分析。工具不可用时，必须在优化日志和最终回复中记录具体不可用原因和替代证据。
9. 只有在有 roofline/硬件指标、PTX/SASS/NCU 证据、baseline 算法证据或项目约束证据时，才可以判断“理论上或当前技术路径上难以达到”。否则只能说“当前实现尚未达标，尚不能证明理论不可达”。
10. 优先使用 `custom_op_contract.md` 指定的测试和性能测试约定。如果该文件没有规定，则从当前任务上下文中推断，并把新发现的具体仓库规则补充到该文件。
11. 只有当关键语义、接口要求或性能测试基准无法从代码、文档和 `custom_op_contract.md` 中推断，并且会阻塞实现或验证时，才向用户提问。

## 工作流

### 步骤 1：读取仓库契约并整理算子契约

读取 [references/custom_op_contract.md](references/custom_op_contract.md)。识别用户请求的算子、期望接口、参考实现、实现位置、测试位置、性能测试位置、依赖假设和可用 GPU 运行环境，并整理当前算子的具体契约。

算子契约至少包含：

- 接口：公开 API、参数签名、输入输出、in-place 或别名关系。
- 语义：参考实现、shape 规则、dtype 规则、layout/stride 规则、数值误差容忍度。
- 功能测试：必测用例、覆盖维度、测试命令。
- 性能测试：基准实现、候选实现、测试 shape、指标、通过条件和测试命令。

如果发现新的具体仓库规则，先补充到 `custom_op_contract.md`，再继续实现。如果接口、语义、误差容忍度或性能测试基准缺失，先记录显式假设；只有阻塞实现或验证的字段才向用户确认。

### 步骤 2：实现完整功能 kernel

读取 [references/kernel_impl_optimization.md](references/kernel_impl_optimization.md)。根据算子契约和 `custom_op_contract.md` 的硬约束实现 Triton kernel、Python 封装函数、启动网格、输出分配、dtype/shape 处理和集成代码。初版实现优先保证正确性，同时避免明显的性能反模式。

### 步骤 3：编写并执行功能测试

根据算子契约和 `custom_op_contract.md` 编写功能测试。覆盖必需 shape、dtype、边界用例、layout 约束、广播或 stride 行为，以及数值误差容忍度。执行测试。

如果任何测试失败，先判断失败类别，再修改代码或测试，并重新执行功能测试。重复该过程直到功能测试通过。

常见失败类别：接口不匹配、shape/索引 bug、数值 bug、内存越界或别名关系问题、测试期望与契约不一致。

### 步骤 4：编写并执行性能测试

功能测试通过后，根据算子契约和 `custom_op_contract.md` 编写性能测试。将候选 Triton kernel 与约定基准实现对比，并为每个必测用例计算 speedup。

性能结论必须先确认 benchmark 有效：候选和 baseline 使用相同 dtype/layout/shape、相同 warmup/repeat/synchronization 策略，没有把编译、dump、一次性 graph capture 或异常处理计入候选侧耗时。如果排除某个 shape，必须记录触发原因、复现现象和为什么该 shape 不应计入必测集合。

如果所有必测用例的 `speedup >= 0.9`，则性能合格。报告完成前，重新执行最终功能测试和性能测试命令。最终性能结论不要默认使用 `TRITON_ALWAYS_COMPILE=1`，除非确认编译时间没有计入 benchmark。

### 步骤 5：性能迭代优化

如果任一必测用例的 `speedup < 0.9`，创建或更新临时优化日志。优先使用当前任务目录下的 `.kernel-generate/optimization-log-<operator>.md`；如果当前任务不适合写入临时文件，则使用 `/tmp/kernel-generate-optimization-log-<operator>.md`。

性能调优过程中，每次优化代码后，不要重复执行完整功能测试；只有优化可能影响正确性时，才执行最小相关功能测试子集。所有性能测试都合格后，最后再执行完整功能测试。

性能归因按以下顺序进行，并把证据写入临时优化日志：

1. benchmark 有效性：确认计时方式、编译缓存、warmup、同步、异常 shape 排除和输出 speedup 计算没有偏差。
2. 包装层/graph 归因：如果性能测试经过上层 graph、runner、prepared op 或其他包装层，必须用受控对比隔离包装层开销；没有受控数据时，不能声称包装层不是瓶颈。
3. kernel 本体归因：检查 kernel 算法结构、tile 映射、访存连续性、冗余 load/store、mask、整数 div/rem、寄存器压力、occupancy 和 autotune 配置。
4. baseline 差距归因：如果 baseline 可能使用 Tensor Core、Winograd、FFT、implicit GEMM、fusion、specialized layout 或持久化 kernel 等更强算法，必须记录证据，并说明当前 Triton 路径是否有等价实现空间。

每一轮执行：

1. 根据性能测试结果、kernel 结构、访存模式、启动配置、occupancy、寄存器压力线索和 shape 特征分析当前瓶颈。
2. 对 kernel 或启动配置应用一次有针对性的优化。
3. 如果正确性可能受影响，重新执行相关功能测试。
4. 重新执行性能测试并观察 speedup。
5. 将本轮慢用例、实测 speedup、证据类型、瓶颈假设、针对性优化、修改文件、功能测试结果、性能测试结果、Triton autotune 输出摘要、PTX/SASS/NCU 状态和下一步假设追加到临时优化日志。

如果普通参数调节、少量 autotune 配置和代码级检查仍无法解释或修复性能问题，按 [references/kernel_impl_optimization.md](references/kernel_impl_optimization.md) 的“PTX / SASS / NCU 深度调优”流程升级分析。深度调优不是可选总结材料：报告阻塞或性能未达标前，必须固定至少一个代表性慢用例，dump PTX/cubin；如果工具可用，反汇编 SASS 并运行 NCU，用硬件指标判断真实瓶颈；每轮只改一个主要点，并把 PTX/SASS/NCU 证据写入临时优化日志。

重复该循环直到性能合格。如果同一瓶颈反复出现，且已经完成深度调优或明确记录工具不可用，并且已经没有可信的下一步优化路径，则带上优化日志路径、当前最佳测试结果、瓶颈分类和证据向用户报告阻塞点。

### 步骤 6：报告结果

总结最终修改的算子文件、功能测试命令和结果、性能测试命令和结果、最终 speedup，以及临时优化日志路径（如果创建过）。同时说明算子契约中仍保留的假设。

如果性能未达标，最终回复必须包含：

- 当前最佳 speedup 和最慢用例。
- 已完成的优化轮次和深度调优证据摘要。
- 主要瓶颈分类：benchmark 偏差、包装层/graph 开销、kernel 本体、baseline 算法差距、工具/环境阻塞或仍未归因。
- 为什么当前还不能达标；如果不能证明理论不可达，必须明确说“尚不能证明理论不可达”。
- 下一步最可信的优化方向，或者说明为什么没有可信路径。

## 示例

**示例 1：新增算子**

```text
用户说：“使用 kernel-generate 为当前仓库实现一个 Triton RMSNorm kernel。”
动作：
  1. 读取 custom_op_contract.md，确认当前仓库规则。
  2. 整理算子契约并实现 kernel。
  3. 添加功能测试并迭代直到通过。
  4. 添加性能测试并优化直到 speedup >= 0.9。
结果：得到一个已集成到当前仓库、功能测试和性能测试均通过的 Triton kernel。
```

**示例 2：优化已有慢 kernel**

```text
用户说：“优化这个 Triton softmax kernel；当前 speedup 只有 0.7。”
动作：
  1. 读取 custom_op_contract.md，确认功能契约和性能测试基准。
  2. 运行或重建失败的性能测试。
  3. 创建优化日志。
  4. 围绕具体瓶颈迭代修改，直到 speedup >= 0.9。
结果：得到优化后的 kernel，以及一份简短记录瓶颈、修改和性能测试结果的优化日志。
```

## 排障

| 问题 | 原因 | 处理 |
|------|------|------|
| 算子接口不明确 | 当前任务缺少公开封装、签名或文档 | 先从上下文整理草案契约，只向用户确认阻塞字段 |
| 具体仓库规则散落在其他文件 | 文档边界被破坏 | 将仓库名、路径、测试框架、性能测试框架和项目约束迁回 `custom_op_contract.md` |
| 优化后功能测试失败 | 优化改动影响了索引、mask、dtype、别名关系或误差行为 | 用最小失败用例复现，先修复正确性，再继续性能测试 |
| speedup 低于 0.9 | benchmark、包装层、kernel 本体或 baseline 算法差距均可能导致 | 追加本轮优化日志，按分层归因收集证据；报告失败前必须完成 PTX，工具可用时完成 SASS/NCU |
