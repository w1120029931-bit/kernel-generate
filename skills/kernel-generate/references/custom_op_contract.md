# 具体仓库算子契约

本文件是 `kernel-generate` 中唯一允许记录具体实现仓库规则和信息的文档。凡是涉及具体仓库名、目录路径、代码风格、注册方式、测试框架、性能测试框架、硬性项目约束或本仓库特有开发流程的内容，都必须写在这里；其他文档只能保留通用流程和通用 kernel 方法。

## 使用规则

1. 开始任何算子实现、功能测试或性能测试前，先读取本文件。
2. 如果开发过程中发现新的仓库专属规则，先补充到本文件，再继续实现。
3. 其他文档不得复制本文件中的具体仓库名、路径、目录、命令或框架规则。
4. 如果本文件与通用流程文档冲突，以本文件为当前仓库的硬约束。
5. 本文件可以随目标仓库变化而替换或重写；通用文档不应因此修改。

## 当前目标仓库

- 仓库名称：FlagDNN
- kernel 语言：Triton
- 目标：为算子开发 Triton kernel、功能测试和性能测试，并按性能测试结果迭代优化。

## 运行环境

- Python 解释器：`/home/wangbingjie/py312/bin/python3`（venv，含 torch/triton/pytest/cudnn-frontend）。
- 以脚本文件方式运行 Python 时，必须前缀
  `LD_LIBRARY_PATH=/root/micromamba-root/envs/toolchain/lib:$LD_LIBRARY_PATH`，
  否则 `import cudnn` 因系统 libstdc++ 缺少 GLIBCXX_3.4.32 失败。
- 测试命令示例：
  `cd FlagDNN && LD_LIBRARY_PATH=/root/micromamba-root/envs/toolchain/lib:$LD_LIBRARY_PATH /home/wangbingjie/py312/bin/python3 -m pytest tests/test_<op>.py -v`
- GPU：NVIDIA H100 80GB（sm90）。

## 目录与文件位置

| 内容 | 位置 | 说明 |
|------|------|------|
| kernel 实现 | `FlagDNN/src/flag_dnn/ops` | 新增或修改 Triton 算子实现的位置。 |
| 功能测试 | `FlagDNN/tests` | 新增或修改功能测试的位置。 |
| 性能测试 | `FlagDNN/benchmark` | 新增或修改性能测试的位置。 |
| NVIDIA 自动调优配置 | `FlagDNN/src/flag_dnn/runtime/backend/_nvidia/tune_configs.yaml` | FlagDNN 默认且首选的自动调优配置位置。算子存在 tunable META 时，在此添加同名条目，并通过 `runtime.get_tuned_config()` 接入（详见“自动调优规则”）。 |

## kernel 开发规则

- 必须实现 Triton 代码。
- 代码和注释都不能使用中文。
- kernel 内绝对不能 fallback 到 `torch`、`cublas`、`cudnn` 或其他库完成核心计算。
- 算子存在 tunable META 时，必须使用仓库级 `tune_configs.yaml` + `runtime.get_tuned_config()` 机制（详见“自动调优规则”），不要内联硬编码 `triton.Config` 列表。
- 具体算子的 public API、参数签名、注册方式和导入路径应从 FlagDNN 现有算子中推断；如果无法推断，先记录假设并向用户确认阻塞字段。
- 待实现的算子的功能和接口应对标 cuDNN 算子；cuDNN 算子详情可通过 `/home/wangbingjie/cudnn-frontend` 库查找。

## cuDNN frontend 对标 API 选择规则

- 为 FlagDNN 选择待实现算子的 public API 前，必须先在 `/home/wangbingjie/cudnn-frontend` 中确认对应 cuDNN frontend 算子的 Python 绑定签名、C++ graph attributes、输入输出 tensor 语义和示例用法。
- API 签名优先参考 cuDNN frontend Python 绑定中用户实际调用的参数名和默认值；C++ graph attributes 用于确认完整语义、内部字段、overload 关系和 shorthand 映射。
- FlagDNN public API 应保持 Python 友好，并与现有 FlagDNN 算子命名、参数顺序和默认值风格一致；但不能因此丢失 cuDNN frontend 完整能力。
- 当 cuDNN frontend 同一算子存在 shorthand API 和完整 attributes API 时，先判断两者关系：如果 shorthand 只是完整 API 的受限映射，FlagDNN 应支持 shorthand，并通过可选高级参数支持完整 API；如果两个 overload 表达不同语义，则必须在算子契约中分别列出。
- 对称参数与非对称参数都存在时，public API 应优先提供常用对称形式，并保留完整非对称形式。例如 `padding` 只是 `pre_padding == post_padding` 的简写时，应明确记录映射关系，不能把两者误认为两个不同算子。
- 不要把 cuDNN frontend 的 graph `Tensor_attributes`、C++ builder 对象或仅用于内部 graph 构建的字段直接暴露为 FlagDNN public API，除非 FlagDNN 现有同类算子已经采用这种模式。
- 如果 cuDNN frontend 的完整能力与 FlagDNN 现有同类算子 API 风格冲突，优先保留 FlagDNN 常用调用方式，并增加无歧义的高级可选参数；如果无法无歧义兼容，先在算子契约中记录冲突并向用户确认。
- 功能测试必须覆盖 public API 的常用简写形式；如果实现了完整 attributes 对标能力，还必须覆盖完整形式与简写形式的等价映射或差异语义。

## 功能测试规则

- 功能测试必须放在 `FlagDNN/tests`。
- 功能测试必须是 FlagDNN 的 triton 算子与 cudnn 算子做对比。
- 编写测试时参考 FlagDNN 其他算子的测试框架、命名方式、fixture 和断言风格。
- 功能测试 shape 需要根据当前算子的语义做针对性适配。
- shape 覆盖要充分，至少覆盖典型 shape、边界 shape 和能暴露索引/mask 问题的非规则 shape。
- dtype、layout、stride、广播、in-place 或 output-buffer 行为是否需要覆盖，以当前算子接口和现有同类算子规则为准。
- 如果测试框架要求固定随机种子、特定 device 初始化或特定图模式入口，遵循 FlagDNN 现有测试写法。

## 性能测试规则

- 性能测试必须放在 `FlagDNN/benchmark`。
- 性能测试必须是 FlagDNN 的 triton 算子与 cudnn 算子做对比。
- 编写性能测试时参考 FlagDNN 其他算子的性能测试框架、命名方式、输入构造、计时方式和输出格式。
- 性能测试文件的必测 shape 必须恰好 8 个，且这 8 个 shape 必须是该算子的重点性能 shape。
- 8 个重点性能 shape 需要根据当前算子的语义做针对性适配，优先来自真实模型或业务路径、已有 benchmark、典型生产规模、边界但性能敏感场景，以及能暴露访存/启动开销瓶颈的场景。
- 不得用重复、随意或只验证正确性的 shape 填数；如果无法确定 8 个重点 shape，先从代码、文档、已有 benchmark、算子契约和 cuDNN frontend 语义推断并记录假设，只有关键 shape 无法推断且会影响性能结论时才向用户确认。
- 根据仓库已有框架实现性能测试后，执行性能测试会输出 `speedup` 信息，因此不要在性能测试代码中编写 assert `speedup >= 0.9` 等类似代码。
- 当 `speedup < 0.9` 时，视为当前实现性能不合格。由于性能测试使用了 graph 功能，必须先量化性能差是否由 graph、runner、prepared op 或 capture/dispatch 开销引起；只有实测显示 graph 不是主要瓶颈后，才能把主要优化方向转向 kernel 本体。
- 第一次正式性能测试前，存在 tunable META 的算子必须完成 `tune_configs.yaml`/`runtime.get_tuned_config()` 接入，并用 `TRITON_PRINT_AUTOTUNING=1` 记录候选配置来源、调优输出和最终命中配置。若 `speedup < 0.9` 且 tunable META 仍硬编码，下一步必须接入或修正自动调优配置，不能继续只改固定 tile/warps/stages 作为主要优化路线。

## Graph 性能归因规则

- 如果声称 graph 不是主要瓶颈，必须在优化日志中记录受控对比数据，例如 FlagDNN graph path、direct op 或 prepared runner path、可访问时的 Triton kernel wrapper path 的耗时和 speedup。
- 如果 graph path 和 direct path 差距明显，先定位并优化 graph、runner、prepared op、capture 缓存或 dispatch 逻辑；不要直接归因到 kernel 算法。
- 如果 direct path 仍明显慢于 cuDNN frontend，继续按通用 PTX/SASS/NCU 深度调优流程分析 kernel 本体。
- 临时归因脚本或 benchmark 变体可以放在任务临时目录；只有对仓库后续有价值时才合入 `FlagDNN/benchmark`。

## 自动调优规则

- FlagDNN 的默认调优机制是仓库级 `tune_configs.yaml` + `runtime.get_tuned_config()`，当前 60+ 算子采用此方式，是本仓库的标准做法。新算子或正在优化的算子只要存在 tunable META，就必须优先采用此机制，不要内联硬编码 `triton.Config` 列表。
- 硬性门槛：对新实现或正在优化的 FlagDNN Triton 算子，只要存在 `BLOCK_*`、tile、split、`num_warps`、`num_stages`、grouping 等 tunable META，功能测试通过后必须先添加或更新 `_nvidia/tune_configs.yaml` 并通过 `runtime.get_tuned_config()` 接入，再运行第一次正式性能测试、进行多轮手动参数优化或下性能结论。固定 META 只允许作为正确性 bootstrap、受控单轮探索，或在有明确证据说明自动调优不适用/被阻塞时作为例外。
- 标准接法（参考 `src/flag_dnn/ops/relu.py`、`src/flag_dnn/ops/softmax.py`）：在算子 kernel 上用 `runtime.get_tuned_config("<op_name>")` 作为 `@triton.autotune` 的 `configs`：

  ```python
  from flag_dnn import runtime

  @triton.autotune(
      configs=runtime.get_tuned_config("<op_name>"),  # 读取 tune_configs.yaml 中的同名条目
      key=[...],                                       # 影响最优配置的 shape/dtype 等字段
  )
  @triton.jit
  def <op>_kernel(...):
      ...
  ```

- 在 `src/flag_dnn/runtime/backend/_nvidia/tune_configs.yaml` 中新增一个与 `get_tuned_config("<op_name>")` 同名的顶层条目。结构必须对齐已有条目：`gen: true` + `param_map`（META 字段映射、`num_warps`、`num_stages`）+ 各参数（如 `block_size`、`warps`、`stages`）的候选值列表。`configloader.py` 会按 `param_map` 自动把这些候选展开成 `triton.Config` 组合。
- 注意：`batch_norm.py` 等少数算子直接硬编码 `@triton.autotune(configs=[...])`、不读 YAML，属于历史/特例写法，不要作为新算子的参照模板。
- 配置项名称、层级结构和读取方式必须与现有 YAML 条目保持一致。
- 配置网格要围绕瓶颈假设收窄；不要为了穷举搜索铺过大网格，优先记录瓶颈假设并选择少量候选配置。

## 每个算子的补充契约模板

为具体算子开发时，可在临时优化日志或任务文档中使用以下模板；如果其中包含 FlagDNN 专属新规则，必须回填到本文件。

```markdown
# <operator> - FlagDNN 补充契约

## 集成位置
- kernel 文件：
- 功能测试文件：
- 性能测试文件：
- 自动调优配置：（tune_configs.yaml 中的条目名 + get_tuned_config 接入点）

## 接口规则
- public API：
- 参数签名：
- 输出约定：
- 注册或导入规则：

## 测试规则
- 功能测试 shape：
- dtype 覆盖：
- layout/stride 覆盖：
- 参考实现：

## 性能规则
- 基准实现：
- 性能测试 shape：恰好 8 个重点性能 shape + 选择依据
- speedup 输出方式：
- graph/direct/kernel 归因方式：
- 深度调优证据：PTX / SASS / NCU / 工具不可用原因
- 必须达到：speedup >= 0.9
```
