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

## 目录与文件位置

| 内容 | 位置 | 说明 |
|------|------|------|
| kernel 实现 | `FlagDNN/src/flag_dnn/ops` | 新增或修改 Triton 算子实现的位置。 |
| 功能测试 | `FlagDNN/tests_graph` | 新增或修改功能测试的位置。 |
| 性能测试 | `FlagDNN/benchmark_graph` | 新增或修改性能测试的位置。 |
| NVIDIA 自动调优配置 | `FlagDNN/src/flag_dnn/runtime/backend/_nvidia/tune_configs.yaml` | 如需使用仓库级自动调优配置，在此添加或修改配置。 |

## kernel 开发规则

- 必须实现 Triton 代码。
- 代码和注释都不能使用中文。
- kernel 内绝对不能 fallback 到 `torch`、`cublas`、`cudnn` 或其他库完成核心计算。
- 可以使用 Triton 原生自动调优配置。
- 也可以在 `FlagDNN/src/flag_dnn/runtime/backend/_nvidia/tune_configs.yaml` 中添加自动调优配置。
- 添加仓库级自动调优配置时，参考已有算子的配置格式和调用方式。
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

- 功能测试必须放在 `FlagDNN/tests_graph`。
- 编写测试时参考 FlagDNN 其他算子的测试框架、命名方式、fixture 和断言风格。
- 功能测试 shape 需要根据当前算子的语义做针对性适配。
- shape 覆盖要充分，至少覆盖典型 shape、边界 shape 和能暴露索引/mask 问题的非规则 shape。
- dtype、layout、stride、广播、in-place 或 output-buffer 行为是否需要覆盖，以当前算子接口和现有同类算子规则为准。
- 如果测试框架要求固定随机种子、特定 device 初始化或特定图模式入口，遵循 FlagDNN 现有测试写法。

## 性能测试规则

- 性能测试必须放在 `FlagDNN/benchmark_graph`。
- 编写性能测试时参考 FlagDNN 其他算子的性能测试框架、命名方式、输入构造、计时方式和输出格式。
- 性能测试 shape 需要根据当前算子的语义做针对性适配。
- shape 覆盖要充分，至少覆盖典型生产 shape、边界 shape 和能暴露访存/启动开销瓶颈的 shape。
- 根据仓库已有框架实现性能测试后，执行性能测试会输出 `speedup`信息，因此不要在性能测试代码中编写 assert `speedup >= 0.9` 等类似代码。
- 当 `speedup < 0.9` 时，视为当前实现性能不合格，需要继续分析瓶颈并优化。

## 自动调优规则

- 优先参考 FlagDNN 已有算子的调优配置风格。
- 如果算子性能对 block size、warps、stages 或其他 meta 参数敏感，允许增加少量有针对性的 Triton autotune 配置。
- 如果使用 `tune_configs.yaml`，配置项名称、层级结构和读取方式必须与现有算子保持一致。
- 不要为了穷举搜索创建过大的 autotune 配置网格；优先记录瓶颈假设并选择少量候选配置。

## 每个算子的补充契约模板

为具体算子开发时，可在临时优化日志或任务文档中使用以下模板；如果其中包含 FlagDNN 专属新规则，必须回填到本文件。

```markdown
# <operator> - FlagDNN 补充契约

## 集成位置
- kernel 文件：
- 功能测试文件：
- 性能测试文件：
- 自动调优配置：

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
- 性能测试 shape：
- speedup 输出方式：
- 必须达到：speedup >= 0.9
```
