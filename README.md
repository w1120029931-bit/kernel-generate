# kernel-generate skills 仓库

本仓库只包含一个智能体 skill：`kernel-generate`。

这个 skill 用于指导 AI 智能体完成通用的高性能 Triton kernel 开发流程：

1. 从当前任务和具体仓库契约中整理算子接口、测试、约束和性能测试契约。
2. 编写符合当前任务接口的完整功能性 Triton kernel。
3. 编写并执行功能测试，直到正确性通过。
4. 编写并执行性能测试，并与基准实现对比。
5. 当 `speedup < 0.9` 时，围绕明确瓶颈迭代优化 kernel，并把每轮瓶颈、优化方式和测试结果记录到临时优化日志中，直到性能合格。

## 目录结构

```text
.
|-- README.md
|-- pyproject.toml
|-- scripts/
|   |-- install_to_codex.sh
|   |-- install_to_claude.sh
|   `-- validate_skills.py
`-- skills/
    `-- kernel-generate/
        |-- LICENSE.txt
        |-- SKILL.md
        |-- scripts/
        |   `-- check_kernel_tools.py
        `-- references/
            |-- custom_op_contract.md
            `-- kernel_impl_optimization.md
```

## 文档拆分

- `SKILL.md`：描述 skill 的入口、资源加载方式、算子契约整理、功能测试、性能测试和优化主流程。
- `references/custom_op_contract.md`：唯一记录具体实现仓库规则、路径、测试框架、性能测试框架和项目约束的文件。
- `references/kernel_impl_optimization.md`：只描述通用 Triton kernel 实现、启动配置设计、访存和性能优化。

## 安装到 Codex

本地开发推荐软链接安装：

```bash
cd /home/wangbingjie/kernel-generate
./scripts/install_to_codex.sh
```

默认安装到 `${CODEX_HOME:-$HOME/.codex}/skills/kernel-generate`。软链接安装后，仓库内容更新会直接反映到 skill；重启 Codex 或开启新会话即可重新加载。

如果需要复制安装当前版本：

```bash
./scripts/install_to_codex.sh --copy
```

如果仓库已发布到 GitHub，也可以在 Codex 中使用 `$skill-installer` 从远端安装：

```text
$skill-installer install https://github.com/<owner>/<repo>/tree/main/skills/kernel-generate
```

注意：`$skill-installer` 从 GitHub 安装时，目标 skill 已存在会中止。需要更新时，先删除旧目录，或继续使用本地软链接安装。

## 安装到 Claude Code

本仓库的 skill 使用 Claude Code 原生的 `SKILL.md` + frontmatter 格式，无需改造即可被 Claude CLI 识别。

本地开发推荐软链接安装：

```bash
cd /home/wangbingjie/kernel-generate
./scripts/install_to_claude.sh
```

默认安装到 `${CLAUDE_CONFIG_DIR:-$HOME/.claude}/skills/kernel-generate`。软链接安装后，仓库内容更新会直接反映到 skill；重启 Claude Code 或开启新会话即可重新加载。

如果需要复制安装当前版本：

```bash
./scripts/install_to_claude.sh --copy
```

加载后可用 `/kernel-generate` 显式调用，或用自然语言触发（匹配 `SKILL.md` 中的 `description`）。

如需随仓库共享给团队，也可改装到项目级目录 `<repo>/.claude/skills/kernel-generate/` 并提交进 git，clone 仓库的人即可自动获得该 skill。

## 校验

执行：

```bash
python3 scripts/validate_skills.py
```

该脚本会检查仓库中是否只包含一个 skill，校验 `SKILL.md` 的 frontmatter 元数据，并确认本地引用文件存在。
