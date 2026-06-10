#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SKILL_SRC="${REPO_ROOT}/skills/kernel-generate"
CLAUDE_HOME_DIR="${CLAUDE_CONFIG_DIR:-${HOME}/.claude}"
SKILLS_DIR="${CLAUDE_HOME_DIR}/skills"
SKILL_DEST="${SKILLS_DIR}/kernel-generate"
MODE="link"

usage() {
  cat <<'EOF'
用法：scripts/install_to_claude.sh [--link|--copy]

把 kernel-generate skill 安装到 Claude Code 用户级 skill 目录
（${CLAUDE_CONFIG_DIR:-$HOME/.claude}/skills/kernel-generate）。

--link  使用软链接安装。适合本地开发；仓库更新后开启新会话即可生效。默认模式。
--copy  复制安装。适合想固定当前版本；仓库更新后需要重新运行本脚本。
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --link)
      MODE="link"
      shift
      ;;
    --copy)
      MODE="copy"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数：$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${SKILL_SRC}/SKILL.md" ]]; then
  echo "错误：未找到 skill 源目录：${SKILL_SRC}" >&2
  exit 1
fi

mkdir -p "${SKILLS_DIR}"
rm -rf "${SKILL_DEST}"

if [[ "${MODE}" == "link" ]]; then
  ln -s "${SKILL_SRC}" "${SKILL_DEST}"
else
  cp -R "${SKILL_SRC}" "${SKILL_DEST}"
fi

echo "已安装 kernel-generate 到：${SKILL_DEST}"
echo "安装模式：${MODE}"
echo "请重启 Claude Code 或开启新会话以加载 skill。"
echo "加载后可用 /kernel-generate 显式调用，或用自然语言触发。"
