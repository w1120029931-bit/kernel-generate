#!/usr/bin/env python3
"""校验单 skill 仓库布局。"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REQUIRED_FIELDS = {"name", "description"}
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$")
MAX_NAME_LEN = 64
MAX_DESC_LEN = 1024
CUSTOM_CONTRACT = Path("references/custom_op_contract.md")


def parse_frontmatter(text: str) -> tuple[dict[str, str], list[str]]:
    errors: list[str] = []
    fields: dict[str, str] = {}

    if not text.startswith("---\n"):
        return fields, ["缺少 YAML frontmatter"]

    parts = text.split("---", 2)
    if len(parts) < 3:
        return fields, ["YAML frontmatter 格式不完整"]

    current_key: str | None = None
    current_value: list[str] = []

    def flush() -> None:
        if current_key is not None:
            fields[current_key] = " ".join(current_value).strip()

    for line in parts[1].splitlines():
        if not line.strip():
            continue

        if line.startswith((" ", "\t")) and current_key is not None:
            current_value.append(line.strip())
            continue

        if ":" not in line:
            continue

        flush()
        key, _, value = line.partition(":")
        current_key = key.strip()
        value = value.strip().strip("'\"")
        current_value = [] if value in {">", "|", ">-", "|-"} else [value]

    flush()
    return fields, errors


def find_local_refs(markdown: str) -> list[str]:
    refs: list[str] = []
    for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", markdown):
        ref = match.group(1)
        if not ref.startswith(("http://", "https://", "#")):
            refs.append(ref)
    return refs


def collect_custom_terms(custom_text: str) -> set[str]:
    """从具体仓库契约中提取不应出现在其他 Markdown 文档中的仓库专属词。"""
    terms: set[str] = set()

    repo_match = re.search(r"仓库名称：\s*([^\n`]+)", custom_text)
    if repo_match:
        terms.add(repo_match.group(1).strip())

    for code_span in re.findall(r"`([^`]+)`", custom_text):
        if "/" in code_span or code_span.endswith((".yaml", ".yml", ".toml", ".py")):
            terms.add(code_span.strip())
            for part in re.split(r"[/\\]", code_span):
                part = part.strip()
                if len(part) >= 6 and re.search(r"[A-Za-z_]", part):
                    terms.add(part)

    return {term for term in terms if term and term != CUSTOM_CONTRACT.as_posix()}


def validate_custom_contract_isolation(skill_dir: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    custom_path = skill_dir / CUSTOM_CONTRACT

    if not custom_path.exists():
        errors.append(f"{skill_dir.name}: 缺少具体仓库契约文件 {CUSTOM_CONTRACT}")
        return errors, warnings

    custom_text = custom_path.read_text(encoding="utf-8")
    terms = collect_custom_terms(custom_text)
    if not terms:
        warnings.append(f"{skill_dir.name}: {CUSTOM_CONTRACT} 未提取到具体仓库关键词")
        return errors, warnings

    markdown_files = [skill_dir / "SKILL.md", *sorted((skill_dir / "references").glob("*.md"))]
    for markdown_file in markdown_files:
        if markdown_file == custom_path:
            continue
        text = markdown_file.read_text(encoding="utf-8")
        for term in sorted(terms, key=len, reverse=True):
            if term in text:
                rel = markdown_file.relative_to(skill_dir)
                errors.append(f"{skill_dir.name}: 具体仓库信息 '{term}' 不应出现在 {rel}")

    return errors, warnings


def validate_skill(skill_dir: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    skill_md = skill_dir / "SKILL.md"

    if not skill_md.exists():
        return [f"{skill_dir.name}: 缺少 SKILL.md"], warnings

    text = skill_md.read_text(encoding="utf-8")
    fields, fm_errors = parse_frontmatter(text)
    errors.extend(f"{skill_dir.name}: {error}" for error in fm_errors)

    for field in REQUIRED_FIELDS:
        if not fields.get(field):
            errors.append(f"{skill_dir.name}: 缺少必要字段 '{field}'")

    name = fields.get("name", "")
    if name:
        if name != skill_dir.name:
            errors.append(f"{skill_dir.name}: frontmatter 中的 name 与目录名不一致")
        if len(name) > MAX_NAME_LEN:
            errors.append(f"{skill_dir.name}: name 超过 {MAX_NAME_LEN} 个字符")
        if not NAME_PATTERN.match(name):
            errors.append(f"{skill_dir.name}: skill 名称不合法 '{name}'")

    description = fields.get("description", "")
    if description and len(description) > MAX_DESC_LEN:
        errors.append(f"{skill_dir.name}: description 超过 {MAX_DESC_LEN} 个字符")
    if description and len(description) < 40:
        warnings.append(f"{skill_dir.name}: description 偏短")

    body = text.split("---", 2)[-1].strip()
    if len(body) < 100:
        errors.append(f"{skill_dir.name}: SKILL.md 正文过短")

    if "[TODO" in text:
        warnings.append(f"{skill_dir.name}: SKILL.md 仍包含 TODO 标记")

    for ref in find_local_refs(body):
        if not (skill_dir / ref).exists():
            errors.append(f"{skill_dir.name}: 引用文件不存在: {ref}")

    if not (skill_dir / "LICENSE.txt").exists():
        warnings.append(f"{skill_dir.name}: 缺少 LICENSE.txt")

    isolation_errors, isolation_warnings = validate_custom_contract_isolation(skill_dir)
    errors.extend(isolation_errors)
    warnings.extend(isolation_warnings)

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repo",
        nargs="?",
        default=Path(__file__).resolve().parent.parent,
        type=Path,
        help="需要校验的仓库根目录",
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    skills_dir = repo / "skills"
    if not skills_dir.exists():
        print(f"错误：缺少 skills 目录: {skills_dir}")
        return 1

    skill_dirs = [path for path in sorted(skills_dir.iterdir()) if path.is_dir()]
    errors: list[str] = []
    warnings: list[str] = []

    if len(skill_dirs) != 1:
        errors.append(f"仓库必须只包含一个 skill，当前数量为 {len(skill_dirs)}")

    for skill_dir in skill_dirs:
        skill_errors, skill_warnings = validate_skill(skill_dir)
        errors.extend(skill_errors)
        warnings.extend(skill_warnings)

    for error in errors:
        print(f"错误：{error}")
    for warning in warnings:
        print(f"警告：{warning}")

    if errors:
        print(f"失败：{len(errors)} 个错误，{len(warnings)} 个警告")
        return 1

    print(f"通过：{len(skill_dirs)} 个 skill，{len(warnings)} 个警告")
    return 0


if __name__ == "__main__":
    sys.exit(main())
