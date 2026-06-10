#!/usr/bin/env python3
"""Check PTX, SASS, and NCU tools before kernel tuning."""

from __future__ import annotations

import shutil
import subprocess


TOOLS = [
    ("PTX", "ptxas", ["--version"]),
    ("SASS", "cuobjdump", ["--version"]),
    ("SASS", "nvdisasm", ["--version"]),
    ("NCU", "ncu", ["--version"]),
]


def version_line(command: str, args: list[str]) -> tuple[str, str]:
    path = shutil.which(command)
    if path is None:
        return "MISSING", "not found in PATH"

    try:
        result = subprocess.run(
            [path, *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
    except Exception as exc:  # noqa: BLE001
        return "ERROR", str(exc)

    output = (result.stdout or result.stderr).strip().splitlines()
    detail = output[0].strip() if output else "no version output"
    if result.returncode != 0:
        return "ERROR", detail
    return "OK", f"{path} ({detail})"


def main() -> int:
    statuses: dict[str, list[str]] = {"PTX": [], "SASS": [], "NCU": []}

    for group, command, args in TOOLS:
        status, detail = version_line(command, args)
        statuses[group].append(status)
        print(f"{group:4} {command:9} {status:7} {detail}")

    print()
    for group in ("PTX", "SASS", "NCU"):
        available = "available" if "OK" in statuses[group] else "unavailable"
        print(f"{group}: {available}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
