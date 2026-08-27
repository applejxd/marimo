#!/usr/bin/env python3
"""Reject display-math syntax that marimo Markdown does not render reliably."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
UNSUPPORTED = re.compile(
    r"(?<!\\)\\\[|(?<!\\)\\\]|"
    r"\\(?:begin|end)\{"
    r"(?:equation\*?|align\*?|alignat\*?|split|gather\*?|multline\*?|eqnarray\*?)"
    r"\}"
)


def markdown_strings(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "md"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            continue
        values.append((node.lineno, node.args[0].value))
    return values


def main() -> int:
    failures = []
    for path in sorted(NOTEBOOKS.rglob("*.py")):
        for line, markdown in markdown_strings(path):
            if match := UNSUPPORTED.search(markdown):
                failures.append(
                    f"{path.relative_to(ROOT)}:{line}: unsupported display math "
                    f"delimiter {match.group(0)!r}; use $$...$$ and aligned"
                )
            if markdown.count("$$") % 2:
                failures.append(
                    f"{path.relative_to(ROOT)}:{line}: unbalanced $$ delimiters"
                )
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("Notebook display math delimiters are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
