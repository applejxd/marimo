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
DISPLAY_DELIMITER = re.compile(r"^\s*\$\$\s*$")
MARKDOWN_LIST_ITEM = re.compile(r"^\s*(?:[-+*]|\d+\.)\s")
CODE_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
INLINE_CODE = re.compile(r"(`+).*?\1")


def markdown_strings(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "md"
            and node.args
        ):
            continue
        argument = node.args[0]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            markdown = argument.value
        elif isinstance(argument, ast.JoinedStr):
            markdown = "".join(
                part.value if isinstance(part, ast.Constant) else "EXPR"
                for part in argument.values
            )
        else:
            continue
        values.append((argument.lineno, markdown))
    return values


def prose_lines(
    markdown: str,
) -> tuple[list[tuple[int, str, str]], int | None]:
    lines = []
    fence = None
    fence_offset = None
    for offset, line in enumerate(markdown.splitlines()):
        if match := CODE_FENCE.match(line):
            marker = match.group(1)
            if fence is None:
                fence = marker
                fence_offset = offset
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
                fence_offset = None
            continue
        if fence is None:
            lines.append((offset, line, INLINE_CODE.sub("", line)))
    return lines, fence_offset


def main() -> int:
    failures = []
    for path in sorted(NOTEBOOKS.rglob("*.py")):
        for line, markdown in markdown_strings(path):
            lines, fence_offset = prose_lines(markdown)
            if fence_offset is not None:
                failures.append(
                    f"{path.relative_to(ROOT)}:{line + fence_offset}: "
                    "unclosed Markdown code fence"
                )
            for offset, _, markdown_line in lines:
                if match := UNSUPPORTED.search(markdown_line):
                    failures.append(
                        f"{path.relative_to(ROOT)}:{line + offset}: unsupported "
                        f"display math delimiter {match.group(0)!r}; use "
                        "$$...$$ and aligned"
                    )
            delimiter_lines = [
                offset for offset, _, text in lines if "$$" in text
            ]
            if sum(text.count("$$") for _, _, text in lines) % 2:
                failures.append(
                    f"{path.relative_to(ROOT)}:"
                    f"{line + delimiter_lines[-1]}: unbalanced $$ delimiters"
                )
            inside_display = False
            for offset, raw_line, markdown_line in lines:
                if "$$" in markdown_line:
                    if not DISPLAY_DELIMITER.fullmatch(raw_line):
                        failures.append(
                            f"{path.relative_to(ROOT)}:{line + offset}: put $$ "
                            "on a line by itself"
                        )
                    if markdown_line.count("$$") % 2:
                        inside_display = not inside_display
                elif inside_display and MARKDOWN_LIST_ITEM.match(markdown_line):
                    failures.append(
                        f"{path.relative_to(ROOT)}:{line + offset}: display "
                        "math line resembles a Markdown list item; remove the "
                        "space after its leading operator"
                    )
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("Notebook display math delimiters are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
