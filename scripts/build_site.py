#!/usr/bin/env python3
"""Execute every marimo notebook and build the static GitHub Pages site."""

from __future__ import annotations

import argparse
import ast
import hashlib
import html
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
SITE = ROOT / "site"
LOGS = ROOT / "build" / "logs"
ERROR_MARKERS = (
    "some cells failed to execute",
    "MarimoExceptionRaisedError",
    "Traceback (most recent call last)",
)
HEADING_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Notebook:
    source: Path
    relative: Path
    title: str

    @property
    def output(self) -> Path:
        return SITE / self.relative.with_suffix(".html")


def is_marimo_notebook(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return False
    imports_marimo = any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            any(alias.name == "marimo" for alias in node.names)
            if isinstance(node, ast.Import)
            else node.module == "marimo"
        )
        for node in tree.body
    )
    has_app = any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "app" for target in node.targets)
        for node in tree.body
    )
    return imports_marimo and has_app


def extract_title(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    for string in (
        node.value
        for node in ast.walk(ast.parse(source, filename=str(path)))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ):
        match = HEADING_RE.search(string)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()
    return path.stem.replace("_", " ").replace("-", " ").title()


def discover() -> list[Notebook]:
    notebooks = []
    for source in sorted(NOTEBOOKS.rglob("*.py")):
        if is_marimo_notebook(source):
            relative = source.relative_to(NOTEBOOKS)
            notebooks.append(Notebook(source, relative, extract_title(source)))
    return notebooks


def export(notebook: Notebook, log_dir: Path) -> str | None:
    notebook.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / notebook.relative.with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "uv",
        "run",
        "--all-groups",
        "marimo",
        "export",
        "html",
        str(notebook.source),
        "-o",
        str(notebook.output),
        "--force",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(result.stdout, encoding="utf-8")
    output_text = ""
    if notebook.output.exists():
        output_text = notebook.output.read_text(encoding="utf-8", errors="replace")
    marker = next(
        (candidate for candidate in ERROR_MARKERS if candidate in result.stdout or candidate in output_text),
        None,
    )
    if result.returncode == 0 and marker is None and notebook.output.exists():
        return None
    notebook.output.unlink(missing_ok=True)
    reason = marker or f"marimo exited with status {result.returncode}"
    return f"{notebook.relative}: {reason} (log: {log_path})"


def write_index(notebooks: list[Notebook]) -> None:
    groups: dict[str, list[Notebook]] = {}
    for notebook in notebooks:
        category = notebook.relative.parts[0] if len(notebook.relative.parts) > 1 else "other"
        groups.setdefault(category, []).append(notebook)
    sections = []
    for category, entries in sorted(groups.items()):
        links = "\n".join(
            f'          <li><a href="{html.escape(entry.relative.with_suffix(".html").as_posix())}">'
            f"{html.escape(entry.title)}</a></li>"
            for entry in entries
        )
        sections.append(
            f"      <section><h2>{html.escape(category.title())}</h2><ul>\n{links}\n"
            "        </ul></section>"
        )
    payload = f"""<!doctype html>
<html lang="ja">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="description" content="実行済み marimo scientific notebooks">
    <title>Scientific marimo notebooks</title>
    <style>
      :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
      body {{ max-width: 64rem; margin: 0 auto; padding: 2rem 1rem 4rem; line-height: 1.6; }}
      main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr)); gap: 1rem; }}
      section {{ border: 1px solid #8886; border-radius: .75rem; padding: .5rem 1.25rem; }}
      li {{ margin: .45rem 0; }}
    </style>
  </head>
  <body>
    <h1>Scientific marimo notebooks</h1>
    <p>{len(notebooks)} notebooks are generated automatically from <code>notebooks/</code>.</p>
    <main>
{chr(10).join(sections)}
    </main>
  </body>
</html>
"""
    (SITE / "index.html").write_text(payload, encoding="utf-8")
    (SITE / ".nojekyll").touch()


def write_manifest(
    notebooks: list[Notebook], updated_notebooks: list[Notebook] | None = None
) -> None:
    manifest_path = SITE / "notebooks-manifest.json"
    if updated_notebooks is None:
        entries = {}
        targets = notebooks
    else:
        if not manifest_path.exists():
            raise FileNotFoundError(
                "A partial build requires an existing manifest; run a full build first."
            )
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = existing.get("notebooks", {})
        targets = updated_notebooks
    for notebook in targets:
        if not notebook.output.exists():
            raise FileNotFoundError(f"Missing exported notebook: {notebook.output}")
        entries[notebook.relative.as_posix()] = {
            "source_sha256": hashlib.sha256(notebook.source.read_bytes()).hexdigest(),
            "html": notebook.relative.with_suffix(".html").as_posix(),
            "html_sha256": hashlib.sha256(notebook.output.read_bytes()).hexdigest(),
        }
    expected_sources = {notebook.relative.as_posix() for notebook in notebooks}
    if updated_notebooks is not None:
        # リネームや削除で消えた source を落とす。これをしないと部分ビルドでは
        # 旧名のエントリが残り、下の照合に失敗して全再ビルドを強いられる。
        entries = {
            name: entry for name, entry in entries.items() if name in expected_sources
        }
    if set(entries) != expected_sources:
        raise ValueError(
            "Manifest source set differs from discovered notebooks; run a full build."
        )
    manifest = {"version": 1, "notebooks": entries}
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def copy_assets() -> None:
    assets = ROOT / "assets"
    if assets.exists():
        shutil.copytree(assets, SITE / "assets", dirs_exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Generate the index from discovered notebooks without executing them.",
    )
    parser.add_argument(
        "--notebook",
        action="append",
        type=Path,
        help="Build only this path relative to notebooks/; repeat to select multiple.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    notebooks = discover()
    if args.notebook:
        selected = {path.with_suffix(".py") for path in args.notebook}
        notebooks = [notebook for notebook in notebooks if notebook.relative in selected]
    if not notebooks:
        print("No marimo notebooks found.", file=sys.stderr)
        return 1

    SITE.mkdir(exist_ok=True)
    if args.discover_only:
        write_index(notebooks)
        copy_assets()
        print(json.dumps({"notebooks": len(notebooks), "executed": False}))
        return 0

    if not args.notebook:
        for stale_html in SITE.rglob("*.html"):
            stale_html.unlink()
        shutil.rmtree(LOGS, ignore_errors=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for index, notebook in enumerate(notebooks, start=1):
        print(f"[{index}/{len(notebooks)}] {notebook.relative}", flush=True)
        if failure := export(notebook, LOGS):
            failures.append(failure)
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1

    all_notebooks = discover()
    write_index(all_notebooks)
    write_manifest(all_notebooks, notebooks if args.notebook else None)
    copy_assets()
    print(json.dumps({"notebooks": len(notebooks), "executed": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
