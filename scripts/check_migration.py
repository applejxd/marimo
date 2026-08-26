#!/usr/bin/env python3
"""Check that archived, marimo, and published notebooks correspond one-to-one."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from build_site import discover

ROOT = Path(__file__).resolve().parents[1]


def relative_files(root: Path, suffix: str) -> set[Path]:
    return {path.relative_to(root).with_suffix("") for path in root.rglob(f"*{suffix}")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-site", action="store_true")
    args = parser.parse_args()

    archived = relative_files(ROOT / "legacy", ".ipynb")
    converted = {notebook.relative.with_suffix("") for notebook in discover()}
    overrides = {
        Path(source).with_suffix(""): Path(target).with_suffix("")
        for source, target in json.loads(
            (ROOT / "migration-map.json").read_text(encoding="utf-8")
        ).items()
    }
    expected = {overrides.get(source, source) for source in archived}
    failures = []
    if expected != converted:
        failures.append(f"missing marimo: {sorted(expected - converted)}")
        failures.append(f"without archive: {sorted(converted - expected)}")
    if args.require_site:
        published = relative_files(ROOT / "site", ".html") - {Path("index")}
        if expected != published:
            failures.append(f"missing HTML: {sorted(expected - published)}")
            failures.append(f"unexpected HTML: {sorted(published - expected)}")
        manifest_path = ROOT / "site" / "notebooks-manifest.json"
        if not manifest_path.exists():
            failures.append("missing site/notebooks-manifest.json")
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entries = manifest.get("notebooks", {})
            expected_sources = {notebook.relative.as_posix() for notebook in discover()}
            if set(entries) != expected_sources:
                failures.append(
                    f"manifest source mismatch: expected {len(expected_sources)}, "
                    f"found {len(entries)}"
                )
            for source_name, entry in entries.items():
                source_path = ROOT / "notebooks" / source_name
                html_path = ROOT / "site" / entry["html"]
                if not source_path.exists() or not html_path.exists():
                    continue
                source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
                html_hash = hashlib.sha256(html_path.read_bytes()).hexdigest()
                if source_hash != entry.get("source_sha256"):
                    failures.append(f"stale HTML source mapping: {source_name}")
                if html_hash != entry.get("html_sha256"):
                    failures.append(f"modified generated HTML: {entry['html']}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(
        f"Migration mapping is complete: {len(archived)} notebooks "
        f"({len(overrides)} renamed)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
