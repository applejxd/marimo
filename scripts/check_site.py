#!/usr/bin/env python3
"""Validate the generated index, published notebook HTML files, and the manifest."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from build_site import discover

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
ERROR_RE = re.compile(r"MarimoExceptionRaisedError|Traceback \(most recent call last\)")


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in {"a", "img", "script", "link", "source", "video"}:
            return
        values = dict(attrs)
        attribute = "href" if tag in {"a", "link"} else "src"
        if values.get(attribute):
            self.links.append(values[attribute] or "")


def local_target(document: Path, link: str) -> Path | None:
    parsed = urlsplit(link)
    if parsed.scheme or parsed.netloc or link.startswith(("#", "data:", "blob:")):
        return None
    return (document.parent / unquote(parsed.path)).resolve()


def check_manifest() -> list[str]:
    """Verify that the committed HTML is up to date with the committed notebooks.

    CI は notebook を実行せず、コミット済みの HTML をそのまま配信する。
    manifest に記録した SHA-256 と実ファイルが食い違っていれば、
    notebook を変更したのに再生成していないことになる。
    """
    failures: list[str] = []
    notebooks = discover()
    expected_sources = {notebook.relative.as_posix() for notebook in notebooks}
    expected_pages = {notebook.relative.with_suffix(".html") for notebook in notebooks}
    published = {
        path.relative_to(SITE) for path in SITE.rglob("*.html") if path != SITE / "index.html"
    }
    if expected_pages != published:
        failures.append(f"missing HTML: {sorted(expected_pages - published)}")
        failures.append(f"unexpected HTML: {sorted(published - expected_pages)}")

    manifest_path = SITE / "notebooks-manifest.json"
    if not manifest_path.exists():
        failures.append("missing site/notebooks-manifest.json")
        return failures
    entries = json.loads(manifest_path.read_text(encoding="utf-8")).get("notebooks", {})
    if set(entries) != expected_sources:
        failures.append(
            f"manifest source mismatch: expected {len(expected_sources)}, found {len(entries)}"
        )
    for source_name, entry in entries.items():
        source_path = ROOT / "notebooks" / source_name
        html_path = SITE / entry["html"]
        if not source_path.exists() or not html_path.exists():
            continue
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != entry.get("source_sha256"):
            failures.append(f"stale HTML source mapping: {source_name}")
        if hashlib.sha256(html_path.read_bytes()).hexdigest() != entry.get("html_sha256"):
            failures.append(f"modified generated HTML: {entry['html']}")
    return failures


def main() -> int:
    index = SITE / "index.html"
    if not index.exists():
        print("site/index.html does not exist", file=sys.stderr)
        return 1
    failures: list[str] = []
    notebook_pages = sorted(path for path in SITE.rglob("*.html") if path != index)
    for document in [index, *notebook_pages]:
        source = document.read_text(encoding="utf-8", errors="replace")
        if ERROR_RE.search(source):
            failures.append(f"{document.relative_to(ROOT)} contains an exception marker")
        parser = LinkParser()
        parser.feed(source)
        for link in parser.links:
            target = local_target(document, link)
            if target is not None and not target.exists():
                failures.append(
                    f"{document.relative_to(ROOT)} has missing local target {link!r}"
                )
    index_parser = LinkParser()
    index_parser.feed(index.read_text(encoding="utf-8"))
    indexed_pages = {
        local_target(index, link)
        for link in index_parser.links
        if urlsplit(link).path.endswith(".html")
    }
    expected_pages = {page.resolve() for page in notebook_pages}
    if indexed_pages != expected_pages:
        failures.append(f"index missing pages: {sorted(expected_pages - indexed_pages)}")
        failures.append(f"index has extra pages: {sorted(indexed_pages - expected_pages)}")
    failures.extend(check_manifest())
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Site is complete: {len(notebook_pages)} notebook pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
