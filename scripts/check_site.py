#!/usr/bin/env python3
"""Validate the generated index and published notebook HTML files."""

from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

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
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Site is complete: {len(notebook_pages)} notebook pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
