"""The published page is one self-contained file that says who made it.

`scripts/build_dashboard.py` inlines the two typefaces from `scripts/fonts/`
into the page, so it loads nothing from any other host; the footer's promise
that nothing typed leaves the page depends on that, so it is checked here on
the page GitHub Pages actually serves.
"""
from __future__ import annotations

import importlib.util
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "index.html"
TEMPLATE = ROOT / "scripts" / "dashboard_template.html"
FONTS_DIR = ROOT / "scripts" / "fonts"

_spec = importlib.util.spec_from_file_location(
    "build_dashboard", ROOT / "scripts" / "build_dashboard.py")
build_dashboard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_dashboard)


@pytest.fixture(scope="module")
def page() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_page_embeds_both_typefaces(page):
    assert page.count("@font-face") == len(build_dashboard.FONT_FACES) == 2
    assert page.count("url(data:font/woff2;base64,") == 2
    assert build_dashboard.FONTS_PLACEHOLDER not in page


class _Resources(HTMLParser):
    """Every URL the page would load by itself: links a reader clicks (<a>)
    and the canonical address are not loads, everything else is."""

    LOADING_ATTRS = ("src", "href", "srcset", "data", "poster", "action", "formaction", "background")

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.loads: list[tuple[str, str, str]] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "a" or (tag == "link" and (a.get("rel") or "").lower() == "canonical"):
            return
        for name in self.LOADING_ATTRS:
            if a.get(name):
                self.loads.append((tag, name, a[name]))


def test_page_fetches_nothing_from_anywhere_else(page):
    parser = _Resources()
    parser.feed(page)
    outside = [(tag, attr, value[:60]) for tag, attr, value in parser.loads
               if not value.startswith(("data:", "#"))]
    assert outside == [], f"the page would load: {outside}"
    for m in re.finditer(r"url\(\s*['\"]?(?!data:|#)", page):
        raise AssertionError("CSS loads a URL: " + page[m.start():m.start() + 60])
    assert "@import" not in page
    for call in ("fetch(", "XMLHttpRequest", "sendBeacon", "WebSocket", "EventSource",
                 "Worker(", "importScripts", "import(", "new Image", ".src=", ".src =",
                 "serviceWorker"):
        assert call not in page, call


def test_font_files_ship_with_their_licences():
    for _family, filename, _weight in build_dashboard.FONT_FACES:
        assert (FONTS_DIR / filename).is_file(), filename
    for licence in ("OFL-eczar.txt", "OFL-anek-latin.txt"):
        text = (FONTS_DIR / licence).read_text(encoding="utf-8")
        assert "SIL Open Font License" in text


def test_build_refuses_a_template_without_its_fonts_placeholder(tmp_path):
    template = tmp_path / "template.html"
    template.write_text(TEMPLATE.read_text(encoding="utf-8").replace(
        build_dashboard.FONTS_PLACEHOLDER, ""), encoding="utf-8")
    with pytest.raises(ValueError, match="__FONTS__"):
        build_dashboard.build("gurgaon", template=template, output=tmp_path / "out.html")


def test_page_carries_its_authorship_marks(page):
    assert '<meta name="author" content="Safdar Hussain">' in page
    assert re.search(r"<!--[^>]*Safdar Hussain[^>]*-->", page)
    assert re.search(r"console\.log\('[^']*Safdar Hussain", page)
    footer = re.search(r"<footer\b.*?</footer>", page, re.S)
    assert footer and "Safdar Hussain" in footer.group(0)
