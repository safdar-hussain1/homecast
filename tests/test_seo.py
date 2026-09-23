"""The search and link-preview metadata must survive into the published page.

`scripts/dashboard_template.html` carries the <head> block (title, description,
canonical, Open Graph and Twitter tags, JSON-LD) and `scripts/build_dashboard.py`
bakes it into `docs/index.html`. These tests read the page GitHub Pages
actually serves, plus the two files that sit beside it (`og-image.png`,
`sitemap.xml`), so a template edit that drops or breaks any of it fails here
instead of silently on the live site.
"""
from __future__ import annotations

import importlib.util
import json
import re
import struct
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PAGE = DOCS / "index.html"
TEMPLATE = ROOT / "scripts" / "dashboard_template.html"
MODEL_JSON = ROOT / "models" / "gurgaon" / "model.json"

URL = "https://safdar-hussain1.github.io/homecast/"
REPO = "https://github.com/safdar-hussain1/homecast"
TOKEN = "0SIEfExLTSQj1qvnHWF5A5fY58KVl2lpIEnePP9CtI0"
AUTHOR = "Safdar Hussain"


class _Head(HTMLParser):
    """Collects <head> metadata, the JSON-LD blocks and the <h1> count."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.lang = None
        self.title = None
        self.meta: dict[str, list[str]] = {}
        self.links: list[dict] = []
        self.ld: list[str] = []
        self.h1 = 0
        self._in_head = self._in_title = self._in_ld = False
        self._buf = ""

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "html":
            self.lang = a.get("lang")
        elif tag == "head":
            self._in_head = True
        elif tag == "title" and self._in_head and self.title is None:
            self._in_title, self._buf = True, ""
        elif tag == "meta" and (a.get("name") or a.get("property")):
            key = (a.get("name") or a.get("property")).lower()
            self.meta.setdefault(key, []).append((a.get("content") or "").strip())
        elif tag == "link":
            self.links.append(a)
        elif tag == "script" and a.get("type") == "application/ld+json":
            self._in_ld, self._buf = True, ""
        elif tag == "h1":
            self.h1 += 1

    def handle_endtag(self, tag):
        if tag == "head":
            self._in_head = False
        elif tag == "title" and self._in_title:
            self._in_title, self.title = False, " ".join(self._buf.split())
        elif tag == "script" and self._in_ld:
            self._in_ld = False
            self.ld.append(self._buf)

    def handle_data(self, data):
        if self._in_title or self._in_ld:
            self._buf += data

    def one(self, key: str) -> str:
        values = self.meta.get(key, [])
        assert len(values) == 1, f"expected exactly one {key!r} tag, found {len(values)}"
        return values[0]


@pytest.fixture(scope="module")
def head() -> _Head:
    parser = _Head()
    parser.feed(PAGE.read_text(encoding="utf-8"))
    return parser


def test_committed_page_is_what_the_template_builds(tmp_path):
    """The metadata lives in the template, not hand-edited into docs/."""
    spec = importlib.util.spec_from_file_location("build_dashboard", ROOT / "scripts" / "build_dashboard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fresh = module.build("gurgaon", output=tmp_path / "index.html")
    assert fresh.read_bytes() == PAGE.read_bytes(), (
        "docs/index.html differs from a fresh build -- run scripts/build_dashboard.py")


def test_language_title_and_description(head):
    assert head.lang == "en"
    assert head.title and len(head.title) <= 60, head.title
    description = head.one("description")
    assert 120 <= len(description) <= 160, f"{len(description)} characters"
    assert head.one("author") == AUTHOR
    assert head.one("google-site-verification") == TOKEN
    assert "viewport" in head.meta


def test_canonical_and_favicon(head):
    canonical = [l.get("href") for l in head.links if (l.get("rel") or "").lower() == "canonical"]
    assert canonical == [URL]
    icons = [l.get("href") or "" for l in head.links
             if "icon" in (l.get("rel") or "").lower().split()]
    assert icons, "no <link rel=icon>"
    assert icons[0].startswith("data:image/svg+xml") or (DOCS / icons[0]).is_file()


def test_open_graph_and_twitter_cards(head):
    title, description = head.title, head.one("description")
    assert head.one("og:type") == "website"
    assert head.one("og:site_name") == AUTHOR
    assert head.one("og:url") == URL
    assert head.one("og:title") == head.one("twitter:title") == title
    assert head.one("og:description") == head.one("twitter:description") == description
    assert head.one("og:image") == head.one("twitter:image") == URL + "og-image.png"
    assert head.one("og:image:width") == "1200"
    assert head.one("og:image:height") == "630"
    assert head.one("og:image:alt") and head.one("og:image:alt") == head.one("twitter:image:alt")
    assert head.one("twitter:card") == "summary_large_image"


def test_json_ld_graph_names_the_app_the_code_and_the_author(head):
    assert len(head.ld) == 1, "expected one JSON-LD block"
    graph = {node["@type"]: node for node in json.loads(head.ld[0])["@graph"]}
    app, code = graph["WebApplication"], graph["SoftwareSourceCode"]
    assert app["url"] == URL
    assert app["image"] == URL + "og-image.png"
    assert app["description"] == head.one("description")
    assert app["applicationCategory"] == "BusinessApplication"
    assert code["codeRepository"] == REPO
    assert "Python" in code["programmingLanguage"]
    for node in (app, code):
        assert node["author"]["@type"] == "Person"
        assert node["author"]["name"] == AUTHOR
        assert "https://github.com/safdar-hussain1" in node["author"]["sameAs"]


def test_exactly_one_h1(head):
    assert head.h1 == 1


def _js_string(expression: str, city: str) -> str:
    """Evaluate a JS concatenation of '...' literals and DATA.city."""
    parts = re.findall(r"'((?:[^'\\]|\\.)*)'|(DATA\.city)", expression)
    return "".join(city if is_city else literal for literal, is_city in parts)


def test_runtime_title_and_description_match_the_head(head):
    """The page rewrites both per city once its script runs, and crawlers that
    run scripts index the rewritten values -- so for the public city they must
    be the same strings as the static <head>."""
    template = TEMPLATE.read_text(encoding="utf-8")
    title_js = re.search(r"if \(!SELFTEST_ACTIVE\) document\.title = (.*?);\n", template)
    desc_js = re.search(r"metaDesc\.setAttribute\('content',(.*?)\);", template, re.S)
    assert title_js and desc_js, "runtime title/description rewrite moved -- update this test"
    city = json.loads(MODEL_JSON.read_text(encoding="utf-8"))["city"]
    assert _js_string(title_js.group(1), city) == head.title
    assert _js_string(desc_js.group(1), city) == head.one("description")


def test_og_image_is_a_1200x630_png_under_500kb():
    data = (DOCS / "og-image.png").read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", data[16:24]) == (1200, 630)
    assert len(data) < 500 * 1024


def test_sitemap_lists_the_page_with_a_fixed_lastmod():
    sitemap = (DOCS / "sitemap.xml").read_text(encoding="utf-8")
    assert f"<loc>{URL}</loc>" in sitemap
    assert re.search(r"<lastmod>\d{4}-\d{2}-\d{2}</lastmod>", sitemap)
