#!/usr/bin/env python3
"""Build the local all-cities hub.

One page that switches between every city HomeCast can price: the public
Gurgaon dashboard, each private city's dashboard, and the Amaravathi rate
reference. Each city keeps its own already-built page; the hub just frames
them, so there is exactly one copy of every payload and nothing can drift.

    python scripts/build_hub.py

Local only. The output lands in `private/dashboards/index.html` and the script
refuses to write under `docs/` — the hub links to private city pages, so
publishing it would leak the private tier.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASH_DIR = ROOT / "private" / "dashboards"
OUT = DASH_DIR / "index.html"
DOCS_DIR = ROOT / "docs"
PUBLIC_DASH = ROOT / "docs" / "index.html"


def _discover() -> list[dict]:
    """Every city page that exists on disk, with how it prices."""
    sys.path.insert(0, str(ROOT / "src"))
    from homecast.cities import CITIES

    found: list[dict] = []
    for key, city in CITIES.items():
        page = PUBLIC_DASH if city.public else DASH_DIR / f"{key}.html"
        if not page.exists():
            continue
        served, note = "model", ""
        metrics = city.models_dir / "metrics.json"
        if metrics.exists():
            m = json.loads(metrics.read_text())
            # `serving_status` is authoritative (homecast.model.serving_status);
            # `baseline_served` is also True for not-served cities, so checking
            # it first would mislabel them.
            status = m.get("serving_status", "model")
            if status == "not_served":
                served, note = "none", "no price — below the quality floor"
            elif status == "baseline":
                served, note = "baseline", "priced by the ₹/sq.ft. rule"
            else:
                mp = m.get("model", {}).get("mape_pct")
                note = f"model · {mp:.1f}% MAPE" if mp is not None else "model"
        found.append({
            "key": key, "label": city.display, "href": _rel(page),
            "public": city.public, "served": served, "note": note,
        })

    ref = DASH_DIR / "amaravathi_ap.html"
    if ref.exists():
        found.append({
            "key": "amaravathi_ap", "label": "Amaravathi, AP", "href": ref.name,
            "public": False, "served": "rates",
            "note": "published rates — no model",
        })
    return found


def _rel(page: Path) -> str:
    """Path from the hub (in private/dashboards/) to a city page."""
    try:
        return page.relative_to(DASH_DIR).as_posix()
    except ValueError:
        # the public dashboard lives outside private/dashboards/
        import os
        return os.path.relpath(page, DASH_DIR).replace("\\", "/")


def _render(cities: list[dict]) -> str:
    tabs = "\n".join(
        f'      <button class="tab" data-src="{html.escape(c["href"])}" '
        f'data-served="{c["served"]}" type="button">'
        f'<span class="nm">{html.escape(c["label"])}</span>'
        f'<span class="nt">{html.escape(c["note"])}</span></button>'
        for c in cities)
    payload = json.dumps(cities, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<!-- HomeCast — designed and built by Safdar Hussain · github.com/safdar-hussain1/homecast -->
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HomeCast — all cities (local)</title>
<meta name="author" content="Safdar Hussain">
<meta name="color-scheme" content="light dark">
<meta name="robots" content="noindex">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,100..900&family=IBM+Plex+Mono:wght@400;500;600&family=Instrument+Sans:wdth,wght@75..100,400..700&display=swap" rel="stylesheet">
<style>
:root{{
  color-scheme:light dark;
  --f-display:"Archivo",ui-sans-serif,system-ui,sans-serif;
  --f-body:"Instrument Sans",ui-sans-serif,system-ui,sans-serif;
  --f-mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
  --plane:#e7ebf2; --surface:#fafbfd; --ink:#0c1016; --ink-2:#465061; --ink-3:#727e8d;
  --line:#d0d8e3; --accent:#1747b8; --model:#0d7a34; --baseline:#9a5407; --none:#8f1d1d;
}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
  --plane:#080b11; --surface:#121820; --ink:#eaf0f7; --ink-2:#9fadbd; --ink-3:#727f8f;
  --line:#28323e; --accent:#8ab3f7; --model:#42c163; --baseline:#e0a35c; --none:#f08a8a;
}}}}
:root[data-theme="dark"]{{
  --plane:#080b11; --surface:#121820; --ink:#eaf0f7; --ink-2:#9fadbd; --ink-3:#727f8f;
  --line:#28323e; --accent:#8ab3f7; --model:#42c163; --baseline:#e0a35c; --none:#f08a8a;
}}
*{{box-sizing:border-box}} html,body{{height:100%}}
body{{margin:0;background:var(--plane);color:var(--ink);font-family:var(--f-body);
  display:flex;flex-direction:column;overflow:hidden}}
header{{background:var(--surface);border-bottom:1px solid var(--line);flex:none}}
.bar{{display:flex;align-items:center;gap:14px;padding:9px clamp(12px,2vw,20px);flex-wrap:wrap}}
.brand{{font-family:var(--f-display);font-variation-settings:"wdth" 112,"wght" 780;
  font-size:16px;letter-spacing:-.02em;white-space:nowrap}}
.brand span{{font-family:var(--f-mono);font-variation-settings:normal;font-size:11px;
  letter-spacing:.13em;color:var(--ink-3);margin-left:8px}}
.tabs{{display:flex;gap:6px;flex-wrap:wrap}}
.tab{{font:inherit;cursor:pointer;background:transparent;color:var(--ink-2);
  border:1px solid var(--line);border-radius:9px;padding:5px 11px;text-align:left;
  display:flex;flex-direction:column;gap:1px;line-height:1.25}}
.tab .nm{{font-size:13.5px;font-weight:600;color:var(--ink)}}
.tab .nt{{font-family:var(--f-mono);font-size:10.5px;color:var(--ink-3);white-space:nowrap}}
.tab:hover{{border-color:var(--accent)}}
.tab[aria-selected="true"]{{border-color:var(--accent);background:var(--plane);
  box-shadow:inset 0 0 0 1px var(--accent)}}
.tab[data-served="model"] .nt{{color:var(--model)}}
.tab[data-served="baseline"] .nt{{color:var(--baseline)}}
.tab[data-served="none"] .nt{{color:var(--none)}}
.tab[data-served="rates"] .nt{{color:var(--baseline)}}
.right{{margin-left:auto;display:flex;gap:9px;align-items:center}}
.right a,.right button{{font-family:var(--f-mono);font-size:12px;color:var(--ink-2);
  text-decoration:none;border:1px solid var(--line);border-radius:999px;padding:5px 11px;
  background:transparent;cursor:pointer}}
.right a:hover,.right button:hover{{border-color:var(--accent);color:var(--accent)}}
.note{{font-family:var(--f-mono);font-size:11px;color:var(--ink-3);
  padding:0 clamp(12px,2vw,20px) 8px}}
iframe{{flex:1;width:100%;border:0;background:var(--plane)}}
</style>
</head>
<body>
<header>
  <div class="bar">
    <div class="brand">HomeCast<span>ALL CITIES · LOCAL</span></div>
    <div class="tabs" role="tablist">
{tabs}
    </div>
    <div class="right">
      <button id="theme" type="button">Theme</button>
      <a id="open" href="#" target="_blank" rel="noopener">Open ↗</a>
    </div>
  </div>
  <p class="note">Local hub. Private cities never leave this machine — this page is gitignored and is not published.</p>
</header>
<iframe id="frame" title="City dashboard"></iframe>
<script>
const CITIES = {payload};
const frame = document.getElementById('frame');
const tabs = [...document.querySelectorAll('.tab')];

function show(i){{
  tabs.forEach((t,j) => t.setAttribute('aria-selected', String(i === j)));
  const src = tabs[i].dataset.src;
  frame.src = src;
  document.getElementById('open').href = src;
  try {{ localStorage.setItem('hc-hub-city', String(i)); }} catch(e) {{}}
}}
tabs.forEach((t,i) => t.addEventListener('click', () => show(i)));

document.getElementById('theme').addEventListener('click', () => {{
  const cur = document.documentElement.getAttribute('data-theme')
    || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  const next = cur === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  // city pages read the same key, so the choice carries into the frame
  try {{ localStorage.setItem('hc-theme', next); }} catch(e) {{}}
  frame.contentWindow.location.reload();
}});

let start = 0;
try {{ const s = Number(localStorage.getItem('hc-hub-city')); if (s >= 0 && s < tabs.length) start = s; }} catch(e) {{}}
show(start);
console.log('HomeCast all-cities hub — built by Safdar Hussain');
</script>
</body>
</html>
"""


def build(output: Path = OUT) -> Path:
    output = output.resolve()
    if output == DOCS_DIR or DOCS_DIR in output.parents:
        raise ValueError(
            f"refusing to write the hub into {DOCS_DIR} — it links to private "
            f"city dashboards and must stay local")
    cities = _discover()
    if not cities:
        raise ValueError(
            "no city dashboards found — build them first, e.g. "
            "`python scripts/build_dashboard.py --city gurgaon` and "
            "`python scripts/build_reference_dashboard.py`")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render(cities), encoding="utf-8")
    return output


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build the local all-cities hub")
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args(argv)
    try:
        out = build(Path(a.out))
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
