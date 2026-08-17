"""Render a :class:`PipelineTrace` as a self-contained, zoomable HTML timeline.

The PNG answers "where did the time go" at a glance and cannot be zoomed; a trace
viewer zooms but organises rows by instruction. This is the union: **rows are
hardware resources**, and the time axis zooms and pans.

One file, no server, no download, no CDN — open it with ``file://``. Everything
is inlined: the data as JSON, the drawing as a few hundred lines of vanilla JS
against an SVG. That is a deliberate constraint. A figure that needs a fetched
viewer, a port and a background process is a figure people stop looking at.

Interaction, all of it discoverable from the legend in the page:

- **wheel** zooms about the cursor, so the interesting region stays put
- **drag** pans
- **double-click** resets
- **hover** gives the span's own numbers — bytes and achieved bandwidth for a
  load, operations and achieved rate for a compute step

Zoom is x-only: the y axis is a list of resources, not a scale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{
    --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --ink-3: #8a8983;
    --grid: #e6e5e1; --box: #f2f1ed;
    --dram: #2a78d6; --sram: #8a8983; --core: #eb6834; --idle: #e6e5e1;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 28px 32px 40px; background: var(--surface); color: var(--ink);
    font: 14px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }}
  h1 {{ font-size: 21px; margin: 0 0 6px; }}
  .sub {{ color: var(--ink-2); margin: 0 0 20px; font-size: 13px; }}
  .boxes {{ display: flex; gap: 14px; margin-bottom: 26px; flex-wrap: wrap; }}
  .box {{
    flex: 1 1 240px; border: 1.5px solid var(--grid); border-radius: 7px;
    background: var(--box); padding: 12px 14px;
  }}
  .box h2 {{ font-size: 10.5px; letter-spacing: .04em; margin: 0 0 6px; }}
  .box .big {{ font-size: 26px; font-weight: 700; line-height: 1.1; }}
  .box .det {{ font-size: 11.5px; color: var(--ink-2); margin-top: 6px; white-space: pre-line; }}
  .box.dram {{ border-color: var(--dram); }} .box.dram h2 {{ color: var(--dram); }}
  .box.sram {{ border-color: var(--sram); }} .box.sram h2 {{ color: var(--sram); }}
  .box.core {{ border-color: var(--core); }} .box.core h2 {{ color: var(--core); }}
  .hint {{ font-size: 11.5px; color: var(--ink-3); margin: 0 0 10px; }}
  #wrap {{ position: relative; border-top: 1px solid var(--grid); padding-top: 12px; }}
  svg {{ display: block; width: 100%; touch-action: none; cursor: grab; }}
  svg.drag {{ cursor: grabbing; }}
  .rowline {{ fill: none; stroke: var(--grid); stroke-width: 1; }}
  .rowname {{ font-size: 12.5px; font-weight: 600; fill: var(--ink); }}
  .rowdetail {{ font-size: 9.5px; fill: var(--ink-3); }}
  .rowqty {{ font-size: 11px; fill: var(--ink); }}
  .rowqty.off {{ fill: var(--ink-3); }}
  .tick {{ font-size: 10.5px; fill: var(--ink-2); }}
  .tickline {{ stroke: var(--grid); stroke-width: 1; }}
  #tip {{
    position: absolute; pointer-events: none; opacity: 0; transition: opacity .08s;
    background: #17171a; color: #fff; padding: 7px 10px; border-radius: 5px;
    font-size: 11.5px; line-height: 1.45; white-space: pre; z-index: 5;
  }}
  footer {{ margin-top: 22px; font-size: 11px; color: var(--ink-3); }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; }}
</style>

<h1>{title}</h1>
<p class="sub">{subtitle}</p>
<div class="boxes">{boxes}</div>
<p class="hint">Rows are hardware resources, not steps. <b>Wheel</b> zooms about the cursor ·
<b>drag</b> pans · <b>double-click</b> resets · <b>hover</b> a bar for its own numbers.
Filled bars on the DRAM row are loads, hollow ones are results written back.
Grey rows are declared by the chip and unused by this model.</p>
<div id="wrap"><svg id="chart"></svg><div id="tip"></div></div>
<footer>{footer}</footer>

<script>
const DATA = {data};

const LEFT = 210, RIGHT = 190, ROW = 42, TOP = 26, BOT = 30;
const COLOUR = {{dram: "#2a78d6", sram: "#8a8983", core: "#eb6834"}};
const svg = document.getElementById("chart");
const tip = document.getElementById("tip");
const wrap = document.getElementById("wrap");
const NS = "http://www.w3.org/2000/svg";

let view = {{lo: 0, hi: DATA.total}};   // visible time window, seconds
let width = 0;

function fmtTime(s) {{
  if (s === 0) return "0 s";
  const u = [[1, "s"], [1e-3, "ms"], [1e-6, "\\u00b5s"], [1e-9, "ns"], [1e-12, "ps"]];
  for (const [scale, name] of u) if (Math.abs(s) >= scale) return trim(s / scale) + " " + name;
  return trim(s / 1e-12) + " ps";
}}
function trim(v) {{
  return (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));
}}
const el = (name, attrs, text) => {{
  const node = document.createElementNS(NS, name);
  for (const k in attrs) node.setAttribute(k, attrs[k]);
  if (text !== undefined) node.textContent = text;
  return node;
}};

function x(t) {{ return LEFT + (t - view.lo) / (view.hi - view.lo) * (width - LEFT - RIGHT); }}
function tAt(px) {{ return view.lo + (px - LEFT) / (width - LEFT - RIGHT) * (view.hi - view.lo); }}
// offsetX is relative to the event *target*, which is a bar as soon as the
// cursor is over one — so zoom would anchor on the wrong instant exactly when
// you are aiming at something. Measure against the svg instead.
function localX(e) {{ return e.clientX - svg.getBoundingClientRect().left; }}

function ticks() {{
  const span = view.hi - view.lo, target = 6;
  const raw = span / target, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) || 10 * mag;
  const out = [];
  for (let t = Math.ceil(view.lo / step) * step; t <= view.hi; t += step) out.push(t);
  return out;
}}

function draw() {{
  width = svg.clientWidth || svg.parentNode.clientWidth || 1200;
  const height = TOP + DATA.rows.length * ROW + BOT;
  svg.setAttribute("height", height);
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const plotW = width - LEFT - RIGHT;

  for (const t of ticks()) {{
    const px = x(t);
    if (px < LEFT - 1 || px > width - RIGHT + 1) continue;
    const bottom = TOP + DATA.rows.length * ROW;
    svg.appendChild(el("line", {{class: "tickline", x1: px, x2: px, y1: TOP, y2: bottom}}));
    svg.appendChild(el("text",
      {{class: "tick", x: px, y: height - 10, "text-anchor": "middle"}}, fmtTime(t)));
  }}

  DATA.rows.forEach((row, i) => {{
    const y0 = TOP + i * ROW;
    const end = {{x: LEFT - 12, "text-anchor": "end"}};
    svg.appendChild(el("line",
      {{class: "rowline", x1: LEFT, x2: width - RIGHT, y1: y0 + ROW, y2: y0 + ROW}}));
    svg.appendChild(el("text", {{...end, class: "rowname", y: y0 + 17}}, row.title));
    svg.appendChild(el("text", {{...end, class: "rowdetail", y: y0 + 29}}, row.detail));
    svg.appendChild(el("text", {{...end, class: "rowdetail", y: y0 + 39}}, row.note));
    svg.appendChild(el("text", {{
      class: "rowqty" + (row.lane ? "" : " off"), x: width - RIGHT + 12, y: y0 + ROW / 2 + 4,
    }}, row.quantity));

    if (!row.lane) {{
      svg.appendChild(el("rect",
        {{x: LEFT, y: y0 + ROW / 2 - 5, width: plotW, height: 10, fill: "var(--idle)"}}));
      return;
    }}
    for (const s of DATA.spans) {{
      if (s.lane !== row.lane) continue;
      if (s.end < view.lo || s.start > view.hi) continue;
      const x0 = Math.max(x(s.start), LEFT), x1 = Math.min(x(s.end), width - RIGHT);
      // Stores hollow, loads filled: the direction of DRAM traffic should be
      // readable without a legend, and they never overlap because it is one port.
      const store = s.store;
      const rect = el("rect", {{
        x: x0, y: y0 + 7, width: Math.max(x1 - x0, 1.2), height: ROW - 16,
        fill: store ? "var(--surface)" : COLOUR[row.lane],
        stroke: store ? COLOUR[row.lane] : "var(--surface)",
        "stroke-width": store ? 1.2 : 0.7,
      }});
      rect.dataset.tip = s.tip;
      svg.appendChild(rect);
    }}
  }});
}}

svg.addEventListener("wheel", e => {{
  e.preventDefault();
  const anchor = tAt(localX(e));
  const factor = Math.exp(e.deltaY * 0.0015);
  let lo = anchor - (anchor - view.lo) * factor, hi = anchor + (view.hi - anchor) * factor;
  // A floor on the window keeps float precision sane at extreme zoom; the
  // ceiling is the whole run, since there is nothing outside it to look at.
  if (hi - lo < DATA.total * 1e-6) return;
  view = {{lo: Math.max(lo, -DATA.total * 0.02), hi: Math.min(hi, DATA.total * 1.02)}};
  draw();
}}, {{passive: false}});

let dragging = null;
svg.addEventListener("pointerdown", e => {{
  dragging = {{px: localX(e), lo: view.lo, hi: view.hi}};
  svg.classList.add("drag"); svg.setPointerCapture(e.pointerId);
}});
svg.addEventListener("pointerup", e => {{
  dragging = null; svg.classList.remove("drag"); svg.releasePointerCapture(e.pointerId);
}});
svg.addEventListener("pointermove", e => {{
  if (dragging) {{
    const moved = localX(e) - dragging.px;
    const shift = moved / (width - LEFT - RIGHT) * (dragging.hi - dragging.lo);
    view = {{lo: dragging.lo - shift, hi: dragging.hi - shift}};
    draw();
    return;
  }}
  const target = e.target.dataset && e.target.dataset.tip;
  if (target) {{
    tip.textContent = target;
    tip.style.opacity = 1;
    const box = wrap.getBoundingClientRect();
    tip.style.left = Math.min(e.clientX - box.left + 14, box.width - 260) + "px";
    tip.style.top = (e.clientY - box.top + 14) + "px";
  }} else {{
    tip.style.opacity = 0;
  }}
}});
svg.addEventListener("dblclick", () => {{ view = {{lo: 0, hi: DATA.total}}; draw(); }});
addEventListener("resize", draw);
draw();
</script>
"""


@dataclass(frozen=True)
class Box:
    """One headline figure above the chart."""

    lane: str
    heading: str
    headline: str
    detail: str


def render(
    *,
    title: str,
    subtitle: str,
    footer: str,
    boxes: list[Box],
    rows: list[dict[str, str | None]],
    spans: list[dict[str, object]],
    total_s: float,
) -> str:
    """Build the page. Pure: returns text, writes nothing."""
    boxes_html = "".join(
        f'<div class="box {b.lane}"><h2>{b.heading}</h2>'
        f'<div class="big">{b.headline}</div>'
        f'<div class="det">{b.detail}</div></div>'
        for b in boxes
    )
    data = json.dumps({"rows": rows, "spans": spans, "total": total_s})
    return TEMPLATE.format(
        title=title, subtitle=subtitle, footer=footer, boxes=boxes_html, data=data
    )
