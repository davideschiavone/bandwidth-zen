"""Render a matmul's or the ad-hoc encoder's schedule as a self-contained,
playable resource-flow animation.

The timeline (``timeline_html.py``) answers "where did the time go, on which
resource" as a static, zoomable strip, with one row per resource a chip profile
declares — grey for what v1 doesn't cost (D20). This answers a different
question — "what does that schedule actually look like happening" — by playing
the same trace (``bwz.analysis.pipeline.PipelineTrace``) back as motion, using
the *same* per-resource station list ``rows_for`` gives the timeline (D43), not
a bespoke three-station shape: a chip with more declared memory levels or a
second compute engine gets more stations, sized from each event's own bytes but
not pixel-accurate to them (docs/CORRECTIONS.md D40).

One file, no server, no download, no CDN — open it with ``file://``, same
constraint as ``timeline_html.py``, same visual vocabulary (the ``--dram``/
``--sram``/``--core``/``--vector`` colours, the streaming-A hatch) copied
verbatim rather than imported, because each self-contained page has to stand
completely alone.

Matmul or the ad-hoc encoder, not a full model: a station diagram assumes one
tile- or operation-shaped stream of events, which a whole model's per-operation
trace can run to hundreds of and this isn't built to usefully show (D42).
"""

from __future__ import annotations

import json

TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{
    --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --ink-3: #8a8983;
    --grid: #e6e5e1; --box: #f2f1ed;
    --dram: #2a78d6; --sram: #8a8983; --core: #eb6834; --vector: #1baf7a;
    --idle: #e6e5e1;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 28px 32px 40px; background: var(--surface); color: var(--ink);
    font: 14px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }}
  h1 {{ font-size: 21px; margin: 0 0 6px; }}
  .sub {{ color: var(--ink-2); margin: 0 0 20px; font-size: 13px; }}
  .hint {{ font-size: 11.5px; color: var(--ink-3); margin: 0 0 14px; }}
  .banner {{
    border: 1.5px solid var(--grid); border-radius: 7px; background: var(--box);
    padding: 10px 14px; margin-bottom: 16px; font-size: 12.5px; color: var(--ink-2);
  }}
  .banner b {{ color: var(--ink); }}
  .banner ul {{ margin: 6px 0 0; padding-left: 18px; }}
  #controls {{
    display: flex; align-items: center; gap: 10px; margin: 4px 0 14px; flex-wrap: wrap;
  }}
  button {{
    font: inherit; font-size: 12.5px; padding: 6px 12px; border-radius: 6px;
    border: 1.5px solid var(--grid); background: var(--box); color: var(--ink); cursor: pointer;
  }}
  button:hover {{ border-color: var(--ink-3); }}
  select {{ font: inherit; font-size: 12.5px; padding: 5px 8px; border-radius: 6px;
            border: 1.5px solid var(--grid); background: var(--box); color: var(--ink); }}
  input[type=range] {{ flex: 1 1 260px; min-width: 160px; }}
  #clockLabel {{ font-size: 12px; color: var(--ink-2); min-width: 150px; text-align: right; }}
  #debugrow {{ display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; }}
  #wrap {{
    flex: 3 1 440px; position: relative; border-top: 1px solid var(--grid); padding-top: 12px;
  }}
  svg {{ display: block; width: 100%; }}
  #codepane {{
    flex: 2 1 340px; margin: 0; padding: 10px 0; max-height: 440px; overflow-y: auto;
    background: var(--box); border: 1.5px solid var(--grid); border-radius: 7px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px;
    line-height: 1.55;
  }}
  .codeline {{
    white-space: pre; padding: 0 12px; border-left: 3px solid transparent; color: var(--ink-2);
  }}
  .codeline.hot-dram {{
    background: rgba(42, 120, 214, 0.14); border-left-color: var(--dram); color: var(--ink);
    font-weight: 600;
  }}
  .codeline.hot-core {{
    background: rgba(235, 104, 52, 0.14); border-left-color: var(--core); color: var(--ink);
    font-weight: 600;
  }}
  .codeline.hot-vector {{
    background: rgba(27, 175, 122, 0.14); border-left-color: var(--vector); color: var(--ink);
    font-weight: 600;
  }}
  .station {{ fill: var(--box); stroke: var(--grid); stroke-width: 1.5; }}
  .station-label {{ font-size: 13px; font-weight: 700; fill: var(--ink); }}
  .station-detail {{ font-size: 10.5px; fill: var(--ink-3); }}
  .lane-track {{ stroke: var(--grid); stroke-width: 1; stroke-dasharray: 3 4; }}
  .glow {{ fill: none; stroke: var(--core); stroke-width: 3; opacity: 0; }}
  #georow {{ margin-top: 18px; border-top: 1px solid var(--grid); padding-top: 12px; }}
  #geometry {{ display: block; width: 100%; max-width: 480px; }}
  .geo-rect {{ fill: var(--box); stroke: var(--grid); stroke-width: 1.5; }}
  .geo-grid {{ stroke: var(--grid); stroke-width: 1; }}
  .geo-label {{ font-size: 11px; fill: var(--ink-2); }}
  .geo-highlight {{ fill-opacity: 0.55; }}
  .geo-boundary {{ stroke: var(--ink-3); stroke-width: 0.6; }}
  #geocaption {{
    background: #17171a; color: #fff; border-radius: 7px; padding: 10px 14px;
    margin-top: 10px; font-size: 12px; line-height: 1.55; max-width: 480px;
  }}
  #geocaption .geo-static {{ color: #c9c8c3; }}
  #geocaption .geo-active {{ margin-top: 4px; color: #fff; }}
  #tip {{
    position: absolute; pointer-events: none; opacity: 0; transition: opacity .08s;
    background: #17171a; color: #fff; padding: 7px 10px; border-radius: 5px;
    font-size: 11.5px; line-height: 1.45; white-space: pre; z-index: 5;
  }}
  #annotation {{
    border: 1.5px solid var(--grid); border-radius: 7px; background: var(--box);
    padding: 12px 14px; margin-top: 14px; font-size: 12.5px; min-height: 64px;
  }}
  #annotation .now {{ color: var(--ink-2); font-size: 11px; margin-bottom: 6px; }}
  #annotation .active {{ margin: 2px 0; white-space: pre-line; }}
  #annotation .empty {{ color: var(--ink-3); }}
  footer {{ margin-top: 22px; font-size: 11px; color: var(--ink-3); }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; }}
</style>

<h1>{title}</h1>
<p class="sub">{subtitle}</p>
<div class="banner">
  {banner}
  {notes}
</div>
<p class="hint">Stations are the chip profile's own declared resources — one per memory level and
compute unit, grey for the ones this model doesn't cost (hover a grey station for why, same as
the timeline's rows, D20/D43). Blocks are sized from each event's own bytes, log-compressed so the
smallest and largest both stay visible — not to scale against each other or against the stations.
Solid blocks are operand B, hatched blocks are operand A streaming (D31), hollow blocks are the
result written back. Each compute station glows independently while it executes — never "entering"
it, since the byte/flop model has no event distinct from the arithmetic itself for that moment; two
different engines never glow together, because operations run in strict sequence in this model
(D5a). The pseudo-C on the right lights up the line(s) executing right now — more than one at once
when double buffering means more than one statement is truly concurrent (D41). Below, A/B/C's own
shapes (schematic, not to scale, D48): A is cut only along K into k-slices — the whole M height
reads one staged slice, never tiled along M — while B genuinely has a 2-D tile grid, and C mirrors
B's column cuts. The lit cell tracks whichever operand the current instant actually touches.</p>

<div id="controls">
  <button id="playBtn">Play</button>
  <button id="resetBtn">Reset</button>
  <select id="rateSelect">
    <option value="0.25">0.25x</option>
    <option value="1" selected>1x</option>
    <option value="4">4x</option>
    <option value="16">16x</option>
  </select>
  <input id="scrub" type="range" min="0" max="1000" value="0">
  <span id="clockLabel"></span>
</div>

<div id="debugrow">
  <div id="wrap"><svg id="chart"></svg><div id="tip"></div></div>
  <pre id="codepane">{code}</pre>
</div>
<div id="georow"><svg id="geometry"></svg></div>
<div id="geocaption"></div>
<div id="annotation"></div>

<footer>{footer}</footer>

<script>
const DATA = {data};

const COLOUR = {{dram: "#2a78d6", sram: "#8a8983", core: "#eb6834", vector: "#1baf7a"}};
const svg = document.getElementById("chart");
const tip = document.getElementById("tip");
const wrap = document.getElementById("wrap");
const annotation = document.getElementById("annotation");
const NS = "http://www.w3.org/2000/svg";
const el = (name, attrs, text) => {{
  const node = document.createElementNS(NS, name);
  for (const k in attrs) node.setAttribute(k, attrs[k]);
  if (text !== undefined) node.textContent = text;
  return node;
}};

function fmtTime(s) {{
  if (s === 0) return "0 s";
  const u = [[1, "s"], [1e-3, "ms"], [1e-6, "\\u00b5s"], [1e-9, "ns"], [1e-12, "ps"]];
  for (const [scale, name] of u) if (Math.abs(s) >= scale) return trim(s / scale) + " " + name;
  return trim(s / 1e-12) + " ps";
}}
function trim(v) {{
  return (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));
}}

// One station per DATA.stations entry (rows_for's own resource list, D43),
// left to right, evenly spaced and shrunk to fit however many a chip
// declares. Not to scale with each other or with the blocks that move
// between them (the request this page answers: shapes and strategy, not a
// floorplan).
const STATION_H = 74, LANE_Y = 130, MARGIN = 40, GAP = 16;
function stationLayout() {{
  const width = svg.clientWidth || svg.parentNode.clientWidth || 900;
  const n = DATA.stations.length;
  const sw = Math.max(90, Math.min(168, (width - 2 * MARGIN - (n - 1) * GAP) / n));
  const totalW = n * sw + (n - 1) * GAP;
  const left = MARGIN + Math.max(0, (width - 2 * MARGIN - totalW) / 2);
  const positions = [];
  for (let i = 0; i < n; i++) positions.push(left + i * (sw + GAP));
  return {{sw, positions}};
}}
function stationIndexByLane(lane) {{
  return DATA.stations.findIndex(s => s.lane === lane);
}}

// Log-compressed size: a straight linear map either vanishes the smallest event
// or blows out the largest one, and both a 4 kB tile and a 4 MB k-slice ramp
// need to read as blocks on the same page.
const byteValues = DATA.flow.filter(f => f.bytes > 0).map(f => f.bytes);
const MIN_BYTES = byteValues.length ? Math.min(...byteValues) : 1;
const MAX_BYTES = byteValues.length ? Math.max(...byteValues) : 1;
const MIN_PX = 12, MAX_PX = 46;
function blockSize(bytes) {{
  if (!bytes || MAX_BYTES <= MIN_BYTES) return (MIN_PX + MAX_PX) / 2;
  const lo = Math.log1p(MIN_BYTES), hi = Math.log1p(MAX_BYTES);
  const t = (Math.log1p(bytes) - lo) / (hi - lo);
  return MIN_PX + Math.max(0, Math.min(1, t)) * (MAX_PX - MIN_PX);
}}

// load_b/load_a move DRAM -> SRAM; store moves SRAM -> DRAM (the result stages
// through the same port on its way out); hold sits at SRAM; exec moves nothing
// and glows its own engine's station instead (D40/D43 — the model has no
// "entering the array" event distinct from the arithmetic itself). Named by
// lane, not station index, so a chip's own grey intermediate levels (Metis:
// LPDDR4x -> L2 -> L1 -> D-IMC) just sit on the path a block crosses without
// being a stop -- an honest picture of "declared, not modelled" (D20).
const PATH = {{
  load_b: ["dram", "sram"], load_a: ["dram", "sram"],
  store: ["sram", "dram"], hold: ["sram", "sram"],
}};

function activeAt(t) {{
  return DATA.flow.filter(f => f.start <= t && t < Math.max(f.end, f.start + 1e-15));
}}

// ---- the debugger-style code pane -------------------------------------------
const codepane = document.getElementById("codepane");
const codeLineEls = Array.from(codepane.querySelectorAll(".codeline"));
let lastHotLines = new Set();

function updateCodeHighlight(events) {{
  // Every event already carries its real lane. Try a lane-specific tag first
  // ("exec_core"/"exec_vector", D43 — the network path's matrix and vector
  // branches get distinct lines); fall back to the plain stage name, which is
  // what the tiled matmul path still uses unchanged (its exec line is always
  // Lane.CORE, nothing to disambiguate).
  const hotLineLane = new Map();
  events.forEach(f => {{
    const lines = DATA.stage_lines[f.stage + "_" + f.lane] || DATA.stage_lines[f.stage];
    (lines || []).forEach(l => hotLineLane.set(l, f.lane));
  }});

  codeLineEls.forEach((lineEl, i) => {{
    lineEl.classList.remove("hot-dram", "hot-core", "hot-vector");
    const lane = hotLineLane.get(i);
    if (lane) lineEl.classList.add("hot-" + lane);
  }});

  // Only scroll when the hot set actually changed — every frame would fight a
  // reader who scrolled up to read the header/#defines.
  const hotLines = new Set(hotLineLane.keys());
  const changed =
    hotLines.size !== lastHotLines.size || [...hotLines].some(l => !lastHotLines.has(l));
  if (changed && hotLines.size) {{
    const firstEl = codeLineEls[Math.min(...hotLines)];
    if (firstEl) {{
      const paneBox = codepane.getBoundingClientRect();
      const lineBox = firstEl.getBoundingClientRect();
      if (lineBox.top < paneBox.top || lineBox.bottom > paneBox.bottom) {{
        firstEl.scrollIntoView({{block: "center", behavior: "smooth"}});
      }}
    }}
  }}
  lastHotLines = hotLines;
}}

// ---- the tile-geometry panel (A/B/C, D48) -----------------------------------
// A is M x K, cut only along K into k_slices — the whole M height reads one
// staged slice, never tiled along M (D30/D33). B is K x N, cut into a real
// k_slices x tiles_per_ks grid. C mirrors B's column cuts (a result tile is
// the full M height x one B tile's width). The classic GEMM diagram makes the
// shared axes visible for free: A's width and B's height are both K, drawn to
// the same pixel scale; B's width and C's width are both N; A's height and
// C's height are both M.
const geoSvg = document.getElementById("geometry");
const geoCaption = document.getElementById("geocaption");
const georow = document.getElementById("georow");
const GEO = DATA.geometry;
const GEO_GAP = 14, GEO_MIN_PX = 40, GEO_MAX_PX = 200, GEO_GRID_CAP = 40;

if (!GEO) {{
  georow.style.display = "none";
  geoCaption.style.display = "none";
}}

function geoAxisScale() {{
  // Same log-compression principle as blockSize: a batch-1 M and a 4096 K
  // both have to stay legible on the same page, not scaled 1:1 against size.
  const vals = [GEO.m, GEO.n, GEO.k];
  const lo = Math.log1p(Math.min(...vals)), hi = Math.log1p(Math.max(...vals));
  return v => {{
    if (hi <= lo) return (GEO_MIN_PX + GEO_MAX_PX) / 2;
    const frac = (Math.log1p(v) - lo) / (hi - lo);
    return GEO_MIN_PX + frac * (GEO_MAX_PX - GEO_MIN_PX);
  }};
}}

function geoSegments(f) {{
  // Split one flow event's [tile_start, tile_end) into per-k-slice-row
  // segments {{kRow, nStart, nEnd}}: a coalesced step (many real tiles per
  // drawn frame) can span more than one k-slice row, and each row's own
  // column range has to be drawn separately — never one fake single cell.
  if (!GEO || f.tile_start == null || f.tile_end == null) return [];
  const segments = [];
  let t = f.tile_start;
  while (t < f.tile_end) {{
    const kRow = Math.floor(t / GEO.tiles_per_ks);
    const rowEnd = (kRow + 1) * GEO.tiles_per_ks;
    const segEnd = Math.min(f.tile_end, rowEnd);
    segments.push({{
      kRow, nStart: t - kRow * GEO.tiles_per_ks, nEnd: segEnd - kRow * GEO.tiles_per_ks,
    }});
    t = segEnd;
  }}
  return segments;
}}

function mergeColumnRanges(segs) {{
  // C has no k-slice-row dimension — merge overlapping/touching [nStart,
  // nEnd) ranges from different rows into one, since a result tile is the
  // full M height x one n-tile's width, not one per k-slice row it
  // happened to come from (D48).
  const sorted = segs.map(s => [s.nStart, s.nEnd]).sort((a, b) => a[0] - b[0]);
  const merged = [];
  sorted.forEach(([start, end]) => {{
    const last = merged[merged.length - 1];
    if (last && start <= last[1]) last[1] = Math.max(last[1], end);
    else merged.push([start, end]);
  }});
  return merged;
}}

function drawGeometry(events) {{
  if (!GEO) return;
  while (geoSvg.firstChild) geoSvg.removeChild(geoSvg.firstChild);

  const scale = geoAxisScale();
  const mPx = scale(GEO.m), nPx = scale(GEO.n), kPx = scale(GEO.k);
  // B sits at the very top with nothing above it, so — unlike A and C, which
  // have the gap between B and A/C to put a label in — it needs its own top
  // margin, or its label draws off the top of the viewBox entirely (D48).
  const TOP_MARGIN = 16;
  const aX = 0, aY = TOP_MARGIN + kPx + GEO_GAP;
  const bX = kPx + GEO_GAP, bY = TOP_MARGIN;
  const cX = bX, cY = aY;
  geoSvg.setAttribute("viewBox", `0 0 ${{bX + nPx + 10}} ${{aY + mPx + 16}}`);

  geoSvg.appendChild(el("rect", {{class: "geo-rect", x: aX, y: aY, width: kPx, height: mPx}}));
  geoSvg.appendChild(el("rect", {{class: "geo-rect", x: bX, y: bY, width: nPx, height: kPx}}));
  geoSvg.appendChild(el("rect", {{class: "geo-rect", x: cX, y: cY, width: nPx, height: mPx}}));
  geoSvg.appendChild(el("text", {{
    class: "geo-label", x: aX + kPx / 2, y: aY - 5, "text-anchor": "middle",
  }}, `A  ${{GEO.m}} x ${{GEO.k}}`));
  geoSvg.appendChild(el("text", {{
    class: "geo-label", x: bX + nPx / 2, y: bY - 5, "text-anchor": "middle",
  }}, `B  ${{GEO.k}} x ${{GEO.n}}`));
  geoSvg.appendChild(el("text", {{
    class: "geo-label", x: cX + nPx / 2, y: cY - 5, "text-anchor": "middle",
  }}, `C  ${{GEO.m}} x ${{GEO.n}}`));

  // Grid lines, capped: past GEO_GRID_CAP a k-slice/n-tile count draws at a
  // coarser stride instead of one line per tile, and the caption says so —
  // never a silent truncation that would read as "this is the whole grid".
  const strideK = Math.max(1, Math.ceil(GEO.k_slices / GEO_GRID_CAP));
  const strideN = Math.max(1, Math.ceil(GEO.tiles_per_ks / GEO_GRID_CAP));
  for (let g = strideK; g < GEO.k_slices; g += strideK) {{
    const x = aX + (g / GEO.k_slices) * kPx;
    geoSvg.appendChild(el("line", {{class: "geo-grid", x1: x, x2: x, y1: aY, y2: aY + mPx}}));
    const y = bY + (g / GEO.k_slices) * kPx;
    geoSvg.appendChild(el("line", {{class: "geo-grid", x1: bX, x2: bX + nPx, y1: y, y2: y}}));
  }}
  for (let u = strideN; u < GEO.tiles_per_ks; u += strideN) {{
    const xB = bX + (u / GEO.tiles_per_ks) * nPx;
    geoSvg.appendChild(el("line", {{class: "geo-grid", x1: xB, x2: xB, y1: bY, y2: bY + kPx}}));
    const xC = cX + (u / GEO.tiles_per_ks) * nPx;
    geoSvg.appendChild(el("line", {{class: "geo-grid", x1: xC, x2: xC, y1: cY, y2: cY + mPx}}));
  }}

  // Highlights: A lights on load_a (A is being staged), B on exec (the tile
  // actually in the array right now), C on store (the result landing) — each
  // tied to the event that genuinely touches that operand at this instant.
  const aRows = new Set(), bSegs = [], cSegs = [];
  events.forEach(f => {{
    const segs = geoSegments(f);
    if (f.stage === "load_a") segs.forEach(s => aRows.add(s.kRow));
    else if (f.stage === "exec") bSegs.push(...segs);
    else if (f.stage === "store") cSegs.push(...segs);
  }});
  // Boundary lines at the highlight's own real edges, regardless of the
  // coarse stride above — a highlighted cell must never sit unbounded by any
  // visible line just because its own boundary fell off the capped grid, and
  // a multi-row highlight (a coalesced step crossing a k-slice boundary) must
  // read as distinct rows, not one jagged, unexplained blob. Deduplicated:
  // two touched rows share one edge, and drawing it twice (once per row)
  // piles thin, near-identical lines on top of each other when rows are only
  // a pixel or two tall — the exact failure mode that made a real highlight
  // read as a washed-out sliver instead of a visible block.
  const drawnBoundaries = new Set();
  function rowBoundary(x1, y, x2) {{
    const key = "row:" + y;
    if (drawnBoundaries.has(key)) return;
    drawnBoundaries.add(key);
    geoSvg.appendChild(el("line", {{class: "geo-boundary", x1, x2, y1: y, y2: y}}));
  }}
  function colBoundary(x, y1, y2) {{
    const key = "col:" + x + ":" + y1 + ":" + y2;
    if (drawnBoundaries.has(key)) return;
    drawnBoundaries.add(key);
    geoSvg.appendChild(el("line", {{class: "geo-boundary", x1: x, x2: x, y1, y2}}));
  }}
  aRows.forEach(kRow => {{
    const x0 = aX + (kRow / GEO.k_slices) * kPx, x1 = aX + ((kRow + 1) / GEO.k_slices) * kPx;
    colBoundary(x0, aY, aY + mPx);
    colBoundary(x1, aY, aY + mPx);
    geoSvg.appendChild(el("rect", {{
      class: "geo-highlight", x: x0, y: aY, width: x1 - x0, height: mPx,
      style: "fill:" + COLOUR.dram,
    }}));
  }});
  bSegs.forEach(s => {{
    const x0 = bX + (s.nStart / GEO.tiles_per_ks) * nPx;
    const x1 = bX + (s.nEnd / GEO.tiles_per_ks) * nPx;
    const y0 = bY + (s.kRow / GEO.k_slices) * kPx;
    const y1 = bY + ((s.kRow + 1) / GEO.k_slices) * kPx;
    rowBoundary(bX, y0, bX + nPx);
    rowBoundary(bX, y1, bX + nPx);
    colBoundary(x0, y0, y1);
    colBoundary(x1, y0, y1);
    geoSvg.appendChild(el("rect", {{
      class: "geo-highlight", x: x0, y: y0, width: x1 - x0, height: y1 - y0,
      style: "fill:" + COLOUR.core,
    }}));
  }});
  // C has no k-slice-row dimension: several segments from different rows
  // routinely land on the same columns (a wave's tiles span many rows, most
  // covering all or most of one row's width), and must draw/read as one C
  // column range, not once per row it happened to come from.
  const cRanges = mergeColumnRanges(cSegs);
  cRanges.forEach(([n0, n1]) => {{
    const x0 = cX + (n0 / GEO.tiles_per_ks) * nPx;
    const x1 = cX + (n1 / GEO.tiles_per_ks) * nPx;
    colBoundary(x0, cY, cY + mPx);
    colBoundary(x1, cY, cY + mPx);
    geoSvg.appendChild(el("rect", {{
      class: "geo-highlight", x: x0, y: cY, width: x1 - x0, height: mPx,
      style: "fill:" + COLOUR.dram,
    }}));
  }});

  geoCaption.innerHTML = geoCaptionHtml(aRows, bSegs, cRanges, strideK, strideN);
}}

function formatIndexRanges(nums) {{
  // Collapse consecutive integers into "start..end" so a whole-A ramp (every
  // k-slice at once, e.g. --a-strategy whole) reads as one span instead of
  // every one of 256 indices spelled out.
  const sorted = [...nums].sort((a, b) => a - b);
  const parts = [];
  let start = sorted[0], prev = sorted[0];
  for (let i = 1; i <= sorted.length; i++) {{
    const v = sorted[i];
    if (v === prev + 1) {{ prev = v; continue; }}
    parts.push(start === prev ? `${{start}}` : `${{start}}..${{prev}}`);
    start = prev = v;
  }}
  return parts.join(",");
}}

function geoCaptionHtml(aRows, bSegs, cRanges, strideK, strideN) {{
  const g = GEO;
  const gridNote =
    strideK > 1 || strideN > 1
      ? ` (gridlines every ${{strideK}} k-slice(s), ${{strideN}} n-tile(s) — ` +
        `${{g.k_slices}}x${{g.tiles_per_ks}} total)`
      : "";
  const line1 =
    `A tile = ${{g.m}} rows x ${{g.rows}} cols &middot; ${{g.k_slices}} k-slices &middot; ` +
    `A(:,0) &hellip; A(:,${{g.k_slices - 1}})`;
  const line2 =
    `B tile = ${{g.rows}} rows x ${{g.cols}} cols &middot; ${{g.k_slices}}x${{g.tiles_per_ks}} ` +
    `tiles &middot; B(0,0) &hellip; B(${{g.k_slices - 1}},${{g.tiles_per_ks - 1}})${{gridNote}}`;
  const parts = [];
  if (aRows.size) {{
    parts.push("A(:," + formatIndexRanges(aRows) + ")");
  }}
  bSegs.forEach(s => {{
    const cols = s.nEnd - s.nStart === 1 ? `${{s.nStart}}` : `${{s.nStart}}..${{s.nEnd - 1}}`;
    parts.push(`B(${{s.kRow}},${{cols}})`);
  }});
  cRanges.forEach(([n0, n1]) => {{
    const cols = n1 - n0 === 1 ? `${{n0}}` : `${{n0}}..${{n1 - 1}}`;
    parts.push(`C(:,${{cols}})`);
  }});
  const active = parts.length
    ? `<div class="geo-active">active: ${{parts.join(" &middot; ")}}</div>`
    : `<div class="geo-active geo-static">nothing in flight</div>`;
  const line1Html = `<div class="geo-static">${{line1}}</div>`;
  const line2Html = `<div class="geo-static">${{line2}}</div>`;
  return line1Html + line2Html + active;
}}

function draw(t) {{
  const width = svg.clientWidth || svg.parentNode.clientWidth || 900;
  const height = LANE_Y + STATION_H + 30;
  svg.setAttribute("height", height);
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  const defs = el("defs", {{}});
  const pat = el("pattern", {{
    id: "streaming", width: 5, height: 5, patternUnits: "userSpaceOnUse",
    patternTransform: "rotate(45)",
  }});
  pat.appendChild(el("rect", {{width: 5, height: 5, fill: "var(--surface)"}}));
  pat.appendChild(el("line",
    {{x1: 0, y1: 0, x2: 0, y2: 5, stroke: COLOUR.dram, "stroke-width": 2.6}}));
  defs.appendChild(pat);
  svg.appendChild(defs);

  const layout = stationLayout();
  const dramIndex = stationIndexByLane("dram");
  const lastIndex = DATA.stations.length - 1;
  if (dramIndex >= 0) {{
    // Spans the whole declared chain, DRAM to the last compute station, so
    // grey intermediate levels sit visibly on the same track a block crosses
    // without stopping there (D20/D43).
    svg.appendChild(el("line", {{
      class: "lane-track",
      x1: layout.positions[dramIndex] + layout.sw / 2,
      x2: layout.positions[lastIndex] + layout.sw / 2,
      y1: LANE_Y, y2: LANE_Y,
    }}));
  }}

  const events = activeAt(t);

  DATA.stations.forEach((station, i) => {{
    const x0 = layout.positions[i];
    const cssVar = station.lane ? "var(--" + station.lane + ")" : "var(--idle)";
    const rect = el("rect", {{
      class: "station", x: x0, y: LANE_Y - STATION_H / 2,
      width: layout.sw, height: STATION_H, rx: 8,
      style: "stroke:" + cssVar,
    }});
    if (station.note) rect.dataset.tip = station.name + "\\n" + station.note;
    svg.appendChild(rect);
    svg.appendChild(el("text", {{
      class: "station-label", x: x0 + layout.sw / 2, y: LANE_Y - 8, "text-anchor": "middle",
      style: "fill:" + cssVar,
    }}, station.name));
    svg.appendChild(el("text", {{
      class: "station-detail", x: x0 + layout.sw / 2, y: LANE_Y + 10, "text-anchor": "middle",
    }}, station.detail));
    // Every compute station glows independently on its own matching-lane exec
    // event, never together — operations run in strict sequence in this
    // model (D5a), so at most one engine is ever really active at once.
    if (station.lane === "core" || station.lane === "vector") {{
      const active = events.some(f => f.stage === "exec" && f.lane === station.lane);
      const glow = el("rect", {{
        class: "glow", x: x0 - 4, y: LANE_Y - STATION_H / 2 - 4, width: layout.sw + 8,
        height: STATION_H + 8, rx: 10,
        style: "opacity:" + (active ? 0.9 : 0) + "; stroke:" + cssVar,
      }});
      svg.appendChild(glow);
    }}
  }});

  events.forEach((f, i) => {{
    if (f.stage === "exec") return;  // glow only, drawn on the stations above
    const [fromLane, toLane] = PATH[f.stage] || ["sram", "sram"];
    const fromIndex = stationIndexByLane(fromLane), toIndex = stationIndexByLane(toLane);
    if (fromIndex < 0 || toIndex < 0) return;
    const x0c = layout.positions[fromIndex] + layout.sw / 2;
    const x1c = layout.positions[toIndex] + layout.sw / 2;
    const span = Math.max(f.end - f.start, 1e-15);
    const progress = Math.max(0, Math.min(1, (t - f.start) / span));
    // hold has no travel; everything else eases toward its destination.
    const cx = f.stage === "hold" ? x0c : x0c + (x1c - x0c) * progress;
    // A small per-event jitter off the lane so simultaneous blocks (double
    // buffering) do not fully overlap.
    const jitter = ((f.step % 3) - 1) * 16;
    const cy = LANE_Y + jitter + (f.stage === "hold" ? 46 : 0);
    const size = blockSize(f.bytes);
    const hollow = f.stage === "store";
    const fillColour = COLOUR[f.lane] || COLOUR.sram;
    const rect = el("rect", {{
      x: cx - size / 2, y: cy - size / 2, width: size, height: size, rx: 3,
      fill: hollow ? "var(--surface)" : (f.streaming ? "url(#streaming)" : fillColour),
      stroke: hollow || f.streaming ? fillColour : "var(--surface)",
      "stroke-width": hollow || f.streaming ? 1.4 : 0.8,
    }});
    rect.dataset.tip = f.tip;
    svg.appendChild(rect);
  }});

  renderAnnotation(t, events);
  updateCodeHighlight(events);
  drawGeometry(events);
}}

function renderAnnotation(t, events) {{
  const lines = events
    .filter(f => f.stage !== "hold" || events.length === 1)
    .map(f => `<div class="active">${{f.tip.replace(/\\n/g, " · ")}}</div>`);
  const fillDrain = DATA.fill_drain_s > 0
    ? `, +${{fmtTime(DATA.fill_drain_s)}} fill/drain this trace shows`
    : "";
  annotation.innerHTML =
    `<div class="now">t = ${{fmtTime(t)}} of ${{fmtTime(DATA.total)}} — ` +
    `reported latency ${{fmtTime(DATA.reported_latency_s)}}${{fillDrain}}</div>` +
    (lines.length ? lines.join("") : `<div class="active empty">nothing in flight</div>`);
}}

// ---- playback --------------------------------------------------------------
let clock = 0, playing = false, rate = 1, lastFrame = null;
const WALL_SECONDS_PER_RUN = 12;  // real seconds for one full playthrough at 1x

const playBtn = document.getElementById("playBtn");
const resetBtn = document.getElementById("resetBtn");
const rateSelect = document.getElementById("rateSelect");
const scrub = document.getElementById("scrub");
const clockLabel = document.getElementById("clockLabel");

function syncScrub() {{
  scrub.value = String(Math.round((clock / DATA.total) * 1000) || 0);
  clockLabel.textContent = fmtTime(clock) + " / " + fmtTime(DATA.total);
}}

function frame(now) {{
  if (playing) {{
    if (lastFrame !== null) {{
      const dtWall = (now - lastFrame) / 1000;
      clock = Math.min(DATA.total, clock + dtWall * (DATA.total / WALL_SECONDS_PER_RUN) * rate);
      if (clock >= DATA.total) playing = false;
    }}
    lastFrame = now;
    draw(clock);
    syncScrub();
    if (playing) requestAnimationFrame(frame);
    else playBtn.textContent = "Play";
  }}
}}

playBtn.addEventListener("click", () => {{
  if (clock >= DATA.total) clock = 0;
  playing = !playing;
  playBtn.textContent = playing ? "Pause" : "Play";
  lastFrame = null;
  if (playing) requestAnimationFrame(frame);
}});
resetBtn.addEventListener("click", () => {{
  playing = false; playBtn.textContent = "Play"; clock = 0; draw(clock); syncScrub();
}});
rateSelect.addEventListener("change", () => {{ rate = parseFloat(rateSelect.value); }});
scrub.addEventListener("input", () => {{
  playing = false; playBtn.textContent = "Play";
  clock = (parseInt(scrub.value, 10) / 1000) * DATA.total;
  draw(clock); syncScrub();
}});

svg.addEventListener("pointermove", e => {{
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

addEventListener("resize", () => draw(clock));
draw(0);
syncScrub();
</script>
"""


def render(
    *,
    title: str,
    subtitle: str,
    footer: str,
    flow: list[dict[str, object]],
    total_s: float,
    reported_latency_s: float,
    fill_drain_s: float,
    stations: list[dict[str, object]],
    a_strategy: str | None,
    b_dataflow: str | None,
    notes: list[str],
    code_lines: list[str],
    stage_lines: dict[str, list[int]],
    geometry: dict[str, int] | None = None,
) -> str:
    """Build the page. Pure: returns text, writes nothing.

    ``code_lines``/``stage_lines`` are ``Deployment.code``, split, and
    ``Deployment.stage_lines`` — the same pseudo-C the timeline page's "How it
    is deployed on the chip" section shows, here with each line addressable so
    the debug-session-style highlight (D41) can light up the ones live at the
    animation's current time.

    ``a_strategy``/``b_dataflow`` are ``None`` for a workload with no single
    A/B dataflow strategy to name — a network's operations run in sequence
    (D5a), not as one matmul (D42) — and the banner says so instead.

    ``geometry`` is ``{m, n, k, rows, cols, k_slices, tiles_per_ks}`` for a
    lone matmul (D48) — ``Deployment``'s own tile-grid numbers, straight
    through with no re-derivation. ``None`` for a network workload, which has
    no A/B tile grid; the geometry panel renders nothing in that case.
    """
    data = json.dumps(
        {
            "flow": flow,
            "total": total_s,
            "reported_latency_s": reported_latency_s,
            "fill_drain_s": fill_drain_s,
            "stations": stations,
            "stage_lines": stage_lines,
            "geometry": geometry,
        }
    )
    banner = (
        f"<b>a-strategy</b> {_escape(a_strategy)} &nbsp; · &nbsp; <b>b-dataflow</b> "
        f"{_escape(b_dataflow)}"
        if a_strategy is not None and b_dataflow is not None
        else "<b>workload</b> a network graph — operations run in sequence (D5a), "
        "no single A/B dataflow strategy to name"
    )
    notes_html = (
        "<ul>" + "".join(f"<li>{_escape(n)}</li>" for n in notes) + "</ul>" if notes else ""
    )
    code_html = "".join(
        f'<div class="codeline" data-line="{i}">{_escape(line) or " "}</div>'
        for i, line in enumerate(code_lines)
    )
    return TEMPLATE.format(
        title=title,
        subtitle=subtitle,
        footer=footer,
        data=data,
        banner=banner,
        notes=notes_html,
        code=code_html,
    )


def _escape(text: str) -> str:
    """Minimal HTML escaping, matching ``timeline_html._escape``."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
