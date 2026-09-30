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
  .geo-discarded {{ fill: none; stroke: var(--ink-3); stroke-width: 1; stroke-dasharray: 3 2; }}
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
<p class="hint">{intro}</p>

<div id="controls">
  <button id="playBtn">Play</button>
  <button id="resetBtn">Reset</button>
  <button id="nextBtn">Next</button>
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
let lastScrollAt = -Infinity;
const SCROLL_COOLDOWN_MS = 700;

function updateCodeHighlight(events) {{
  // Every event already carries its real lane. Try a lane-specific tag first
  // ("exec_core"/"exec_vector", D43 — the network path's matrix and vector
  // branches get distinct lines); fall back to the plain stage name, which is
  // what the tiled matmul path still uses unchanged (its exec line is always
  // Lane.CORE, nothing to disambiguate).
  const hotLineLane = new Map();
  events.forEach(f => {{
    // A span's own key first: FlashAttention's two array bars share a stage and
    // a lane, and only the key says which statement is running (D71).
    const lines = (f.key && DATA.stage_lines[f.key]) ||
      DATA.stage_lines[f.stage + "_" + f.lane] || DATA.stage_lines[f.stage];
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
    const paneBox = codepane.getBoundingClientRect();
    // Scroll only when NOTHING hot is on screen. Checking just the first
    // (topmost) hot line instead would yank the pane whenever the prefetch
    // block lit up while the exec/store block was already visible — and
    // during playback the hot set alternates between those two blocks
    // constantly, which is exactly what made the pane dance up and down.
    const anyHotVisible = [...hotLines].some(l => {{
      const el = codeLineEls[l];
      if (!el) return false;
      const b = el.getBoundingClientRect();
      return b.bottom > paneBox.top && b.top < paneBox.bottom;
    }});
    // Even then, rate-limit: two hot blocks further apart than the pane is
    // tall can never both be visible, so without a cooldown they would take
    // turns scrolling each other off screen for the whole run.
    const now = performance.now();
    if (!anyHotVisible && now - lastScrollAt > SCROLL_COOLDOWN_MS) {{
      const target = codeLineEls[Math.min(...hotLines)];
      if (target) {{
        // Move the pane's own scrollTop rather than calling scrollIntoView,
        // which walks every scrollable ancestor and drags the whole page
        // along with it.
        const lineBox = target.getBoundingClientRect();
        const delta =
          (lineBox.top + lineBox.height / 2) - (paneBox.top + paneBox.height / 2);
        codepane.scrollTop += delta;
        lastScrollAt = now;
      }}
    }}
  }}
  lastHotLines = hotLines;
}}

// ---- the tile-geometry panel (A/B/C, D48/D53) -------------------------------
// Which operand has a 2-D tile grid is the STATIONARITY's doing, not a fixed
// fact: the grid has a row axis and a column axis (GEO.row_dim/col_dim), and a
// third dimension every tile sweeps in full. An operand is cut along a
// dimension exactly when that dimension is one of the grid's two axes, and
// covered whole along the one that is swept. Under ws that makes B the 2-D one
// and C a set of column bands; under os it makes C the 2-D one and B the bands.
// The classic GEMM diagram makes the shared axes visible for free: A's width
// and B's height are both K, drawn to the same pixel scale; B's width and C's
// width are both N; A's height and C's height are both M.
const GEO_DIMS = {{A: ["M", "K"], B: ["K", "N"], C: ["M", "N"]}};   // [vertical, horizontal]
const geoSvg = document.getElementById("geometry");
const geoCaption = document.getElementById("geocaption");
const georow = document.getElementById("georow");
const GEO = DATA.geometry;
const FLASH = DATA.flash;
const GEO_GAP = 14, GEO_MIN_PX = 40, GEO_MAX_PX = 200, GEO_GRID_CAP = 40;

if (!GEO && !FLASH) {{
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
  // Split one flow event's [tile_start, tile_end) into per-grid-row segments
  // {{row, colStart, colEnd}}: a coalesced step (many real tiles per drawn
  // frame) can span more than one row, and each row's own column range has to
  // be drawn separately — never one fake single cell.
  if (!GEO || f.tile_start == null || f.tile_end == null) return [];
  const segments = [];
  let t = f.tile_start;
  while (t < f.tile_end) {{
    const row = Math.floor(t / GEO.grid_cols) % GEO.grid_rows;
    const rowStart = Math.floor(t / GEO.grid_cols) * GEO.grid_cols;
    const segEnd = Math.min(f.tile_end, rowStart + GEO.grid_cols);
    segments.push({{row, colStart: t - rowStart, colEnd: segEnd - rowStart}});
    t = segEnd;
  }}
  return segments;
}}

function geoSpan(seg, dim) {{
  // Where one segment sits along `dim`, as a [lo, hi) fraction of it. A
  // dimension the grid does not carry is covered in FULL by every tile — that
  // is what "swept" means — so it spans the whole extent (D53).
  if (dim === GEO.row_dim) return [seg.row / GEO.grid_rows, (seg.row + 1) / GEO.grid_rows];
  if (dim === GEO.col_dim) return [seg.colStart / GEO.grid_cols, seg.colEnd / GEO.grid_cols];
  return [0, 1];
}}

function geoDivisions(dim) {{
  if (dim === GEO.row_dim) return GEO.grid_rows;
  if (dim === GEO.col_dim) return GEO.grid_cols;
  return 1;
}}

function mergeSpans(pairs) {{
  const sorted = pairs.map(p => [p[0], p[1]]).sort((a, b) => a[0] - b[0]);
  const merged = [];
  sorted.forEach(([start, end]) => {{
    const last = merged[merged.length - 1];
    if (last && start <= last[1] + 1e-9) last[1] = Math.max(last[1], end);
    else merged.push([start, end]);
  }});
  return merged;
}}

function geoRects(segs, dimV, dimH) {{
  // Segments that land on the same band of the vertical dimension merge along
  // the horizontal one. That is what stops an operand whose vertical dimension
  // is *swept* — C under ws, B under os — from drawing one repeated rectangle
  // per grid row it happened to come from (D48).
  const byBand = new Map();
  segs.forEach(seg => {{
    const v = geoSpan(seg, dimV), h = geoSpan(seg, dimH);
    const key = v[0] + ":" + v[1];
    if (!byBand.has(key)) byBand.set(key, {{v, hs: []}});
    byBand.get(key).hs.push(h);
  }});
  const out = [];
  byBand.forEach(({{v, hs}}) => mergeSpans(hs).forEach(h => out.push({{v, h}})));
  return out;
}}

// ---- the FlashAttention panel (D71) ----------------------------------------
// Q on the left, Kᵀ (and V, the same blocks) on top, the score matrix S in the
// middle, O on the right — one head, cut into Br x Bc blocks. The block being
// computed lights on the array, then on the vector unit while the softmax runs;
// the blocks this wave already consumed are drawn dashed, because FlashAttention
// never stores S: only O ever reaches DRAM.
function drawFlash(events) {{
  while (geoSvg.firstChild) geoSvg.removeChild(geoSvg.firstChild);
  const F = FLASH;
  const vals = [F.q_len, F.kv_len, F.d];
  const lo = Math.log1p(Math.min(...vals)), hi = Math.log1p(Math.max(...vals));
  const px = v => hi === lo ? (GEO_MIN_PX + GEO_MAX_PX) / 2
    : GEO_MIN_PX + (Math.log1p(v) - lo) / (hi - lo) * (GEO_MAX_PX - GEO_MIN_PX);
  const qPx = px(F.q_len), kvPx = px(F.kv_len), dPx = px(F.d);
  // A left margin, or Q's label — centred over a column only d wide — runs off
  // the viewBox for any wide shape.
  const TOP = 16, LEFT = 36;
  const qX = LEFT, sX = qX + dPx + GEO_GAP, oX = sX + kvPx + GEO_GAP;
  const kY = TOP, sY = TOP + dPx + GEO_GAP;
  geoSvg.setAttribute("viewBox", `0 0 ${{oX + dPx + 10}} ${{sY + qPx + 16}}`);

  const box = (x, y, w, h, label) => {{
    geoSvg.appendChild(el("rect", {{class: "geo-rect", x, y, width: w, height: h}}));
    geoSvg.appendChild(el("text", {{
      class: "geo-label", x: x + w / 2, y: y - 5, "text-anchor": "middle",
    }}, label));
  }};
  box(qX, sY, dPx, qPx, `Q ${{F.q_len}}x${{F.d}}`);
  box(sX, kY, kvPx, dPx, `Kᵀ ${{F.d}}x${{F.kv_len}} (and V)`);
  box(sX, sY, kvPx, qPx, "S — never stored");
  box(oX, sY, dPx, qPx, `O ${{F.q_len}}x${{F.d}}`);

  const rowsCut = Math.max(1, Math.ceil(F.q_blocks / GEO_GRID_CAP));
  const colsCut = Math.max(1, Math.ceil(F.kv_blocks / GEO_GRID_CAP));
  const hLines = (x, w) => {{
    for (let i = rowsCut; i < F.q_blocks; i += rowsCut) {{
      const y = sY + (i / F.q_blocks) * qPx;
      geoSvg.appendChild(el("line", {{class: "geo-grid", x1: x, x2: x + w, y1: y, y2: y}}));
    }}
  }};
  const vLines = (y, h) => {{
    for (let i = colsCut; i < F.kv_blocks; i += colsCut) {{
      const x = sX + (i / F.kv_blocks) * kvPx;
      geoSvg.appendChild(el("line", {{class: "geo-grid", x1: x, x2: x, y1: y, y2: y + h}}));
    }}
  }};
  hLines(qX, dPx); hLines(sX, kvPx); hLines(oX, dPx);
  vLines(kY, dPx); vLines(sY, qPx);

  const live = events.filter(f => f.step != null && f.stage !== "hold");
  if (F.coalesced || !live.length) {{
    geoCaption.innerHTML = '<div class="geo-static">' + (F.coalesced
      ? "Steps are coalesced at this size, so no single block is live — pass a larger " +
        "--steps, or a smaller shape, to watch S fill block by block."
      : "S is computed one Br x Bc block at a time on the array, softmaxed on the vector " +
        "unit, used for O += P·V and thrown away.") + "</div>";
    return;
  }}
  const step = live[0].step;
  const wave = Math.floor(step / F.kv_blocks), j = step % F.kv_blocks;
  const first = wave * F.used, last = Math.min((wave + 1) * F.used, F.programs) - 1;
  const rowBlocks = new Set(), heads = new Set();
  for (let p = first; p <= last; p++) {{
    rowBlocks.add(p % F.q_blocks);
    heads.add(Math.floor(p / F.q_blocks));
  }}
  const keys = new Set(live.map(f => f.key));
  const rowSpan = r => [sY + (r / F.q_blocks) * qPx, sY + ((r + 1) / F.q_blocks) * qPx];
  const colSpan = c => [sX + (c / F.kv_blocks) * kvPx, sX + ((c + 1) / F.kv_blocks) * kvPx];
  const paint = (x0, y0, x1, y1, colour) => geoSvg.appendChild(el("rect", {{
    class: "geo-highlight", x: x0, y: y0, width: x1 - x0, height: y1 - y0, style: "fill:" + colour,
  }}));
  rowBlocks.forEach(r => {{
    const [y0, y1] = rowSpan(r);
    for (let c = 0; c < j; c++) {{
      const [x0, x1] = colSpan(c);
      geoSvg.appendChild(el("rect", {{
        class: "geo-discarded", x: x0, y: y0, width: x1 - x0, height: y1 - y0,
      }}));
    }}
    const [x0, x1] = colSpan(j);
    if (keys.has("load_q")) paint(qX, y0, qX + dPx, y1, COLOUR.dram);
    if (keys.has("qk")) paint(x0, y0, x1, y1, COLOUR.core);
    if (keys.has("softmax")) paint(x0, y0, x1, y1, COLOUR.vector);
    if (keys.has("pv")) paint(oX, y0, oX + dPx, y1, COLOUR.core);
    if (keys.has("normalise")) paint(oX, y0, oX + dPx, y1, COLOUR.vector);
    if (keys.has("store_o")) paint(oX, y0, oX + dPx, y1, COLOUR.dram);
  }});
  if (keys.has("load_kv")) {{
    const [x0, x1] = colSpan(j);
    paint(x0, kY, x1, kY + dPx, COLOUR.dram);
  }}
  const doing = [];
  if (keys.has("load_q")) doing.push("loading Q for this wave's programs");
  if (keys.has("load_kv")) {{
    doing.push(`staging K_${{j}} and V_${{j}}, shared by the wave's programs of each head`);
  }}
  if (keys.has("qk")) doing.push(`S = Q·Kᵀ for block ${{j}} on the array`);
  if (keys.has("softmax")) {{
    doing.push(j ? "online softmax and the rescale of O on the vector unit"
      : "online softmax on the vector unit");
  }}
  if (keys.has("pv")) doing.push(`O += P·V_${{j}} on the array`);
  if (keys.has("normalise")) doing.push("O = O / l");
  if (keys.has("store_o")) doing.push("writing O back");
  const hs = [...heads];
  geoCaption.innerHTML =
    `<div class="geo-static">wave ${{wave + 1}} of ${{F.waves}} · kv block ${{j + 1}} of ` +
    `${{F.kv_blocks}} · head${{hs.length > 1 ? "s" : ""}} ${{Math.min(...hs)}}` +
    `${{hs.length > 1 ? ".." + Math.max(...hs) : ""}} · Q row block(s) ` +
    `${{formatIndexRanges([...rowBlocks])}} of ${{F.q_blocks}} (Br=${{F.br}}, Bc=${{F.bc}}). ` +
    `Dashed: S blocks this wave already used and discarded.</div>` +
    `<div class="geo-active">${{doing.join(" · ") || "waiting"}}</div>`;
}}

function drawGeometry(events) {{
  if (FLASH) {{ drawFlash(events); return; }}
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
  }}, `A  M=${{GEO.m}} x K=${{GEO.k}}`));
  geoSvg.appendChild(el("text", {{
    class: "geo-label", x: bX + nPx / 2, y: bY - 5, "text-anchor": "middle",
  }}, `B  K=${{GEO.k}} x N=${{GEO.n}}`));
  geoSvg.appendChild(el("text", {{
    class: "geo-label", x: cX + nPx / 2, y: cY - 5, "text-anchor": "middle",
  }}, `C  M=${{GEO.m}} x N=${{GEO.n}}`));

  // Grid lines, capped: past GEO_GRID_CAP a division count draws at a coarser
  // stride instead of one line per tile, and the caption says so — never a
  // silent truncation that would read as "this is the whole grid". Which
  // operand gets lines along which axis follows from the grid, not from a
  // fixed assumption about B (D53).
  const strideRow = Math.max(1, Math.ceil(GEO.grid_rows / GEO_GRID_CAP));
  const strideCol = Math.max(1, Math.ceil(GEO.grid_cols / GEO_GRID_CAP));
  function drawGridLines(x, y, w, h, dimV, dimH) {{
    const dv = geoDivisions(dimV), dh = geoDivisions(dimH);
    const sv = dimV === GEO.row_dim ? strideRow : strideCol;
    const sh = dimH === GEO.row_dim ? strideRow : strideCol;
    for (let i = sv; i < dv; i += sv) {{
      const yy = y + (i / dv) * h;
      geoSvg.appendChild(el("line", {{class: "geo-grid", x1: x, x2: x + w, y1: yy, y2: yy}}));
    }}
    for (let i = sh; i < dh; i += sh) {{
      const xx = x + (i / dh) * w;
      geoSvg.appendChild(el("line", {{class: "geo-grid", x1: xx, x2: xx, y1: y, y2: y + h}}));
    }}
  }}
  drawGridLines(aX, aY, kPx, mPx, "M", "K");
  drawGridLines(bX, bY, nPx, kPx, "K", "N");
  drawGridLines(cX, cY, nPx, mPx, "M", "N");

  // Highlights: A lights on load_a (A is being staged), the RESIDENT operand
  // on exec (the tile actually in the array right now), C on store (the result
  // landing), and B on its own load when B is not the resident one — each tied
  // to the event that genuinely touches that operand at this instant (D53).
  const aRows = new Set(), execSegs = [], cSegs = [], bSegs = [];
  events.forEach(f => {{
    if (f.stage === "load_a") {{
      // Not geoSegments (the raw *touched* tile range — right for EXEC, which
      // genuinely spans several rows at once). A's byte cost is openings-based
      // (D33/D48): this event *completes* grid rows [start//w, end//w) — the
      // same window the hover's label uses — never the block its last tile
      // merely touches but a later event finishes and gets billed for.
      if (f.tile_start == null || f.tile_end == null) return;
      const first = Math.floor(f.tile_start / GEO.grid_cols);
      const last = Math.floor(f.tile_end / GEO.grid_cols) - 1;
      for (let k = first; k <= last; k++) aRows.add(k % GEO.grid_rows);
    }} else if (f.stage === "exec") {{
      execSegs.push(...geoSegments(f));
    }} else if (f.stage === "store") {{
      cSegs.push(...geoSegments(f));
    }} else if (f.stage === "load_b" && GEO.resident !== "B") {{
      bSegs.push(...geoSegments(f));
    }}
  }});
  const aSegs = [...aRows].map(row => ({{row, colStart: 0, colEnd: GEO.grid_cols}}));
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
  const boxes = {{
    A: [aX, aY, kPx, mPx], B: [bX, bY, nPx, kPx], C: [cX, cY, nPx, mPx],
  }};
  function highlight(operand, segs, colour) {{
    if (!segs.length) return [];
    const [x, y, w, h] = boxes[operand];
    const [dimV, dimH] = GEO_DIMS[operand];
    const rects = geoRects(segs, dimV, dimH);
    rects.forEach(r => {{
      const y0 = y + r.v[0] * h, y1 = y + r.v[1] * h;
      const x0 = x + r.h[0] * w, x1 = x + r.h[1] * w;
      if (r.v[1] - r.v[0] < 1) {{ rowBoundary(x, y0, x + w); rowBoundary(x, y1, x + w); }}
      colBoundary(x0, y0, y1);
      colBoundary(x1, y0, y1);
      geoSvg.appendChild(el("rect", {{
        class: "geo-highlight", x: x0, y: y0, width: x1 - x0, height: y1 - y0,
        style: "fill:" + colour,
      }}));
    }});
    return rects;
  }}
  // DRAM-side highlights first, the array's on top: where the resident operand
  // is also the one being written back (C under os), the arithmetic is what a
  // reader should see at that instant.
  highlight("A", aSegs, COLOUR.dram);
  highlight("B", bSegs, COLOUR.dram);
  highlight("C", cSegs, COLOUR.dram);
  highlight(GEO.resident, execSegs, COLOUR.core);

  geoCaption.innerHTML = geoCaptionHtml(aRows, execSegs, cSegs, strideRow, strideCol);
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

function geoIndex(operand, seg) {{
  // The same rule the hover uses (plot_pipeline._index_notation): a dimension
  // on the grid's row axis takes the row number, one on its column axis takes
  // the column range, and a swept one is ":" because the tile covers it whole.
  const part = dim => {{
    if (dim === GEO.row_dim) return String(seg.row);
    if (dim === GEO.col_dim) {{
      return seg.colEnd - seg.colStart === 1
        ? String(seg.colStart) : `${{seg.colStart}}..${{seg.colEnd - 1}}`;
    }}
    return ":";
  }};
  const [dimV, dimH] = GEO_DIMS[operand];
  return `${{operand}}(${{part(dimV)}},${{part(dimH)}})`;
}}

function geoCaptionHtml(aRows, execSegs, cSegs, strideRow, strideCol) {{
  const g = GEO;
  const gridNote =
    strideRow > 1 || strideCol > 1
      ? ` (gridlines every ${{strideRow}} row(s), ${{strideCol}} column(s) — ` +
        `${{g.grid_rows}}x${{g.grid_cols}} total)`
      : "";
  const line1 =
    `${{g.stationarity}}: ${{g.resident}} stays resident &middot; A ${{g.group_name}} = ` +
    `${{g.band_rows}} x ${{g.band_cols}} &middot; ${{g.a_events}} of them`;
  const line2 =
    `${{g.resident}} tile = ${{g.rows}} rows x ${{g.cols}} cols &middot; ` +
    `${{g.grid_rows}}x${{g.grid_cols}} tiles (${{g.row_dim}} x ${{g.col_dim}}), each sweeping ` +
    `${{g.swept_dim}}${{gridNote}}`;
  const parts = [];
  if (aRows.size) {{
    const bands = formatIndexRanges(aRows);
    parts.push(GEO.row_dim === "M" ? `A(${{bands}},:)` : `A(:,${{bands}})`);
  }}
  execSegs.forEach(s => parts.push(geoIndex(GEO.resident, s)));
  // C's own entries merge the same way its rectangles do: when C does not
  // carry the grid's row axis (ws, where M is swept), every row's segment
  // names the SAME C columns and must read as one range, not one per row.
  const cKeyed = new Map();
  cSegs.forEach(s => {{
    const key = GEO_DIMS.C.includes(GEO.row_dim) ? s.row : "swept";
    if (!cKeyed.has(key)) cKeyed.set(key, []);
    cKeyed.get(key).push([s.colStart, s.colEnd]);
  }});
  cKeyed.forEach((ranges, key) => {{
    mergeSpans(ranges).forEach(([c0, c1]) => {{
      parts.push(geoIndex("C", {{row: key === "swept" ? 0 : key, colStart: c0, colEnd: c1}}));
    }});
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
const nextBtn = document.getElementById("nextBtn");
const rateSelect = document.getElementById("rateSelect");
const scrub = document.getElementById("scrub");
const clockLabel = document.getElementById("clockLabel");

// Every moment the active-event set actually changes — a span starting or
// ending — not every individual event's own start, so "Next" steps through
// meaningful transitions (load -> hold -> exec -> store) one at a time,
// debugger-style, matching the code pane's own step-by-step highlight (D41).
const BOUNDARIES = [...new Set(DATA.flow.flatMap(f => [f.start, f.end]))].sort((a, b) => a - b);

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
nextBtn.addEventListener("click", () => {{
  playing = false; playBtn.textContent = "Play";
  const upcoming = BOUNDARIES.find(t => t > clock + 1e-15);
  clock = upcoming === undefined ? DATA.total : upcoming;
  draw(clock); syncScrub();
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


MATMUL_INTRO = """Stations are the chip profile's own declared resources — one per memory level and
compute unit, grey for the ones this model doesn't cost (hover a grey station for why, same as
the timeline's rows, D20/D43). Blocks are sized from each event's own bytes, log-compressed so the
smallest and largest both stay visible — not to scale against each other or against the stations.
Solid blocks are operand B, hatched blocks are operand A streaming (D31), hollow blocks are the
result written back. Each compute station glows independently while it executes — never "entering"
it, since the byte/flop model has no event distinct from the arithmetic itself for that moment; two
different engines never glow together, because operations run in strict sequence in this model
(D5a). The pseudo-C on the right lights up the line(s) executing right now — more than one at once
when double buffering means more than one statement is truly concurrent (D41). Below, A/B/C's own
shapes (schematic, not to scale, D48): the tile grid has a row axis and a column axis, and every
tile sweeps the third dimension in full, so an operand is cut along a dimension exactly when the
grid carries it and covered whole along the one it sweeps (D53). Under weight-stationary that
gives B the 2-D grid and C a set of column bands; under output-stationary it is the other way
round. The caption below names which. The lit cell tracks whichever operand the current instant
actually touches. <b>Reading a tile address:</b> a tile is named
<code>Operand(row,col)</code>, row-major within the grid, 0-indexed; <code>:</code> in either
position means that dimension is swept in full rather than cut, so <code>C(:,col)</code> is one
column band of C over the whole of M. A comma-range like <code>0..124</code> is not one tile — it
names *every* tile in that row from column 0 through 124 inclusive, compacted so a wave of
hundreds of tiles reads as a few ranges instead of being spelled out one by one:
<code>B(0,0..124); B(1,0..124); B(2,0..124); B(3,0..56)</code> is
<code>125 + 125 + 125 + 57 = 432</code> tiles, not 4."""
"""The help paragraph a matmul or network playback carries."""

FLASH_INTRO = (
    "Stations are the chip profile's own declared resources — one per memory level and compute "
    "unit, grey for the ones this model doesn't cost (D20/D43). Blocks are sized from each "
    "event's own bytes, log-compressed. Hatched blocks are <b>Q</b> (read on a wave's first kv "
    "block), solid blocks a <b>K and V</b> block (shared by every program of the head in the "
    "wave, D33), hollow blocks <b>O</b> written back. The array glows for S = Q&middot;K&#7488; "
    "and O += P&middot;V, the vector unit for the online softmax between them — never both at "
    "once, because FlashAttention-2 runs that chain serially. The program on the right lights "
    "the statement executing right now. Below: one head's Q, K&#7488; and O, and the score "
    "matrix S cut into Br x Bc blocks. The live block lights on the engine working on it; "
    "dashed blocks were already used by this wave and thrown away — S is never stored, which "
    "is the whole of FlashAttention (D71)."
)
"""The help paragraph a FlashAttention playback carries (D71)."""


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
    geometry: dict[str, object] | None = None,
    flash: dict[str, object] | None = None,
    banner: str | None = None,
    intro: str | None = None,
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

    ``geometry`` is the lone matmul's tile grid (D48/D53) — the shape, the
    array, and the grid's own axes (``row_dim``/``col_dim``/``swept_dim``,
    ``resident``): ``Deployment``'s own numbers, straight through with no
    re-derivation, so the panel draws whichever operand this chip's
    stationarity actually keeps resident. ``None`` for a network workload,
    which has no tile grid; the geometry panel renders nothing in that case.

    ``flash`` replaces it for a FlashAttention plan (D71): the shape and blocks —
    ``q_len``, ``kv_len``, ``d``, ``br``, ``bc``, ``q_blocks``, ``kv_blocks``,
    ``waves``, ``used``, ``programs`` and whether steps were ``coalesced`` — from
    which the panel works out which (wave, kv block) each event is. ``banner``
    overrides the strategy line for a workload the matmul vocabulary does not fit.
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
            "flash": flash,
        }
    )
    # Stationarity first: it decides the whole decomposition the other two knobs
    # then place events within, so a reader who sees "a-strategy stage" without
    # it does not know what a staged group even is (D53).
    parts: list[str] = []
    if geometry is not None:
        parts.append(
            f"<b>stationarity</b> {_escape(str(geometry['stationarity']))} "
            f"({_escape(str(geometry['resident']))} resident, "
            f"{geometry['grid_rows']}x{geometry['grid_cols']} tiles "
            f"{_escape(str(geometry['row_dim']))}x{_escape(str(geometry['col_dim']))}, "
            f"sweeping {_escape(str(geometry['swept_dim']))})"
        )
        if int(str(geometry["splits"])) > 1:
            parts.append(f"<b>split-K</b> {geometry['splits']}")
    if a_strategy is not None and b_dataflow is not None:
        parts.append(f"<b>a-strategy</b> {_escape(a_strategy)}")
        parts.append(f"<b>b-dataflow</b> {_escape(b_dataflow)}")
    elif not parts:
        parts.append(
            "<b>workload</b> a network graph — operations run in sequence (D5a), "
            "no single A/B dataflow strategy to name"
        )
    banner = banner if banner is not None else " &nbsp; · &nbsp; ".join(parts)
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
        intro=intro if intro is not None else (FLASH_INTRO if flash is not None else MATMUL_INTRO),
    )


def _escape(text: str) -> str:
    """Minimal HTML escaping, matching ``timeline_html._escape``."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
