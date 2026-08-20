"""Render a matmul's tile schedule as a self-contained, playable DRAM -> SRAM ->
Accelerator flow animation.

The timeline (``timeline_html.py``) answers "where did the time go, on which
resource" as a static, zoomable strip. This answers a different question — "what
does the chosen A-strategy/B-dataflow actually look like happening" — by playing
the same schedule (``bwz.analysis.pipeline.PipelineTrace``) back as motion between
three fixed stations, sized from each tile's own bytes but not pixel-accurate to
them (docs/CORRECTIONS.md D40).

One file, no server, no download, no CDN — open it with ``file://``, same
constraint as ``timeline_html.py``, same visual vocabulary (the ``--dram``/
``--sram``/``--core`` colours, the streaming-A hatch) copied verbatim rather than
imported, because each self-contained page has to stand completely alone.

Matmul only: a station diagram assumes one tile-shaped stream of DRAM/SRAM/
compute events, which is what a lone matmul's trace is and a whole network's
per-operation trace is not (``docs/CLI.md`` §3 vs §4).
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
  .station {{ fill: var(--box); stroke: var(--grid); stroke-width: 1.5; }}
  .station-label {{ font-size: 13px; font-weight: 700; fill: var(--ink); }}
  .station-detail {{ font-size: 10.5px; fill: var(--ink-3); }}
  .lane-track {{ stroke: var(--grid); stroke-width: 1; stroke-dasharray: 3 4; }}
  .glow {{ fill: none; stroke: var(--core); stroke-width: 3; opacity: 0; }}
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
<p class="hint">Blocks are sized from each event's own bytes, log-compressed so the smallest and
largest both stay visible — not to scale against each other or against the stations. Solid blocks
are operand B, hatched blocks are operand A streaming (D31), hollow blocks are the result written
back. The Accelerator station glows while it is executing; nothing "enters" it, since the byte/flop
model does not distinguish that moment from the rest of a wave's arithmetic. The pseudo-C on the
right lights up the line(s) executing right now — more than one at once when double buffering means
more than one statement is truly concurrent (D41).</p>

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
<div id="annotation"></div>

<footer>{footer}</footer>

<script>
const DATA = {data};

const COLOUR = {{dram: "#2a78d6", sram: "#8a8983", core: "#eb6834"}};
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

// Three fixed stations, left to right. Not to scale with each other or with the
// blocks that move between them (the request this page answers: shapes and
// strategy, not a floorplan).
const W = {{dram: 0, sram: 1, acc: 2}};
const STATION_W = 168, STATION_H = 74, LANE_Y = 130, MARGIN = 40;
function stationX(which) {{
  const width = svg.clientWidth || svg.parentNode.clientWidth || 900;
  const span = width - 2 * MARGIN - STATION_W;
  return MARGIN + which * (span / 2);
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
// and glows the Accelerator station instead (D40 — the model has no "entering
// the array" event distinct from the arithmetic itself).
const PATH = {{
  load_b: [W.dram, W.sram], load_a: [W.dram, W.sram],
  store: [W.sram, W.dram], hold: [W.sram, W.sram],
}};

function activeAt(t) {{
  return DATA.flow.filter(f => f.start <= t && t < Math.max(f.end, f.start + 1e-15));
}}

// ---- the debugger-style code pane -------------------------------------------
const codepane = document.getElementById("codepane");
const codeLineEls = Array.from(codepane.querySelectorAll(".codeline"));
const STAGE_LANE = {{load_b: "dram", load_a: "dram", store: "dram", exec: "core"}};
let lastHotLines = new Set();

function updateCodeHighlight(events) {{
  const hotStages = new Set(events.map(f => f.stage));
  const hotLines = new Set();
  hotStages.forEach(stage => (DATA.stage_lines[stage] || []).forEach(l => hotLines.add(l)));

  codeLineEls.forEach((lineEl, i) => {{
    lineEl.classList.remove("hot-dram", "hot-core");
    if (!hotLines.has(i)) return;
    for (const stage of hotStages) {{
      if ((DATA.stage_lines[stage] || []).includes(i)) {{
        lineEl.classList.add("hot-" + (STAGE_LANE[stage] || "dram"));
        break;
      }}
    }}
  }});

  // Only scroll when the hot set actually changed — every frame would fight a
  // reader who scrolled up to read the header/#defines.
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

  svg.appendChild(el("line", {{
    class: "lane-track", x1: stationX(W.dram) + STATION_W / 2, x2: stationX(W.acc) + STATION_W / 2,
    y1: LANE_Y, y2: LANE_Y,
  }}));

  const stations = [
    ["dram", DATA.stations.dram], ["sram", DATA.stations.sram], ["acc", DATA.stations.accelerator],
  ];
  stations.forEach(([key, info], i) => {{
    const x0 = stationX(i);
    svg.appendChild(el("rect", {{
      class: "station", x: x0, y: LANE_Y - STATION_H / 2,
      width: STATION_W, height: STATION_H, rx: 8,
      style: "stroke: var(--" + key + ")",
    }}));
    svg.appendChild(el("text", {{
      class: "station-label", x: x0 + STATION_W / 2, y: LANE_Y - 8, "text-anchor": "middle",
      style: "fill: var(--" + key + ")",
    }}, info.name));
    svg.appendChild(el("text", {{
      class: "station-detail", x: x0 + STATION_W / 2, y: LANE_Y + 10, "text-anchor": "middle",
    }}, info.detail));
    if (key === "acc") {{
      const active = activeAt(t).some(f => f.stage === "exec");
      const glow = el("rect", {{
        class: "glow", x: x0 - 4, y: LANE_Y - STATION_H / 2 - 4, width: STATION_W + 8,
        height: STATION_H + 8, rx: 10, style: "opacity:" + (active ? 0.9 : 0),
      }});
      svg.appendChild(glow);
    }}
  }});

  const events = activeAt(t);
  events.forEach((f, i) => {{
    if (f.stage === "exec") return;  // glow only, drawn on the station above
    const [fromW, toW] = PATH[f.stage] || [W.sram, W.sram];
    const x0c = stationX(fromW) + STATION_W / 2, x1c = stationX(toW) + STATION_W / 2;
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
    stations: dict[str, dict[str, str]],
    a_strategy: str | None,
    b_dataflow: str | None,
    notes: list[str],
    code_lines: list[str],
    stage_lines: dict[str, list[int]],
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
    """
    data = json.dumps(
        {
            "flow": flow,
            "total": total_s,
            "reported_latency_s": reported_latency_s,
            "fill_drain_s": fill_drain_s,
            "stations": stations,
            "stage_lines": stage_lines,
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
