"""Render a :class:`PipelineTrace` as a Kanata log, readable by Konata.

Konata (github.com/shioyadan/Konata) draws a pipeline as one row per instruction
and one coloured span per stage. That is the right shape for this: a matmul's
tile steps are the rows, and ``Ld``/``Ex`` are the stages, so the double-buffered
overlap appears as the staircase it is.

Format (Kanata 0004), one command per line, tab-separated:

===========================  ====================================================
``Kanata\t0004``             header
``C=\t<cycle>``              starting cycle
``I\t<id>\t<seq>\t<tid>``    a new instruction
``L\t<id>\t<kind>\t<text>``  label; kind 0 is the compact left pane, 1 the tooltip
``S\t<id>\t<lane>\t<stage>`` a stage starts on a lane
``E\t<id>\t<lane>\t<stage>`` a stage ends
``R\t<id>\t<seq>\t<flush>``  retire; flush 1 means squashed
``C\t<n>``                   advance n cycles
===========================  ====================================================

**Ticks, not cycles.** A Kanata cycle is whatever this file says it is, and the
engine has no cycle-accurate notion of one — so a tick here is
``total_time / resolution``, printed in the header comment. That is exactly the
"scaled to the total time" reading: the whole application spans ``resolution``
ticks whether it took 71 microseconds or 8 milliseconds, and two chips drawn at
the same resolution are directly comparable in shape even when their absolute
times differ by orders of magnitude.

Every span is given at least one tick, so a stage that rounds to nothing still
appears; the resulting drift is bounded by one tick per span and is reported in
the header rather than hidden.
"""

from __future__ import annotations

from collections import defaultdict

from bwz.analysis.pipeline import Lane, PipelineTrace, Span
from bwz.units import format_time

DEFAULT_RESOLUTION = 2000
"""Ticks the whole trace spans. Fine enough that a 64-step schedule shows each
step's fill and drain, coarse enough that Konata opens instantly."""

_LANE_INDEX = {Lane.DRAM: 0, Lane.SRAM: 1, Lane.CORE: 0}
"""Konata lane within a row. DRAM and CORE share lane 0 because they are
sequential for one tile — load then execute — and reading them on one line is the
point. SRAM occupancy goes on lane 1, where its overlap with the *next* tile's
load is visible."""


def to_kanata(
    trace: PipelineTrace,
    *,
    title: str,
    resolution: int = DEFAULT_RESOLUTION,
) -> str:
    """Serialise *trace*. Pure: returns text, writes nothing."""
    if not trace.spans or trace.total_s <= 0:
        return f"Kanata\t0004\nC=\t0\n// {title}: nothing to draw\n"

    tick_s = trace.total_s / resolution

    def tick(seconds: float) -> int:
        return round(seconds / tick_s)

    by_step: dict[int, list[Span]] = defaultdict(list)
    for span in trace.spans:
        by_step[span.step].append(span)

    # (tick, order, text). Order within a tick: I, then L, then any E, then any S,
    # then R — so an instruction exists before its stages, a lane is released
    # before it is re-claimed, and a row retires after its last stage ends.
    events: list[tuple[int, int, str]] = []
    for ident, step in enumerate(sorted(by_step)):
        spans = sorted(by_step[step], key=lambda s: (s.start_s, s.lane.value))
        start = min(tick(s.start_s) for s in spans)
        end = max(tick(s.end_s) for s in spans)
        end = max(end, start + 1)

        label = _label(spans, trace)
        events.append((start, 0, f"I\t{ident}\t{ident}\t0"))
        events.append((start, 1, f"L\t{ident}\t0\t{label}"))
        events.append((start, 1, f"L\t{ident}\t1\t{_detail(spans, tick_s)}"))
        for span in spans:
            lane = _LANE_INDEX[span.lane]
            s_tick = tick(span.start_s)
            e_tick = max(tick(span.end_s), s_tick + 1)
            # E before S at the same tick: a lane must be free before the next
            # stage claims it, and one-tick minimum durations make ties common.
            events.append((s_tick, 3, f"S\t{ident}\t{lane}\t{span.stage.value}"))
            events.append((e_tick, 2, f"E\t{ident}\t{lane}\t{span.stage.value}"))
        events.append((end, 4, f"R\t{ident}\t{ident}\t0"))

    events.sort(key=lambda e: (e[0], e[1]))

    lines = [
        "Kanata\t0004",
        f"// {title}",
        f"// 1 tick = {format_time(tick_s)}; {resolution} ticks = "
        f"{format_time(trace.total_s)} total",
        "// lanes: row lane 0 = DRAM load then core execute, lane 1 = on-chip residency",
        f"// {trace.steps} steps"
        + (f" coalesced from {trace.tiles}" if trace.coalesced else "")
        + f"; double buffered: {'yes' if trace.double_buffered else 'no'}",
        "C=\t0",
    ]
    cycle = 0
    for at, _order, text in events:
        if at > cycle:
            lines.append(f"C\t{at - cycle}")
            cycle = at
        lines.append(text)
    lines.append("")
    return "\n".join(lines)


def _label(spans: list[Span], trace: PipelineTrace) -> str:
    head = spans[0]
    prefix = "" if trace.kind == "tiles" else f"{head.phase.value}: "
    return f"{prefix}{head.label}"


def _detail(spans: list[Span], tick_s: float) -> str:
    parts = [
        f"{span.lane.value}/{span.stage.value} {format_time(span.duration_s)}" for span in spans
    ]
    return "  ".join(parts) + f"  (tick = {format_time(tick_s)})"
