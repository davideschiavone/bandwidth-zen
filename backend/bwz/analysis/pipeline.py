"""A tile-level schedule behind the roofline's ``max(load, compute)``.
``docs/MODEL.md`` §6.5.

The roofline reports one number per operation. This module decomposes that
number into the schedule it implies — which resource is busy when — so the
overlap can be *seen* rather than asserted. It invents no new cost: every span
below is a slice of a quantity ``op_roofline`` already produced, and the spans on
each lane sum back to it.

**The matmul is the interesting case**, because its tiling is not a metaphor. A
weight-stationary array holds a ``rows x cols`` slice of ``B`` and streams ``M``
rows of ``A`` through it, so the schedule has

    tiles = ceil(K/rows) * ceil(N/cols)

steps, and that is the same tile count :func:`analysis.tiling.systolic_utilisation`
divides by. Each step loads ``t_dram/tiles`` and computes ``t_compute/tiles``.

Two resources, each serial in itself: one DRAM channel, one array.

    depth            = 2 if double buffered else 1          [tiles that fit at once]
    load_start(i)    = max(load_end(i-1), compute_end(i-depth))
    compute_start(i) = max(load_end(i), compute_end(i-1))

The ``compute_end(i-depth)`` term is the buffer being *freed*: with two buffers a
tile cannot be fetched until the one two steps back has been consumed. Drop it
and the schedule quietly assumes infinite on-chip capacity — every tile loaded at
once — which is the thing capacity planning exists to prevent.

**What the picture shows that the number hides.** Under double buffering the
schedule costs one extra step of the *non-binding* resource — the pipeline fill
when compute dominates, the drain when DRAM does:

    total = max(t_dram, t_compute) + min(t_dram, t_compute) / tiles

so ``max(load, compute)`` is the ``tiles -> infinity`` limit, and the error it
carries is ``min(t_dram, t_compute)/tiles``. That is negligible for a large
matmul (625x625 tiles) and is not negligible for a decode step, where a
projection is a handful of tiles. Recorded in ``report.assumptions`` and reported
as :attr:`PipelineTrace.fill_drain_s` rather than folded silently into a latency.

Without double buffering the schedule costs ``t_dram + t_compute`` exactly, which
is what the roofline already says — there is nothing to hide when nothing
overlaps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from bwz.analysis.roofline import MachineModel
from bwz.analysis.tiling import padded
from bwz.graph.ops import ComputeGraph, GraphPhase, MatmulAttrs, Operation
from bwz.report import OpResult, PhaseResult

MAX_DRAWN_STEPS = 64
"""Steps a trace draws before coalescing. A 10000-cubed matmul on a 16x16 array
has 390 625 tiles; no one reads 390 625 rows. Coalescing groups them and says so
— the alternative, silently drawing the first few, would misrepresent the shape
of the schedule."""


class Lane(StrEnum):
    """A resource that can be busy. One row of the picture."""

    DRAM = "dram"
    SRAM = "sram"
    CORE = "core"


class Stage(StrEnum):
    """What a lane is doing. Konata calls these stages; a Gantt calls them bars."""

    DISPATCH = "Dis"
    LOAD = "Ld"
    HOLD = "Hold"
    EXEC = "Ex"


@dataclass(frozen=True, slots=True)
class Span:
    """One resource busy over one interval."""

    lane: Lane
    stage: Stage
    label: str
    start_s: float
    end_s: float
    step: int
    phase: GraphPhase

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True, slots=True)
class PipelineTrace:
    """The schedule implied by a phase's roofline numbers."""

    spans: tuple[Span, ...]
    total_s: float
    reported_latency_s: float
    steps: int
    """Steps actually drawn, after coalescing."""
    tiles: int
    """Steps the schedule really has. Equal to ``steps`` unless coalesced."""
    double_buffered: bool
    fill_drain_s: float
    """Pipeline fill/drain the roofline's ``max()`` omits. Zero when not
    double-buffered, because then nothing overlaps and the sum is exact."""
    kind: str
    """``"tiles"`` for a single matmul, ``"operations"`` for a graph."""

    @property
    def coalesced(self) -> bool:
        return self.tiles > self.steps

    @property
    def busy_s(self) -> dict[Lane, float]:
        """Summed span duration per lane.

        For DRAM and the array this is time occupied, because their spans never
        overlap — each is a single serial resource. For SRAM it is **not**: a
        double buffer holds two tiles at once, so the sum runs to about twice the
        wall clock. Divide it by the span and you get a *depth*, not a duty
        cycle; :attr:`concurrency` does that division with the right label
        attached.
        """
        out = dict.fromkeys(Lane, 0.0)
        for span in self.spans:
            out[span.lane] += span.duration_s
        return out

    @property
    def concurrency(self) -> dict[Lane, tuple[float, int]]:
        """``(mean, peak)`` spans in flight per lane.

        The mean is ``busy_s / total_s`` — a time-average count of how many
        things that resource was holding at once. For DRAM and the array it
        cannot exceed 1 and reads as a duty cycle. For SRAM it is the number of
        tile buffers occupied, which is 2 under double buffering and 1 without;
        reporting *that* as a percentage is what made a correct schedule look
        like a broken one.
        """
        busy = self.busy_s
        out: dict[Lane, tuple[float, int]] = {}
        for lane in Lane:
            spans = [s for s in self.spans if s.lane is lane]
            edges = sorted([(s.start_s, 1) for s in spans] + [(s.end_s, -1) for s in spans])
            peak = live = 0
            for _at, delta in edges:
                live += delta
                peak = max(peak, live)
            mean = busy[lane] / self.total_s if self.total_s > 0 else 0.0
            out[lane] = (mean, peak)
        return out


def tile_count(op: Operation, machine: MachineModel) -> int:
    """Weight tiles a matmul walks: ``ceil(K/rows) * ceil(N/cols)``.

    The same decomposition ``systolic_utilisation`` costs, so a trace built from
    it cannot disagree with the utilisation the report quotes. Returns 1 when the
    profile declares no array geometry — there is then no tile to speak of.
    """
    if not isinstance(op.attrs, MatmulAttrs):
        return 1
    dims = machine.unit.systolic_dims
    if dims is None:
        return 1
    rows, cols = dims
    return (padded(op.attrs.k, rows) // rows) * (padded(op.attrs.n, cols) // cols)


def build_trace(
    graph: ComputeGraph,
    phase: PhaseResult,
    machine: MachineModel,
    *,
    double_buffered: bool,
    max_steps: int = MAX_DRAWN_STEPS,
) -> PipelineTrace:
    """Decompose *phase* into resource spans.

    A single-matmul graph is decomposed into tile steps; anything else is
    decomposed into one step per operation, which is the granularity at which the
    engine actually models a network (no cross-operation overlap, D5a).
    """
    if len(graph.ops) == 1 and isinstance(graph.ops[0].attrs, MatmulAttrs):
        return _tile_trace(
            graph.ops[0],
            phase,
            machine,
            double_buffered=double_buffered,
            max_steps=max_steps,
        )
    return _operation_trace(graph, phase, double_buffered=double_buffered, max_steps=max_steps)


def _tile_trace(
    op: Operation,
    phase: PhaseResult,
    machine: MachineModel,
    *,
    double_buffered: bool,
    max_steps: int,
) -> PipelineTrace:
    result = phase.ops[0]
    tiles = tile_count(op, machine)
    steps = min(tiles, max(1, max_steps))
    per_step = tiles / steps

    load = result.t_dram_s / steps
    execute = result.t_compute_s / steps
    attrs = op.attrs
    assert isinstance(attrs, MatmulAttrs)
    dims = machine.unit.systolic_dims
    shape = f"{dims[0]}x{dims[1]}" if dims is not None else "untiled"
    label = f"B tile {shape}" if per_step == 1 else f"{per_step:.0f} B tiles {shape}"

    spans = _pipelined_tiles(
        [load] * steps,
        [execute] * steps,
        labels=[f"{label} [{i + 1}/{steps}]" for i in range(steps)],
        phase=phase.phase,
        dispatch_s=result.t_fixed_s,
        double_buffered=double_buffered,
    )
    return PipelineTrace(
        spans=tuple(spans),
        total_s=max((s.end_s for s in spans), default=0.0),
        reported_latency_s=result.latency_s,
        steps=steps,
        tiles=tiles,
        double_buffered=double_buffered,
        fill_drain_s=(min(result.t_dram_s, result.t_compute_s) / steps if double_buffered else 0.0),
        kind="tiles",
    )


def _operation_trace(
    graph: ComputeGraph,
    phase: PhaseResult,
    *,
    double_buffered: bool,
    max_steps: int,
) -> PipelineTrace:
    """One step per operation, coalescing runs of operations when there are many.

    A Llama-3-8B decode step is 451 nodes; drawn one per row that is a wall of
    hairlines. Coalescing sums adjacent operations into a block, which is the
    honest reduction: the model already assumes no overlap between them, so a
    block's load and compute are simply the sums of its members'.
    """
    results = list(phase.ops)
    if not results:
        return PipelineTrace((), 0.0, phase.latency_s, 0, 0, double_buffered, 0.0, "operations")

    # Chunking to a fixed group size yields fewer groups than requested whenever
    # the count does not divide, so the step count follows the chunking rather
    # than the other way round.
    groups = _chunk(results, min(len(results), max(1, max_steps)))
    steps = len(groups)
    loads = [sum(r.t_dram_s for r in g) for g in groups]
    executes = [sum(r.t_compute_s for r in g) for g in groups]
    dispatch = sum(r.t_fixed_s for r in results)
    labels = [
        g[0].op_id if len(g) == 1 else f"{g[0].op_id} .. {g[-1].op_id}  ({len(g)} ops)"
        for g in groups
    ]

    # A group's duration is the SUM of its members' latencies, not the latency of
    # their summed terms: coalescing must not silently re-schedule the ops it
    # groups. max(sum L, sum C) <= sum max(L, C), so taking the former would draw
    # a picture faster than the report.
    durations = [
        sum(
            max(r.t_dram_s, r.t_compute_s) if double_buffered else r.t_dram_s + r.t_compute_s
            for r in g
        )
        for g in groups
    ]
    spans = _serial_steps(
        loads,
        executes,
        durations=durations,
        labels=labels,
        phase=phase.phase,
        dispatch_s=dispatch,
        double_buffered=double_buffered,
    )
    return PipelineTrace(
        spans=tuple(spans),
        total_s=max((s.end_s for s in spans), default=0.0),
        reported_latency_s=phase.latency_s,
        steps=steps,
        tiles=len(results),
        double_buffered=double_buffered,
        # Operations do not pipeline against each other in this model, so there
        # is no fill or drain to account for: the trace reproduces the reported
        # latency exactly.
        fill_drain_s=0.0,
        kind="operations",
    )


def _pipelined_tiles(
    loads: list[float],
    executes: list[float],
    *,
    labels: list[str],
    phase: GraphPhase,
    dispatch_s: float,
    double_buffered: bool,
) -> list[Span]:
    """Software-pipeline the tiles of ONE operation, per the constraints above.

    Legitimate here and only here: the tiles of a single matmul are issued by one
    kernel, which is exactly the thing a double buffer overlaps.
    """
    steps = len(loads)
    spans: list[Span] = []
    if dispatch_s > 0:
        # Step -1: the dispatch is not a tile, and giving it step 0 would merge it
        # with the first tile's row.
        spans.append(Span(Lane.CORE, Stage.DISPATCH, "kernel dispatch", 0.0, dispatch_s, -1, phase))

    depth = 2 if double_buffered else 1
    load_end = dispatch_s
    exec_end = dispatch_s
    exec_ends: list[float] = []
    for i in range(steps):
        freed = exec_ends[i - depth] if i >= depth else dispatch_s
        load_start = max(load_end, freed)
        load_end = load_start + loads[i]
        exec_start = max(load_end, exec_end)
        exec_end = exec_start + executes[i]
        exec_ends.append(exec_end)

        if loads[i] > 0:
            spans.append(Span(Lane.DRAM, Stage.LOAD, labels[i], load_start, load_end, i, phase))
        # The tile occupies a buffer from the moment its fetch begins until the
        # array is done with it. This lane is what makes double buffering
        # visible: exactly `depth` bars overlap at any instant, so the SRAM row
        # is busy for about `depth` times the span — which is the sense in which
        # capacity, not bandwidth, is what SRAM contributes.
        spans.append(Span(Lane.SRAM, Stage.HOLD, labels[i], load_start, exec_end, i, phase))
        if executes[i] > 0:
            spans.append(Span(Lane.CORE, Stage.EXEC, labels[i], exec_start, exec_end, i, phase))
    return spans


def _serial_steps(
    loads: list[float],
    executes: list[float],
    *,
    durations: list[float],
    labels: list[str],
    phase: GraphPhase,
    dispatch_s: float,
    double_buffered: bool,
) -> list[Span]:
    """Lay operations end to end, overlapping load and compute only *within* one.

    This is the model's own schedule for a network and not a weaker version of
    it: "no overlap is modelled between one kernel's prefetch and the previous
    kernel's arithmetic" (D5a). Pipelining across operations here would draw a
    picture faster than the report it illustrates.
    """
    spans: list[Span] = []
    now = 0.0
    if dispatch_s > 0:
        spans.append(Span(Lane.CORE, Stage.DISPATCH, "kernel dispatch", 0.0, dispatch_s, -1, phase))
        now = dispatch_s

    for i, (load, execute, duration) in enumerate(zip(loads, executes, durations, strict=True)):
        start = now
        end = start + duration
        # Load leads, arithmetic trails: within a step the two overlap exactly as
        # far as the double buffer allows, and the step ends when the arithmetic
        # does.
        load_end = start + load
        exec_start = end - execute
        now = end

        if load > 0:
            spans.append(Span(Lane.DRAM, Stage.LOAD, labels[i], start, load_end, i, phase))
        spans.append(Span(Lane.SRAM, Stage.HOLD, labels[i], start, end, i, phase))
        if execute > 0:
            spans.append(Span(Lane.CORE, Stage.EXEC, labels[i], exec_start, end, i, phase))
    return spans


def _chunk(items: list[OpResult], groups: int) -> list[list[OpResult]]:
    """Split *items* into *groups* contiguous chunks of near-equal size."""
    size = math.ceil(len(items) / groups)
    return [items[i : i + size] for i in range(0, len(items), size)]
