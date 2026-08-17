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
    """What a lane is doing — one bar on the timeline."""

    DISPATCH = "Dis"
    LOAD = "Ld"
    HOLD = "Hold"
    EXEC = "Ex"
    STORE = "St"
    """Writing the result back. A separate stage because it happens *after* the
    arithmetic and shares the DRAM port with the next tile's load (D22)."""


@dataclass(frozen=True, slots=True)
class Span:
    """One resource busy over one interval, and how much work that was.

    The quantity matters as much as the interval: "DRAM was busy 4% of the span"
    is a much weaker statement than "DRAM moved 539 MB at 2.04 TB/s", and only
    the second lets two architectures be compared. Each lane carries the quantity
    that is meaningful for it and leaves the others at zero.
    """

    lane: Lane
    stage: Stage
    label: str
    start_s: float
    end_s: float
    step: int
    phase: GraphPhase
    bytes_moved: float = 0.0
    """DRAM lane: bytes crossing the one modelled link in this span."""
    flops: float = 0.0
    """Core lane: operations retired in this span."""
    resident_bytes: float = 0.0
    """SRAM lane: bytes this buffer holds while occupied."""

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def rate_bytes_per_s(self) -> float:
        """Achieved bandwidth over this span, which is the effective rate by
        construction — the span was derived from it."""
        return self.bytes_moved / self.duration_s if self.duration_s > 0 else 0.0

    @property
    def rate_flops_per_s(self) -> float:
        return self.flops / self.duration_s if self.duration_s > 0 else 0.0


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
    def totals(self) -> dict[Lane, float]:
        """The quantity each lane accounts for: bytes for DRAM, operations for the
        core, peak bytes held for SRAM (a stock, not a flow — it does not sum)."""
        out = {
            Lane.DRAM: sum(s.bytes_moved for s in self.spans),
            Lane.CORE: sum(s.flops for s in self.spans),
            Lane.SRAM: max((s.resident_bytes for s in self.spans), default=0.0)
            * max(self.concurrency[Lane.SRAM][1], 1),
        }
        return out

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
    def direction_bytes(self) -> tuple[float, float]:
        """``(read, written)`` over the DRAM lane. Two numbers rather than one
        because a comparison cares which way the bus was busy."""
        reads = sum(s.bytes_moved for s in self.spans if s.stage is Stage.LOAD)
        writes = sum(s.bytes_moved for s in self.spans if s.stage is Stage.STORE)
        return reads, writes

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

    # Split the DRAM time by direction: loads lead the arithmetic, stores trail
    # it, and both queue on the one port.
    traffic = result.dram_bytes or 1.0
    load = result.t_dram_s * (result.dram_read_bytes / traffic) / steps
    store = result.t_dram_s * (result.dram_write_bytes / traffic) / steps
    execute = result.t_compute_s / steps
    attrs = op.attrs
    assert isinstance(attrs, MatmulAttrs)
    dims = machine.unit.systolic_dims
    shape = f"{dims[0]}x{dims[1]}" if dims is not None else "untiled"
    label = f"B tile {shape}" if per_step == 1 else f"{per_step:.0f} B tiles {shape}"

    spans = _pipelined_tiles(
        [load] * steps,
        [execute] * steps,
        [store] * steps,
        labels=[f"{label} [{i + 1}/{steps}]" for i in range(steps)],
        phase=phase.phase,
        dispatch_s=result.t_fixed_s,
        double_buffered=double_buffered,
        bytes_per_step=[result.dram_read_bytes / steps] * steps,
        stored_per_step=[result.dram_write_bytes / steps] * steps,
        flops_per_step=[result.flops / steps] * steps,
        # What one buffer holds: the B tile the array is stationary on, plus the
        # A rows streaming against it. Sized from the schedule, not asserted.
        resident_per_step=[result.weight_bytes / tiles * per_step] * steps,
    )
    return PipelineTrace(
        spans=tuple(spans),
        total_s=max((s.end_s for s in spans), default=0.0),
        reported_latency_s=result.latency_s,
        steps=steps,
        tiles=tiles,
        double_buffered=double_buffered,
        # Measured, not derived: with stores in the schedule the fill/drain is a
        # load at the head plus a store at the tail, and a formula for it would
        # be one more thing to keep in step with the scheduler.
        fill_drain_s=max(0.0, max((s.end_s for s in spans), default=0.0) - result.latency_s),
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
        bytes_per_step=[sum(r.dram_read_bytes for r in g) for g in groups],
        stored_per_step=[sum(r.dram_write_bytes for r in g) for g in groups],
        flops_per_step=[sum(r.flops for r in g) for g in groups],
        resident_per_step=[max((r.weight_bytes for r in g), default=0.0) for g in groups],
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
    stores: list[float],
    *,
    labels: list[str],
    phase: GraphPhase,
    dispatch_s: float,
    double_buffered: bool,
    bytes_per_step: list[float],
    stored_per_step: list[float],
    flops_per_step: list[float],
    resident_per_step: list[float],
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
    dram_free = dispatch_s
    exec_end = dispatch_s
    load_ends: list[float] = []
    exec_starts: list[float] = []
    exec_ends: list[float] = []
    store_ends: list[float] = []
    store_starts: list[float] = []
    load_starts: list[float] = []

    def place_store(index: int) -> None:
        """Drain one result, once it exists and once the port is free."""
        nonlocal dram_free
        start = max(exec_ends[index], dram_free)
        store_starts.append(start)
        dram_free = start + stores[index]
        store_ends.append(dram_free)

    for i in range(steps):
        # A buffer is free once its tile has been *written out*, not merely
        # computed — the hold below runs to the store, so the reuse test must too.
        freed = store_ends[i - depth] if i >= depth else dispatch_s
        load_start = max(dram_free, freed)
        load_end = load_start + loads[i]
        dram_free = load_end
        load_starts.append(load_start)
        load_ends.append(load_end)

        exec_start = max(load_end, exec_end)
        exec_end = exec_start + executes[i]
        exec_starts.append(exec_start)
        exec_ends.append(exec_end)

        # Issue order on the one port: the *next* tile's fetch outranks this
        # tile's write-back, which is what a memory controller does with a write
        # buffer. Draining stores first would stall the array behind them and
        # quietly cancel the double buffer.
        if i >= 1:
            place_store(i - 1)
    if steps:
        place_store(steps - 1)

    for i in range(steps):
        load_start, load_end = load_starts[i], load_ends[i]
        # Kept, not recomputed: `exec_ends[i] - executes[i]` can land a float
        # hair before the previous span's end and read as an overlap.
        exec_start, exec_end = exec_starts[i], exec_ends[i]
        store_start, store_end = store_starts[i], store_ends[i]

        if loads[i] > 0:
            spans.append(
                Span(
                    Lane.DRAM,
                    Stage.LOAD,
                    labels[i],
                    load_start,
                    load_end,
                    i,
                    phase,
                    bytes_moved=bytes_per_step[i],
                )
            )
        # A buffer is occupied from the moment its fetch begins until its result
        # has been written out — not merely until the arithmetic ends. This lane
        # is what makes double buffering visible: exactly `depth` bars overlap at
        # any instant, which is the sense in which capacity, not bandwidth, is
        # what SRAM contributes.
        spans.append(
            Span(
                Lane.SRAM,
                Stage.HOLD,
                labels[i],
                load_start,
                store_end,
                i,
                phase,
                resident_bytes=resident_per_step[i],
            )
        )
        if executes[i] > 0:
            spans.append(
                Span(
                    Lane.CORE,
                    Stage.EXEC,
                    labels[i],
                    exec_start,
                    exec_end,
                    i,
                    phase,
                    flops=flops_per_step[i],
                )
            )
        if stores[i] > 0:
            spans.append(
                Span(
                    Lane.DRAM,
                    Stage.STORE,
                    labels[i],
                    store_start,
                    store_end,
                    i,
                    phase,
                    bytes_moved=stored_per_step[i],
                )
            )
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
    bytes_per_step: list[float],
    stored_per_step: list[float],
    flops_per_step: list[float],
    resident_per_step: list[float],
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
        # Within a step: operands in, arithmetic, result out. The step's length
        # is the reported latency for those operations, so the three are placed
        # inside it rather than summed — a decomposition, not a re-schedule.
        share = stored_per_step[i] / ((bytes_per_step[i] + stored_per_step[i]) or 1.0)
        store_time = load * share
        load_time = load - store_time
        load_end = start + load_time
        store_start = end - store_time
        # Clamped into the step: its length is the reported latency for these
        # operations, and a decomposition may not run past what it decomposes.
        exec_start = min(max(load_end, store_start - execute), max(start, end - execute))
        now = end

        if load_time > 0:
            spans.append(
                Span(
                    Lane.DRAM,
                    Stage.LOAD,
                    labels[i],
                    start,
                    load_end,
                    i,
                    phase,
                    bytes_moved=bytes_per_step[i],
                )
            )
        spans.append(
            Span(
                Lane.SRAM,
                Stage.HOLD,
                labels[i],
                start,
                end,
                i,
                phase,
                resident_bytes=resident_per_step[i],
            )
        )
        if execute > 0:
            spans.append(
                Span(
                    Lane.CORE,
                    Stage.EXEC,
                    labels[i],
                    exec_start,
                    exec_start + execute,
                    i,
                    phase,
                    flops=flops_per_step[i],
                )
            )
        if store_time > 0:
            spans.append(
                Span(
                    Lane.DRAM,
                    Stage.STORE,
                    labels[i],
                    store_start,
                    end,
                    i,
                    phase,
                    bytes_moved=stored_per_step[i],
                )
            )
    return spans


def _chunk(items: list[OpResult], groups: int) -> list[list[OpResult]]:
    """Split *items* into *groups* contiguous chunks of near-equal size."""
    size = math.ceil(len(items) / groups)
    return [items[i : i + size] for i in range(0, len(items), size)]
