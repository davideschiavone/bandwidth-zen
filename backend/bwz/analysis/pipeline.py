"""A tile-level schedule behind the roofline's ``max(load, compute)``.
``docs/MODEL.md`` §6.5.

The roofline reports one number per operation. This module decomposes that
number into the schedule it implies — which resource is busy when — so the
overlap can be *seen* rather than asserted. It invents no new cost: every span
below is a slice of a quantity ``op_roofline`` already produced, and the spans on
each lane sum back to it.

**The matmul is the interesting case**, because its tiling is not a metaphor.
Which tiles there are follows from the machine's stationarity (D53), and
``analysis/stationarity.py`` is the one place that decides: a weight-stationary
array holds a ``rows x cols`` slice of ``B`` and streams ``M`` past it, giving
``ceil(K/rows) * ceil(N/cols)`` tiles; an output-stationary one holds ``C``'s
accumulator and sweeps ``K`` inside each of ``ceil(M/rows) * ceil(N/cols)``. The
schedule takes whichever grid it is handed — the same one
:func:`analysis.tiling.systolic_utilisation` divides by, so the picture cannot
disagree with the utilisation. Each step loads ``t_dram/tiles`` and computes
``t_compute/tiles``.

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
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING

from bwz.analysis.roofline import MATRIX_OP_TYPES, MachineModel
from bwz.analysis.stationarity import Operand, TileGrid, grid_for
from bwz.graph.ops import ComputeGraph, GraphPhase, MatmulAttrs, Operation
from bwz.report import OpResult, PhaseResult, ReductionPlacement
from bwz.spec.deployment import AStrategy, BDataflow

if TYPE_CHECKING:
    # analysis.dataflow imports tile_count/tiles_per_a_event from this module,
    # so importing DataflowPlan back at runtime would be circular. The type is
    # only ever used in annotations, which `from __future__ import annotations`
    # already defers, so a TYPE_CHECKING-only import is enough.
    from bwz.analysis.dataflow import DataflowPlan

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
    """The matrix engine: matmul, attention, conv, and the dispatches."""
    VECTOR = "vector"
    """Everything else — norms, activations, residuals. A separate lane because
    it is separate silicon: a tensor core does matrix-multiply-accumulate and a
    norm has to go somewhere else (D27)."""


class Stage(StrEnum):
    """What a lane is doing — one bar on the timeline."""

    DISPATCH = "Dis"
    LOAD = "Ld"
    """Operand B. Stationary under ``ws``, streamed past a resident accumulator
    under ``os`` — either way it is B's bytes that cross here (D53)."""
    LOAD_A = "LdA"
    """Operand A, the one that streams through the array. A separate stage
    because the two obey different residency fractions and spill at different
    times — capacity is granted to activations before weights (D15) — so one
    combined LOAD figure cannot say which operand crossed the bus. For a lone
    matmul this stage is instead the per-grid-row staging: the operand enters
    once per row of the tile grid — a k-slice under ``ws``, a band of M rows
    under ``os`` — and every tile of that row reads the staging (D33/D53)."""
    HOLD = "Hold"
    EXEC = "Ex"
    REDUCE = "Red"
    """Split-K's second kernel: the partial results read back and summed (D53).
    Its own stage because it is its own *kernel* — it starts after the GEMM has
    finished, on a different engine, moving bytes that are neither operand."""
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
    op_type: str = ""
    """Which operator family this span serves — ``matmul``, ``attention``, ``norm``…
    A phase mixes them, and "what was computed" is a different question from "how
    long did it take"."""
    bytes_moved: float = 0.0
    """DRAM lane: bytes crossing the one modelled link in this span."""
    flops: float = 0.0
    """Core lane: operations retired in this span."""
    resident_bytes: float = 0.0
    """SRAM lane: bytes this buffer holds while occupied."""
    a_fetch_mode: str = "stream"
    """DRAM lane, ``Stage.LOAD_A`` only: which ``a_strategy`` produced this span
    — ``"stage"`` (a k-slice staged once, D33), ``"stream"`` (a per-tile
    re-fetch, D31) or ``"whole"`` (every k-slice ramped upfront, same bytes as
    stage). Meaningless on any other stage. Defaults to ``"stream"``: a
    network's inter-op activation traffic has no k-slice structure to stage —
    it genuinely streams, the same physical picture ``a_strategy=stream``
    deliberately reproduces for a lone matmul."""
    tile_start: int | None = None
    tile_end: int | None = None
    """A lone matmul only: the global, row-major tile-index range ``[tile_start,
    tile_end)`` this step covers — the same numbering ``deploy.py``'s
    ``GROUP``/``tile()`` macros use. ``None`` for a network's per-operation
    trace, which has no tile grid to index into (D48)."""
    grid: TileGrid | None = None
    """A lone matmul only: the decomposition this trace was built from (D53).
    ``grid.decode`` turns ``tile_start``/``tile_end`` into a (row, col) position
    and ``grid.row_dim``/``col_dim`` say which of M/N/K each axis is, so a
    caller holding just a ``Span`` (D48's shared hover text,
    ``bwz.figures``'s ``_tip``) can name the tile without importing the
    deployment's own geometry. Same object for every span of one trace: it
    replaces the bare ``tiles_per_ks`` width, which could only describe the
    weight-stationary grid."""

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
    work_by_op: tuple[tuple[str, float, float], ...] = ()
    """``(op_type, operations, compute seconds)`` per operator family, biggest
    first. Built from the operations themselves rather than from the drawn spans:
    coalescing 451 nodes into 57 blocks would otherwise report only whichever
    family dominates each block, and the mixture is the point — a decode step is
    matmul *and* attention *and* norms."""

    @property
    def coalesced(self) -> bool:
        return self.tiles > self.steps

    @property
    def totals(self) -> dict[Lane, float]:
        """The quantity each lane accounts for: bytes for DRAM, operations for the
        core, peak bytes held for SRAM (a stock, not a flow — it does not sum)."""
        out = {
            Lane.DRAM: sum(s.bytes_moved for s in self.spans),
            Lane.CORE: sum(s.flops for s in self.spans if s.lane is Lane.CORE),
            Lane.VECTOR: sum(s.flops for s in self.spans if s.lane is Lane.VECTOR),
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
        loads = (Stage.LOAD, Stage.LOAD_A)
        reads = sum(s.bytes_moved for s in self.spans if s.stage in loads)
        writes = sum(s.bytes_moved for s in self.spans if s.stage is Stage.STORE)
        return reads, writes

    @property
    def operand_bytes(self) -> tuple[float, float]:
        """``(B, A)`` read over the DRAM lane — the stationary operand and the
        streaming one, kept apart because they spill at different times."""
        weights = sum(s.bytes_moved for s in self.spans if s.stage is Stage.LOAD)
        activations = sum(s.bytes_moved for s in self.spans if s.stage is Stage.LOAD_A)
        return weights, activations

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


def grid_of(op: Operation, machine: MachineModel) -> TileGrid | None:
    """The tile grid this machine's stationarity implies for *op* (D53).

    ``None`` when there is nothing to tile — a non-matmul, or a profile that
    declares no array geometry. Every caller that needs to know how the work is
    cut up goes through here, so the trace, the utilisation term, the listing
    and the renderers cannot disagree about the decomposition.
    """
    if not isinstance(op.attrs, MatmulAttrs):
        return None
    dims = machine.unit.systolic_dims
    if dims is None:
        return None
    rows, cols = dims
    return grid_for(machine.stationarity, op.attrs, rows, cols, k_partitions=machine.k_partitions)


def tile_count(op: Operation, machine: MachineModel) -> int:
    """Independent tiles a matmul walks, per the machine's stationarity (D53).

    Weight-stationary walks ``ceil(K/rows) * ceil(N/cols)`` weight tiles;
    output-stationary walks ``ceil(M/rows) * ceil(N/cols)`` result tiles and
    sweeps K inside each. The same decomposition ``systolic_utilisation``
    costs, so a trace built from it cannot disagree with the utilisation the
    report quotes. Returns 1 when there is no array geometry to tile against.
    """
    grid = grid_of(op, machine)
    return grid.tiles if grid is not None else 1


def tiles_per_a_event(op: Operation, machine: MachineModel) -> int:
    """Tiles one A staging event serves: the width of a grid row (D33/D53).

    ``ceil(N/cols)`` under every stationarity whose column axis is N — which is
    all of them but ``is`` — so this is the same number D33 called
    ``NTILES_PER_KS``, generalised to grids whose rows are bands of M rather
    than slices of K. The same divisor :func:`_tile_trace` and :mod:`bwz.deploy`
    use, so the dataflow strategies cannot compute a different figure than the
    schedule and the listing draw. Returns 1 without declared array geometry,
    where there is no tile structure to speak of.
    """
    grid = grid_of(op, machine)
    return grid.cols if grid is not None else 1


def _resident_tile_bytes(
    grid: TileGrid | None, result: OpResult, dataflow: DataflowPlan | None
) -> float:
    """Bytes one tile of the resident operand occupies on chip (D53).

    Derived from the traffic the report already charged rather than asserted:
    the operand's own total, divided by the number of grid *cells*. Cells, not
    tiles — under split-K each partition holds a full-size partial of the same
    cell, so dividing by ``grid.tiles`` would shrink the buffer as split-K grew
    it. ``dataflow.a_bytes`` is preferred over the charged activation read for
    an A-resident grid, since ``stream`` inflates the latter by re-reads (D31)
    and a buffer holds one copy however many times it is filled.
    """
    if grid is None:
        return result.weight_bytes
    cells = max(1, grid.rows * grid.cols)
    if grid.resident is Operand.B:
        total = result.weight_bytes
    elif grid.resident is Operand.C:
        total = result.dram_write_bytes
    else:
        total = dataflow.a_bytes if dataflow is not None else result.dram_activation_read_bytes
    return total / cells


def _reduction_spans(
    result: OpResult,
    phase: GraphPhase,
    op_type: str,
    grid: TileGrid | None,
    *,
    after: list[Span],
) -> list[Span]:
    """The reduction, drawn where its placement puts it (D53/D62).

    Two shapes, and the difference is the whole of D62. A **DRAM** reduction is
    a second kernel: the partials' round trip and the additions both start where
    the tile schedule ends, because the GEMM has to have finished everywhere
    first. An **on-chip** reduction pipelines against the GEMM instead — the
    vector unit sums slice *n* while the matrix engine builds slice *n+1* — so
    its span starts at the first wave's end and runs *alongside* the core lane.
    Drawing it as a tail would contradict the ``max(matrix, vector)`` the report
    charged.

    Empty when nothing is reduced, which keeps the ordinary trace as it was. The
    DRAM span's duration is the same seconds-per-byte the tile spans were scaled
    by, so the lane still sums to ``t_dram`` exactly.
    """
    if grid is None:
        return []
    if result.reduction_placement is ReductionPlacement.ON_CHIP:
        return _overlapped_reduction_spans(result, phase, op_type, grid, alongside=after)
    if result.dram_reduction_bytes <= 0:
        return []
    start = max((span.end_s for span in after), default=0.0)
    per_byte = result.t_dram_s / result.dram_bytes if result.dram_bytes > 0 else 0.0
    dram_s = result.dram_reduction_bytes * per_byte
    partials = grid.k_partitions
    spans = [
        Span(
            Lane.DRAM,
            Stage.REDUCE,
            f"{partials} split-K partials written, then read back",
            start,
            start + dram_s,
            -2,
            phase,
            op_type=op_type,
            bytes_moved=result.dram_reduction_bytes,
            grid=grid,
        )
    ]
    if result.t_reduce_s > 0:
        spans.append(
            Span(
                Lane.VECTOR,
                Stage.REDUCE,
                f"batched reduction: {partials - 1} x M x N adds into C",
                start,
                start + result.t_reduce_s,
                -2,
                phase,
                op_type=op_type,
                flops=(partials - 1) * float(grid.m) * float(grid.n),
                grid=grid,
            )
        )
    return spans


def _overlapped_reduction_spans(
    result: OpResult,
    phase: GraphPhase,
    op_type: str,
    grid: TileGrid,
    *,
    alongside: list[Span],
) -> list[Span]:
    """The vector unit's share of a reduction that stays on chip (D62).

    One span, on the vector lane, starting when the first wave has produced
    something to add and running for the ``t_reduce_s`` the report charged. It
    overlaps the core lane deliberately: that overlap **is** the model — compute
    is ``max(matrix, vector)``, and a picture that queued the adds behind the
    arithmetic would draw a cost the report did not charge.

    No DRAM span, because no bytes are charged: the partials cross on-chip
    memory, which the v1 machine has no bandwidth term for (D5a), and that is
    named in ``report.assumptions`` rather than drawn as free traffic.
    """
    if result.t_reduce_s <= 0:
        return []
    executes = sorted(
        (span for span in alongside if span.stage is Stage.EXEC), key=lambda s: s.end_s
    )
    start = executes[0].end_s if executes else 0.0
    slices = grid.k_slices
    return [
        Span(
            Lane.VECTOR,
            Stage.REDUCE,
            f"{slices - 1} x M x N adds, one k-slice behind the array",
            start,
            start + result.t_reduce_s,
            -2,
            phase,
            op_type=op_type,
            flops=(slices - 1) * float(grid.m) * float(grid.n),
            grid=grid,
        )
    ]


def _engine_work(group: list[OpResult], *, matrix: bool) -> tuple[float, float, str]:
    """``(seconds, operations, dominant family)`` for one engine's share of *group*.

    The family is picked from the operations that engine actually ran, so a
    coalesced block reports ``matmul`` on the array and ``norm`` on the vector
    unit rather than ``matmul`` on both. Returns an empty name when the engine
    did nothing in this group, and the caller then emits no span for it.
    """
    mine = [r for r in group if (r.op_type in MATRIX_OP_TYPES) is matrix]
    if not mine:
        return 0.0, 0.0, ""
    return (
        sum(r.t_compute_s for r in mine),
        sum(r.flops for r in mine),
        max(mine, key=lambda r: r.flops).op_type.value,
    )


def _work_by_op(results: tuple[OpResult, ...]) -> tuple[tuple[str, float, float], ...]:
    """Arithmetic and compute time per operator family, biggest first."""
    totals: dict[str, list[float]] = {}
    for result in results:
        entry = totals.setdefault(result.op_type.value, [0.0, 0.0])
        entry[0] += result.flops
        entry[1] += result.t_compute_s
    return tuple(
        sorted(
            ((name, flops, seconds) for name, (flops, seconds) in totals.items()),
            key=lambda row: row[1],
            reverse=True,
        )
    )


def build_trace(
    graph: ComputeGraph,
    phase: PhaseResult,
    machine: MachineModel,
    *,
    double_buffered: bool,
    max_steps: int = MAX_DRAWN_STEPS,
    dataflow: DataflowPlan | None = None,
) -> PipelineTrace:
    """Decompose *phase* into resource spans.

    A single-matmul graph is decomposed into tile steps; anything else is
    decomposed into one step per operation, which is the granularity at which the
    engine actually models a network (no cross-operation overlap, D5a).

    *dataflow* is the plan ``analysis.dataflow.plan_dataflow`` already resolved
    for this op — the same one that decided the bytes ``phase`` carries — so the
    schedule this draws cannot disagree with the traffic it charged. ``None``
    (the default: no caller has to know about the dataflow strategies) draws
    exactly what a matmul always drew, ``a_strategy=stage``."""
    if len(graph.ops) == 1 and isinstance(graph.ops[0].attrs, MatmulAttrs):
        return _tile_trace(
            graph.ops[0],
            phase,
            machine,
            double_buffered=double_buffered,
            max_steps=max_steps,
            dataflow=dataflow,
        )
    return _operation_trace(graph, phase, double_buffered=double_buffered, max_steps=max_steps)


def _tile_trace(
    op: Operation,
    phase: PhaseResult,
    machine: MachineModel,
    *,
    double_buffered: bool,
    max_steps: int,
    dataflow: DataflowPlan | None = None,
) -> PipelineTrace:
    result = phase.ops[0]
    grid = grid_of(op, machine)
    tiles = grid.tiles if grid is not None else 1
    # A step is a WAVE, not a tile. The chip has `units` arrays and runs that
    # many weight tiles at once, so drawing one bar per tile showed a 4-core NPU
    # chewing through four tiles in series when it does all four together — a
    # picture that contradicted the utilisation the same report quotes (D30).
    units = max(machine.unit.count, 1)
    waves = math.ceil(tiles / units) if tiles else 1
    in_flight = min(tiles, units)
    steps = min(waves, max(1, max_steps))
    per_step = waves / steps

    # Split the DRAM time by direction, and the read by operand: B is the
    # stationary operand, A streams through. They obey different residency
    # fractions and spill at different times (D15), so they get their own bars.
    traffic = result.dram_bytes or 1.0
    scale = result.t_dram_s / traffic / steps
    load_b = result.dram_weight_read_bytes * scale
    store = result.dram_write_bytes * scale
    execute = result.t_compute_s / steps
    attrs = op.attrs
    assert isinstance(attrs, MatmulAttrs)
    dims = machine.unit.systolic_dims
    shape = f"{dims[0]}x{dims[1]}" if dims is not None else "untiled"
    a_strategy = dataflow.a_strategy if dataflow is not None else AStrategy.STAGE
    # The real, row-major tile-index range [open_tile, end_tile) each drawn step
    # covers — independent of a_strategy, so computed once here and reused both
    # for A's staging bookkeeping below and to tag every span this step produces
    # with the tiles it actually represents (D48). Capped at the real `tiles`
    # count, not waves * units (the array's theoretical capacity): when tiles
    # doesn't divide evenly into units, the last wave leaves some array slots
    # idle, and an idle slot is not a tile any span should claim (D46).
    tile_ranges: list[tuple[int, int]] | None = None
    if grid is not None:
        tile_ranges = [
            (math.floor(i * per_step * units), min(tiles, math.floor((i + 1) * per_step * units)))
            for i in range(steps)
        ]
    # D33, generalised by D53: under stage/whole, A is not a stream — one grid
    # ROW's tiles all read the same slice of A, staged once. Concentrate A's
    # DRAM time into one event per grid row, at the step that opens it, instead
    # of a per-wave trickle that reads as a re-read. Tiles are row-major, so
    # tile t belongs to the row `grid.cols` divides it into. What that row *is*
    # depends on the stationarity: a k-slice under `ws` (D33's own case), a band
    # of M rows under `os`/`rs`/`is`. Either way A crosses DRAM exactly once, in
    # `grid.a_events` pieces. Under stream (D31) or without declared geometry,
    # the honest picture *is* the per-wave trickle: `result` already carries the
    # inflated bytes a per-tile re-fetch costs (analysis/dataflow.py), and this
    # function only has to schedule what it is given.
    ramp_s = 0.0
    a_events = 1
    if grid is not None and a_strategy is not AStrategy.STREAM:
        assert tile_ranges is not None
        a_events = grid.a_events
        a_bytes_step: list[float] = []
        # (first, last) 1-based grid row *completed* this step, or None if this
        # step completes none. `openings = end//w - start//w` is the number of
        # rows this step finishes — consecutive steps' [start//w, end//w)
        # windows partition [0, a_events) exactly, with no gap and no overlap
        # (start_{i+1} = end_i, so last_i = end_i//w - 1 = start_{i+1}//w =
        # first_{i+1} - 1) — so the label's range must be *this same* window,
        # not the block merely *touched* by the step's last tile
        # (`(end_tile - 1) // grid.cols`), which straddles into whichever
        # later step actually finishes it: that block would then be named by
        # two consecutive steps' labels while its bytes were only ever
        # charged to the second, undercounting this step's own label by
        # exactly the bytes of the one block it doesn't yet own.
        # Per-row byte weight, not a uniform total/a_events average: neither M
        # nor K need divide evenly by rows (188 slices of 16 rows is 3008, not a
        # 3000-wide K), so the *last* band is narrower than the rest. A uniform
        # average charges every band the same ~31.9 kB regardless, silently
        # under-charging the 187 full-width bands and over-charging the ragged
        # last one — small in total, but a step naming three full bands must
        # charge exactly 3 x (1000 rows x 16 cols), not three shares of a
        # fleet-wide average. `grid.a_event_elements` is what knows the shape.
        bytes_per_element = result.dram_activation_read_bytes / (attrs.m * attrs.k)
        ks_opened: list[tuple[int, int] | None] = []
        for open_tile, end_tile in tile_ranges:
            first_g0 = open_tile // grid.cols
            last_g0 = end_tile // grid.cols - 1
            if last_g0 >= first_g0:
                a_bytes_step.append(
                    sum(
                        grid.a_event_elements(g) * bytes_per_element
                        for g in range(first_g0, last_g0 + 1)
                    )
                )
                ks_opened.append((first_g0 + 1, last_g0 + 1))
            else:
                a_bytes_step.append(0.0)
                ks_opened.append(None)
        if a_strategy is AStrategy.WHOLE:
            # Same total bytes as stage (D33) — only the timing changes: every
            # band ramps in before wave 0 instead of landing at the wave that
            # opens it. Zero the per-step shares here; the ramp itself is a
            # single span prepended after `_pipelined_tiles` returns, and every
            # other span shifts to start after it (below).
            ramp_s = sum(a_bytes_step) * scale * steps
            load_a = [0.0] * steps
            activation_bytes = [0.0] * steps
            ks_opened = [None] * steps
        else:
            load_a = [b * scale * steps for b in a_bytes_step]
            activation_bytes = a_bytes_step
    else:
        load_a = [result.dram_activation_read_bytes * scale] * steps
        activation_bytes = [result.dram_activation_read_bytes / steps] * steps
        ks_opened = [None] * steps
    tiles_here = in_flight * per_step
    # Two spaces separate the bar text from the qualifier: `_short` in the plot
    # script splits there, so the bar stays legible and the hover keeps it all.
    # The tile is named after the operand that stays resident in it — B under
    # weight-stationary, C's accumulator under output-stationary (D53) — because
    # that is what the grid is a grid *of*.
    resident = grid.resident.value if grid is not None else "B"
    if tiles_here == 1:
        label, qualifier = f"{resident} tile {shape}", ""
    elif per_step > 1:
        # One bar coalesces several waves, so "in parallel" would overstate it:
        # this many tiles pass through, `in_flight` of them at any instant.
        label = f"{tiles_here:.0f} {resident} tiles {shape}"
        qualifier = f"{in_flight} at a time"
    elif units > 1:
        label, qualifier = f"{tiles_here:.0f} {resident} tiles {shape}", "all in parallel"
    else:
        label, qualifier = f"{tiles_here:.0f} {resident} tiles {shape}", ""
    tail = f"  {qualifier}" if qualifier else ""
    labels = [f"{label} [{i + 1}/{steps}]{tail}" for i in range(steps)]

    # The staging openings' A bars carry their own names under stage; the other
    # steps have no A traffic at all under D33, so nothing else needs one. Under
    # stream or whole there is no per-step opening to name — stream falls back
    # to the tile labels (D31's per-tile share), whole's A traffic is a single
    # ramp named separately below. The band is named by its own row axis: a
    # k-slice under ws, a row-band under os (D53).
    group = grid.group_name if grid is not None else "k-slice"
    band = f"{grid.a_event_shape[0]}x{grid.a_event_shape[1]}" if grid is not None else ""

    def _stage_label(opened: tuple[int, int] | None) -> str:
        if opened is None:
            return ""
        first, last = opened
        if first == last:
            return f"A {group} {first}/{a_events} ({band}) — staged once, feeds its tiles"
        return f"A {group}s {first}-{last}/{a_events} ({band}) — staged once, feed their tiles"

    activation_labels = (
        [_stage_label(g) for g in ks_opened] if a_strategy is AStrategy.STAGE else None
    )

    spans = _pipelined_tiles(
        [load_b] * steps,
        [execute] * steps,
        [store] * steps,
        labels=labels,
        op_types=[op.op_type.value] * steps,
        phase=phase.phase,
        dispatch_s=result.t_fixed_s,
        double_buffered=double_buffered,
        bytes_per_step=[result.dram_weight_read_bytes / steps] * steps,
        stored_per_step=[result.dram_write_bytes / steps] * steps,
        flops_per_step=[result.flops / steps] * steps,
        # What one buffer holds: the resident operand's tiles the arrays work on
        # for this wave — B under weight-stationary, C's accumulator under
        # output-stationary (D53). Sized from the schedule, not asserted, and
        # divided by the grid's CELL count rather than its tile count: under
        # split-K every partition holds a full-size partial of the same cell.
        resident_per_step=[_resident_tile_bytes(grid, result, dataflow) * tiles_here] * steps,
        activation_loads=load_a,
        activation_bytes_per_step=activation_bytes,
        activation_labels=activation_labels,
        a_fetch_mode=a_strategy.value,
        depth_override=(dataflow.a_prefetch_depth if dataflow is not None else None),
        b_on_demand=(dataflow is not None and dataflow.b_dataflow is BDataflow.ON_DEMAND),
        tile_ranges=tile_ranges,
        grid=grid,
    )
    if a_strategy is AStrategy.WHOLE and ramp_s > 0:
        # Every k-slice staged before wave 0: one span for the whole ramp, and
        # the rest of the schedule — which has no A load left to place — shifts
        # to start after it. DRAM busy still sums to t_dram exactly: the ramp's
        # bytes are the same A total the per-k-slice events would have moved.
        ramp = Span(
            Lane.DRAM,
            Stage.LOAD_A,
            f"A staged whole  {a_events} {group}s before wave 0 ({band} each)",
            0.0,
            ramp_s,
            -1,
            phase.phase,
            op_type=op.op_type.value,
            bytes_moved=result.dram_activation_read_bytes,
            a_fetch_mode=AStrategy.WHOLE.value,
            tile_start=0,
            tile_end=tiles,
            grid=grid,
        )
        spans = [
            ramp,
            *(replace(s, start_s=s.start_s + ramp_s, end_s=s.end_s + ramp_s) for s in spans),
        ]
    spans.extend(_reduction_spans(result, phase.phase, op.op_type.value, grid, after=spans))
    return PipelineTrace(
        spans=tuple(spans),
        total_s=max((s.end_s for s in spans), default=0.0),
        reported_latency_s=result.latency_s,
        steps=steps,
        # The schedule's real step count is WAVES, not tiles: `units` arrays run
        # a wave together, so that is what `coalesced` must compare against.
        tiles=waves,
        double_buffered=double_buffered,
        # Measured, not derived: with stores in the schedule the fill/drain is a
        # load at the head plus a store at the tail, and a formula for it would
        # be one more thing to keep in step with the scheduler.
        fill_drain_s=max(0.0, max((s.end_s for s in spans), default=0.0) - result.latency_s),
        kind="tiles",
        work_by_op=_work_by_op((result,)),
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
    # A coalesced group mixes families, so the group's compute time is split by
    # engine rather than attributed to whichever family happens to dominate it.
    # Without this the vector lane came out empty on a model that spends real
    # time on norms and activations (D28).
    #
    # Each half also names *itself*. Naming both halves after the group's
    # dominant family — which is a matmul in essentially every group — labelled
    # the vector lane's spans `matmul`, so a norm bar on the vector unit hovered
    # as "EXEC — matmul on the vector unit". A matrix family cannot execute
    # there; that is the whole content of D27.
    matrix = [_engine_work(g, matrix=True) for g in groups]
    vector = [_engine_work(g, matrix=False) for g in groups]
    labels = [
        g[0].op_id if len(g) == 1 else f"{g[0].op_id} .. {g[-1].op_id}  ({len(g)} ops)"
        for g in groups
    ]
    # The family that dominates the group's arithmetic — a coalesced block is
    # usually a run of like operators, and naming the biggest is more honest than
    # naming the first.
    op_types = [max(g, key=lambda r: r.flops).op_type.value if g else "" for g in groups]

    # A group's duration is the SUM of its members' latencies, not the latency of
    # their summed terms: coalescing must not silently re-schedule the ops it
    # groups. max(sum L, sum C) <= sum max(L, C), so taking the former would draw
    # a picture faster than the report.
    # Dispatch is charged to the operation that pays it, so it is part of that
    # step's duration rather than a prologue (D25).
    fixed = [sum(r.t_fixed_s for r in g) for g in groups]
    durations = [
        fixed[index]
        + sum(
            max(r.t_dram_s, r.t_compute_s) if double_buffered else r.t_dram_s + r.t_compute_s
            for r in g
        )
        for index, g in enumerate(groups)
    ]
    spans = _serial_steps(
        loads,
        executes,
        durations=durations,
        labels=labels,
        op_types=op_types,
        matrix_per_step=matrix,
        vector_per_step=vector,
        phase=phase.phase,
        fixed_per_step=fixed,
        double_buffered=double_buffered,
        bytes_per_step=[sum(r.dram_read_bytes for r in g) for g in groups],
        weight_bytes_per_step=[sum(r.dram_weight_read_bytes for r in g) for g in groups],
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
        work_by_op=_work_by_op(tuple(results)),
    )


def _pipelined_tiles(
    loads: list[float],
    executes: list[float],
    stores: list[float],
    *,
    labels: list[str],
    op_types: list[str],
    phase: GraphPhase,
    dispatch_s: float,
    double_buffered: bool,
    bytes_per_step: list[float],
    stored_per_step: list[float],
    flops_per_step: list[float],
    resident_per_step: list[float],
    activation_loads: list[float] | None = None,
    activation_bytes_per_step: list[float] | None = None,
    activation_labels: list[str] | None = None,
    a_fetch_mode: str = "stream",
    depth_override: int | None = None,
    b_on_demand: bool = False,
    tile_ranges: list[tuple[int, int]] | None = None,
    grid: TileGrid | None = None,
) -> list[Span]:
    """Software-pipeline the tiles of ONE operation, per the constraints above.

    Legitimate here and only here: the tiles of a single matmul are issued by one
    kernel, which is exactly the thing a double buffer overlaps.

    ``loads`` is operand B's time and ``activation_loads`` operand A's. They
    queue on the same port, so the schedule uses their sum and the drawing keeps
    them as two adjacent bars. ``a_fetch_mode`` names which ``a_strategy`` these
    A bars are (D33/D31), and ``activation_labels`` names them; both default to
    the per-step leg of a stream.

    ``depth_override`` is ``--a-prefetch-depth``: how many steps back a buffer
    must free before its slot can be reused, in place of the depth on-chip
    capacity would derive (``2`` double buffered, ``1`` otherwise). Schedule-only
    — the loop below is depth-agnostic beyond ``i >= depth``, so any depth >= 1
    is valid, not just the two capacity ever produces.

    ``b_on_demand`` is ``--b-dataflow on-demand`` (D33/D40): B's load ordinarily
    prefetches as early as the buffer and the DRAM port allow (write-ahead,
    D33's default — hidden behind an earlier wave's compute). Setting this
    forces tile i's load to wait for tile i-1's compute to finish first,
    exposing the same, already-costed load duration on the critical path
    instead of hiding it. Schedule-only, like ``depth_override``: it never
    changes a byte count or the reported latency, only how much of the trace's
    span is fill/drain (D19). ``persistent`` gets no parameter here — D40 shows
    write-ahead's overlap is already this model's best case, so persistent's
    trace is write-ahead's, unmodified.

    ``tile_ranges[i]`` is the ``[start, end)`` global tile-index range step
    ``i`` covers (D48) — the same range for every span that step produces,
    since they all cover the same real tiles. ``None`` when the caller has no
    tile grid to index into (no declared systolic geometry). ``grid`` is the
    decomposition those indices are indices *into* (D53); every span carries it
    so a renderer can decode a tile position without re-deriving the geometry.
    """
    steps = len(loads)
    a_loads = activation_loads if activation_loads is not None else [0.0] * steps
    a_bytes = activation_bytes_per_step if activation_bytes_per_step is not None else [0.0] * steps
    a_labels = activation_labels if activation_labels is not None else labels
    ranges: list[tuple[int, int] | None] = (
        list(tile_ranges) if tile_ranges is not None else [None] * steps
    )
    spans: list[Span] = []
    if dispatch_s > 0:
        # Step -1: the dispatch is not a tile, and giving it step 0 would merge it
        # with the first tile's row.
        spans.append(Span(Lane.CORE, Stage.DISPATCH, "kernel dispatch", 0.0, dispatch_s, -1, phase))

    depth = depth_override if depth_override is not None else (2 if double_buffered else 1)
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
        # With a single buffer (depth 1, no double buffering) there is no slack
        # to reorder: tile i-1's store must be placed before tile i's load even
        # asks whether the buffer is free, because that load needs its answer.
        # Deferring it (the depth >= 2 branch below) would read store_ends one
        # entry short of what has actually been placed.
        if depth == 1 and i >= 1:
            place_store(i - 1)
        # A buffer is free once its tile has been *written out*, not merely
        # computed — the hold below runs to the store, so the reuse test must too.
        freed = store_ends[i - depth] if i >= depth else dispatch_s
        # on-demand (D40): B's load may not start until the *previous* tile's
        # compute has actually finished — no prefetch-ahead. write-ahead (the
        # default) omits this term, so `load_start` stays gated only on the
        # port and the buffer, exactly as it always has.
        on_demand_floor = exec_end if b_on_demand else 0.0
        load_start = max(dram_free, freed, on_demand_floor)
        load_end = load_start + loads[i] + a_loads[i]
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
        # quietly cancel the double buffer. Only meaningful with >= 2 buffers —
        # depth 1 has no next-fetch to prioritise over, and already placed this
        # store above.
        if depth >= 2 and i >= 1:
            place_store(i - 1)
    if steps:
        place_store(steps - 1)

    for i in range(steps):
        load_start, load_end = load_starts[i], load_ends[i]
        # Kept, not recomputed: `exec_ends[i] - executes[i]` can land a float
        # hair before the previous span's end and read as an overlap.
        exec_start, exec_end = exec_starts[i], exec_ends[i]
        store_start, store_end = store_starts[i], store_ends[i]
        step_range = ranges[i]
        tile_start, tile_end = step_range if step_range is not None else (None, None)

        # B first, then A, adjacent on the one port: the array cannot start
        # until its stationary operand has arrived.
        if loads[i] > 0:
            spans.append(
                Span(
                    Lane.DRAM,
                    Stage.LOAD,
                    labels[i],
                    load_start,
                    load_start + loads[i],
                    i,
                    phase,
                    op_type=op_types[i],
                    bytes_moved=bytes_per_step[i],
                    tile_start=tile_start,
                    tile_end=tile_end,
                    grid=grid,
                )
            )
        if a_loads[i] > 0:
            spans.append(
                Span(
                    Lane.DRAM,
                    Stage.LOAD_A,
                    a_labels[i],
                    load_start + loads[i],
                    load_end,
                    i,
                    phase,
                    op_type=op_types[i],
                    bytes_moved=a_bytes[i],
                    a_fetch_mode=a_fetch_mode,
                    tile_start=tile_start,
                    tile_end=tile_end,
                    grid=grid,
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
                op_type=op_types[i],
                resident_bytes=resident_per_step[i],
                tile_start=tile_start,
                tile_end=tile_end,
                grid=grid,
            )
        )
        if executes[i] > 0:
            spans.append(
                Span(
                    _exec_lane(op_types[i]),
                    Stage.EXEC,
                    labels[i],
                    exec_start,
                    exec_end,
                    i,
                    phase,
                    op_type=op_types[i],
                    flops=flops_per_step[i],
                    tile_start=tile_start,
                    tile_end=tile_end,
                    grid=grid,
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
                    op_type=op_types[i],
                    bytes_moved=stored_per_step[i],
                    tile_start=tile_start,
                    tile_end=tile_end,
                    grid=grid,
                )
            )
    return spans


def _serial_steps(
    loads: list[float],
    executes: list[float],
    *,
    durations: list[float],
    labels: list[str],
    op_types: list[str],
    matrix_per_step: list[tuple[float, float, str]],
    vector_per_step: list[tuple[float, float, str]],
    phase: GraphPhase,
    fixed_per_step: list[float],
    double_buffered: bool,
    bytes_per_step: list[float],
    weight_bytes_per_step: list[float],
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

    for i, (load, execute, duration) in enumerate(zip(loads, executes, durations, strict=True)):
        start = now
        end = start + duration
        # Dispatch belongs to the operation that pays it, not to a prologue.
        # Summed into one block at t=0 it made a launch-bound model draw as a
        # single bar with every operation invisible behind it (D25).
        if fixed_per_step[i] > 0:
            spans.append(
                Span(
                    Lane.CORE,
                    Stage.DISPATCH,
                    labels[i],
                    start,
                    start + fixed_per_step[i],
                    i,
                    phase,
                    op_type=op_types[i],
                )
            )
        start += fixed_per_step[i]
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
            # Split the load bar by operand in proportion to the bytes each
            # contributed, B first: the stationary operand has to land before
            # the array can start (D31).
            read = bytes_per_step[i] or 1.0
            b_share = weight_bytes_per_step[i] / read
            b_end = start + load_time * b_share
            if weight_bytes_per_step[i] > 0:
                spans.append(
                    Span(
                        Lane.DRAM,
                        Stage.LOAD,
                        labels[i],
                        start,
                        b_end,
                        i,
                        phase,
                        op_type=op_types[i],
                        bytes_moved=weight_bytes_per_step[i],
                    )
                )
            if bytes_per_step[i] - weight_bytes_per_step[i] > 0:
                spans.append(
                    Span(
                        Lane.DRAM,
                        Stage.LOAD_A,
                        labels[i],
                        b_end,
                        load_end,
                        i,
                        phase,
                        op_type=op_types[i],
                        bytes_moved=bytes_per_step[i] - weight_bytes_per_step[i],
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
                op_type=op_types[i],
                resident_bytes=resident_per_step[i],
            )
        )
        # One span per engine that did work in this step: the array, then the
        # vector unit that follows it. Two bars rather than one is the point —
        # they are different silicon and a reader should see which was busy.
        cursor = exec_start
        for lane, (seconds, work, family) in (
            (Lane.CORE, matrix_per_step[i]),
            (Lane.VECTOR, vector_per_step[i]),
        ):
            if seconds <= 0:
                continue
            spans.append(
                Span(
                    lane,
                    Stage.EXEC,
                    labels[i],
                    cursor,
                    cursor + seconds,
                    i,
                    phase,
                    # This engine's own dominant family, never the group's.
                    op_type=family,
                    flops=work,
                )
            )
            cursor += seconds
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
                    op_type=op_types[i],
                    bytes_moved=stored_per_step[i],
                )
            )
    return spans


def _exec_lane(op_type: str) -> Lane:
    """Which engine executes this family. Matrix work goes to the array; norms,
    activations and residuals go to the vector unit (D27)."""
    return Lane.CORE if op_type in {t.value for t in MATRIX_OP_TYPES} else Lane.VECTOR


def _chunk(items: list[OpResult], groups: int) -> list[list[OpResult]]:
    """Split *items* into *groups* contiguous chunks of near-equal size."""
    size = math.ceil(len(items) / groups)
    return [items[i : i + size] for i in range(0, len(items), size)]
