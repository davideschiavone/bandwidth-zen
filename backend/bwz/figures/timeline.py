"""Where the time went, on which piece of hardware — one self-contained, zoomable
HTML page per chip, or one page for two chips.

    uv run --group plots python scripts/plot_pipeline.py --chip a100_80gb
    uv run --group plots python scripts/plot_pipeline.py \\
        --chip a100_80gb --chip metis_aipu --compare --model gemma3_4b -S 512

**Rows are resources, not steps.** A per-instruction view — one row per tile step,
stages within the row — answers "what happened to this tile" and not "what was the
memory system doing while the array worked". For comparing two architectures only
the second question matters, so this figure gives every declared memory level and
every declared compute unit its own row and writes the quantity beside it: bytes
moved and at what rate, operations retired and at what fraction of peak, how much
the buffers hold.

Rows come from the chip profile, so the picture shows what the machine *has*, and
a resource the v1 model does not use is drawn grey rather than quietly omitted.
A100 declares L1, L2 and HBM plus tensor and CUDA cores; only HBM (bandwidth), L1
(capacity) and the tensor cores carry anything here, and the grey rows are
exactly where the model's boundary lies.

One register: the whole run, at ``--steps`` resolution (default 256). There is no
separate zoomed register — the page itself zooms (wheel, about the cursor) and
pans (drag) — so a 65 536-tile matmul's first tile is a scroll away rather than a
second figure.

**``--compare`` puts every ``--chip`` in one page** (docs/CORRECTIONS.md D29).
Three things change and nothing else does:

1. The x axis becomes **shared and absolute** instead of normalised per chip, so a
   bar three times as long took three times as long. That is the entire point, and
   it is why the per-chip normalised view is *kept* rather than replaced: it is
   still the better view of a single machine's internal balance.
2. Rows are **banded by chip**, because two profiles declare different numbers of
   memory levels and compute units and no correspondence between them exists to
   draw. Each band is introduced by a header row carrying the machine's headline
   numbers, its row counts and its total span.
3. A **roofline register** is added below, both chips' ceilings on one chart, since
   the timeline shows what happened and the roofline shows why it had to.

Every chip runs the *same* workload at the *same* precision — a comparison across
two precisions would be comparing two different amounts of traffic — so
``--compare`` insists on a dtype every chip supports and says so if there is none.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import bwz
from bwz.analysis import idealised, machine_model
from bwz.analysis.dataflow import DataflowPlan, plan_dataflow
from bwz.analysis.pipeline import Lane, PipelineTrace, Span, Stage, build_trace, grid_of
from bwz.analysis.roofline import MATRIX_OP_TYPES, MachineModel, compute_dtype
from bwz.analysis.stationarity import Dim, TileGrid, residency_phrase
from bwz.analysis.tiling import operation_cores
from bwz.deploy import check as check_deployment
from bwz.deploy import deployment_of
from bwz.emit import EmittedProgram, emit_matmul
from bwz.emit import check as check_program
from bwz.explain import Explanation, explain_graph
from bwz.figures.dataflow_html import render as render_animation
from bwz.figures.timeline_html import Box, render
from bwz.graph import GraphPhase, build_graph, build_graphs
from bwz.graph.ops import Operation
from bwz.operators.base import cost_of
from bwz.report import Bound, PhaseResult, ReductionPlacement, Report
from bwz.spec import AnyModelSpec, DeploymentSpec, DType, HardwareSpec, MatmulSpec
from bwz.units import format_bandwidth, format_bytes, format_quantity, format_time

# Chip identity, used only where two machines share one chart: the band rules on
# a comparison timeline and the roofs on a comparison roofline. Slots 4 and 5 of
# the documented categorical palette, deliberately *not* the three lane hues
# above — a bar's colour must keep meaning "which resource", so "which chip" gets
# its own pair, and every mark carrying one is also directly labelled.
CHIP_COLOURS = ["#5b53c9", "#c2185b", "#0f766e", "#a16207"]


@dataclass(frozen=True)
class Workload:
    """What was run, and the four numbers the roofline needs to place it."""

    name: str
    trace: PipelineTrace
    explanations: tuple[Explanation, ...]
    flops: float
    dram_bytes: float
    latency_s: float
    bound: Bound
    phase: PhaseResult | None = None
    """The phase this trace decomposes; the deployment listing reads its bytes."""
    operation: Operation | None = None
    """The single matmul, when there is one, so the listing can quote the same
    tile count the schedule and the utilisation model both divide by."""
    dataflow: DataflowPlan | None = None
    """The effective A/B dataflow strategy for a lone matmul — already clamped by
    ``analysis.dataflow.plan_dataflow``, the same plan that decided the bytes this
    trace draws — so the deployment listing cannot render a strategy the schedule
    above it did not actually run."""
    machine: MachineModel | None = None
    """The machine this workload was analysed with, carrying the effective
    stationarity and split-K (D53). ``None`` where the defaults were used and a
    freshly derived model is identical."""
    units_used: int = 0
    """Arrays of the matrix engine that ever receive a tile in this workload —
    ``min(count, tiles)``, maximised over its operations (D30). **Not** the
    declared count and not wave occupancy: a two-tile matmul occupies two of
    A100's 432 tensor cores and the other 430 never start, which is what a
    resource row has to say rather than quoting the datasheet's 432. 0 where it
    was not computed."""
    program: EmittedProgram | None = None
    """The runnable version of this decomposition (D54), when ``--emit`` asked
    for one. Built here rather than in the page so it comes from the same
    ``Report``, grid and ``DataflowPlan`` the timeline is drawn from — a program
    walking a different decomposition than the picture beside it would be worse
    than no program at all."""


@dataclass(frozen=True)
class Panel:
    """One chip's run of the workload — the unit a comparison repeats.

    A single-chip figure is a comparison of one, so there is one code path and
    the per-chip view cannot drift from the compared one.
    """

    chip: HardwareSpec
    dtype: DType
    work: Workload

    @property
    def machine(self) -> MachineModel:
        """The machine the workload was analysed with.

        Deriving a fresh one here would silently drop the stationarity and
        split-K the run actually used, and the listing would then describe a
        decomposition the timeline above it is not of (D53).
        """
        return self.work.machine or machine_model(self.chip, self.dtype)

    @property
    def colour(self) -> str:
        return CHIP_COLOURS[0]


@dataclass(frozen=True)
class Row:
    """One hardware resource, drawn whether or not this workload touches it."""

    title: str
    detail: str
    lane: Lane | None
    """The trace lane whose spans belong on this row, or None for a resource the
    model declares and never uses — the rows that show where it stops."""
    note: str = ""
    panel: int = 0
    """Which chip's trace this row reads. Always 0 on a single-chip figure."""
    header: bool = False
    """A band header naming the chip whose rows follow, rather than a resource.
    Two profiles declare different numbers of memory levels and compute units, so
    the rows cannot line up and are banded instead (docs/CORRECTIONS.md D29)."""
    quantity: str = ""
    """Overrides the computed right-hand figure. Set on header rows, whose
    quantity is the chip's whole span rather than one resource's share."""
    colour: str = ""


def rows_for(panel: Panel) -> list[Row]:
    """Resource rows: what the profile declares, and how much of it this run uses.

    The v1 machine is three elements (``docs/CORRECTIONS.md`` D5a): the deepest
    memory level supplies bandwidth, the shallowest supplies capacity, the
    fastest compute unit supplies TOPS. Everything else a profile declares is
    drawn idle with the reason, which is more honest than leaving it out — an
    A100 has 40 MB of L2 this model never spends, and that omission is worth
    seeing next to a chip whose SRAM is the whole story.

    A compute row names the arrays **this workload uses**, not the count on the
    datasheet. "432 x 16x16 array" is what the chip has; a 17-cubed matmul runs
    on two of them and the picture said 432, which is the number a reader would
    have divided by. The panel is taken whole rather than a chip and a dtype so
    that the row can ask its trace what happened.
    """
    chip, dtype = panel.chip, panel.dtype
    rows: list[Row] = []
    deepest = chip.memory[-1]
    shallowest = chip.memory[0]
    machine = machine_model(chip, dtype)

    for level in reversed(chip.memory):
        detail = (
            f"{format_bytes(level.capacity_bytes)} · "
            f"{format_bandwidth(level.bandwidth_bytes_per_s)}"
        )
        if level is deepest:
            rows.append(Row(level.name, detail, Lane.DRAM, "the only bandwidth ceiling (D5a)"))
        elif level is shallowest:
            rows.append(
                Row(level.name, detail, Lane.SRAM, "capacity only — no bandwidth term (D5b)")
            )
        else:
            rows.append(
                Row(level.name, detail, None, "declared, not modelled — the roofline is flat (D5)")
            )

    vector_busy = panel.work.trace.busy_s[Lane.VECTOR] > 0
    for unit in chip.compute_units:
        geometry = (
            f"{unit.systolic_dims[0]}x{unit.systolic_dims[1]} array"
            if unit.systolic_dims
            else f"{unit.ops_per_cycle_per_unit:g} MAC/cycle"
        )
        peak = unit.peak_flops_per_s(chip.clock_hz, dtype) if unit.supports(dtype) else 0.0
        rate = f"peak {format_quantity(peak, 'OP/s')} at {dtype.value}"
        if unit is machine.unit:
            # min(count, tiles): the arrays that ever hold one. Distinct from
            # wave occupancy, which averages over the run and would report the
            # same 0.5% whether one array worked or all 432 half-worked (D30).
            used = min(panel.work.units_used or unit.count, unit.count)
            idle = unit.count - used
            why = (
                f" · the grid has {used} tile{'s' if used != 1 else ''} at its widest, so "
                f"{idle} of these arrays never start (D30)"
                if idle > 0
                else " · every array gets a tile"
            )
            rows.append(
                Row(
                    unit.name,
                    _used_of(used, unit.count, geometry),
                    Lane.CORE,
                    f"matrix work — {rate}{why}",
                )
            )
        elif unit is machine.vector_unit:
            # A tensor core does matrix-multiply-accumulate and nothing else, so
            # norms, activations, residuals and any reduction have their own row
            # on their own silicon (D27/D28/D62). How much of that unit runs them
            # is not modelled: the cost is charged at the whole unit's rate, so
            # the row says all of it is engaged and says that it assumed so.
            rows.append(
                Row(
                    unit.name,
                    _used_of(unit.count if vector_busy else 0, unit.count, geometry),
                    Lane.VECTOR,
                    f"norms, activations, reductions — {rate}"
                    + (
                        " · charged at the whole unit's rate, so all of it is assumed engaged"
                        if vector_busy
                        else " · no vector work in this graph"
                    ),
                )
            )
        elif unit.supports(dtype):
            rows.append(
                Row(
                    unit.name,
                    _used_of(0, unit.count, geometry),
                    None,
                    "idle — no work of its kind in this graph",
                )
            )
        else:
            rows.append(
                Row(
                    unit.name,
                    _used_of(0, unit.count, geometry),
                    None,
                    f"idle — no {dtype.value} datapath",
                )
            )
    return rows


def _used_of(used: int, count: int, geometry: str) -> str:
    """``"2 of 432 x 16x16 array"`` — what this run uses, of what there is.

    Both numbers, always, including when they are equal: a row reading "432 x
    16x16 array" is a statement about the datasheet, and the question a reader
    brings to a resource row is how much of it the workload reached.
    """
    return f"{used} of {count} x {geometry}"


def panel_rows(panels: list[Panel]) -> list[Row]:
    """Every panel's resource rows, banded by chip when there is more than one.

    The rows of two chips cannot be aligned — A100 declares three memory levels
    and two compute units, Metis four and two, ``chip_a`` two and one — so no
    correspondence is invented. Each chip keeps its own band, introduced by a
    header row carrying the machine's headline numbers and its total span, and
    the shared thing is the **time axis** rather than the rows
    (docs/CORRECTIONS.md D29).
    """
    banded = len(panels) > 1
    rows: list[Row] = []
    for index, panel in enumerate(panels):
        if banded:
            machine = machine_model(panel.chip, panel.dtype)
            phase = panel.work.phase
            assert phase is not None, "every workload this script builds carries its phase"
            rows.append(
                Row(
                    panel.chip.name,
                    f"{format_quantity(machine.peak_flops_per_s, 'OP/s')} {panel.dtype.value} · "
                    f"{format_bandwidth(panel.chip.dram.bandwidth_bytes_per_s)} "
                    f"{panel.chip.dram.name}",
                    None,
                    # The row counts, said out loud: this is why the bands exist
                    # rather than one shared row list (D29).
                    note=(
                        f"{format_bytes(panel.chip.on_chip_capacity_bytes)} on chip · "
                        f"{len(panel.chip.memory)} memory levels, "
                        f"{len(panel.chip.compute_units)} engines"
                    ),
                    panel=index,
                    header=True,
                    quantity=(
                        # Reported latency, not the drawn span: the bars run past
                        # it by the fill/drain the roofline omits, and a band
                        # header quoting the longer number would disagree with
                        # the report (D35). Achieved throughput is the other
                        # number D35 requires the same discipline of: it comes
                        # from this same reported latency, not the drawn span,
                        # so it is the rate the chip *delivered* on this
                        # workload — not the rate the array ran at while busy
                        # (that number is in the COMPUTED box below, labelled
                        # apart so the two are never read as the same claim).
                        f"{format_time(panel.work.trace.reported_latency_s)} · "
                        f"{panel.work.bound.value.replace('_', ' ').lower()}\n"
                        f"{format_quantity(phase.achieved_flops_per_s, 'OP/s')} achieved · "
                        f"{phase.utilization:.0%} of peak"
                    ),
                    colour=CHIP_COLOURS[index % len(CHIP_COLOURS)],
                )
            )
        for row in rows_for(panel):
            rows.append(replace(row, panel=index))
    return rows


def _op_mix(trace: PipelineTrace, *, matrix: bool | None = None) -> str:
    """The operator families that did the arithmetic, biggest first.

    "137 GOP" does not say whether that was one matmul or a decode step's worth of
    matmul, attention and norms, and for a comparison the mixture is the point.
    """
    families = {t.value for t in MATRIX_OP_TYPES}
    rows = [
        (name, flops)
        for name, flops, _ in trace.work_by_op
        if flops > 0 and (matrix is None or (name in families) is matrix)
    ]
    if not rows:
        return "no arithmetic"
    total = sum(flops for _, flops in rows) or 1.0
    shown = [f"{name} {flops / total:.0%}" for name, flops in rows[:3] if flops / total >= 0.005]
    if len(rows) == 1:
        return rows[0][0]
    return " · ".join(shown) if shown else rows[0][0]


def _boxes(panels: list[Panel]) -> list[Box]:
    """The three headline figures per chip, shared by the PNG and the HTML.

    Banded rather than merged: a "MOVED OVER DRAM" box that tried to carry two
    chips at once would have to pick one headline number, and which chip moved
    how much is exactly the thing being compared.
    """
    banded = len(panels) > 1
    out: list[Box] = []
    for panel in panels:
        trace, chip, dtype = panel.work.trace, panel.chip, panel.dtype
        phase = panel.work.phase
        assert phase is not None, "every workload this script builds carries its phase"
        totals, busy, concurrency = trace.totals, trace.busy_s, trace.concurrency
        span = trace.total_s or 1.0
        machine = machine_model(chip, dtype)
        peak = machine.peak_flops_per_s
        # The vector engine has a peak too, and without it "2.11 GOP @ 13.6 TOP/s"
        # is a rate with no denominator — you cannot tell 70% of the vector unit
        # from 4% of it. A profile that declares no vector unit runs this work on
        # the array, and the box says so rather than printing the array's peak
        # twice as though it were a second engine (D27).
        vector_peak = machine.vector_unit.peak_flops_per_s(chip.clock_hz, dtype)
        of_vector = (
            f" of {format_quantity(vector_peak, 'OP/s')}"
            if machine.has_vector_unit
            else f"\nno vector unit at {dtype.value} — charged to the array"
        )
        buffers = max(concurrency[Lane.SRAM][1], 1)
        rate = format_bandwidth(totals[Lane.DRAM] / busy[Lane.DRAM]) if busy[Lane.DRAM] else "—"
        core_rate = totals[Lane.CORE] / busy[Lane.CORE] if busy[Lane.CORE] else 0.0
        vector_rate = totals[Lane.VECTOR] / busy[Lane.VECTOR] if busy[Lane.VECTOR] else 0.0
        band = f"{chip.name} — {format_time(trace.reported_latency_s)}" if banded else ""
        out += [
            Box(
                "dram",
                "MOVED OVER DRAM",
                format_bytes(totals[Lane.DRAM]),
                # `B=` rather than `B `: the operand's name and the byte unit are
                # the same letter, so "LOAD B 0 B · A 512 B" read as an imperative
                # ("load B") followed by an unparseable "0 B · A". The equals sign
                # is what makes it a label.
                f"LOAD B={format_bytes(trace.operand_bytes[0])} · "
                f"A={format_bytes(trace.operand_bytes[1])}\n"
                f"STORE C={format_bytes(trace.direction_bytes[1])} · {rate} while active\n"
                f"{format_time(busy[Lane.DRAM])} — {busy[Lane.DRAM] / span:.0%} of the span",
                band=band,
            ),
            Box(
                "sram",
                "HELD ON CHIP",
                format_bytes(totals[Lane.SRAM]),
                f"{buffers} buffer{'s' if buffers != 1 else ''} x "
                f"{format_bytes(totals[Lane.SRAM] / buffers)}\n"
                f"of {format_bytes(chip.on_chip_capacity_bytes)} capacity",
                band=band,
            ),
            Box(
                "core",
                f"COMPUTED — {_op_mix(trace)}",
                format_quantity(totals[Lane.CORE] + totals[Lane.VECTOR], "OP"),
                # Achieved throughput first, and from phase.achieved_flops_per_s —
                # derived from the REPORTED latency (D35), never the drawn span —
                # because it is the number that makes two chips comparable
                # independently of workload size: "the chip delivered N TOP/s on
                # this workload". `core_rate` below is a different claim, `totals
                # / busy` — the rate while the array specifically was busy, which
                # can exceed the achieved figure whenever the array is not the
                # whole critical path. Keeping both, labelled apart, is the point.
                f"{format_quantity(phase.achieved_flops_per_s, 'OP/s')} achieved · "
                f"{phase.utilization:.0%} of peak\n"
                f"array {format_quantity(totals[Lane.CORE], 'OP')} @ "
                f"{format_quantity(core_rate, 'OP/s')} while busy, of "
                f"{format_quantity(peak, 'OP/s')}\n"
                f"vector {format_quantity(totals[Lane.VECTOR], 'OP')} @ "
                f"{format_quantity(vector_rate, 'OP/s')}{of_vector}",
                band=band,
            ),
        ]
    return out


@dataclass(frozen=True)
class Roof:
    """One chip's ceilings and the point this workload sits at under them."""

    name: str
    colour: str
    peak: float
    bandwidth: float
    derated_peak: float
    derated_bandwidth: float
    tail: float
    intensity: float
    achieved: float
    label: str
    tip: str

    @property
    def ridge(self) -> float:
        return self.peak / self.bandwidth


def _roof(panel: Panel, colour: str) -> Roof:
    """The ceilings and this workload's place under them.

    Both roofs are drawn: the datasheet one and, when the profile derates, the
    fitted-constants one — the gap between them is the unfitted part of any
    prediction. The point's x is intensity against **DRAM** traffic, not
    compulsory traffic, so residency moves it right exactly as it does on the PNG.
    """
    chip, dtype, work = panel.chip, panel.dtype, panel.work
    datasheet = machine_model(idealised(chip), dtype)
    derated = machine_model(chip, dtype)
    dims = datasheet.unit.systolic_dims
    tail = datasheet.effective_flops_per_s / (1 + dims[0]) if dims else 0.0

    intensity = work.flops / work.dram_bytes if work.dram_bytes > 0 else 0.0
    achieved = work.flops / work.latency_s if work.latency_s > 0 else 0.0
    return Roof(
        name=chip.id,
        colour=colour,
        peak=datasheet.effective_flops_per_s,
        bandwidth=datasheet.effective_bandwidth_bytes_per_s,
        derated_peak=derated.effective_flops_per_s,
        derated_bandwidth=derated.effective_bandwidth_bytes_per_s,
        tail=tail,
        intensity=max(intensity, 0.11),
        achieved=max(achieved, 1.0),
        label=work.bound.value.replace("_", " ").lower(),
        tip=(
            f"{chip.name} — {work.name}\n"
            f"{format_quantity(work.flops, 'OP')} over "
            f"{format_bytes(work.dram_bytes)} of DRAM traffic\n"
            f"{intensity:.1f} OP/byte · {format_quantity(achieved, 'OP/s')}\n"
            f"ridge {datasheet.ridge_point:.0f} OP/byte — {work.bound.value}"
        ),
    )


def roofs_for(panels: list[Panel]) -> list[Roof]:
    return [_roof(panel, CHIP_COLOURS[i % len(CHIP_COLOURS)]) for i, panel in enumerate(panels)]


def _roofline_limits(roofs: list[Roof]) -> dict[str, float]:
    """Axis bounds wide enough for every chip on the chart.

    Shared across the PNG panel and the HTML panel, and — the point of taking the
    min and max over all roofs — across chips, so two machines are read against
    one pair of scales rather than each against its own.
    """
    return {
        "x_lo": 0.1,
        "x_hi": max([1e4] + [r.intensity * 3 for r in roofs]),
        "y_lo": min(min(r.peak for r in roofs) / 3e3, min(r.achieved for r in roofs) / 3),
        "y_hi": max(r.peak for r in roofs) * 3,
    }


def _roofline_data(roofs: list[Roof]) -> dict[str, object]:
    """The chart's shared axes plus one entry per chip.

    Always a list, even for one chip, so the single-chip page and the comparison
    page draw through the same code and cannot tell different stories.
    """
    return {
        **_roofline_limits(roofs),
        "chips": [
            {
                "name": r.name,
                "colour": r.colour,
                "peak": r.peak,
                "bw": r.bandwidth,
                "derated_peak": r.derated_peak,
                "derated_bw": r.derated_bandwidth,
                "tail": r.tail,
                "points": [
                    {"ai": r.intensity, "achieved": r.achieved, "label": r.label, "tip": r.tip}
                ],
            }
            for r in roofs
        ],
    }


def _deployments(panels: list[Panel]) -> list[dict[str, str]]:
    """One listing per chip, for the workloads that have no runnable program.

    A lone matmul's decomposition is emitted as real Python instead (D54), which
    is the section below this one; what is left here is the *sequence* listing a
    network gets, where there is no single tile grid to walk and so nothing to
    emit. Every constant is still checked against the schedule before it reaches
    the page (``deploy.check``), on the same reasoning as ``explain.check``.
    """
    out: list[dict[str, str]] = []
    for panel in panels:
        work = panel.work
        if work.phase is None or work.program is not None:
            continue
        listing = deployment_of(
            panel.chip,
            panel.machine,
            work.phase,
            work.trace,
            workload=f"{work.name} at {panel.dtype.value}",
            operation=work.operation,
        )
        check_deployment(listing, work.trace)
        out.append({"title": listing.title, "code": listing.code})
    return out


def _programs(panels: list[Panel]) -> list[dict[str, str]]:
    """Each panel's runnable program, carried in the page.

    This is what replaced the pseudo-C listing for a tiled matmul (D54). The page
    is for reading; ``--emit`` saves the same program to disk to run, from the
    same ``EmittedProgram``, so the two cannot differ.
    """
    return [
        {
            "title": _program_title(panel, panel.work.program),
            "filename": panel.work.program.filename,
            "source": panel.work.program.source,
        }
        for panel in panels
        if panel.work.program is not None
    ]


def _program_title(panel: Panel, program: EmittedProgram) -> str:
    """Chip, precision and decomposition — the three things that pick a file."""
    return f"{panel.chip.name} — {panel.dtype.value} — {program.stationarity.value}"


def _stationarity_banner(panels: list[Panel]) -> str:
    """The decomposition the timeline is of, for the page's banner strip (D53).

    Empty for a workload with no single tile grid — a network's operations run
    in sequence (D5a) and have no one grid between them — and for a comparison
    whose chips disagree, where one line could only be wrong about one of them;
    each chip's own listing states its grid in that case.
    """
    # The FIRST span carrying a grid, not the first span: span 0 is the kernel
    # dispatch, which has no tile to address, so reading its grid returned None
    # and this banner silently never appeared on a matmul page at all.
    grids = {
        next((span.grid for span in panel.work.trace.spans if span.grid is not None), None)
        for panel in panels
    }
    only = {grid for grid in grids if grid is not None}
    if len(only) != 1:
        return ""
    grid = next(iter(only))
    # "B stays resident" is a misnomer on a unit with no weight banks (D62), and
    # two chips in a comparison can disagree about that even on one grid — so the
    # phrase is only used when every panel's unit gives the same one.
    phrases = {residency_phrase(grid, panel.machine.unit) for panel in panels}
    held = phrases.pop() if len(phrases) == 1 else f"{grid.resident.value} stays resident"
    banner = (
        f"<b>stationarity</b> {grid.stationarity.value} — {held}, "
        f"in a {grid.rows:,} x {grid.cols:,} tile grid "
        f"({grid.row_dim.value} x {grid.col_dim.value}) whose tiles each sweep "
        f"{grid.swept_dim.value}. Tile addresses on the bars below index into it."
    )
    if grid.k_partitions > 1:
        banner += (
            f" <b>split-K</b> {grid.k_partitions}: the contraction is cut that many ways and "
            f"summed by a second kernel, drawn at the end of the DRAM and vector rows."
        )
    return banner + _reduction_banner(panels, grid)


def _reduction_banner(panels: list[Panel], grid: TileGrid) -> str:
    """Where this grid's partials meet, when every panel agrees (D62).

    The vector row grows a bar for a K-on-grid reduction and nothing on the page
    said why; this is that sentence. Silent when the chips disagree — Metis sums
    a contraction in its own periphery where A100 pays the CUDA cores, and one
    line cannot be right about both. Silent too under split-K, whose own
    sentence above already covers it.
    """
    placements = {
        panel.work.phase.ops[0].reduction_placement
        for panel in panels
        if panel.work.phase is not None and panel.work.phase.ops
    }
    if grid.k_partitions > 1 or len(placements) != 1:
        return ""
    placement = placements.pop()
    slices = grid.k_slices
    adds = f"{(slices - 1) * grid.m * grid.n:,} additions"
    if placement is ReductionPlacement.LOCAL:
        return (
            f" <b>reduction</b> local: each output element ends up with {slices} partial values, "
            f"and this unit declares an accumulator deep enough to sum them in its own periphery "
            f"— nothing is charged and no bar is drawn."
        )
    if placement is ReductionPlacement.ON_CHIP:
        return (
            f" <b>reduction</b> on chip: each output element ends up with {slices} partial "
            f"values, so {adds} run on the vector row — overlapped with the matrix row, since "
            f"the array builds the next slice while the vector unit sums the last (D62)."
        )
    if placement is ReductionPlacement.DRAM:
        return (
            f" <b>reduction</b> through DRAM: the {grid.accumulator_elements:,} live "
            f"accumulators do not fit on chip, so the partials cross DRAM and {adds} follow the "
            f"matrix work rather than overlapping it (D62)."
        )
    return ""


def write_timeline(panels: list[Panel], command: str, out: Path) -> None:
    """The same figure, zoomable, as one self-contained file.

    A comparison page carries every chip's rows against one shared, absolute time
    axis, and the arithmetic **once** — the operator list is a property of the
    workload, which is the same on both machines, so repeating it per chip would
    be repeating the identical text.
    """
    rows = panel_rows(panels)
    total = max(p.work.trace.total_s for p in panels)
    page = render(
        title=_title(panels),
        subtitle=_subtitle(panels),
        footer=(
            f"bwz {bwz.__version__}{_git()} — every bar is a slice of the reported latency: "
            f"the DRAM row sums to t_dram and the compute row to t_compute. "
            f"<br><code>$ {command}</code>"
        ),
        boxes=_boxes(panels),
        banner=_stationarity_banner(panels),
        rows=[
            {
                "title": row.title,
                "detail": row.detail,
                "note": row.note,
                "lane": row.lane.value if row.lane else None,
                "quantity": _quantity(row, panels[row.panel].work.trace),
                "panel": row.panel,
                "header": row.header,
                "colour": row.colour,
            }
            for row in rows
        ],
        roofline=_roofline_data(roofs_for(panels)),
        explanations=[
            {
                "op_id": e.op_id,
                "shapes": e.shapes,
                "algebra": e.algebra,
                "arithmetic": e.arithmetic,
                "arithmetic_short": format_quantity(e.flops, "OP"),
                "code": e.code,
            }
            for e in panels[0].work.explanations
        ],
        spans=[
            {
                "lane": span.lane.value,
                "start": span.start_s,
                "end": span.end_s,
                "store": span.stage is Stage.STORE,
                "streaming": span.stage is Stage.LOAD_A,
                "tip": _tip(span),
                "panel": index,
            }
            for index, panel in enumerate(panels)
            for span in panel.work.trace.spans
        ],
        deployments=_deployments(panels),
        programs=_programs(panels),
        total_s=total,
        hint=(
            " Rows are banded by chip and the time axis is <b>shared and absolute</b>, so a bar "
            "twice as long took twice as long."
            if len(panels) > 1
            else ""
        ),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")


def _tile_segments(tile_start: int, tile_end: int, grid_cols: int) -> list[tuple[int, int, int]]:
    """Split a global ``[tile_start, tile_end)`` tile range into per-grid-row
    segments ``(row, col_start, col_end)`` — the same decomposition the
    animation's geometry panel does in JS (D48's ``geoSegments``), so the
    hover and the panel never name a tile differently. What a row *is* depends
    on the stationarity (D53): a k-slice under ``ws``, a band of M under ``os``.
    """
    segments: list[tuple[int, int, int]] = []
    t = tile_start
    while t < tile_end:
        row = t // grid_cols
        row_end = (row + 1) * grid_cols
        seg_end = min(tile_end, row_end)
        segments.append((row, t - row * grid_cols, seg_end - row * grid_cols))
        t = seg_end
    return segments


def _format_index_ranges(nums: list[int]) -> str:
    """Collapse consecutive integers into ``start..end`` — a whole-A ramp
    lights every one of e.g. 256 k-slices at once, and spelling out all 256
    would swamp the hover (D48)."""
    if not nums:
        return ""
    ordered = sorted(nums)
    parts: list[str] = []
    start = prev = ordered[0]
    for v in ordered[1:]:
        if v == prev + 1:
            prev = v
            continue
        parts.append(str(start) if start == prev else f"{start}..{prev}")
        start = prev = v
    parts.append(str(start) if start == prev else f"{start}..{prev}")
    return ",".join(parts)


def _merge_column_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping/touching half-open ``[n0, n1)`` ranges.

    For an operand whose other dimension is *swept* rather than on the grid,
    several grid-row segments touching the *same* columns (the common case: a
    wave's tiles span many rows, each covering most or all of one row's
    columns) must read as one column range, not one repeated per row it
    happened to come from. Under ``ws`` that operand is C, whose result tile is
    the full M height x one n-tile's width.
    """
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _operand_axes(grid: TileGrid, first: Dim, second: Dim) -> tuple[bool, bool]:
    """Whether each of an operand's two dimensions is the grid's row axis.

    ``(is_row, is_col)`` per dimension is all a caller needs to place a grid
    position into that operand's own index: a dimension on the row axis takes
    the row number, one on the column axis takes the column, and one the tiles
    sweep takes ``:`` because the operand is covered in full (D53).
    """
    return (first is grid.row_dim, second is grid.col_dim)


def _index_notation(span: Span) -> str:
    """``A(:,g)``/``B(row,col)``/``C(:,col)`` — the same tile-grid vocabulary
    the geometry panel's caption uses (D48), for whichever grid the machine's
    stationarity implies (D53). Which index an operand takes follows from where
    its dimensions sit: on the grid's rows, on its columns, or swept. Empty for
    a network's per-operation trace, which has no tile grid to index into (D42),
    and for the kernel dispatch span, which is not a tile at all.
    """
    grid = span.grid
    if span.tile_start is None or span.tile_end is None or grid is None:
        return ""
    if span.stage is Stage.LOAD_A:
        # Not `_tile_segments` (which decomposes the step's raw *touched* tile
        # range — right for B/EXEC, which genuinely spans several rows at
        # once). A's own byte cost is `openings`-based (D33/D48): this event
        # *completes* rows [start//w, end//w) — the same window `_tile_trace`'s
        # label uses — never the block its last tile merely touches but a
        # *later* event finishes and gets billed for.
        first = span.tile_start // grid.cols
        last = span.tile_end // grid.cols - 1
        bands = _format_index_ranges(list(range(first, last + 1)))
        # A is M x K. Under ws the staged band is a slice of K; under os it is
        # a band of M rows. Whichever it is, the other axis is read in full.
        return f"A({bands},:)" if grid.row_dim is Dim.M else f"A(:,{bands})"
    segments = _tile_segments(span.tile_start, span.tile_end, grid.cols)
    operand, first_dim, second_dim = {
        Stage.LOAD: ("B", Dim.K, Dim.N),
        Stage.STORE: ("C", Dim.M, Dim.N),
    }.get(span.stage, (grid.resident.value, grid.row_dim, grid.col_dim))
    row_on_grid, col_on_grid = _operand_axes(grid, first_dim, second_dim)
    if not row_on_grid:
        # One index only: the rows all cover the same columns of this operand,
        # so they collapse into one range rather than repeating per row.
        merged = _merge_column_ranges([(c0, c1) for _row, c0, c1 in segments])
        cells = [f"{c0}" if c1 - c0 == 1 else f"{c0}..{c1 - 1}" for c0, c1 in merged]
        joiner = f"); {operand}(:,"
        return f"{operand}(:," + joiner.join(cells) + ")"
    if not col_on_grid:
        rows = _format_index_ranges([row for row, _c0, _c1 in segments])
        return f"{operand}({rows},:)"
    cells = [f"{row},{c0}" if c1 - c0 == 1 else f"{row},{c0}..{c1 - 1}" for row, c0, c1 in segments]
    return f"{operand}(" + f"); {operand}(".join(cells) + ")"


def _tip(span: Span) -> str:
    """A span's own numbers, for the hover.

    The first line names the transaction — LOAD, STORE, EXEC, HOLD — because a
    bar's colour tells you which resource it is on and nothing about what it was
    doing there.

    **Both compute lanes are arithmetic.** D28 gave the vector engine its own
    lane; this function still tested only ``Lane.CORE``, so every norm and
    activation bar fell through to the SRAM branch and hovered as
    ``HOLD — elementwise on chip … holding 0 B`` — the one lane whose operations
    and rate the reader most needs, reported as a buffer occupancy of nothing.

    **The index line (D48)** names the same ``A(:,g)``/``B(row,col)`` position
    the geometry panel highlights, straight off ``Span.tile_start``/
    ``tile_end``/``grid`` — no re-derivation, just the tooltip finally
    saying what the label's own bytes and band count already implied.
    """
    when = f"{format_time(span.start_s)} + {format_time(span.duration_s)}"
    idx = _index_notation(span)
    idx_line = f"\n{idx}" if idx else ""
    if span.lane is Lane.DRAM:
        # "band" rather than "k-slice": what one A staging event covers follows
        # the grid, and is a slice of K only under weight-stationary (D53).
        band = span.grid.group_name if span.grid is not None else "k-slice"
        kind = {
            Stage.STORE: "STORE — result C written back",
            Stage.LOAD: "LOAD — operand B",
            Stage.REDUCE: "REDUCE — split-K partials out and back, between the two kernels (D53)",
            Stage.LOAD_A: {
                "stage": f"STAGE — operand A: {band} staging, read once in total (D33)",
                "whole": f"STAGE — operand A: whole-A ramp, every {band} before wave 0 (D33)",
                "stream": "STREAM — operand A, re-fetched per tile (D31)",
            }.get(span.a_fetch_mode, "LOAD — operands in"),
        }.get(span.stage, "LOAD — operands in")
        note = (
            {
                "stage": (
                    f"\nThis bar is one whole {band}: every tile of the group reads this"
                    " staging, and A crosses DRAM exactly once"
                ),
                "whole": (
                    f"\nEvery {band} of A lands before the first tile computes — the same"
                    f" total bytes as staging per {band}, ramped upfront instead"
                ),
            }.get(span.a_fetch_mode, "")
            if span.stage is Stage.LOAD_A
            else ""
        )
        return (
            f"{kind}\n{span.label}{idx_line}\n{when}\n"
            f"{format_bytes(span.bytes_moved)} @ {format_bandwidth(span.rate_bytes_per_s)}"
            f"{note}"
        )
    if span.lane in (Lane.CORE, Lane.VECTOR):
        engine = "array" if span.lane is Lane.CORE else "vector unit"
        if span.stage is Stage.REDUCE:
            # Two different reductions share this stage (D62): split-K's second
            # kernel, which runs after the GEMM, and partials from OTHER units
            # summed alongside it on chip. Naming the second as the first told a
            # Metis reader its DPU adds were a CUTLASS kernel.
            second_kernel = span.grid is not None and span.grid.materialises_partials
            kind = (
                f"REDUCE — split-K's second kernel (CUTLASS), on the {engine} (D27/D53)"
                if second_kernel
                else f"REDUCE — partials from other units, summed on the {engine} alongside"
                " the matrix work (on chip, D62)"
            )
            return (
                f"{kind}\n"
                f"{span.label}\n{when}\n"
                f"{format_quantity(span.flops, 'OP')} @ "
                f"{format_quantity(span.rate_flops_per_s, 'OP/s')}\n"
                "Not new arithmetic — 2*M*N*K already counts these adds; they have "
                "merely left the matrix engine's accumulator"
            )
        kind = (
            f"EXEC — {span.op_type} on the {engine}"
            if span.stage is Stage.EXEC
            else "DISPATCH — kernel launch"
        )
        return (
            f"{kind}\n{span.label}{idx_line}\n{when}\n"
            f"{format_quantity(span.flops, 'OP')} @ "
            f"{format_quantity(span.rate_flops_per_s, 'OP/s')}"
        )
    return (
        f"HOLD — {span.op_type or 'tile'} on chip\n{span.label}{idx_line}\n{when}\n"
        f"holding {format_bytes(span.resident_bytes)}"
    )


_ANIMATION_STAGE = {
    Stage.LOAD: "load_b",
    Stage.LOAD_A: "load_a",
    Stage.HOLD: "hold",
    Stage.EXEC: "exec",
    Stage.STORE: "store",
    Stage.REDUCE: "reduce",
}
"""Kernel dispatch (Stage.DISPATCH) is fixed overhead, not a DRAM/SRAM/compute
transaction, so it has nothing to animate and is left out of the map.
``Stage.REDUCE`` is real vector arithmetic — split-K's second kernel after the
GEMM (D53), or partials from other units summed alongside it (D62) — so it plays
back like any other event; the listing's own ``reduce`` tag lights the line it
comes from."""


def _flow_spans(trace: PipelineTrace) -> list[dict[str, object]]:
    """A trace's spans, as the events ``--animate`` plays back.

    Reuses the trace and ``_tip`` verbatim — the animation is a player over the
    same schedule the timeline draws, not a second model of it.
    """
    out: list[dict[str, object]] = []
    for span in trace.spans:
        stage = _ANIMATION_STAGE.get(span.stage)
        if stage is None:
            continue
        out.append(
            {
                "stage": stage,
                "lane": span.lane.value,
                "start": span.start_s,
                "end": span.end_s,
                "bytes": span.resident_bytes if span.stage is Stage.HOLD else span.bytes_moved,
                "tip": _tip(span),
                "streaming": span.stage is Stage.LOAD_A and span.a_fetch_mode == "stream",
                "step": span.step,
                "tile_start": span.tile_start,
                "tile_end": span.tile_end,
            }
        )
    return out


def write_animation(panel: Panel, command: str, out: Path) -> None:
    """One chip's schedule as a self-contained DRAM -> SRAM -> Accelerator flow
    animation (docs/CLI.md §3, docs/CORRECTIONS.md D40/D42/D43).

    Matmul or the ad-hoc encoder — the caller never reaches this with a
    `--model` workload (`_reject_flags_for_the_wrong_workload` refuses that
    combination). A lone matmul carries a real ``dataflow`` plan (D33/D36); the
    encoder's per-operation trace has none — it is a sequence of named
    operations, not one A/B dataflow strategy to name (D5a) — so every
    dataflow-specific argument below is threaded through only when there is one.

    Stations are ``rows_for``'s own resource list (D43) — the same one the
    timeline draws, grey for what v1 declares but does not cost (D20) — not a
    bespoke three-station shape, so a chip with more declared memory levels or
    a second compute engine gets more stations, not a collapsed picture of one.
    """
    work = panel.work
    dataflow = work.dataflow
    assert work.phase is not None, "every --animate workload carries its phase"
    # The code pane shows the RUNNABLE program where there is one (D54): the
    # highlighted lines are then statements that perform the transfer rather
    # than a pseudo-C paraphrase of one, and a reader can run the file the
    # animation is stepping through. A network has no tile grid to walk, so it
    # keeps the sequence listing, which is all `deployment_of` still produces.
    program = work.program
    code_lines = program.source.split("\n") if program is not None else []
    stage_lines: dict[str, list[int]] = {}
    if program is not None:
        stage_lines = {tag: list(indices) for tag, indices in program.stage_lines}
    if program is None:
        listing = deployment_of(
            panel.chip,
            panel.machine,
            work.phase,
            work.trace,
            workload=f"{work.name} at {panel.dtype.value}",
            operation=work.operation,
        )
        check_deployment(listing, work.trace)
        code_lines = listing.code.split("\n")
        stage_lines = {tag: list(indices) for tag, indices in listing.stage_lines}
    geometry: dict[str, object] | None = None
    if program is not None and program.grid is not None:
        # Straight off the program's own grid (D53): the panel has to draw the
        # operand THIS chip keeps resident, and which one that is — along with
        # which dimension each grid axis carries — is exactly what the grid
        # says. Re-deriving it here is how the panel and the program would drift
        # apart.
        grid = program.grid
        band_rows, band_cols = grid.a_event_shape
        geometry = {
            "m": grid.m,
            "n": grid.n,
            "k": grid.k,
            "rows": program.array_rows,
            "cols": program.array_cols,
            "grid_rows": grid.rows,
            "grid_cols": grid.cols,
            "row_dim": grid.row_dim.value,
            "col_dim": grid.col_dim.value,
            "swept_dim": grid.swept_dim.value,
            "resident": grid.resident.value,
            "stationarity": grid.stationarity.value,
            "a_events": grid.a_events,
            "group_name": grid.group_name,
            "band_rows": band_rows,
            "band_cols": band_cols,
            "splits": grid.k_partitions,
        }
    stations: list[dict[str, object]] = [
        {
            "name": row.title,
            "detail": row.detail,
            "note": row.note,
            "lane": row.lane.value if row.lane else None,
        }
        for row in rows_for(panel)
    ]
    page = render_animation(
        title=_title([panel]),
        subtitle=_subtitle([panel]),
        footer=(
            f"bwz {bwz.__version__}{_git()} — playback of the same schedule the timeline "
            f"draws; the reported latency, not the drawn span, is the ground truth."
            f"<br><code>$ {command}</code>"
        ),
        flow=_flow_spans(work.trace),
        total_s=work.trace.total_s,
        reported_latency_s=work.trace.reported_latency_s,
        fill_drain_s=work.trace.fill_drain_s,
        stations=stations,
        a_strategy=dataflow.a_strategy.value if dataflow is not None else None,
        b_dataflow=dataflow.b_dataflow.value if dataflow is not None else None,
        notes=list(dataflow.notes) if dataflow is not None else [],
        code_lines=code_lines,
        stage_lines=stage_lines,
        geometry=geometry,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")


def _quantity(row: Row, trace: PipelineTrace) -> str:
    """What went through this resource. The part an instruction-centric view
    cannot give: a row means nothing for a comparison until it carries a number."""
    if row.quantity:
        return row.quantity
    totals, busy, concurrency = trace.totals, trace.busy_s, trace.concurrency
    if row.lane is Lane.DRAM:
        if totals[Lane.DRAM] <= 0:
            return "0 B — nothing crossed"
        rate = totals[Lane.DRAM] / busy[Lane.DRAM] if busy[Lane.DRAM] else 0.0
        _, writes = trace.direction_bytes
        weights, activations = trace.operand_bytes
        return (
            f"{format_bytes(totals[Lane.DRAM])} @ {format_bandwidth(rate)}\n"
            f"LOAD B={format_bytes(weights)} · A={format_bytes(activations)}\n"
            f"STORE C={format_bytes(writes)}"
        )
    if row.lane is Lane.SRAM:
        mean, peak = concurrency[Lane.SRAM]
        plural = "buffers" if peak != 1 else "buffer"
        return f"{format_bytes(totals[Lane.SRAM])} in {peak} {plural} (x{mean:.2f} avg)"
    if row.lane in (Lane.CORE, Lane.VECTOR):
        lane = row.lane
        if totals[lane] <= 0:
            return "idle — nothing of its kind"
        rate = totals[lane] / busy[lane] if busy[lane] else 0.0
        mix = _op_mix(trace, matrix=lane is Lane.CORE)
        return f"{format_quantity(totals[lane], 'OP')} @ {format_quantity(rate, 'OP/s')}\n{mix}"
    return "not used"


def _title(panels: list[Panel]) -> str:
    work = panels[0].work
    if len(panels) == 1:
        return f"{panels[0].chip.name} — {work.name} at {panels[0].dtype.value}"
    chips = " vs ".join(panel.chip.name for panel in panels)
    return f"{work.name} at {panels[0].dtype.value} — {chips}"


def _subtitle(panels: list[Panel]) -> str:
    """One chip: the span and how it was scheduled. Two: that, plus the ratio the
    figure exists to show, stated rather than left to be measured off the axis.

    The ratio is quoted from the **reported latency**, never from the drawn span.
    The two differ by pipeline fill/drain, which the roofline's ``max()`` omits
    and which differs wildly between machines: on an 8192-cubed INT8 matmul it is
    0.0% of A100's latency and 17.7% of Metis's, so a ratio read off the bars
    said 3.93x where the report says 3.34x. A headline disagreeing with the
    report by 18% is the failure D19 exists to prevent (D35).
    """
    if len(panels) == 1:
        return _trace_subtitle(panels[0].work.trace)
    ordered = sorted(panels, key=lambda p: p.work.trace.reported_latency_s)
    fastest, slowest = ordered[0], ordered[-1]
    quickest = fastest.work.trace.reported_latency_s or 1.0
    ratio = slowest.work.trace.reported_latency_s / quickest
    spans = " · ".join(
        f"{panel.chip.id} {format_time(panel.work.trace.reported_latency_s)}" for panel in panels
    )
    # Name the gap between what is drawn and what is reported wherever it is big
    # enough to see, rather than dropping the disclosure the single-chip
    # subtitle has always carried (D35).
    drawn = [
        f"{panel.chip.id} +{format_time(panel.work.trace.fill_drain_s)}"
        for panel in panels
        if panel.work.trace.reported_latency_s > 0
        and panel.work.trace.fill_drain_s / panel.work.trace.reported_latency_s >= 0.01
    ]
    gap = " · ".join(drawn)
    tail = (
        f" Bars run past it by the pipeline fill/drain the roofline omits ({gap})." if drawn else ""
    )
    return (
        f"Shared absolute time axis, reported latency. {spans}. "
        f"{fastest.chip.id} is {ratio:.2f}x faster than {slowest.chip.id} on this workload.{tail}"
    )


def _trace_subtitle(trace: PipelineTrace) -> str:
    steps = (
        f"{trace.steps} steps drawn, coalesced from {trace.tiles} "
        f"{'operations' if trace.kind == 'operations' else 'tiles'}. "
        if trace.coalesced
        else f"{trace.steps} {'operations' if trace.kind == 'operations' else 'tile steps'}. "
    )
    if trace.kind == "operations":
        buffering = (
            "Operations do not pipeline against each other in this model (D5a), so the span is "
            "the reported latency exactly. Hatched bars are kernel dispatch."
        )
    elif not trace.double_buffered:
        buffering = "No double buffer: loads and arithmetic alternate."
    elif trace.fill_drain_s > 0:
        buffering = (
            f"Double buffered — the reported latency omits {format_time(trace.fill_drain_s)} "
            f"of pipeline fill/drain."
        )
    else:
        buffering = (
            "Double buffered, but nothing crosses DRAM to overlap, so the span is the reported "
            "latency exactly."
        )
    return f"Span {format_time(trace.total_s)}. {steps}{buffering}"


def _git() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""
    return f" @ {sha}"


def _units_used(ops: Sequence[Operation], machine: MachineModel) -> int:
    """Arrays the matrix engine ever has busy at once, over a whole phase (D30).

    The **max** over the phase's operations, not a sum or an average: the
    resource row is a claim about the machine — this many arrays are reached by
    this workload at its widest — and a graph whose largest GEMM fills the chip
    has reached all of it even if a projection later occupies four cores.

    Only matrix work counts. Norms and elementwise operations run on the vector
    unit, which has no tile grid here (D27).
    """
    return max(
        (
            operation_cores(
                op,
                machine.unit,
                stationarity=machine.stationarity,
                k_partitions=machine.k_partitions,
            )
            for op in ops
            if op.op_type in MATRIX_OP_TYPES
        ),
        default=0,
    )


def build_matmul(
    chip: HardwareSpec,
    spec: MatmulSpec,
    deployment: DeploymentSpec,
    report: Report,
    *,
    steps: int,
    command: str,
) -> Workload:
    """One matmul's schedule, drawn from the report the caller already has.

    The ``Report`` and the ``DeploymentSpec`` come in rather than being rebuilt
    here (D55): ``bwz matmul`` prints its table from one analysis and draws its
    figure from the same one, so the picture cannot illustrate a run the numbers
    above it did not come from. Everything else below is a pure function of those
    two, recomputed rather than threaded through — ``machine_model``,
    ``build_graph`` and ``plan_dataflow`` are deterministic, so recomputing is
    cheaper than widening the signature and cannot disagree.
    """
    machine = machine_model(
        chip,
        spec.operand_dtype,
        stationarity=deployment.stationarity,
        k_partitions=deployment.split_k,
    )
    graph = build_graph(spec, deployment, GraphPhase.STATIC)
    dataflow = plan_dataflow(
        graph.ops[0],
        machine,
        chip,
        deployment,
        a_bytes=cost_of(graph.ops[0], graph.tensors).input_bytes,
    )
    trace = build_trace(
        graph,
        report.phases[0],
        machine,
        double_buffered=report.memory.double_buffered,
        max_steps=steps,
        dataflow=dataflow,
    )
    op = report.phases[0].ops[0]
    # The page's "how it is deployed" section IS this program (D54). A chip whose
    # fastest unit for this dtype declares no array geometry (fp32 on A100 runs
    # on the CUDA cores) has no tile grid to walk, and falls back to the sequence
    # listing instead.
    program: EmittedProgram | None = None
    grid = grid_of(graph.ops[0], machine)
    if grid is not None:
        program = emit_matmul(
            chip,
            machine,
            grid,
            dataflow,
            op,
            a_dtype=spec.a_dtype,
            b_dtype=spec.b_dtype,
            c_dtype=spec.result_dtype,
            acc_dtype=deployment.precision.accumulate,
            double_buffered=report.memory.double_buffered,
            command=command,
            version=bwz.__version__,
        )
        check_program(program)
    return Workload(
        spec.name,
        trace,
        explain_graph(graph),
        op.flops,
        op.dram_bytes,
        op.latency_s,
        op.bound,
        phase=report.phases[0],
        operation=graph.ops[0],
        dataflow=dataflow,
        machine=machine,
        units_used=_units_used(graph.ops, machine),
        program=program,
    )


def build_phases(
    chip: HardwareSpec,
    model: AnyModelSpec,
    deployment: DeploymentSpec,
    report: Report,
    *,
    tokens: int,
    steps: int,
) -> list[Workload]:
    """One workload per phase of *model* — prefill and decode are different
    machines (CLAUDE.md #6), so they get different figures rather than an
    average.

    Same contract as :func:`build_matmul`: the caller's own report and
    deployment, so the page and the table it came with describe one run.
    """
    machine = machine_model(
        chip,
        compute_dtype(chip, deployment.precision.weights, deployment.precision.activations),
    )
    graphs = build_graphs(model, deployment)
    out = []
    for phase in report.phases:
        trace = build_trace(
            graphs[phase.phase],
            phase,
            machine,
            double_buffered=report.memory.double_buffered,
            max_steps=steps,
        )
        out.append(
            Workload(
                f"{model.name} {phase.phase.value} S={tokens}",
                trace,
                explain_graph(graphs[phase.phase]),
                phase.flops,
                phase.dram_bytes,
                phase.latency_s,
                phase.bound,
                phase=phase,
                units_used=_units_used(graphs[phase.phase].ops, machine),
            )
        )
    return out


PREFERRED_DTYPES = (DType.FP16, DType.INT8, DType.BF16, DType.FP32)


def shared_dtype(chips: list[HardwareSpec], requested: str | None) -> DType:
    """One precision for every chip in a comparison.

    Per-chip defaults would silently compare *different workloads*: A100 defaults
    to fp16 and Metis has no fp16 datapath at all, so the two figures would move
    different numbers of bytes and the shared time axis would be meaningless. The
    comparison therefore insists on a precision both machines can execute
    (docs/CORRECTIONS.md D29).
    """
    if requested is not None:
        dtype = DType(requested)
        missing = [c.id for c in chips if not c.supports(dtype)]
        if missing:
            raise SystemExit(
                f"bwz: {', '.join(missing)} has no {dtype.value} datapath, so --compare-with "
                f"cannot run the same workload on every chip. Supported by all: "
                f"{_common_dtypes(chips) or 'nothing — these chips share no precision'}"
            )
        return dtype
    for candidate in PREFERRED_DTYPES:
        if all(c.supports(candidate) for c in chips):
            return candidate
    common = _common_dtypes(chips)
    if not common:
        raise SystemExit(
            "bwz: --compare-with needs one precision every chip supports, and "
            + "; ".join(f"{c.id} supports {_dtypes_of(c)}" for c in chips)
        )
    return DType(common.split(", ")[0])


def _dtypes_of(chip: HardwareSpec) -> str:
    return ", ".join(sorted({d.value for u in chip.compute_units for d in u.supported_dtypes}))


def _common_dtypes(chips: list[HardwareSpec]) -> str:
    shared = set.intersection(
        *({d for u in c.compute_units for d in u.supported_dtypes} for c in chips)
    )
    return ", ".join(sorted(d.value for d in shared))
