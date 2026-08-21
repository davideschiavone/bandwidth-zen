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

import argparse
import shlex
import subprocess
import sys
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path

from dataflow_html import render as render_animation
from timeline_html import Box, render

import bwz
from bwz.analysis import analyze, idealised, machine_model
from bwz.analysis.dataflow import DataflowPlan, plan_dataflow
from bwz.analysis.pipeline import Lane, PipelineTrace, Span, Stage, build_trace
from bwz.analysis.roofline import MATRIX_OP_TYPES, compute_dtype
from bwz.deploy import check as check_deployment
from bwz.deploy import deployment_of
from bwz.explain import Explanation, explain_graph
from bwz.graph import GraphPhase, build_graph, build_graphs
from bwz.graph.ops import Operation
from bwz.kernels import encoder_layer_kernel, matmul_kernel
from bwz.operators.base import cost_of
from bwz.report import Bound, PhaseResult
from bwz.spec import (
    AnyModelSpec,
    AStrategy,
    BDataflow,
    DeploymentSpec,
    DType,
    HardwareSpec,
    load_chip,
    load_model,
)
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


def rows_for(chip: HardwareSpec, dtype: DType) -> list[Row]:
    """Resource rows, read off the chip profile.

    The v1 machine is three elements (``docs/CORRECTIONS.md`` D5a): the deepest
    memory level supplies bandwidth, the shallowest supplies capacity, the
    fastest compute unit supplies TOPS. Everything else a profile declares is
    drawn idle with the reason, which is more honest than leaving it out — an
    A100 has 40 MB of L2 this model never spends, and that omission is worth
    seeing next to a chip whose SRAM is the whole story.
    """
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

    for unit in chip.compute_units:
        geometry = (
            f"{unit.systolic_dims[0]}x{unit.systolic_dims[1]} array"
            if unit.systolic_dims
            else f"{unit.ops_per_cycle_per_unit:g} MAC/cycle"
        )
        detail = f"{unit.count} x {geometry}"
        peak = unit.peak_flops_per_s(chip.clock_hz, dtype) if unit.supports(dtype) else 0.0
        rate = f"peak {format_quantity(peak, 'OP/s')} at {dtype.value}"
        if unit is machine.unit:
            rows.append(Row(unit.name, detail, Lane.CORE, f"matrix work — {rate}"))
        elif unit is machine.vector_unit:
            # A tensor core does matrix-multiply-accumulate and nothing else, so
            # norms, activations and residuals have their own row on their own
            # silicon (D27/D28).
            rows.append(Row(unit.name, detail, Lane.VECTOR, f"norms, activations — {rate}"))
        elif unit.supports(dtype):
            rows.append(Row(unit.name, detail, None, "idle — no work of its kind in this graph"))
        else:
            rows.append(Row(unit.name, detail, None, f"idle — no {dtype.value} datapath"))
    return rows


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
        for row in rows_for(panel.chip, panel.dtype):
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
                f"LOAD B {format_bytes(trace.operand_bytes[0])} · "
                f"A {format_bytes(trace.operand_bytes[1])}\n"
                f"STORE C {format_bytes(trace.direction_bytes[1])} · {rate} while active\n"
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
    """One loop-nest listing per chip — the section a comparison must not merge.

    Every constant is checked against the schedule before it reaches the page
    (``deploy.check``), on the same reasoning as ``explain.check``: a listing
    that disagreed with the timeline above it would be believed.
    """
    out: list[dict[str, str]] = []
    for panel in panels:
        work = panel.work
        if work.phase is None:
            continue
        dataflow = work.dataflow
        listing = deployment_of(
            panel.chip,
            machine_model(panel.chip, panel.dtype),
            work.phase,
            work.trace,
            workload=f"{work.name} at {panel.dtype.value}",
            operation=work.operation,
            **(
                {"a_strategy": dataflow.a_strategy, "b_dataflow": dataflow.b_dataflow}
                if dataflow is not None
                else {}
            ),
        )
        check_deployment(listing, work.trace)
        out.append({"title": listing.title, "code": listing.code})
    return out


def write_html(panels: list[Panel], command: str, out: Path) -> None:
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
    print(f"wrote {out}")


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
    """
    when = f"{format_time(span.start_s)} + {format_time(span.duration_s)}"
    if span.lane is Lane.DRAM:
        kind = {
            Stage.STORE: "STORE — result C written back",
            Stage.LOAD: "LOAD — operand B, the tile the array holds",
            Stage.LOAD_A: {
                "stage": "STAGE — operand A: k-slice staging, read once in total (D33)",
                "whole": "STAGE — operand A: whole-A ramp, every k-slice before wave 0 (D33)",
                "stream": "STREAM — operand A, re-fetched per tile (D31)",
            }.get(span.a_fetch_mode, "LOAD — operands in"),
        }.get(span.stage, "LOAD — operands in")
        note = (
            {
                "stage": (
                    "\nThis bar is one whole k-slice: every tile of the group reads this"
                    " staging, and A crosses DRAM exactly once"
                ),
                "whole": (
                    "\nEvery k-slice of A lands before the first tile computes — the same"
                    " total bytes as staging per k-slice, ramped upfront instead"
                ),
            }.get(span.a_fetch_mode, "")
            if span.stage is Stage.LOAD_A
            else ""
        )
        return (
            f"{kind}\n{span.label}\n{when}\n"
            f"{format_bytes(span.bytes_moved)} @ {format_bandwidth(span.rate_bytes_per_s)}"
            f"{note}"
        )
    if span.lane in (Lane.CORE, Lane.VECTOR):
        engine = "array" if span.lane is Lane.CORE else "vector unit"
        kind = (
            f"EXEC — {span.op_type} on the {engine}"
            if span.stage is Stage.EXEC
            else "DISPATCH — kernel launch"
        )
        return (
            f"{kind}\n{span.label}\n{when}\n"
            f"{format_quantity(span.flops, 'OP')} @ "
            f"{format_quantity(span.rate_flops_per_s, 'OP/s')}"
        )
    return (
        f"HOLD — {span.op_type or 'tile'} on chip\n{span.label}\n{when}\n"
        f"holding {format_bytes(span.resident_bytes)}"
    )


_ANIMATION_STAGE = {
    Stage.LOAD: "load_b",
    Stage.LOAD_A: "load_a",
    Stage.HOLD: "hold",
    Stage.EXEC: "exec",
    Stage.STORE: "store",
}
"""Kernel dispatch (Stage.DISPATCH) is fixed overhead, not a DRAM/SRAM/compute
transaction, so it has nothing to animate and is left out of the map."""


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
            }
        )
    return out


def write_animation_html(panel: Panel, command: str, out: Path) -> None:
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
    listing = deployment_of(
        panel.chip,
        machine_model(panel.chip, panel.dtype),
        work.phase,
        work.trace,
        workload=f"{work.name} at {panel.dtype.value}",
        operation=work.operation,
        **(
            {"a_strategy": dataflow.a_strategy, "b_dataflow": dataflow.b_dataflow}
            if dataflow is not None
            else {}
        ),
    )
    check_deployment(listing, work.trace)
    stations: list[dict[str, object]] = [
        {
            "name": row.title,
            "detail": row.detail,
            "note": row.note,
            "lane": row.lane.value if row.lane else None,
        }
        for row in rows_for(panel.chip, panel.dtype)
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
        code_lines=listing.code.split("\n"),
        stage_lines={tag: list(indices) for tag, indices in listing.stage_lines},
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}")


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
            f"LOAD B {format_bytes(weights)} · A {format_bytes(activations)}\n"
            f"STORE C {format_bytes(writes)}"
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


def build_matmul(
    chip: HardwareSpec,
    m: int,
    n: int,
    k: int,
    dtype: DType,
    steps: int,
    *,
    a_strategy: AStrategy = AStrategy.STAGE,
    b_dataflow: BDataflow = BDataflow.WRITE_AHEAD,
    a_residency_tiles: int | None = None,
    a_prefetch_depth: int | None = None,
    iterations: int = 1,
) -> Workload:
    spec = matmul_kernel(m, n, k, a_dtype=dtype, b_dtype=dtype)
    deployment = DeploymentSpec.model_validate(
        {
            "batch": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "a_strategy": a_strategy,
            "b_dataflow": b_dataflow,
            "a_residency_tiles": a_residency_tiles,
            "a_prefetch_depth": a_prefetch_depth,
            "iterations": iterations,
        }
    )
    report = analyze(spec, chip, deployment)
    if not report.feasible:
        raise SystemExit(f"bwz: infeasible on {chip.id}: {report.infeasibility[0]}")
    machine = machine_model(chip, spec.operand_dtype)
    graph = build_graph(spec, deployment, GraphPhase.STATIC)
    # The same plan_dataflow call analyze() made internally to charge the bytes
    # above: recomputed rather than threaded out, because it is pure — same
    # inputs, same plan — so the trace this draws and the report's numbers
    # cannot disagree on which strategy actually ran (whole/persistent clamp).
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
    )


def build_model(
    chip: HardwareSpec, model_id: str, tokens: int, dtype: DType, steps: int
) -> list[Workload]:
    """One workload per phase of a profile: prefill and decode are different
    machines, so they get different figures rather than being averaged."""
    # Profile ids use underscores; accept the hyphenated form people type after
    # seeing the command name.
    return _workloads_for(chip, load_model(model_id.replace("-", "_")), tokens, dtype, steps)


def build_encoder(
    chip: HardwareSpec,
    *,
    dmodel: int,
    nheads: int,
    ffn: int,
    vocab: int,
    tokens: int,
    dtype: DType,
    steps: int,
) -> list[Workload]:
    """A single-layer encoder built from dimensions, mirroring `bwz encoder-layer`.

    The same reason that command exists: a shape you can change one term of and
    watch the picture move, without writing a profile for every experiment.
    """
    try:
        spec = encoder_layer_kernel(
            dmodel=dmodel, nheads=nheads, ffn=ffn, vocab=vocab, tokens=tokens
        )
    except ValueError as exc:
        raise SystemExit(f"bwz: {exc}") from exc
    return _workloads_for(chip, spec, tokens, dtype, steps)


def _workloads_for(
    chip: HardwareSpec, model: AnyModelSpec, tokens: int, dtype: DType, steps: int
) -> list[Workload]:
    """One workload per phase of *model*."""
    deployment = DeploymentSpec.model_validate(
        {
            "batch": 1,
            "input_tokens": tokens,
            "output_tokens": 1,
            "precision": {"weights": dtype, "activations": dtype, "kv_cache": dtype},
        }
    )
    report = analyze(model, chip, deployment)
    if not report.feasible:
        raise SystemExit(f"bwz: infeasible on {chip.id}: {report.infeasibility[0]}")
    machine = machine_model(chip, compute_dtype(chip, dtype, dtype))
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
            )
        )
    return out


PREFERRED_DTYPES = (DType.FP16, DType.INT8, DType.BF16, DType.FP32)


def _default_dtype(chip: HardwareSpec) -> DType:
    for candidate in PREFERRED_DTYPES:
        if chip.supports(candidate):
            return candidate
    return chip.compute_units[0].supported_dtypes[0]


def _shared_dtype(chips: list[HardwareSpec], requested: str | None) -> DType:
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
                f"bwz: {', '.join(missing)} has no {dtype.value} datapath, so --compare cannot "
                f"run the same workload on every chip. Supported by all: "
                f"{_common_dtypes(chips) or 'nothing — these chips share no precision'}"
            )
        return dtype
    for candidate in PREFERRED_DTYPES:
        if all(c.supports(candidate) for c in chips):
            return candidate
    common = _common_dtypes(chips)
    if not common:
        raise SystemExit(
            "bwz: --compare needs one precision every chip supports, and "
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


def _build_parser() -> argparse.ArgumentParser:
    """Three questions, in order: what to run, on what, and (for a lone matmul
    only) how its operands should move. Flags are grouped by which question
    they answer, and a flag from one workload's group used with another
    workload is a hard error rather than a silent no-op — CLAUDE.md #8's
    "actionable errors" applies to the CLI surface as much as to a `Report`.
    """
    parser = argparse.ArgumentParser(
        prog="plot_pipeline.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Draw one self-contained, zoomable HTML timeline: where the time went, on "
            "which piece of hardware, for one chip or a head-to-head of several.\n\n"
            "Three questions: WORKLOAD (what to run — pick exactly one), CHIP (what to "
            "run it on — repeatable, add --compare for a head-to-head), and, only for "
            "the default matmul workload, DATAFLOW STRATEGY (how A and B move, "
            "docs/CLI.md §2.5). See docs/plots/README.md for how to read the page."
        ),
        epilog=textwrap.dedent(
            """\
            examples:
              # a matmul, the default workload
              plot_pipeline.py --chip a100_80gb --matmul 8192,8192,8192

              # a model instead -- one page per phase (prefill, decode)
              plot_pipeline.py --chip a100_80gb --model llama3_8b -S 512

              # an ad-hoc single-layer encoder, sized from the shape flags
              plot_pipeline.py --chip a100_80gb --encoder --dmodel 4096 --nheads 64 \\
                  --ffn 16384 -S 4096

              # two chips, one shared page, head to head
              plot_pipeline.py --chip a100_80gb --chip metis_aipu --compare --model gemma3_4b -S 512

              # a lone matmul's dataflow strategy (docs/CLI.md §2.5)
              plot_pipeline.py --chip metis_aipu --matmul 8192,8192,8192 \\
                  --a-strategy stream --b-dataflow persistent

            --matmul / --model / --encoder are mutually exclusive: pick one workload.
            The encoder shape flags (--dmodel --nheads --ffn --vocab) and
            --tokens/-S only mean anything for --encoder or --model; the dataflow
            strategy flags (--a-strategy --b-dataflow --a-residency-tiles
            --a-prefetch-depth --iterations) only mean anything for the default matmul
            workload. Passing one with the wrong workload is rejected rather than
            silently ignored.
            """
        ),
    )

    workload = parser.add_argument_group(
        "workload — pick exactly one (default: --matmul 4096,4096,4096)"
    ).add_mutually_exclusive_group()
    workload.add_argument(
        "--matmul", default="4096,4096,4096", metavar="M,N,K", help="A[M,K] x B[K,N] -> C[M,N]"
    )
    workload.add_argument(
        "--model", default=None, metavar="ID", help="Draw a model instead; one page per phase"
    )
    workload.add_argument(
        "--encoder",
        action="store_true",
        help="Draw a single-layer encoder sized from the shape flags below, not a profile",
    )

    shape = parser.add_argument_group("encoder shape — only with --encoder")
    shape.add_argument("--dmodel", type=int, default=None, help="Model width (default 8)")
    shape.add_argument(
        "--nheads",
        type=int,
        default=None,
        help="Attention heads (default 2); dmodel must divide evenly by this "
        "-- head_dim is always dmodel // nheads, never set separately",
    )
    shape.add_argument(
        "--ffn",
        type=int,
        default=None,
        help="FFN inner width (default 16, or 4x --dmodel when --dmodel is set)",
    )
    shape.add_argument("--vocab", type=int, default=None, help="Vocabulary (default 16)")
    shape.add_argument(
        "--tokens",
        "-S",
        type=int,
        default=None,
        help="Sequence length for --model / --encoder (default 512)",
    )

    chip = parser.add_argument_group("chip & precision")
    chip.add_argument("--chip", action="append", default=None, help="Chip id; repeatable")
    chip.add_argument(
        "--compare",
        action="store_true",
        help="Draw every --chip in ONE page on a shared, absolute time axis, rows banded per "
        "chip, with both rooflines below. Without it each chip gets its own page, x normalised "
        "to that chip's own span — both views are kept because they answer different questions",
    )
    chip.add_argument("--weights", default=None, help="Precision; defaults per chip")
    chip.add_argument("--ideal", action="store_true", help="Both de-ratings at 1.0")

    dataflow = parser.add_argument_group(
        "dataflow strategy — only with the default matmul workload (docs/CLI.md §2.5)"
    )
    dataflow.add_argument(
        "--a-strategy",
        choices=[s.value for s in AStrategy],
        default=None,
        help="How A is loaded: stage (once per k-slice, D33, the default), stream "
        "(per tile, D31) or whole (all of A before the first tile)",
    )
    dataflow.add_argument(
        "--b-dataflow",
        choices=[d.value for d in BDataflow],
        default=None,
        help="When B's array write lands: write-ahead (a wave early, hidden behind "
        "compute, the default), on-demand (at compute, exposed) or persistent "
        "(once, never displaced)",
    )
    dataflow.add_argument(
        "--a-residency-tiles",
        type=int,
        default=None,
        help="Override tiles served per A staging event under stage/whole; must be a "
        "power-of-2 divisor of NTILES_PER_KS (clamped otherwise)",
    )
    dataflow.add_argument(
        "--a-prefetch-depth",
        type=int,
        default=None,
        help="Override the double-buffered staging depth for A. Schedule-only",
    )
    dataflow.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Invocations this report represents (default 1); only b_dataflow=persistent reads it",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--steps",
        type=int,
        default=256,
        help="Steps in the trace (default 256). The only resolution knob — the page "
        "zooms, so there is no separate static-figure register to keep legible",
    )
    output.add_argument("--out", type=Path, default=Path("../docs/plots"), help="Output directory")
    output.add_argument(
        "--animate",
        action="store_true",
        help="Also write a self-contained DRAM->SRAM->Accelerator flow animation "
        "(matmul only; opt-in, not part of `make plots`)",
    )
    return parser


def _reject_flags_for_the_wrong_workload(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Fill in the real defaults for the chosen workload, and refuse a flag
    scoped to a workload that was not chosen — a flag combination this script
    would otherwise silently ignore, which is worse than an error naming it
    (CLAUDE.md #8): `--ffn` with `--matmul` never reaches an encoder to apply
    to, and previously said nothing about that at all.
    """

    def given(**flags: object) -> list[str]:
        return [f"--{name.replace('_', '-')}" for name, value in flags.items() if value is not None]

    if args.encoder:
        # ffn's default depends on whether dmodel was actually typed, not on
        # its resolved value: the bare --encoder invocation has to keep
        # matching bwz encoder-layer's own bare defaults, hand-countable at
        # 664 params/5280 ops (D24) — the 4x-dmodel convention only applies
        # once dmodel was itself an explicit choice (D45).
        dmodel_given = args.dmodel is not None
        args.dmodel = 8 if args.dmodel is None else args.dmodel
        args.nheads = 2 if args.nheads is None else args.nheads
        if args.ffn is None:
            args.ffn = 4 * args.dmodel if dmodel_given else 16
        args.vocab = 16 if args.vocab is None else args.vocab
    else:
        bad = given(
            dmodel=args.dmodel,
            nheads=args.nheads,
            ffn=args.ffn,
            vocab=args.vocab,
        )
        if bad:
            verb = "applies" if len(bad) == 1 else "apply"
            them = "it" if len(bad) == 1 else "them"
            parser.error(
                f"{', '.join(bad)} only {verb} to --encoder; pass --encoder or drop {them}"
            )

    if args.model or args.encoder:
        args.tokens = 512 if args.tokens is None else args.tokens
    elif args.tokens is not None:
        parser.error("--tokens/-S only applies to --model / --encoder; pass one or drop it")

    if args.model or args.encoder:
        bad = given(
            a_strategy=args.a_strategy,
            b_dataflow=args.b_dataflow,
            a_residency_tiles=args.a_residency_tiles,
            a_prefetch_depth=args.a_prefetch_depth,
            iterations=args.iterations,
        )
        if bad:
            verb = "applies" if len(bad) == 1 else "apply"
            them = "it" if len(bad) == 1 else "them"
            parser.error(
                f"{', '.join(bad)} only {verb} to the default matmul workload, not "
                f"--model/--encoder; drop {them} or drop --model/--encoder (docs/CLI.md §2.5)"
            )
    else:
        args.a_strategy = args.a_strategy or AStrategy.STAGE.value
        args.b_dataflow = args.b_dataflow or BDataflow.WRITE_AHEAD.value
        args.iterations = 1 if args.iterations is None else args.iterations

    if args.animate and args.model:
        parser.error(
            "--animate does not support --model yet (only the default matmul and --encoder "
            "workloads); drop --model or drop --animate (docs/CLI.md §3)"
        )
    if args.animate and args.compare:
        parser.error("--animate draws one chip's schedule; drop --compare or drop --animate")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _reject_flags_for_the_wrong_workload(parser, args)

    m, n, k = (int(part) for part in args.matmul.split(","))
    command = "uv run --group plots python " + " ".join(shlex.quote(a) for a in sys.argv)

    def slug(prefix: str, dtype: DType, work: Workload, index: int) -> str:
        """File stem. A model gets one figure per phase, so the phase is in the name."""
        if args.encoder:
            return f"{prefix}-encoder-d{args.dmodel}-S{args.tokens}-{dtype.value}"
        if args.model is None:
            return f"{prefix}-{dtype.value}"
        phase = work.trace.spans[0].phase.value if work.trace.spans else str(index)
        return f"{prefix}-{args.model.replace('-', '_')}-{phase}-{dtype.value}"

    def workloads(chip: HardwareSpec, dtype: DType, steps: int) -> list[Workload]:
        """One workload per phase — one for a matmul, two for a decoder."""
        if args.encoder:
            return build_encoder(
                chip,
                dmodel=args.dmodel,
                nheads=args.nheads,
                ffn=args.ffn,
                vocab=args.vocab,
                tokens=args.tokens,
                dtype=dtype,
                steps=steps,
            )
        if args.model:
            return build_model(chip, args.model, args.tokens, dtype, steps)
        return [
            build_matmul(
                chip,
                m,
                n,
                k,
                dtype,
                steps,
                a_strategy=AStrategy(args.a_strategy),
                b_dataflow=BDataflow(args.b_dataflow),
                a_residency_tiles=args.a_residency_tiles,
                a_prefetch_depth=args.a_prefetch_depth,
                iterations=args.iterations,
            )
        ]

    chips = [load_chip(chip_id) for chip_id in args.chip or ["a100_80gb", "chip_a"]]
    if args.ideal:
        chips = [idealised(chip) for chip in chips]

    if args.compare:
        if len(chips) < 2:
            raise SystemExit(
                f"bwz: --compare puts two or more chips in one figure and got {len(chips)}; "
                f"pass --chip twice, or drop --compare for the per-chip view"
            )
        dtype = _shared_dtype(chips, args.weights)
        prefix = "compare-" + "-vs-".join(chip.id for chip in chips)

        def groups(steps: int) -> list[list[Panel]]:
            """Transposed: per phase, one panel per chip — so prefill is compared
            against prefill and decode against decode, never across."""
            per_chip = [workloads(chip, dtype, steps) for chip in chips]
            return [
                [Panel(chip, dtype, work) for chip, work in zip(chips, phase, strict=True)]
                for phase in zip(*per_chip, strict=True)
            ]

        for index, panels in enumerate(groups(args.steps)):
            write_html(
                panels,
                command,
                args.out / f"timeline-{slug(prefix, dtype, panels[0].work, index)}.html",
            )
        return

    for chip in chips:
        dtype = DType(args.weights) if args.weights else _default_dtype(chip)
        for index, work in enumerate(workloads(chip, dtype, args.steps)):
            panel = Panel(chip, dtype, work)
            write_html(
                [panel], command, args.out / f"timeline-{slug(chip.id, dtype, work, index)}.html"
            )
            if args.animate:
                write_animation_html(
                    panel, command, args.out / f"animate-{slug(chip.id, dtype, work, index)}.html"
                )


if __name__ == "__main__":
    main()
