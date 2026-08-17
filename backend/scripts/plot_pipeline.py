"""Where the time went, on which piece of hardware — one figure per chip, or one
figure for two chips.

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

Two registers: the whole run at total scale, and the first steps zoomed, since at
total scale one step of a 65 536-tile matmul is a hairline.

**``--compare`` puts every ``--chip`` in one figure** (docs/CORRECTIONS.md D29).
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
import math
import shlex
import subprocess
import sys
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from timeline_html import Box, render

import bwz
from bwz.analysis import analyze, idealised, machine_model
from bwz.analysis.pipeline import Lane, PipelineTrace, Span, Stage, build_trace
from bwz.analysis.roofline import MATRIX_OP_TYPES, compute_dtype
from bwz.explain import Explanation, explain_graph
from bwz.graph import GraphPhase, build_graph, build_graphs
from bwz.report import Bound
from bwz.spec import (
    AnyModelSpec,
    DeploymentSpec,
    DType,
    HardwareSpec,
    MatmulSpec,
    TransformerSpec,
    load_chip,
    load_model,
)
from bwz.units import format_bandwidth, format_bytes, format_quantity, format_time

matplotlib.use("Agg")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e6e5e1"
BOX = "#f2f1ed"
# Slots 1 and 2 of the documented categorical palette, plus a neutral for the
# buffer. Three roles, not eight, and every row is directly labelled.
COLOUR = {
    Lane.DRAM: "#2a78d6",
    Lane.SRAM: "#8a8983",
    Lane.CORE: "#eb6834",
    Lane.VECTOR: "#1baf7a",  # slot 3: a third engine, not a shade of the array
}
IDLE = "#e6e5e1"

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
                        f"{format_time(panel.work.trace.total_s)} · "
                        f"{panel.work.bound.value.replace('_', ' ').lower()}"
                    ),
                    colour=CHIP_COLOURS[index % len(CHIP_COLOURS)],
                )
            )
        for row in rows_for(panel.chip, panel.dtype):
            rows.append(replace(row, panel=index))
    return rows


def _bars(
    ax: plt.Axes, rows: list[Row], span_lists: list[list[Span]], window: tuple[float, float]
) -> None:
    """One rectangle per span, on its resource's row, x normalised to *window*.

    On a comparison *window* is the same absolute interval for every panel — the
    whole point being to see that one span is several times the other — and each
    row draws only its own panel's spans.
    """
    start, end = window
    width_s = (end - start) or 1.0
    for index, row in enumerate(rows):
        if row.header:
            ax.add_patch(
                Rectangle(
                    (0, index + 0.90),
                    1,
                    0.045,
                    facecolor=row.colour or INK_MUTED,
                    edgecolor="none",
                    zorder=2,
                )
            )
            continue
        if row.lane is None:
            ax.add_patch(
                Rectangle(
                    (0, index + 0.34),
                    1,
                    0.32,
                    facecolor=IDLE,
                    edgecolor="none",
                    zorder=1,
                )
            )
            continue
        for span in span_lists[row.panel]:
            if span.lane is not row.lane:
                continue
            # Stores are drawn hollow so the direction of DRAM traffic is
            # visible at a glance: filled bars bring operands in, outlined bars
            # take results out, and they never overlap because it is one port.
            store = span.stage is Stage.STORE
            dispatch = span.stage is Stage.DISPATCH
            x0 = (span.start_s - start) / width_s
            width = max(span.duration_s / width_s, 0.0015)
            ax.add_patch(
                Rectangle(
                    (x0, index + 0.18),
                    width,
                    0.64,
                    facecolor=SURFACE if store else COLOUR[row.lane],
                    edgecolor=COLOUR[row.lane] if (store or dispatch) else SURFACE,
                    linewidth=1.1 if (store or dispatch) else 0.7,
                    hatch="///" if dispatch else None,
                    alpha=0.5 if dispatch else 1.0,
                    zorder=4 if store else 3,
                )
            )
            # Name the bar when there is room for it. On a small graph this is
            # the difference between "something happened" and "q_proj happened",
            # and on a large one no bar is ever wide enough so nothing is drawn.
            if width > 0.05:
                ax.text(
                    x0 + width / 2,
                    index + 0.5,
                    _short(span.label),
                    ha="center",
                    va="center",
                    fontsize=6.4,
                    color=INK if (store or dispatch) else SURFACE,
                    zorder=6,
                )


def _short(label: str) -> str:
    """Bar text: the operation, without the layer prefix that every bar shares."""
    head = label.split("  ")[0]
    return head.rsplit(".", 1)[-1] if "." in head else head


def _row_axis(ax: plt.Axes, rows: list[Row]) -> None:
    ax.set_ylim(len(rows), 0)
    ax.set_xlim(0, 1)
    ax.set_yticks([i + 0.5 for i in range(len(rows))])
    ax.set_yticklabels([""] * len(rows))  # titles are drawn explicitly, with their detail
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.5, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def _ticks(ax: plt.Axes, end: float, count: int) -> None:
    fractions = [i / (count - 1) for i in range(count)]
    ax.set_xticks(fractions)
    ax.set_xticklabels([format_time(f * end) for f in fractions], fontsize=8.5)


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
        band = f"{chip.name} — {format_time(trace.total_s)}" if banded else ""
        out += [
            Box(
                "dram",
                "MOVED OVER DRAM",
                format_bytes(totals[Lane.DRAM]),
                f"LOAD {format_bytes(trace.direction_bytes[0])} · "
                f"STORE {format_bytes(trace.direction_bytes[1])}\n{rate} while active\n"
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
                f"array {format_quantity(totals[Lane.CORE], 'OP')} @ "
                f"{format_quantity(core_rate, 'OP/s')} of {format_quantity(peak, 'OP/s')}\n"
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
                "tip": _tip(span),
                "panel": index,
            }
            for index, panel in enumerate(panels)
            for span in panel.work.trace.spans
        ],
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
        kind = "STORE — result written back" if span.stage is Stage.STORE else "LOAD — operands in"
        return (
            f"{kind}\n{span.label}\n{when}\n"
            f"{format_bytes(span.bytes_moved)} @ {format_bandwidth(span.rate_bytes_per_s)}"
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
        reads, writes = trace.direction_bytes
        return (
            f"{format_bytes(totals[Lane.DRAM])} @ {format_bandwidth(rate)}\n"
            f"LOAD {format_bytes(reads)} · STORE {format_bytes(writes)}"
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


BOX_H = 1.30
BAND_LABEL_H = 0.34
BAND_GAP = 0.22


def _boxes_region_in(count: int, banded: bool) -> float:
    """Inches the info-box block occupies for *count* chips."""
    pitch = BAND_LABEL_H + BOX_H + BAND_GAP if banded else BOX_H
    return count * pitch - (BAND_GAP if banded else 0.0)


def _info_boxes(
    fig: plt.Figure,
    boxes: list[Box],
    top_in: float,
    fig_h: float,
    band_colours: dict[str, str],
) -> None:
    """The three numbers a comparison turns on: what was copied, what was held,
    what was computed — each with its rate and its share of the span.

    Drawn from the very :class:`Box` objects the HTML page renders, so the PNG and
    the page cannot quote different figures for the same run. On a comparison the
    boxes come in one band per chip, each band introduced by the machine's name.
    """
    banded = any(box.band for box in boxes)
    pitch = BAND_LABEL_H + BOX_H + BAND_GAP if banded else BOX_H
    bands: list[str] = []
    for box in boxes:
        if box.band not in bands:
            bands.append(box.band)

    for box_index, box in enumerate(boxes):
        band_index = bands.index(box.band)
        column = box_index % 3
        x = 0.038 + column * 0.312
        base = top_in + band_index * pitch + (BAND_LABEL_H if banded else 0.0)
        colour = COLOUR[Lane(box.lane)]

        def at(offset_in: float, anchor: float = 0.0) -> float:
            return 1.0 - (anchor + offset_in) / fig_h

        if banded and column == 0:
            fig.text(
                0.038,
                at(base - 0.12),
                box.band,
                fontsize=9.5,
                color=band_colours.get(box.band, INK),
                fontweight="bold",
            )
        fig.patches.append(
            FancyBboxPatch(
                (x, at(BOX_H, base)),
                0.286,
                BOX_H / fig_h,
                boxstyle="round,pad=0.004,rounding_size=0.006",
                transform=fig.transFigure,
                facecolor=BOX,
                edgecolor=colour,
                linewidth=1.4,
                zorder=1,
            )
        )
        fig.text(
            x + 0.014, at(0.26, base), box.heading, fontsize=8, color=colour, fontweight="bold"
        )
        fig.text(x + 0.014, at(0.62, base), box.headline, fontsize=17, color=INK, fontweight="bold")
        fig.text(
            x + 0.014,
            at(0.74, base),
            box.detail,
            fontsize=8,
            color=INK_SECONDARY,
            linespacing=1.5,
            va="top",
        )


def draw(panels: list[Panel], command: str, out: Path, zoom_steps: int) -> None:
    """Lay the figure out in inches and convert once.

    Row count varies with the chip — A100 declares five resources, Metis six,
    ``chip_a`` three — so every vertical position is derived from it rather than
    guessed as a fraction, which is what stops the second register from landing
    on the footer. With more than one panel the row count is the sum plus a band
    header each, and the x window becomes the **same absolute interval** for
    every chip instead of each chip's own span.
    """
    panels = list(panels)
    compare = len(panels) > 1
    rows = panel_rows(panels)
    span_lists = [list(p.work.trace.spans) for p in panels]
    total = max(p.work.trace.total_s for p in panels)
    boxes = _boxes(panels)
    band_colours = {
        box.band: CHIP_COLOURS[i % len(CHIP_COLOURS)]
        for i, box in enumerate(boxes[::3])
        if box.band
    }

    row_h, gap, footer = 0.48, 1.15, 1.05
    header = 1.15 + _boxes_region_in(len(panels), compare) + 0.85
    band = row_h * len(rows)
    roof_h = 4.30 if compare else 0.0
    fig_w, fig_h = 13.0, header + band + gap + band + roof_h + footer
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=160, facecolor=SURFACE)

    def y(inches_from_top: float) -> float:
        return 1.0 - inches_from_top / fig_h

    left, width = 2.55 / fig_w, (fig_w - 2.55 - 2.85) / fig_w
    top = fig.add_axes((left, y(header + band), width, band / fig_h))
    bottom = fig.add_axes((left, y(header + 2 * band + gap), width, band / fig_h))

    # "The first N steps" of each chip, on one window: the union, so the slower
    # machine's steps are not cropped away to fit the faster one's.
    zoom = min(zoom_steps, max(p.work.trace.steps for p in panels))
    cutoff = max(
        max((s.end_s for s in spans if s.step < zoom), default=trace.total_s)
        for spans, trace in zip(span_lists, (p.work.trace for p in panels), strict=True)
    )

    _bars(top, rows, span_lists, (0.0, total))
    _row_axis(top, rows)
    _ticks(top, total, 5)
    _bars(
        bottom,
        rows,
        [[s for s in spans if s.start_s < cutoff] for spans in span_lists],
        (0.0, cutoff),
    )
    _row_axis(bottom, rows)
    _ticks(bottom, cutoff, 3)

    for axes in (top, bottom):
        for index, row in enumerate(rows):
            centre = index + 0.5
            axes.text(
                -0.012,
                centre - 0.03,
                row.title,
                transform=axes.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=11 if row.header else 9.5,
                color=row.colour if row.header else INK,
                fontweight="bold",
            )
            axes.text(
                -0.012,
                centre + 0.04,
                f"{row.detail}\n{row.note}",
                transform=axes.get_yaxis_transform(),
                ha="right",
                va="top",
                fontsize=6.6,
                color=INK_MUTED,
                linespacing=1.3,
            )
    for index, row in enumerate(rows):
        top.text(
            1.015,
            index + 0.5,
            _quantity(row, panels[row.panel].work.trace),
            transform=top.get_yaxis_transform(),
            va="center",
            fontsize=8.5,
            color=INK if (row.lane is not None or row.header) else INK_MUTED,
            fontweight="bold" if row.header else "normal",
        )

    fig.text(0.038, y(0.50), _title(panels), fontsize=16, fontweight="bold", color=INK)
    fig.text(0.038, y(0.84), _subtitle(panels), fontsize=10, color=INK_SECONDARY)
    _info_boxes(fig, boxes, 1.15, fig_h, band_colours)
    fig.text(
        0.038,
        y(header - 0.52),
        (
            "Every resource both profiles declare — the whole run, one absolute axis"
            if compare
            else "Every resource the profile declares — the whole run"
        ),
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.038,
        y(header - 0.28),
        (
            "Rows are banded by chip because two profiles declare different resources; the time "
            "axis is shared, so bar lengths are directly comparable. Grey rows are declared and "
            "unused by this model."
            if compare
            else "Grey rows are declared by the chip and unused by this model: that is where its "
            "boundary lies."
        ),
        fontsize=8.5,
        color=INK_MUTED,
    )
    fig.text(
        0.038,
        y(header + band + gap - 0.52),
        f"First {zoom} steps — {format_time(cutoff)} of it",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.038,
        y(header + band + gap - 0.28),
        (
            "One bar per operation: dispatch, then its loads, arithmetic and stores."
            if panels[0].work.trace.kind == "operations"
            else "A load and the previous tile's arithmetic overlap exactly as far as capacity "
            "allowed a second buffer."
        ),
        fontsize=8.5,
        color=INK_MUTED,
    )
    if compare:
        _draw_roofline(fig, roofs_for(panels), header + 2 * band + gap, fig_w, fig_h)

    for offset, line in enumerate(textwrap.wrap(f"$ {command}", width=132)):
        fig.text(
            0.038,
            y(fig_h - 0.62 + 0.20 * offset),
            line,
            fontsize=8.5,
            color=INK_SECONDARY,
            family="monospace",
        )
    fig.text(
        0.038,
        y(fig_h - 0.22),
        f"bwz {bwz.__version__}{_git()} — every bar is a slice of the reported latency: "
        f"the DRAM row sums to t_dram and the compute row to t_compute.",
        fontsize=8,
        color=INK_MUTED,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def _title(panels: list[Panel]) -> str:
    work = panels[0].work
    if len(panels) == 1:
        return f"{panels[0].chip.name} — {work.name} at {panels[0].dtype.value}"
    chips = " vs ".join(panel.chip.name for panel in panels)
    return f"{work.name} at {panels[0].dtype.value} — {chips}"


def _past(value: float, limits: dict[str, float], fraction: float) -> bool:
    """True when *value* sits past *fraction* of the way across the log x axis —
    the test for flipping a label inward before it runs off the chart."""
    lo, hi = math.log10(limits["x_lo"]), math.log10(limits["x_hi"])
    return math.log10(value) > lo + fraction * (hi - lo)


def _draw_roofline(
    fig: plt.Figure, roofs: list[Roof], top_in: float, fig_w: float, fig_h: float
) -> None:
    """Both machines' ceilings on one log-log chart, with each one's point on it.

    This is the panel that says *why* the timeline above looks the way it does:
    two chips can move the same bytes and do the same arithmetic and still land
    on opposite sides of their own ridge point.
    """
    limits = _roofline_limits(roofs)
    heading = top_in + 0.30
    ax = fig.add_axes(
        (
            2.55 / fig_w,
            1.0 - (top_in + 0.75 + 3.05) / fig_h,
            5.30 / fig_w,
            3.05 / fig_h,
        )
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits["x_lo"], limits["x_hi"])
    ax.set_ylim(limits["y_lo"], limits["y_hi"])
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.set_xlabel(
        "arithmetic intensity — OP per byte of DRAM traffic", fontsize=8.5, color=INK_SECONDARY
    )
    ax.set_ylabel("achieved OP/s", fontsize=8.5, color=INK_SECONDARY)

    for index, roof in enumerate(roofs):
        for peak, bandwidth, style in (
            (roof.peak, roof.bandwidth, "-"),
            (roof.derated_peak, roof.derated_bandwidth, "--"),
        ):
            if style == "--" and (peak, bandwidth) == (roof.peak, roof.bandwidth):
                continue  # the profile does not derate: one roof, and the docs say why
            knee = peak / bandwidth
            ax.plot(
                [limits["x_lo"], knee, limits["x_hi"]],
                [max(bandwidth * limits["x_lo"], limits["y_lo"]), peak, peak],
                style,
                color=roof.colour,
                lw=1.9,
                zorder=3,
            )
        # Labels flip to the inside near the right edge. Both chips run the same
        # workload, so their points share an x and only the achieved rate
        # separates them — hence the per-chip vertical stagger as well.
        crowded = _past(roof.ridge, limits, 0.55)
        ax.axvline(roof.ridge, color=roof.colour, ls=":", lw=1.0, zorder=2)
        ax.text(
            roof.ridge,
            limits["y_hi"],
            f"{roof.name} ridge {roof.ridge:,.0f} "
            if crowded
            else f" {roof.name} ridge {roof.ridge:,.0f}",
            fontsize=7.4,
            color=roof.colour,
            va="top",
            ha="right" if crowded else "left",
        )
        if roof.tail:
            ax.axhline(roof.tail, color=roof.colour, ls=":", lw=0.9, alpha=0.7, zorder=2)
            # Left edge: the right half is where both ridge lines and both point
            # labels already are.
            ax.text(
                limits["x_lo"],
                roof.tail,
                f" M=1 ceiling {format_quantity(roof.tail, 'OP/s')}",
                fontsize=7.4,
                color=roof.colour,
                va="bottom",
                ha="left",
            )
        ax.plot(
            [roof.intensity],
            [roof.achieved],
            "o",
            color=roof.colour,
            ms=8,
            mec=SURFACE,
            mew=1.5,
            zorder=5,
        )
        inside = _past(roof.intensity, limits, 0.62)
        ax.annotate(
            f"{roof.name} — {roof.label}",
            (roof.intensity, roof.achieved),
            textcoords="offset points",
            xytext=(-11 if inside else 11, -3 + index * 13),
            ha="right" if inside else "left",
            fontsize=8,
            color=roof.colour,
            fontweight="bold",
            annotation_clip=False,
        )

    fig.text(
        0.038,
        1.0 - heading / fig_h,
        "Why — both rooflines on one chart",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.038,
        1.0 - (heading + 0.24) / fig_h,
        "Colour is the chip here, not the resource. Solid roof = datasheet, dashed = after the "
        "unfitted calibration constants.",
        fontsize=8.5,
        color=INK_MUTED,
    )
    notes = "\n\n".join(
        f"{roof.name}\n"
        f"  peak      {format_quantity(roof.peak, 'OP/s')}\n"
        f"  bandwidth {format_bandwidth(roof.bandwidth)}\n"
        f"  ridge     {roof.ridge:,.0f} OP/byte\n"
        f"  this run  {roof.intensity:,.1f} OP/byte → {roof.label}"
        for roof in roofs
    )
    fig.text(
        (2.55 + 5.30 + 0.55) / fig_w,
        1.0 - (top_in + 0.75) / fig_h,
        notes,
        fontsize=8,
        color=INK_SECONDARY,
        family="monospace",
        va="top",
        linespacing=1.45,
    )


def _subtitle(panels: list[Panel]) -> str:
    """One chip: the span and how it was scheduled. Two: that, plus the ratio the
    figure exists to show, stated rather than left to be measured off the axis."""
    if len(panels) == 1:
        return _trace_subtitle(panels[0].work.trace)
    ordered = sorted(panels, key=lambda p: p.work.trace.total_s)
    fastest, slowest = ordered[0], ordered[-1]
    ratio = slowest.work.trace.total_s / (fastest.work.trace.total_s or 1.0)
    spans = " · ".join(
        f"{panel.chip.id} {format_time(panel.work.trace.total_s)}" for panel in panels
    )
    return (
        f"Shared absolute time axis. {spans}. "
        f"{fastest.chip.id} is {ratio:.2f}x faster than {slowest.chip.id} on this workload."
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


def build_matmul(chip: HardwareSpec, m: int, n: int, k: int, dtype: DType, steps: int) -> Workload:
    spec = MatmulSpec.model_validate(
        {
            "id": "p",
            "name": f"matmul {m}x{n}x{k}",
            "family": "matmul",
            "m": m,
            "n": n,
            "k": k,
            "a_dtype": dtype,
            "b_dtype": dtype,
        }
    )
    deployment = DeploymentSpec.model_validate({"batch": 1, "input_tokens": 1, "output_tokens": 1})
    report = analyze(spec, chip, deployment)
    if not report.feasible:
        raise SystemExit(f"bwz: infeasible on {chip.id}: {report.infeasibility[0]}")
    graph = build_graph(spec, deployment, GraphPhase.STATIC)
    trace = build_trace(
        graph,
        report.phases[0],
        machine_model(chip, spec.operand_dtype),
        double_buffered=report.memory.double_buffered,
        max_steps=steps,
    )
    op = report.phases[0].ops[0]
    return Workload(
        spec.name, trace, explain_graph(graph), op.flops, op.dram_bytes, op.latency_s, op.bound
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
    hidden: int,
    heads: int,
    head_dim: int | None,
    ffn: int,
    vocab: int,
    tokens: int,
    dtype: DType,
    steps: int,
) -> list[Workload]:
    """A single-layer encoder built from dimensions, mirroring `bwz single-layer-encoder`.

    The same reason that command exists: a shape you can change one term of and
    watch the picture move, without writing a profile for every experiment.
    """
    spec = TransformerSpec.model_validate(
        {
            "id": "single_layer_encoder_cli",
            "name": f"1-layer encoder d={hidden} h={heads} ffn={ffn} S={tokens}",
            "family": "transformer_encoder",
            "hypothetical": True,
            "params": {
                "layers": 1,
                "hidden": hidden,
                "heads": heads,
                "head_dim": head_dim,
                "ffn_hidden": ffn,
                "ffn_type": "relu",
                "vocab": vocab,
                "max_context": max(tokens, 1),
                "norm": "rmsnorm",
                "positional": "none",
                "tie_embeddings": True,
            },
        }
    )
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chip", action="append", default=None, help="Chip id; repeatable")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Draw every --chip in ONE figure on a shared, absolute time axis, rows banded per "
        "chip, with both rooflines below. Without it each chip gets its own figure, x normalised "
        "to that chip's own span — both views are kept because they answer different questions",
    )
    parser.add_argument("--matmul", default="4096,4096,4096", metavar="M,N,K")
    parser.add_argument(
        "--model", default=None, help="Draw a model instead of a matmul; one figure per phase"
    )
    parser.add_argument(
        "--tokens", "-S", type=int, default=512, help="Sequence length for --model / --encoder"
    )
    parser.add_argument(
        "--encoder",
        action="store_true",
        help="Draw a single-layer encoder built from the shape flags below, not a profile",
    )
    parser.add_argument("--hidden", type=int, default=8, help="--encoder: model width")
    parser.add_argument("--heads", type=int, default=2, help="--encoder: attention heads")
    parser.add_argument("--head-dim", type=int, default=None, help="--encoder: defaults to d/h")
    parser.add_argument("--ffn", type=int, default=16, help="--encoder: FFN inner width")
    parser.add_argument("--vocab", type=int, default=16, help="--encoder: vocabulary")
    parser.add_argument("--weights", default=None, help="Precision; defaults per chip")
    parser.add_argument("--ideal", action="store_true", help="Both de-ratings at 1.0")
    parser.add_argument("--steps", type=int, default=32, help="Steps to draw")
    parser.add_argument("--zoom", type=int, default=6, help="Steps in the zoomed register")
    parser.add_argument("--html", action="store_true", help="Also write the zoomable HTML timeline")
    parser.add_argument(
        "--html-steps",
        type=int,
        default=256,
        help="Steps in the HTML trace (default 256). Higher than --steps on purpose: a static "
        "figure has to stay legible at one scale and a zoomable one does not",
    )
    parser.add_argument("--out", type=Path, default=Path("../docs/plots"))
    args = parser.parse_args()

    m, n, k = (int(part) for part in args.matmul.split(","))
    command = "uv run --group plots python " + " ".join(shlex.quote(a) for a in sys.argv)

    def slug(prefix: str, dtype: DType, work: Workload, index: int) -> str:
        """File stem. A model gets one figure per phase, so the phase is in the name."""
        if args.encoder:
            return f"{prefix}-encoder-d{args.hidden}-S{args.tokens}-{dtype.value}"
        if args.model is None:
            return f"{prefix}-{dtype.value}"
        phase = work.trace.spans[0].phase.value if work.trace.spans else str(index)
        return f"{prefix}-{args.model.replace('-', '_')}-{phase}-{dtype.value}"

    def workloads(chip: HardwareSpec, dtype: DType, steps: int) -> list[Workload]:
        """One workload per phase — one for a matmul, two for a decoder."""
        if args.encoder:
            return build_encoder(
                chip,
                hidden=args.hidden,
                heads=args.heads,
                head_dim=args.head_dim,
                ffn=args.ffn,
                vocab=args.vocab,
                tokens=args.tokens,
                dtype=dtype,
                steps=steps,
            )
        if args.model:
            return build_model(chip, args.model, args.tokens, dtype, steps)
        return [build_matmul(chip, m, n, k, dtype, steps)]

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
            draw(
                panels,
                command,
                args.out / f"timeline-{slug(prefix, dtype, panels[0].work, index)}.png",
                args.zoom,
            )
        if args.html:
            for index, panels in enumerate(groups(args.html_steps)):
                write_html(
                    panels,
                    command,
                    args.out / f"timeline-{slug(prefix, dtype, panels[0].work, index)}.html",
                )
        return

    for chip in chips:
        dtype = DType(args.weights) if args.weights else _default_dtype(chip)
        for index, work in enumerate(workloads(chip, dtype, args.steps)):
            draw(
                [Panel(chip, dtype, work)],
                command,
                args.out / f"timeline-{slug(chip.id, dtype, work, index)}.png",
                args.zoom,
            )
        if args.html:
            # A separate, finer trace: the PNG stays readable at 32 steps while
            # the page has something to zoom into.
            for index, work in enumerate(workloads(chip, dtype, args.html_steps)):
                write_html(
                    [Panel(chip, dtype, work)],
                    command,
                    args.out / f"timeline-{slug(chip.id, dtype, work, index)}.html",
                )


if __name__ == "__main__":
    main()
