"""Where the time went, on which piece of hardware — one figure per chip.

    uv run --group plots python scripts/plot_pipeline.py --chip a100_80gb

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
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from timeline_html import Box, render

import bwz
from bwz.analysis import analyze, idealised, machine_model
from bwz.analysis.pipeline import Lane, PipelineTrace, Span, build_trace
from bwz.graph import GraphPhase, build_graph
from bwz.spec import DeploymentSpec, DType, HardwareSpec, MatmulSpec, load_chip
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
COLOUR = {Lane.DRAM: "#2a78d6", Lane.SRAM: "#8a8983", Lane.CORE: "#eb6834"}
IDLE = "#e6e5e1"


@dataclass(frozen=True)
class Row:
    """One hardware resource, drawn whether or not this workload touches it."""

    title: str
    detail: str
    lane: Lane | None
    """The trace lane whose spans belong on this row, or None for a resource the
    model declares and never uses — the rows that show where it stops."""
    note: str = ""


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
    governing = max(
        (u for u in chip.compute_units if u.supports(dtype)),
        key=lambda u: u.peak_flops_per_s(chip.clock_hz, dtype),
    )

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
        if unit is governing:
            peak = unit.peak_flops_per_s(chip.clock_hz, dtype)
            rows.append(
                Row(
                    unit.name,
                    detail,
                    Lane.CORE,
                    f"peak {format_quantity(peak, 'OP/s')} at {dtype.value}",
                )
            )
        elif unit.supports(dtype):
            rows.append(
                Row(unit.name, detail, None, "idle — peak is the max over units, not the sum")
            )
        else:
            rows.append(Row(unit.name, detail, None, f"idle — no {dtype.value} datapath"))
    return rows


def _bars(ax: plt.Axes, rows: list[Row], spans: list[Span], window: tuple[float, float]) -> None:
    """One rectangle per span, on its resource's row, x normalised to *window*."""
    start, end = window
    width_s = (end - start) or 1.0
    for index, row in enumerate(rows):
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
        for span in spans:
            if span.lane is not row.lane:
                continue
            ax.add_patch(
                Rectangle(
                    ((span.start_s - start) / width_s, index + 0.18),
                    max(span.duration_s / width_s, 0.0015),
                    0.64,
                    facecolor=COLOUR[row.lane],
                    edgecolor=SURFACE,
                    linewidth=0.7,
                    zorder=3,
                )
            )


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


def _boxes(trace: PipelineTrace, chip: HardwareSpec, dtype: DType) -> list[Box]:
    """The three headline figures, shared by the PNG and the HTML."""
    totals, busy, concurrency = trace.totals, trace.busy_s, trace.concurrency
    span = trace.total_s or 1.0
    peak = machine_model(chip, dtype).peak_flops_per_s
    buffers = max(concurrency[Lane.SRAM][1], 1)
    dram_rate = format_bandwidth(totals[Lane.DRAM] / busy[Lane.DRAM]) if busy[Lane.DRAM] else "—"
    core_rate = totals[Lane.CORE] / busy[Lane.CORE] if busy[Lane.CORE] else 0.0
    return [
        Box(
            "dram",
            "COPIED FROM DRAM",
            format_bytes(totals[Lane.DRAM]),
            f"{dram_rate} while active\n"
            f"{format_time(busy[Lane.DRAM])} — {busy[Lane.DRAM] / span:.0%} of the span",
        ),
        Box(
            "sram",
            "HELD ON CHIP",
            format_bytes(totals[Lane.SRAM]),
            f"{buffers} buffers x {format_bytes(totals[Lane.SRAM] / buffers)}\n"
            f"of {format_bytes(chip.on_chip_capacity_bytes)} capacity",
        ),
        Box(
            "core",
            "COMPUTED",
            format_quantity(totals[Lane.CORE], "OP"),
            f"{format_quantity(core_rate, 'OP/s')} of {format_quantity(peak, 'OP/s')} peak\n"
            f"{format_time(busy[Lane.CORE])} — {busy[Lane.CORE] / span:.0%} of the span",
        ),
    ]


def write_html(
    chip: HardwareSpec,
    dtype: DType,
    trace: PipelineTrace,
    spec: MatmulSpec,
    command: str,
    out: Path,
) -> None:
    """The same figure, zoomable, as one self-contained file."""
    rows = rows_for(chip, dtype)
    page = render(
        title=f"{chip.name} — {spec.name} at {dtype.value}",
        subtitle=_subtitle(trace),
        footer=(
            f"bwz {bwz.__version__}{_git()} — every bar is a slice of the reported latency: "
            f"the DRAM row sums to t_dram and the compute row to t_compute. "
            f"<br><code>$ {command}</code>"
        ),
        boxes=_boxes(trace, chip, dtype),
        rows=[
            {
                "title": row.title,
                "detail": row.detail,
                "note": row.note,
                "lane": row.lane.value if row.lane else None,
                "quantity": _quantity(row, trace),
            }
            for row in rows
        ],
        spans=[
            {
                "lane": span.lane.value,
                "start": span.start_s,
                "end": span.end_s,
                "tip": _tip(span),
            }
            for span in trace.spans
        ],
        total_s=trace.total_s,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}")


def _tip(span: Span) -> str:
    """A span's own numbers, for the hover."""
    head = f"{span.label}\n{format_time(span.start_s)} + {format_time(span.duration_s)}"
    if span.lane is Lane.DRAM:
        return (
            f"{head}\n{format_bytes(span.bytes_moved)} @ {format_bandwidth(span.rate_bytes_per_s)}"
        )
    if span.lane is Lane.CORE:
        return (
            f"{head}\n{format_quantity(span.flops, 'OP')} @ "
            f"{format_quantity(span.rate_flops_per_s, 'OP/s')}"
        )
    return f"{head}\nholding {format_bytes(span.resident_bytes)}"


def _quantity(row: Row, trace: PipelineTrace) -> str:
    """What went through this resource. The part an instruction-centric view
    cannot give: a row means nothing for a comparison until it carries a number."""
    totals, busy, concurrency = trace.totals, trace.busy_s, trace.concurrency
    if row.lane is Lane.DRAM:
        if totals[Lane.DRAM] <= 0:
            return "0 B — nothing crossed"
        rate = totals[Lane.DRAM] / busy[Lane.DRAM] if busy[Lane.DRAM] else 0.0
        return f"{format_bytes(totals[Lane.DRAM])} @ {format_bandwidth(rate)}"
    if row.lane is Lane.SRAM:
        mean, peak = concurrency[Lane.SRAM]
        return f"{format_bytes(totals[Lane.SRAM])} in {peak} buffers (x{mean:.2f} avg)"
    if row.lane is Lane.CORE:
        rate = totals[Lane.CORE] / busy[Lane.CORE] if busy[Lane.CORE] else 0.0
        return f"{format_quantity(totals[Lane.CORE], 'OP')} @ {format_quantity(rate, 'OP/s')}"
    return "not used"


def _info_boxes(
    fig: plt.Figure,
    trace: PipelineTrace,
    chip: HardwareSpec,
    dtype: DType,
    top_in: float,
    fig_h: float,
) -> None:
    """The three numbers a comparison turns on: what was copied, what was held,
    what was computed — each with its rate and its share of the span."""
    totals, busy, concurrency = trace.totals, trace.busy_s, trace.concurrency
    span = trace.total_s or 1.0
    peak = machine_model(chip, dtype).peak_flops_per_s
    buffers = max(concurrency[Lane.SRAM][1], 1)
    core_rate = totals[Lane.CORE] / busy[Lane.CORE] if busy[Lane.CORE] else 0.0

    boxes = [
        (
            COLOUR[Lane.DRAM],
            "COPIED FROM DRAM",
            format_bytes(totals[Lane.DRAM]),
            f"{format_bandwidth(totals[Lane.DRAM] / busy[Lane.DRAM]) if busy[Lane.DRAM] else '—'}"
            f" while active\n{format_time(busy[Lane.DRAM])} — {busy[Lane.DRAM] / span:.0%}"
            f" of the span",
        ),
        (
            COLOUR[Lane.SRAM],
            "HELD ON CHIP",
            format_bytes(totals[Lane.SRAM]),
            f"{buffers} buffers x {format_bytes(totals[Lane.SRAM] / buffers)}\n"
            f"of {format_bytes(chip.on_chip_capacity_bytes)} capacity",
        ),
        (
            COLOUR[Lane.CORE],
            "COMPUTED",
            format_quantity(totals[Lane.CORE], "OP"),
            f"{format_quantity(core_rate, 'OP/s')} of {format_quantity(peak, 'OP/s')} peak\n"
            f"{format_time(busy[Lane.CORE])} — {busy[Lane.CORE] / span:.0%} of the span",
        ),
    ]

    box_h = 1.05
    for index, (colour, heading, headline, detail) in enumerate(boxes):
        x = 0.038 + index * 0.312

        def at(offset_in: float, base: float = top_in) -> float:
            return 1.0 - (base + offset_in) / fig_h

        fig.patches.append(
            FancyBboxPatch(
                (x, at(box_h)),
                0.286,
                box_h / fig_h,
                boxstyle="round,pad=0.004,rounding_size=0.006",
                transform=fig.transFigure,
                facecolor=BOX,
                edgecolor=colour,
                linewidth=1.4,
                zorder=1,
            )
        )
        fig.text(x + 0.014, at(0.26), heading, fontsize=8, color=colour, fontweight="bold")
        fig.text(x + 0.014, at(0.62), headline, fontsize=17, color=INK, fontweight="bold")
        fig.text(
            x + 0.014,
            at(0.74),
            detail,
            fontsize=8,
            color=INK_SECONDARY,
            linespacing=1.5,
            va="top",
        )


def draw(
    chip: HardwareSpec,
    dtype: DType,
    trace: PipelineTrace,
    spec: MatmulSpec,
    command: str,
    out: Path,
    zoom_steps: int,
) -> None:
    """Lay the figure out in inches and convert once.

    Row count varies with the chip — A100 declares five resources, chip_a three —
    so every vertical position is derived from it rather than guessed as a
    fraction, which is what stops the second register from landing on the footer.
    """
    rows = rows_for(chip, dtype)
    row_h, header, gap, footer = 0.48, 3.05, 1.15, 1.05
    band = row_h * len(rows)
    fig_w, fig_h = 13.0, header + band + gap + band + footer
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=160, facecolor=SURFACE)

    def y(inches_from_top: float) -> float:
        return 1.0 - inches_from_top / fig_h

    left, width = 2.55 / fig_w, (fig_w - 2.55 - 2.85) / fig_w
    top = fig.add_axes((left, y(header + band), width, band / fig_h))
    bottom = fig.add_axes((left, y(header + 2 * band + gap), width, band / fig_h))

    zoom = min(zoom_steps, trace.steps)
    cutoff = max((s.end_s for s in trace.spans if s.step < zoom), default=trace.total_s)

    _bars(top, rows, list(trace.spans), (0.0, trace.total_s))
    _row_axis(top, rows)
    _ticks(top, trace.total_s, 5)
    _bars(bottom, rows, [s for s in trace.spans if s.start_s < cutoff], (0.0, cutoff))
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
                fontsize=9.5,
                color=INK,
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
            _quantity(row, trace),
            transform=top.get_yaxis_transform(),
            va="center",
            fontsize=8.5,
            color=INK if row.lane is not None else INK_MUTED,
        )

    fig.text(
        0.038,
        y(0.50),
        f"{chip.name} — {spec.name} at {dtype.value}",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    fig.text(0.038, y(0.84), _subtitle(trace), fontsize=10, color=INK_SECONDARY)
    _info_boxes(fig, trace, chip, dtype, 1.15, fig_h)
    fig.text(
        0.038,
        y(header - 0.52),
        "Every resource the profile declares — the whole run",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.038,
        y(header - 0.28),
        "Grey rows are declared by the chip and unused by this model: that is where its "
        "boundary lies.",
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
        "A load and the previous tile's arithmetic overlap exactly as far as capacity "
        "allowed a second buffer.",
        fontsize=8.5,
        color=INK_MUTED,
    )

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


def _subtitle(trace: PipelineTrace) -> str:
    steps = (
        f"{trace.steps} steps drawn, coalesced from {trace.tiles} tiles. "
        if trace.coalesced
        else f"{trace.steps} tile steps. "
    )
    if not trace.double_buffered:
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


def build(
    chip: HardwareSpec, m: int, n: int, k: int, dtype: DType, steps: int
) -> tuple[PipelineTrace, MatmulSpec]:
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
    trace = build_trace(
        build_graph(spec, deployment, GraphPhase.STATIC),
        report.phases[0],
        machine_model(chip, spec.operand_dtype),
        double_buffered=report.memory.double_buffered,
        max_steps=steps,
    )
    return trace, spec


def _default_dtype(chip: HardwareSpec) -> DType:
    for candidate in (DType.FP16, DType.INT8, DType.BF16, DType.FP32):
        if chip.supports(candidate):
            return candidate
    return chip.compute_units[0].supported_dtypes[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chip", action="append", default=None, help="Chip id; repeatable")
    parser.add_argument("--matmul", default="4096,4096,4096", metavar="M,N,K")
    parser.add_argument("--weights", default=None, help="Precision; defaults per chip")
    parser.add_argument("--ideal", action="store_true", help="Both de-ratings at 1.0")
    parser.add_argument("--steps", type=int, default=32, help="Steps to draw")
    parser.add_argument("--zoom", type=int, default=6, help="Steps in the zoomed register")
    parser.add_argument("--html", action="store_true", help="Also write the zoomable HTML timeline")
    parser.add_argument("--out", type=Path, default=Path("../docs/plots"))
    args = parser.parse_args()

    m, n, k = (int(part) for part in args.matmul.split(","))
    command = "uv run --group plots python " + " ".join(shlex.quote(a) for a in sys.argv)

    for chip_id in args.chip or ["a100_80gb", "chip_a"]:
        chip = load_chip(chip_id)
        if args.ideal:
            chip = idealised(chip)
        dtype = DType(args.weights) if args.weights else _default_dtype(chip)
        trace, spec = build(chip, m, n, k, dtype, args.steps)
        draw(
            chip,
            dtype,
            trace,
            spec,
            command,
            args.out / f"timeline-{chip.id}-{dtype.value}.png",
            args.zoom,
        )
        if args.html:
            write_html(
                chip,
                dtype,
                trace,
                spec,
                command,
                args.out / f"timeline-{chip.id}-{dtype.value}.html",
            )


if __name__ == "__main__":
    main()
