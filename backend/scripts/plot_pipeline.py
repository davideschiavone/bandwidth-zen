"""Draw the resource schedule behind a chip's latency — the Konata view, offline.

One figure per chip, because the schedule is a property of the machine: the same
matmul is a compute-bound staircase on A100 and a DRAM-starved one on an edge NPU
with a tenth the bandwidth, and that difference is the thing worth seeing.

    uv run --group plots python scripts/plot_pipeline.py --chip a100_80gb

Two registers, stacked:

- **the whole run**, one bar per lane at total scale — where the time went;
- **the first few steps**, zoomed, so the double-buffered overlap is visible at
  all. At total scale a tile step of a 390 625-tile matmul is a hairline.

The x-axis is normalised to the total time, which is what makes two chips
comparable at a glance: the shapes can be laid side by side even when the
absolute times differ by three orders of magnitude. Absolute figures are printed
on the bars.

Same data as ``--kanata``; this is the version that does not need Konata
installed.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import bwz
from bwz.analysis import analyze, idealised, machine_model
from bwz.analysis.pipeline import Lane, PipelineTrace, Span, build_trace
from bwz.graph import GraphPhase, build_graph
from bwz.kanata import to_kanata
from bwz.spec import DeploymentSpec, DType, HardwareSpec, MatmulSpec, load_chip
from bwz.units import format_bytes, format_quantity, format_time

matplotlib.use("Agg")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e6e5e1"
# Slots 1 and 2 of the documented categorical palette, plus a muted neutral for
# the buffer. Three lanes, and every bar is directly labelled, so colour is
# reinforcing identity rather than carrying it alone.
LANE_COLOUR = {Lane.DRAM: "#2a78d6", Lane.SRAM: "#8a8983", Lane.CORE: "#eb6834"}
LANE_TITLE = {
    Lane.DRAM: "DRAM",
    Lane.SRAM: "on-chip SRAM",
    Lane.CORE: "compute array",
}
LANE_LEGEND = (
    "DRAM = the one modelled link · SRAM = capacity, tiles in flight (two deep when "
    "double buffered) · array = arithmetic"
)


def build(
    chip: HardwareSpec, m: int, n: int, k: int, dtype: DType, steps: int
) -> tuple[PipelineTrace, MatmulSpec, dict[str, float]]:
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

    machine = machine_model(chip, spec.operand_dtype)
    trace = build_trace(
        build_graph(spec, deployment, GraphPhase.STATIC),
        report.phases[0],
        machine,
        double_buffered=report.memory.double_buffered,
        max_steps=steps,
    )
    op = report.phases[0].ops[0]
    facts = {
        "flops": op.flops,
        "dram_bytes": op.dram_bytes,
        "utilization": op.utilization,
        "peak": machine.peak_flops_per_s,
        "on_chip": chip.on_chip_capacity_bytes,
        "bandwidth": chip.dram.bandwidth_bytes_per_s,
        "resident": report.memory.resident_fraction,
    }
    return trace, spec, facts


def _bars(ax: plt.Axes, spans: list[Span], span_s: float, *, origin: float = 0.0) -> None:
    """One row per lane, one rectangle per span, x normalised to *span_s*."""
    for span in spans:
        row = list(Lane).index(span.lane)
        x0 = (span.start_s - origin) / span_s
        width = max(span.duration_s / span_s, 0.0015)
        ax.add_patch(
            Rectangle(
                (x0, row + 0.18),
                width,
                0.64,
                facecolor=LANE_COLOUR[span.lane],
                edgecolor=SURFACE,
                linewidth=0.8,
                zorder=3,
            )
        )


def _idle_note(ax: plt.Axes, spans: list[Span]) -> None:
    """Say so when a lane never runs. An empty row reads as a rendering failure;
    "nothing crossed DRAM" is a result, and on an NPU holding all of B it is the
    headline result."""
    busy_lanes = {span.lane for span in spans}
    for row, lane in enumerate(Lane):
        if lane not in busy_lanes:
            ax.text(
                0.5,
                row + 0.5,
                "idle — nothing crossed this link",
                ha="center",
                va="center",
                fontsize=8.5,
                color=INK_MUTED,
                style="italic",
            )


def _lane_axis(ax: plt.Axes) -> None:
    ax.set_ylim(len(Lane), 0)
    ax.set_yticks([i + 0.5 for i in range(len(Lane))])
    ax.set_yticklabels([LANE_TITLE[lane] for lane in Lane], fontsize=9.5, color=INK)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.5, length=0)
    for spine in ax.spines.values():
        ax.spines[spine.spine_type].set_visible(False)
    ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def draw(
    chip: HardwareSpec,
    trace: PipelineTrace,
    spec: MatmulSpec,
    facts: dict[str, float],
    command: str,
    out: Path,
    zoom_steps: int,
) -> None:
    fig = plt.figure(figsize=(11.5, 6.6), dpi=160, facecolor=SURFACE)
    top = fig.add_axes((0.135, 0.600, 0.83, 0.175))
    bottom = fig.add_axes((0.135, 0.255, 0.83, 0.175))

    # -- whole run ---------------------------------------------------------
    _bars(top, list(trace.spans), trace.total_s)
    _idle_note(top, list(trace.spans))
    _lane_axis(top)
    top.set_xlim(0, 1)
    top.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    top.set_xticklabels(
        [format_time(f * trace.total_s) for f in (0, 0.25, 0.5, 0.75, 1.0)], fontsize=8.5
    )
    busy = trace.busy_s
    for row, lane in enumerate(Lane):
        share = busy[lane] / trace.total_s if trace.total_s else 0.0
        top.text(
            1.008,
            row + 0.5,
            f"{share:.0%}",
            transform=top.get_yaxis_transform(),
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
        )

    # -- zoom --------------------------------------------------------------
    zoom = min(zoom_steps, trace.steps)
    cutoff = max(
        (s.end_s for s in trace.spans if s.step < zoom),
        default=trace.total_s,
    )
    window = [s for s in trace.spans if s.start_s < cutoff]
    _bars(bottom, window, cutoff)
    _idle_note(bottom, window)
    _lane_axis(bottom)
    bottom.set_xlim(0, 1)
    bottom.set_xticks([0, 0.5, 1.0])
    bottom.set_xticklabels([format_time(f * cutoff) for f in (0, 0.5, 1.0)], fontsize=8.5)

    _headings(fig, chip, trace, spec, facts, zoom, cutoff)
    for offset, line in enumerate(textwrap.wrap(f"$ {command}", width=118)):
        fig.text(
            0.045,
            0.105 - 0.028 * offset,
            line + (" \\" if offset == 0 and len(command) > 116 else ""),
            fontsize=8.5,
            color=INK_SECONDARY,
            family="monospace",
        )
    fig.text(
        0.045,
        0.038,
        f"bwz {bwz.__version__}{_git()} — spans are a decomposition of the reported latency, "
        f"not a second model.",
        fontsize=8,
        color=INK_MUTED,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def _headings(
    fig: plt.Figure,
    chip: HardwareSpec,
    trace: PipelineTrace,
    spec: MatmulSpec,
    facts: dict[str, float],
    zoom: int,
    cutoff: float,
) -> None:
    fig.text(
        0.045,
        0.948,
        f"{chip.name} — {spec.name} at {spec.operand_dtype.value}",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.045,
        0.898,
        f"{format_quantity(facts['flops'], 'OP')} against "
        f"{format_bytes(facts['dram_bytes'])} of DRAM traffic; "
        f"{format_bytes(facts['on_chip'])} on chip holds "
        f"{facts['resident']:.0%} of operand B. Span {format_time(trace.total_s)}.",
        fontsize=10,
        color=INK_SECONDARY,
    )
    fig.text(
        0.045,
        0.860,
        f"{trace.steps} steps"
        + (f" drawn, coalesced from {trace.tiles} tiles" if trace.coalesced else " (tiles)")
        + f"; double buffered: {'yes' if trace.double_buffered else 'no'}"
        + (
            f"; fill/drain {format_time(trace.fill_drain_s)} on top of the reported "
            f"{format_time(trace.reported_latency_s)}"
            if trace.fill_drain_s > 0
            else "; span equals the reported latency"
        ),
        fontsize=10,
        color=INK_SECONDARY,
    )

    fig.text(0.045, 0.806, "The whole run", fontsize=11, fontweight="bold", color=INK)
    fig.text(
        0.135,
        0.782,
        LANE_LEGEND,
        fontsize=8.5,
        color=INK_MUTED,
    )
    fig.text(
        0.135,
        0.545,
        "x normalised to the total, so two chips can be laid side by side. "
        "The percentage at the right is each lane's occupancy.",
        fontsize=8.5,
        color=INK_MUTED,
    )

    fig.text(
        0.045,
        0.470,
        f"First {zoom} steps, zoomed",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.135,
        0.200,
        _zoom_caption(trace, cutoff),
        fontsize=8.5,
        color=INK_MUTED,
    )


def _zoom_caption(trace: PipelineTrace, cutoff: float) -> str:
    """What the zoom is showing — which depends on whether DRAM ran at all."""
    head = f"{format_time(cutoff)} of it. "
    if all(span.lane is not Lane.DRAM for span in trace.spans):
        return (
            head + "Operand B is entirely on chip, so there is no load to overlap: the array "
            "runs back to back and SRAM holds two tiles only because the schedule keeps a "
            "spare buffer."
        )
    if trace.double_buffered:
        return (
            head + "A load and the previous tile's arithmetic overlap exactly because capacity "
            "held two tiles; SRAM is two deep throughout."
        )
    return head + "Capacity held one tile, so each load waits for the previous tile's arithmetic."


def _git() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""
    return f" @ {sha}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chip", action="append", default=None, help="Chip id; repeatable")
    parser.add_argument("--matmul", default="4096,4096,4096", metavar="M,N,K")
    parser.add_argument("--weights", default=None, help="Precision; defaults per chip")
    parser.add_argument("--ideal", action="store_true", help="Both de-ratings at 1.0")
    parser.add_argument("--steps", type=int, default=32, help="Steps to draw")
    parser.add_argument("--zoom", type=int, default=6, help="Steps in the zoomed register")
    parser.add_argument("--kanata", action="store_true", help="Also write the Kanata log")
    parser.add_argument("--out", type=Path, default=Path("../docs/plots"))
    args = parser.parse_args()

    m, n, k = (int(part) for part in args.matmul.split(","))
    command = "uv run --group plots python " + " ".join(shlex.quote(a) for a in sys.argv)

    for chip_id in args.chip or ["a100_80gb", "chip_a"]:
        chip = load_chip(chip_id)
        if args.ideal:
            chip = idealised(chip)
        # Default to the widest dtype the chip actually has a unit for, so an
        # INT8-only NPU is not asked to do fp16 and declared infeasible.
        dtype = DType(args.weights) if args.weights else _default_dtype(chip)
        trace, spec, facts = build(chip, m, n, k, dtype, args.steps)
        draw(
            chip,
            trace,
            spec,
            facts,
            command,
            args.out / f"pipeline-matmul-{chip.id}-{dtype.value}.png",
            args.zoom,
        )
        if args.kanata:
            target = args.out / f"pipeline-matmul-{chip.id}-{dtype.value}.kanata"
            target.write_text(
                to_kanata(trace, title=f"{spec.name} on {chip.name}"), encoding="utf-8"
            )
            print(f"wrote {target}")


def _default_dtype(chip: HardwareSpec) -> DType:
    for candidate in (DType.FP16, DType.INT8, DType.BF16, DType.FP32):
        if chip.supports(candidate):
            return candidate
    return chip.compute_units[0].supported_dtypes[0]


if __name__ == "__main__":
    main()
