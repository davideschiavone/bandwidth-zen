"""Draw a chip's roofline, and the memory path the model actually charges for.

Two PNGs, both computed by calling ``analyze`` — nothing here is a transcribed
number. If the engine changes, the figures change with it, which is the whole
reason this replaced a hand-drawn chart.

    uv run --group plots python scripts/plot_roofline.py --chip a100_80gb

Every figure carries the command that produced it, so a PNG that turns up in a
slide deck six months from now can be regenerated rather than guessed at.

Deliberately outside the ``bwz`` package: the analysis core imports no plotting
library (CLAUDE.md #3), and this script imports the core, never the reverse.
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
from matplotlib.patches import FancyArrowPatch, Rectangle

import bwz
from bwz.analysis import analyze, idealised, machine_model
from bwz.analysis.roofline import MachineModel
from bwz.graph.ops import GraphPhase
from bwz.report import Bound
from bwz.spec import DeploymentSpec, DType, HardwareSpec, MatmulSpec, load_chip, load_model
from bwz.units import format_bandwidth, format_bytes, format_quantity, format_time

# -- palette ---------------------------------------------------------------
# Light-surface slots 1 and 2 of the documented categorical palette, which is
# the pair validated for all-pairs use. Two hues is all this chart needs: the
# encoding is polarity (datasheet vs derated), not identity, and every mark also
# carries a direct label, so colour never has to work alone.

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e6e5e1"
DATASHEET = "#2a78d6"  # slot 1, blue
DERATED = "#eb6834"  # slot 2, orange

# Headless: this only ever writes files, and importing a GUI toolkit on a machine
# that has none is the classic way for a figure script to fail in CI.
matplotlib.use("Agg")

BOUND_LABEL = {
    Bound.COMPUTE_BOUND: "compute-bound",
    Bound.DRAM_BW_BOUND: "DRAM-bound",
    Bound.LATENCY_BOUND: "latency-bound",
}


@dataclass(frozen=True)
class Point:
    """One workload placed on the roofline."""

    label: str
    intensity: float
    achieved_flops_per_s: float
    latency_s: float
    bound: Bound
    detail: str


def _deployment(dtype: DType, **overrides: object) -> DeploymentSpec:
    document: dict[str, object] = {
        "batch": 1,
        "input_tokens": 2048,
        "output_tokens": 1,
        "precision": {"weights": dtype, "activations": dtype, "kv_cache": dtype},
    }
    document.update(overrides)
    return DeploymentSpec.model_validate(document)


def matmul_point(chip: HardwareSpec, dtype: DType, m: int, n: int, k: int) -> Point:
    spec = MatmulSpec.model_validate(
        {
            "id": "p",
            "name": "p",
            "family": "matmul",
            "m": m,
            "n": n,
            "k": k,
            "a_dtype": dtype,
            "b_dtype": dtype,
        }
    )
    report = analyze(spec, chip, _deployment(dtype))
    if not report.feasible:
        raise SystemExit(f"bwz: matmul {m}x{n}x{k} is infeasible: {report.infeasibility[0]}")
    op = report.phases[0].ops[0]
    # Against *DRAM* traffic, not compulsory traffic, so the point lands on the
    # roof. Residency is exactly what moves a workload to the right: bytes that
    # stay on chip never cross the one modelled link. Using the compulsory
    # intensity here would float a resident workload above its own ceiling.
    # A workload that fits entirely on chip has no DRAM traffic and therefore
    # infinite intensity -- it has walked off the right-hand edge of the roofline,
    # which is a real answer, not an error.
    return Point(
        label=f"matmul M={m}",
        intensity=_intensity(op.flops, op.dram_bytes),
        achieved_flops_per_s=op.flops / op.latency_s,
        latency_s=op.latency_s,
        bound=op.bound,
        detail=(
            f"{m}x{n}x{k}  {format_quantity(op.flops, 'OP')}  "
            f"{format_bytes(op.dram_bytes)} from DRAM  util {op.utilization:.1%}  "
            f"compulsory AI {op.arithmetic_intensity:.1f}"
        ),
    )


def model_points(chip: HardwareSpec, dtype: DType, model_id: str, tokens: int) -> list[Point]:
    model = load_model(model_id)
    report = analyze(model, chip, _deployment(dtype, input_tokens=tokens))
    if not report.feasible:
        raise SystemExit(f"bwz: {model_id} is infeasible: {report.infeasibility[0]}")
    points = []
    for phase in report.phases:
        name = {GraphPhase.PREFILL: f"prefill {tokens}", GraphPhase.DECODE: "decode B=1"}.get(
            phase.phase, phase.phase.value
        )
        points.append(
            Point(
                label=f"{model_id} {name}",
                intensity=_intensity(phase.flops, phase.dram_bytes),
                achieved_flops_per_s=phase.achieved_flops_per_s,
                latency_s=phase.latency_s,
                bound=phase.bound,
                detail=(
                    f"{format_quantity(phase.flops, 'OP')}  "
                    f"{format_bytes(phase.dram_bytes)} from DRAM  util {phase.utilization:.1%}"
                ),
            )
        )
    return points


def _intensity(flops: float, dram_bytes: float) -> float:
    """OP per byte crossing the one modelled link; ``inf`` when nothing crosses."""
    return flops / dram_bytes if dram_bytes > 0 else float("inf")


def _roof(machine: MachineModel, xs: list[float]) -> list[float]:
    """min(compute ceiling, bandwidth x intensity) — the roofline itself."""
    return [
        min(machine.effective_flops_per_s, machine.effective_bandwidth_bytes_per_s * x) for x in xs
    ]


def _wrap_command(command: str) -> list[str]:
    """The command, wrapped to the figure width and continued with a backslash so
    the wrapped form is still runnable."""
    lines = textwrap.wrap(f"$ {command}", width=118, break_long_words=False)
    return [line + " \\" for line in lines[:-1]] + lines[-1:]


def _floor(datasheet: MachineModel, points: list[Point]) -> float:
    """Bottom of the y-axis: three decades below peak, or low enough to hold the
    slowest workload — a batch-1 decode can sit further down than that."""
    return min(
        [datasheet.effective_flops_per_s / 3e3]
        + [p.achieved_flops_per_s * 0.4 for p in points if p.achieved_flops_per_s > 0]
    )


def draw_roofline(
    chip: HardwareSpec,
    dtype: DType,
    points: list[Point],
    command: str,
    out: Path,
) -> None:
    datasheet = machine_model(idealised(chip), dtype)
    derated = machine_model(chip, dtype)

    x_lo, x_hi = 0.1, 1e4
    xs = [x_lo * (x_hi / x_lo) ** (i / 400) for i in range(401)]

    fig = plt.figure(figsize=(11.5, 8.6), dpi=160, facecolor=SURFACE)
    ax = fig.add_axes((0.085, 0.34, 0.885, 0.53))
    ax.set_facecolor(SURFACE)

    # A profile may declare both efficiencies as 1.0 — chip_a does, because the
    # worked example it comes from quotes datasheet peak. Then the two roofs are
    # the same line, and drawing a second one under a legend that promises a
    # difference would be a lie about the chip.
    derated_differs = (
        abs(derated.effective_flops_per_s - datasheet.effective_flops_per_s) > 1.0
        or abs(derated.effective_bandwidth_bytes_per_s - datasheet.effective_bandwidth_bytes_per_s)
        > 1.0
    )
    ax.plot(
        xs,
        _roof(datasheet, xs),
        color=DATASHEET,
        lw=2.0,
        zorder=3,
        label="datasheet ceiling"
        if derated_differs
        else "ceiling (this profile declares no derating)",
    )
    if derated_differs:
        ax.plot(
            xs,
            _roof(derated, xs),
            color=DERATED,
            lw=2.0,
            ls=(0, (5, 3)),
            zorder=3,
            label="with unfitted calibration derating",
        )

    # The ridge point, where the two ceilings meet: left of it a workload is
    # bandwidth-starved, right of it the array is the limit.
    ax.axvline(datasheet.ridge_point, color=INK_MUTED, lw=1.0, ls=":", zorder=2)
    ax.annotate(
        f"ridge {datasheet.ridge_point:.0f} OP/byte",
        xy=(datasheet.ridge_point, _floor(datasheet, points) * 1.15),
        xytext=(-13, 0),
        textcoords="offset points",
        fontsize=8.5,
        color=INK_SECONDARY,
        rotation=90,
        va="bottom",
    )

    # The single-row ceiling. NOT a derating: --ideal leaves it exactly here,
    # because it follows from the array's declared geometry.
    dims = datasheet.unit.systolic_dims
    if dims is not None:
        rows = dims[0]
        tail = datasheet.effective_flops_per_s / (1 + rows)
        ax.plot(
            [tail / datasheet.effective_bandwidth_bytes_per_s, x_hi],
            [tail, tail],
            color=INK_MUTED,
            lw=1.2,
            ls=(0, (1, 2)),
            zorder=2,
        )
        ax.annotate(
            f"M=1 ceiling {format_quantity(tail, 'OP/s')} — 1/{1 + rows} of peak\n"
            f"({rows}x{dims[1]} array tail; not a derating, --ideal leaves it here)",
            xy=(datasheet.ridge_point * 2.2, tail),
            xytext=(0, 6),
            textcoords="offset points",
            fontsize=8.5,
            color=INK_SECONDARY,
            ha="left",
        )

    # Numbered keys rather than labels-on-marks: on a log-log plot the interesting
    # workloads bunch up against the compute roof, and five two-line callouts
    # collide there every time. The number is still a direct label — the table
    # below resolves it, and carries more than a callout could.
    resident = [p for p in points if p.intensity == float("inf")]
    if resident:
        # A workload with no DRAM traffic has infinite intensity: it is off the
        # right-hand edge, not at the edge. Say so rather than let the clamp
        # read as a measured position.
        ax.axvline(x_hi * 0.78, color=INK_MUTED, lw=1.0, ls=":", zorder=2)
        ax.annotate(
            "no DRAM traffic →\n(fully resident:\ninfinite intensity)",
            xy=(x_hi * 0.76, datasheet.effective_flops_per_s / 40),
            fontsize=8.5,
            color=INK_SECONDARY,
            ha="right",
            va="top",
        )

    for index, point in enumerate(points, start=1):
        x = min(point.intensity, x_hi * 0.90)
        ax.plot(
            x,
            point.achieved_flops_per_s,
            marker="o",
            ms=15,
            mfc=INK,
            mec=SURFACE,
            mew=2,
            ls="none",
            zorder=5,
        )
        ax.annotate(
            str(index),
            xy=(x, point.achieved_flops_per_s),
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=SURFACE,
            zorder=6,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(_floor(datasheet, points), datasheet.effective_flops_per_s * 3)
    ax.set_xlabel("arithmetic intensity  (OP per byte of DRAM traffic)", fontsize=9.5, color=INK)
    ax.set_ylabel("achieved throughput  (OP/s)", fontsize=9.5, color=INK)
    ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
    ax.grid(True, which="minor", color=GRID, lw=0.4, alpha=0.6, zorder=0)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)

    legend = ax.legend(loc="upper left", fontsize=9, frameon=True, framealpha=1.0)
    legend.get_frame().set_facecolor(SURFACE)
    legend.get_frame().set_edgecolor(GRID)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.text(
        0.085,
        0.955,
        f"{chip.name} — roofline at {dtype.value}",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.085,
        0.915,
        f"Compute ceiling {format_quantity(datasheet.peak_flops_per_s, 'OP/s')} against "
        f"{format_bandwidth(chip.dram.bandwidth_bytes_per_s)} of DRAM.",
        fontsize=10,
        color=INK_SECONDARY,
    )
    fig.text(
        0.085,
        0.890,
        (
            "The solid roof is the datasheet — what --ideal reports. The dashed roof applies the "
            "two unfitted calibration constants; the gap is the unfitted part of any prediction."
            if derated_differs
            else "This profile declares both efficiencies as 1.0, so the datasheet roof is the "
            "only roof: --ideal changes nothing here."
        ),
        fontsize=10,
        color=INK_SECONDARY,
    )

    _table(fig, points, datasheet, derated)
    for offset, line in enumerate(_wrap_command(command)):
        fig.text(
            0.085,
            0.048 - 0.018 * offset,
            line,
            fontsize=8.5,
            color=INK_SECONDARY,
            family="monospace",
        )
    fig.text(
        0.085,
        0.012,
        f"bwz {bwz.__version__}{_git_describe()} — every value computed by "
        f"bwz.analysis.analyze, not transcribed.",
        fontsize=8,
        color=INK_MUTED,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def _table(
    fig: plt.Figure, points: list[Point], datasheet: MachineModel, derated: MachineModel
) -> None:
    """The numbers behind the marks, in text.

    A log-log plot is good at showing which side of the ridge a workload sits on
    and bad at showing how far apart two latencies are. The table is the relief:
    it is also the accessible view, since it does not depend on reading a colour.
    """
    fig.text(0.085, 0.268, "Behind the marks", fontsize=10, fontweight="bold", color=INK)
    header = (
        f"    {'workload':<22}{'intensity':>13}  {'latency':>9}  {'verdict':<14}  "
        f"shape / arithmetic / DRAM traffic / shape utilisation"
    )
    fig.text(0.085, 0.244, header, fontsize=8, color=INK_MUTED, family="monospace")
    y = 0.222
    for index, point in enumerate(points, start=1):
        intensity = (
            "no DRAM traffic"
            if point.intensity == float("inf")
            else f"{point.intensity:>9.1f} OP/b"
        )
        row = (
            f"{index:>2}  {point.label:<22}{intensity:>15}  "
            f"{format_time(point.latency_s):>9}  {BOUND_LABEL[point.bound]:<14}  {point.detail}"
        )
        fig.text(0.085, y, row, fontsize=8.5, color=INK_SECONDARY, family="monospace")
        y -= 0.023
    fig.text(
        0.085,
        y - 0.012,
        (
            f"ridge point {datasheet.ridge_point:.0f} OP/byte at the datasheet ceiling, "
            f"{derated.ridge_point:.0f} once derated — a workload between the two is the only "
            f"place the verdict can flip."
            if datasheet.ridge_point != derated.ridge_point
            else f"ridge point {datasheet.ridge_point:.0f} OP/byte, derated and not: this "
            f"profile declares no efficiency derating."
        ),
        fontsize=8.5,
        color=INK_MUTED,
    )


def draw_machine(chip: HardwareSpec, dtype: DType, command: str, out: Path) -> None:
    """The three-element machine, drawn as it is modelled (docs/CORRECTIONS.md D5a).

    The point of the picture is what is *missing*: exactly one link carries a
    bandwidth number. The on-chip path is drawn as a plain arrow because the v1
    model gives it no rate at all — SRAM contributes capacity, nothing else.
    """
    machine = machine_model(idealised(chip), dtype)
    on_chip = chip.on_chip_capacity_bytes

    fig = plt.figure(figsize=(11.5, 5.0), dpi=160, facecolor=SURFACE)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    boxes = [
        (
            6,
            "DRAM",
            f"{format_bytes(chip.dram.capacity_bytes)}\n"
            f"{format_bandwidth(chip.dram.bandwidth_bytes_per_s)}",
            DATASHEET,
        ),
        (40, "on-chip SRAM", f"{format_bytes(on_chip)}\ncapacity only", INK_SECONDARY),
        (
            74,
            "compute array",
            f"{format_quantity(machine.peak_flops_per_s, 'OP/s')} at {dtype.value}\n"
            + (
                f"{machine.unit.systolic_dims[0]}x{machine.unit.systolic_dims[1]} systolic"
                if machine.unit.systolic_dims
                else machine.unit.name
            ),
            DATASHEET,
        ),
    ]
    for x, title, body, colour in boxes:
        ax.add_patch(
            Rectangle(
                (x, 12),
                20,
                14,
                facecolor=SURFACE,
                edgecolor=colour,
                lw=2.0,
                joinstyle="round",
                zorder=3,
            )
        )
        ax.text(x + 10, 22.5, title, ha="center", fontsize=11, fontweight="bold", color=INK)
        ax.text(x + 10, 16.5, body, ha="center", fontsize=9, color=INK_SECONDARY, linespacing=1.6)

    ax.add_patch(
        FancyArrowPatch(
            (26.8, 19),
            (39.2, 19),
            arrowstyle="-|>",
            mutation_scale=18,
            lw=2.5,
            color=DATASHEET,
            zorder=4,
        )
    )
    ax.text(
        33,
        30.5,
        "the one modelled link",
        ha="center",
        fontsize=9.5,
        color=DATASHEET,
        fontweight="bold",
    )
    ax.text(
        33,
        27.3,
        f"t_dram = bytes / {format_bandwidth(chip.dram.bandwidth_bytes_per_s)}",
        ha="center",
        fontsize=8.5,
        color=INK_SECONDARY,
        family="monospace",
    )

    ax.add_patch(
        FancyArrowPatch(
            (60.8, 19),
            (73.2, 19),
            arrowstyle="-|>",
            mutation_scale=18,
            lw=1.5,
            color=INK_MUTED,
            linestyle=(0, (4, 3)),
            zorder=4,
        )
    )
    ax.text(67, 30.5, "no bandwidth term", ha="center", fontsize=9.5, color=INK_MUTED)
    ax.text(
        67,
        27.3,
        "assumed free (D5b)",
        ha="center",
        fontsize=8.5,
        color=INK_MUTED,
        family="monospace",
    )

    ax.text(
        6,
        9.5,
        "SRAM earns its place through capacity: two tiles fit, so a load overlaps the previous\n"
        "tile's arithmetic and latency is max(load, compute) rather than their sum. Resident\n"
        "weights stop being traffic at all. Neither effect needs an on-chip rate.",
        fontsize=9,
        color=INK_SECONDARY,
        linespacing=1.7,
        va="top",
    )

    fig.text(
        0.052,
        0.93,
        f"{chip.name} — the machine bwz models, at {dtype.value}",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    for offset, line in enumerate(_wrap_command(command)):
        fig.text(
            0.052,
            0.055 - 0.038 * offset,
            line,
            fontsize=8.5,
            color=INK_MUTED,
            family="monospace",
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def _git_describe() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""
    return f" @ {sha}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chip", default="a100_80gb", help="Chip profile id or path")
    parser.add_argument("--weights", default="fp16", help="Precision to plot the roofline at")
    parser.add_argument(
        "--matmul",
        action="append",
        default=None,
        metavar="M,N,K",
        help="A matmul shape to place on the roofline; repeatable",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        metavar="ID",
        help="A model profile whose prefill and decode go on the roofline; repeatable",
    )
    parser.add_argument("--tokens", type=int, default=2048, help="Prefill length for --model")
    parser.add_argument("--out", type=Path, default=Path("../docs/plots"))
    args = parser.parse_args()

    dtype = DType(args.weights)
    chip = load_chip(args.chip)
    # Points are placed at the datasheet ceiling, so they sit on the solid roofs.
    # The derated roofs stay in the picture as the band a real run falls into.
    ideal_chip = idealised(chip)

    shapes = args.matmul or ["10000,10000,10000", "512,10000,10000", "1,10000,10000"]
    points = []
    for shape in shapes:
        try:
            m, n, k = (int(part) for part in shape.split(","))
        except ValueError:
            raise SystemExit(f"bwz: --matmul expects M,N,K; got {shape!r}") from None
        points.append(matmul_point(ideal_chip, dtype, m, n, k))
    for model_id in args.model or []:
        points.extend(model_points(ideal_chip, dtype, model_id, args.tokens))

    command = "uv run --group plots python " + " ".join(shlex.quote(a) for a in sys.argv)
    draw_roofline(chip, dtype, points, command, args.out / f"roofline-{chip.id}-{dtype.value}.png")
    draw_machine(chip, dtype, command, args.out / f"machine-{chip.id}.png")


if __name__ == "__main__":
    main()
