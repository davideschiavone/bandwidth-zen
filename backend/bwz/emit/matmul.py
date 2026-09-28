"""Write one matmul's decomposition out as a runnable Python program.

``docs/CORRECTIONS.md`` D54. :mod:`bwz.deploy` prints a pseudo-C loop nest whose
*constants* are checked against the schedule; nothing checks that the
decomposition it narrates computes a matmul at all, or that the bytes the
roofline charged are the bytes such a schedule would move. This module closes
that: for a given chip and the strategy chosen for it, it writes a self-contained
program that walks the same grid, counts what it moves, and checks those counts
against the prediction embedded in its own source.

Every section of the emitted file is built from the objects the ``Report`` was
built from — :class:`~bwz.analysis.stationarity.TileGrid`,
:class:`~bwz.analysis.dataflow.DataflowPlan`, :class:`~bwz.report.OpResult` — and
never from geometry re-derived here. The grid already knows which operand is
resident, how many tiles there are along which dimensions, and which one each
tile sweeps (D53); if this module ever needs something the grid cannot say, the
grid is under-described and that is where the fix belongs.

**Legibility is the priority.** Where a clearer loop nest and a tighter check
pull against each other, the loop nest wins and the check moves elsewhere: a
vectorised walk that verifies perfectly while hiding which dimension is swept has
failed at the job. The measurement half is what makes the teaching trustworthy,
not the point of it.
"""

from __future__ import annotations

import ast
import inspect
import math
import re
from dataclasses import dataclass

from bwz.analysis.dataflow import DataflowPlan
from bwz.analysis.roofline import MachineModel
from bwz.analysis.stationarity import Dim, TileGrid
from bwz.analysis.tiling import padded
from bwz.emit import _harness
from bwz.report import OpResult, ReductionPlacement
from bwz.spec.deployment import AStrategy, BDataflow
from bwz.spec.dtypes import DType, accumulator_for, bytes_per_element, is_integer
from bwz.spec.hardware_spec import ComputeUnit, Dataflow, HardwareSpec
from bwz.spec.loaders import largest_declared_array
from bwz.units import format_bytes

COMMENT_COLUMN = 40
"""Where a constant's trailing comment starts. Every constant in the emitted file
carries one naming where it came from, and they line up so the whole block reads
as a table."""

Line = tuple[str, str | None]
"""One emitted line and the animation stage it belongs to, if any."""


def _plain(lines: list[str]) -> list[Line]:
    """Untagged lines — everything but the loop nest's own load/exec/store."""
    return [(line, None) for line in lines]


@dataclass(frozen=True, slots=True)
class EmittedProgram:
    """A runnable program, and the numbers it was told to expect.

    ``predicted`` is kept beside the source so :func:`bwz.emit.check` can compare
    it against the ``Report`` without re-parsing the file — the same contract
    :class:`bwz.deploy.Deployment` had with ``deploy.check``.
    """

    chip_id: str
    filename: str
    source: str
    predicted: dict[str, float]
    stationarity: Dataflow
    working_set_bytes: float
    stage_lines: tuple[tuple[str, tuple[int, ...]], ...] = ()
    """0-indexed line numbers within :attr:`source` for each animated stage
    (``"load_b"``, ``"load_a"``, ``"exec"``, ``"store"``, ``"reduce"`` — the
    vocabulary ``bwz.figures``'s ``_ANIMATION_STAGE`` already uses). Lets
    the animation's debug view light up the lines live at its current time
    (D41), now against the real program rather than a pseudo-C paraphrase of it
    (D54). A tuple of pairs rather than a ``dict``, to keep a frozen dataclass
    hashable in substance."""
    grid: TileGrid | None = None
    """The decomposition this program walks. Carried so the animation's geometry
    panel (D48) draws the operand *this* chip keeps resident without re-deriving
    the emitter's own arithmetic."""
    array_rows: int = 0
    array_cols: int = 0


_STATIONARITY_TITLE: dict[Dataflow, str] = {
    Dataflow.OUTPUT_STATIONARY: "output-stationary",
    Dataflow.WEIGHT_STATIONARY: "weight-stationary",
    Dataflow.INPUT_STATIONARY: "input-stationary",
    Dataflow.ROW_STATIONARY: "row-stationary",
}


def decomposition_suffix(stationarity: Dataflow, k_partitions: int) -> str:
    """``-<stationarity>[-splitk<N>]`` — what makes two runs of one shape differ.

    Shared with the figures deliberately (D65). A page and the program beside it
    are two views of the *same* decomposition, so they are named alike, and two
    stationarities of one shape can never overwrite each other's files. Naming
    only the shape would have `--stationarity is` land on top of `os`'s page —
    which is exactly what it did.
    """
    splits = f"-splitk{k_partitions}" if k_partitions > 1 else ""
    return f"-{stationarity.value}{splits}"


def default_filename(chip_id: str, dtype: DType, stationarity: Dataflow, k_partitions: int) -> str:
    """``matmul-<chip>-<dtype>-<stationarity>[-splitk<N>].py``."""
    return f"matmul-{chip_id}-{dtype.value}{decomposition_suffix(stationarity, k_partitions)}.py"


def predicted_for(
    machine: MachineModel,
    grid: TileGrid,
    dataflow: DataflowPlan,
    result: OpResult,
) -> dict[str, float]:
    """The report's own numbers, in the vocabulary the emitted program counts in.

    Every entry is read straight off the ``Report`` or derived from the grid by
    the same formula the analysis uses, so a disagreement at runtime is one
    between the model and a walk of it — never between two spellings of the
    model.

    The two ratios a reader cares about, wave occupancy and shape padding, are
    pinned here as **integers** (``idle_core_waves``, ``mac_slots``) rather than
    as floats. It is the same claim without a float comparison: ``occupancy = 1 -
    idle_core_waves / (waves * available_cores)`` and ``padding = macs /
    mac_slots``, both of which the program prints from counts it asserted.
    """
    unit = machine.unit
    rows, cols = unit.systolic_dims if unit.systolic_dims is not None else (1, 1)
    tiles = grid.tiles
    available = max(unit.count, 1)
    used = max(1, min(available, tiles))
    waves = math.ceil(tiles / used)
    events = grid.a_events * (dataflow.tiles_per_a_event // max(1, dataflow.residency_tiles))
    return {
        "tiles": tiles,
        "waves": waves,
        "available_cores": available,
        "used_cores": used,
        "idle_core_waves": waves * available - tiles,
        "macs": grid.m * grid.n * grid.k,
        "mac_slots": padded(grid.m, rows) * padded(grid.n, cols) * padded(grid.k, rows),
        "staging_events": events,
        # (p-1) per output element, where p is the k-slices it ends up with —
        # the grid's own K axis, or split-K's partitions. The report charges
        # exactly this many additions to the vector unit (D62), so the walk
        # counting a different number means one of the two is wrong.
        "partial_sum_adds": (grid.k_slices - 1) * grid.m * grid.n,
        "a_dram_bytes": result.dram_activation_read_bytes,
        "c_dram_bytes": result.dram_write_bytes,
        "partial_dram_bytes": result.dram_reduction_bytes,
        "b_dram_bytes_charged": result.dram_weight_read_bytes,
        "utilization": result.utilization,
    }


def emit_matmul(
    chip: HardwareSpec,
    machine: MachineModel,
    grid: TileGrid,
    dataflow: DataflowPlan,
    result: OpResult,
    *,
    a_dtype: DType,
    b_dtype: DType,
    c_dtype: DType,
    acc_dtype: DType | None,
    double_buffered: bool,
    command: str,
    version: str,
    seed: int = 20240501,
) -> EmittedProgram:
    """Build the runnable program for this chip, dtype and strategy.

    *acc_dtype* is ``None`` for "whatever this arithmetic accumulates into",
    which is the normal case and the only correct one for an integer matmul: the
    width follows from the operand format, not from a deployment knob
    (:func:`~bwz.spec.dtypes.accumulator_for`, D56). Pass one to override.
    """
    # machine.dtype rather than either operand's: mixed widths run at the wider
    # one, through a single datapath, and it is that datapath that accumulates
    # (docs/CORRECTIONS.md D18).
    accumulate = acc_dtype if acc_dtype is not None else accumulator_for(machine.dtype)
    unit = machine.unit
    if unit.systolic_dims is None:
        raise ValueError(
            f"{chip.id}: compute unit {unit.name!r} declares no systolic_dims, so there is no "
            f"tile grid to walk and nothing to emit. Pick a chip whose profile describes an "
            f"array."
        )
    predicted = predicted_for(machine, grid, dataflow, result)
    working_set = (
        grid.m * grid.k * bytes_per_element(a_dtype)
        + grid.k * grid.n * bytes_per_element(b_dtype)
        + grid.m * grid.n * bytes_per_element(c_dtype)
    )

    # Assembled as (line, stage tag) pairs so the animation can light up the
    # lines live at its current time (D41/D43) against the real program rather
    # than against a pseudo-C paraphrase of it. Tagging after the fact, by
    # matching line text, would be a second place that has to know what the
    # emitter wrote; tagging here is one place.
    lines: list[Line] = [("#!/usr/bin/env python3", None)]
    lines += _plain(
        _docstring(
            chip,
            machine,
            grid,
            dataflow,
            command,
            version,
            working_set,
            dtypes=(a_dtype, b_dtype, c_dtype),
        )
    )
    lines += _plain(["from __future__ import annotations", ""])
    lines += _plain(_harness_imports())
    lines += _plain(["", ""])
    lines += _plain(
        _constants(
            chip,
            machine,
            grid,
            dataflow,
            predicted,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            c_dtype=c_dtype,
            acc_dtype=accumulate,
            double_buffered=double_buffered,
            seed=seed,
        )
    )
    lines += _plain(["", ""])
    lines += _walk(grid, dataflow, unit, predicted, result.reduction_placement)
    lines += _plain(["", ""])
    lines += _main(chip, machine, grid, dataflow, result.reduction_placement)
    # The runtime goes last, against convention and on purpose: the loop nest is
    # what a reader is here for, and 400 lines of machinery between the constants
    # and the walk would bury it. Python does not mind — nothing below runs until
    # the guard at the very bottom calls main().
    lines += _plain(["", ""])
    lines += _plain(_harness_body())
    lines += _plain(
        [
            "",
            "",
            'if __name__ == "__main__":',
            "    raise SystemExit(main())",
        ]
    )

    source, stage_lines = _tidy(lines)
    return EmittedProgram(
        chip_id=chip.id,
        filename=default_filename(chip.id, machine.dtype, grid.stationarity, grid.k_partitions),
        source=source,
        predicted=predicted,
        stationarity=grid.stationarity,
        working_set_bytes=working_set,
        stage_lines=stage_lines,
        grid=grid,
        array_rows=unit.systolic_dims[0],
        array_cols=unit.systolic_dims[1],
    )


# ---------------------------------------------------------------------------- header


def _docstring(
    chip: HardwareSpec,
    machine: MachineModel,
    grid: TileGrid,
    dataflow: DataflowPlan,
    command: str,
    version: str,
    working_set: float,
    *,
    dtypes: tuple[DType, DType, DType],
) -> list[str]:
    """The file's own header: what it is, what it is not, and what it models."""
    title = _STATIONARITY_TITLE[grid.stationarity]
    out = [
        f'"""{chip.name} · {machine.dtype.value} · {title} — the decomposition bwz costed.',
        "",
        f"    $ {command}",
        f"    bwz {version}",
        "",
        *_shapes_picture(grid.m, grid.n, grid.k, dtypes),
        "",
        "Not an illustration. This walks the same tile grid, stages A on the same events",
        "and hands tiles to cores the same way the cost model charged; it counts what it",
        "moves along the way and checks those counts against the prediction written into",
        "section 5 below. When the two disagree, one of them is wrong — which is the",
        "whole reason this file exists.",
        "",
        "NOT A BENCHMARK. It validates counts, not time. Its wall clock has no",
        "relationship to the latency the report predicts: it makes no attempt to be fast,",
        "it runs one OS thread per modelled core whatever this host actually has, and the",
        "prediction it checks itself against came from the model rather than a stopwatch.",
        "Timing this program and comparing it to the reported latency is a category",
        "error — what is being checked here is the model's bookkeeping.",
        "",
        f"Working set: {format_bytes(working_set)} of operands and result.",
        "",
        "Read it top to bottom: the constants, the prediction, then walk(). The",
        "runtime it needs — counted DRAM, the shared staging buffer, the lockstep wave",
        "loop — is at the BOTTOM, out of the way of the part you came for.",
        "",
        "    $ python <this file> --debug",
        "",
        "narrates the walk: which core takes which tile in which wave, when A is",
        "staged, and every instruction tile issued with its operand ranges. One line",
        "per instruction tile, so it is for small shapes — the run above issues one",
        "line for every mma() call, which a large matmul has millions of.",
        "",
        "What is really executed here, and what is only written down:",
        "",
        "  stationarity   real — a different loop nest, a different resident buffer",
        "  split-K        real — per-partition partials and a second reduction pass",
        "  reduction      half real — the additions are performed and COUNTED here; the",
        "                 report's reduction OVERLAP is written down only. That the",
        "                 vector unit sums slice n while the array builds slice n+1 is a",
        "                 claim about time, and this program counts (D62)",
        "  a_strategy     real — changes how often A is staged, and A's measured bytes",
        "  residency      real — the staging buffer serves that many tiles before refill",
        "  b_dataflow     written down only — a placement in *time*, moving no byte",
        "                 within one pass (D30/D33), and this program measures bytes",
        "  double buffer  written down only — 'latency is max(load, compute)' is a claim",
        "                 about time, and a prefetch queue here would change no count",
        "  sub-cycles     written down only — a rate, not a structure",
        "",
        "Results are held at the accumulator's width throughout. Narrowing them to the",
        f"declared {machine.dtype.value} result width is a store-side rounding the byte "
        "counts below",
        "already reflect and the arithmetic here does not emulate.",
    ]
    if dataflow.notes:
        out += ["", "The report's own clamps, which this run inherits:", ""]
        out += [f"  · {note}" for note in dataflow.notes]
    out.append('"""')
    return out


# ------------------------------------------------------------------------- constants


def _shapes_picture(m: int, n: int, k: int, dtypes: tuple[DType, DType, DType]) -> list[str]:
    """Which operand is which shape: ``A[M, K] @ B[K, N] -> C[M, N]``, drawn.

    B sits above C and A to its left, the textbook layout, so the shared edges
    line up: A's width and B's height are both K, C takes its rows from A and its
    columns from B, and K — the contraction — is the one dimension C does not
    have. The boxes are not to scale; the labels carry this run's sizes.
    """
    a_dtype, b_dtype, c_dtype = (dtype.value for dtype in dtypes)
    a_width, bc_width = 16, 12
    m_label, n_label, k_label = f"M = {m:,}", f"N = {n:,}", f"K = {k:,}"
    pad = len(m_label) + 1
    b_at = pad + a_width + 3
    blank_a, blank_bc = "│" + " " * a_width + "│", "│" + " " * bc_width + "│"
    return [
        "The shapes:  A[M, K]  @  B[K, N]  ->  C[M, N]",
        "",
        " " * b_at + n_label.center(bc_width + 2),
        " " * b_at + "┌" + "─" * bc_width + "┐",
        " " * b_at + blank_bc,
        k_label.rjust(b_at - 1) + " │" + f"B · {b_dtype}".center(bc_width) + "│",
        " " * b_at + blank_bc,
        " " * b_at + "└" + "─" * bc_width + "┘",
        " " * pad + k_label.center(a_width + 2) + " " + n_label.center(bc_width + 2),
        " " * pad + "┌" + "─" * a_width + "┐ ┌" + "─" * bc_width + "┐",
        " " * pad + blank_a + " " + blank_bc,
        m_label.ljust(pad)
        + "│"
        + f"A · {a_dtype}".center(a_width)
        + "│ │"
        + f"C · {c_dtype}".center(bc_width)
        + "│ "
        + m_label,
        " " * pad + blank_a + " " + blank_bc,
        " " * pad + "└" + "─" * a_width + "┘ └" + "─" * bc_width + "┘",
        "",
        "  M   rows of A and rows of C          -M  (a batch folds in here)",
        "  N   columns of B and columns of C    -N",
        "  K   columns of A = rows of B         -K  (the contraction: summed away, so",
        "                                            C has no K at all)",
    ]


def _constant(statement: str, *comments: str) -> list[str]:
    """One constant and the comment naming where its value came from."""
    if not comments:
        return [statement]
    head = statement.ljust(COMMENT_COLUMN - 1) + f" # {comments[0]}"
    return [head, *(" " * COMMENT_COLUMN + f"# {rest}" for rest in comments[1:])]


def _constants(
    chip: HardwareSpec,
    machine: MachineModel,
    grid: TileGrid,
    dataflow: DataflowPlan,
    predicted: dict[str, float],
    *,
    a_dtype: DType,
    b_dtype: DType,
    c_dtype: DType,
    acc_dtype: DType,
    double_buffered: bool,
    seed: int,
) -> list[str]:
    """Sections 1-5: the shape, the chip, the strategy, the grid, the prediction."""
    unit = machine.unit
    rows, cols = unit.systolic_dims if unit.systolic_dims is not None else (1, 1)
    multiplier = unit.dtype_multipliers.get(machine.dtype, 1.0)
    sub_cycles = _sub_cycles(machine)
    widths = ", ".join(f'"{dtype.value}"' for dtype in (a_dtype, b_dtype, c_dtype))
    k_tiles = max(1, math.ceil(grid.k / rows))
    bounds = _k_tile_bounds(grid, rows)

    out = [
        _rule("1. the shape"),
        "# From the command line at the top. Every constant below is that, the chip",
        "# profile, or derived from the two — nothing here is a free parameter.",
        *_constant(f"M, N, K = {grid.m}, {grid.n}, {grid.k}", "-M / -N / -K"),
        *_constant(f"A_DTYPE, B_DTYPE, C_DTYPE = {widths}", "-d, --a, --b, --out"),
        *_constant(f'ACC_DTYPE = "{acc_dtype.value}"', *_accumulator_note(machine.dtype)),
        *_constant(
            f"A_BYTES_PER_ELEMENT = {bytes_per_element(a_dtype)}",
            "spec/dtypes.py — a definition, not a fit",
        ),
        *_constant(f"B_BYTES_PER_ELEMENT = {bytes_per_element(b_dtype)}", "likewise"),
        *_constant(
            f"C_BYTES_PER_ELEMENT = {bytes_per_element(c_dtype)}",
            "the result width: bytes only, never operations",
        ),
        *_constant(
            f"PARTIAL_BYTES_PER_ELEMENT = {bytes_per_element(c_dtype)}",
            "the model sizes split-K's partials at C's",
            "own width, not ACC_DTYPE's (roofline._reduction_for)",
        ),
        *_constant(
            f"INTEGER = {is_integer(machine.dtype)}", "decides the reference's own arithmetic"
        ),
        *_constant(f"SEED = {seed}", "operands are dyadic: nothing here rounds"),
        *_constant(
            f"TOLERANCE = {_tolerance(grid, acc_dtype, machine.dtype)!r}",
            "so a difference is a WALK error, never a rounding",
            "one. bwz/emit/matmul.py::_tolerance derives it.",
        ),
        "",
        _rule("2. the chip"),
        f"# profiles/chips/{chip.id}.yaml, compute unit {unit.name!r}.",
        *_constant(f"ROWS, COLS = {rows}, {cols}", "systolic_dims"),
        *_constant(f"AVAILABLE_CORES = {unit.count}", f"count: {unit.name}"),
        *_constant(
            f"WEIGHT_SETS = {unit.weight_sets}", "array-sized weight tiles one core holds (D30)"
        ),
        *_constant(
            f"ON_CHIP_BYTES = {int(chip.on_chip_capacity_bytes):_}",
            "the capacity A's staging comes out of",
        ),
        *_constant(
            f"SUB_CYCLES = {sub_cycles}",
            f"{machine.dtype.value} multiplier {multiplier:g}: a rate,",
            "annotated here and never executed",
        ),
        *_constant(
            f"LARGEST_DECLARED_CORES = {_largest_array()[0]}",
            "the biggest array any bundled profile declares",
            f"({_largest_array()[1]}). Only used to decide whether",
            "starting one thread per core is worth remarking on",
        ),
        *_constant(
            f"SLOTS_PER_MMA = {rows * rows * cols:_}",
            "ROWS * ROWS * COLS — MAC positions one",
            "instruction tile issues, used or not (D52)",
        ),
        "",
        _rule("3. the strategy"),
        "# Where each of these came from, since they come from four different",
        "# places: a flag you passed, the default of one you did not, the chip's",
        "# own declaration, or a capacity test. Only DEPTH is derived, and it is",
        "# a yes/no rather than a measurement — see its own comment.",
        *_constant(
            f'STATIONARITY = "{grid.stationarity.value}"', *_stationarity_note(grid, dataflow, unit)
        ),
        *_constant(f"SPLIT_K = {grid.k_partitions}", *_split_k_note(grid)),
        *_constant(f'A_STRATEGY = "{dataflow.a_strategy.value}"', *_a_strategy_note(dataflow)),
        *_constant(f"A_RESIDENCY_TILES = {dataflow.residency_tiles}", *_residency_note(dataflow)),
        *_constant(
            f'B_DATAFLOW = "{dataflow.b_dataflow.value}"',
            *_b_dataflow_note(dataflow, unit.weight_sets),
        ),
        *_constant(
            f"ITERATIONS = {dataflow.iterations}",
            _flag("--iterations", str(dataflow.iterations), is_default=dataflow.iterations == 1)
            + ": invocations the report's",
            "bytes stand for. Only b_dataflow=persistent reads it",
        ),
        *_constant(
            f"DEPTH = {2 if double_buffered else 1}",
            *_depth_note(chip, machine, rows, cols, double_buffered),
        ),
        "",
        _rule("4. the grid"),
        f"# What STATIONARITY implies (D53). Under {grid.stationarity.value} the grid is "
        f"{grid.resident.value}'s:",
        f"# {grid.rows} x {grid.cols} tiles along {grid.row_dim.value} x {grid.col_dim.value}, "
        f"each sweeping {grid.swept_dim.value}.",
        *_constant(
            f"GRID_ROWS = {grid.rows}", f"ceil({grid.row_dim.value} / {_tile_of(grid.row_dim)})"
        ),
        *_constant(
            f"GRID_COLS = {grid.cols}", f"ceil({grid.col_dim.value} / {_tile_of(grid.col_dim)})"
        ),
        *_constant(f"K_TILES = {k_tiles}", "ceil(K / ROWS)"),
        *_constant(f"M_TILES = {max(1, math.ceil(grid.m / rows))}", "ceil(M / ROWS)"),
        *_constant(f"N_TILES = {max(1, math.ceil(grid.n / cols))}", "ceil(N / COLS)"),
        *(
            _constant(
                f"K_TILE_BOUNDS = {bounds!r}",
                "the K tiles each split-K piece owns: the",
                "sweep is cut in whole instruction tiles",
            )
            if grid.swept_dim is Dim.K
            else []
        ),
        *_constant(
            f"TILES = {grid.tiles}",
            "GRID_ROWS * GRID_COLS" + (" * SPLIT_K" if grid.k_partitions > 1 else ""),
        ),
        "",
        "# Two core counts, and the gap between them is the whole of wave occupancy:",
        *_constant(
            f"USED_CORES = {int(predicted['used_cores'])}",
            "min(AVAILABLE_CORES, TILES) — what there is",
            "work for. One OS thread each, deliberately NOT",
            "capped to this host's CPUs: structure, not speed.",
        ),
        *_constant(f"WAVES = {int(predicted['waves'])}", "ceil(TILES / USED_CORES)"),
        "",
        _rule("5. the prediction"),
        "# What bwz said. Everything here is asserted after the run except the two rows",
        "# the printed table marks tier 2 — quantities the model reaches through a",
        "# capacity heuristic this file deliberately does not imitate, so they are shown",
        "# side by side and left for a reader to judge.",
        "PREDICTED = {",
    ]
    for key, value in predicted.items():
        rendered = f"{int(value):_}" if float(value).is_integer() else repr(value)
        out.append(f'    "{key}": {rendered},')
    out.append("}")
    return out


def _largest_array() -> tuple[int, str]:
    """The biggest array any bundled profile declares, cached for one emission.

    Read from the profiles rather than hardcoded, so the bar moves with the
    repository instead of going stale (D59). Cached because emitting a comparison
    page calls this once per chip and it walks every profile on disk.
    """
    return largest_declared_array()


def _accumulator_note(operand: DType) -> tuple[str, ...]:
    """Why the accumulator is the width it is, in this dtype's own terms.

    Not one sentence for every format: the integer rule and the float one are
    different claims with different force. Saying "a product does not fit one
    operand" of ``fp16`` would imply fp16 accumulate is impossible, and it is a
    real MMA mode — just not anyone's default (D56).
    """
    if is_integer(operand):
        return (
            f"{operand.value} x {operand.value} -> int32, always: the sum",
            "of K products does not fit 8 bits, and int32 is the",
            "only integer MMA shape these arrays issue (spec/dtypes.py)",
        )
    return (
        "what cuBLAS accumulates in by default",
        "(CUBLAS_COMPUTE_32F). fp16 accumulate is a real mode and",
        "not modelled here; bf16 and tf32 have no narrower one",
    )


def _depth_note(
    chip: HardwareSpec, machine: MachineModel, rows: int, cols: int, double_buffered: bool
) -> tuple[str, ...]:
    """Why DEPTH is 2, which is *not* "capacity fits exactly two tiles".

    ``report.memory.double_buffered`` is a **bool**: the planner asks whether
    spare capacity holds *at least* two working tiles, because two is what
    overlapping one load with one compute requires, and the cost model has
    exactly two states — ``max(load, compute)`` or ``load + compute`` (D5a).
    There is no DEPTH 3 to report even when capacity would hold thousands, and a
    comment reading "capacity fits two tiles" invited exactly the question it
    should have answered (D58).

    The headroom quoted is ``ON_CHIP_BYTES`` over one ``ROWS x COLS`` tile — both
    already constants on the page, so a reader can check it — and is deliberately
    *before* the planner's subtraction for resident weights and activations,
    which is what the real test uses (``analysis/memory.py``).
    """
    tile = rows * cols * bytes_per_element(machine.dtype)
    held = int(chip.on_chip_capacity_bytes / tile) if tile > 0 else 0
    if not double_buffered:
        return (
            "DERIVED, and a yes/no: spare capacity does NOT hold two",
            f"{rows}x{cols} tiles at once, so a load cannot overlap the",
            "compute before it and the report's latency is their SUM,",
            "not max(load, compute) (D5a). Not read below.",
        )
    return (
        "DERIVED, and a yes/no dressed as a number: the model asks",
        f"only whether capacity holds TWO {rows}x{cols} tiles at once —",
        "what overlapping one load with one compute needs — not how",
        f"many it would really hold, which here is ~{held:,} before",
        "anything else is resident. There is no DEPTH 3: latency is",
        "max(load, compute) or their sum, nothing between (D5a).",
        "Not read below; this program counts bytes, not time.",
    )


def _tile_of(dim: Dim) -> str:
    return "COLS" if dim is Dim.N else "ROWS"


def _flag(name: str, value: str, *, is_default: bool) -> str:
    """How a strategy constant got its value, in one phrase.

    The distinction the emitted file has to keep is between *you asked for this*
    and *nobody asked, this is the default* — they look identical in the value
    and mean different things to a reader deciding what to try next (D57).
    """
    return f"{name}, not passed" if is_default else f"{name} {value}"


def _stationarity_note(
    grid: TileGrid, dataflow: DataflowPlan, unit: ComputeUnit
) -> tuple[str, ...]:
    """Where the decomposition came from: the chip, or the command line."""
    what = f"{grid.resident.value} resident, {grid.swept_dim.value} swept"
    if dataflow.requested_stationarity is None:
        return (
            f"{unit.name}'s OWN declared dataflow, not a choice made",
            f"here: {what}, straight off the chip profile.",
            "--stationarity picks another the unit declares; one it",
            "does not is refused, never clamped — it would be a",
            "different decomposition, not a slower one (D53)",
        )
    return (
        f"--stationarity {grid.stationarity.value}, requested: {what}.",
        f"{unit.name} declares it, or the run would have been",
        "refused rather than quietly given the chip's own (D53)",
    )


def _split_k_note(grid: TileGrid) -> tuple[str, ...]:
    """Where the split factor came from — which is never a search."""
    if grid.k_partitions > 1:
        return (
            f"--split-k {grid.k_partitions}: the contraction is cut that many",
            "ways, so its partials cross DRAM and CUTLASS's second",
            "kernel sums them (D53). Chosen by you, not searched for",
        )
    return (
        "--split-k, not passed. NOT computed: nothing here looks",
        "for a good split factor — the flag selects, it does not",
        "optimise (D53). 1 means one GEMM kernel and no reduction",
    )


def _a_strategy_note(dataflow: DataflowPlan) -> tuple[str, ...]:
    """Where A's strategy came from, including a clamp if there was one."""
    if dataflow.a_requested is not dataflow.a_strategy:
        return (
            f"--a-strategy {dataflow.a_requested.value} did not fit and was",
            f"clamped to {dataflow.a_strategy.value} — the header says why (D36)",
        )
    flag = _flag(
        "--a-strategy", dataflow.a_strategy.value, is_default=dataflow.a_strategy is AStrategy.STAGE
    )
    if dataflow.a_strategy is AStrategy.STREAM:
        return (
            f"{flag}: A is re-read per tile (D31),",
            f"{dataflow.tiles_per_a_event}x the staged total",
        )
    if dataflow.a_strategy is AStrategy.WHOLE:
        return (f"{flag}: ramped in before wave 0;", "the same bytes as stage, only earlier (D33)")
    return (f"{flag}:", f"A is staged once per {dataflow.group_name} (D33)")


def _residency_note(dataflow: DataflowPlan) -> tuple[str, ...]:
    """Where the residency came from: the strategy, the default, or an override."""
    if dataflow.a_strategy is AStrategy.STREAM:
        return (
            "1 by construction: stream fetches per tile (D31), so",
            "one staging event serves exactly one tile",
        )
    if dataflow.residency_tiles == dataflow.tiles_per_a_event:
        tiles = dataflow.tiles_per_a_event
        return (
            f"default: one staging serves this {dataflow.group_name}'s",
            f"{tiles} tile{'s' if tiles != 1 else ''}, so A crosses DRAM once (D33)",
        )
    return (
        f"--a-residency-tiles: {dataflow.residency_tiles} of the {dataflow.group_name}'s",
        f"{dataflow.tiles_per_a_event}, clamped to a power-of-2 divisor (D36)",
    )


def _sub_cycles(machine: MachineModel) -> int:
    """Sub-cycles one operand takes on this unit at this dtype: 1 unless bit-serial.

    The same partition ``analysis/tiling.py`` makes (D34/D52): a multiplier below
    1 marks a combinational crossbar, whose K-side loss carries a sub-cycle row of
    fill on top of the padding. That is a *rate*, so an emitted program can name
    it and cannot measure it.
    """
    multiplier = machine.unit.dtype_multipliers.get(machine.dtype, 1.0)
    return round(1 / multiplier) if 0 < multiplier < 1 else 1


def _b_dataflow_note(dataflow: DataflowPlan, weight_sets: int) -> tuple[str, ...]:
    """Where B's placement came from, and whether it means anything here."""
    if dataflow.b_requested is not dataflow.b_dataflow:
        return (
            f"--b-dataflow {dataflow.b_requested.value} did not fit and was",
            f"clamped to {dataflow.b_dataflow.value} — the header says why (D36)",
        )
    flag = _flag(
        "--b-dataflow",
        dataflow.b_dataflow.value,
        is_default=dataflow.b_dataflow is BDataflow.WRITE_AHEAD,
    )
    if weight_sets <= 1:
        return (
            flag + ", and moot here: this array stores",
            "no weights, so there is no write to place (D30)",
        )
    if dataflow.b_dataflow is BDataflow.ON_DEMAND:
        return (f"{flag}:", "the write is exposed at compute (D33) — timing only")
    if dataflow.b_dataflow is BDataflow.PERSISTENT:
        return (f"{flag}:", "written once into the array, never displaced (D33)")
    return (f"{flag}:", "the write lands a wave early, behind compute (D33)")


_ACC_MANTISSA_BITS: dict[DType, int] = {
    DType.FP32: 24,
    DType.TF32: 24,
    DType.FP16: 11,
    DType.BF16: 8,
    DType.FP8: 4,
}


def _tolerance(grid: TileGrid, acc_dtype: DType, operand_dtype: DType) -> float:
    """Bound on ``max |C - reference|``, given operands on the dyadic grid.

    Operands are drawn from ``{-7 .. 7}``, divided by 8 for a float format, so
    every product is a multiple of 1/64 no larger than 49/64 and a sum of ``K``
    of them is a multiple of 1/64 no larger than ``49*K/64``. A binary
    accumulator holds every such value **exactly** while ``49*K`` stays inside
    its integer range — the normal case, and then the right tolerance is zero
    and any difference at all is a walk error rather than a rounding one.

    That separation is the point. This program checks a decomposition; whether
    fp16 rounds is a different question, and letting the two share one tolerance
    would make a failure ambiguous. The program prints the achieved error either
    way, so the margin is visible rather than merely asserted.
    """
    if is_integer(operand_dtype):
        return 0.0
    mantissa = _ACC_MANTISSA_BITS.get(acc_dtype, 11)
    if 49 * grid.k < (1 << mantissa):
        return 0.0
    return 4.0 * grid.k * (49.0 / 64.0) * 2.0**-mantissa


# -------------------------------------------------------------------------- the walk
#
# The emitted program has ONE function that walks the decomposition, `walk()`,
# and every split lives in it, outermost first: waves, the core inside a wave,
# the tile's place in the grid (and its split-K piece), A's staging, what stays
# put, the sweep, one instruction, and where the result lands after the last
# wave (D67). Only plumbing that is identical for every decomposition — counted
# DRAM, the scratchpad, starting the threads — stays in the harness. The cost is
# repetition in the generated file; the gain is that `diff` of two programs for
# different stationarities lands every difference inside one function.

_TILE_INDEX: dict[Dim, str] = {Dim.M: "mt", Dim.N: "nt", Dim.K: "kt"}
"""The loop variable naming a tile's position along each dimension, in every nest."""

_TILE_EXTENT: dict[Dim, str] = {Dim.M: "ROWS", Dim.N: "COLS", Dim.K: "ROWS"}
"""The array side that cuts each dimension: M and K by its rows, N by its columns."""

_BODY = " " * 12
"""Indent of one tile's work: ``walk`` > ``core`` > ``for wave`` > here."""


def _commented(code: str, comment: str) -> str:
    """*code* with a trailing comment, aligned so a nest's comments read as a column."""
    return f"{code.ljust(max(60, len(code) + 2))}# {comment}"


_TRAILING_COMMENT = re.compile(r"^(\s*\S.*?)\s{2,}# (.*)$")
"""A statement and its trailing comment, as :func:`_commented` wrote them."""


def _indented(lines: list[Line], by: str) -> list[Line]:
    """*lines* shifted right by *by*, blank lines left blank, tags kept.

    Trailing comments are re-aligned after the shift, so a block built at one
    depth and moved to another still reads as one column.
    """
    out: list[Line] = []
    for text, tag in lines:
        shifted = f"{by}{text}" if text else text
        match = _TRAILING_COMMENT.match(shifted)
        out.append((_commented(match[1], match[2]) if match else shifted, tag))
    return out


def _walk(
    grid: TileGrid,
    dataflow: DataflowPlan,
    unit: ComputeUnit,
    predicted: dict[str, float],
    placement: ReductionPlacement,
) -> list[Line]:
    """``walk()``: the whole decomposition in one function, every split in it.

    Returned as tagged lines. The tags are the animation's own stage vocabulary
    (D42/D43) and they point at real statements — ``dram.read_b(...)`` is the B
    load, ``mma(...)`` is the arithmetic — so the debug view highlights code that
    performs the transfer rather than code that describes one (D54).
    """
    needs_partials = grid.swept_dim is not Dim.K
    out: list[Line] = _plain(
        [
            _rule("the walk"),
            "def walk(dram: Dram, pad: Scratchpad, counters: Counters) -> None:",
            *_levels_docstring(grid, dataflow, unit, predicted, placement),
        ]
    )
    if needs_partials:
        out += _plain(
            [
                "    # Where the k-slices of one output block meet. It exists at all because K is",
                "    # on the grid (D53); under os there is nothing here.",
                "    partials = Partials(M, N, ROWS, COLS, ACC_DTYPE, counters)",
                "",
            ]
        )
    if dataflow.a_strategy is AStrategy.WHOLE:
        out += _plain(
            [
                "    # a_strategy=whole: every band of A is ramped in before wave 0 instead of at",
                "    # the row boundary that needs it. The same statements as the A step below,",
                "    # the same bytes, the same event count — only the timing moves (D33), and",
                "    # timing is the one thing this program does not measure.",
                "    for tile in range(TILES):",
            ]
        )
        out += _position(grid, " " * 8)
        out += _stage_a(grid, dataflow, " " * 8, keep=False)
        out += _plain([""])
    out += _plain(
        [
            "    # == LEVEL 1 + 2: waves, and the cores inside each wave ====================",
            "    # One thread per core, started by run_cores below. Every core runs this same",
            "    # function; they differ only in core_id. end_of_wave() is a barrier: nobody",
            "    # starts wave w+1 until every core has finished wave w (lockstep, D30).",
            "    def core(core_id: int, end_of_wave: Callable[[], None]) -> None:",
            _commented("        for wave in range(WAVES):", "LEVEL 1: waves, one after another"),
            _commented(
                "            tile = wave * USED_CORES + core_id", "LEVEL 2: this core's tile"
            ),
            "            if tile >= TILES:",
            "                # More cores than tiles left: this core sits the wave out. THIS is",
            "                # wave occupancy (D30), counted rather than assumed.",
            "                if DEBUG:",
            "                    log(",
            '                        f"core {core_id:<5} wave {wave:<4} "',
            '                        "idle -- no tile left (D30)"',
            "                    )",
            "                counters.count_idle_core_wave()",
            "                end_of_wave()",
            "                continue",
            "            if DEBUG:",
            '                log_block(f"core {core_id:<5} wave {wave:<4} tile {tile}")',
            "",
        ]
    )
    out += _position(grid, _BODY)
    # Dispatch on the SWEPT dimension, not on the resident operand: what decides
    # the shape of the nest is whether K is swept inside one tile — where it
    # accumulates locally and owes nobody a partial (os, and rs, whose extra
    # spreading of K happens inside one array) — or carried on the grid, where
    # the tile owns a slice of the contraction and must hand partials on (D53).
    if grid.swept_dim is Dim.K:
        body = _accumulator_body(grid, dataflow)
    elif grid.swept_dim is Dim.M:
        body = _weight_body(grid, dataflow, unit)
    else:
        body = _input_body(grid, dataflow)
    out += _indented(body, " " * 8)
    out += _plain(
        [
            f"{_BODY}counters.count_tile()",
            f"{_BODY}if DEBUG:",
            f"{_BODY}    log_flush()",
            _commented(f"{_BODY}end_of_wave()", "wait for every other core"),
            "",
            "    run_cores(",
            "        USED_CORES, core,",
            "        warn_above=LARGEST_DECLARED_CORES,",
            f'        warn_source="{_largest_array()[1]}",',
            "    )",
        ]
    )
    out += _after_the_waves(grid, needs_partials)
    return out


def _k_tile_bounds(grid: TileGrid, rows: int) -> tuple[tuple[int, int], ...]:
    """The K instruction tiles each split-K piece sweeps, as ``(first, past_last)``.

    One piece — the whole of K — when there is no split-K.
    """
    # TODO(D67-open): "partition", "part", "piece", "k_partitions" and K_TILE_BOUNDS
    # all name split-K's cut of K, and "k-slice" names the unrelated cut ws/is make
    # by putting K on the grid. Rename to one clear vocabulary when the Axelera
    # work starts; kept as-is until then so the emitted files stay diffable.
    k_tiles = max(1, math.ceil(grid.k / rows))
    return tuple(
        (part * k_tiles // grid.k_partitions, (part + 1) * k_tiles // grid.k_partitions)
        for part in range(grid.k_partitions)
    )


_PICTURE_MAX = 8
"""The most grid rows or columns :func:`_grid_picture` draws; past it the middle is elided."""


def _shown(count: int) -> list[int | None]:
    """Which indices of *count* to draw: all of them, or the first and last few.

    ``None`` marks the elided middle. Past :data:`_PICTURE_MAX` the picture keeps
    the first four and the last three, so both edges — the ragged last tile
    included — stay visible and the picture never grows past 8 x 8.
    """
    if count <= _PICTURE_MAX:
        return list(range(count))
    tail = (_PICTURE_MAX - 1) // 2
    head = _PICTURE_MAX - 1 - tail
    return [*range(head), None, *range(count - tail, count)]


def _grid_picture(grid: TileGrid, unit: ComputeUnit) -> list[str]:
    """LEVEL 3 drawn: the grid as a box of numbered tiles, then one tile's work.

    The grid is the stationary operand cut into array-sized blocks, so its axes
    are that operand's two dimensions (C: M x N, B: K x N, A: M x K) and the
    third is swept (D53). The picture says so with this run's numbers, then
    follows one tile through its sweep and says where its result goes — which
    is where ``os`` and ``ws``/``is`` part company.
    """
    rows, cols = unit.systolic_dims or (1, 1)
    side = {Dim.M: rows, Dim.N: cols, Dim.K: rows}
    row_dim, col_dim, swept = grid.row_dim, grid.col_dim, grid.swept_dim
    row_var, col_var, swept_var = _TILE_INDEX[row_dim], _TILE_INDEX[col_dim], _TILE_INDEX[swept]
    shown_rows, shown_cols = _shown(grid.rows), _shown(grid.cols)
    heads = [f"{col_var}={c}" if c is not None else "..." for c in shown_cols]
    labels = [
        f"t{r * grid.cols + c}"
        for r in shown_rows
        if r is not None
        for c in shown_cols
        if c is not None
    ]
    width = max(len(text) for text in [*heads, *labels]) + 2

    def rule(left: str, middle: str, right: str) -> str:
        return "  " + left + middle.join("─" * width for _ in shown_cols) + right

    out = [
        f"columns: {col_dim.value} = {grid.extent(col_dim)} -> {grid.cols} chunks of"
        f" {side[col_dim]} ({col_var})",
        f"rows:    {row_dim.value} = {grid.extent(row_dim)} -> {grid.rows} chunks of"
        f" {side[row_dim]} ({row_var})",
        "",
        rule("┌", "┬", "┐"),
    ]
    for position, r in enumerate(shown_rows):
        if r is None:
            cells, tail = ["⋮".center(width) for _ in shown_cols], ""
        else:
            cells = [
                (f"t{r * grid.cols + c}" if c is not None else "…").center(width)
                for c in shown_cols
            ]
            tail = f"  {row_var} = {r}"
        out.append("  │" + "│".join(cells) + "│" + tail)
        last = position == len(shown_rows) - 1
        out.append(rule("└", "┴", "┘") if last else rule("├", "┼", "┤"))
    out.append("   " + " ".join(head.center(width) for head in heads))
    out.append("")

    operand = grid.resident.value
    block = f"{side[row_dim]}x{side[col_dim]}"
    grid_size = f"{grid.rows} x {grid.cols} = {grid.rows * grid.cols:,} tiles"
    if swept is Dim.K:
        out.append(f"{grid_size}, each one a {block} block of C, accumulated in place.")
    elif unit.weight_sets <= 1 and operand == "B":
        out.append(f"{grid_size}, each one a {block} block of B (stays put in name only here).")
    else:
        out.append(f"{grid_size}, each one a {block} block of {operand} that stays put.")
    if grid.k_partitions > 1:
        out += [
            f"x SPLIT_K = {grid.k_partitions}: the whole grid is repeated once per piece of K,",
            f"so there are {grid.tiles:,} tiles; tile = part x {grid.rows * grid.cols}"
            f" + {row_var} x {grid.cols} + {col_var}. Only piece 0 is drawn.",
        ]

    # One tile, followed through its sweep.
    at = {row_dim: min(1, grid.rows - 1), col_dim: min(1, grid.cols - 1)}
    tile = at[row_dim] * grid.cols + at[col_dim]

    def span(dim: Dim) -> str:
        if dim is swept:
            return f"{swept_var}-chunk"
        first = at[dim] * side[dim]
        return f"{first}:{min(first + side[dim], grid.extent(dim))}"

    if swept is Dim.K:
        low, high = _k_tile_bounds(grid, rows)[0]
    else:
        low, high = 0, max(1, math.ceil(grid.extent(swept) / side[swept]))
    held = {"A": (Dim.M, Dim.K), "B": (Dim.K, Dim.N), "C": (Dim.M, Dim.N)}[operand]
    out += [
        "",
        f"Tile t{tile} ({row_var}={at[row_dim]}, {col_var}={at[col_dim]}) holds"
        f" {operand}[{span(held[0])}, {span(held[1])}]; its sweep is:",
        "",
        f"    for {swept_var} in {low}..{high - 1}:",
        f"        C[{span(Dim.M)}, {span(Dim.N)}]  +=  A[{span(Dim.M)}, {span(Dim.K)}]"
        f"  @  B[{span(Dim.K)}, {span(Dim.N)}]",
        "",
    ]
    if swept is Dim.K and grid.k_partitions > 1:
        out += [
            "Every step adds into the same block of C, but only over piece 0's share of K:",
            f"what the tile ends with is a PARTIAL. The other {grid.k_partitions - 1} pieces'"
            " partials go",
            "through DRAM too, and KERNEL 2 adds them all after the last wave.",
        ]
    elif swept is Dim.K:
        out += [
            "Every step adds into the same block of C in this core's accumulator, so",
            "when the loop ends that block is finished: no partial ever leaves the tile.",
        ]
    else:
        others = (grid.rows if row_dim is Dim.K else grid.cols) - 1
        fixed = col_var if row_dim is Dim.K else row_var
        whole = f"C[:, {span(Dim.N)}]" if row_dim is Dim.K else f"C[{span(Dim.M)}, :]"
        out += [
            f"Each step is only a PARTIAL: {whole} also needs the other {others} tile(s)",
            f"with the same {fixed}, one per kt, which may run on other cores. Their",
            "partials meet in `partials`, on chip.",
        ]
    return out


def _levels_docstring(
    grid: TileGrid,
    dataflow: DataflowPlan,
    unit: ComputeUnit,
    predicted: dict[str, float],
    placement: ReductionPlacement,
) -> list[str]:
    """``walk``'s docstring: the levels as a table with this run's numbers, then why.

    The numbers are the constants above — the grid's, and the prediction's
    waves and cores — restated, so a reader comparing two files sees the whole
    difference in the first twenty lines before reading any code.
    """
    rows, cols = unit.systolic_dims or (1, 1)
    side = {Dim.M: rows, Dim.N: cols, Dim.K: rows}
    swept = grid.swept_dim
    steps = max(1, math.ceil(grid.extent(swept) / side[swept]))
    tiles, waves = grid.tiles, int(predicted["waves"])
    used, available = int(predicted["used_cores"]), unit.count
    grid_text = (
        f"{grid.row_dim.value} x {grid.col_dim.value} = {grid.rows} x {grid.cols}"
        + (f" x {grid.k_partitions} split-K pieces" if grid.k_partitions > 1 else "")
        + f" = {tiles:,} tiles"
    )
    sweep_text = f"{swept.value}: {steps:,} steps of {side[swept]}" + (
        f", cut into {grid.k_partitions} pieces" if grid.k_partitions > 1 else ""
    )
    lines = [
        f'    """The whole {_STATIONARITY_TITLE[grid.stationarity]} decomposition,'
        " in ONE function.",
        "",
        "    Every split this program makes is below, outermost first:",
        "",
        f"      LEVEL 1  waves        {waves:,} = ceil({tiles:,} tiles / {available:,} cores)",
        f"      LEVEL 2  cores        {used:,} of {available:,} in use;"
        " each takes one tile per wave",
        f"      LEVEL 3  tile grid    {grid_text}",
        f"               (parallel)   the two dimensions ON the grid: {grid.row_dim.value}"
        f" and {grid.col_dim.value}",
        f"      LEVEL 4  the sweep    {sweep_text}",
        f"               (in order)   the one dimension NOT on the grid: {swept.value}",
        f"      LEVEL 5  one mma()    {rows} x {cols} x {rows} = {rows * rows * cols:,} MAC slots,"
        " the bottom",
        "",
        "    The code of LEVELS 1, 2 and 5 is the same for every stationarity. LEVEL 3 and",
        "    LEVEL 4 are where they differ: which two of M, N, K form the grid, and which",
        "    one is swept. Inside a tile, A is staged into a buffer every core shares (once",
        f"    per {dataflow.group_name}), one operand stays put, and the others stream past it.",
        "",
        "    LEVEL 3, drawn. Each box is one tile, numbered as `tile` is below:",
        "",
    ]
    lines += [f"      {line}" if line else "" for line in _grid_picture(grid, unit)]
    lines += [""]
    lines += [f"    {line}" if line else "" for line in _explanation(grid, unit)]
    if grid.swept_dim is not Dim.K:
        lines += _placement_note(placement)
    lines.append('    """')
    return lines


def _explanation(grid: TileGrid, unit: ComputeUnit) -> list[str]:
    """The paragraph that says what this decomposition means, per stationarity."""
    if grid.stationarity is Dataflow.ROW_STATIONARY:
        return [
            "Row-stationary: a row of A lives in each PE and K reduces across the array's",
            "own columns, so no partial sum leaves a core. The difference from",
            "output-stationary is INSIDE one array, which a program at this granularity",
            "cannot show and does not pretend to (Eyeriss, ISCA 2016).",
        ]
    if grid.swept_dim is Dim.K:
        return [
            "Output-stationary: C stays in one core's accumulator while K is swept inside",
            "the tile, which is exactly why it owes no reduction: no partial sum ever",
            "leaves a tile (D53)."
            + (
                " Except under split-K, which cuts K on purpose and pays a second kernel."
                if grid.k_partitions > 1
                else ""
            ),
        ]
    if grid.swept_dim is Dim.M:
        holds = unit.weight_sets > 1
        lines = [
            "One weight tile. B stays resident; M streams past it."
            if holds
            else "One k-slice of the contraction. Nothing is held — K is on the grid.",
            "",
            "Weight-stationary: K is on the grid, so each tile owns one k-slice and",
            "produces a PARTIAL over K, not a finished result. The slices of one output",
            "block meet in `partials`, which counts the additions it performs — that it",
            "has to exist at all is the difference from output-stationary.",
        ]
        if not holds:
            lines += [
                "",
                f"{unit.name} declares weight_sets=1, so it stores no weights of its own and",
                "the NAME of this dataflow is a misnomer here: the tile of B is read per",
                "instruction like everything else (D30). What ws really changed is the",
                "GRID — its rows are slices of K rather than bands of M.",
            ]
        return lines
    return [
        "Input-stationary: A's tile stays put and N streams past it. K is on the grid",
        "(its columns), so each tile owns a slice of the contraction exactly as a",
        "weight-stationary one does, and produces PARTIALS over K (D53). They meet in",
        "`partials`, which counts the additions it performs.",
    ]


def _position(grid: TileGrid, indent: str) -> list[Line]:
    """LEVEL 3: which tile of the grid this is, and the slice of each dimension it owns.

    Inline rather than behind ``tile_row``/``tile_col`` helpers: the index
    arithmetic is where the grid's two axes get their names, and a helper would
    hide exactly the line that differs between stationarities. Split-K stacks
    ``SPLIT_K`` whole grids one after another, so only a K-swept grid decodes a
    ``part``; a K-on-grid one has none to decode (D53).
    """
    row, col = _TILE_INDEX[grid.row_dim], _TILE_INDEX[grid.col_dim]
    rows_label = f"grid row    -> {grid.row_dim.value}"
    cols_label = f"grid column -> {grid.col_dim.value}"
    lines = [
        f"{indent}# == LEVEL 3: the tile grid — where this tile sits in the"
        f" {grid.row_dim.value} x {grid.col_dim.value} grid"
    ]
    if grid.swept_dim is Dim.K:
        lines += [
            # TODO(D67-open): "part"/"piece" is split-K's name for a cut of K; rename
            # with the rest of that vocabulary (see _k_tile_bounds).
            _commented(f"{indent}part = tile // (GRID_ROWS * GRID_COLS)", "split-K piece"),
            _commented(
                f"{indent}{row} = (tile % (GRID_ROWS * GRID_COLS)) // GRID_COLS", rows_label
            ),
        ]
    else:
        lines += [_commented(f"{indent}{row} = tile // GRID_COLS", rows_label)]
    lines += [_commented(f"{indent}{col} = tile % GRID_COLS", cols_label)]
    for dim in (grid.row_dim, grid.col_dim):
        name, index, extent = dim.value.lower(), _TILE_INDEX[dim], _TILE_EXTENT[dim]
        lines.append(
            f"{indent}{name}0, {name}1 = {index} * {extent}, "
            f"min({index} * {extent} + {extent}, {dim.value})"
        )
    if grid.swept_dim is Dim.K:
        lines += [
            _commented(
                f"{indent}kt0, kt1 = K_TILE_BOUNDS[part]", "this piece's slice of the sweep"
            ),
            f"{indent}k_lo, k_hi = kt0 * ROWS, min(kt1 * ROWS, K)",
        ]
    return _plain(lines)


def _stage_a(grid: TileGrid, dataflow: DataflowPlan, indent: str, *, keep: bool) -> list[Line]:
    """A's band, staged into the scratchpad every core shares.

    The same lines open ``walk`` under ``a_strategy=whole`` (*keep* false,
    nothing to bind the band to): whole and stage move the same bytes and differ
    only in *when* the events happen (D33), which this file shows by running the
    same statement earlier. The key is (split-K piece, grid row, residency
    chunk); one staging event per key is what makes the three A strategies one
    mechanism (D31/D33).
    """
    row, col = _TILE_INDEX[grid.row_dim], _TILE_INDEX[grid.col_dim]
    piece = "part" if grid.swept_dim is Dim.K else "0"
    if grid.row_dim is Dim.K:
        fetch = _commented(
            f"{indent}    lambda: dram.read_a(0, M, k0, k1),", "the whole M height of one k-slice"
        )
    elif grid.col_dim is Dim.K:
        fetch = _commented(
            f"{indent}    lambda: dram.read_a(m0, m1, 0, K),", "one band of M, all of K"
        )
    else:
        fetch = _commented(
            f"{indent}    lambda: dram.read_a(m0, m1, k_lo, k_hi),",
            "one band of M, this piece of K",
        )
    head = [
        f"-- A: staged once per {dataflow.group_name}, into ONE buffer all cores share",
        "   (consecutive tiles of a row run on DIFFERENT cores). A_RESIDENCY_TILES is",
        "   how many tiles one key serves: the whole row for stage/whole, so A crosses",
        "   DRAM once; 1 for stream, so every tile re-fetches (D31/D33).",
    ]
    if keep:
        # Inside core(): every core runs this line, which reads as if each one
        # loaded its own band. Say plainly that it is a lookup, not a load.
        head += [
            "",
            "   pad behaves like a CACHE (a scratchpad shared by every core). Every core",
            "   runs this line, and every core whose tile is in the same grid row asks for",
            "   the same key. Only the FIRST to arrive misses: its fetch reads the band",
            "   from DRAM, and that is one staging event. After that the cache is hot, so",
            "   for every other core this call just reads the cache — no DRAM traffic,",
            "   nothing counted.",
        ]
        if dataflow.a_strategy is AStrategy.WHOLE:
            head += [
                "   Here, under whole, the prologue before the waves already filled every",
                "   key, so no core misses at all: every call below is a cache read.",
            ]
        else:
            head += ["   (Under stream every tile has its own key, so every call misses.)"]
    return [
        *_plain([f"{indent}# {line}" for line in head]),
        (f"{indent}{'band = ' if keep else ''}pad.band(", "load_a"),
        (
            _commented(f"{indent}    ({piece}, {row}, {col} // A_RESIDENCY_TILES),", "the key"),
            "load_a",
        ),
        (fetch, "load_a"),
        (f"{indent})", "load_a"),
    ]


# One tile's work, per stationarity. Written at indent 4 and shifted into place
# by `_walk`, so each reads as a block and the three can be compared here too.


def _accumulator_body(grid: TileGrid, dataflow: DataflowPlan) -> list[Line]:
    """``os``/``rs``: C stays in the accumulator and K is swept inside one tile."""
    out: list[Line] = _plain(
        [
            "    if DEBUG:",
            '        log(f"  C[{m0}:{m1}, {n0}:{n1}]  piece {part}, sweeping kt {kt0}..{kt1}")',
            "",
        ]
    )
    out += _stage_a(grid, dataflow, "    ", keep=True)
    out += _plain(
        [
            "",
            "    # -- what stays put: C, in this core's accumulator",
            _commented("    acc = zeros(m1 - m0, n1 - n0, ACC_DTYPE)", "stays put"),
            "",
            "    # == LEVEL 4: the sweep — K, INSIDE this tile",
            _commented("    for kt in range(kt0, kt1):", "K is swept INSIDE this tile"),
            "        k0, k1 = kt * ROWS, min(kt * ROWS + ROWS, K)",
            _commented(
                "        a = sub(band, 0, m1 - m0, k0 - k_lo, k1 - k_lo)", "already on chip"
            ),
        ]
    )
    out += [(_commented("        b = dram.read_b(k0, k1, n0, n1)", "crosses DRAM"), "load_b")]
    out += _plain(
        [
            "        if DEBUG:",
            "            log(",
            '                f"    kt={kt:<4} A[{m0}:{m1}, {k0}:{k1}] @ B[{k0}:{k1}, {n0}:{n1}]"',
            '                f"  {(m1 - m0) * (k1 - k0) * (n1 - n0):>9,} useful of "',
            '                f"{SLOTS_PER_MMA:,} slots"',
            "            )",
            "        # == LEVEL 5: one instruction",
        ]
    )
    out += [
        ("        mma(acc, a, b, counters, SLOTS_PER_MMA)", "exec"),
        ("", None),
    ]
    if grid.materialises_partials:
        out += _plain(
            [
                "    # Kernel 1 of two. This piece owns K_TILE_BOUNDS[part] of the contraction and",
                "    # nothing else, so what it holds is a partial — and the GEMM ends before the",
                "    # reduction begins, so that partial cannot stay in a register (D53).",
            ]
        )
        out += [("    dram.write_partial(part, m0, n0, acc)", "store")]
    else:
        out += [(_commented("    dram.write_c(m0, n0, acc)", "finished, not a partial"), "store")]
    return out


def _weight_body(grid: TileGrid, dataflow: DataflowPlan, unit: ComputeUnit) -> list[Line]:
    """``ws``: K is on the grid and M streams past a tile of B, giving partials.

    "Weight-stationary" is a claim about hardware and is a misnomer on an MMA
    unit, which holds nothing (D30/D62); the comment says which one this is.
    """
    out: list[Line] = _plain(
        [
            "    if DEBUG:",
            '        log(f"  B[{k0}:{k1}, {n0}:{n1}] resident, M streams past it")',
            "",
        ]
    )
    out += _stage_a(grid, dataflow, "    ", keep=True)
    out += _plain(
        [
            "",
            "    # -- what stays put: B's tile"
            + ("" if unit.weight_sets > 1 else " (in name only here: see the docstring)"),
        ]
    )
    out += [(_commented("    b = dram.read_b(k0, k1, n0, n1)", "stays put"), "load_b")]
    out += _plain(
        [
            "",
            "    # == LEVEL 4: the sweep — M streams past; each step writes a DIFFERENT block of C",
            _commented("    for mt in range(M_TILES):", "M streams past the resident tile"),
            "        m0, m1 = mt * ROWS, min(mt * ROWS + ROWS, M)",
            _commented("        a = sub(band, m0, m1, 0, k1 - k0)", "already on chip"),
            "        product = zeros(m1 - m0, n1 - n0, ACC_DTYPE)",
            "        if DEBUG:",
            "            log(",
            '                f"    mt={mt:<4} A[{m0}:{m1}, {k0}:{k1}] @ B[{k0}:{k1}, {n0}:{n1}]"',
            '                f" -> PARTIAL into C[{m0}:{m1}, {n0}:{n1}]"',
            "            )",
            "        # == LEVEL 5: one instruction",
        ]
    )
    return out + _partial_step(grid)


def _input_body(grid: TileGrid, dataflow: DataflowPlan) -> list[Line]:
    """``is``: A stays resident and N streams past it, producing partials over K."""
    out: list[Line] = _plain(
        [
            "    if DEBUG:",
            '        log(f"  A[{m0}:{m1}, {k0}:{k1}] resident, N streams past it")',
            "",
        ]
    )
    out += _stage_a(grid, dataflow, "    ", keep=True)
    out += _plain(
        [
            "",
            "    # -- what stays put: A's tile, cut from the staged band",
            _commented("    a = sub(band, 0, m1 - m0, k0, k1)", "stays put"),
            "",
            "    # == LEVEL 4: the sweep — N streams past; each step writes a DIFFERENT block of C",
            _commented("    for nt in range(N_TILES):", "N streams past the resident tile"),
            "        n0, n1 = nt * COLS, min(nt * COLS + COLS, N)",
        ]
    )
    out += [(_commented("        b = dram.read_b(k0, k1, n0, n1)", "crosses DRAM"), "load_b")]
    out += _plain(
        [
            "        product = zeros(m1 - m0, n1 - n0, ACC_DTYPE)",
            "        if DEBUG:",
            "            log(",
            '                f"    nt={nt:<4} A[{m0}:{m1}, {k0}:{k1}] @ B[{k0}:{k1}, {n0}:{n1}]"',
            '                f" -> PARTIAL into C[{m0}:{m1}, {n0}:{n1}]"',
            "            )",
            "        # == LEVEL 5: one instruction",
        ]
    )
    return out + _partial_step(grid)


def _partial_step(grid: TileGrid) -> list[Line]:
    """The instruction and the partial it hands on, shared by ``ws`` and ``is``.

    The comment above ``partials.accumulate`` is the one place the program says
    who else writes the block this step produces: nothing in the loop shows it,
    because the other writers are other tiles, on other cores.
    """
    k_on_rows = grid.row_dim is Dim.K
    slices = grid.rows if k_on_rows else grid.cols
    swept_var = _TILE_INDEX[grid.swept_dim]
    fixed_var = _TILE_INDEX[grid.col_dim if k_on_rows else grid.row_dim]
    if slices <= 1:
        note = [
            "-- where the product goes: K fits in ONE instruction tile here, so this",
            "   tile is the only writer of every block it touches. Nothing is added;",
            "   `partials` just holds C until partials.drain() writes it out.",
        ]
    else:
        note = [
            "-- where the product goes. It is C[m0:m1, n0:n1] over K = k0:k1 ONLY:",
            f"   1 of the {slices} k-slices that block needs. The other {slices - 1} come from",
            f"   the tiles with the same {fixed_var} and a different kt, running on OTHER",
            "   cores, maybe in this same wave. All of them land in ONE shared copy of",
            "   C held on chip (`partials`, one lock per block):",
            "",
            "     the FIRST tile to arrive   → copies its product in (C starts at zero,",
            "                                  so this is not counted as an addition)",
            f"     each of the other {slices - 1}".ljust(32)
            + "→ adds its product on top (counted: the",
            "                                  report charges these to the vector unit)",
            "",
            "   Which tile arrives first is thread timing and changes nothing. Steps of",
            f"   THIS loop never add to each other — each {swept_var} is a different block of C.",
            "   The finished C leaves the chip once, in partials.drain(), after the last",
            "   wave.",
        ]
    return [
        ("        mma(product, a, b, counters, SLOTS_PER_MMA)", "exec"),
        ("", None),
        *_plain([f"        # {line}".rstrip() if line else "        #" for line in note]),
        (
            _commented(
                "        partials.accumulate(m0, n0, product)", "this tile's share; see above"
            ),
            # "reduce", not "exec": this line IS the reduction the report charges
            # to the vector unit and the trace draws on the vector lane (D62), so
            # tagging it "exec" left the animation with a vector bar playing and
            # no line lit under it (D66).
            "reduce",
        ),
    ]


def _after_the_waves(grid: TileGrid, needs_partials: bool) -> list[Line]:
    """What happens once every wave is done: where C finally lands.

    ``os`` wrote C inside each tile, so nothing. A K-on-grid walk drains its
    shared accumulator. Split-K runs its second kernel — inline here rather than
    as its own function, because it is a split too: the one that undoes
    ``SPLIT_K``.
    """
    if needs_partials:
        return [
            *_plain(
                [
                    "",
                    "    # == after the last wave: C lands. The accumulator never crossed DRAM;",
                    "    # C's own compulsory write does (D53).",
                ]
            ),
            ("    partials.drain(dram, M, N)", "store"),
        ]
    if not grid.materialises_partials:
        return []
    out: list[Line] = _plain(
        [
            "",
            "    # == after the last wave: KERNEL 2, sum the SPLIT_K partials into C (D53).",
            "    # A separate launch in CUTLASS: the GEMM has to finish everywhere before any",
            "    # of this can start, which is precisely why the partials could not stay in",
            "    # registers and crossed DRAM twice. Not new arithmetic — 2*M*N*K already",
            "    # counts these adds; they have merely left the array's own accumulator for",
            "    # the vector unit, which is slower, and that is the trade split-K makes.",
            "    for m0 in range(0, M, ROWS):",
            "        for n0 in range(0, N, COLS):",
            "            m1, n1 = min(m0 + ROWS, M), min(n0 + COLS, N)",
            "            acc = copy_of(dram.read_partial(0, m0, m1, n0, n1))",
            "            for part in range(1, SPLIT_K):",
        ]
    )
    out += [
        ("                add_into(acc, dram.read_partial(part, m0, m1, n0, n1))", "reduce"),
        (
            "                counters.count_partial_sum_adds((m1 - m0) * (n1 - n0))",
            "reduce",
        ),
        ("            dram.write_c(m0, n0, acc)", "store"),
    ]
    return out


def _placement_note(placement: ReductionPlacement) -> tuple[str, ...]:
    """Where the report says these additions happen, and what it charged (D62).

    The count is the same under every placement and the program measures it the
    same way; what differs is the *price*, and quoting one placement's price in
    a file emitted for another would be exactly the kind of adjacent-to-true
    comment this repository keeps having to correct.
    """
    if placement is ReductionPlacement.NONE:
        return (
            "",
            "    Not that there are any here: K fits inside ONE instruction tile at this shape,",
            "    so the grid has a single k-slice, every block is written once and nothing is",
            "    summed. The count below is 0, and the report charges no reduction.",
        )
    if placement is ReductionPlacement.LOCAL:
        return (
            "",
            "    The report charges NOTHING for them: this unit declares an accumulator deep",
            "    enough for the whole contraction, so the partials are summed in its own",
            "    periphery and never reach on-chip memory (D62). The count below is still",
            "    real — it is the work the hardware absorbed.",
        )
    if placement is ReductionPlacement.ON_CHIP:
        return (
            "",
            "    The report charges these to the VECTOR unit — a matrix engine does",
            "    matrix-multiply-accumulate and nothing else (D27) — and OVERLAPS them with",
            "    the arithmetic above: it costs max(matrix, vector), not their sum. The",
            "    overlap is the one part of that this program cannot check; it counts.",
        )
    if placement is ReductionPlacement.DRAM:
        return (
            "",
            "    The report charges these to the VECTOR unit (D27) and, because the whole",
            "    M x N accumulator does not fit on chip, a DRAM round trip besides — so they",
            "    serialise behind the arithmetic rather than overlapping it (D62).",
        )
    return ()


# -------------------------------------------------------------------------- the main


_CHECKS = """    checks = [
        Check("tiles", PREDICTED["tiles"], counters.tiles_run, True),
        Check("waves", PREDICTED["waves"], WAVES, True),
        Check("MACs", PREDICTED["macs"], counters.macs, True),
        Check("MAC slots issued", PREDICTED["mac_slots"], counters.mac_slots, True),
        Check(
            "idle core-waves",
            PREDICTED["idle_core_waves"],
            counters.idle_core_waves + (AVAILABLE_CORES - USED_CORES) * WAVES,
            True,
            note="a core with no tile this wave. This IS wave occupancy (D30).",
        ),
        Check("A staging events", PREDICTED["staging_events"], counters.staging_events, True),
        Check("A bytes", PREDICTED["a_dram_bytes"], counters.a_dram_bytes, True),
        Check("C bytes", PREDICTED["c_dram_bytes"], counters.c_dram_bytes, True),
{partial_bytes}
        Check(
            "partial-sum additions",
            PREDICTED["partial_sum_adds"],
            counters.partial_sum_adds,
            True,
            note=(
                "(p-1) x M x N, where p is how many k-slices each output element ends\\n"
                "up with. Not new arithmetic — 2*M*N*K already counts them — but the\\n"
                "report charges them to the VECTOR unit, because they have left the\\n"
                "matrix engine's own accumulator (D27/D62)."
            ),
        ),
        Check(
            "B bytes fetched",
            PREDICTED["b_dram_bytes_charged"],
            counters.b_dram_bytes,
            False,
            note=(
                "tier 2. The report charges compulsory traffic and then discounts it by a\\n"
                "residency fraction a capacity heuristic supplies; this walk fetches what\\n"
                "the tile order asks for. The gap above it is the tiling re-read that\\n"
                "docs/MODEL.md 6.2 declines to model, measured rather than argued about."
            ),
        ),
        Check(
            "B bytes, first touch",
            PREDICTED["b_dram_bytes_charged"],
            counters.b_compulsory_bytes,
            False,
            note="tier 2. What B costs if every byte of it crosses the bus exactly once.",
        ),
    ]
    failed = report_checks(checks)"""


_PARTIAL_BYTES_TIER_1 = (
    '        Check("partial bytes", PREDICTED["partial_dram_bytes"], '
    "counters.partial_dram_bytes, True),"
)

_PARTIAL_BYTES_TIER_2 = """        Check(
            "partial bytes",
            PREDICTED["partial_dram_bytes"],
            counters.partial_dram_bytes,
            False,
            note=(
                "tier 2, and the one place this file and the report model different\\n"
                "machines. The report found the M x N accumulator too big for on-chip\\n"
                "capacity, so it charged the partials a DRAM round trip (D62); this walk\\n"
                "keeps them in `partials` whatever their size, because where an\\n"
                "accumulator lives is a capacity heuristic and not a decomposition. The\\n"
                "gap above IS that heuristic, which is why it is shown and not asserted."
            ),
        ),"""


def _checks(grid: TileGrid, placement: ReductionPlacement) -> list[str]:
    """The check list, with the partial-bytes row at the tier it can defend.

    Tier 1 everywhere the program and the report agree about where the partials
    live — nowhere at all (no reduction), on chip (this file's `Partials`), or
    in DRAM because there are two kernels (split-K, which this file emits). Tier
    2 for the one case they disagree: a K-on-grid walk whose accumulator the
    report found too big to hold. That is a capacity judgement about a machine,
    not a step of the walk, so the file shows the gap rather than asserting a
    round trip it has no reason to perform (D54's rule, D62's case).
    """
    spilled = placement is ReductionPlacement.DRAM and not grid.materialises_partials
    row = _PARTIAL_BYTES_TIER_2 if spilled else _PARTIAL_BYTES_TIER_1
    return _CHECKS.format(partial_bytes=row).splitlines()


def _main(
    chip: HardwareSpec,
    machine: MachineModel,
    grid: TileGrid,
    dataflow: DataflowPlan,
    placement: ReductionPlacement,
) -> list[Line]:
    title = f"{chip.name} · {machine.dtype.value} · {_STATIONARITY_TITLE[grid.stationarity]}"
    lines = [
        _rule("run it"),
        "def main() -> int:",
        f'    print("{title}")',
        '    print(f"backend: {BACKEND};  {USED_CORES} threads, one per modelled core in use")',
        "    if DEBUG:",
        "        # One line per instruction tile, so the count is PREDICTED's own:",
        "        # mac_slots / SLOTS_PER_MMA is exactly how many mma() calls there are.",
        '        issued = PREDICTED["mac_slots"] // SLOTS_PER_MMA',
        "        log(",
        '            f"--debug: {TILES:,} tile(s) over {USED_CORES:,} core(s) in "',
        '            f"{WAVES:,} wave(s), issuing {issued:,} instruction tile(s)."',
        "        )",
        '        log("         core/wave assignment, A staging events, and every")',
        '        log("         instruction tile follow. Expect one line each.")',
        "",
        "    counters = Counters()",
        "    a = operand(M, K, A_DTYPE, SEED)",
        "    b = operand(K, N, B_DTYPE, SEED + 1)",
        "    c = zeros(M, N, ACC_DTYPE)",
        "    dram = Dram(",
        "        a, b, c, counters,",
        "        a_bytes_per_element=A_BYTES_PER_ELEMENT,",
        "        b_bytes_per_element=B_BYTES_PER_ELEMENT,",
        "        c_bytes_per_element=C_BYTES_PER_ELEMENT,",
        "        acc_bytes_per_element=PARTIAL_BYTES_PER_ELEMENT,",
        "        partitions=SPLIT_K,",
        "        acc_dtype=ACC_DTYPE,",
        "    )",
        "    pad = Scratchpad(counters)",
    ]
    lines += [
        "",
        "    walk(dram, pad, counters)                    # every split is in here",
    ]
    out: list[Line] = _plain(lines)
    tail = [
        "",
        "    error = max_abs_diff(c, reference(a, b, INTEGER))",
        # Split rather than appended whole: every element of this list is one
        # emitted line, and the stage-line numbering counts elements.
        *_checks(grid, placement),
        "",
        "    occupancy = counters.occupancy(WAVES, USED_CORES, AVAILABLE_CORES)",
        "    padding = counters.padding_efficiency()",
        '    print("")',
        "    print(",
        '        f"wave occupancy   {occupancy:.4f}   "',
        '        "1 - idle core-waves / (WAVES * AVAILABLE_CORES)"',
        "    )",
        '    print(f"shape padding    {padding:.4f}   useful MACs / MAC slots issued (D52)")',
        "    print(",
        '        f"utilisation      {padding * occupancy:.4f}   against the report\'s "',
        "        f\"{PREDICTED['utilization']:.4f}\"",
        "    )",
    ]
    if _sub_cycles(machine) > 1:
        tail += [
            '    print("                 The report\'s figure is the smaller of the two: on a")',
            '    print("                 bit-serial array it also charges a sub-cycle row of")',
            '    print("                 fill on K (D34), which is a RATE. Nothing here")',
            '    print("                 measures rates, so this program cannot see it.")',
        ]
    tail += [
        '    print("")',
        '    print(f"numerics: max |C - A@B| = {error:g}   (tolerance {TOLERANCE:g})")',
        # A one-number fingerprint of C: two decompositions of the same matmul
        # must agree on it, which is a stronger claim than each passing alone.
        '    print(f"C checksum: {checksum(c):.0f}")',
        "",
        "    if failed or error > TOLERANCE:",
        "        raise SystemExit(",
        '            f"{len(failed)} tier-1 count(s) disagree with the report"',
        '            + ("" if error <= TOLERANCE else " and the result itself is wrong")',
        '            + ". One of the two is wrong, and finding out which is what this file"',
        '            + " is for."',
        "        )",
        '    print("")',
        '    print("every tier-1 count matches the report, and C == A @ B.")',
        "    return 0",
    ]
    return out + _plain(tail)


# ----------------------------------------------------------------------- inline glue


def _rule(title: str) -> str:
    """A section banner, right-aligned to the emitted file's own gutter."""
    return f"# {'-' * max(3, 86 - len(title))} {title}"


def _harness_imports() -> list[str]:
    """The harness's own top-level imports, hoisted to the emitted file's head."""
    return _harness_split()[0]


def _harness_body() -> list[str]:
    """The harness, minus its imports, with its module docstring as a comment."""
    return _harness_split()[1]


def _harness_split() -> tuple[list[str], list[str]]:
    """Split ``_harness.py`` into (imports, body), by parsing rather than by regex.

    ``inspect.getsource`` needs the module on disk as source — true for this
    checkout and for an installed wheel; the failure mode is a clear error, never
    a corrupt emission. Inlining by source rather than by copy is what keeps
    exactly one copy of the runtime, and it is the copy ruff, mypy and the unit
    tests see.
    """
    source = inspect.getsource(_harness)
    lines = source.splitlines()
    tree = ast.parse(source)
    drop: set[int] = set()
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            end = node.end_lineno or node.lineno
            drop.update(range(node.lineno, end + 1))
            if not (isinstance(node, ast.ImportFrom) and node.module == "__future__"):
                imports.extend(lines[node.lineno - 1 : end])
    banner: list[str] = []
    docstring = ast.get_docstring(tree, clean=False)
    if docstring is not None:
        first = tree.body[0]
        drop.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
        banner = [_rule("the harness, inlined from bwz/emit/_harness.py")]
        banner += [f"# {line}".rstrip() for line in docstring.strip().splitlines()]
    body = [line for number, line in enumerate(lines, 1) if number not in drop]
    return imports, banner + body


def _tidy(lines: list[Line]) -> tuple[str, tuple[tuple[str, tuple[int, ...]], ...]]:
    """Render the pairs, and report where each tagged line ended up.

    The line *numbers* have to be computed here rather than while building,
    because this is where blank-line runs are collapsed and leading blanks
    trimmed — do it earlier and every index below the first collapse is wrong.
    """
    kept: list[Line] = []
    blanks = 0
    for text, tag in lines:
        blanks = blanks + 1 if not text.strip() else 0
        if blanks <= 2:
            kept.append((text.rstrip(), tag))
    while kept and not kept[0][0]:
        kept.pop(0)
    while kept and not kept[-1][0]:
        kept.pop()

    stages: dict[str, list[int]] = {}
    for index, (_text, tag) in enumerate(kept):
        if tag is not None:
            stages.setdefault(tag, []).append(index)
    source = "\n".join(text for text, _tag in kept) + "\n"
    return source, tuple((tag, tuple(indices)) for tag, indices in stages.items())
