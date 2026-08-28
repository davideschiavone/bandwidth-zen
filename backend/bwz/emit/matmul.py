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
from dataclasses import dataclass

from bwz.analysis.dataflow import DataflowPlan
from bwz.analysis.roofline import MachineModel
from bwz.analysis.stationarity import Dim, TileGrid
from bwz.analysis.tiling import padded
from bwz.emit import _harness
from bwz.report import OpResult
from bwz.spec.deployment import AStrategy, BDataflow
from bwz.spec.dtypes import DType, bytes_per_element, is_integer
from bwz.spec.hardware_spec import Dataflow, HardwareSpec
from bwz.units import format_bytes

COMMENT_COLUMN = 40
"""Where a constant's trailing comment starts. Every constant in the emitted file
carries one naming where it came from, and they line up so the whole block reads
as a table."""


@dataclass(frozen=True, slots=True)
class EmittedProgram:
    """A runnable program, and the numbers it was told to expect.

    ``predicted`` is kept beside the source so :func:`bwz.emit.check` can compare
    it against the ``Report`` without re-parsing the file — the same contract
    :class:`bwz.deploy.Deployment` has with ``deploy.check``.
    """

    chip_id: str
    filename: str
    source: str
    predicted: dict[str, float]
    stationarity: Dataflow
    working_set_bytes: float


_STATIONARITY_TITLE: dict[Dataflow, str] = {
    Dataflow.OUTPUT_STATIONARY: "output-stationary",
    Dataflow.WEIGHT_STATIONARY: "weight-stationary",
    Dataflow.INPUT_STATIONARY: "input-stationary",
    Dataflow.ROW_STATIONARY: "row-stationary",
}


def default_filename(chip_id: str, dtype: DType, stationarity: Dataflow, k_partitions: int) -> str:
    """``matmul-<chip>-<dtype>-<stationarity>[-splitk<N>].py``."""
    suffix = f"-splitk{k_partitions}" if k_partitions > 1 else ""
    return f"matmul-{chip_id}-{dtype.value}-{stationarity.value}{suffix}.py"


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
    acc_dtype: DType,
    double_buffered: bool,
    command: str,
    version: str,
    seed: int = 20240501,
) -> EmittedProgram:
    """Build the runnable program for this chip, dtype and strategy."""
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

    lines: list[str] = ["#!/usr/bin/env python3"]
    lines += _docstring(chip, machine, grid, dataflow, command, version, working_set)
    lines += ["from __future__ import annotations", ""]
    lines += _harness_imports()
    lines += ["", ""]
    lines += _constants(
        chip,
        machine,
        grid,
        dataflow,
        predicted,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        acc_dtype=acc_dtype,
        double_buffered=double_buffered,
        seed=seed,
    )
    lines += ["", ""]
    lines += _harness_body()
    lines += ["", ""]
    lines += _loop_nest(grid, dataflow)
    lines += ["", ""]
    lines += _main(chip, machine, grid, dataflow)

    return EmittedProgram(
        chip_id=chip.id,
        filename=default_filename(chip.id, machine.dtype, grid.stationarity, grid.k_partitions),
        source=_tidy("\n".join(lines)),
        predicted=predicted,
        stationarity=grid.stationarity,
        working_set_bytes=working_set,
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
) -> list[str]:
    """The file's own header: what it is, what it is not, and what it models."""
    title = _STATIONARITY_TITLE[grid.stationarity]
    out = [
        f'"""{chip.name} · {machine.dtype.value} · {title} — the decomposition bwz costed.',
        "",
        f"    $ {command}",
        f"    bwz {version}",
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
        "What is really executed here, and what is only written down:",
        "",
        "  stationarity   real — a different loop nest, a different resident buffer",
        "  split-K        real — per-partition partials and a second reduction pass",
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
    bounds = tuple(
        (part * k_tiles // grid.k_partitions, (part + 1) * k_tiles // grid.k_partitions)
        for part in range(grid.k_partitions)
    )

    out = [
        _rule("1. the shape"),
        "# From the command line at the top. Every constant below is that, the chip",
        "# profile, or derived from the two — nothing here is a free parameter.",
        *_constant(f"M, N, K = {grid.m}, {grid.n}, {grid.k}", "-M / -N / -K"),
        *_constant(f"A_DTYPE, B_DTYPE, C_DTYPE = {widths}", "-d, --a, --b, --out"),
        *_constant(
            f'ACC_DTYPE = "{acc_dtype.value}"',
            "deployment.precision.accumulate — wider than the",
            f"operands on purpose: a product of two {a_dtype.value}s",
            "does not fit one of them",
        ),
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
            f"SLOTS_PER_MMA = {rows * rows * cols:_}",
            "ROWS * ROWS * COLS — MAC positions one",
            "instruction tile issues, used or not (D52)",
        ),
        "",
        _rule("3. the strategy"),
        *_constant(
            f'STATIONARITY = "{grid.stationarity.value}"',
            f"{grid.resident.value} resident, {grid.swept_dim.value} swept",
        ),
        *_constant(
            f"SPLIT_K = {grid.k_partitions}",
            "CUTLASS's two kernels (D53)" if grid.k_partitions > 1 else "one kernel",
        ),
        *_constant(f'A_STRATEGY = "{dataflow.a_strategy.value}"', _a_strategy_note(dataflow)),
        *_constant(
            f"A_RESIDENCY_TILES = {dataflow.residency_tiles}",
            f"tiles one staging serves, of the {dataflow.tiles_per_a_event}",
            f"in a {dataflow.group_name}",
        ),
        *_constant(
            f'B_DATAFLOW = "{dataflow.b_dataflow.value}"',
            _b_dataflow_note(dataflow, unit.weight_sets),
        ),
        *_constant(
            f"ITERATIONS = {dataflow.iterations}", "invocations the report's bytes stand for"
        ),
        *_constant(
            f"DEPTH = {2 if double_buffered else 1}",
            "capacity fits two tiles" if double_buffered else "no room for a second tile",
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


def _tile_of(dim: Dim) -> str:
    return "COLS" if dim is Dim.N else "ROWS"


def _a_strategy_note(dataflow: DataflowPlan) -> str:
    if dataflow.a_strategy is AStrategy.STREAM:
        return f"re-read per tile (D31): {dataflow.tiles_per_a_event}x the staged total"
    if dataflow.a_strategy is AStrategy.WHOLE:
        return "ramped in before wave 0; same bytes as stage"
    return f"staged once per {dataflow.group_name} (D33)"


def _sub_cycles(machine: MachineModel) -> int:
    """Sub-cycles one operand takes on this unit at this dtype: 1 unless bit-serial.

    The same partition ``analysis/tiling.py`` makes (D34/D52): a multiplier below
    1 marks a combinational crossbar, whose K-side loss carries a sub-cycle row of
    fill on top of the padding. That is a *rate*, so an emitted program can name
    it and cannot measure it.
    """
    multiplier = machine.unit.dtype_multipliers.get(machine.dtype, 1.0)
    return round(1 / multiplier) if 0 < multiplier < 1 else 1


def _b_dataflow_note(dataflow: DataflowPlan, weight_sets: int) -> str:
    if weight_sets <= 1:
        return "moot: the array stores no weights (D30)"
    if dataflow.b_dataflow is BDataflow.ON_DEMAND:
        return "exposed at compute (D33) — timing only"
    if dataflow.b_dataflow is BDataflow.PERSISTENT:
        return "written once, never displaced (D33)"
    return "a wave early, behind compute (D33)"


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


def _loop_nest(grid: TileGrid, dataflow: DataflowPlan) -> list[str]:
    """The part a reader is meant to read: one tile, walked."""
    out = [
        _rule("the walk"),
        "COUNTERS = Counters()",
        "",
        "",
        "def tile_row(tile: int) -> int:",
        f'    """This tile\'s position along the grid\'s {grid.row_dim.value} axis."""',
        "    return tile % (GRID_ROWS * GRID_COLS) // GRID_COLS",
        "",
        "",
        "def tile_col(tile: int) -> int:",
        f'    """This tile\'s position along the grid\'s {grid.col_dim.value} axis."""',
        "    return tile % (GRID_ROWS * GRID_COLS) % GRID_COLS",
        "",
        "",
        "def partition_of(tile: int) -> int:",
        '    """Which split-K piece of the contraction this tile owns."""',
        "    return tile // (GRID_ROWS * GRID_COLS)",
        "",
        "",
        *_stage_function(grid, dataflow),
        "",
        "",
    ]
    # Dispatch on the SWEPT dimension, not on the resident operand: what decides
    # the shape of the nest is whether K is swept inside one tile — where it
    # accumulates locally and owes nobody a partial (os, and rs, whose extra
    # spreading of K happens inside one array) — or carried on the grid, where
    # the tile owns a slice of the contraction and must hand partials on (D53).
    if grid.swept_dim is Dim.K:
        out += _accumulator_nest(grid)
    elif grid.swept_dim is Dim.M:
        out += _weight_nest()
    else:
        out += _input_nest()
    if grid.materialises_partials:
        out += ["", "", *_reduction_kernel()]
    return out


def _stage_function(grid: TileGrid, dataflow: DataflowPlan) -> list[str]:
    """``stage_a``: the one place A crosses DRAM, in the grid's own vocabulary.

    Its own function rather than two lines inside ``run_tile`` because
    ``a_strategy=whole`` calls it as a prologue: whole and stage move the same
    bytes and differ only in *when* the events happen (D33), which is a
    difference this file can show by calling the same function earlier.
    """
    head = [
        "def stage_a(tile: int, dram: Dram, pad: Scratchpad) -> Tile:",
        f'    """The band of A this tile reads, staged on chip once per {dataflow.group_name}.',
        "",
        "    One staging event per (grid row, residency chunk), which is what makes the",
        "    three A strategies one mechanism: with A_RESIDENCY_TILES equal to the whole",
        "    row — stage, and whole — A crosses DRAM exactly once (D33); with it set to 1",
        "    — stream — every tile becomes its own event and re-fetches (D31).",
        "",
        "    Shared by every core, not per core: consecutive tiles of a row land on",
        "    DIFFERENT cores in the same wave, so 'A is staged once per row' is a claim",
        "    about one chip-wide buffer. Per-core staging would multiply A's traffic by",
        "    the core count.",
        '    """',
    ]
    if grid.row_dim is Dim.K:
        body = [
            "    kt = tile_row(tile)",
            "    k0, k1 = kt * ROWS, min(kt * ROWS + ROWS, K)",
            "    return pad.band(",
            "        (0, kt, tile_col(tile) // A_RESIDENCY_TILES),",
            "        lambda: dram.read_a(0, M, k0, k1),   # the whole M height of one k-slice",
            "    )",
        ]
    elif grid.col_dim is Dim.K:
        body = [
            "    mt = tile_row(tile)",
            "    m0, m1 = mt * ROWS, min(mt * ROWS + ROWS, M)",
            "    return pad.band(",
            "        (0, mt, tile_col(tile) // A_RESIDENCY_TILES),",
            "        lambda: dram.read_a(m0, m1, 0, K),   # one band of M rows, all of K",
            "    )",
        ]
    else:
        body = [
            "    mt, part = tile_row(tile), partition_of(tile)",
            "    m0, m1 = mt * ROWS, min(mt * ROWS + ROWS, M)",
            "    kt0, kt1 = K_TILE_BOUNDS[part]",
            "    k_lo, k_hi = kt0 * ROWS, min(kt1 * ROWS, K)",
            "    return pad.band(",
            "        (part, mt, tile_col(tile) // A_RESIDENCY_TILES),",
            "        lambda: dram.read_a(m0, m1, k_lo, k_hi),   # one band of M, this piece of K",
            "    )",
        ]
    return head + body


def _accumulator_nest(grid: TileGrid) -> list[str]:
    """``os``/``rs``: C stays in the accumulator and K is swept inside one tile."""
    out = ["def run_tile(tile: int, dram: Dram, pad: Scratchpad) -> None:"]
    if grid.stationarity is Dataflow.ROW_STATIONARY:
        out += [
            '    """One output tile. A row of A lives in each PE and K reduces across the',
            "    array's own columns, so no partial sum leaves this core either. The",
            "    difference from output-stationary is INSIDE one array, which a program at",
            "    this granularity cannot show and does not pretend to (Eyeriss, ISCA 2016).",
            '    """',
        ]
    else:
        out += [
            '    """One output tile. C stays in the accumulator; K is swept INSIDE it.',
            "",
            "    The whole contraction for this output block happens in one core's own",
            "    accumulator, which is exactly why output-stationary owes no reduction: no",
            "    partial sum ever leaves this function (D53).",
            '    """',
        ]
    out += [
        "    mt, nt, part = tile_row(tile), tile_col(tile), partition_of(tile)",
        "    m0, m1 = mt * ROWS, min(mt * ROWS + ROWS, M)",
        "    n0, n1 = nt * COLS, min(nt * COLS + COLS, N)",
        "    kt0, kt1 = K_TILE_BOUNDS[part]               # this piece's slice of the sweep",
        "    k_lo = kt0 * ROWS",
        "    band = stage_a(tile, dram, pad)",
        "",
        "    acc = zeros(m1 - m0, n1 - n0, ACC_DTYPE)     # the accumulator that stays put",
        "    for kt in range(kt0, kt1):                   # K is swept INSIDE this tile",
        "        k0, k1 = kt * ROWS, min(kt * ROWS + ROWS, K)",
        "        a = sub(band, 0, m1 - m0, k0 - k_lo, k1 - k_lo)   # already on chip",
        "        b = dram.read_b(k0, k1, n0, n1)                   # crosses DRAM",
        "        mma(acc, a, b, COUNTERS, SLOTS_PER_MMA)",
        "",
    ]
    if grid.materialises_partials:
        out += [
            "    # Kernel 1 of two. This piece owns K_TILE_BOUNDS[part] of the contraction and",
            "    # nothing else, so what it holds is a partial — and the GEMM ends before the",
            "    # reduction begins, so that partial cannot stay in a register (D53).",
            "    dram.write_partial(part, m0, n0, acc)",
        ]
    else:
        out += [
            "    dram.write_c(m0, n0, acc)                    # finished, not a partial",
        ]
    out += ["    COUNTERS.count_tile()"]
    return out


def _weight_nest() -> list[str]:
    """``ws``: B stays resident and M streams past it, producing partials over K."""
    return [
        "def run_tile(tile: int, dram: Dram, pad: Scratchpad, partials: Partials) -> None:",
        '    """One weight tile. B stays resident; M streams past it.',
        "",
        "    This tile owns one slice of the contraction — a k-slice — so what it produces",
        "    is a PARTIAL over K, not a finished result. The slices for one output block",
        "    meet in `partials`, in an accumulator on chip, on a later wave (D53). That",
        "    `partials` has to exist at all is the difference between this decomposition",
        "    and output-stationary, in one object.",
        '    """',
        "    kt, nt = tile_row(tile), tile_col(tile)",
        "    k0, k1 = kt * ROWS, min(kt * ROWS + ROWS, K)",
        "    n0, n1 = nt * COLS, min(nt * COLS + COLS, N)",
        "    b = dram.read_b(k0, k1, n0, n1)              # the operand that stays put",
        "    band = stage_a(tile, dram, pad)",
        "",
        "    for mt in range(M_TILES):                    # M streams past the resident tile",
        "        m0, m1 = mt * ROWS, min(mt * ROWS + ROWS, M)",
        "        a = sub(band, m0, m1, 0, k1 - k0)        # already on chip",
        "        product = zeros(m1 - m0, n1 - n0, ACC_DTYPE)",
        "        mma(product, a, b, COUNTERS, SLOTS_PER_MMA)",
        "        partials.accumulate(m0, n0, product)     # a PARTIAL over K; nothing is stored",
        "    COUNTERS.count_tile()",
    ]


def _input_nest() -> list[str]:
    """``is``: A stays resident and N streams past it, producing partials over K."""
    return [
        "def run_tile(tile: int, dram: Dram, pad: Scratchpad, partials: Partials) -> None:",
        '    """One input tile. A stays resident; N streams past it.',
        "",
        "    The grid's columns are slices of K here — K cut by the array's depth rather",
        "    than its width — so this tile owns a slice of the contraction exactly as a",
        "    weight-stationary one does, and produces PARTIALS over K (D53).",
        '    """',
        "    mt, kt = tile_row(tile), tile_col(tile)",
        "    m0, m1 = mt * ROWS, min(mt * ROWS + ROWS, M)",
        "    k0, k1 = kt * ROWS, min(kt * ROWS + ROWS, K)",
        "    band = stage_a(tile, dram, pad)",
        "    a = sub(band, 0, m1 - m0, k0, k1)            # the operand that stays put",
        "",
        "    for nt in range(N_TILES):                    # N streams past the resident tile",
        "        n0, n1 = nt * COLS, min(nt * COLS + COLS, N)",
        "        b = dram.read_b(k0, k1, n0, n1)          # crosses DRAM",
        "        product = zeros(m1 - m0, n1 - n0, ACC_DTYPE)",
        "        mma(product, a, b, COUNTERS, SLOTS_PER_MMA)",
        "        partials.accumulate(m0, n0, product)     # a PARTIAL over K; nothing is stored",
        "    COUNTERS.count_tile()",
    ]


def _reduction_kernel() -> list[str]:
    return [
        "def reduce_partials(dram: Dram) -> None:",
        '    """Kernel 2: sum the SPLIT_K partial results into C (D53).',
        "",
        "    A separate launch rather than more lines above: the GEMM has to finish",
        "    everywhere before any of this can start, which is precisely why the partials",
        "    could not stay in registers and crossed DRAM twice. Not new arithmetic —",
        "    2*M*N*K already counts these adds; they have merely left the array's own",
        "    accumulator for the vector unit, which is slower, and that is the trade",
        "    split-K makes for its occupancy.",
        '    """',
        "    for m0 in range(0, M, ROWS):",
        "        for n0 in range(0, N, COLS):",
        "            m1, n1 = min(m0 + ROWS, M), min(n0 + COLS, N)",
        "            acc = copy_of(dram.read_partial(0, m0, m1, n0, n1))",
        "            for part in range(1, SPLIT_K):",
        "                add_into(acc, dram.read_partial(part, m0, m1, n0, n1))",
        "            dram.write_c(m0, n0, acc)",
    ]


# -------------------------------------------------------------------------- the main


_CHECKS = """    checks = [
        Check("tiles", PREDICTED["tiles"], COUNTERS.tiles_run, True),
        Check("waves", PREDICTED["waves"], WAVES, True),
        Check("MACs", PREDICTED["macs"], COUNTERS.macs, True),
        Check("MAC slots issued", PREDICTED["mac_slots"], COUNTERS.mac_slots, True),
        Check(
            "idle core-waves",
            PREDICTED["idle_core_waves"],
            COUNTERS.idle_core_waves + (AVAILABLE_CORES - USED_CORES) * WAVES,
            True,
            note="a core with no tile this wave. This IS wave occupancy (D30).",
        ),
        Check("A staging events", PREDICTED["staging_events"], COUNTERS.staging_events, True),
        Check("A bytes", PREDICTED["a_dram_bytes"], COUNTERS.a_dram_bytes, True),
        Check("C bytes", PREDICTED["c_dram_bytes"], COUNTERS.c_dram_bytes, True),
        Check("partial bytes", PREDICTED["partial_dram_bytes"], COUNTERS.partial_dram_bytes, True),
        Check(
            "B bytes fetched",
            PREDICTED["b_dram_bytes_charged"],
            COUNTERS.b_dram_bytes,
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
            COUNTERS.b_compulsory_bytes,
            False,
            note="tier 2. What B costs if every byte of it crosses the bus exactly once.",
        ),
    ]
    failed = report_checks(checks)"""


def _main(
    chip: HardwareSpec, machine: MachineModel, grid: TileGrid, dataflow: DataflowPlan
) -> list[str]:
    needs_partials = grid.swept_dim is not Dim.K
    call = "run_tile(tile, dram, pad, partials)" if needs_partials else "run_tile(tile, dram, pad)"
    title = f"{chip.name} · {machine.dtype.value} · {_STATIONARITY_TITLE[grid.stationarity]}"
    out = [
        _rule("run it"),
        "def main() -> int:",
        f'    print("{title}")',
        '    print(f"backend: {BACKEND};  {USED_CORES} threads, one per modelled core in use")',
        "",
        "    a = operand(M, K, A_DTYPE, SEED)",
        "    b = operand(K, N, B_DTYPE, SEED + 1)",
        "    c = zeros(M, N, ACC_DTYPE)",
        "    dram = Dram(",
        "        a, b, c, COUNTERS,",
        "        a_bytes_per_element=A_BYTES_PER_ELEMENT,",
        "        b_bytes_per_element=B_BYTES_PER_ELEMENT,",
        "        c_bytes_per_element=C_BYTES_PER_ELEMENT,",
        "        acc_bytes_per_element=PARTIAL_BYTES_PER_ELEMENT,",
        "        partitions=SPLIT_K,",
        "        acc_dtype=ACC_DTYPE,",
        "    )",
        "    pad = Scratchpad(COUNTERS)",
    ]
    if needs_partials:
        out.append("    partials = Partials(M, N, ROWS, COLS, ACC_DTYPE)")
    if dataflow.a_strategy is AStrategy.WHOLE:
        out += [
            "",
            "    # a_strategy=whole: every band is ramped in before wave 0 instead of at the",
            "    # row boundary that needs it. The same function, the same bytes, the same",
            "    # event count — only the timing moves, which is D33's point and is also the",
            "    # one thing this program does not measure.",
            "    for tile in range(TILES):",
            "        stage_a(tile, dram, pad)",
        ]
    out += [
        "",
        f"    run_waves(USED_CORES, WAVES, TILES, COUNTERS, lambda tile: {call})",
    ]
    if needs_partials:
        out += [
            "",
            "    # The accumulator never crossed DRAM; C's own compulsory write does (D53).",
            "    partials.drain(dram, M, N)",
        ]
    if grid.materialises_partials:
        out += ["", "    reduce_partials(dram)"]
    out += [
        "",
        "    error = max_abs_diff(c, reference(a, b, INTEGER))",
        _CHECKS,
        "",
        "    occupancy = COUNTERS.occupancy(WAVES, USED_CORES, AVAILABLE_CORES)",
        "    padding = COUNTERS.padding_efficiency()",
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
        out += [
            '    print("                 The report\'s figure is the smaller of the two: on a")',
            '    print("                 bit-serial array it also charges a sub-cycle row of")',
            '    print("                 fill on K (D34), which is a RATE. Nothing here")',
            '    print("                 measures rates, so this program cannot see it.")',
        ]
    out += [
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
        "",
        "",
        'if __name__ == "__main__":',
        "    raise SystemExit(main())",
    ]
    return out


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


def _tidy(source: str) -> str:
    """Collapse the blank-line runs the import surgery leaves behind."""
    out: list[str] = []
    blanks = 0
    for line in source.splitlines():
        blanks = blanks + 1 if not line.strip() else 0
        if blanks <= 2:
            out.append(line.rstrip())
    return "\n".join(out).strip("\n") + "\n"
