"""How the model deploys a workload onto one chip, written as a loop nest.

``docs/CORRECTIONS.md`` D32. Sibling of :mod:`bwz.explain`, and the same
contract: **pure**, holds no counts of its own, and every constant it prints is
read back out of the schedule the figure draws. ``check()`` asserts that, so the
listing cannot drift from the timeline above it the way prose does.

The difference between the two is what they answer. ``explain`` answers *what
arithmetic is performed* — operand shapes, the algebra, the flop count — and is
the same on every chip. This answers *how that arithmetic reaches this
particular silicon*: how B is cut into array-sized tiles, how many arrays take a
wave of them at once, whether a tile has to be written into the array before it
can compute, and where the loads and stores sit around it.

Nothing here is vendor knowledge. Every branch is driven by a field the profile
declares — ``systolic_dims``, ``count``, ``weight_sets``, the dtype multiplier —
so a chip that declares an in-memory array gets a listing with weight sets in it
and a chip that does not gets one without, without this module knowing what
either is called.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from bwz.analysis.pipeline import Lane, PipelineTrace, grid_of, tile_count
from bwz.analysis.roofline import MachineModel
from bwz.analysis.stationarity import Dim, TileGrid
from bwz.graph.ops import MatmulAttrs, Operation
from bwz.report import PhaseResult
from bwz.spec.deployment import AStrategy, BDataflow
from bwz.spec.hardware_spec import Dataflow, HardwareSpec
from bwz.units import format_bytes, format_time


@dataclass(frozen=True, slots=True)
class Deployment:
    """One chip's listing, plus the numbers it was built from.

    The numbers are kept alongside the text so :func:`check` can compare them
    against the trace without re-parsing the listing.
    """

    chip_id: str
    title: str
    code: str
    tiles: int
    waves: int
    units: int
    resident_tiles: int
    reloads: int
    kind: str = "tiles"
    """``"tiles"`` for a tiled matmul, ``"operations"`` for a graph. The wave
    relation ``waves = ceil(tiles / units)`` only holds for the first: a network
    is a sequence of operations with no wave structure to check (D5a)."""
    stage_lines: tuple[tuple[str, tuple[int, ...]], ...] = ()
    """0-indexed line numbers within ``code`` for each animated stage
    (``"load_b"``, ``"load_a"``, ``"exec"``, ``"store"`` — the vocabulary
    ``plot_pipeline.py``'s ``_ANIMATION_STAGE`` already uses). A tuple of pairs
    rather than a ``dict``, matching ``PipelineTrace.work_by_op``'s own reason:
    keep a frozen dataclass hashable-in-substance. Empty for the network-sequence
    listing (:func:`_network_deployment`), which has no per-wave stages to point
    at. Lets a debug view highlight the line(s) live at the animation's current
    time — more than one at once when double buffering means more than one
    statement is truly concurrent (D41)."""
    array_rows: int = 0
    array_cols: int = 0
    grid: TileGrid | None = None
    """The decomposition a ``"tiles"`` listing was built from (D53): which
    operand is resident, how many tiles there are and along which dimensions,
    and which dimension each tile sweeps. ``None`` for an ``"operations"``
    network listing, which has no tile grid at all. Lets the ``--animate``
    geometry panel (D48) draw A/B/C without recomputing this module's own
    arithmetic — and lets it draw the *right* grid, which is the resident
    operand's and therefore not the same one on every chip."""


def _int(value: float) -> str:
    return f"{round(value):,}"


_STATIONARITY_HEADERS: dict[Dataflow, tuple[str, ...]] = {
    Dataflow.WEIGHT_STATIONARY: (
        "Weight-stationary: B is cut into {rows}x{cols} tiles and held by the array;",
        "{swept} streams past it. Each tile owns a slice of the contraction, so the",
        "partials meet in an accumulator on a later wave (D53).",
    ),
    Dataflow.OUTPUT_STATIONARY: (
        "Output-stationary — what cuBLAS/CUTLASS do: C is cut into {rows}x{cols}",
        "accumulator tiles and K is swept INSIDE each one, in registers. No partial",
        "sum ever leaves a core, so there is no reduction to pay for (D53).",
    ),
    Dataflow.INPUT_STATIONARY: (
        "Input-stationary: A is cut into {rows}x{cols} tiles and held by the array;",
        "{swept} streams past it. Each tile owns a slice of the contraction (D53).",
    ),
    Dataflow.ROW_STATIONARY: (
        "Row-stationary, after Eyeriss (Chen/Emer/Sze, ISCA 2016): one A row per PE,",
        "with K spread across the array's own columns so partial sums reduce INSIDE",
        "the array. No shipped profile declares this — these numbers are UNVALIDATED.",
    ),
}
"""How each stationarity cuts the problem, in the listing's own voice. A table
rather than a branch because every entry says the same three things — what is
resident, what streams, and where the partial sums go — and the reader should be
able to compare two chips' listings line for line (D53)."""

_MMA_WEIGHT_STATIONARY_HEADER = (
    "Weight-stationary: B is cut into {rows}x{cols} tiles; each instruction tile",
    "pairs one with {rows} rows of A. Nothing is held — this unit reads both",
    "operands per instruction (D30) — but K is still on the grid, so each tile",
    "owns a slice of the contraction (D53).",
)
"""Weight-stationary on a unit that holds no weights. The grid is the same, but
saying "held by the array" of an MMA unit would contradict D30 and D52 in the
same breath the cost model denies it — the reason D52's addendum exists."""


def _stationarity_header(grid: TileGrid, issues_instruction_tiles: bool) -> list[str]:
    """The header lines naming the decomposition this listing is of."""
    template = _STATIONARITY_HEADERS[grid.stationarity]
    if grid.stationarity is Dataflow.WEIGHT_STATIONARY and issues_instruction_tiles:
        template = _MMA_WEIGHT_STATIONARY_HEADER
    lines = [
        text.format(rows=grid.tile_rows, cols=grid.tile_cols, swept=grid.swept_dim.value)
        for text in template
    ]
    lines.append(
        f"Grid: {_int(grid.rows)} x {_int(grid.cols)} tiles "
        f"({grid.row_dim.value} x {grid.col_dim.value}), each sweeping {grid.swept_dim.value}"
        + (
            " in whole instruction tiles (D52)."
            if issues_instruction_tiles
            else " element by element."
        )
    )
    return lines


def _sweep_note(grid: TileGrid) -> str:
    """What the swept-dimension loop is doing, for its own trailing comment."""
    axis = grid.swept_dim.value
    if grid.stationarity is Dataflow.OUTPUT_STATIONARY:
        return f"{axis} accumulates in the tile's own accumulator — nothing spills (D53)"
    if grid.stationarity is Dataflow.ROW_STATIONARY:
        return f"{axis} is ALSO spread across the array's columns; this walks the rest"
    return f"{axis} is spatial: whole instruction tiles, no serial pipeline (D52)"


def _operand_indices(grid: TileGrid, *, instruction_tiles: bool) -> tuple[str, str]:
    """``A[..][..]``/``B[..][..]`` subscripts for one tile's inner loop.

    A dimension on the grid's row axis is ``GROUP(w, u)``, one on its column
    axis is ``COL(w, u)``, and the swept one is the loop variable — so which
    subscript varies is exactly the stationarity's choice, and the listing says
    the same thing the cost model does (D53). An MMA unit walks the swept
    dimension a whole instruction tile at a time (``kt * ROWS``); a streaming
    array walks it element by element (``m``).
    """

    def axis(dim: Dim) -> str:
        if dim is grid.row_dim:
            return "GROUP(w, u)"
        if dim is grid.col_dim:
            return "COL(w, u)"
        name = dim.value.lower()
        if not instruction_tiles:
            return name
        return f"{name}t * {'COLS' if dim is Dim.N else 'ROWS'}"

    return (
        f"[{axis(Dim.M)}][{axis(Dim.K)}]",
        f"[{axis(Dim.K)}][{axis(Dim.N)}]",
    )


def _reduction_kernel(
    grid: TileGrid, machine: MachineModel, dram_bytes: float
) -> list[tuple[str, str | None]]:
    """The second kernel split-K needs, as its own listing block (D53).

    A separate block rather than more lines in the loop nest above, because it is
    a separate *launch*: the GEMM has to finish everywhere before any of this can
    start. That is also why its bytes are real — the partials cannot stay in
    registers across a kernel boundary, so they go out to DRAM and come back.
    """
    adds = (grid.k_partitions - 1) * grid.m * grid.n
    return [
        ("/* ---- kernel 2: the batched reduction (split-K, D53) ---- ", None),
        (" * The GEMM above wrote SPLITS partial results per output element instead", None),
        (" * of one finished one. Summing them is a second launch, on the vector", None),
        (f" * unit rather than the array (D27): {machine.vector_unit.name}.", None),
        (f" * {format_bytes(dram_bytes)} crosses DRAM here — the partials out of", None),
        (f" * kernel 1 and back into kernel 2 — for {_int(adds)} adds. */", None),
        ("for (int i = 0; i < M * N; ++i) {", None),
        ("    acc_t acc = partial[0][i];", None),
        ("    for (int s = 1; s < SPLITS; ++s)", None),
        ("        acc += partial[s][i];   /* not new arithmetic: 2*M*N*K already", "reduce"),
        ("                                 * counts these adds — they have merely left", None),
        ("                                 * the array's accumulator (D53) */", None),
        ("    C[i] = acc;", None),
        ("}", None),
    ]


def deployment_of(
    chip: HardwareSpec,
    machine: MachineModel,
    phase: PhaseResult,
    trace: PipelineTrace,
    *,
    workload: str,
    operation: Operation | None = None,
    a_strategy: AStrategy = AStrategy.STAGE,
    b_dataflow: BDataflow = BDataflow.WRITE_AHEAD,
) -> Deployment:
    """Build the listing for *chip* running *workload*.

    *operation* is the single matmul when there is one; the tile count comes from
    :func:`analysis.pipeline.tile_count`, the same function the schedule and the
    utilisation model both divide by, so all three cannot disagree.

    *a_strategy* and *b_dataflow* are the **effective** choices — already
    clamped by :func:`analysis.dataflow.plan_dataflow` — so the loop nest this
    prints cannot claim a strategy the schedule above it does not actually run.
    Their defaults reproduce the listing this module always printed.
    """
    unit = machine.unit
    dims = unit.systolic_dims
    units = max(unit.count, 1)
    result = phase.ops[0] if phase.ops else None

    if trace.kind != "tiles" or dims is None or result is None or operation is None:
        return _network_deployment(chip, machine, phase, trace, workload=workload)
    if not isinstance(operation.attrs, MatmulAttrs):
        return _network_deployment(chip, machine, phase, trace, workload=workload)

    rows, cols = dims
    grid = grid_of(operation, machine)
    assert grid is not None, "systolic_dims present and attrs is a matmul"
    waves = trace.tiles
    tiles = tile_count(operation, machine)
    resident = unit.resident_tile_capacity()
    reloads = max(0, tiles - resident)
    multiplier = unit.dtype_multipliers.get(machine.dtype, 1.0)
    sub_cycles = round(1 / multiplier) if 0 < multiplier < 1 else 1
    # An MMA unit — tensor/matrix core: full-rate (not bit-serial) and holding
    # no resident weights, so it issues fixed instruction tiles rather than
    # streaming the swept dimension past a stationary operand. Same partition
    # the cost model makes in analysis/tiling.py (D52), plus the
    # weight-residency test, so a full-rate array that DOES hold weights still
    # reads as a streaming one.
    issues_instruction_tiles = sub_cycles == 1 and unit.weight_sets <= 1
    # What each tile sweeps, and how far: M under weight-stationary, K under
    # output-stationary, N under input-stationary (D53). N is the dimension the
    # array's columns cut; M and K are both cut by its rows.
    swept = grid.swept_dim
    swept_extent = grid.extent(swept)
    swept_tile = cols if swept is Dim.N else rows
    swept_tiles = math.ceil(swept_extent / swept_tile)
    depth = 2 if trace.double_buffered else 1
    # One representative tile's share of the traffic the report charged. This
    # must divide by the real tile count, not waves * units (the array's
    # theoretical capacity): when tiles doesn't divide evenly into units, the
    # last wave leaves slots idle, and waves * units overcounts how many tiles
    # actually shared the total, understating each one's share (D46).
    per = max(tiles, 1)
    b_bytes = format_bytes(result.dram_weight_read_bytes / per)
    c_bytes = format_bytes(result.dram_write_bytes / per)
    # The staging structure behind the listing: tiles are counted row-major,
    # each grid row holds `grid.cols` of them, and A is staged once per row — a
    # k-slice under ws, a band of M rows under os (D53).
    group = grid.group_name
    a_events = grid.a_events
    band_bytes = result.dram_activation_read_bytes / a_events
    a_stage = format_bytes(band_bytes)

    header = [
        f"/* {chip.name} — how this model deploys the run",
        f" * {workload}",
        " *",
        *(f" * {line}" for line in _stationarity_header(grid, issues_instruction_tiles)),
        f" * {_int(tiles)} tiles over {units} array{'s' if units != 1 else ''}"
        f" -> {_int(waves)} wave{'s' if waves != 1 else ''}.",
    ]
    if a_strategy is AStrategy.STREAM:
        a_lines = [
            f" * A is re-read per tile ({grid.cols}x the staged total, D31) — the honest",
            " * picture for a machine whose GEMMs genuinely refetch operands.",
        ]
    elif a_strategy is AStrategy.WHOLE:
        a_lines = [
            f" * A is staged whole ({format_bytes(result.dram_activation_read_bytes)} of",
            f" * scratchpad), every {group} before wave 0 — the same total as staging per",
            f" * {group} (D33), only the timing changes.",
        ]
    else:
        a_lines = [
            f" * A is staged once per {group} ({_int(band_bytes)} B of scratchpad); every",
            f" * tile of that {group} reads the staging. A crosses DRAM exactly once",
            " * (D33).",
        ]

    if unit.weight_sets > 1:
        header.append(
            f" * {unit.weight_sets} weight sets per array, {resident} chip-wide: a tile must be"
        )
        header.append(" * written INTO the array before it can compute.")
        if b_dataflow is BDataflow.ON_DEMAND:
            header.append(" * On-demand: the write lands at compute, exposed on the critical path")
            header.append(" * rather than hidden behind an earlier wave's arithmetic (D33).")
        elif b_dataflow is BDataflow.PERSISTENT:
            header.append(f" * Persistent: all {_int(tiles)} tiles fit the {resident} resident")
            header.append(" * sets, so each is written once, in wave 0, and never displaced —")
            header.append(" * no later wave writes the array again.")
        else:
            header.append(" * Sets rotate: the next wave's tile lands in the set that finished")
            header.append(" * computing three waves ago, so each write hides behind this wave's")
            header.append(" * arithmetic — the independent-address overlap weight sets exist for")
            header.append(" * (D33). The first wave's sets fill during the first A staging.")
        header.append(" *")
        header.extend(a_lines)
        header.append(" *")
        # Within ONE pass every tile is written once whether or not it fits: M is
        # the innermost loop, so a tile is used once and never revisited. What
        # capacity decides is the cost of the NEXT invocation on the same B.
        if reloads:
            header.append(
                f" * B is {_int(tiles)} tiles and only {resident} fit, so {_int(reloads)} of them"
            )
            header.append(" * displace an earlier one. Each is still written once in this pass —")
            header.append(" * what capacity costs is the NEXT run on the same B: all of it again.")
        else:
            header.append(f" * All {_int(tiles)} tiles fit, so B is written once and a second run")
            header.append(" * on the same weights writes nothing at all.")
    else:
        header.append(" * The array stores no weights: both operands are re-read per instruction,")
        header.append(" * so there is no residency limit on the array itself.")
        header.extend(a_lines)
    header.append(" */")

    row_axis, col_axis = grid.row_dim.value, grid.col_dim.value
    row_tile = "COLS" if grid.row_dim is Dim.N else "ROWS"
    col_tile = "COLS" if grid.col_dim is Dim.N else "ROWS"
    defines = [
        f"#define UNITS        {units:<10} /* {unit.name} */",
        f"#define ROWS         {rows:<10} /* array rows -> M or K per tile */",
        f"#define COLS         {cols:<10} /* array cols -> N per tile */",
        f"#define GRID_ROWS    {grid.rows:<10} /* ceil({row_axis} / {row_tile}) */",
        f"#define GRID_COLS    {grid.cols:<10} /* ceil({col_axis} / {col_tile}) */",
        f"#define TILES        {tiles:<10} /* GRID_ROWS * GRID_COLS"
        + (f" * SPLITS ({grid.k_partitions} split-K pieces)" if grid.k_partitions > 1 else "")
        + " */",
        f"#define WAVES        {waves:<10} /* ceil(TILES / UNITS) */",
        f"#define DEPTH        {depth:<10} /* "
        + (
            "double buffered: capacity fits two tiles"
            if depth == 2
            else "no room for a second tile"
        )
        + " */",
        # Both macros used to be spelled in k-slice vocabulary (NTILES_PER_KS,
        # KSLICE), which named a decomposition only weight-stationary runs. The
        # arithmetic is unchanged; what a "group" is now follows the grid (D53).
        f"#define TILES_PER_GROUP {grid.cols:<6} /* tiles per grid row: GRID_COLS */",
        f"#define GROUP(w, u)   (((w) * UNITS + (u)) / TILES_PER_GROUP % GRID_ROWS * {row_tile})"
        f"  /* this tile's {row_axis} */",
        f"#define COL(w, u)     (((w) * UNITS + (u)) % TILES_PER_GROUP * {col_tile})"
        f"  /* this tile's {col_axis} */",
    ]
    if unit.weight_sets > 1:
        defines.insert(
            1, f"#define WEIGHT_SETS  {unit.weight_sets:<10} /* resident tiles per array */"
        )
    if sub_cycles > 1:
        defines.append(
            f"#define SUB_CYCLES   {sub_cycles:<10} /* {machine.dtype.value} multiplier"
            f" {multiplier:g} -> {sub_cycles} cycles per operand */"
        )
    if issues_instruction_tiles:
        defines.append(
            f"#define {swept.value}TILES       {swept_tiles:<10} /* ceil({swept.value} /"
            f" {'COLS' if swept is Dim.N else 'ROWS'}): the swept dimension is padded to whole"
            f" instruction tiles (D52) */"
        )
    if grid.k_partitions > 1:
        defines.append(
            f"#define SPLITS       {grid.k_partitions:<10} /* split-K: independent pieces of the"
            f" contraction, summed by the second kernel below (D53) */"
        )

    # An in-memory array must be *written* before it can compute, but the sets
    # are independently addressed, so a write-ahead lands a full wave early and
    # hides behind that wave's arithmetic (D33); an array that reads its
    # operands per instruction has nothing to write, and saying otherwise would
    # invent a residency it does not have (D30). on-demand and persistent are
    # the other two placements the same fact admits: exposed at compute, or
    # paid once and never again.
    def b_load_line(wave_expr: str, indent: str) -> tuple[str, str]:
        return (
            f"{indent}load_B(u, tile({wave_expr}, u));  "
            f"/* {b_bytes} — solid bar; each tile fetched once */",
            "load_b",
        )

    def a_staging_lines(wave_expr: str, indent: str) -> list[tuple[str, str]]:
        # WHOLE has no per-wave staging line at all — it is a prologue-only ramp
        # (a_prologue_line below), same total bytes as staging per group (D33).
        if a_strategy is AStrategy.WHOLE:
            return []
        if a_strategy is AStrategy.STREAM:
            return [
                (
                    f"{indent}load_A(u, tile({wave_expr}, u));"
                    f" /* {a_stage} — hatched bar; re-read every tile (D31) */",
                    "load_a",
                )
            ]
        return [
            (
                f"{indent}if (tile({wave_expr}, u) % TILES_PER_GROUP == 0)  "
                f"/* this tile opens a {group} */",
                "load_a",
            ),
            (
                f"{indent}    stage_A(u, GROUP({wave_expr}, u)); "
                f"/* {a_stage} — hatched bar; staged once (D33) */",
                "load_a",
            ),
        ]

    a_prologue_line: tuple[str, str] | None = (
        (
            f"stage_all_of_A();  /* {format_bytes(result.dram_activation_read_bytes)} — hatched "
            f"bar; every {group} ramped in before wave 0, same total as staging per {group} (D33) "
            f"*/",
            "load_a",
        )
        if a_strategy is AStrategy.WHOLE
        else None
    )

    # How one tile reads its operands, in the vocabulary of the grid it belongs
    # to: `GROUP(w, u)` is the tile's position along the grid's row axis and
    # `COL(w, u)` along its column axis, so the swept dimension is the only one
    # the inner loop varies (D53).
    a_index, b_index = _operand_indices(grid, instruction_tiles=issues_instruction_tiles)
    inner: list[str] = (
        [
            "            for (int c = 0; c < SUB_CYCLES; ++c)",
            f"                feed(u, w % WEIGHT_SETS, &A{a_index});"
            "   /* one sub-cycle of the operand */",
        ]
        if sub_cycles > 1
        else (
            [
                f"            mma(u, &A{a_index}, &B{b_index});"
                f"   /* one {rows}x{cols}x{rows} instruction tile */"
            ]
            if issues_instruction_tiles
            else [f"            mac(u, &A{a_index});"]
        )
    )
    if sub_cycles > 1 and unit.weight_sets == 1:
        inner = [line.replace("w % WEIGHT_SETS, ", "") for line in inner]
    # The exec line is always the last physical line of `inner` — a sub-cycle
    # loop's header carries no arithmetic of its own to tag.
    inner_lines: list[tuple[str, str | None]] = [(line, None) for line in inner[:-1]]
    inner_lines.append((inner[-1], "exec"))

    on_demand_line: tuple[str, str | None] | None = (
        (
            "        imc_write(u, w % WEIGHT_SETS, tile(w, u));"
            "  /* on-demand (D33): exposed on the critical path */",
            None,
        )
        if unit.weight_sets > 1 and b_dataflow is BDataflow.ON_DEMAND
        else None
    )

    def compute_lines() -> list[tuple[str, str | None]]:
        # An MMA unit does not stream the swept dimension past a resident tile —
        # it issues fixed instruction tiles, so that dimension is spatial and
        # quantises into whole tiles of it (D52). A resident-weight array (Metis)
        # genuinely does stream, and keeps the element loop. Which dimension is
        # swept is the stationarity's doing: M under ws, K under os (D53).
        axis = swept.value
        sweep_line = (
            (
                f"        for (int {axis.lower()}t = 0; {axis.lower()}t < {axis}TILES;"
                f" ++{axis.lower()}t)"
                f"   /* {_sweep_note(grid)} */",
                None,
            )
            if issues_instruction_tiles
            else (
                f"        for (int {axis.lower()} = 0; {axis.lower()} < {swept_extent};"
                f" ++{axis.lower()})"
                f"   /* {axis} streams past the resident {grid.resident.value} tile */",
                None,
            )
        )
        if on_demand_line is None:
            return [
                ("    parallel_for (int u = 0; u < UNITS; ++u)", None),
                sweep_line,
                *inner_lines,
            ]
        return [
            ("    parallel_for (int u = 0; u < UNITS; ++u) {", None),
            on_demand_line,
            sweep_line,
            *inner_lines,
            ("    }", None),
        ]

    store_line: tuple[str, str] = (
        f"        store_C(u, tile(w, u));       /* {c_bytes} — hollow bar */",
        "store",
    )

    body: list[tuple[str, str | None]] = []
    if a_prologue_line is not None:
        body.append(a_prologue_line)
    if depth == 2:
        # Double buffered: a real prologue primes wave 0, and every steady-state
        # iteration prefetches wave w+1 alongside wave w's own compute — the
        # schedule this decomposes (`_pipelined_tiles`) genuinely overlaps them
        # (`load_start(i) = max(dram_free, store_ends[i-2])`, never gated on the
        # previous wave's compute), so the listing reads that way too (D41).
        body.append(("/* prologue: prime the pipeline — wave 0's tile loads before the", None))
        body.append((" * loop starts, so wave 0's compute below already has an operand", None))
        body.append((" * when the loop's first prefetch (wave 1) begins beside it,", None))
        body.append((" * DEPTH=2 apart. */", None))
        body.append(("for (int u = 0; u < UNITS; ++u) {", None))
        body.append(b_load_line("0", "    "))
        body.extend(a_staging_lines("0", "    "))
        if unit.weight_sets > 1:
            if b_dataflow is BDataflow.WRITE_AHEAD:
                body.append(
                    (
                        "    imc_write(u, 0, tile(0, u));  "
                        "/* write-ahead (D33): priming set 0 — nothing to hide behind yet */",
                        None,
                    )
                )
            elif b_dataflow is BDataflow.PERSISTENT:
                body.append(
                    (
                        "    imc_write(u, u, tile(0, u));  "
                        "/* persistent (D33): loaded once, never displaced */",
                        None,
                    )
                )
        body.append(("}", None))
        body.append(("", None))
        body.append(("for (int w = 0; w < WAVES; ++w) {", None))
        body.append(("", None))
        body.append(
            (
                "    /* ---- prefetch wave w+1 while wave w computes below "
                "(double buffered) ---- */",
                None,
            )
        )
        body.append(("    if (w + 1 < WAVES) {", None))
        body.append(("        for (int u = 0; u < UNITS; ++u) {", None))
        body.append(b_load_line("w + 1", "            "))
        body.extend(a_staging_lines("w + 1", "            "))
        if unit.weight_sets > 1 and b_dataflow is BDataflow.WRITE_AHEAD:
            body.append(
                (
                    "            imc_write(u, (w + 1) % WEIGHT_SETS, tile(w + 1, u));"
                    "  /* hidden behind this wave's arithmetic */",
                    None,
                )
            )
        body.append(("        }", None))
        body.append(("    }", None))
        body.append(("", None))
        body.append(("    /* ---- the arrays, all UNITS of them at once ---- */", None))
        body.extend(compute_lines())
        body.append(("", None))
        body.append(
            (
                "    /* ---- DRAM again: wave w's result drains, behind the "
                "prefetch issued above ---- */",
                None,
            )
        )
        body.append(("    for (int u = 0; u < UNITS; ++u)", None))
        body.append(store_line)
        body.append(("}", None))
    else:
        # Not double buffered: load, compute and store for wave w genuinely
        # serialise before wave w+1 can begin — `_pipelined_tiles` places wave
        # w's store before it ever computes wave w+1's load_start — so nothing
        # here should read as overlapping.
        body.append(("for (int w = 0; w < WAVES; ++w) {", None))
        body.append(("", None))
        body.append(("    /* ---- DRAM, one port, in issue order (D22) ---- */", None))
        body.append(("    for (int u = 0; u < UNITS; ++u) {", None))
        body.append(b_load_line("w", "        "))
        body.extend(a_staging_lines("w", "        "))
        if unit.weight_sets > 1 and b_dataflow is BDataflow.WRITE_AHEAD:
            body.append(
                (
                    "        if (w + 1 < WAVES)  "
                    "/* write-ahead (D33): the next wave's tile lands */",
                    None,
                )
            )
            body.append(
                (
                    "            imc_write(u, (w + 1) % WEIGHT_SETS, tile(w + 1, u));"
                    "   /* in the set freed 3 waves ago — hidden behind this wave */",
                    None,
                )
            )
        elif unit.weight_sets > 1 and b_dataflow is BDataflow.PERSISTENT:
            body.append(
                (
                    "        if (w == 0)  /* persistent (D33): loaded once, never displaced */",
                    None,
                )
            )
            body.append(("            imc_write(u, u, tile(0, u));", None))
        body.append(("    }", None))
        body.append(("", None))
        body.append(("    /* ---- the arrays, all UNITS of them at once ---- */", None))
        body.extend(compute_lines())
        body.append(("", None))
        body.append(
            ("    /* ---- DRAM again: the result drains behind the next fetch ---- */", None)
        )
        body.append(("    for (int u = 0; u < UNITS; ++u)", None))
        body.append(store_line)
        body.append(("}", None))

    # Both numbers, always: the REPORTED latency is what the engine predicts and
    # what every table quotes, and the drawn span is that plus the fill/drain the
    # roofline's max() omits. Printing only the span invites it to be read as the
    # prediction, and on Metis at 8192-cubed the two differ by 18% (D35).
    terms = (
        f"max(t_dram {format_time(result.t_dram_s)}, t_compute {format_time(result.t_compute_s)})"
        if trace.double_buffered
        else f"t_dram {format_time(result.t_dram_s)} + t_compute {format_time(result.t_compute_s)}"
    )
    footer = f"\n/* reported latency {format_time(trace.reported_latency_s)} = {terms}"
    if trace.fill_drain_s > 0:
        footer += (
            f";\n * drawn span {format_time(trace.total_s)} adds "
            f"{format_time(trace.fill_drain_s)} of pipeline fill/drain, which the\n"
            f" * roofline's max() leaves out of the prediction (D19)"
        )
    footer += " */"

    combined: list[tuple[str, str | None]] = [(h, None) for h in header]
    combined.append(("", None))
    combined.extend((d, None) for d in defines)
    combined.append(("", None))
    combined.extend(body)
    if grid.materialises_partials:
        combined.append(("", None))
        combined.extend(
            (line, tag)
            for line, tag in _reduction_kernel(grid, machine, result.dram_reduction_bytes)
        )
    combined.extend((f, None) for f in footer.split("\n"))

    code = "\n".join(text for text, _tag in combined)
    stage_line_map: dict[str, list[int]] = {}
    for index, (_text, tag) in enumerate(combined):
        if tag is not None:
            stage_line_map.setdefault(tag, []).append(index)

    return Deployment(
        chip_id=chip.id,
        title=f"{chip.name} — {unit.name}",
        code=code,
        tiles=tiles,
        waves=waves,
        units=units,
        resident_tiles=resident,
        reloads=reloads,
        stage_lines=tuple((tag, tuple(indices)) for tag, indices in stage_line_map.items()),
        array_rows=rows,
        array_cols=cols,
        grid=grid,
    )


def _tiles_from(result: object, rows: int, cols: int) -> int:
    """Upper bound on the tile count; the schedule's wave count is authoritative."""
    return 1 << 62


def _network_deployment(
    chip: HardwareSpec,
    machine: MachineModel,
    phase: PhaseResult,
    trace: PipelineTrace,
    *,
    workload: str,
) -> Deployment:
    """A graph of operations rather than one tiled matmul.

    The model runs a network as a sequence with no overlap between operations
    (D5a), so the listing is a loop over operations rather than a tile nest, and
    it says so.
    """
    unit = machine.unit
    steps = max(trace.steps, 1)
    header_lines = [
        f"/* {chip.name} — how this model deploys the run",
        f" * {workload}",
        " *",
        " * A network is a SEQUENCE here: no overlap is modelled between one operation's",
        " * prefetch and the previous operation's arithmetic (D5a). Within an operation,",
        " * load and compute overlap when capacity granted a second buffer.",
        " *",
        f" * {trace.tiles} operations, drawn as {steps} step{'s' if steps != 1 else ''}.",
        " */",
    ]
    defines = [
        f"#define OPS   {trace.tiles:<10} /* graph nodes in this phase */",
        f"#define UNITS {unit.count:<10} /* {unit.name} */",
    ]
    # A generic, un-unrolled loop: real operation names (Span.label, e.g.
    # "q_proj") live in the trace and the animation's blocks/hover text, not
    # here — one static "load_B(op[i])" line stands for every operation's load,
    # so tagging it "load_b" highlights it whenever *any* op is loading, not a
    # specific one. The two compute branches get distinct tags (exec_core,
    # exec_vector, D43) so the debug view can glow the matrix and vector
    # stations independently — which one runs depends on op[i], which this
    # text never names, but *which engine* is a real, known distinction the
    # trace already carries per span (D42's original single "exec" tag
    # couldn't make that distinction; splitting it is what let per-engine
    # animation stations exist at all).
    imc_write_line: tuple[str, str | None] = (
        "    imc_write(op[i]);                /* the array's weight set is written before "
        "this op's arithmetic; no per-op write-ahead/on-demand/persistent placement is "
        "modelled for a network trace (D43) */",
        None,
    )
    body: list[tuple[str, str | None]] = [
        ("for (int i = 0; i < OPS; ++i) {", None),
        ("    dispatch(op[i]);                  /* hatched bar, if the op costs one */", None),
        ("", None),
        ("    load_B(op[i]);                    /* weights   — solid bar   */", "load_b"),
        *([imc_write_line] if unit.weight_sets > 1 else []),
        ("    load_A(op[i]);                    /* activations — hatched bar */", "load_a"),
        ("", None),
        ("    if (is_matrix(op[i]))", None),
        ("        parallel_for (int u = 0; u < UNITS; ++u)", None),
        (f"            {unit.name}(u, op[i]);", "exec_core"),
        ("    else", None),
        (
            f"        {machine.vector_unit.name}(op[i]);   /* norms, activations (D27) */",
            "exec_vector",
        ),
        ("", None),
        (
            "    store_C(op[i]);                   /* whatever no later op reads — hollow bar */",
            "store",
        ),
        ("}", None),
    ]
    footer_line = (
        f"/* span {format_time(trace.total_s)}; DRAM {format_bytes(trace.totals[Lane.DRAM])}, "
        f"arithmetic {_int(trace.totals[Lane.CORE] + trace.totals[Lane.VECTOR])} OP */"
    )

    combined: list[tuple[str, str | None]] = [(h, None) for h in header_lines]
    combined.append(("", None))
    combined.extend((d, None) for d in defines)
    combined.append(("", None))
    combined.extend(body)
    combined.append(("", None))
    combined.append((footer_line, None))

    code = "\n".join(text for text, _tag in combined)
    stage_line_map: dict[str, list[int]] = {}
    for index, (_text, tag) in enumerate(combined):
        if tag is not None:
            stage_line_map.setdefault(tag, []).append(index)

    return Deployment(
        chip_id=chip.id,
        title=f"{chip.name} — {unit.name}",
        code=code,
        tiles=trace.tiles,
        waves=trace.tiles,
        units=max(unit.count, 1),
        resident_tiles=unit.resident_tile_capacity(),
        reloads=0,
        kind="operations",
        stage_lines=tuple((tag, tuple(indices)) for tag, indices in stage_line_map.items()),
    )


def check(deployment: Deployment, trace: PipelineTrace) -> None:
    """Assert the listing's constants are the schedule's.

    The same guarantee :mod:`bwz.explain` gives the arithmetic section: a
    deployment listing that disagreed with the timeline above it would be worse
    than no listing, because a reader would believe it.
    """
    if deployment.waves != trace.tiles:
        raise ValueError(
            f"{deployment.chip_id}: listing says {deployment.waves} waves, schedule has "
            f"{trace.tiles}"
        )
    if deployment.tiles < deployment.waves:
        raise ValueError(
            f"{deployment.chip_id}: {deployment.tiles} tiles cannot fill {deployment.waves} waves"
        )
    if deployment.kind != "tiles":
        return
    if deployment.units > 0 and deployment.waves != math.ceil(deployment.tiles / deployment.units):
        raise ValueError(
            f"{deployment.chip_id}: {deployment.tiles} tiles over {deployment.units} units is "
            f"{math.ceil(deployment.tiles / deployment.units)} waves, listing says "
            f"{deployment.waves}"
        )
