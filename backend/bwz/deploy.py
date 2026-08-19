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

from bwz.analysis.pipeline import Lane, PipelineTrace, tile_count
from bwz.analysis.roofline import MachineModel
from bwz.graph.ops import MatmulAttrs, Operation
from bwz.report import PhaseResult
from bwz.spec.deployment import AStrategy, BDataflow
from bwz.spec.hardware_spec import HardwareSpec
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


def _int(value: float) -> str:
    return f"{round(value):,}"


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
    waves = trace.tiles
    tiles = tile_count(operation, machine)
    stream_rows = operation.attrs.m
    resident = unit.resident_tile_capacity()
    reloads = max(0, tiles - resident)
    multiplier = unit.dtype_multipliers.get(machine.dtype, 1.0)
    sub_cycles = round(1 / multiplier) if 0 < multiplier < 1 else 1
    depth = 2 if trace.double_buffered else 1
    # Per-wave, per-array shares of the traffic the report charged.
    per = max(waves, 1) * units
    b_bytes = format_bytes(result.dram_weight_read_bytes / per)
    c_bytes = format_bytes(result.dram_write_bytes / per)
    # The k-slice structure behind the listing: tiles are counted k-major, each
    # k-slice holds ceil(N/COLS) of them, and A is staged once per k-slice.
    n_tiles_per_ks = math.ceil(operation.attrs.n / cols)
    k_slices = math.ceil(operation.attrs.k / rows)
    k_slice_bytes = result.dram_activation_read_bytes / k_slices
    a_stage = format_bytes(k_slice_bytes)

    header = [
        f"/* {chip.name} — how this model deploys the run",
        f" * {workload}",
        " *",
        f" * B is cut into {rows}x{cols} tiles and held by the array; M streams past it.",
        f" * {_int(tiles)} tiles over {units} array{'s' if units != 1 else ''}"
        f" -> {_int(waves)} wave{'s' if waves != 1 else ''}.",
    ]
    if a_strategy is AStrategy.STREAM:
        a_lines = [
            f" * A is re-read per tile ({n_tiles_per_ks}x the staged total, D31) — the honest",
            " * picture for a machine whose GEMMs genuinely refetch operands.",
        ]
    elif a_strategy is AStrategy.WHOLE:
        a_lines = [
            f" * A is staged whole ({format_bytes(result.dram_activation_read_bytes)} of",
            " * scratchpad), every k-slice before wave 0 — the same total as staging per",
            " * k-slice (D33), only the timing changes.",
        ]
    else:
        a_lines = [
            f" * A is staged once per k-slice ({_int(k_slice_bytes)} B of scratchpad); every",
            " * tile of that k-slice reads the staging. A crosses DRAM exactly once",
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

    defines = [
        f"#define UNITS        {units:<10} /* {unit.name} */",
        f"#define ROWS         {rows:<10} /* array rows -> K per tile */",
        f"#define COLS         {cols:<10} /* array cols -> N per tile */",
        f"#define TILES        {tiles:<10} /* ceil(K/ROWS) * ceil(N/COLS) */",
        f"#define WAVES        {waves:<10} /* ceil(TILES / UNITS) */",
        f"#define DEPTH        {depth:<10} /* "
        + (
            "double buffered: capacity fits two tiles"
            if depth == 2
            else "no room for a second tile"
        )
        + " */",
        f"#define NTILES_PER_KS {n_tiles_per_ks:<7} /* n-tiles per k-slice: ceil(N / COLS) */",
        "#define KSLICE(w, u)  (((w) * UNITS + (u)) / NTILES_PER_KS * ROWS) /* this tile's k */",
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

    # An in-memory array must be *written* before it can compute, but the sets
    # are independently addressed, so a write-ahead lands a full wave early and
    # hides behind that wave's arithmetic (D33); an array that reads its
    # operands per instruction has nothing to write, and saying otherwise would
    # invent a residency it does not have (D30). on-demand and persistent are
    # the other two placements the same fact admits: exposed at compute, or
    # paid once and never again.
    write_ahead = (
        "        if (w + 1 < WAVES) /* write-ahead (D33): the next wave's tile lands */\n"
        "            imc_write(u, (w + 1) % WEIGHT_SETS, tile(w + 1, u));"
        "   /* in the set freed 3 waves ago — hidden behind this wave */\n"
        if unit.weight_sets > 1 and b_dataflow is BDataflow.WRITE_AHEAD
        else ""
    )
    persistent_write = (
        "        if (w == 0)  /* persistent (D33): loaded once, never displaced */\n"
        "            imc_write(u, u, tile(0, u));\n"
        if unit.weight_sets > 1 and b_dataflow is BDataflow.PERSISTENT
        else ""
    )
    on_demand_write = (
        "        imc_write(u, w % WEIGHT_SETS, tile(w, u));"
        "  /* on-demand (D33): exposed on the critical path */\n"
        if unit.weight_sets > 1 and b_dataflow is BDataflow.ON_DEMAND
        else ""
    )
    prologue = (
        "/* prologue: wave 0's B tiles land in sets 0..UNITS-1 while the first A\n"
        " * k-slice stages; afterwards the write-ahead keeps every set a full wave\n"
        " * ahead of its compute. */\n"
        if unit.weight_sets > 1 and b_dataflow is BDataflow.WRITE_AHEAD
        else "/* prologue: wave 0's B tiles land in sets 0..UNITS-1 and are never\n"
        " * displaced — TILES <= UNITS * WEIGHT_SETS holds, or this fell back to\n"
        " * write-ahead (checked upstream). */\n"
        if unit.weight_sets > 1 and b_dataflow is BDataflow.PERSISTENT
        else ""
    )
    a_prologue = (
        f"stage_all_of_A();  /* {format_bytes(result.dram_activation_read_bytes)} — hatched "
        f"bar; every k-slice ramped in before wave 0, same total as staging per k-slice (D33) "
        f"*/\n"
        if a_strategy is AStrategy.WHOLE
        else ""
    )
    if a_strategy is AStrategy.STREAM:
        a_dram_line = (
            "        load_A(u, tile(w, u));"
            f" /* {a_stage} — hatched bar; re-read every tile (D31) */\n"
        )
    elif a_strategy is AStrategy.WHOLE:
        a_dram_line = ""  # staged once, up front, in the prologue above.
    else:
        a_dram_line = (
            "        if (tile(w, u) % NTILES_PER_KS == 0)  /* this tile opens a k-slice */\n"
            f"            stage_A(u, KSLICE(w, u)); /* {a_stage} — hatched bar; staged once "
            f"(D33) */\n"
        )
    inner = (
        "            for (int c = 0; c < SUB_CYCLES; ++c)\n"
        "                feed(u, w % WEIGHT_SETS, &A[m][KSLICE(w, u)]);"
        "   /* one sub-cycle of the operand */\n"
        if sub_cycles > 1
        else "            mac(u, &A[m][KSLICE(w, u)]);\n"
    )
    if sub_cycles > 1 and unit.weight_sets == 1:
        inner = inner.replace("w % WEIGHT_SETS, ", "")

    if on_demand_write:
        compute_block = f"""    parallel_for (int u = 0; u < UNITS; ++u) {{
{on_demand_write}        for (int m = 0; m < {stream_rows}; ++m)   /* M streams; it never tiles */
{inner}    }}
"""
    else:
        compute_block = f"""    parallel_for (int u = 0; u < UNITS; ++u)
        for (int m = 0; m < {stream_rows}; ++m)   /* M streams; it never tiles */
{inner}"""

    body = f"""
{a_prologue}{prologue}for (int w = 0; w < WAVES; ++w) {{

    /* ---- DRAM, one port, in issue order (D22) ------------------------- */
    for (int u = 0; u < UNITS; ++u) {{
        load_B(u, tile(w, u));        /* {b_bytes} — solid bar; each tile fetched once */
{a_dram_line}{write_ahead}{persistent_write}    }}

    /* ---- the arrays, all UNITS of them at once ------------------------ */
{compute_block}    /* ---- DRAM again: the result drains behind the next fetch ---------- */
    for (int u = 0; u < UNITS; ++u)
        store_C(u, tile(w, u));       /* {c_bytes} — hollow bar */
}}
""".rstrip()

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

    code = "\n".join(header) + "\n\n" + "\n".join(defines) + "\n" + body + footer
    return Deployment(
        chip_id=chip.id,
        title=f"{chip.name} — {unit.name}",
        code=code,
        tiles=tiles,
        waves=waves,
        units=units,
        resident_tiles=resident,
        reloads=reloads,
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
    header = f"""/* {chip.name} — how this model deploys the run
 * {workload}
 *
 * A network is a SEQUENCE here: no overlap is modelled between one operation's
 * prefetch and the previous operation's arithmetic (D5a). Within an operation,
 * load and compute overlap when capacity granted a second buffer.
 *
 * {trace.tiles} operations, drawn as {steps} step{"s" if steps != 1 else ""}.
 */

#define OPS   {trace.tiles:<10} /* graph nodes in this phase */
#define UNITS {unit.count:<10} /* {unit.name} */

for (int i = 0; i < OPS; ++i) {{
    dispatch(op[i]);                  /* hatched bar, if the op costs one */

    load_B(op[i]);                    /* weights   — solid bar   */
    load_A(op[i]);                    /* activations — hatched bar */

    if (is_matrix(op[i]))
        parallel_for (int u = 0; u < UNITS; ++u)
            {unit.name}(u, op[i]);
    else
        {machine.vector_unit.name}(op[i]);   /* norms, activations (D27) */

    store_C(op[i]);                   /* whatever no later op reads — hollow bar */
}}

/* span {format_time(trace.total_s)}; DRAM {format_bytes(trace.totals[Lane.DRAM])},"""
    footer = f" arithmetic {_int(trace.totals[Lane.CORE] + trace.totals[Lane.VECTOR])} OP */"
    return Deployment(
        chip_id=chip.id,
        title=f"{chip.name} — {unit.name}",
        code=header + footer,
        tiles=trace.tiles,
        waves=trace.tiles,
        units=max(unit.count, 1),
        resident_tiles=unit.resident_tile_capacity(),
        reloads=0,
        kind="operations",
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
