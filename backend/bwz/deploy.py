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
    def b_load_line(wave_expr: str, indent: str) -> tuple[str, str]:
        return (
            f"{indent}load_B(u, tile({wave_expr}, u));  "
            f"/* {b_bytes} — solid bar; each tile fetched once */",
            "load_b",
        )

    def a_staging_lines(wave_expr: str, indent: str) -> list[tuple[str, str]]:
        # WHOLE has no per-wave staging line at all — it is a prologue-only ramp
        # (a_prologue_line below), same total bytes as staging per k-slice (D33).
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
                f"{indent}if (tile({wave_expr}, u) % NTILES_PER_KS == 0)  "
                f"/* this tile opens a k-slice */",
                "load_a",
            ),
            (
                f"{indent}    stage_A(u, KSLICE({wave_expr}, u)); "
                f"/* {a_stage} — hatched bar; staged once (D33) */",
                "load_a",
            ),
        ]

    a_prologue_line: tuple[str, str] | None = (
        (
            f"stage_all_of_A();  /* {format_bytes(result.dram_activation_read_bytes)} — hatched "
            f"bar; every k-slice ramped in before wave 0, same total as staging per k-slice (D33) "
            f"*/",
            "load_a",
        )
        if a_strategy is AStrategy.WHOLE
        else None
    )

    inner: list[str] = (
        [
            "            for (int c = 0; c < SUB_CYCLES; ++c)",
            "                feed(u, w % WEIGHT_SETS, &A[m][KSLICE(w, u)]);"
            "   /* one sub-cycle of the operand */",
        ]
        if sub_cycles > 1
        else ["            mac(u, &A[m][KSLICE(w, u)]);"]
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
        m_line = (
            f"        for (int m = 0; m < {stream_rows}; ++m)   /* M streams; it never tiles */",
            None,
        )
        if on_demand_line is None:
            return [
                ("    parallel_for (int u = 0; u < UNITS; ++u)", None),
                m_line,
                *inner_lines,
            ]
        return [
            ("    parallel_for (int u = 0; u < UNITS; ++u) {", None),
            on_demand_line,
            m_line,
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
