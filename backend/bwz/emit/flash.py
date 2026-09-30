"""Write the FlashAttention plan bwz chose out as a runnable Python program.

``docs/CORRECTIONS.md`` D70, the attention counterpart of :mod:`bwz.emit.matmul`
(D54). :func:`bwz.analysis.flash.plan_flash` costs every block size and inner
dataflow the chip can run and keeps the fastest; this writes that plan as a
self-contained program which

1. computes ``O = softmax(Q Kᵀ / √d) V`` block by block, with the online softmax
   written out where a reader can see it, and checks the result against the
   textbook formula computed all at once;
2. walks the plan's programs across the chip's units in the same lockstep
   waves, and each inner matmul tile by tile under the stationarity the plan
   chose for it;
3. counts what it moves and what the vector unit does, and asserts those counts
   against the prediction embedded in its own source.

Everything in the file comes from the :class:`~bwz.analysis.flash.FlashPlan`
— never from geometry re-derived here — and the runtime is the same
``_harness.py`` every emitted matmul inlines.
"""

from __future__ import annotations

from dataclasses import dataclass

from bwz.analysis.flash import (
    PV,
    QK,
    FlashCandidate,
    FlashPlan,
    local_group_slices,
    versus_chosen,
)
from bwz.analysis.roofline import machine_model
from bwz.emit.matmul import COMMENT_COLUMN, _harness_split, _rule
from bwz.spec.dtypes import bytes_per_element, is_integer
from bwz.spec.hardware_spec import Dataflow, HardwareSpec
from bwz.units import format_bytes, format_time

_TITLE = {
    Dataflow.OUTPUT_STATIONARY: "output-stationary",
    Dataflow.WEIGHT_STATIONARY: "weight-stationary",
    Dataflow.INPUT_STATIONARY: "input-stationary",
}

TOLERANCE = 1e-3
"""Largest ``|O - reference|`` the emitted check accepts. Not a calibration
constant: it bounds fp32 rounding in the running softmax against the float64
reference, on dyadic operands no larger than 7/8 (or ±7 at an integer width).
The walk being wrong moves ``O`` by far more than this."""


@dataclass(frozen=True, slots=True)
class FlashProgram:
    """A runnable FlashAttention program and the numbers it was told to expect."""

    chip_id: str
    filename: str
    source: str
    predicted: dict[str, float]
    working_set_bytes: float


def default_filename(plan: FlashPlan, chosen: FlashCandidate) -> str:
    """``flash-<chip>-<dtype>-br<Br>-bc<Bc>-<qk flow>-<pv flow>.py``."""
    flows = f"{chosen.qk.chosen.stationarity.value}-{chosen.pv.chosen.stationarity.value}"
    return f"flash-{plan.chip_id}-{plan.shape.dtype.value}-br{chosen.br}-bc{chosen.bc}-{flows}.py"


def predicted_for(plan: FlashPlan, chosen: FlashCandidate) -> dict[str, float]:
    """The plan's own numbers, in the vocabulary the program counts in."""
    return {
        "programs": chosen.programs,
        "waves": chosen.waves,
        "used_cores": chosen.used_cores,
        "available_cores": plan.units,
        "idle_core_waves": chosen.waves * plan.units - chosen.programs,
        "macs": plan.shape.macs,
        "mac_slots": chosen.mac_slots,
        "staging_events": chosen.staging_events,
        "q_dram_bytes": chosen.q_bytes,
        "k_dram_bytes": chosen.k_bytes,
        "v_dram_bytes": chosen.v_bytes,
        "o_dram_bytes": chosen.o_bytes,
        "scores": chosen.scores,
        "rescaled": chosen.rescaled,
        "normalised": chosen.normalised,
        "inner_vector_adds": chosen.inner_vector_adds,
    }


def emit_flash(
    chip: HardwareSpec, plan: FlashPlan, *, command: str, version: str, seed: int = 20240501
) -> FlashProgram:
    """Build the runnable program for *plan*'s chosen candidate on *chip*."""
    chosen = plan.chosen
    if chosen is None:
        raise ValueError(
            f"{plan.chip_id}: the plan is infeasible ({'; '.join(plan.infeasibility)}), so "
            f"there is no decomposition to emit."
        )
    unit = machine_model(chip, plan.shape.dtype).unit
    qk_group = local_group_slices(unit, chosen.qk.chosen.stationarity, chosen.qk.chosen.k)
    pv_group = local_group_slices(unit, chosen.pv.chosen.stationarity, chosen.pv.chosen.k)
    shape = plan.shape
    operand = bytes_per_element(shape.dtype)
    working_set = 4 * shape.heads_total * shape.q_len * shape.head_dim * operand
    predicted = predicted_for(plan, chosen)
    qk_flow = chosen.qk.chosen.stationarity
    pv_flow = chosen.pv.chosen.stationarity
    flows = [qk_flow] if qk_flow is pv_flow else [qk_flow, pv_flow]

    lines = ["#!/usr/bin/env python3"]
    lines += _docstring(plan, chosen, command, version, working_set)
    lines += ["from __future__ import annotations", "", *_harness_split()[0], "", ""]
    lines += _constants(plan, chosen, predicted, seed, groups=(qk_group, pv_group))
    lines += ["", ""]
    for flow in flows:
        lines += _inner_matmul(flow)
        lines += ["", ""]
    lines += _program(qk_flow, pv_flow)
    lines += ["", ""]
    lines += _walk()
    lines += ["", ""]
    lines += _main(plan, chosen)
    lines += ["", ""]
    lines += _harness_split()[1]
    lines += ["", "", 'if __name__ == "__main__":', "    raise SystemExit(main())"]
    return FlashProgram(
        chip_id=plan.chip_id,
        filename=default_filename(plan, chosen),
        source=_tidy(lines),
        predicted=predicted,
        working_set_bytes=working_set,
    )


# ---------------------------------------------------------------------------- header


def _docstring(
    plan: FlashPlan, chosen: FlashCandidate, command: str, version: str, working_set: float
) -> list[str]:
    shape = plan.shape
    out = [
        f'"""{plan.chip_name} · {shape.dtype.value} · FlashAttention-2, Br={chosen.br}, '
        f"Bc={chosen.bc} — the plan bwz chose.",
        "",
        f"    $ {command}",
        f"    bwz {version}",
        "",
        "What it computes, for every (batch, head):",
        "",
        "    O = softmax(Q Kᵀ / sqrt(d)) V        Q, K, V, O are [seq, d]",
        "",
        "without ever holding the [seq x seq] score matrix. Each PROGRAM owns one",
        "Br-row block of Q and its O, on one matrix unit, and streams K and V past",
        "them Bc rows at a time:",
        "",
        "    for each Bc-row block j of K and V:",
        "        S  = Q_i Kᵀ_j / sqrt(d)              inner matmul 1: [Br,d] x [d,Bc]",
        "        m' = max(m, rowmax S)                the running max",
        "        P  = exp(S - m')",
        "        l  = exp(m - m') l + rowsum P        the running sum",
        "        O  = exp(m - m') O + P V_j           inner matmul 2: [Br,Bc] x [Bc,d]",
        "        m  = m'",
        "    O = O / l",
        "",
        "Rescaling O by exp(m - m') whenever the max moves is the whole trick: it makes",
        "the softmax exact without seeing a row's scores all at once.",
        "",
        *_loop_map(plan, chosen),
        "",
        *_glossary(plan, chosen),
        "",
        "Why this plan and not another: bwz costed every block size the chip can hold,",
        "each inner matmul under every dataflow the unit declares, with the same",
        "formula `bwz matmul` uses, and kept the fastest. What it beat:",
        "",
        "    Br     Bc   programs  waves    latency   vs chosen",
    ]
    for candidate in [c for c in plan.candidates if c.fits][:6]:
        mark = "  <- chosen" if candidate is chosen else f"  {versus_chosen(candidate, chosen)}"
        out.append(
            f"    {candidate.br:<5} {candidate.bc:>5}  {candidate.programs:>9,} "
            f"{candidate.waves:>6}  {format_time(candidate.latency_s):>9}{mark}"
        )
    for choice in (chosen.qk, chosen.pv):
        out += [
            "",
            f"{choice.name} at [{choice.chosen.m},{choice.chosen.k}] x "
            f"[{choice.chosen.k},{choice.chosen.n}] on one {plan.unit_name}:",
        ]
        for alt in choice.alternatives:
            tag = " <- chosen" if alt is choice.chosen else ""
            reduction = (
                f"{alt.partials} partials/output, {alt.vector_adds:,} vector adds"
                if alt.vector_adds
                else f"reduction {alt.placement.value}"
            )
            out.append(
                f"    {alt.stationarity.value}  {format_time(alt.t_s):>9}  ({reduction}){tag}"
            )
        if choice.forced:
            out.append("    (pinned with --stationarity, not chosen by the formula)")
    out += [
        "",
        "Not an illustration. This walks the same programs, deals them to units in the",
        "same waves, runs each inner matmul tile by tile the way its stationarity",
        "says, and counts what it moves. Section 4 holds the prediction it checks",
        "those counts against.",
        "",
        "NOT A BENCHMARK. It validates counts, not time: its wall clock has nothing to",
        "do with the latency the report predicts.",
        "",
        f"Working set: {format_bytes(working_set)} of Q, K, V and O.",
        "",
        "What is really executed here, and what is only written down:",
        "",
        "  blocks          real — Br x Bc, the ragged last block included",
        "  online softmax  real — computed, counted, and checked against the textbook",
        "  inner dataflow  real — a different loop nest per stationarity",
        "  K/V staging     real — one fetch per (wave, head, block), shared by the",
        "                  programs of that head running in that wave",
        "  overlap         written down only — 'latency = max(t_dram, t_compute)' is a",
        "                  claim about time, and this program counts",
        "",
        "S and P are held in fp32 here. On an integer chip a real kernel requantises P",
        "before P V — a rounding this program does not emulate, and which moves no",
        "byte this program counts, because P never leaves the unit.",
        '"""',
    ]
    return out


def _loop_map(plan: FlashPlan, chosen: FlashCandidate) -> list[str]:
    """The whole kernel as one loop nest, each level named by the constant it runs to."""
    qk = chosen.qk.chosen.stationarity.value
    pv = chosen.pv.chosen.stationarity.value
    return [
        "The map — every loop below, and the constant that bounds it:",
        "",
        "    for wave in range(WAVES):                      walk(): lockstep, all units at once",
        "      for core_id in range(USED_CORES):            one thread per modelled unit",
        "        program = wave * USED_CORES + core_id      round-robin deal (D30)",
        "        head, block = divmod(program, Q_BLOCKS)    which (batch, head), which Q rows",
        "        Q_i = Q[block*BR : (block+1)*BR]           run_program(): Q_i stays on the unit",
        "        for j in range(KV_BLOCKS):                 stream K_j, V_j: BC rows at a time",
        f"          S  = Q_i · K_jᵀ                          matmul_{qk}(): [BR, HEAD_DIM] x "
        "[HEAD_DIM, BC]",
        "          online softmax on S                      vector unit: max, exp, sum, rescale",
        f"          O += P · V_j                             matmul_{pv}(): [BR, BC] x "
        "[BC, HEAD_DIM]",
        "        O = O / l ; write O_i                      once per program",
        "",
        "  and inside each matmul_*() the array's own tile loops — ROWS x COLS instruction",
        "  tiles, ordered by that function's stationarity (see its docstring).",
    ]


def _glossary(plan: FlashPlan, chosen: FlashCandidate) -> list[str]:
    """Every parameter: its constant, the flag that sets it, what it means, this run's value."""
    shape = plan.shape
    qk = chosen.qk.chosen.stationarity.value
    pv = chosen.pv.chosen.stationarity.value
    formula = "chosen by the formula"
    rows = [
        ("BATCH", "-b/--batch", "independent sequences", f"{shape.batch}"),
        ("HEADS", "--heads", "attention heads per sequence", f"{shape.heads}"),
        ("Q_LEN", "-S/--seq", "query rows per head: the sequence length", f"{shape.q_len}"),
        ("KV_LEN", "--kv-len", "key/value rows per head (default: Q_LEN)", f"{shape.kv_len}"),
        (
            "HEAD_DIM",
            "--head-dim",
            "d: one head's width, the contraction of Q·Kᵀ",
            f"{shape.head_dim}",
        ),
        ("DTYPE", "-d/--dtype", "width of Q, K, V and O in DRAM", shape.dtype.value),
        (
            "BR",
            "--br",
            "Q rows one program owns on one unit",
            f"{chosen.br} ({'pinned' if plan.requested_br is not None else formula})",
        ),
        (
            "BC",
            "--bc",
            "K/V rows streamed past a program per step",
            f"{chosen.bc} ({'pinned' if plan.requested_bc is not None else formula})",
        ),
        (
            "QK_STATIONARITY",
            "--stationarity",
            "dataflow of S = Q·Kᵀ: which operand stays put",
            f"{qk} ({'pinned' if chosen.qk.forced else formula})",
        ),
        (
            "PV_STATIONARITY",
            "--stationarity",
            "dataflow of O += P·V",
            f"{pv} ({'pinned' if chosen.pv.forced else formula})",
        ),
        (
            "ROWS, COLS",
            "the chip profile",
            "one instruction tile of the array",
            f"{plan.array_rows} x {plan.array_cols}",
        ),
        ("AVAILABLE_CORES", "the chip profile", "matrix units on the chip", f"{plan.units}"),
        ("Q_BLOCKS", "derived", "ceil(Q_LEN / BR): programs per head", f"{chosen.q_blocks}"),
        ("KV_BLOCKS", "derived", "ceil(KV_LEN / BC): steps per program", f"{chosen.kv_blocks}"),
        ("PROGRAMS", "derived", "BATCH x HEADS x Q_BLOCKS", f"{chosen.programs}"),
        ("USED_CORES", "derived", "min(AVAILABLE_CORES, PROGRAMS)", f"{chosen.used_cores}"),
        ("WAVES", "derived", "ceil(PROGRAMS / USED_CORES)", f"{chosen.waves}"),
    ]
    out = [
        "The parameters — the constant, the flag that sets it, what it means, this run:",
        "",
    ]
    for name, flag, meaning, value in rows:
        out.append(f"  {name:<16} {flag:<17} {meaning}")
        out.append(f"  {'':<16} {'':<17} = {value}")
    out += [
        "",
        "  Stationarity, for one [M,K] x [K,N] matmul on a ROWS x COLS array (D53):",
        "    os  output-stationary: a block of the RESULT stays in the accumulator and all",
        "        of K is swept through it — nothing is partial, nothing to sum afterwards",
        "    ws  weight-stationary: a tile of the SECOND operand (Kᵀ, or V) stays put and",
        "        the rows of the first stream past; K is cut into slices whose partials",
        "        must be added — free in a periphery accumulator, else on the vector unit",
        "    is  input-stationary: a tile of the FIRST operand (Q, or P) stays put and the",
        "        columns of the second stream past; K is cut the same way",
    ]
    return out


# ------------------------------------------------------------------------- constants


def _constant(statement: str, *comments: str) -> list[str]:
    if not comments:
        return [statement]
    head = statement.ljust(COMMENT_COLUMN - 1) + f" # {comments[0]}"
    return [head, *(" " * COMMENT_COLUMN + f"# {rest}" for rest in comments[1:])]


def _constants(
    plan: FlashPlan,
    chosen: FlashCandidate,
    predicted: dict[str, float],
    seed: int,
    *,
    groups: tuple[int, int],
) -> list[str]:
    shape = plan.shape
    dtype = shape.dtype
    acc = plan.acc_dtype
    qk = chosen.qk.chosen
    pv = chosen.pv.chosen
    chip_unit = plan.unit_name
    out = [_rule("1. the shape")]
    out += _constant(f"BATCH, HEADS = {shape.batch}, {shape.heads}", "-b/--batch, --heads")
    out += _constant("HEADS_TOTAL = BATCH * HEADS", "every (batch, head) is independent")
    out += _constant(
        f"Q_LEN, KV_LEN = {shape.q_len}, {shape.kv_len}",
        "-S/--seq, --kv-len: query rows and key/value",
        "rows per head",
    )
    out += _constant(
        f"HEAD_DIM = {shape.head_dim}",
        "--head-dim: d, one head's width — the",
        "contraction of Q·Kᵀ",
    )
    out += _constant("SCALE = 1.0 / math.sqrt(HEAD_DIM)", "softmax(Q·Kᵀ / sqrt(d))")
    out += _constant(f'DTYPE = "{dtype.value}"', "-d/--dtype: Q, K, V and O in DRAM")
    out += _constant(
        f"DTYPE_BYTES = {bytes_per_element(dtype):g}", "what every DRAM byte count charges"
    )
    out += _constant(f'S_ACC_DTYPE = "{acc.value}"', "what Q·Kᵀ accumulates into")
    out += _constant('SOFTMAX_DTYPE = "fp32"', "S, P and O on the unit (see the header)")
    out += _constant(f"SEED = {seed}", "fixes the random Q, K and V, so runs repeat")
    out += [""]
    out += [_rule("2. the chip")]
    out += _constant(
        f"ROWS, COLS = {plan.array_rows}, {plan.array_cols}", f"{chip_unit}: systolic_dims"
    )
    out += _constant(
        "SLOTS_PER_MMA = ROWS * ROWS * COLS", "MAC positions one instruction tile issues"
    )
    out += _constant(f"AVAILABLE_CORES = {plan.units}", f"{chip_unit}: count")
    out += [""]
    out += [_rule("3. the plan bwz chose")]
    out += _constant(
        f"BR = {chosen.br}", "--br: Q rows one program owns on one unit", _pinned(plan.requested_br)
    )
    out += _constant(
        f"BC = {chosen.bc}",
        "--bc: K/V rows streamed past a program per step",
        _pinned(plan.requested_bc),
    )
    out += _constant(
        f'QK_STATIONARITY = "{qk.stationarity.value}"',
        f"{QK}: {_TITLE.get(qk.stationarity, qk.stationarity.value)}",
        "pinned with --stationarity"
        if chosen.qk.forced
        else "chosen by the formula; --stationarity pins it",
    )
    out += _constant(
        f'PV_STATIONARITY = "{pv.stationarity.value}"',
        f"{PV}: {_TITLE.get(pv.stationarity, pv.stationarity.value)}",
        "pinned with --stationarity"
        if chosen.pv.forced
        else "chosen by the formula; --stationarity pins it",
    )
    out += _constant(
        f"QK_GROUP_SLICES = {groups[0]}",
        "k-slices one periphery sums before a partial",
        "leaves for the vector unit (1: every slice does)",
    )
    out += _constant(
        f"PV_GROUP_SLICES = {groups[1]}",
        "the same, for P·V",
    )
    out += _constant("Q_BLOCKS = math.ceil(Q_LEN / BR)", "programs per head")
    out += _constant("KV_BLOCKS = math.ceil(KV_LEN / BC)", "blocks each program streams")
    out += _constant("PROGRAMS = HEADS_TOTAL * Q_BLOCKS", "one per (batch, head, Q block)")
    out += _constant(f"USED_CORES = {chosen.used_cores}", "min(AVAILABLE_CORES, PROGRAMS)")
    out += _constant(f"WAVES = {chosen.waves}", "ceil(PROGRAMS / USED_CORES), lockstep (D30)")
    out += _constant(f"TOLERANCE = {TOLERANCE:g}", "fp32 softmax against a float64 reference")
    out += [""]
    out += [_rule("4. the prediction")]
    out += ["# The plan's own numbers. main() asserts every one against what the walk counts."]
    out += ["PREDICTED = {"]
    for key, value in predicted.items():
        literal = f"{int(value)}" if float(value).is_integer() else repr(float(value))
        out += _constant(f'    "{key}": {literal},', _MEANING[key])
    out += ["}"]
    return out


_MEANING = {
    "programs": "BATCH x HEADS x Q_BLOCKS, each run once",
    "waves": "lockstep rounds of one program per unit",
    "used_cores": "units that ever get a program",
    "available_cores": "units the chip has",
    "idle_core_waves": "unit-rounds with no program: WAVES x AVAILABLE - PROGRAMS",
    "macs": "useful multiply-adds: 2 x heads x Q_LEN x KV_LEN x d",
    "mac_slots": "MAC positions issued, tile padding included",
    "staging_events": "K and V block fetches, one per (wave, head, block)",
    "q_dram_bytes": "Q read once",
    "k_dram_bytes": "K read once per wave its head spans",
    "v_dram_bytes": "V, the same",
    "o_dram_bytes": "O written once",
    "scores": "entries of S computed: heads x Q_LEN x KV_LEN",
    "rescaled": "O elements scaled by exp(m - m'), blocks 2 onward",
    "normalised": "O elements divided by l at the end",
    "inner_vector_adds": "k-slice partials summed on the vector unit",
}
"""What each prediction counts, written beside it in the emitted file."""


def _pinned(requested: int | None) -> str:
    return "pinned with the flag" if requested is not None else "chosen by the formula"


# ------------------------------------------------------------------------ the inner matmuls


_INNER_HEAD = {
    Dataflow.OUTPUT_STATIONARY: [
        '    """acc += a @ b, OUTPUT-STATIONARY: each ROWS x COLS block of acc sits in one',
        "    accumulator while all of K is swept through it. Nothing is ever partial, so",
        '    nothing is ever summed after the fact."""',
    ],
    Dataflow.WEIGHT_STATIONARY: [
        '    """acc += a @ b, WEIGHT-STATIONARY: each ROWS x COLS tile of b stays put while',
        "    every row band of a streams past it. K is cut into slices on the grid, so",
        "    each output meets one partial per slice; a periphery that sums GROUP_SLICES",
        '    of them does it for free, and the rest land on the vector unit (D62)."""',
    ],
    Dataflow.INPUT_STATIONARY: [
        '    """acc += a @ b, INPUT-STATIONARY: each ROWS x ROWS tile of a stays put while',
        "    every column band of b streams past it. K is cut on the grid's column axis,",
        '    so every slice beyond the first is a partial the vector unit adds (D62)."""',
    ],
}


def _inner_matmul(flow: Dataflow) -> list[str]:
    name = f"matmul_{flow.value}"
    head = [
        _rule(f"inner matmul, {_TITLE[flow]}"),
        f"def {name}(acc, a, b, acc_dtype, counters, ledger, group_slices):",
        *_INNER_HEAD[flow],
        "    m, k = shape_of(a)",
        "    n = shape_of(b)[1]",
    ]
    if flow is Dataflow.OUTPUT_STATIONARY:
        body = [
            "    for r0 in range(0, m, ROWS):",
            "        r1 = min(r0 + ROWS, m)",
            "        for c0 in range(0, n, COLS):",
            "            c1 = min(c0 + COLS, n)",
            "            block = copy_of(sub(acc, r0, r1, c0, c1))   # the accumulator",
            "            for k0 in range(0, k, ROWS):                 # K swept INSIDE the tile",
            "                k1 = min(k0 + ROWS, k)",
            "                mma(block, sub(a, r0, r1, k0, k1), sub(b, k0, k1, c0, c1),",
            "                    counters, SLOTS_PER_MMA)",
            "            put_block(acc, r0, c0, block)",
        ]
    elif flow is Dataflow.WEIGHT_STATIONARY:
        body = [
            "    for kt, k0 in enumerate(range(0, k, ROWS)):      # grid rows: k-slices",
            "        k1 = min(k0 + ROWS, k)",
            "        for c0 in range(0, n, COLS):                 # grid columns",
            "            c1 = min(c0 + COLS, n)",
            "            resident = sub(b, k0, k1, c0, c1)        # the stationary tile",
            "            for r0 in range(0, m, ROWS):             # M swept past it",
            "                r1 = min(r0 + ROWS, m)",
            "                partial = zeros(r1 - r0, c1 - c0, acc_dtype)",
            "                mma(partial, sub(a, r0, r1, k0, k1), resident,",
            "                    counters, SLOTS_PER_MMA)",
            "                add_block(acc, r0, c0, partial)",
            "                _count_partial(ledger, kt, group_slices, (r1 - r0) * (c1 - c0))",
        ]
    else:
        body = [
            "    for r0 in range(0, m, ROWS):                     # grid rows: bands of M",
            "        r1 = min(r0 + ROWS, m)",
            "        for kt, k0 in enumerate(range(0, k, ROWS)):  # grid columns: k-slices",
            "            k1 = min(k0 + ROWS, k)",
            "            resident = sub(a, r0, r1, k0, k1)        # the stationary tile",
            "            for c0 in range(0, n, COLS):             # N swept past it",
            "                c1 = min(c0 + COLS, n)",
            "                partial = zeros(r1 - r0, c1 - c0, acc_dtype)",
            "                mma(partial, resident, sub(b, k0, k1, c0, c1),",
            "                    counters, SLOTS_PER_MMA)",
            "                add_block(acc, r0, c0, partial)",
            "                _count_partial(ledger, kt, group_slices, (r1 - r0) * (c1 - c0))",
        ]
    return head + body


_COUNT_PARTIAL = [
    "def _count_partial(ledger, kt, group_slices, elements):",
    '    """Where a k-slice\'s partial met its accumulator, and who paid for the add.',
    "",
    "    Slice 0 lands on what is already there — the accumulator's own start — and",
    "    costs nothing. A slice inside its group is summed in the unit's periphery,",
    "    free. The first slice of every later group is a partial that LEFT the",
    '    periphery: the vector unit adds it, and that is what the report charges."""',
    "    if kt == 0:",
    "        return",
    "    if kt % group_slices:",
    '        ledger.add("periphery_adds", elements)',
    "    else:",
    '        ledger.add("inner_vector_adds", elements)',
]


# ------------------------------------------------------------------------ one program


def _program(qk_flow: Dataflow, pv_flow: Dataflow) -> list[str]:
    return [
        *_COUNT_PARTIAL,
        "",
        "",
        _rule("one program: one (batch, head, row block), on one unit"),
        "def run_program(program, wave, core_id, tensors, staging, counters, ledger):",
        "    Q, K, V, O = tensors",
        "    head, block = divmod(program, Q_BLOCKS)",
        "    r0, r1 = block * BR, min((block + 1) * BR, Q_LEN)",
        "    rows = r1 - r0",
        "    if DEBUG:",
        '        log_block(f"core {core_id} wave {wave}: program {program} = head {head}, "',
        '                  f"query rows {r0}:{r1}")',
        "",
        "    q = Q.read(head, r0, r1, 0, HEAD_DIM)             # Q_i: read once, kept on the unit",
        "    o = zeros(rows, HEAD_DIM, SOFTMAX_DTYPE)           # the O accumulator, on the unit",
        "    m = [-math.inf] * rows                             # running row max",
        "    l = [0.0] * rows                                   # running row sum",
        "",
        "    for j in range(KV_BLOCKS):",
        "        c0, c1 = j * BC, min((j + 1) * BC, KV_LEN)",
        "        # Staged chip-wide: every program of this head in this wave shares the",
        "        # fetch, so K and V cross DRAM once per wave the head spans (D33).",
        '        k = staging.band(("K", wave, head, j), lambda: K.read(head, c0, c1, 0, HEAD_DIM))',
        '        v = staging.band(("V", wave, head, j), lambda: V.read(head, c0, c1, 0, HEAD_DIM))',
        "",
        "        s = zeros(rows, c1 - c0, S_ACC_DTYPE)          # S = Q_i Kᵀ_j, never leaves",
        f"        matmul_{qk_flow.value}(s, q, transpose(k), S_ACC_DTYPE, counters, ledger,",
        "                  QK_GROUP_SLICES)                # QK_STATIONARITY",
        "        s = as_dtype(s, SOFTMAX_DTYPE)",
        "        scale_all(s, SCALE)",
        "",
        "        m_new = [max(old, new) for old, new in zip(m, row_max(s))]",
        "        p = exp_shifted(s, m_new, SOFTMAX_DTYPE)        # P = exp(S - m')",
        "        alpha = [math.exp(old - new) for old, new in zip(m, m_new)]",
        "        l = [a * old + new for a, old, new in zip(alpha, l, row_sum(p))]",
        '        ledger.add("scores", rows * (c1 - c0))',
        "        if j > 0:",
        "            # The max moved, so everything O holds was weighted against the old",
        "            # one: rescale it. On block 0, O is still zero and there is nothing to do.",
        "            scale_rows(o, alpha)",
        '            ledger.add("rescaled", rows * HEAD_DIM)',
        "",
        f"        matmul_{pv_flow.value}(o, p, v, SOFTMAX_DTYPE, counters, ledger,",
        "                  PV_GROUP_SLICES)                # PV_STATIONARITY",
        "        m = m_new",
        "        if DEBUG:",
        '            log(f"  kv block {j}: keys {c0}:{c1}, row max now {max(m):.4g}")',
        "",
        "    divide_rows(o, l)                                   # O = O / l",
        '    ledger.add("normalised", rows * HEAD_DIM)',
        "    O.write(head, r0, 0, o)",
        "    counters.count_tile()",
        "    if DEBUG:",
        "        log_flush()",
    ]


def _walk() -> list[str]:
    return [
        _rule("the walk: programs dealt to units, one per unit per wave"),
        "def walk(tensors, counters, ledger):",
        '    """Round-robin, lockstep (D30): in wave w, unit c runs program w*USED_CORES + c."""',
        "    staging = Scratchpad(counters)",
        "",
        "    def core(core_id, end_of_wave):",
        "        for wave in range(WAVES):",
        "            program = wave * USED_CORES + core_id",
        "            if program < PROGRAMS:",
        "                run_program(program, wave, core_id, tensors, staging, counters, ledger)",
        "            else:",
        "                counters.count_idle_core_wave()          # the last wave, part empty",
        "            end_of_wave()",
        "",
        "    run_cores(USED_CORES, core)",
    ]


# ------------------------------------------------------------------------------ main


def _main(plan: FlashPlan, chosen: FlashCandidate) -> list[str]:
    integer = is_integer(plan.shape.dtype)
    title = (
        f"{plan.chip_name} · {plan.shape.dtype.value} · FlashAttention-2 "
        f"Br={chosen.br} Bc={chosen.bc}"
    )
    return [
        _rule("the textbook answer, to check against"),
        "def reference_attention(q, k, v):",
        '    """softmax(Q Kᵀ / sqrt(d)) V all at once, in float64: the whole score matrix,',
        '    exactly what the walk above never holds."""',
        f"    s = reference(q, transpose(k), {integer})",
        '    s = as_dtype(s, "fp64")',
        "    scale_all(s, SCALE)",
        '    p = exp_shifted(s, row_max(s), "fp64")',
        "    divide_rows(p, row_sum(p))",
        "    return reference(p, v, False)",
        "",
        "",
        _rule("run it"),
        "def main() -> int:",
        f'    print("{title}")',
        '    print(f"backend: {BACKEND};  {USED_CORES} threads, one per modelled unit in use")',
        "    counters = Counters()",
        "    ledger = Ledger()",
        "    host = {",
        "        name: [operand(rows, HEAD_DIM, DTYPE, SEED + 4 * h + i)",
        "               for h in range(HEADS_TOTAL)]",
        '        for i, (name, rows) in enumerate((("Q", Q_LEN), ("K", KV_LEN), ("V", KV_LEN)))',
        "    }",
        "    out = [zeros(Q_LEN, HEAD_DIM, SOFTMAX_DTYPE) for _ in range(HEADS_TOTAL)]",
        "    tensors = (",
        '        DramTensor("q", host["Q"], DTYPE_BYTES, ledger),',
        '        DramTensor("k", host["K"], DTYPE_BYTES, ledger),',
        '        DramTensor("v", host["V"], DTYPE_BYTES, ledger),',
        '        DramTensor("o", out, DTYPE_BYTES, ledger),',
        "    )",
        "",
        "    walk(tensors, counters, ledger)",
        "",
        "    error = max(",
        "        max_abs_diff(",
        '            out[h], reference_attention(host["Q"][h], host["K"][h], host["V"][h])',
        "        )",
        "        for h in range(HEADS_TOTAL)",
        "    )",
        "    never_used = (AVAILABLE_CORES - USED_CORES) * WAVES",
        "    measured = {",
        '        "programs": counters.tiles_run,',
        '        "waves": WAVES,',
        '        "used_cores": USED_CORES,',
        '        "available_cores": AVAILABLE_CORES,',
        '        "idle_core_waves": counters.idle_core_waves + never_used,',
        '        "macs": counters.macs,',
        '        "mac_slots": counters.mac_slots,',
        '        "staging_events": counters.staging_events,',
        '        "q_dram_bytes": ledger["q_dram_bytes"],',
        '        "k_dram_bytes": ledger["k_dram_bytes"],',
        '        "v_dram_bytes": ledger["v_dram_bytes"],',
        '        "o_dram_bytes": ledger["o_dram_bytes"],',
        '        "scores": ledger["scores"],',
        '        "rescaled": ledger["rescaled"],',
        '        "normalised": ledger["normalised"],',
        '        "inner_vector_adds": ledger["inner_vector_adds"],',
        "    }",
        "    checks = [",
        "        Check(name, float(PREDICTED[name]), float(measured[name]), asserted=True)",
        "        for name in PREDICTED",
        "    ]",
        "    failed = report_checks(checks)",
        '    print("")',
        "    print(f\"periphery adds   {ledger['periphery_adds']:,.0f}   summed where they were \"",
        '          "made: free, and not charged")',
        '    print(f"numerics: max |O - softmax(QKᵀ/√d)V| = {error:g}   "',
        '          f"(tolerance {TOLERANCE:g})")',
        "    if failed or error > TOLERANCE:",
        "        raise SystemExit(",
        '            f"{len(failed)} count(s) disagree with the plan"',
        '            + ("" if error <= TOLERANCE else " and O itself is wrong")',
        '            + ". One of the two is wrong, and finding out which is what this file"',
        '            + " is for."',
        "        )",
        '    print("")',
        '    print("every count matches the plan, and O == softmax(Q Kᵀ / sqrt(d)) V.")',
        "    return 0",
    ]


def check(program: FlashProgram) -> None:
    """Assert the emitted source says what the program object says, without running it.

    The flash counterpart of :func:`bwz.emit.check`: an emitter that wrote the
    wrong ``PREDICTED`` block would produce a program that passes while checking
    the wrong thing. Raises ``ValueError`` naming the key, the emitted value and
    the expected one (CLAUDE.md #8).
    """
    from bwz.emit import constants_of  # the package imports this module

    constants = constants_of(program.source)
    predicted = constants.get("PREDICTED")
    if not isinstance(predicted, dict):
        raise ValueError(f"{program.filename}: no PREDICTED dict literal at module level.")
    for key, expected in program.predicted.items():
        actual = predicted.get(key)
        if actual is None or float(actual) != float(expected):
            raise ValueError(
                f"{program.filename}: PREDICTED[{key!r}] is {actual!r}, but the plan says "
                f"{expected!r}."
            )
    for name, key in (("WAVES", "waves"), ("USED_CORES", "used_cores")):
        emitted = constants.get(name)
        if not isinstance(emitted, int) or emitted != program.predicted[key]:
            raise ValueError(
                f"{program.filename}: constant {name} is {emitted!r} but PREDICTED[{key!r}] is "
                f"{program.predicted[key]!r}; the walk and the check disagree."
            )


def _tidy(lines: list[str]) -> str:
    """At most two blank lines in a row, none at either end."""
    kept: list[str] = []
    blanks = 0
    for line in lines:
        blanks = blanks + 1 if not line.strip() else 0
        if blanks <= 2:
            kept.append(line.rstrip())
    while kept and not kept[0]:
        kept.pop(0)
    while kept and not kept[-1]:
        kept.pop()
    return "\n".join(kept) + "\n"


__all__ = ["FlashProgram", "check", "default_filename", "emit_flash", "predicted_for"]
