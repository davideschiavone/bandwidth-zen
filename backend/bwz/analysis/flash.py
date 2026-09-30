"""FlashAttention-2 as a decomposition: which blocks, which inner dataflow, and why.

``docs/MODEL.md`` §6.9, ``docs/CORRECTIONS.md`` D70. The attention *operator*
(:mod:`bwz.operators.attention`) prices FlashAttention as a byte count and
nothing else. This module prices it the way ``bwz matmul`` prices a GEMM: as a
grid of work dealt to the chip's matrix units, and it chooses the plan by
running that formula over every plan the chip can run.

**The decomposition.** For every ``(batch, head)`` the query rows are cut into
``Br``-row blocks, and each block is one *program*: it keeps its ``Q_i`` block
and its output accumulator on one matrix unit and streams every ``Bc``-row block
of K and V past them::

    for j in kv blocks:
        S   = Q_i @ K_j^T / sqrt(d)          inner matmul 1: M=Br, N=Bc, K=d
        m'  = max(m, rowmax(S))              online softmax (Milakov & Gimelshein 2018)
        P   = exp(S - m')
        l   = exp(m - m') * l + rowsum(P)
        O   = exp(m - m') * O + P @ V_j      inner matmul 2: M=Br, N=d,  K=Bc
        m   = m'
    O = O / l

``Br x Bc`` never leaves the unit, which is the whole of FlashAttention; the
vector work it adds is the rescale of ``O`` on every block after the first
(Dao 2023, FlashAttention-2, Algorithm 1).

**The cost, per program**, is the chain above run serially on one unit::

    t_program = sum_j [ t(S = Q K^T) + t_softmax(Br x Bc) + t(O += P V) ]
                + t_rescale + t_normalise

each inner matmul costed by the *same* formula as a lone matmul on one unit —
shape padding against the array (D52), and the reduction its stationarity owes
if it cuts K (D62). Programs are dealt round-robin, one per unit per lockstep
wave (D30), so::

    t_compute = sum over waves of the slowest program in that wave
    t_dram    = (Q + O once, K and V once per wave a head's programs span) / bw
    latency   = max(t_dram, t_compute) + t_fixed     [K/V double buffered]

**The choice.** ``Br``, ``Bc`` and each inner matmul's stationarity are free.
Every combination the chip can hold is costed and the fastest kept, and every
one is returned so that the reader can see what the winner beat and by how
much — the choice is an output of the formula, not a heuristic beside it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from bwz.analysis.roofline import MachineModel, classify, idealised, machine_model
from bwz.analysis.stationarity import (
    K_ON_GRID,
    accumulates_locally,
    accumulation_depth,
    grid_for,
    partials_per_output,
    refusal_reason,
)
from bwz.analysis.tiling import padded, systolic_utilisation
from bwz.calibration import SOFTMAX_FLOPS_PER_SCORE
from bwz.graph.ops import MatmulAttrs
from bwz.report import Bound, ReductionPlacement
from bwz.spec.dtypes import DType, accumulator_for, bytes_per_element
from bwz.spec.hardware_spec import ComputeUnit, Dataflow, HardwareSpec
from bwz.units import format_bytes

QK = "S = Q·Kᵀ"
PV = "O += P·V"

_PREFERENCE = (
    Dataflow.OUTPUT_STATIONARY,
    Dataflow.WEIGHT_STATIONARY,
    Dataflow.INPUT_STATIONARY,
    Dataflow.ROW_STATIONARY,
)
"""Tie-break between inner dataflows that cost the same, after the unit's own:
the one that cuts K least. ``os`` owes no reduction at all."""


@dataclass(frozen=True, slots=True)
class FlashShape:
    """One attention call: ``O = softmax(Q Kᵀ / √d) V`` for every (batch, head).

    Bidirectional — every query sees every key. This is the encoder's attention,
    which is what ``bwz encoder-layer`` models; a causal mask would skip blocks
    above the diagonal and is not modelled here (D70).
    """

    batch: int
    heads: int
    q_len: int
    kv_len: int
    head_dim: int
    dtype: DType

    @property
    def heads_total(self) -> int:
        return self.batch * self.heads

    @property
    def macs(self) -> int:
        """``2 · Sq · Skv · d`` per head: ``Q·Kᵀ`` and ``P·V`` issue the same count."""
        return 2 * self.heads_total * self.q_len * self.kv_len * self.head_dim


@dataclass(frozen=True, slots=True)
class InnerCost:
    """One inner matmul, costed on one unit under one stationarity."""

    stationarity: Dataflow
    m: int
    n: int
    k: int
    grid_rows: int
    grid_cols: int
    utilisation: float
    """Shape padding against one array (D52). No occupancy term: the program
    owns its unit, and the chip's other units are running other programs."""
    placement: ReductionPlacement
    k_slices: int
    partials: int
    """Partial values per output element that leave the periphery (D62/D69)."""
    vector_adds: int
    """``(partials - 1) · M · N`` — what the vector unit is charged."""
    t_matrix_s: float
    t_reduce_s: float

    @property
    def t_s(self) -> float:
        """The reduction overlaps the matrix work on chip (D62): the slower of the two."""
        return max(self.t_matrix_s, self.t_reduce_s)


@dataclass(frozen=True, slots=True)
class InnerChoice:
    """Which stationarity one inner matmul runs, and what the others would have cost."""

    name: str
    chosen: InnerCost
    alternatives: tuple[InnerCost, ...]
    """Every dataflow the unit declares, the chosen one included, cheapest first."""
    forced: bool
    """True when ``--stationarity`` chose, not the formula."""


@dataclass(frozen=True, slots=True)
class FlashCandidate:
    """One block size, fully costed."""

    br: int
    bc: int
    qk: InnerChoice
    pv: InnerChoice
    q_blocks: int
    kv_blocks: int
    programs: int
    waves: int
    used_cores: int
    occupancy: float
    kv_streams: int
    """Times a head's K (and V) cross DRAM, summed over heads: one per wave that
    head's programs span, since programs of one head in one wave share the
    staged block (D33's chip-wide staging, applied to K and V)."""
    staging_events: int
    """K and V block fetches: ``2 · kv_streams · kv_blocks``."""
    q_bytes: float
    k_bytes: float
    v_bytes: float
    o_bytes: float
    working_set_bytes: float
    """The worst wave's on-chip footprint, with K/V double buffered when it fits."""
    double_buffered: bool
    fits: bool
    mac_slots: int
    scores: int
    rescaled: int
    """``O`` elements multiplied by ``exp(m - m')`` — every block after a program's first."""
    normalised: int
    inner_vector_adds: int
    vector_ops: float
    t_matrix_s: float
    t_vector_s: float
    t_compute_s: float
    t_dram_s: float
    t_fixed_s: float
    latency_s: float
    bound: Bound

    @property
    def dram_bytes(self) -> float:
        return self.q_bytes + self.k_bytes + self.v_bytes + self.o_bytes


@dataclass(frozen=True, slots=True)
class FlashPlan:
    """The chosen plan, every candidate it was chosen from, and what it rests on."""

    shape: FlashShape
    chip_id: str
    chip_name: str
    unit_name: str
    vector_unit_name: str
    array_rows: int
    array_cols: int
    units: int
    on_chip_capacity_bytes: float
    per_unit_matrix_flops_per_s: float
    per_unit_vector_flops_per_s: float
    bandwidth_bytes_per_s: float
    acc_dtype: DType
    feasible: bool
    infeasibility: tuple[str, ...]
    candidates: tuple[FlashCandidate, ...]
    """Every block size costed, fastest first; infeasible ones last."""
    chosen: FlashCandidate | None
    requested_br: int | None
    requested_bc: int | None
    requested_stationarity: Dataflow | None
    assumptions: tuple[str, ...]


def plan_flash(
    chip: HardwareSpec,
    shape: FlashShape,
    *,
    br: int | None = None,
    bc: int | None = None,
    stationarity: Dataflow | None = None,
    ideal: bool = False,
) -> FlashPlan:
    """Cost every FlashAttention-2 plan *chip* can hold, and keep the fastest.

    *br*, *bc* and *stationarity* pin a choice instead of searching it; a pinned
    stationarity the unit does not declare is refused, not clamped (D53), and a
    pinned block that does not fit on chip is reported as infeasible rather than
    shrunk — both answer the question asked or say why they cannot.
    """
    hardware = idealised(chip) if ideal else chip
    refusal = _refusal(hardware, shape, stationarity, br, bc)
    if refusal:
        return _refused(hardware, shape, refusal, br, bc, stationarity)
    machine = machine_model(hardware, shape.dtype)
    unit = machine.unit
    assert unit.systolic_dims is not None  # _refusal checked it
    rows, cols = unit.systolic_dims
    step = max(rows, cols)
    brs = [br] if br is not None else _block_sizes(shape.q_len, step)
    bcs = [bc] if bc is not None else _block_sizes(shape.kv_len, step)

    costed = [
        _candidate(machine, shape, block_r, block_c, stationarity)
        for block_r in brs
        for block_c in bcs
    ]
    costed.sort(key=_rank)
    feasible = [c for c in costed if c.fits]
    chosen = feasible[0] if feasible else None
    infeasibility: tuple[str, ...] = ()
    if chosen is None:
        smallest = min(costed, key=lambda c: c.working_set_bytes)
        infeasibility = (
            f"no block size fits on chip: the smallest working set is "
            f"{format_bytes(smallest.working_set_bytes)} (Br={smallest.br}, Bc={smallest.bc}) "
            f"against {format_bytes(hardware.on_chip_capacity_bytes)} of on-chip capacity. "
            f"Fewer heads or a shorter head_dim shrinks it; pinning a smaller --br/--bc cannot "
            f"go below one array tile ({step}).",
        )
    return FlashPlan(
        shape=shape,
        chip_id=hardware.id,
        chip_name=hardware.name,
        unit_name=unit.name,
        vector_unit_name=machine.vector_unit.name,
        array_rows=rows,
        array_cols=cols,
        units=unit.count,
        on_chip_capacity_bytes=hardware.on_chip_capacity_bytes,
        per_unit_matrix_flops_per_s=machine.effective_flops_per_s / unit.count,
        per_unit_vector_flops_per_s=machine.effective_vector_flops_per_s / unit.count,
        bandwidth_bytes_per_s=machine.effective_bandwidth_bytes_per_s,
        acc_dtype=accumulator_for(shape.dtype),
        feasible=chosen is not None,
        infeasibility=infeasibility,
        candidates=tuple(costed),
        chosen=chosen,
        requested_br=br,
        requested_bc=bc,
        requested_stationarity=stationarity,
        assumptions=_assumptions(hardware, machine, shape, chosen),
    )


RANK_DIGITS = 6
"""Significant figures two latencies must share to count as a tie. Not a
calibration constant: two plans whose latencies differ in the tenth digit differ
by float rounding in how the same terms were summed, and the tie-break below —
less DRAM traffic — is then the real difference between them."""


def _rank(candidate: FlashCandidate) -> tuple[bool, float, float, float, int, int]:
    """Feasible first, then fastest; ties to less traffic, less vector work, bigger blocks.

    Bigger blocks last because they are the tie-break a kernel author would
    take: fewer programs to launch and fewer K/V block fetches, at no cost the
    model can see.
    """
    return (
        not candidate.fits,
        float(f"{candidate.latency_s:.{RANK_DIGITS}g}"),
        candidate.dram_bytes,
        candidate.vector_ops,
        -candidate.br,
        -candidate.bc,
    )


def versus_chosen(candidate: FlashCandidate, chosen: FlashCandidate) -> str:
    """How much slower *candidate* is than the plan kept: ``+3.2%``, or ``tie``.

    A tie is two latencies equal to :data:`RANK_DIGITS` figures — the same
    test the ranking applies — and is then decided by the traffic column.
    """
    ranked = f"{candidate.latency_s:.{RANK_DIGITS}g}" == f"{chosen.latency_s:.{RANK_DIGITS}g}"
    if ranked:
        return "tie"
    return f"+{candidate.latency_s / chosen.latency_s - 1:.1%}"


def _block_sizes(extent: int, step: int) -> list[int]:
    """``step · 2^i`` up to the first that covers *extent* — every tile-aligned power of two.

    A block that is not a whole number of array tiles pads every one of its
    tiles, so only multiples of the array are worth costing; powers of two keep
    the search to ``log2(extent / step)`` sizes a side. One block covering the
    whole extent is always included, so a short sequence is never cut at all.
    """
    sizes = []
    size = step
    while True:
        sizes.append(size)
        if size >= extent:
            return sizes
        size *= 2


def _refusal(
    chip: HardwareSpec,
    shape: FlashShape,
    stationarity: Dataflow | None,
    br: int | None,
    bc: int | None,
) -> tuple[str, ...]:
    """Why no plan can be costed at all, naming the field (CLAUDE.md #8)."""
    for name, value in (
        ("batch", shape.batch),
        ("heads", shape.heads),
        ("q_len", shape.q_len),
        ("kv_len", shape.kv_len),
        ("head_dim", shape.head_dim),
    ):
        if value < 1:
            return (f"{name}={value} is not a size; it must be at least 1.",)
    if not chip.supports(shape.dtype):
        available = sorted({d.value for u in chip.compute_units for d in u.supported_dtypes})
        return (
            f"dtype={shape.dtype.value!r} is not supported by {chip.id}; it declares "
            f"{', '.join(available)}.",
        )
    machine = machine_model(chip, shape.dtype)
    unit = machine.unit
    if unit.systolic_dims is None:
        return (
            f"{chip.id}: compute unit {unit.name!r} declares no systolic_dims at "
            f"{shape.dtype.value}, so there is no array to cut Q, K and V into tiles for.",
        )
    if not machine.has_vector_unit:
        # D62's rule, for the same reason: elementwise work charged to the matrix
        # engine would price exp and max at array rate, the opposite of the truth.
        return (
            f"{chip.id} declares no non-systolic unit supporting {shape.dtype.value}: the "
            f"online softmax's exp, max and rescale would have to run on {unit.name}, which "
            f"only multiplies and accumulates. Refused rather than costed at the array's rate "
            f"(D62). Add a vector unit to the profile, or use a chip that has one.",
        )
    if stationarity is not None:
        reason = refusal_reason(unit, stationarity)
        if reason is not None:
            return (reason,)
    rows, cols = unit.systolic_dims
    step = max(rows, cols)
    for name, block in (("br", br), ("bc", bc)):
        if block is not None and (block < step or block % step):
            return (
                f"{name}={block} is not a whole number of {unit.name} tiles; it must be a "
                f"positive multiple of {step}, or every tile it cuts is padded.",
            )
    return ()


def _refused(
    chip: HardwareSpec,
    shape: FlashShape,
    reasons: tuple[str, ...],
    br: int | None,
    bc: int | None,
    stationarity: Dataflow | None,
) -> FlashPlan:
    return FlashPlan(
        shape=shape,
        chip_id=chip.id,
        chip_name=chip.name,
        unit_name="",
        vector_unit_name="",
        array_rows=0,
        array_cols=0,
        units=0,
        on_chip_capacity_bytes=chip.on_chip_capacity_bytes,
        per_unit_matrix_flops_per_s=0.0,
        per_unit_vector_flops_per_s=0.0,
        bandwidth_bytes_per_s=0.0,
        acc_dtype=accumulator_for(shape.dtype) if chip.supports(shape.dtype) else shape.dtype,
        feasible=False,
        infeasibility=reasons,
        candidates=(),
        chosen=None,
        requested_br=br,
        requested_bc=bc,
        requested_stationarity=stationarity,
        assumptions=(),
    )


# ------------------------------------------------------------------ inner matmuls


def inner_cost(machine: MachineModel, m: int, n: int, k: int, stationarity: Dataflow) -> InnerCost:
    """One ``[M,K]x[K,N]`` on ONE unit, under *stationarity*.

    The lone-matmul formula with the chip cut down to a single unit::

        t_matrix = 2·M·N·K / (per_unit_rate · U_shape)
        t_reduce = (p - 1)·M·N / per_unit_vector_rate
        t        = max(t_matrix, t_reduce)                  [partials meet on chip]

    ``U_shape`` is the D52 padding product (a bit-serial array's K carries its
    sub-cycle fill, D34). ``p`` is :func:`~bwz.analysis.stationarity.partials_per_output`
    with no occupancy grouping — there are no idle units to borrow inside one
    program — so it is ``k_slices`` on a tensor core under ``ws``/``is``, and
    1 (``LOCAL``, free) on a unit whose periphery holds the whole contraction.

    Worked example, A100 fp16, ``S = Q·Kᵀ`` at Br=Bc=64, d=128, ``ws``: K=128
    is 8 slices of 16, so ``p = 8`` and the CUDA-core share adds
    ``7 · 64 · 64 = 28 672`` partials at 19.5 T / 432 = 45.1 GOP/s, 0.64 µs,
    against ``2·64·64·128 / (312 T / 432) = 1.45 µs`` of matrix work — hidden,
    exactly as D62 finds for a whole matmul.
    """
    unit = machine.unit
    assert unit.systolic_dims is not None
    rows, cols = unit.systolic_dims
    grid = grid_for(stationarity, MatmulAttrs(m=m, n=n, k=k), rows, cols)
    multiplier = unit.dtype_multipliers.get(machine.dtype, 1.0)
    fill = round(1 / multiplier) if 0 < multiplier < 1 else None
    utilisation = systolic_utilisation(m, k, n, rows, cols, units=1, fill_cycles=fill, tiles=1)
    per_unit = machine.effective_flops_per_s / unit.count
    per_unit_vector = machine.effective_vector_flops_per_s / unit.count
    t_matrix = 2.0 * m * n * k / (per_unit * utilisation)

    if not grid.needs_reduction or grid.k_slices <= 1:
        placement, partials = ReductionPlacement.NONE, 1
    else:
        # vector_adder=False: the occupancy grouping of D69 borrows idle units,
        # and a program has none — its unit is the only one it owns.
        partials = partials_per_output(grid, unit, vector_adder=False)
        placement = ReductionPlacement.LOCAL if partials == 1 else ReductionPlacement.ON_CHIP
    adds = (partials - 1) * m * n
    return InnerCost(
        stationarity=stationarity,
        m=m,
        n=n,
        k=k,
        grid_rows=grid.rows,
        grid_cols=grid.cols,
        utilisation=utilisation,
        placement=placement,
        k_slices=grid.k_slices,
        partials=partials,
        vector_adds=adds,
        t_matrix_s=t_matrix,
        t_reduce_s=adds / per_unit_vector,
    )


def local_group_slices(unit: ComputeUnit, stationarity: Dataflow, k: int) -> int:
    """k-slices one periphery sums before a partial must leave it, on one unit.

    The emitted program walks groups of this many slices, and hands the vector
    unit ``ceil(k_slices / this)`` partials per output — the same ``p`` that
    :func:`inner_cost` charges, for the full block *k* and any ragged one below
    it. ``floor(depth / tile_rows)`` where the unit declares a depth (Metis:
    16384 / 512 = 32); every slice of *k* where the depth is unbounded; 1 on a
    unit with no periphery accumulator at all, whose every slice leaves it.
    """
    assert unit.systolic_dims is not None
    rows, cols = unit.systolic_dims
    grid = grid_for(stationarity, MatmulAttrs(m=1, n=1, k=k), rows, cols)
    if stationarity not in K_ON_GRID or not accumulates_locally(grid, unit):
        return 1
    depth = accumulation_depth(unit)
    return grid.k_slices if math.isinf(depth) else max(1, int(depth // grid.tile_rows))


def _choose(
    machine: MachineModel, name: str, m: int, n: int, k: int, forced: Dataflow | None
) -> InnerChoice:
    """Cost *name* under every dataflow the unit declares and keep the cheapest.

    Ties go to the unit's own dataflow, then to the one that cuts K least — a
    tie means the reduction is hidden, and what it still costs then is
    accumulator capacity the formula does not charge here.
    """
    unit = machine.unit
    flows = (forced,) if forced is not None else unit.dataflows()
    costs = [inner_cost(machine, m, n, k, flow) for flow in flows]

    def order(cost: InnerCost) -> tuple[float, bool, int]:
        return (
            cost.t_s,
            cost.stationarity is not unit.dataflow,
            _PREFERENCE.index(cost.stationarity),
        )

    costs.sort(key=order)
    return InnerChoice(
        name=name, chosen=costs[0], alternatives=tuple(costs), forced=forced is not None
    )


# ------------------------------------------------------------------ one candidate


def _candidate(
    machine: MachineModel,
    shape: FlashShape,
    br: int,
    bc: int,
    forced: Dataflow | None,
) -> FlashCandidate:
    unit = machine.unit
    assert unit.systolic_dims is not None
    rows, cols = unit.systolic_dims
    d = shape.head_dim
    operand = bytes_per_element(shape.dtype)
    acc = bytes_per_element(accumulator_for(shape.dtype))
    per_unit_vector = machine.effective_vector_flops_per_s / unit.count

    # The full block decides the dataflow; the ragged last block runs the same one.
    qk = _choose(machine, QK, br, bc, d, forced)
    pv = _choose(machine, PV, br, d, bc, forced)

    q_blocks = math.ceil(shape.q_len / br)
    kv_blocks = math.ceil(shape.kv_len / bc)
    kv_widths = [min(bc, shape.kv_len - j * bc) for j in range(kv_blocks)]

    def program(rows_i: int) -> tuple[float, float, int, int, int, int, int]:
        """(t_matrix, t_vector, mac_slots, scores, rescaled, normalised, inner adds)."""
        t_mat = t_vec = 0.0
        slots = scores = rescaled = adds = 0
        for j, width in enumerate(kv_widths):
            first = inner_cost(machine, rows_i, width, d, qk.chosen.stationarity)
            second = inner_cost(machine, rows_i, d, width, pv.chosen.stationarity)
            block_scores = rows_i * width
            block_rescale = rows_i * d if j > 0 else 0
            # Serial: S must exist before softmax, P before P·V (FA2, one warp
            # group). The inner reductions overlap their own matmul (D62).
            t_mat += first.t_s + second.t_s
            t_vec += (SOFTMAX_FLOPS_PER_SCORE * block_scores + block_rescale) / per_unit_vector
            slots += _slots(rows_i, width, d, rows, cols) + _slots(rows_i, d, width, rows, cols)
            scores += block_scores
            rescaled += block_rescale
            adds += first.vector_adds + second.vector_adds
        normalised = rows_i * d
        t_vec += normalised / per_unit_vector
        return t_mat, t_vec, slots, scores, rescaled, normalised, adds

    full = program(min(br, shape.q_len))
    tail_rows = shape.q_len - (q_blocks - 1) * br
    tail = program(tail_rows) if tail_rows != min(br, shape.q_len) else full

    programs = shape.heads_total * q_blocks
    used = min(unit.count, programs)
    waves = math.ceil(programs / used)

    t_matrix = t_vector = 0.0
    worst_programs = worst_heads = 0
    for wave in range(waves):
        first_p, last_p = wave * used, min((wave + 1) * used, programs) - 1
        # Program p is row-block p % q_blocks of its head, and only the last
        # row-block is ragged. The wave is lockstep (D30): it lasts as long as
        # its slowest member, and a full block is never faster than the tail.
        tails = (last_p + 1) // q_blocks - first_p // q_blocks
        only_tails = tail is not full and tails == last_p - first_p + 1
        slowest = tail if only_tails else full
        t_matrix += slowest[0]
        t_vector += slowest[1]
        worst_programs = max(worst_programs, last_p - first_p + 1)
        worst_heads = max(worst_heads, last_p // q_blocks - first_p // q_blocks + 1)

    kv_streams = sum(
        ((h + 1) * q_blocks - 1) // used - (h * q_blocks) // used + 1
        for h in range(shape.heads_total)
    )
    per_program = min(br, shape.q_len) * (
        d * operand + d * acc + min(bc, shape.kv_len) * acc + 2 * acc
    )
    kv_block = 2 * min(bc, shape.kv_len) * d * operand
    single = worst_programs * per_program + worst_heads * kv_block
    double = worst_programs * per_program + 2 * worst_heads * kv_block
    capacity = machine.chip.on_chip_capacity_bytes
    double_buffered = double <= capacity
    fits = single <= capacity

    per_head_rows = shape.q_len * d * operand
    q_bytes = float(shape.heads_total * per_head_rows)
    o_bytes = q_bytes
    k_bytes = float(kv_streams * shape.kv_len * d * operand)
    v_bytes = k_bytes

    full_count = shape.heads_total * (q_blocks - 1) if tail is not full else programs
    tail_count = programs - full_count

    def total(index: int) -> int:
        return int(full[index]) * full_count + int(tail[index]) * tail_count

    scores, rescaled, normalised, adds = total(3), total(4), total(5), total(6)
    vector_ops = SOFTMAX_FLOPS_PER_SCORE * scores + rescaled + normalised + adds

    t_compute = t_matrix + t_vector
    t_dram = (q_bytes + k_bytes + v_bytes + o_bytes) / machine.effective_bandwidth_bytes_per_s
    t_fixed = machine.per_op_overhead_s
    overlapped = max(t_dram, t_compute) if double_buffered else t_dram + t_compute
    return FlashCandidate(
        br=br,
        bc=bc,
        qk=qk,
        pv=pv,
        q_blocks=q_blocks,
        kv_blocks=kv_blocks,
        programs=programs,
        waves=waves,
        used_cores=used,
        occupancy=programs / (waves * unit.count),
        kv_streams=kv_streams,
        staging_events=2 * kv_streams * kv_blocks,
        q_bytes=q_bytes,
        k_bytes=k_bytes,
        v_bytes=v_bytes,
        o_bytes=o_bytes,
        working_set_bytes=float(double if double_buffered else single),
        double_buffered=double_buffered,
        fits=fits,
        mac_slots=total(2),
        scores=scores,
        rescaled=rescaled,
        normalised=normalised,
        inner_vector_adds=adds,
        vector_ops=vector_ops,
        t_matrix_s=t_matrix,
        t_vector_s=t_vector,
        t_compute_s=t_compute,
        t_dram_s=t_dram,
        t_fixed_s=t_fixed,
        latency_s=overlapped + t_fixed,
        bound=classify(t_dram, t_compute, t_fixed),
    )


def _slots(m: int, n: int, k: int, rows: int, cols: int) -> int:
    """MAC positions issued for one ``[M,K]x[K,N]`` on a ``rows x cols`` array (D52)."""
    return padded(m, rows) * padded(n, cols) * padded(k, rows)


# ------------------------------------------------------------------ assumptions


def _assumptions(
    chip: HardwareSpec,
    machine: MachineModel,
    shape: FlashShape,
    chosen: FlashCandidate | None,
) -> tuple[str, ...]:
    unit = machine.unit
    out = [
        "Bidirectional attention: every query sees every key, as in an encoder. A causal "
        "mask would skip the blocks above the diagonal and roughly halve both matmuls; it "
        "is not modelled here (D70).",
        f"One program — one (batch, head, Br-row block) — runs on one {unit.name}, and "
        f"programs are dealt round-robin, one per unit per lockstep wave (D30). On a GPU a "
        f"thread block spans several tensor cores, so Br here is the rows ONE unit owns.",
        "Inside a program the chain S → softmax → P·V runs serially, as FlashAttention-2 "
        "issues it. Overlapping one block's softmax with the next block's matmul "
        "(FlashAttention-3's ping-pong) is not modelled: t_compute is the sum.",
        "K and V are staged chip-wide and shared by every program of the same head in the "
        "same wave — D33's staging, applied to K/V — so they cross DRAM once per wave a "
        "head's programs span. Nothing survives from one wave to the next.",
        f"Vector work runs at {machine.vector_unit.name}'s rate divided evenly over the "
        f"{unit.count} {unit.name} units: each program gets its share, never the whole chip's.",
        f"Softmax is charged {SOFTMAX_FLOPS_PER_SCORE:g} operations per score (calibration."
        "SOFTMAX_FLOPS_PER_SCORE): scale, max, subtract, exp, sum. The divide moves off the "
        "scores onto O — d per query row — and every block after a program's first rescales "
        "its O by exp(m - m'): d more per row per block. That rescale is the vector work "
        "FlashAttention adds; the matrix work, 2·Sq·Skv·d MACs per head, is unchanged.",
        "S, P, the running max and sum and the O accumulator never leave the unit. They are "
        "charged as on-chip capacity only: v1 has no on-chip bandwidth term (D5a).",
        f"A block fits when the worst wave's programs plus its heads' staged K/V blocks fit "
        f"the chip's whole on-chip capacity ({format_bytes(chip.on_chip_capacity_bytes)}, "
        f"every level above DRAM), as residency is judged everywhere in v1. A real kernel "
        f"keeps S and O in the innermost level — registers and shared memory on a GPU — "
        f"which caps Bc far lower than this does (FlashAttention-2 uses 64-128).",
        "Each inner matmul is costed by the lone-matmul formula on one unit — shape padding "
        "(D52) and its own reduction (D62) — with no wave-occupancy term: occupancy comes "
        "from how many programs there are, not from one program's tiles.",
    ]
    if unit.weight_sets > 1:
        out.append(
            f"{unit.name} is an in-memory array: under ws the stationary operand of S = Q·Kᵀ "
            f"is Kᵀ and of O += P·V is V, so every K and V block must be WRITTEN into the "
            f"array's banks before it can multiply anything. The capacity is modelled; the "
            f"time of that write is not, because it needs an on-chip bandwidth term v1 lacks "
            f"(D5a, D30). For attention — where both operands are activations — that write is "
            f"not amortised over a batch the way a weight's is."
        )
    # One line per distinct note, naming every field it covers: --ideal gives all
    # three the same paragraph, and repeating it three times buries the rest.
    notes: dict[str, list[str]] = {}
    for field in (
        "dram_bandwidth_efficiency",
        "achieved_flops_fraction",
        "kernel_launch_overhead_s",
    ):
        note = chip.estimates.get(field)
        if note is not None:
            notes.setdefault(note, []).append(field)
    out += [f"{', '.join(fields)}: {note}" for note, fields in notes.items()]
    if chosen is not None and not chosen.double_buffered:
        out.append(
            f"A second K/V buffer does not fit beside the working set "
            f"({format_bytes(chosen.working_set_bytes)} of "
            f"{format_bytes(chip.on_chip_capacity_bytes)}), so loads and compute serialise: "
            f"latency is t_dram + t_compute."
        )
    return tuple(out)
