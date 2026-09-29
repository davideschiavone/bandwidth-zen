"""Per-operation roofline: two ceilings and a fixed cost. ``docs/MODEL.md`` §6.3.

The v1 machine is three elements (``docs/CORRECTIONS.md`` D5a): DRAM supplies the
only bandwidth ceiling, SRAM supplies capacity, compute supplies TOPS. For one
operation::

    dram_bytes = (1 - r) * weight_bytes + input + output + scratch
    t_dram     = dram_bytes / (dram_bandwidth * bandwidth_efficiency)
    t_matrix   = flops / (peak_flops * achieved_fraction * shape_utilisation)
    t_vector   = (p - 1) * M * N / vector_flops        [a cut contraction, D62]
    t_compute  = max(t_matrix, t_vector)   [partials summed on chip, overlapped]
               = t_matrix + t_vector       [summed through DRAM, serialised]
    t_fixed    = per_op_overhead, for dispatched operations only
    latency    = max(t_dram, t_compute) + t_fixed      [double buffering fits]
               = t_dram + t_compute      + t_fixed      [otherwise]

Weights and activations get **separate** residency fractions, because on-chip
capacity is allocated to whichever saves more traffic per byte (see
``analysis/memory.py``). Ignoring activation residency would have charged
chip_a 2 GB of DRAM traffic for a 15 MB working set that plainly fits in its
55 MB of SRAM, inflating prefill TTFT by 58%.
"""

from __future__ import annotations

from dataclasses import dataclass

from bwz.analysis.stationarity import (
    NO_REDUCTION,
    ReductionCost,
    accumulates_locally,
    grid_for,
    reduction_cost,
)
from bwz.analysis.tiling import operation_grid, operation_utilisation
from bwz.calibration import (
    DEFAULT_ACHIEVED_FLOPS_FRACTION,
    DEFAULT_DRAM_BANDWIDTH_EFFICIENCY,
    DEFAULT_KERNEL_LAUNCH_OVERHEAD_S,
)
from bwz.graph.ops import ConvAttrs, MatmulAttrs, Operation, OpType
from bwz.operators.base import OpCost
from bwz.report import Bound, OpResult, ReductionPlacement
from bwz.spec.dtypes import DType
from bwz.spec.hardware_spec import ComputeUnit, Dataflow, HardwareSpec

MATRIX_OP_TYPES = frozenset({OpType.MATMUL, OpType.ATTENTION, OpType.CONV})
"""Operations a systolic array or tensor core can execute. Everything else —
norms, GELU, softmax tails, residuals, pooling — is elementwise or transcendental
work for the vector units, which on A100 run 16x slower than the tensor cores
(19.5 against 312 TOP/s). Charging them at the matrix rate overstated them by
that factor (docs/CORRECTIONS.md D27)."""

DISPATCHED_OP_TYPES = frozenset(
    {OpType.MATMUL, OpType.ATTENTION, OpType.CONV, OpType.POOL, OpType.EMBEDDING}
)
"""Operations that cost a dispatch. Norms and elementwise ops are assumed fused
into an adjacent kernel, which every serving stack does; counting all 451 graph
nodes as launches would triple the fixed cost of a transformer decode step."""


@dataclass(frozen=True, slots=True)
class MachineModel:
    """A chip reduced to the four numbers the roofline needs."""

    chip: HardwareSpec
    dtype: DType
    unit: ComputeUnit
    """The matrix engine: what a GEMM runs on, and what the headline peak quotes."""
    peak_flops_per_s: float
    effective_flops_per_s: float
    vector_unit: ComputeUnit
    """Where non-matrix work goes. The same unit when the profile declares only
    one, which is optimistic and recorded as an assumption."""
    effective_vector_flops_per_s: float
    effective_bandwidth_bytes_per_s: float
    per_op_overhead_s: float
    stationarity: Dataflow = Dataflow.WEIGHT_STATIONARY
    """The *effective* dataflow — the unit's own unless a deployment overrode it
    with one the unit supports. Carried here so the tile grid, the utilisation
    term, the trace and the listing all read the same choice and cannot
    disagree (D53)."""
    k_partitions: int = 1
    """Split-K slices, from ``DeploymentSpec.split_k`` (D53)."""

    def rate_for(self, op_type: OpType) -> float:
        """Effective throughput available to *op_type*."""
        return (
            self.effective_flops_per_s
            if op_type in MATRIX_OP_TYPES
            else self.effective_vector_flops_per_s
        )

    @property
    def has_vector_unit(self) -> bool:
        return self.vector_unit is not self.unit

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity where the two effective ceilings meet."""
        return self.effective_flops_per_s / self.effective_bandwidth_bytes_per_s


IDEAL_NOTE = (
    "Forced to 1.0 by --ideal. The result is a hardware ceiling, not a prediction: it assumes "
    "the memory system hits its pin rate and the array issues without a stall. Use it to reason "
    "about the machine; do not quote it as a throughput figure."
)


def idealised(chip: HardwareSpec) -> HardwareSpec:
    """*chip* with both efficiency de-ratings set to 1.0.

    Separates the two things that are conflated in a single latency number: the
    part that follows from published quantities (MACs, clock, bandwidth,
    capacity) and the part that follows from ``calibration.py`` constants nobody
    has fitted yet. Under ``--ideal`` only the first remains, so a number can be
    checked by hand against a datasheet.

    All three unfitted constants go: the two efficiencies and the per-dispatch
    overhead. Shape utilisation is **not** disabled — a batch-1 GEMM on a 512x512
    array still runs at 1/513 of peak. That is geometry, not a fudge factor: it follows
    from the array's declared dimensions and would be there on ideal silicon.

    The override is recorded in ``estimates`` so it propagates into
    ``report.assumptions`` through the normal path (D7). Nothing else has to know
    the flag exists.
    """
    return chip.model_copy(
        update={
            "dram_bandwidth_efficiency": 1.0,
            "achieved_flops_fraction": 1.0,
            # The launch overhead is the third unfitted constant, and leaving it
            # in was an inconsistency: on a small graph it *is* the latency, so
            # an "ideal" run came back 100% dispatch with the arithmetic
            # invisible beneath it (D25).
            "kernel_launch_overhead_s": 0.0,
            "estimates": {
                **chip.estimates,
                "dram_bandwidth_efficiency": IDEAL_NOTE,
                "achieved_flops_fraction": IDEAL_NOTE,
                "kernel_launch_overhead_s": IDEAL_NOTE,
            },
        }
    )


def machine_model(
    chip: HardwareSpec,
    dtype: DType,
    *,
    stationarity: Dataflow | None = None,
    k_partitions: int = 1,
) -> MachineModel:
    """Derive effective rates, honouring per-chip overrides over defaults (D6).

    *stationarity* defaults to the matrix unit's own declared dataflow, so a
    caller that does not care about the decomposition gets the chip's native one
    (D53). Validating a requested override against what the unit can actually
    run is :func:`analysis.dataflow.resolve_stationarity`'s job, not this one.
    """
    efficiency = (
        chip.dram_bandwidth_efficiency
        if chip.dram_bandwidth_efficiency is not None
        else DEFAULT_DRAM_BANDWIDTH_EFFICIENCY
    )
    achieved = (
        chip.achieved_flops_fraction
        if chip.achieved_flops_fraction is not None
        else DEFAULT_ACHIEVED_FLOPS_FRACTION
    )
    overhead = (
        chip.kernel_launch_overhead_s
        if chip.kernel_launch_overhead_s is not None
        else DEFAULT_KERNEL_LAUNCH_OVERHEAD_S
    )
    peak = chip.peak_flops_per_s(dtype)
    supported = [u for u in chip.compute_units if u.supports(dtype)]
    unit = max(supported, key=lambda u: u.peak_flops_per_s(chip.clock_hz, dtype))
    # The fastest unit that is *not* a systolic array is where elementwise and
    # transcendental work goes. A profile that declares only an array leaves us
    # charging that work at the array's rate, which is optimistic and said so in
    # the assumptions.
    vector_candidates = [u for u in supported if u.systolic_dims is None]
    vector = (
        max(vector_candidates, key=lambda u: u.peak_flops_per_s(chip.clock_hz, dtype))
        if vector_candidates
        else unit
    )
    return MachineModel(
        chip=chip,
        dtype=dtype,
        unit=unit,
        peak_flops_per_s=peak,
        effective_flops_per_s=peak * achieved,
        vector_unit=vector,
        effective_vector_flops_per_s=vector.peak_flops_per_s(chip.clock_hz, dtype) * achieved,
        effective_bandwidth_bytes_per_s=chip.dram.bandwidth_bytes_per_s * efficiency,
        per_op_overhead_s=overhead,
        stationarity=stationarity if stationarity is not None else unit.dataflow,
        k_partitions=max(1, k_partitions),
    )


def op_roofline(
    op: Operation,
    cost: OpCost,
    machine: MachineModel,
    *,
    resident_fraction: float,
    activation_resident_fraction: float,
    double_buffered: bool,
    terminal_output_bytes: float = 0.0,
    activation_traffic_multiplier: float = 1.0,
    weight_traffic_multiplier: float = 1.0,
) -> OpResult:
    """Predict one operation's latency and name what limits it.

    ``bound`` is the ``argmax`` over the three terms, so it answers "what would I
    have to change to make this faster" rather than "which is biggest on
    average". A ``LATENCY_BOUND`` operation is one where the dispatch costs more
    than the work.

    ``activation_traffic_multiplier`` and ``weight_traffic_multiplier`` scale the
    *compulsory* share of each operand's traffic and default to 1.0, which is
    every caller but the single-matmul dataflow strategies
    (``docs/CORRECTIONS.md`` D33/D36): ``stream`` re-reads A ``NTILES_PER_KS``
    times instead of staging it once, and a ``persistent`` B amortised over
    several iterations writes less than once. Unlike ``resident_fraction``,
    which only ever discounts traffic toward zero, a multiplier can exceed 1.0 —
    that is the only way to express re-reading an operand that already crosses
    DRAM in full.
    """
    # Reads and writes are charged separately because they are not
    # interchangeable. `terminal_output_bytes` is the share of this operation's
    # result that no later operation reads, and it must reach DRAM however much
    # capacity there is: for a standalone matmul that is the whole of C, and
    # discounting it by activation residency understated traffic (D22).
    resident_output = max(0.0, cost.output_bytes - terminal_output_bytes)
    # The two operands are kept apart all the way to the figure. They obey
    # different residency fractions and spill at different times — capacity goes
    # to activations before weights (D15), so B streams first — and "LOAD 107 MB"
    # does not say which of them crossed the bus.
    weight_read_bytes = weight_traffic_multiplier * (1.0 - resident_fraction) * cost.weight_bytes
    activation_read_bytes = (
        activation_traffic_multiplier
        * (1.0 - activation_resident_fraction)
        * (cost.input_bytes + cost.scratch_bytes)
    )
    read_bytes = weight_read_bytes + activation_read_bytes
    write_bytes = terminal_output_bytes + (1.0 - activation_resident_fraction) * resident_output

    utilisation = operation_utilisation(
        op,
        machine.unit,
        machine.dtype,
        stationarity=machine.stationarity,
        k_partitions=machine.k_partitions,
        vector_adder=machine.has_vector_unit,
    )
    rate = machine.rate_for(op.op_type)
    t_matrix = cost.flops / (rate * utilisation) if cost.flops > 0 and utilisation > 0 else 0.0

    t_fixed = machine.per_op_overhead_s if op.op_type in DISPATCHED_OP_TYPES else 0.0

    # Summing the partials a cut contraction leaves (D53/D62). Nothing here fires
    # for a grid that does not cut K, or for one whose unit accumulates K in its
    # own periphery. The three terms go to the three lanes that actually pay
    # them — the round trip to DRAM, the additions to the *vector* unit (a matrix
    # engine does MAC and nothing else, D27), the second launch to t_fixed.
    reduction = _reduction_for(op, cost, machine)
    t_reduce = (
        reduction.partial_sums / machine.effective_vector_flops_per_s
        if machine.effective_vector_flops_per_s > 0
        else 0.0
    )
    # Reduction overlap, and NOT double buffering — that word is already spoken
    # for, one level up, between the DRAM load and the compute (D5a). This is
    # between the two *engines* inside compute: the matrix cores build slice n+1
    # while the vector unit sums slice n, so the steady state is the slower of
    # the two. Fill and drain are omitted, exactly as max(t_dram, t_compute)
    # omits them a level up (D19). A DRAM reduction cannot pipeline that way —
    # the GEMM has to have finished everywhere before the partials are all
    # there — so it serialises, which is what split-K has always been charged.
    t_compute = (
        max(t_matrix, t_reduce)
        if reduction.placement is ReductionPlacement.ON_CHIP
        else t_matrix + t_reduce
    )
    t_fixed += reduction.dispatches * machine.per_op_overhead_s

    # Kept out of read_bytes/write_bytes rather than split half and half into
    # them: those two are the *operands'* traffic, A and B in and C out, and the
    # trace divides them among tiles. The partials are neither — they are one
    # kernel's output read back as another's input — so they get their own term
    # and their own span (D53).
    dram_bytes = read_bytes + write_bytes + reduction.dram_bytes
    t_dram = dram_bytes / machine.effective_bandwidth_bytes_per_s

    overlapped = max(t_dram, t_compute) if double_buffered else t_dram + t_compute
    latency = overlapped + t_fixed

    return OpResult(
        op_id=op.id,
        op_type=op.op_type,
        layer=op.layer,
        flops=cost.flops,
        weight_bytes=cost.weight_bytes,
        dram_bytes=dram_bytes,
        dram_read_bytes=read_bytes,
        dram_write_bytes=write_bytes,
        dram_weight_read_bytes=weight_read_bytes,
        dram_activation_read_bytes=activation_read_bytes,
        dram_reduction_bytes=reduction.dram_bytes,
        t_arith_s=t_matrix,
        t_reduce_s=t_reduce,
        reduction_placement=reduction.placement,
        arithmetic_intensity=cost.arithmetic_intensity,
        utilization=utilisation,
        t_dram_s=t_dram,
        t_compute_s=t_compute,
        t_fixed_s=t_fixed,
        latency_s=latency,
        bound=classify(t_dram, t_compute, t_fixed),
    )


def _reduction_for(op: Operation, cost: OpCost, machine: MachineModel) -> ReductionCost:
    """What summing this operation's partial results costs (D53/D62).

    :data:`~analysis.stationarity.NO_REDUCTION` for everything but a matmul on a
    grid that cuts the contraction. The accumulator width is read back out of the
    result the operator model already sized — ``output_bytes / (M*N)`` — rather
    than taken from the deployment's precision, so it is the same width the
    report writes C at, and the same width the capacity test in
    :func:`~analysis.stationarity.reduction_placement` measures against.
    """
    if machine.unit.systolic_dims is None:
        return NO_REDUCTION
    if isinstance(op.attrs, ConvAttrs):
        # A convolution's im2col grid is K x N, like ws. Its partials are charged
        # only where a unit groups them (D69): the occupancy the groups buy has a
        # price, and taking one without the other would be a free lunch. Where
        # a conv's k-slices are spread without grouping — every MMA unit — no
        # reduction is charged, as before D69: a gap recorded there, not closed.
        conv_grid = operation_grid(op, machine.unit)
        if conv_grid is None or not accumulates_locally(conv_grid, machine.unit):
            return NO_REDUCTION
        grid = conv_grid
    elif isinstance(op.attrs, MatmulAttrs):
        rows, cols = machine.unit.systolic_dims
        grid = grid_for(
            machine.stationarity, op.attrs, rows, cols, k_partitions=machine.k_partitions
        )
    else:
        return NO_REDUCTION
    elements = grid.m * grid.n
    accumulator_bytes = cost.output_bytes / elements if elements else 0.0
    return reduction_cost(
        grid,
        accumulator_bytes,
        unit=machine.unit,
        on_chip_capacity_bytes=machine.chip.on_chip_capacity_bytes,
        vector_adder=machine.has_vector_unit,
    )


def classify(t_dram: float, t_compute: float, t_fixed: float) -> Bound:
    """Name the largest of the three terms.

    Ties go to DRAM, which is the default state of inference and the one a user
    can most readily act on.
    """
    if t_fixed > t_dram and t_fixed > t_compute:
        return Bound.LATENCY_BOUND
    if t_compute > t_dram:
        return Bound.COMPUTE_BOUND
    return Bound.DRAM_BW_BOUND


def compute_dtype(chip: HardwareSpec, weights: DType, activations: DType) -> DType:
    """Which dtype's peak rate governs.

    Quantised inference runs the GEMM at the weight dtype when the hardware has a
    unit for it — that is the entire point of quantising — and falls back to the
    activation dtype otherwise. Returns the weight dtype even when unsupported so
    that the feasibility check can name it; callers must check ``chip.supports``
    first.
    """
    if chip.supports(weights):
        return weights
    return activations
