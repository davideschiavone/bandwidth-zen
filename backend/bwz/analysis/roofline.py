"""Per-operation roofline: two ceilings and a fixed cost. ``docs/MODEL.md`` §6.3.

The v1 machine is three elements (``docs/CORRECTIONS.md`` D5a): DRAM supplies the
only bandwidth ceiling, SRAM supplies capacity, compute supplies TOPS. For one
operation::

    dram_bytes = (1 - r) * weight_bytes + input + output + scratch
    t_dram     = dram_bytes / (dram_bandwidth * bandwidth_efficiency)
    t_compute  = flops / (peak_flops * achieved_fraction * shape_utilisation)
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

from bwz.analysis.tiling import operation_utilisation
from bwz.calibration import (
    DEFAULT_ACHIEVED_FLOPS_FRACTION,
    DEFAULT_DRAM_BANDWIDTH_EFFICIENCY,
    DEFAULT_KERNEL_LAUNCH_OVERHEAD_S,
)
from bwz.graph.ops import Operation, OpType
from bwz.operators.base import OpCost
from bwz.report import Bound, OpResult
from bwz.spec.dtypes import DType
from bwz.spec.hardware_spec import ComputeUnit, HardwareSpec

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
    peak_flops_per_s: float
    effective_flops_per_s: float
    effective_bandwidth_bytes_per_s: float
    per_op_overhead_s: float

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


def machine_model(chip: HardwareSpec, dtype: DType) -> MachineModel:
    """Derive effective rates, honouring per-chip overrides over defaults (D6)."""
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
    unit = max(
        (u for u in chip.compute_units if u.supports(dtype)),
        key=lambda u: u.peak_flops_per_s(chip.clock_hz, dtype),
    )
    return MachineModel(
        chip=chip,
        dtype=dtype,
        unit=unit,
        peak_flops_per_s=peak,
        effective_flops_per_s=peak * achieved,
        effective_bandwidth_bytes_per_s=chip.dram.bandwidth_bytes_per_s * efficiency,
        per_op_overhead_s=overhead,
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
) -> OpResult:
    """Predict one operation's latency and name what limits it.

    ``bound`` is the ``argmax`` over the three terms, so it answers "what would I
    have to change to make this faster" rather than "which is biggest on
    average". A ``LATENCY_BOUND`` operation is one where the dispatch costs more
    than the work.
    """
    # Reads and writes are charged separately because they are not
    # interchangeable. `terminal_output_bytes` is the share of this operation's
    # result that no later operation reads, and it must reach DRAM however much
    # capacity there is: for a standalone matmul that is the whole of C, and
    # discounting it by activation residency understated traffic (D22).
    resident_output = max(0.0, cost.output_bytes - terminal_output_bytes)
    read_bytes = (1.0 - resident_fraction) * cost.weight_bytes + (
        1.0 - activation_resident_fraction
    ) * (cost.input_bytes + cost.scratch_bytes)
    write_bytes = terminal_output_bytes + (1.0 - activation_resident_fraction) * resident_output
    dram_bytes = read_bytes + write_bytes
    t_dram = dram_bytes / machine.effective_bandwidth_bytes_per_s

    utilisation = operation_utilisation(op, machine.unit)
    if cost.flops > 0 and utilisation > 0:
        t_compute = cost.flops / (machine.effective_flops_per_s * utilisation)
    else:
        t_compute = 0.0

    t_fixed = machine.per_op_overhead_s if op.op_type in DISPATCHED_OP_TYPES else 0.0

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
        arithmetic_intensity=cost.arithmetic_intensity,
        utilization=utilisation,
        t_dram_s=t_dram,
        t_compute_s=t_compute,
        t_fixed_s=t_fixed,
        latency_s=latency,
        bound=classify(t_dram, t_compute, t_fixed),
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
