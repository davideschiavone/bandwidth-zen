"""Classification, flip margins and optimisation suggestions. ``docs/MODEL.md`` §6.6.

The flip margin is the part worth reading twice. A bottleneck label on its own
is a claim without an error bar: "DRAM-bound" reads identically whether the DRAM
term beats the next one by 260x or by 6%. When the binding input is an estimate —
and on an unfitted engine most of them are — the difference between those two
cases is the difference between a result and a coin toss.
"""

from __future__ import annotations

from bwz.analysis.roofline import MachineModel
from bwz.report import Bound, FlipMargin, PhaseResult
from bwz.spec.hardware_spec import HardwareSpec
from bwz.units import format_bandwidth, format_quantity

_TERM_INPUT = {
    Bound.DRAM_BW_BOUND: "dram bandwidth",
    Bound.COMPUTE_BOUND: "peak throughput",
    Bound.LATENCY_BOUND: "per-op overhead",
}

_ESTIMATE_KEYS = {
    Bound.DRAM_BW_BOUND: ("memory", "dram_bandwidth_efficiency"),
    Bound.COMPUTE_BOUND: ("compute_units", "clock_ghz", "achieved_flops_fraction"),
    Bound.LATENCY_BOUND: ("kernel_launch_overhead_s",),
}


def flip_margin(phase: PhaseResult, machine: MachineModel) -> FlipMargin:
    """How far the binding term can move before the verdict changes.

    Returns the ratio of the binding term to the runner-up, plus the concrete
    value at which they would swap — "DRAM-bound; compute takes over above
    X OP/s" is actionable in a way that "DRAM-bound" is not.
    """
    terms = {
        Bound.DRAM_BW_BOUND: phase.t_dram_s,
        Bound.COMPUTE_BOUND: phase.t_compute_s,
        Bound.LATENCY_BOUND: phase.t_fixed_s,
    }
    ordered = sorted(terms.items(), key=lambda item: item[1], reverse=True)
    (winner, winning_time), (runner_up, runner_time) = ordered[0], ordered[1]
    margin = winning_time / runner_time if runner_time > 0 else float("inf")

    description = _describe(winner, runner_up, margin, machine)
    return FlipMargin(
        quantity=_TERM_INPUT[winner],
        bound=winner,
        runner_up=runner_up,
        margin=margin,
        description=description,
        rests_on_estimate=_rests_on_estimate(winner, machine.chip),
    )


def _describe(winner: Bound, runner_up: Bound, margin: float, machine: MachineModel) -> str:
    if margin == float("inf"):
        return f"{winner.value} with no competing term; nothing else is doing measurable work."
    if winner is Bound.DRAM_BW_BOUND:
        threshold = machine.effective_bandwidth_bytes_per_s * margin
        return (
            f"DRAM-bound by {margin:.3g}x over {runner_up.value}; "
            f"{runner_up.value.lower().replace('_', ' ')} takes over above "
            f"{format_bandwidth(threshold)} of effective bandwidth."
        )
    if winner is Bound.COMPUTE_BOUND:
        threshold = machine.effective_flops_per_s * margin
        return (
            f"Compute-bound by {margin:.3g}x over {runner_up.value}; "
            f"DRAM takes over above {format_quantity(threshold, 'OP/s')} of effective throughput."
        )
    return (
        f"Latency-bound by {margin:.3g}x over {runner_up.value}; the work is smaller than the "
        f"cost of dispatching it, so only fusing or batching helps."
    )


def _rests_on_estimate(bound: Bound, chip: HardwareSpec) -> bool:
    """Whether the binding input is one the profile flagged as estimated (D7)."""
    if chip.hypothetical:
        return True
    roots = {key.split(".")[0] for key in chip.estimates}
    return bool(roots & set(_ESTIMATE_KEYS[bound]))


def suggestions(phase: PhaseResult, machine: MachineModel) -> tuple[str, ...]:
    """What to change, ordered by how much it would buy.

    Suggestions are derived from the binding term, not from a fixed list: telling
    a DRAM-bound user to buy a faster array is worse than saying nothing.
    """
    out: list[str] = []
    if phase.bound is Bound.DRAM_BW_BOUND:
        out.append(
            "Reduce bytes moved: quantise the weights further, or raise the batch size so each "
            "weight read serves more tokens."
        )
        if phase.utilization < 0.05:
            out.append(
                f"Utilisation is {phase.utilization:.2%} of peak — the compute array is nearly "
                f"idle. A cheaper chip with the same bandwidth would perform identically."
            )
    elif phase.bound is Bound.COMPUTE_BOUND:
        worst = min((op.utilization for op in phase.ops if op.flops > 0), default=1.0)
        if worst < 0.5:
            out.append(
                f"Shape utilisation bottoms out at {worst:.2%}: operands do not fill the "
                f"{machine.unit.systolic_dims} array. Larger batches or fused projections help."
            )
        out.append("Lower-precision arithmetic would raise the compute ceiling directly.")
    else:
        out.append(
            f"{phase.n_dispatched_ops} dispatches cost {phase.t_fixed_s * 1e3:.3g} ms of the "
            f"{phase.latency_s * 1e3:.3g} ms total. Fuse operations or raise the batch size; "
            f"faster memory and faster compute both buy nothing here."
        )
    return tuple(out)


def rank_operations(phase: PhaseResult, limit: int = 10) -> tuple[str, ...]:
    """The operations that dominate the phase, most expensive first."""
    ranked = sorted(phase.ops, key=lambda op: op.latency_s, reverse=True)[:limit]
    total = phase.latency_s or 1.0
    return tuple(
        f"{op.op_id}: {op.latency_s * 1e3:.3g} ms ({op.latency_s / total:.1%}), {op.bound.value}"
        for op in ranked
    )
