"""Assemble per-operation results into a phase. ``docs/MODEL.md`` §6.4.

v1 schedules a phase as the **sum** of its operation latencies, not as a critical
path over the DAG. That is the right call for a transformer, whose graph is a
chain — the critical path *is* the sum — and it is conservative for a CNN with
parallel branches. The DAG machinery to do better already exists in
``graph/dag.py`` and is used to report the serial lower bound alongside; a real
list scheduler with resource lanes is M6 work.

Within an operation, load and compute overlap or do not according to whether
on-chip capacity has room to double-buffer (``docs/CORRECTIONS.md`` D5a). Across
operations, no overlap is assumed: one kernel's weight prefetch does not hide the
previous kernel's arithmetic in this model.
"""

from __future__ import annotations

from bwz.analysis.roofline import DISPATCHED_OP_TYPES, MachineModel, classify, op_roofline
from bwz.graph.ops import ComputeGraph
from bwz.operators.base import cost_of
from bwz.report import PhaseResult


def run_phase(
    graph: ComputeGraph,
    machine: MachineModel,
    *,
    resident_fraction: float,
    activation_resident_fraction: float,
    double_buffered: bool,
) -> PhaseResult:
    """Cost every operation and aggregate.

    The phase ``bound`` is classified from the *summed* terms rather than by
    counting per-op verdicts: an operation that is latency-bound but takes 3 µs
    should not outvote one that is DRAM-bound and takes 3 ms.
    """
    results = tuple(
        op_roofline(
            op,
            cost_of(op, graph.tensors),
            machine,
            resident_fraction=resident_fraction,
            activation_resident_fraction=activation_resident_fraction,
            double_buffered=double_buffered,
        )
        for op in graph.ops
    )

    t_dram = sum(r.t_dram_s for r in results)
    t_compute = sum(r.t_compute_s for r in results)
    t_fixed = sum(r.t_fixed_s for r in results)
    latency = sum(r.latency_s for r in results)
    flops = sum(r.flops for r in results)
    dram_bytes = sum(r.dram_bytes for r in results)

    achieved = flops / latency if latency > 0 else 0.0
    return PhaseResult(
        phase=graph.phase,
        latency_s=latency,
        flops=flops,
        dram_bytes=dram_bytes,
        t_dram_s=t_dram,
        t_compute_s=t_compute,
        t_fixed_s=t_fixed,
        bound=classify(t_dram, t_compute, t_fixed),
        achieved_flops_per_s=achieved,
        utilization=achieved / machine.peak_flops_per_s if machine.peak_flops_per_s > 0 else 0.0,
        n_ops=len(results),
        n_dispatched_ops=sum(1 for op in graph.ops if op.op_type in DISPATCHED_OP_TYPES),
        ops=results,
    )
