"""``analyze(model, hardware, deployment) -> Report`` — the whole engine, one call.

``api/`` and ``cli.py`` are thin shells over this function (CLAUDE.md). The
package is **pure**: no file I/O, no web framework, no global mutable state, no
wall-clock reads. Same inputs, identical output — which is what makes sweeps
parallelisable and snapshot tests meaningful.
"""

from __future__ import annotations

import bwz
from bwz.analysis.bottleneck import flip_margin, rank_operations, suggestions
from bwz.analysis.memory import infeasibility_reasons, plan_memory, usable_memory_fraction
from bwz.analysis.roofline import MachineModel, compute_dtype, machine_model
from bwz.analysis.schedule import run_phase
from bwz.graph.builder import build_graphs, phases_for
from bwz.graph.ops import GraphPhase
from bwz.report import (
    Bound,
    Confidence,
    FlipMargin,
    MemoryPlan,
    Meta,
    PhaseResult,
    Report,
    Summary,
    config_hash,
)
from bwz.spec.deployment import DeploymentSpec
from bwz.spec.hardware_spec import HardwareSpec
from bwz.spec.loaders import AnyModelSpec
from bwz.spec.model_spec import TransformerSpec
from bwz.units import format_bytes

__all__ = [
    "MachineModel",
    "analyze",
    "flip_margin",
    "machine_model",
    "plan_memory",
    "rank_operations",
    "suggestions",
]


def analyze(model: AnyModelSpec, hardware: HardwareSpec, deployment: DeploymentSpec) -> Report:
    """Predict how *model* runs on *hardware* under *deployment*.

    Never raises for a configuration that merely cannot run: an unsupported dtype
    or a model that does not fit returns a ``Report`` with ``feasible: false`` and
    the cheapest fixes (CLAUDE.md #8).
    """
    dtype = compute_dtype(hardware, deployment.precision.weights, deployment.precision.activations)
    meta = Meta(
        model_name=model.name,
        chip_name=hardware.name,
        bwz_version=bwz.__version__,
        config_hash=config_hash(model.id, hardware.id, deployment.model_dump_json()),
    )

    if not hardware.supports(dtype):
        available = sorted({d.value for u in hardware.compute_units for d in u.supported_dtypes})
        return _infeasible(
            meta,
            (
                f"{hardware.name} has no compute unit for {dtype.value!r}; it supports "
                f"{available}. Set precision.weights to one of those, or choose another chip.",
            ),
        )

    graphs = build_graphs(model, deployment)
    sizing_phase = GraphPhase.DECODE if GraphPhase.DECODE in graphs else next(iter(graphs))
    plan = plan_memory(graphs[sizing_phase], hardware, deployment)

    if not plan.fits:
        return _infeasible(meta, infeasibility_reasons(plan, hardware, deployment), plan)

    machine = machine_model(hardware, dtype)
    phases = tuple(
        run_phase(
            graphs[phase],
            machine,
            resident_fraction=plan.resident_fraction,
            activation_resident_fraction=plan.activation_resident_fraction,
            double_buffered=plan.double_buffered,
        )
        for phase in phases_for(model, deployment)
    )

    summary = _summarise(phases, machine, deployment)
    margins = tuple(flip_margin(phase, machine) for phase in phases)
    return Report(
        meta=meta,
        feasible=True,
        memory=plan,
        summary=summary,
        phases=phases,
        assumptions=_assumptions(model, hardware, deployment, plan, phases, machine, margins),
        flip_margins=margins,
        confidence=_confidence(hardware, margins),
    )


def _infeasible(meta: Meta, reasons: tuple[str, ...], plan: MemoryPlan | None = None) -> Report:
    empty = plan or MemoryPlan(
        weight_bytes=0.0,
        kv_cache_bytes=0.0,
        peak_activation_bytes=0.0,
        total_bytes=0.0,
        dram_capacity_bytes=0.0,
        usable_dram_bytes=0.0,
        on_chip_capacity_bytes=0.0,
        resident_fraction=0.0,
        activation_resident_fraction=0.0,
        double_buffered=False,
        fits=False,
    )
    return Report(
        meta=meta,
        feasible=False,
        memory=empty,
        summary=None,
        phases=(),
        assumptions=(),
        flip_margins=(),
        confidence=Confidence.LOW,
        infeasibility=reasons,
    )


def _summarise(
    phases: tuple[PhaseResult, ...], machine: MachineModel, deployment: DeploymentSpec
) -> Summary:
    """Roll the phases into the headline numbers.

    ``ttft`` is the prefill pass; ``tpot`` is one decode step. Total latency is
    ``ttft + output_tokens * tpot`` — the standard decomposition, and the reason
    the two phases are costed separately in the first place.
    """
    by_phase = {phase.phase: phase for phase in phases}
    prefill = by_phase.get(GraphPhase.PREFILL)
    decode = by_phase.get(GraphPhase.DECODE)

    ttft = prefill.latency_s if prefill else None
    tpot = decode.latency_s if decode else None
    tokens_per_s = deployment.batch / tpot if tpot else None

    latency = (ttft or 0.0) + (tpot or 0.0) * (deployment.output_tokens if tpot else 0)
    if not phases:
        latency = 0.0
    elif ttft is None and tpot is None:
        latency = sum(phase.latency_s for phase in phases)

    dominant = max(phases, key=lambda phase: phase.latency_s)
    total_flops = sum(phase.flops for phase in phases)
    achieved = total_flops / latency if latency > 0 else 0.0
    return Summary(
        latency_s=latency,
        ttft_s=ttft,
        tpot_s=tpot,
        tokens_per_s=tokens_per_s,
        throughput_per_s=deployment.batch / latency if latency > 0 else 0.0,
        peak_flops_per_s=machine.peak_flops_per_s,
        achieved_flops_per_s=achieved,
        utilization=achieved / machine.peak_flops_per_s if machine.peak_flops_per_s else 0.0,
        bound=dominant.bound,
    )


def _confidence(hardware: HardwareSpec, margins: tuple[FlipMargin, ...]) -> Confidence:
    """Never ``HIGH`` before Session 5.

    Every prediction currently rests on calibration constants that have not been
    fitted against a single published measurement. Claiming high confidence in
    that state would be the one failure mode this project cannot survive.
    """
    if hardware.hypothetical or any(margin.rests_on_estimate for margin in margins):
        return Confidence.LOW
    if any(margin.margin < 2.0 for margin in margins):
        return Confidence.LOW
    return Confidence.MEDIUM


def _assumptions(
    model: AnyModelSpec,
    hardware: HardwareSpec,
    deployment: DeploymentSpec,
    plan: MemoryPlan,
    phases: tuple[PhaseResult, ...],
    machine: MachineModel,
    margins: tuple[FlipMargin, ...],
) -> tuple[str, ...]:
    """Every shortcut taken and every estimate touched, in one drawer.

    This is the honesty mechanism of the whole tool (CLAUDE.md #4). It is
    assembled centrally rather than appended ad hoc because a user needs to be
    able to read all of it, and because an assumption that only fires sometimes
    is the one most likely to be missed.
    """
    out: list[str] = [
        "Calibration constants are documented defaults, not fitted values: no prediction here "
        "has been checked against a published measurement (docs/PLAN.md Session 5).",
        "Machine model is three elements (docs/CORRECTIONS.md D5a): DRAM supplies the only "
        "bandwidth ceiling, on-chip SRAM supplies capacity only, compute supplies TOPS. There is "
        "no on-chip bandwidth term.",
        f"Residency r = {plan.resident_fraction:.1%} applies to weights only; activations are "
        f"assumed to stream from DRAM even where on-chip capacity could hold them.",
        f"Peak throughput is the maximum over compute units, not their sum: "
        f"{machine.unit.name} at {machine.dtype.value}.",
        f"Achieved-throughput derating and shape utilisation are applied separately and multiply; "
        f"a batch-1 GEMM on a {machine.unit.systolic_dims} array loses far more to shape than to "
        f"derating.",
        "Peak activation footprint assumes an allocator that frees each tensor the instant its "
        "last reader completes — optimistic.",
        "A phase costs the sum of its operations; no overlap is modelled between one kernel's "
        "prefetch and the previous kernel's arithmetic.",
    ]

    if plan.double_buffered:
        out.append(
            f"On-chip capacity ({format_bytes(plan.on_chip_capacity_bytes)}) has room for two "
            f"tiles, so load and compute overlap within an operation: latency is "
            f"max(load, compute), not their sum."
        )
    else:
        out.append(
            f"On-chip capacity ({format_bytes(plan.on_chip_capacity_bytes)}) cannot hold two "
            f"tiles alongside the resident weights, so loads serialise behind compute: latency "
            f"is load + compute."
        )

    dispatched = phases[0].n_dispatched_ops if phases else 0
    total_ops = phases[0].n_ops if phases else 0
    out.append(
        f"{dispatched} of {total_ops} graph operations are charged a dispatch cost; norms and "
        f"elementwise ops are assumed fused into an adjacent kernel."
    )

    if isinstance(model, TransformerSpec):
        out.append("The LM head is computed for the last token only, as every serving stack does.")
        out.append(
            "Prefill attention FLOPs are halved for causal masking (docs/CORRECTIONS.md D8)."
        )
        error = model.declared_vs_derived_error()
        if error is not None and error > 0.01:
            out.append(
                f"Headline parameter count is declared, not derived: {model.declared_params:.3g} "
                f"against {model.parameter_count():.4g} from the hyperparameters, a {error:.1%} "
                f"gap. Weight traffic follows the declared figure."
            )

    if hardware.hypothetical:
        out.append(
            f"{hardware.name} is a hypothetical profile, not a product. Numbers derived from it "
            f"describe a design point (docs/CORRECTIONS.md D7)."
        )
    out.extend(
        f"Estimated input — {field}: {why}" for field, why in sorted(hardware.estimates.items())
    )

    out.append(
        f"Usable DRAM is {usable_memory_fraction(hardware):.0%} of nominal capacity "
        f"({format_bytes(plan.usable_dram_bytes)} of {format_bytes(plan.dram_capacity_bytes)})."
    )
    out.extend(
        f"{margin.bound.value} verdict rests on an estimated input: {margin.description}"
        for margin in margins
        if margin.rests_on_estimate
    )
    if deployment.parallelism.total_chips > 1:
        out.append(
            "Multi-chip parallelism is declared but not modelled in v1; sharding and collectives "
            "land at M5. Treat this as a single-chip result."
        )
    return tuple(out)


def bound_of(report: Report, phase: GraphPhase) -> Bound | None:
    """Convenience for tests and the CLI."""
    result = report.phase(phase)
    return result.bound if result else None
