"""``analyze(model, hardware, deployment) -> Report`` — the whole engine, one call.

``api/`` and ``cli.py`` are thin shells over this function (CLAUDE.md). The
package is **pure**: no file I/O, no web framework, no global mutable state, no
wall-clock reads. Same inputs, identical output — which is what makes sweeps
parallelisable and snapshot tests meaningful.
"""

from __future__ import annotations

import bwz
from bwz.analysis.bottleneck import flip_margin, rank_operations, suggestions
from bwz.analysis.dataflow import DataflowPlan, plan_dataflow
from bwz.analysis.memory import infeasibility_reasons, plan_memory, usable_memory_fraction
from bwz.analysis.pipeline import PipelineTrace, build_trace
from bwz.analysis.roofline import MachineModel, compute_dtype, idealised, machine_model
from bwz.analysis.schedule import run_phase
from bwz.analysis.stationarity import (
    K_ON_GRID,
    UNBOUNDED_ACCUMULATION,
    TileGrid,
    accumulation_depth,
    deal_for,
    refusal_reason,
    residency_phrase,
)
from bwz.graph.builder import build_graphs, phases_for
from bwz.graph.ops import GraphPhase
from bwz.operators.base import cost_of
from bwz.report import (
    Bound,
    Confidence,
    FlipMargin,
    MemoryPlan,
    Meta,
    OpResult,
    PhaseResult,
    ReductionPlacement,
    Report,
    Summary,
    config_hash,
)
from bwz.spec.deployment import AStrategy, BDataflow, DeploymentSpec
from bwz.spec.dtypes import DType
from bwz.spec.hardware_spec import ComputeUnit, Dataflow, HardwareSpec
from bwz.spec.loaders import AnyModelSpec
from bwz.spec.model_spec import MatmulSpec, ModelFamily, TransformerSpec
from bwz.units import format_bytes, format_quantity, format_time

__all__ = [
    "MachineModel",
    "PipelineTrace",
    "analyze",
    "build_trace",
    "flip_margin",
    "idealised",
    "machine_model",
    "plan_memory",
    "rank_operations",
    "suggestions",
    "trace_phases",
]


def analyze(model: AnyModelSpec, hardware: HardwareSpec, deployment: DeploymentSpec) -> Report:
    """Predict how *model* runs on *hardware* under *deployment*.

    Never raises for a configuration that merely cannot run: an unsupported dtype
    or a model that does not fit returns a ``Report`` with ``feasible: false`` and
    the cheapest fixes (CLAUDE.md #8).
    """
    # A bare matmul names its own operand widths and ignores the deployment's
    # weights/activations vocabulary entirely (docs/CORRECTIONS.md D18).
    dtype = (
        model.operand_dtype
        if isinstance(model, MatmulSpec)
        else compute_dtype(hardware, deployment.precision.weights, deployment.precision.activations)
    )
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

    # Refused, not clamped (D53): a stationarity is a whole decomposition, so
    # substituting a supported one would answer a different question than the
    # caller asked. Checked before any work, against the matrix unit that would
    # actually run the GEMM.
    if deployment.stationarity is not None:
        matrix_unit = max(
            (u for u in hardware.compute_units if u.supports(dtype)),
            key=lambda u: u.peak_flops_per_s(hardware.clock_hz, dtype),
        )
        refusal = refusal_reason(matrix_unit, deployment.stationarity) or _reduction_refusal(
            hardware, dtype, matrix_unit, deployment
        )
        if refusal is not None:
            return _infeasible(meta, (refusal,))

    graphs = build_graphs(model, deployment)
    sizing_phase = GraphPhase.DECODE if GraphPhase.DECODE in graphs else next(iter(graphs))
    plan = plan_memory(graphs[sizing_phase], hardware, deployment)

    if not plan.fits:
        return _infeasible(meta, infeasibility_reasons(plan, hardware, deployment), plan)

    machine = machine_model(
        hardware,
        dtype,
        stationarity=deployment.stationarity,
        k_partitions=deployment.split_k,
    )
    phases = tuple(
        run_phase(
            graphs[phase],
            machine,
            hardware,
            deployment,
            resident_fraction=plan.resident_fraction,
            activation_resident_fraction=plan.activation_resident_fraction,
            double_buffered=plan.double_buffered,
        )
        for phase in phases_for(model, deployment)
    )

    summary = _summarise(phases, machine, deployment)
    margins = tuple(flip_margin(phase, machine) for phase in phases)
    dataflow: DataflowPlan | None = None
    if isinstance(model, MatmulSpec):
        # Recomputed rather than threaded out of run_phase: plan_dataflow is pure,
        # so the same (op, machine, hardware, deployment) always gives the same
        # plan, and the assumptions drawer cannot disagree with the bytes charged.
        static_graph = graphs[GraphPhase.STATIC]
        op0_cost = cost_of(static_graph.ops[0], static_graph.tensors)
        dataflow = plan_dataflow(
            static_graph.ops[0], machine, hardware, deployment, a_bytes=op0_cost.input_bytes
        )
    return Report(
        meta=meta,
        feasible=True,
        memory=plan,
        summary=summary,
        phases=phases,
        assumptions=_assumptions(
            model, hardware, deployment, plan, phases, machine, margins, dataflow
        ),
        flip_margins=margins,
        confidence=_confidence(hardware, margins),
    )


def trace_phases(
    model: AnyModelSpec, hardware: HardwareSpec, deployment: DeploymentSpec
) -> tuple[tuple[GraphPhase, PipelineTrace], ...]:
    """The resource schedule behind each phase of :func:`analyze`.

    Rebuilds the graphs rather than carrying them on the ``Report``: building is
    cheap and deterministic, and the report stays a plain data structure that a
    frontend can consume without a graph library (CLAUDE.md).
    """
    report = analyze(model, hardware, deployment)
    if not report.feasible:
        return ()
    dtype = (
        model.operand_dtype
        if isinstance(model, MatmulSpec)
        else compute_dtype(hardware, deployment.precision.weights, deployment.precision.activations)
    )
    # Same stationarity the report was built with, or the trace would draw a
    # different decomposition than the numbers above it (D53).
    machine = machine_model(
        hardware,
        dtype,
        stationarity=deployment.stationarity,
        k_partitions=deployment.split_k,
    )
    graphs = build_graphs(model, deployment)
    dataflow: DataflowPlan | None = None
    if isinstance(model, MatmulSpec):
        static_graph = graphs[GraphPhase.STATIC]
        op0_cost = cost_of(static_graph.ops[0], static_graph.tensors)
        dataflow = plan_dataflow(
            static_graph.ops[0], machine, hardware, deployment, a_bytes=op0_cost.input_bytes
        )
    return tuple(
        (
            result.phase,
            build_trace(
                graphs[result.phase],
                result,
                machine,
                double_buffered=report.memory.double_buffered,
                dataflow=dataflow,
            ),
        )
        for result in report.phases
    )


def _reduction_refusal(
    hardware: HardwareSpec,
    dtype: DType,
    matrix_unit: ComputeUnit,
    deployment: DeploymentSpec,
) -> str | None:
    """Why a *requested* K-on-grid stationarity cannot be costed here (D62).

    Two refusals, both about a cost that would otherwise be charged to the wrong
    engine or counted twice — and both restricted to a stationarity the caller
    asked for, because a unit is never refused its own declared dataflow.

    **No genuine vector unit.** ``machine_model`` falls back to the matrix unit
    when no non-systolic unit supports the dtype, so on A100 at int8 the "vector
    unit" *is* the tensor core. Charging elementwise adds there would price them
    at 437 TOP/s — free, physically nonsense, and wrong in the one direction
    that matters: it would make ``ws`` look nearly as good as ``os``, which is
    the opposite of the finding. Refused, naming the chip, the dtype and what is
    missing. Declaring int8 on the CUDA cores (DP4A) would unlock these cases,
    but ``vector_unit`` also prices every non-matrix op (D27), so that moves
    documented int8 figures and belongs in its own change.

    **``--split-k`` on a grid that already carries K.** Cutting the contraction
    twice is incoherent, and the flag is an output-stationary knob (D53). Said
    rather than accepted-and-ignored (CLAUDE.md #8).
    """
    requested = deployment.stationarity
    if requested is None or requested not in K_ON_GRID:
        return None
    if deployment.split_k > 1:
        return (
            f"--split-k {deployment.split_k} cannot be combined with "
            f"stationarity={requested.value!r}: that grid already carries K on one of its axes, "
            f"so the contraction would be cut twice and its partials summed twice. Split-K is an "
            f"output-stationary knob (D53) — drop one of the two flags."
        )
    machine = machine_model(hardware, dtype, stationarity=requested)
    if machine.has_vector_unit or accumulation_depth(matrix_unit) == UNBOUNDED_ACCUMULATION:
        return None
    non_matrix = sorted(
        {
            d.value
            for u in hardware.compute_units
            if u.systolic_dims is None
            for d in u.supported_dtypes
        }
    )
    return (
        f"stationarity={requested.value!r} puts K on the tile grid, so partial sums leave "
        f"{matrix_unit.name} and something has to add them — but {hardware.name} declares no "
        f"non-systolic compute unit supporting {dtype.value!r}, so the only engine available is "
        f"{matrix_unit.name} itself. A matrix engine does matrix-multiply-accumulate and nothing "
        f"else (D27), and charging elementwise adds at its rate would report this decomposition "
        f"as nearly free. Refused rather than mispriced (D62). Non-systolic units here support "
        f"{non_matrix or 'nothing'}; run at one of those dtypes, or use the chip's own dataflow."
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
    dataflow: DataflowPlan | None = None,
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
        (
            f"Matrix work (matmul, attention, conv) runs on {machine.unit.name} at "
            f"{format_quantity(machine.effective_flops_per_s, 'OP/s')}; norms, activations and "
            f"other elementwise work runs on {machine.vector_unit.name} at "
            f"{format_quantity(machine.effective_vector_flops_per_s, 'OP/s')} — a tensor core "
            f"does matrix-multiply-accumulate and nothing else (D27)."
            if machine.has_vector_unit
            else f"{machine.chip.name} declares no non-systolic compute unit, so elementwise and "
            f"transcendental work is charged at the array's rate. That is optimistic: such work "
            f"does not use the array (D27)."
        ),
        (
            f"{machine.unit.name} and {machine.vector_unit.name} are drawn as separate lanes but "
            f"exchange nothing in this model: each operation's traffic is charged once, and no "
            f"register file or shared memory is modelled. Real silicon hands a GEMM's result to "
            f"the activation that follows it in registers, never through DRAM, so a fused "
            f"epilogue costs less here than the model's two dispatches suggest (D28)."
            if machine.has_vector_unit
            else "Only one compute unit is declared, so no matrix/vector handoff arises."
        ),
        "Non-linear functions are counted algebraically — GELU 8 operations per element, softmax "
        "5 per score, RMSNorm 4 — and charged at the vector rate. Real hardware pays more: an erf "
        "without a special-function unit is a polynomial approximation, and an SFU runs below the "
        "ALU rate. A deliberate simplification (D27), so these costs are a lower bound; on "
        "Llama-3-8B prefill non-matrix work is 1.4% of the phase, so a 4x transcendental penalty "
        "would move the total by 4%.",
        f"Achieved-throughput derating and shape utilisation are applied separately and multiply; "
        f"a batch-1 GEMM on a {machine.unit.systolic_dims} array loses far more to shape than to "
        f"derating.",
        (
            f"{machine.unit.name} is {machine.unit.count} arrays, and work reaches them a wave of "
            f"{machine.unit.count} weight tiles at a time: an operation with fewer tiles than "
            f"that leaves the rest idle, and the last wave of any operation is partly empty. "
            f"Charged as wave occupancy in the utilisation (D30) — without it the aggregate peak "
            f"silently assumes every array always has a tile."
            if machine.unit.count > 1
            else f"{machine.unit.name} is a single array, so there is no wave quantisation."
        ),
        (
            f"{machine.unit.name} holds {machine.unit.weight_sets} weight tiles per array and "
            f"{machine.unit.resident_tile_capacity()} across the chip. An in-memory-compute "
            f"weight cannot join a MAC until it has been written into a bank. Within one pass "
            f"every tile is written exactly once whether or not it fits, since M is the innermost "
            f"loop and a tile is never revisited; what the capacity decides is the cost of the "
            f"NEXT run on the same weights — free if B fits, all of it again if it does not. "
            f"Neither the writes nor that reuse are charged any time: both need an on-chip "
            f"bandwidth term, which the v1 machine model does not have (D5b, D30)."
            if machine.unit.weight_sets > 1
            else f"{machine.unit.name} stores no weights of its own — operands are read per "
            f"instruction — so there is no weight-residency limit on the array itself (D30)."
        ),
        "Peak activation footprint assumes an allocator that frees each tensor the instant its "
        "last reader completes — optimistic.",
        "A phase costs the sum of its operations; no overlap is modelled between one kernel's "
        "prefetch and the previous kernel's arithmetic.",
        "Traffic is charged in exact bytes. Real DRAM moves whole bursts, so a transfer narrower "
        "than one — a gather of short embedding rows, any small tensor — costs more than this "
        "says, and a scattered gather loses row-buffer locality besides. Immaterial at "
        "Llama-3-8B's 8 kB rows; not immaterial on a narrow model.",
        "DRAM traffic is compulsory traffic: each operand crosses the bus once. Re-reads forced "
        "by tiling a working set that does not fit on chip are not modelled, so a DRAM-bound "
        "latency here is a lower bound (docs/MODEL.md 6.2).",
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

    if isinstance(model, MatmulSpec):
        # The one operation a MatmulSpec has, and the only place the reduction's
        # placement and its two engine times are recorded (D62).
        matmul_result = phases[0].ops[0] if phases and phases[0].ops else None
        out.append(
            f"Arithmetic runs at {model.operand_dtype.value}, the wider of the two operands "
            f"(A {model.a_dtype.value}, B {model.b_dtype.value}): both enter the array through "
            f"one datapath, so a narrow operand saves bytes but buys no throughput "
            f"(docs/CORRECTIONS.md D18)."
        )
        out.append(
            f"The {model.result_dtype.value} result is an accumulator width: it changes bytes "
            f"only, never operations. 2*M*N*K is the same at every result width."
        )
        out.append(
            "The activation-residency discount (D15) — inter-operation reuse, which a one-op "
            "graph cannot have — does not apply to a lone matmul's A; a_strategy governs it "
            "instead (docs/CORRECTIONS.md D33)."
        )
        if dataflow is not None:
            out.extend(_stationarity_assumptions(dataflow, machine, matmul_result))
            if dataflow.a_strategy is AStrategy.STREAM:
                a_line = (
                    f"A: streamed per tile (D31), {dataflow.tiles_per_a_event}x the staged total "
                    f"— {format_bytes(dataflow.a_bytes)} would cross DRAM once under stage/whole, "
                    f"{format_bytes(dataflow.a_bytes * dataflow.a_bytes_multiplier)} crosses it "
                    f"under stream"
                )
            elif dataflow.a_strategy is AStrategy.WHOLE:
                a_line = (
                    f"A: staged whole, {format_bytes(dataflow.a_bytes)} before the first tile — "
                    f"crosses DRAM exactly once (D33), same bytes as stage"
                )
            else:
                a_line = (
                    f"A: staged {format_bytes(dataflow.a_bytes_per_event)} per "
                    f"{dataflow.group_name} ({dataflow.a_events} {dataflow.group_name}s) — "
                    f"crosses DRAM exactly once (D33)"
                )
            if dataflow.b_dataflow is BDataflow.PERSISTENT:
                b_line = (
                    f"B: persistent over {dataflow.iterations} iterations — the first writes "
                    f"all {dataflow.tiles} tiles, the rest reuse them; this report charges the "
                    f"amortised share, 1/{dataflow.iterations} of a full write"
                    if dataflow.iterations > 1
                    else "B: persistent — resident for this one-invocation report; a repeat "
                    "invocation (iterations > 1) would write 0 B for the rest"
                )
            elif dataflow.b_dataflow is BDataflow.ON_DEMAND:
                b_line = "B: on-demand, exposed on the critical path"
            elif machine.unit.weight_sets > 1:
                b_line = f"B: write-ahead depth {machine.unit.weight_sets} (D33)"
            else:
                b_line = "B: write-ahead is moot — the array stores no weights (D30)"
            out.append(f"{a_line} · {b_line}.")
            out.extend(dataflow.notes)

    if isinstance(model, TransformerSpec) and model.family is ModelFamily.TRANSFORMER_ENCODER:
        # The deployment can ask for generation; an encoder has none to give, and
        # silently dropping the request would be the kind of omission this drawer
        # exists to prevent (D24).
        out.append(
            "Encoder: one bidirectional pass over all input_tokens, no LM head. "
            + (
                f"output_tokens={deployment.output_tokens} and phase="
                f"{deployment.phase.value} are ignored — there is no decode phase."
                if deployment.output_tokens > 0
                else "There is no decode phase."
            )
        )

    if isinstance(model, TransformerSpec) and model.family is ModelFamily.TRANSFORMER_DECODER:
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


def _stationarity_assumptions(
    dataflow: DataflowPlan, machine: MachineModel, op: OpResult | None = None
) -> list[str]:
    """What the decomposition is, who chose it, and what it costs (D53/D62).

    Every one of these is load-bearing: the tile count, the wave occupancy and
    the reduction all follow from the stationarity, so a reader who cannot see
    which one ran cannot check any of the three. *op* is the matmul's own
    result, which is where the reduction's placement and its two engine times
    are read from — recomputing them here would be a second opinion about the
    same numbers.
    """
    grid = dataflow.grid
    if grid is None:
        return [
            f"stationarity={dataflow.stationarity.value} is declared but this profile gives no "
            f"array geometry, so there is no tile grid to decompose against and no reduction to "
            f"charge (D53)."
        ]
    adds = (grid.k_slices - 1) * grid.m * grid.n
    chosen = (
        f"asked for with stationarity={dataflow.stationarity.value!r}"
        if dataflow.requested_stationarity is not None
        else f"{machine.unit.name}'s own declared dataflow"
    )
    out = [
        f"Stationarity {dataflow.stationarity.value} — {residency_phrase(grid, machine.unit)} "
        f"({chosen}). The parallel grid is {grid.rows} x {grid.cols} tiles "
        f"({grid.row_dim.value} x {grid.col_dim.value}, cut by the "
        f"{grid.tile_rows}x{grid.tile_cols} array), each sweeping {grid.swept_dim.value}; "
        f"{grid.tiles:,} tiles in all. Every stationarity issues the same M*N*K MACs — what "
        f"differs is the quantisation loss and whether partial sums must be reduced (D53)."
    ]
    if dataflow.stationarity is Dataflow.ROW_STATIONARY:
        out.append(
            "Row-stationary is defined for completeness, after Eyeriss (Chen/Emer/Sze, ISCA "
            "2016): one A row per PE with the contraction spread across the array's own "
            "columns. No shipped profile declares it and nothing here has been checked against "
            "a measurement of such a machine, so its numbers are UNVALIDATED (D53)."
        )
    if grid.materialises_partials:
        out.append(
            f"split_k={grid.k_partitions}: the contraction is cut into {grid.k_partitions} "
            f"independent pieces, so this runs as CUTLASS's two kernels — a partitionedK GEMM "
            f"and a batched reduction. Charged: {grid.k_partitions} full M x N partials written "
            f"and read back, {(grid.k_partitions - 1) * grid.m * grid.n:,.0f} additions on "
            f"{machine.vector_unit.name} at "
            f"{format_quantity(machine.effective_vector_flops_per_s, 'OP/s')}, and one extra "
            f"dispatch. Those additions are not new arithmetic — 2*M*N*K already counts them — "
            f"but they leave the matrix engine's accumulator for the vector unit, which is where "
            f"their cost comes from (D27/D53)."
        )
    elif grid.needs_reduction:
        if dataflow.split_k > 1:
            out.append(
                f"split_k={dataflow.split_k} was requested but {dataflow.stationarity.value} "
                f"already carries K on the tile grid, so there is nothing left to split: the "
                f"flag is inert here and costs nothing. It is an output-stationary knob (D53)."
            )
        out.append(
            f"{dataflow.stationarity.value} carries K on the tile grid, so each of the "
            f"{grid.k_slices:,} k-slices computes a partial value for every one of the "
            f"{grid.accumulator_elements:,} output elements, and the {adds:,.0f} additions that "
            f"sum them are NOT new arithmetic — 2*M*N*K already counts them. What changes is "
            f"where they run: they leave the matrix engine's own accumulator, which is the whole "
            f"cost. The sliver this double-counts is those same adds at the matrix rate (D53/D62)."
        )
        out.extend(_placement_assumptions(grid, machine, op))
    return out


def _placement_assumptions(grid: TileGrid, machine: MachineModel, op: OpResult | None) -> list[str]:
    """Where this grid's partials met, and everything that was *not* charged (D62).

    The three placements make three different claims, and each of them rests on
    something this model does not have: an on-chip bandwidth term, a
    synchronisation cost, or an enforced wave assignment. All three are named
    here rather than left implicit — the reduction is the one part of a K-on-grid
    decomposition whose cost is a modelling choice rather than an arithmetic
    consequence.
    """
    if op is None:
        return []
    engine = machine.vector_unit.name
    rate = format_quantity(machine.effective_vector_flops_per_s, "OP/s")
    ratio = (
        machine.effective_flops_per_s / machine.effective_vector_flops_per_s
        if machine.effective_vector_flops_per_s > 0
        else 0.0
    )
    depth = accumulation_depth(machine.unit)

    if op.reduction_placement is ReductionPlacement.LOCAL:
        source = (
            f"{machine.unit.name} declares local_accumulation_inputs={depth:,.0f} and K="
            f"{grid.k:,} is inside it, so every k-slice of an output element is summed in that "
            f"unit's own periphery and never reaches on-chip memory. Nothing is charged, which "
            f"is the hardware's answer rather than a modelling shortcut."
            if depth != UNBOUNDED_ACCUMULATION
            else f"{machine.unit.name} runs {machine.stationarity.value} natively but declares no "
            f"local_accumulation_inputs, so this model assumes K accumulates locally at ANY "
            f"depth and charges nothing. That is the claim it has always made for a K-on-grid "
            f"grid, and it is unfalsifiable as it stands: declaring a depth would bound it."
        )
        dealt = deal_for(grid, machine.unit)
        units = machine.unit.count
        idle = (
            f" Here N gives only {grid.cols:,} output column(s) for {units:,} units, so "
            f"{units - dealt.used_cores:,} of them never receive a tile: wave occupancy "
            f"{dealt.occupancy:.1%}, charged in the utilisation. Spreading the k-slices over "
            f"every unit would fill them, but then the partials would have to leave the unit "
            f"and be summed elsewhere — a different decomposition, not modelled here."
            if dealt.used_cores < units
            else ""
        )
        return [
            f"Reduction: LOCAL. {source}",
            f"The locality that makes it free also needs the {grid.k_slices:,} k-slices of one "
            f"output column to land on the SAME unit, and the deal enforces it (D68): a wave "
            f"never straddles a grid row, so unit u keeps column round * "
            f"{dealt.used_cores:,} + u for every k-slice.{idle}",
        ]

    if op.reduction_placement is ReductionPlacement.ON_CHIP:
        binds = op.t_reduce_s >= op.t_arith_s
        return [
            f"Reduction: ON_CHIP. {machine.unit.name} declares no accumulator that survives "
            f"across k-slices, so the partials go out to on-chip memory and "
            f"{engine} adds them at {rate} — {ratio:.0f}x below {machine.unit.name} (D27). That "
            f"is {format_time(op.t_reduce_s)} of vector work against "
            f"{format_time(op.t_arith_s)} of matrix work, and "
            + (
                "the VECTOR unit is what binds: this operation is compute-bound on the engine "
                "that is not doing the multiplies."
                if binds
                else "the matrix engine still binds, so the reduction costs no latency here at "
                "all — it costs capacity."
            ),
            "Reduction overlap (NOT double buffering, which is the DRAM-to-compute overlap one "
            "level up): the matrix engine builds the next slice while the vector unit sums the "
            "last, so compute is max(matrix, vector) rather than their sum. Steady state only — "
            "pipeline fill and drain are omitted, exactly as max(t_dram, t_compute) omits them "
            "(D19) — and synchronisation between the two engines is NOT charged: the vector unit "
            "is assumed to know when a partial has landed, for free.",
            f"The partials' on-chip traffic is NOT charged: {grid.k_slices:,} passes over the "
            f"whole output cross on-chip memory in each direction, and the v1 machine model has "
            f"no on-chip bandwidth term to price them against (D5a/D5b). What IS charged is the "
            f"capacity — {grid.accumulator_elements:,} live accumulators against "
            f"{format_bytes(machine.chip.on_chip_capacity_bytes)} — and past that the placement "
            f"flips to DRAM, discontinuously.",
        ]

    if op.reduction_placement is ReductionPlacement.DRAM and not grid.materialises_partials:
        return [
            f"Reduction: DRAM, by capacity. The {grid.accumulator_elements:,} live accumulators "
            f"do not fit {format_bytes(machine.chip.on_chip_capacity_bytes)} of on-chip memory, "
            f"so the partials cannot stay there: "
            f"{format_bytes(op.dram_reduction_bytes)} of round trip, "
            f"{format_time(op.t_reduce_s)} of adds on {engine}, and a second dispatch — "
            f"serialised behind the matrix work rather than overlapped with it, since nothing can "
            f"be summed until the slice that feeds it has been written. This is a CLIFF: one "
            f"element of output less and the same decomposition would have been ON_CHIP.",
            "A real compiler would re-block the output instead and re-read A and B, which is the "
            "traffic docs/MODEL.md 6.2 already declines to model. Charging the spill while that "
            "stands prices one horn of the dilemma and not the other, so read this as an upper "
            "bound on the reduction and the re-blocked alternative as the lower one (D62).",
        ]
    return []


def bound_of(report: Report, phase: GraphPhase) -> Bound | None:
    """Convenience for tests and the CLI."""
    result = report.phase(phase)
    return result.bound if result else None
