"""The tile schedule behind the roofline.

The property that matters throughout: the trace is a *decomposition* of numbers
the report already published, so it can never disagree with them. Every test
below is a form of that statement.
"""

from __future__ import annotations

import math

import pytest

from bwz.analysis import analyze, idealised, machine_model
from bwz.analysis.pipeline import Lane, PipelineTrace, Stage, build_trace, tile_count
from bwz.analysis.roofline import MATRIX_OP_TYPES, MachineModel
from bwz.graph import GraphPhase, build_graph
from bwz.report import Report
from bwz.spec import DeploymentSpec, DType, MatmulSpec, load_chip, load_model


def _spec(m: int, n: int, k: int, dtype: str = "fp16") -> MatmulSpec:
    return MatmulSpec.model_validate(
        {
            "id": "t",
            "name": "t",
            "family": "matmul",
            "m": m,
            "n": n,
            "k": k,
            "a_dtype": dtype,
            "b_dtype": dtype,
        }
    )


def _deployment() -> DeploymentSpec:
    return DeploymentSpec.model_validate({"batch": 1, "input_tokens": 1, "output_tokens": 1})


def _trace(
    spec: MatmulSpec, chip_id: str, *, ideal: bool = True, max_steps: int = 64
) -> tuple[PipelineTrace, Report]:
    chip = load_chip(chip_id)
    if ideal:
        chip = idealised(chip)
    deployment = _deployment()
    report = analyze(spec, chip, deployment)
    assert report.feasible, report.infeasibility
    graph = build_graph(spec, deployment, GraphPhase.STATIC)
    machine: MachineModel = machine_model(chip, spec.operand_dtype)
    return (
        build_trace(
            graph,
            report.phases[0],
            machine,
            double_buffered=report.memory.double_buffered,
            max_steps=max_steps,
        ),
        report,
    )


def test_tile_count_is_the_utilisation_decomposition() -> None:
    """A 4096x4096 B on a 16x16 array is 256 x 256 = 65 536 tiles.

    The same product ``systolic_utilisation`` divides by, which is what keeps the
    picture and the utilisation figure from telling different stories.
    """
    machine = machine_model(load_chip("a100_80gb"), DType.FP16)
    graph = build_graph(_spec(1, 4096, 4096), _deployment(), GraphPhase.STATIC)

    assert tile_count(graph.ops[0], machine) == 256 * 256


def test_lane_spans_sum_back_to_the_reported_terms() -> None:
    """DRAM busy is t_dram, core busy is t_compute + t_fixed. No new cost."""
    trace, report = _trace(_spec(4096, 4096, 4096, "int8"), "chip_b", ideal=True)
    op = report.phases[0].ops[0]
    busy = trace.busy_s

    assert busy[Lane.DRAM] == pytest.approx(op.t_dram_s, rel=1e-9)
    assert op.dram_read_bytes + op.dram_write_bytes == pytest.approx(op.dram_bytes)
    assert busy[Lane.CORE] == pytest.approx(op.t_compute_s + op.t_fixed_s, rel=1e-9)


def test_double_buffering_holds_exactly_two_tiles() -> None:
    """SRAM occupancy is two tiles deep, never more.

    The buffer being freed is what bounds it: a tile cannot be fetched until the
    one two steps back has been consumed. Without that constraint the schedule
    would quietly assume infinite capacity.
    """
    trace, _ = _trace(_spec(4096, 4096, 4096, "int8"), "chip_b")
    assert trace.double_buffered

    holds = [s for s in trace.spans if s.stage is Stage.HOLD]
    edges = [(s.start_s, 1) for s in holds] + [(s.end_s, -1) for s in holds]
    edges.sort()
    depth = concurrent = 0
    for _at, delta in edges:
        concurrent += delta
        depth = max(depth, concurrent)

    assert depth == 2


def test_totals_match_the_report() -> None:
    """The quantities on the figure are the report's, not a recount.

    A picture that says "39.9 MB moved" while the report says something else is
    worse than no picture, so the bars carry the report's own bytes and
    operations (D19).
    """
    trace, report = _trace(_spec(10_000, 10_000, 10_000), "a100_80gb")
    op = report.phases[0].ops[0]
    totals = trace.totals

    assert totals[Lane.DRAM] == pytest.approx(op.dram_bytes, rel=1e-9)
    assert totals[Lane.CORE] == pytest.approx(op.flops, rel=1e-9)
    # SRAM is a stock, not a flow: what the buffers hold at once, bounded by the
    # capacity the planner granted.
    assert 0 < totals[Lane.SRAM] <= load_chip("a100_80gb").on_chip_capacity_bytes


def test_concurrency_reads_as_a_depth_not_a_duty_cycle() -> None:
    """SRAM occupancy is ~2 because two buffers are held, not because a resource
    was busy 198% of the time.

    DRAM and the array are single serial resources: their spans never overlap, so
    their mean concurrency is a duty cycle and cannot exceed 1. SRAM is *n*
    buffers, so the same ratio counts buffers. Reporting the second as a
    percentage is what made a correct schedule look broken.
    """
    trace, _ = _trace(_spec(10_000, 10_000, 10_000), "a100_80gb")
    concurrency = trace.concurrency

    dram_mean, dram_peak = concurrency[Lane.DRAM]
    core_mean, core_peak = concurrency[Lane.CORE]
    sram_mean, sram_peak = concurrency[Lane.SRAM]

    assert dram_peak == core_peak == 1
    assert dram_mean <= 1.0 and core_mean <= 1.0
    assert sram_peak == 2
    assert 1.9 < sram_mean <= 2.0


def test_fill_drain_is_what_the_roofline_omits() -> None:
    """The schedule costs more than `max(load, compute)` — a load at the head and
    a store at the tail that nothing overlaps.

    The roofline reports the max alone, which is the many-tiles limit. Here it is
    a fraction of a percent; on a decode projection, which is a handful of tiles
    rather than thousands, it is not.
    """
    trace, report = _trace(_spec(10_000, 10_000, 10_000), "a100_80gb")
    op = report.phases[0].ops[0]

    assert op.t_dram_s > 0, "pick a shape that actually touches DRAM"
    assert trace.total_s > trace.reported_latency_s
    assert trace.fill_drain_s == pytest.approx(trace.total_s - trace.reported_latency_s, rel=1e-9)


def test_operation_trace_reproduces_the_reported_latency_exactly() -> None:
    """A network's steps do not pipeline against each other in this model (D5a),
    so the picture must not be faster than the report it illustrates."""
    chip = load_chip("a100_80gb")
    model = load_model("llama3_8b")
    deployment = DeploymentSpec.model_validate(
        {"batch": 1, "input_tokens": 512, "output_tokens": 1}
    )
    report = analyze(model, chip, deployment)
    decode = report.phase(GraphPhase.DECODE)
    assert decode is not None

    trace = build_trace(
        build_graph(model, deployment, GraphPhase.DECODE),
        decode,
        machine_model(chip, DType.FP16),
        double_buffered=report.memory.double_buffered,
    )

    assert trace.fill_drain_s == 0.0
    assert trace.total_s == pytest.approx(decode.latency_s, rel=1e-9)


def test_each_engines_spans_name_only_families_that_engine_can_run() -> None:
    """A matrix family cannot execute on the vector unit, and vice versa.

    D28 split a coalesced group's compute *time* between the two engines but
    named both halves after the group's dominant family — which is a matmul in
    essentially every group of a transformer. The vector lane's spans therefore
    carried ``op_type="matmul"``, and the timeline's hover read
    ``EXEC — matmul on the vector unit``: a claim that contradicts D27, on the
    one lane D27 exists to separate out.

    Llama-3-8B decode on A100 coalesces 451 operations into 32 blocks, and every
    block mixes matmul with norms and elementwise work, so this asserts over a
    case where the naive answer is wrong on every step.
    """
    chip = load_chip("a100_80gb")
    model = load_model("llama3_8b")
    deployment = DeploymentSpec.model_validate(
        {"batch": 1, "input_tokens": 512, "output_tokens": 1}
    )
    report = analyze(model, chip, deployment)
    decode = report.phase(GraphPhase.DECODE)
    assert decode is not None
    trace = build_trace(
        build_graph(model, deployment, GraphPhase.DECODE),
        decode,
        machine_model(chip, DType.FP16),
        double_buffered=report.memory.double_buffered,
    )

    matrix = {t.value for t in MATRIX_OP_TYPES}
    core = {s.op_type for s in trace.spans if s.lane is Lane.CORE and s.stage is Stage.EXEC}
    vector = {s.op_type for s in trace.spans if s.lane is Lane.VECTOR and s.stage is Stage.EXEC}

    assert core and vector, "this workload must exercise both engines for the test to mean anything"
    assert core <= matrix, f"array span named a non-matrix family: {sorted(core - matrix)}"
    assert not (vector & matrix), f"vector span named a matrix family: {sorted(vector & matrix)}"

    # The split is a relabelling, not a re-costing: the two lanes still sum to
    # the phase's compute time and arithmetic (D19).
    busy = trace.busy_s
    totals = trace.totals
    assert busy[Lane.CORE] + busy[Lane.VECTOR] == pytest.approx(
        decode.t_compute_s + decode.t_fixed_s, rel=1e-9
    )
    assert totals[Lane.CORE] + totals[Lane.VECTOR] == pytest.approx(decode.flops, rel=1e-9)


def test_coalescing_preserves_the_span() -> None:
    """Drawing 64 rows instead of every step changes the picture's resolution and
    not its length.

    Since D30 a step is a **wave**, not a tile: 625x625 = 390 625 tiles over
    A100's 432 tensor cores is ``ceil(390625/432)`` = 905 waves, and that is the
    schedule's real step count.
    """
    fine, _ = _trace(_spec(10_000, 10_000, 10_000), "a100_80gb", max_steps=256)
    coarse, _ = _trace(_spec(10_000, 10_000, 10_000), "a100_80gb", max_steps=8)

    assert fine.tiles == coarse.tiles == math.ceil(625 * 625 / 432) == 905
    assert fine.steps == 256
    assert coarse.steps == 8
    # Coarser steps mean a larger fill/drain, because fill/drain is one step.
    assert coarse.total_s > fine.total_s


def test_loads_are_split_by_operand_and_reconcile_with_the_report() -> None:
    """B and A cross the bus separately, and the two bars sum to the one number.

    They obey different residency fractions and spill at different times, so a
    single LOAD figure cannot say which operand moved. At 8192-cubed INT8 on
    Metis both are now 100% non-resident: B does not fit the 16 resident tiles,
    and A is compulsory traffic read exactly once (D33) — the asymmetry D31
    documented is gone, but the reconciliation it exists to test remains.
    """
    chip = idealised(load_chip("metis_aipu"))
    trace, report = _trace(_spec(8192, 8192, 8192, "int8"), "metis_aipu", max_steps=256)
    op = report.phases[0].ops[0]

    weights, activations = trace.operand_bytes
    reads, writes = trace.direction_bytes

    assert weights == pytest.approx(op.dram_weight_read_bytes, rel=1e-9)
    assert activations == pytest.approx(op.dram_activation_read_bytes, rel=1e-9)
    assert weights + activations == pytest.approx(op.dram_read_bytes, rel=1e-9)
    assert reads == pytest.approx(op.dram_read_bytes, rel=1e-9)
    assert writes == pytest.approx(op.dram_write_bytes, rel=1e-9)
    # B does not fit the array and A is read exactly once: both operands are
    # full compulsory traffic now (D33), where A used to get a residency
    # discount — the 2x gap in D31's numbers is gone.
    assert weights == pytest.approx(activations, rel=1e-9)
    assert weights > 0
    assert chip.on_chip_capacity_bytes > 0


def test_a_is_staged_once_per_k_slice_in_the_trace() -> None:
    """D33: the single-matmul trace does not show A streaming every wave.

    8192-cubed INT8 on Metis tiles to ceil(K/512) = 16 k-slices, and the DRAM
    lane carries exactly one LOAD_A event per k-slice — the whole staging, fed
    once to every tile of the group — instead of a per-wave share that reads as
    a re-read.
    """
    trace, report = _trace(_spec(8192, 8192, 8192, "int8"), "metis_aipu", max_steps=64)
    op = report.phases[0].ops[0]
    a_spans = [s for s in trace.spans if s.stage is Stage.LOAD_A]

    assert len(a_spans) == 16, "one staging event per k-slice, ceil(8192/512)"
    assert all(s.staged_once for s in a_spans)
    expected = op.dram_activation_read_bytes / 16
    assert all(s.bytes_moved == pytest.approx(expected, rel=1e-9) for s in a_spans)
    assert sum(s.bytes_moved for s in a_spans) == pytest.approx(
        op.dram_activation_read_bytes, rel=1e-9
    )
    assert all("A k-slice" in s.label for s in a_spans)

    # The non-matmul path is untouched: a network's activations still stream
    # (inter-op reuse, no k-slice staging to draw).
    chip = load_chip("a100_80gb")
    model = load_model("llama3_8b")
    deployment = DeploymentSpec.model_validate(
        {"batch": 1, "input_tokens": 512, "output_tokens": 1}
    )
    report = analyze(model, chip, deployment)
    decode = report.phase(GraphPhase.DECODE)
    assert decode is not None
    trace_net = build_trace(
        build_graph(model, deployment, GraphPhase.DECODE),
        decode,
        machine_model(chip, DType.FP16),
        double_buffered=report.memory.double_buffered,
    )
    assert not any(s.staged_once for s in trace_net.spans)


def test_a_tile_step_is_a_wave_not_a_single_tile() -> None:
    """The bars must not show a 4-core chip working one tile at a time (D30).

    Metis has four 512x512 arrays. An 8192-cubed INT8 matmul is
    ceil(8192/512)^2 = 256 tiles, which is 64 waves of 4 — and the trace draws
    64 steps, not 256, with each step carrying four tiles' worth of arithmetic.
    Drawing one bar per tile said the arrays ran in series, contradicting the
    wave occupancy the same report quotes.
    """
    trace, report = _trace(_spec(8192, 8192, 8192, "int8"), "metis_aipu", max_steps=256)

    assert trace.tiles == 64, "steps are waves: ceil(256 tiles / 4 arrays)"
    assert not trace.coalesced, "64 waves fit under the drawing cap, so nothing is merged"
    assert "4 B tiles 512x512" in trace.spans[0].label
    assert trace.spans[0].label.endswith("all in parallel")

    # And it stays a decomposition: the waves still sum to the reported terms.
    op = report.phases[0].ops[0]
    assert trace.busy_s[Lane.CORE] == pytest.approx(op.t_compute_s + op.t_fixed_s, rel=1e-9)

    # When one bar has to coalesce several waves, the label must stop claiming
    # they are simultaneous and say how many run at a time instead.
    coarse, _ = _trace(_spec(8192, 8192, 8192, "int8"), "metis_aipu", max_steps=8)
    assert coarse.coalesced
    assert coarse.spans[0].label.endswith("4 at a time")
    assert "in parallel" not in coarse.spans[0].label


def test_fill_drain_differs_enough_between_chips_to_distort_a_ratio() -> None:
    """Why the comparison headline must be quoted from the reported latency.

    D33's k-slice staging concentrates A into a few large events instead of a
    per-wave trickle, so much less of it overlaps compute. On an 8192-cubed INT8
    matmul the pipeline fill/drain is a rounding error on A100 and 17.7% of the
    latency on Metis — so a ratio measured off the drawn bars reads 3.93x where
    the report says 3.34x, an 18% error in the one number a comparison figure
    exists to show (D35).

    This asserts the *engine* side of that: the gap is real and asymmetric. The
    figure's job is to quote `reported_latency_s`, which `_subtitle` now does.
    """
    fast, fast_report = _trace(_spec(8192, 8192, 8192, "int8"), "a100_80gb", max_steps=256)
    slow, slow_report = _trace(_spec(8192, 8192, 8192, "int8"), "metis_aipu", max_steps=256)

    def share(trace: PipelineTrace) -> float:
        return trace.fill_drain_s / trace.reported_latency_s

    assert share(fast) < 0.01, "A100 overlaps almost everything"
    assert share(slow) > 0.10, "Metis stages A in lumps, so much less overlaps"

    reported = slow.reported_latency_s / fast.reported_latency_s
    drawn = slow.total_s / fast.total_s
    assert reported == pytest.approx(3.34, rel=0.02)
    assert drawn == pytest.approx(3.93, rel=0.02)
    assert drawn > reported * 1.15, "the two disagree by enough to mislead"

    # Whichever is quoted, the spans still decompose the reported terms (D19).
    for trace, report in ((fast, fast_report), (slow, slow_report)):
        op = report.phases[0].ops[0]
        assert trace.busy_s[Lane.DRAM] == pytest.approx(op.t_dram_s, rel=1e-9)
        assert trace.reported_latency_s == pytest.approx(op.latency_s, rel=1e-9)
