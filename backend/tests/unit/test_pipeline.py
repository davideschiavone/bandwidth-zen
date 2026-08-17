"""The tile schedule behind the roofline, and its Kanata serialisation.

The property that matters throughout: the trace is a *decomposition* of numbers
the report already published, so it can never disagree with them. Every test
below is a form of that statement.
"""

from __future__ import annotations

import pytest

from bwz.analysis import analyze, idealised, machine_model
from bwz.analysis.pipeline import Lane, PipelineTrace, Stage, build_trace, tile_count
from bwz.analysis.roofline import MachineModel
from bwz.graph import GraphPhase, build_graph
from bwz.kanata import to_kanata
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


def test_fill_drain_is_one_step_of_the_non_binding_resource() -> None:
    """total = max(t_dram, t_compute) + min(t_dram, t_compute)/steps.

    The roofline reports the max alone, so it is the steps -> infinity limit. On
    a 10000^3 fp16 matmul over 64 drawn steps the omission is a fraction of a
    percent; it is the *form* that matters, because a decode projection is a
    handful of tiles rather than thousands.
    """
    trace, report = _trace(_spec(10_000, 10_000, 10_000), "a100_80gb")
    op = report.phases[0].ops[0]

    assert op.t_dram_s > 0, "pick a shape that actually touches DRAM"
    expected = min(op.t_dram_s, op.t_compute_s) / trace.steps
    assert trace.fill_drain_s == pytest.approx(expected, rel=1e-9)
    assert trace.total_s == pytest.approx(
        max(op.t_dram_s, op.t_compute_s) + expected + op.t_fixed_s, rel=1e-9
    )
    assert trace.total_s > trace.reported_latency_s


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


def test_coalescing_preserves_the_span() -> None:
    """Drawing 64 rows instead of 390 625 changes the picture's resolution and
    not its length."""
    fine, _ = _trace(_spec(10_000, 10_000, 10_000), "a100_80gb", max_steps=256)
    coarse, _ = _trace(_spec(10_000, 10_000, 10_000), "a100_80gb", max_steps=8)

    assert fine.tiles == coarse.tiles == 625 * 625
    assert fine.steps == 256
    assert coarse.steps == 8
    # Coarser steps mean a larger fill/drain, because fill/drain is one step.
    assert coarse.total_s > fine.total_s


def test_kanata_is_well_formed() -> None:
    """Structural invariants of the log: monotone ticks, every row opened before
    it is used and retired after, every stage started before it ends."""
    trace, _ = _trace(_spec(4096, 4096, 4096, "int8"), "chip_b", max_steps=16)
    text = to_kanata(trace, title="test", resolution=500)
    lines = [line for line in text.splitlines() if line and not line.startswith("//")]

    assert lines[0] == "Kanata\t0004"
    assert lines[1] == "C=\t0"

    open_rows: set[str] = set()
    retired: set[str] = set()
    active: set[tuple[str, str]] = set()
    cycle = 0
    for line in lines[2:]:
        parts = line.split("\t")
        kind = parts[0]
        if kind == "C":
            step = int(parts[1])
            assert step > 0
            cycle += step
        elif kind == "I":
            assert parts[1] not in open_rows
            open_rows.add(parts[1])
        elif kind == "L":
            assert parts[1] in open_rows
        elif kind == "S":
            assert parts[1] in open_rows and parts[1] not in retired
            active.add((parts[1], parts[3]))
        elif kind == "E":
            assert (parts[1], parts[3]) in active
            active.discard((parts[1], parts[3]))
        elif kind == "R":
            assert parts[1] in open_rows
            retired.add(parts[1])

    assert open_rows == retired
    assert not active, f"stages never ended: {active}"
    assert cycle > 0
