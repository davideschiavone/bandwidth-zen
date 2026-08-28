"""A's residency and B's write timing as command-line strategies.

``docs/CORRECTIONS.md`` D36, ``docs/MODEL.md`` §6.3a. Every expected value here is
hand-computed in the docstring or comment above the assertion (CLAUDE.md #7):
these are the goldens the acceptance criteria for D36 names explicitly, so they
are written first and everything else is judged against them.
"""

from __future__ import annotations

import math

import pytest

from bwz.analysis import analyze, machine_model, trace_phases
from bwz.analysis.dataflow import DataflowPlan, plan_dataflow
from bwz.analysis.pipeline import Lane, tile_count, tiles_per_a_event
from bwz.graph import GraphPhase, build_graph
from bwz.operators.base import cost_of
from bwz.report import Report
from bwz.spec import DeploymentSpec, MatmulSpec, load_chip


def _spec(m: int, n: int, k: int, dtype: str = "int8") -> MatmulSpec:
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


def _plan(
    chip_id: str, spec: MatmulSpec, deployment: DeploymentSpec
) -> tuple[DataflowPlan, Report]:
    """The report plus the same ``DataflowPlan`` ``analyze()`` used to cost it."""
    chip = load_chip(chip_id)
    machine = machine_model(chip, spec.operand_dtype)
    graph = build_graph(spec, deployment, GraphPhase.STATIC)
    cost = cost_of(graph.ops[0], graph.tensors)
    plan = plan_dataflow(graph.ops[0], machine, chip, deployment, a_bytes=cost.input_bytes)
    report = analyze(spec, chip, deployment)
    return plan, report


# 8192-cubed INT8 on Metis: NTILES_PER_KS = ceil(8192/512) = 16, |A| = 8192*8192 = 67 108 864 B.
METIS_8192_A_BYTES = 8192 * 8192
METIS_8192_NTILES_PER_KS = 16


def test_defaults_reproduce_stage_write_ahead_exactly() -> None:
    """D33's own numbers, with the strategy flags left at their defaults.

    A crosses DRAM exactly once: a_bytes_multiplier = NTILES_PER_KS/NTILES_PER_KS = 1.0, so
    dram_activation_read_bytes == |A| == 67 108 864 B exactly.
    """
    spec = _spec(8192, 8192, 8192)
    deployment = DeploymentSpec.model_validate({"batch": 1, "input_tokens": 1, "output_tokens": 1})
    plan, report = _plan("metis_aipu", spec, deployment)

    assert plan.a_strategy.value == "stage"
    assert plan.b_dataflow.value == "write-ahead"
    assert plan.a_bytes_multiplier == pytest.approx(1.0)
    assert plan.notes == ()

    op = report.phases[0].ops[0]
    assert op.dram_activation_read_bytes == pytest.approx(METIS_8192_A_BYTES, rel=1e-9)


def test_stream_multiplies_a_by_ntiles_per_ks() -> None:
    """D31: streamed A is NTILES_PER_KS x the staged total.

    67 108 864 B * 16 = 1 073 741 824 B = 1.07 GB (to 3 s.f.), matching the D36 correction note.
    """
    spec = _spec(8192, 8192, 8192)
    deployment = DeploymentSpec.model_validate(
        {"batch": 1, "input_tokens": 1, "output_tokens": 1, "a_strategy": "stream"}
    )
    plan, report = _plan("metis_aipu", spec, deployment)

    assert plan.a_strategy.value == "stream"
    assert plan.residency_tiles == 1
    assert plan.tiles_per_a_event == METIS_8192_NTILES_PER_KS
    assert plan.a_bytes_multiplier == pytest.approx(METIS_8192_NTILES_PER_KS)

    op = report.phases[0].ops[0]
    expected = METIS_8192_A_BYTES * METIS_8192_NTILES_PER_KS
    assert expected == 1_073_741_824
    assert op.dram_activation_read_bytes == pytest.approx(expected, rel=1e-9)

    # 256 fetch events summing to it: waves * units tiles, each streaming its own
    # share — the same nested loop deploy.py's listing draws for `stream`.
    chip = load_chip("metis_aipu")
    machine = machine_model(chip, spec.operand_dtype)
    graph = build_graph(spec, deployment, GraphPhase.STATIC)
    tiles = tile_count(graph.ops[0], machine)
    assert tiles == 256
    per_event = expected / tiles
    assert per_event * tiles == pytest.approx(expected)


def test_whole_matches_stage_bytes_and_ramps_the_schedule() -> None:
    """D33: `whole` moves the same bytes as `stage` — only the timing changes.

    A100 4096-cubed fp16: A = 4096*4096*2 B = 33 554 432 B fits comfortably in A100's on-chip
    capacity, so `whole` is not clamped and its ramp actually fires.
    """
    spec = _spec(4096, 4096, 4096, dtype="fp16")
    stage_dep = DeploymentSpec.model_validate({"batch": 1, "input_tokens": 1, "output_tokens": 1})
    whole_dep = DeploymentSpec.model_validate(
        {"batch": 1, "input_tokens": 1, "output_tokens": 1, "a_strategy": "whole"}
    )
    stage_plan, stage_report = _plan("a100_80gb", spec, stage_dep)
    whole_plan, whole_report = _plan("a100_80gb", spec, whole_dep)

    assert whole_plan.a_strategy.value == "whole"
    assert whole_plan.notes == (), "A must fit A100's on-chip capacity for this shape"

    stage_op = stage_report.phases[0].ops[0]
    whole_op = whole_report.phases[0].ops[0]
    assert whole_op.dram_activation_read_bytes == pytest.approx(stage_op.dram_activation_read_bytes)
    assert whole_report.phases[0].latency_s == pytest.approx(stage_report.phases[0].latency_s)

    # The reported latency (D35) is untouched; the drawn schedule is not — whole
    # ramps every k-slice in before wave 0, so it can only add fill/drain, never remove it.
    chip = load_chip("a100_80gb")
    machine = machine_model(chip, spec.operand_dtype)
    graph = build_graph(spec, whole_dep, GraphPhase.STATIC)
    from bwz.analysis.pipeline import build_trace

    stage_graph = build_graph(spec, stage_dep, GraphPhase.STATIC)
    stage_trace = build_trace(
        stage_graph,
        stage_report.phases[0],
        machine,
        double_buffered=stage_report.memory.double_buffered,
        dataflow=stage_plan,
    )
    whole_trace = build_trace(
        graph,
        whole_report.phases[0],
        machine,
        double_buffered=whole_report.memory.double_buffered,
        dataflow=whole_plan,
    )
    assert whole_trace.reported_latency_s == pytest.approx(stage_trace.reported_latency_s)
    assert whole_trace.fill_drain_s >= stage_trace.fill_drain_s
    ramp = [s for s in whole_trace.spans if s.a_fetch_mode == "whole"]
    assert len(ramp) == 1
    assert ramp[0].start_s == 0.0
    assert ramp[0].bytes_moved == pytest.approx(whole_op.dram_activation_read_bytes)


def test_whole_clamps_to_stage_when_a_does_not_fit_the_scratchpad() -> None:
    """8192-cubed on Metis: A is 67.1 MB against a 54.5 MB scratchpad — the exact shape
    the D36 correction note names as hitting this fallback."""
    spec = _spec(8192, 8192, 8192)
    deployment = DeploymentSpec.model_validate(
        {"batch": 1, "input_tokens": 1, "output_tokens": 1, "a_strategy": "whole"}
    )
    plan, _report = _plan("metis_aipu", spec, deployment)

    assert plan.a_requested.value == "whole"
    assert plan.a_strategy.value == "stage", "falls back: A does not fit on chip"
    assert len(plan.notes) == 1
    assert "a_strategy=whole" in plan.notes[0]
    assert "fell back to stage" in plan.notes[0]


def test_persistent_clamps_to_write_ahead_when_b_does_not_fit() -> None:
    """Same shape, the B side: 256 tiles against 16 resident (D30's own numbers)."""
    spec = _spec(8192, 8192, 8192)
    deployment = DeploymentSpec.model_validate(
        {"batch": 1, "input_tokens": 1, "output_tokens": 1, "b_dataflow": "persistent"}
    )
    plan, _report = _plan("metis_aipu", spec, deployment)

    assert plan.b_requested.value == "persistent"
    assert plan.b_dataflow.value == "write-ahead"
    assert plan.tiles == 256
    assert plan.resident_tile_capacity == 16
    assert any("fell back to write-ahead" in note for note in plan.notes)


def test_persistent_amortises_b_over_iterations_when_it_fits() -> None:
    """A shape where B genuinely fits the array (16 tiles) and the generic SRAM
    residency discount is forced to zero by an oversized M, isolating the
    persistent/iterations effect from `plan_memory`'s own residency fraction.

    B = 2048*2048*1 B = 4 194 304 B. iterations=1: the first (only) invocation
    writes it in full. iterations=4: amortised to 4 194 304 / 4 = 1 048 576 B.
    """
    spec = _spec(100_000, 2048, 2048)
    chip = load_chip("metis_aipu")

    dep1 = DeploymentSpec.model_validate(
        {
            "batch": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "b_dataflow": "persistent",
            "iterations": 1,
        }
    )
    dep4 = DeploymentSpec.model_validate(
        {
            "batch": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "b_dataflow": "persistent",
            "iterations": 4,
        }
    )
    report1 = analyze(spec, chip, dep1)
    report4 = analyze(spec, chip, dep4)
    machine = machine_model(chip, spec.operand_dtype)
    graph = build_graph(spec, dep1, GraphPhase.STATIC)
    plan1 = plan_dataflow(
        graph.ops[0], machine, chip, dep1, a_bytes=cost_of(graph.ops[0], graph.tensors).input_bytes
    )
    plan4 = plan_dataflow(
        graph.ops[0], machine, chip, dep4, a_bytes=cost_of(graph.ops[0], graph.tensors).input_bytes
    )

    assert plan1.tiles == 16
    assert plan1.resident_tile_capacity == 16
    assert plan1.notes == (), "fits: no fallback"
    assert plan1.b_write_multiplier == pytest.approx(1.0)
    assert plan4.b_write_multiplier == pytest.approx(0.25)

    full_write = 2048 * 2048
    assert full_write == 4_194_304
    op1 = report1.phases[0].ops[0]
    op4 = report4.phases[0].ops[0]
    assert op1.dram_weight_read_bytes == pytest.approx(full_write, rel=1e-6)
    assert op4.dram_weight_read_bytes == pytest.approx(full_write / 4, rel=1e-6)


def test_b_dataflow_never_moves_a_byte_within_one_pass() -> None:
    """D30, proven again here: write-ahead, on-demand and persistent(fits) all
    charge identical bytes and latency at iterations=1 — only deploy.py's listing
    tells them apart."""
    spec = _spec(8192, 8192, 8192)
    chip = load_chip("metis_aipu")
    results = {}
    for choice in ("write-ahead", "on-demand", "persistent"):
        deployment = DeploymentSpec.model_validate(
            {"batch": 1, "input_tokens": 1, "output_tokens": 1, "b_dataflow": choice}
        )
        report = analyze(spec, chip, deployment)
        op = report.phases[0].ops[0]
        results[choice] = (report.phases[0].latency_s, op.dram_weight_read_bytes, op.dram_bytes)

    assert results["write-ahead"] == results["on-demand"] == results["persistent"]


def test_a_residency_tiles_clamps_to_the_largest_power_of_two_divisor() -> None:
    """NTILES_PER_KS = 16 here; 5 is not a power-of-2 divisor of 16, so it clamps
    down to 4 (the largest one that is), not up to 8."""
    spec = _spec(8192, 8192, 8192)
    deployment = DeploymentSpec.model_validate(
        {"batch": 1, "input_tokens": 1, "output_tokens": 1, "a_residency_tiles": 5}
    )
    plan, _report = _plan("metis_aipu", spec, deployment)

    assert plan.tiles_per_a_event == 16
    assert plan.residency_tiles == 4
    assert plan.a_bytes_multiplier == pytest.approx(4.0)
    assert any("clamped to 4" in note for note in plan.notes)


@pytest.mark.parametrize("a_strategy", ["stage", "stream", "whole"])
@pytest.mark.parametrize("b_dataflow", ["write-ahead", "on-demand", "persistent"])
def test_d19_invariant_survives_every_strategy(a_strategy: str, b_dataflow: str) -> None:
    """DRAM busy sums to t_dram and core busy to t_compute + t_fixed, whichever
    dataflow strategy is in effect — asserted, not assumed (D19)."""
    spec = _spec(8192, 8192, 2048)
    chip = load_chip("metis_aipu")
    deployment = DeploymentSpec.model_validate(
        {
            "batch": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "a_strategy": a_strategy,
            "b_dataflow": b_dataflow,
        }
    )
    report = analyze(spec, chip, deployment)
    assert report.feasible, report.infeasibility
    op = report.phases[0].ops[0]

    ((_phase, trace),) = trace_phases(spec, chip, deployment)
    busy = trace.busy_s
    assert busy[Lane.DRAM] == pytest.approx(op.t_dram_s, rel=1e-6)
    assert busy[Lane.CORE] == pytest.approx(op.t_compute_s + op.t_fixed_s, rel=1e-6)
    # D35: the schedule may run past the reported latency (fill/drain) but never short of it.
    assert trace.total_s >= trace.reported_latency_s - 1e-9
    assert trace.reported_latency_s == pytest.approx(report.phases[0].latency_s, rel=1e-9)


def test_dataflow_line_names_the_effective_not_requested_strategy() -> None:
    """report.assumptions states what actually ran, per CLAUDE.md #4 — on a shape
    where both requested strategies get clamped, the line must say the fallback."""
    spec = _spec(8192, 8192, 8192)
    chip = load_chip("metis_aipu")
    deployment = DeploymentSpec.model_validate(
        {
            "batch": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "a_strategy": "whole",
            "b_dataflow": "persistent",
        }
    )
    report = analyze(spec, chip, deployment)
    lines = [a for a in report.assumptions if a.startswith("A: ")]
    assert len(lines) == 1
    assert "staged" in lines[0] and "crosses DRAM exactly once" in lines[0]
    assert "write-ahead" in lines[0]
    assert "persistent" not in lines[0].split("·")[1]


def test_the_emitted_program_agrees_with_the_plan_for_every_strategy() -> None:
    """``emit.check`` must hold for every combination — the program cannot walk a
    strategy the schedule above it did not run.

    This was ``deploy.py``'s pseudo-C listing until D54 retired it. The guarantee
    is the same and the artifact is stronger: a program that named the wrong
    strategy would also *move the wrong bytes*, and its own tier-1 assertions
    would fail when run.
    """
    from bwz.analysis.pipeline import build_trace, grid_of
    from bwz.emit import check as check_program
    from bwz.emit import emit_matmul

    spec = _spec(600, 600, 600)
    chip = load_chip("metis_aipu")
    for a_strategy in ("stage", "stream", "whole"):
        for b_dataflow in ("write-ahead", "on-demand", "persistent"):
            deployment = DeploymentSpec.model_validate(
                {
                    "batch": 1,
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "a_strategy": a_strategy,
                    "b_dataflow": b_dataflow,
                }
            )
            report = analyze(spec, chip, deployment)
            machine = machine_model(chip, spec.operand_dtype)
            graph = build_graph(spec, deployment, GraphPhase.STATIC)
            plan = plan_dataflow(
                graph.ops[0],
                machine,
                chip,
                deployment,
                a_bytes=cost_of(graph.ops[0], graph.tensors).input_bytes,
            )
            build_trace(
                graph,
                report.phases[0],
                machine,
                double_buffered=report.memory.double_buffered,
                max_steps=32,
                dataflow=plan,
            )
            grid = grid_of(graph.ops[0], machine)
            assert grid is not None
            program = emit_matmul(
                chip,
                machine,
                grid,
                plan,
                report.phases[0].ops[0],
                a_dtype=spec.a_dtype,
                b_dtype=spec.b_dtype,
                c_dtype=spec.result_dtype,
                acc_dtype=deployment.precision.accumulate,
                double_buffered=report.memory.double_buffered,
                command="bwz matmul (test)",
                version="0.0.0-test",
            )
            check_program(program)
            assert f'A_STRATEGY = "{plan.a_strategy.value}"' in program.source
            assert f'B_DATAFLOW = "{plan.b_dataflow.value}"' in program.source


def test_a_prefetch_depth_is_schedule_only() -> None:
    """`--a-prefetch-depth` changes the drawn schedule, never a byte or the
    reported latency (D35) — and depth=1 (no double buffering) must not crash,
    which it did before ``_pipelined_tiles`` learned to place a depth-1 store
    before reading it rather than after (the deferred issue-order optimisation
    only has slack to defer when a second buffer exists)."""
    from bwz.analysis.pipeline import build_trace

    spec = _spec(4096, 4096, 4096, dtype="fp16")
    chip = load_chip("a100_80gb")
    machine = machine_model(chip, spec.operand_dtype)

    baseline_dep = DeploymentSpec.model_validate(
        {"batch": 1, "input_tokens": 1, "output_tokens": 1}
    )
    baseline_report = analyze(spec, chip, baseline_dep)
    assert baseline_report.memory.double_buffered, "this shape must double-buffer by default"

    results = {}
    for depth in (None, 1, 2, 8):
        overrides = {"batch": 1, "input_tokens": 1, "output_tokens": 1}
        if depth is not None:
            overrides["a_prefetch_depth"] = depth
        deployment = DeploymentSpec.model_validate(overrides)
        report = analyze(spec, chip, deployment)
        graph = build_graph(spec, deployment, GraphPhase.STATIC)
        plan = plan_dataflow(
            graph.ops[0],
            machine,
            chip,
            deployment,
            a_bytes=cost_of(graph.ops[0], graph.tensors).input_bytes,
        )
        trace = build_trace(
            graph,
            report.phases[0],
            machine,
            double_buffered=report.memory.double_buffered,
            dataflow=plan,
        )
        op = report.phases[0].ops[0]
        results[depth] = (op.dram_bytes, trace.reported_latency_s, trace.fill_drain_s)

    # No byte or reported-latency change at any depth (D35).
    bytes_and_latency = {(b, latency) for b, latency, _fd in results.values()}
    assert len(bytes_and_latency) == 1

    # depth=1 (no double buffering) must not crash and must show strictly more
    # fill/drain than depth=2 — less prefetch slack serialises more of the run.
    assert results[1][2] > results[2][2]
    # depth=None (today's behaviour, derived from double buffering) matches an
    # explicit depth=2 exactly on a shape this profile double-buffers.
    assert results[None][2] == pytest.approx(results[2][2])


def test_ntiles_per_kslice_matches_deploy_and_pipeline() -> None:
    """The one divisor every strategy is built from, read the same way everywhere."""
    spec = _spec(8192, 8192, 8192)
    chip = load_chip("metis_aipu")
    machine = machine_model(chip, spec.operand_dtype)
    deployment = DeploymentSpec.model_validate({"batch": 1, "input_tokens": 1, "output_tokens": 1})
    graph = build_graph(spec, deployment, GraphPhase.STATIC)
    assert tiles_per_a_event(graph.ops[0], machine) == math.ceil(8192 / 512) == 16
