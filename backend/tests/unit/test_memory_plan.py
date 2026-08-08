"""Capacity planning: residency, the allocation order, and infeasibility fixes."""

from __future__ import annotations

import pytest

from bwz.analysis import analyze
from bwz.analysis.memory import plan_memory
from bwz.graph import GraphPhase, build_graph
from bwz.report import MemoryPlan
from bwz.spec import DeploymentSpec, load_chip, load_model

INT8 = {"weights": "int8", "activations": "int8", "kv_cache": "int8"}


def _plan(model_id: str, chip_id: str, **overrides: object) -> MemoryPlan:
    document: dict[str, object] = {
        "batch": 1,
        "input_tokens": 1,
        "output_tokens": 1,
        "kv_context_tokens": 4096,
        "precision": INT8,
    }
    document.update(overrides)
    deployment = DeploymentSpec.model_validate(document)
    model = load_model(model_id)
    graph = build_graph(model, deployment, GraphPhase.DECODE)
    return plan_memory(graph, load_chip(chip_id), deployment)


@pytest.mark.parametrize(
    ("model_id", "chip_id", "expected", "d8_figure"),
    [
        ("gemma3_4b", "chip_a", 0.0140, "1.4%"),
        ("gemma3_preset_2b", "chip_a", 0.0273, "2.7%"),
        ("gemma3_preset_1b", "chip_a", 0.0568, "5.5%"),
        ("gemma3_4b", "chip_b", 0.2575, "25.6%"),
        ("gemma3_preset_2b", "chip_b", 0.5015, "50%"),
        ("gemma3_preset_1b", "chip_b", 1.0, "100%"),
    ],
)
def test_residency_matches_the_d8_figures(
    model_id: str, chip_id: str, expected: float, d8_figure: str
) -> None:
    """docs/CORRECTIONS.md D8's residency table, reproduced to within 0.4 points.

    ``r = available_on_chip / W`` where ``available`` is what remains after the
    double buffer and the activation working set are served. At decode the
    activation set is kilobytes, so the D8 form ``sram/W`` is recovered almost
    exactly.
    """
    plan = _plan(model_id, chip_id)
    assert plan.resident_fraction == pytest.approx(expected, abs=0.004), d8_figure


def test_activations_are_served_before_weights() -> None:
    """On-chip capacity goes where it saves the most DRAM traffic.

    At prefill on chip_a the activation working set is ~15 MB against 3.88 GB of
    weights. Spending 15 MB on activations removes ~2 GB of traffic; spending the
    same 15 MB on weight residency removes 15 MB. Serving weights first would
    inflate TTFT by 58%.
    """
    plan = _plan("gemma3_4b", "chip_a", input_tokens=512, phase="prefill", output_tokens=0)
    assert plan.activation_resident_fraction == 1.0
    assert 0.0 < plan.resident_fraction < 0.02


def test_the_double_buffer_is_reserved_before_residency() -> None:
    """chip_b's 1 GB SRAM would otherwise be 100% claimed by a 3.88 GB model's
    residency, leaving no staging room and forcing loads to serialise."""
    plan = _plan("gemma3_4b", "chip_b")
    assert plan.double_buffered is True
    assert plan.resident_fraction < 1.0


def test_a_fully_resident_model_still_double_buffers() -> None:
    plan = _plan("gemma3_preset_1b", "chip_b")
    assert plan.resident_fraction == 1.0
    assert plan.double_buffered is True


def test_residency_is_zero_when_there_is_no_on_chip_memory() -> None:
    """A single-level chip reports no on-chip capacity, so nothing is resident."""
    chip = load_chip("chip_a")
    single_level = chip.model_copy(update={"memory": [chip.memory[-1]]})
    graph = build_graph(
        load_model("gemma3_4b"),
        DeploymentSpec.model_validate({"batch": 1, "input_tokens": 1, "output_tokens": 1}),
        GraphPhase.DECODE,
    )
    plan = plan_memory(
        graph,
        single_level,
        DeploymentSpec.model_validate({"batch": 1, "input_tokens": 1, "output_tokens": 1}),
    )
    assert plan.resident_fraction == 0.0


# -- feasibility ------------------------------------------------------------


def test_a_model_that_does_not_fit_reports_reasons_not_an_exception() -> None:
    """CLAUDE.md #8: infeasible returns a Report, never raises."""
    report = analyze(
        load_model("llama2_70b"),
        load_chip("jetson_orin"),
        DeploymentSpec(batch=1, input_tokens=2048, output_tokens=256),
    )
    assert report.feasible is False
    assert report.summary is None
    assert report.infeasibility
    joined = " ".join(report.infeasibility)
    assert "Over by" in joined
    assert "halve the weight precision" in joined
    assert "shard across" in joined


def test_infeasibility_names_the_actual_numbers() -> None:
    report = analyze(
        load_model("gpt3"),
        load_chip("h100_sxm"),
        DeploymentSpec(batch=1, input_tokens=2048, output_tokens=256),
    )
    assert report.feasible is False
    first = report.infeasibility[0]
    assert "349 GB" in first or "GB" in first
    assert "NVIDIA H100 SXM5 80GB" in first


def test_an_unsupported_dtype_is_infeasible_with_the_allowed_set() -> None:
    """chip_a is INT8/INT4 only. Asking for fp16 must say so and list what works."""
    report = analyze(
        load_model("gemma3_4b"),
        load_chip("chip_a"),
        DeploymentSpec(batch=1, input_tokens=512, output_tokens=1),
    )
    assert report.feasible is False
    assert "no compute unit for 'fp16'" in report.infeasibility[0]
    assert "int8" in report.infeasibility[0]


def test_quantising_makes_an_infeasible_config_fit() -> None:
    """The cheapest fix, actually applied."""
    fp16 = DeploymentSpec(batch=1, input_tokens=2048, output_tokens=256)
    int8 = DeploymentSpec.model_validate(
        {"batch": 1, "input_tokens": 2048, "output_tokens": 256, "precision": INT8}
    )
    model, chip = load_model("llama2_70b"), load_chip("h100_sxm")
    assert analyze(model, chip, fp16).feasible is False
    assert analyze(model, chip, int8).feasible is True
