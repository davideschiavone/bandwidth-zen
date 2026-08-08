"""End-to-end: the CLAUDE.md sanity checks, the D8 demo, and the property tests.

These are the M3 definition of done. Every expected value is either a CLAUDE.md
sanity check quoted in the docstring, a figure from ``docs/CORRECTIONS.md`` D8,
or derived from the inputs in the docstring.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from bwz.analysis import analyze
from bwz.analysis.compare import decode_vs_bandwidth, head_to_head, prefill_crossover
from bwz.graph import GraphPhase
from bwz.report import Bound, Confidence, PhaseResult, Report, Summary
from bwz.spec import DeploymentSpec, DType, HardwareSpec, load_chip, load_model

INT8 = {"weights": "int8", "activations": "int8", "kv_cache": "int8"}


def _dep(**overrides: object) -> DeploymentSpec:
    document: dict[str, object] = {"batch": 1, "input_tokens": 2048, "output_tokens": 256}
    document.update(overrides)
    return DeploymentSpec.model_validate(document)


def _run(model_id: str, chip_id: str, **overrides: object) -> Report:
    return analyze(load_model(model_id), load_chip(chip_id), _dep(**overrides))


def _summary(report: Report) -> Summary:
    """Narrow away the ``| None`` that only an infeasible report carries."""
    assert report.summary is not None, "expected a feasible report"
    return report.summary


def _phase(report: Report, phase: GraphPhase) -> PhaseResult:
    result = report.phase(phase)
    assert result is not None, f"expected a {phase.value} phase"
    return result


def _need(result: PhaseResult | None) -> PhaseResult:
    """Narrow a phase lookup written inline on a call expression."""
    assert result is not None, "expected a phase"
    return result


def _with_bandwidth(chip: HardwareSpec, bandwidth: float) -> HardwareSpec:
    levels = list(chip.memory)
    levels[-1] = levels[-1].model_copy(update={"bandwidth_bytes_per_s": bandwidth})
    return chip.model_copy(update={"memory": levels})


# -- CLAUDE.md sanity checks ------------------------------------------------


def test_llama3_8b_decode_on_h100_is_dram_bound() -> None:
    """ "Llama-3-8B, fp16, batch 1, H100, decode -> memory-bound, roughly 16 GB of
    weights moved per token."

    Both halves hold: DRAM_BW_BOUND, and 16.1 GB of weights. The rate that
    follows is **165 tok/s**, not the 35-55 the same CLAUDE.md line predicts:
    16 GB at 3.35 TB/s x 0.85 achieved is 5.6 ms. See docs/CORRECTIONS.md D12 --
    the traffic figure is right and the rate that was paired with it is not.
    """
    report = _run("llama3_8b", "h100_sxm")
    decode = _phase(report, GraphPhase.DECODE)
    assert decode.bound is Bound.DRAM_BW_BOUND
    assert report.memory.weight_bytes == pytest.approx(16.1e9, rel=0.02)
    assert _summary(report).tokens_per_s == pytest.approx(165, rel=0.15)


def test_llama3_8b_decode_utilisation_is_far_below_one_percent() -> None:
    """ "A model that shows 80% utilization at batch 1 is broken." This one shows 0.27%."""
    decode = _need(_run("llama3_8b", "h100_sxm").phase(GraphPhase.DECODE))
    assert decode.utilization < 0.01


def test_llama3_8b_prefill_at_2k_is_compute_bound() -> None:
    """ "Llama-3-8B prefill, 2048 tokens -> compute-bound, utilization 40-70%." 67.7%."""
    prefill = _need(_run("llama3_8b", "h100_sxm").phase(GraphPhase.PREFILL))
    assert prefill.bound is Bound.COMPUTE_BOUND
    assert 0.40 <= prefill.utilization <= 0.70


def test_batching_raises_utilisation_sharply() -> None:
    """ "Same model, batch 128, decode -> utilization rises sharply."

    0.27% to 21%, and the bound flips from DRAM to compute once each weight read
    serves 128 tokens instead of one.
    """
    one = _need(_run("llama3_8b", "h100_sxm", batch=1).phase(GraphPhase.DECODE))
    many = _need(_run("llama3_8b", "h100_sxm", batch=128).phase(GraphPhase.DECODE))
    assert many.utilization > 50 * one.utilization
    assert many.bound is Bound.COMPUTE_BOUND


def test_gemma3_4b_batch_1_has_single_digit_utilisation() -> None:
    """ "Gemma-4, batch 1, H100 -> latency/launch-bound, single-digit % utilization.
    Batch 128 -> good utilization."

    The utilisation claim holds (0.24%, then 19.5%). The *label* does not: at
    fp16 a 3.88 G model moves 7.8 GB per token, so t_dram = 2.7 ms against
    t_fixed = 0.82 ms and the verdict is DRAM-bound. Latency is the runner-up,
    beating compute. See docs/CORRECTIONS.md D13 -- the launch-bound regime is
    real but needs a far smaller model, which is why MobileNetV3 lands in it.
    """
    one = _need(_run("gemma3_4b", "h100_sxm", batch=1).phase(GraphPhase.DECODE))
    many = _need(_run("gemma3_4b", "h100_sxm", batch=128).phase(GraphPhase.DECODE))
    assert one.utilization < 0.09
    assert many.utilization > 0.15
    assert one.bound is Bound.DRAM_BW_BOUND
    assert one.t_fixed_s > one.t_compute_s, "latency beats compute, just not DRAM"


def test_mobilenetv3_on_h100_is_latency_bound() -> None:
    """A 5.4 M CNN on a 989 TFLOP/s chip is pure dispatch overhead.

    This is the regime CLAUDE.md's Gemma line describes; it takes a model three
    orders of magnitude smaller to reach it.
    """
    static = _phase(
        _run("mobilenetv3", "h100_sxm", phase="prefill", output_tokens=0), GraphPhase.STATIC
    )
    assert static.bound is Bound.LATENCY_BOUND
    assert static.utilization < 0.01


def test_depthwise_layers_are_the_memory_bound_ones_on_a_big_array() -> None:
    """ "MobileNetV3 depthwise layers -> memory-bound, poor utilization on a large
    systolic array." On chip_a's 512x512 array they bottom out below 1%."""
    report = _run("mobilenetv3", "chip_a", phase="prefill", output_tokens=0, precision=INT8)
    static = _phase(report, GraphPhase.STATIC)
    depthwise = [
        op for op in static.ops if op.op_id.endswith(".depthwise") and op.op_type.value == "conv"
    ]
    assert depthwise
    assert max(op.utilization for op in depthwise) < 0.05


# -- the D8 acceptance demo -------------------------------------------------


@pytest.mark.parametrize(
    ("model_id", "chip_id", "expected_tok_s", "d8"),
    [
        ("gemma3_4b", "chip_a", 8.4, 8.8),
        ("gemma3_preset_2b", "chip_a", 16.8, 17.5),
        ("gemma3_preset_1b", "chip_a", 37.0, 36.0),
    ],
)
def test_d8_decode_rates_on_chip_a(
    model_id: str, chip_id: str, expected_tok_s: float, d8: float
) -> None:
    """docs/CORRECTIONS.md D8's decode table for chip_a, reproduced within 5%.

    chip_a has four times chip_b's compute, so its array never binds and the
    supplied weight-traffic formula is recovered directly. The residual against
    D8 is the KV correction from D9: real ``kv_width`` is 1024, not 512, so the
    KV term is twice what D8 assumed.
    """
    report = _run(
        model_id,
        chip_id,
        input_tokens=512,
        output_tokens=1,
        kv_context_tokens=4096,
        precision=INT8,
        phase="decode",
    )
    assert _summary(report).tokens_per_s == pytest.approx(expected_tok_s, rel=0.05)
    assert _summary(report).tokens_per_s == pytest.approx(d8, rel=0.12)
    assert _phase(report, GraphPhase.DECODE).bound is Bound.DRAM_BW_BOUND


def test_d8_prefill_ttft_on_chip_a() -> None:
    """D8: TTFT at S=512 is ~113 ms for the 4B and ~27.9 ms for the 1B, DRAM-bound.

    Reproduced at 114.6 ms and 27.2 ms.
    """
    four_b = _need(
        _run(
            "gemma3_4b",
            "chip_a",
            input_tokens=512,
            output_tokens=0,
            phase="prefill",
            precision=INT8,
        ).phase(GraphPhase.PREFILL)
    )
    one_b = _need(
        _run(
            "gemma3_preset_1b",
            "chip_a",
            input_tokens=512,
            output_tokens=0,
            phase="prefill",
            precision=INT8,
        ).phase(GraphPhase.PREFILL)
    )
    assert four_b.latency_s == pytest.approx(0.113, rel=0.05)
    assert one_b.latency_s == pytest.approx(0.0279, rel=0.05)
    assert four_b.bound is Bound.DRAM_BW_BOUND


def test_chip_b_is_compute_bound_at_batch_one_contra_d8() -> None:
    """The central M3 finding, and it inverts D8's conclusion.

    D8 computes ``t_compute = 2*params/tops``, which assumes the 512x512 array
    is fully utilised. At batch 1 it runs at 1/513 of peak, so chip_b's quarter
    of chip_a's TOPS becomes the binding term and its 18x SRAM advantage largely
    stops paying: 9.3 tok/s against chip_a's 8.4, not the 11.7 against 8.8 that
    the unutilised model predicts.
    """
    chip_a = _run(
        "gemma3_4b",
        "chip_a",
        input_tokens=512,
        output_tokens=1,
        kv_context_tokens=4096,
        precision=INT8,
        phase="decode",
    )
    chip_b = _run(
        "gemma3_4b",
        "chip_b",
        input_tokens=512,
        output_tokens=1,
        kv_context_tokens=4096,
        precision=INT8,
        phase="decode",
    )
    assert _phase(chip_b, GraphPhase.DECODE).bound is Bound.COMPUTE_BOUND
    assert _phase(chip_a, GraphPhase.DECODE).bound is Bound.DRAM_BW_BOUND
    fast, slow = _summary(chip_b).tokens_per_s, _summary(chip_a).tokens_per_s
    assert fast is not None and slow is not None
    ratio = fast / slow
    assert 1.0 < ratio < 1.3, "chip_b still wins, but by ~11% not 33%"


def test_d8_ridge_points() -> None:
    """6165 and 1541 OP/byte. Both chips set efficiency to 1.0, so the effective
    ridge equals the nominal one (docs/CORRECTIONS.md D6)."""
    from bwz.analysis import machine_model

    assert machine_model(load_chip("chip_a"), DType.INT8).ridge_point == pytest.approx(
        6165, rel=0.01
    )
    assert machine_model(load_chip("chip_b"), DType.INT8).ridge_point == pytest.approx(
        1541, rel=0.01
    )


def test_head_to_head_covers_every_pair() -> None:
    models = [load_model(m) for m in ("gemma3_preset_1b", "gemma3_preset_2b", "gemma3_4b")]
    chips = [load_chip(c) for c in ("chip_a", "chip_b")]
    rows = head_to_head(models, chips, _dep(input_tokens=512, output_tokens=1, precision=INT8))
    assert len(rows) == 6
    assert all(row.feasible for row in rows)
    by_model = {(r.model_id, r.chip_id): r for r in rows}
    assert by_model[("gemma3_preset_1b", "chip_b")].resident_fraction == 1.0


def test_prefill_crossover_is_found_or_honestly_absent() -> None:
    """Bisection over S. When one chip wins throughout, that is reported as such
    rather than as a fabricated crossing point."""
    point = prefill_crossover(
        load_model("gemma3_4b"),
        load_chip("chip_a"),
        load_chip("chip_b"),
        _dep(input_tokens=512, output_tokens=0, phase="prefill", precision=INT8),
        lo=1,
        hi=100_000,
    )
    assert point.faster_below in ("chip_a", "chip_b")
    if point.tokens is not None:
        assert 1 <= point.tokens <= 100_000


def test_decode_scales_linearly_with_bandwidth() -> None:
    """D8's ``decode_vs_bandwidth`` must come out ~linear while DRAM binds."""
    points = decode_vs_bandwidth(
        load_model("gemma3_4b"),
        load_chip("chip_a"),
        _dep(input_tokens=1, output_tokens=1, kv_context_tokens=4096, precision=INT8),
        [1.7e10, 3.4e10, 6.8e10],
    )
    ratios = [points[i + 1][1] / points[i][1] for i in range(len(points) - 1)]
    assert all(1.8 < ratio <= 2.0 for ratio in ratios), ratios


# -- report contract --------------------------------------------------------


def test_every_report_carries_assumptions_and_flip_margins() -> None:
    """CLAUDE.md #4: the honesty mechanism is mandatory, not optional."""
    report = _run("llama3_8b", "h100_sxm")
    assert len(report.assumptions) >= 8
    assert report.flip_margins
    assert all(margin.margin > 0 for margin in report.flip_margins)
    assert any("not fitted" in a or "documented defaults" in a for a in report.assumptions)


def test_confidence_is_never_high_before_calibration() -> None:
    """Session 5 has not run. Claiming high confidence would be the one failure
    mode this project cannot survive."""
    for chip_id in ("h100_sxm", "a100_80gb", "mi300x", "chip_a", "chip_b"):
        report = _run("gemma3_4b", chip_id, precision=INT8)
        assert report.confidence is not Confidence.HIGH


def test_a_hypothetical_chip_says_so_in_its_assumptions() -> None:
    report = _run("gemma3_4b", "chip_b", precision=INT8)
    assert any("hypothetical profile" in a for a in report.assumptions)
    assert report.confidence is Confidence.LOW


def test_analysis_is_deterministic() -> None:
    """No wall clock, no global state: same inputs, identical report (CLAUDE.md #3)."""
    first = _run("llama3_8b", "h100_sxm")
    second = _run("llama3_8b", "h100_sxm")
    assert first.to_json() == second.to_json()
    assert first.meta.generated_at is None, "analyze() must not stamp time"


def test_the_per_op_table_is_available_for_the_d8_layer_report() -> None:
    report = _run(
        "gemma3_4b",
        "chip_a",
        input_tokens=1,
        output_tokens=1,
        kv_context_tokens=4096,
        precision=INT8,
        phase="decode",
    )
    decode = _phase(report, GraphPhase.DECODE)
    layer0 = [op for op in decode.ops if op.layer == 0 and op.op_type.value == "matmul"]
    assert len(layer0) == 7
    assert sum(op.weight_bytes for op in layer0) == pytest.approx(94.4e6, rel=0.01)


# -- property tests ---------------------------------------------------------


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(bandwidth=st.floats(min_value=5e10, max_value=8e12))
def test_more_bandwidth_never_costs_latency(bandwidth: float) -> None:
    """CLAUDE.md: "Doubling DRAM bandwidth never increases predicted latency"."""
    chip = load_chip("h100_sxm")
    model, deployment = load_model("llama3_8b"), _dep(phase="decode")
    slow = analyze(model, _with_bandwidth(chip, bandwidth), deployment)
    fast = analyze(model, _with_bandwidth(chip, bandwidth * 2), deployment)
    assert _summary(fast).latency_s <= _summary(slow).latency_s * 1.000001


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(batch=st.integers(min_value=1, max_value=64))
def test_throughput_never_falls_as_batch_grows(batch: int) -> None:
    """Per-token cost is monotonically non-increasing in batch: the same weight
    read serves more tokens."""
    model, chip = load_model("llama3_8b"), load_chip("h100_sxm")
    small = analyze(model, chip, _dep(batch=batch, phase="decode"))
    large = analyze(model, chip, _dep(batch=batch * 2, phase="decode"))
    if not (small.feasible and large.feasible):
        return
    fast, slow = _summary(large).tokens_per_s, _summary(small).tokens_per_s
    assert fast is not None and slow is not None
    assert fast >= slow * 0.999999


def test_int8_is_never_slower_than_fp16() -> None:
    """CLAUDE.md: "INT8 is never slower than FP16 on hardware that supports both"."""
    model = load_model("llama3_8b")
    for chip_id in ("h100_sxm", "a100_80gb", "mi300x"):
        chip = load_chip(chip_id)
        fp16 = analyze(model, chip, _dep(phase="decode"))
        int8 = analyze(model, chip, _dep(phase="decode", precision=INT8))
        assert _summary(int8).latency_s <= _summary(fp16).latency_s


def test_a_longer_context_never_speeds_decode_up() -> None:
    """KV traffic grows with context; nothing shrinks."""
    model, chip = load_model("llama3_8b"), load_chip("h100_sxm")
    previous = 0.0
    for context in (512, 2048, 8192):
        report = analyze(
            model, chip, _dep(phase="decode", input_tokens=8, kv_context_tokens=context)
        )
        tpot = _summary(report).tpot_s
        assert tpot is not None and tpot >= previous
        previous = tpot
