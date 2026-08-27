"""Every number the docs quote, re-derived from the engine.

``docs/CLI.md`` promises "real output, not a description" and CLAUDE.md requires
it of any command doc. That promise decays silently: a correction moves a
figure, the tests that pin the *code* all pass, and the prose keeps quoting a
number the engine stopped producing. It happened — D52 and D53 between them
invalidated output blocks in four files, including a README table of MLPerf
error bars that no test had ever computed.

So the docs get a test. Each case below names the file and section it defends,
and asserts the **formatted** string, because that is what is on the page —
comparing floats would pass while the page said something else.

This is deliberately not a golden-file diff of the docs. The point is that a
change which moves a documented number **fails here**, so the author has to
decide whether the number or the prose is wrong, rather than finding out from a
reader months later.
"""

from __future__ import annotations

import pytest

from bwz.analysis import analyze, idealised
from bwz.kernels import matmul_kernel
from bwz.report import Report
from bwz.spec import DeploymentSpec, DType, MatmulSpec, load_chip, load_model
from bwz.units import format_bytes, format_time

MATMUL_DEPLOYMENT = DeploymentSpec.model_validate(
    {"batch": 1, "input_tokens": 1, "output_tokens": 0, "phase": "prefill"}
)


def _matmul(
    m: int,
    n: int,
    k: int,
    *,
    chip: str = "a100_80gb",
    dtype: DType = DType.FP16,
    out_dtype: DType | None = None,
    b_dtype: DType | None = None,
    ideal: bool = True,
    **deployment: object,
) -> tuple[Report, MatmulSpec]:
    hardware = load_chip(chip)
    if ideal:
        hardware = idealised(hardware)
    spec = matmul_kernel(m, n, k, a_dtype=dtype, b_dtype=b_dtype or dtype, out_dtype=out_dtype)
    payload = {**MATMUL_DEPLOYMENT.model_dump(mode="json"), **deployment}
    report = analyze(spec, hardware, DeploymentSpec.model_validate(payload))
    assert report.feasible, report.infeasibility
    return report, spec


def test_cli_2_1_the_datasheet_check() -> None:
    """``docs/CLI.md`` §2.1 and the README's `bwz matmul` block.

    The whole point of this one is that it is checkable by hand: 2*10000^3 = 2.000e12
    OP at A100's 312 TOP/s is 6.41 ms, and the 0.09% on top is wave occupancy —
    every dimension is a multiple of 16, so no padding is lost, but 390 625 tiles
    over 432 tensor cores is 905 waves whose last is 89% full.
    """
    report, _ = _matmul(10_000, 10_000, 10_000)
    op = report.phases[0].ops[0]

    assert format_bytes(op.dram_read_bytes) == "400 MB"
    assert format_bytes(op.dram_write_bytes) == "200 MB"
    assert f"{op.arithmetic_intensity:.1f}" == "3333.3"
    assert f"{op.utilization:.2%}" == "99.91%"
    assert format_time(op.t_dram_s) == "294 µs"
    assert format_time(op.latency_s) == "6.42 ms"
    assert f"{report.flip_margins[0].margin:.1f}x" == "21.8x"

    # "Drop --ideal and the same command gives 9.17 ms: the difference is /0.70."
    derated, _ = _matmul(10_000, 10_000, 10_000, ideal=False)
    assert format_time(derated.phases[0].ops[0].latency_s) == "9.17 ms"


def test_cli_2_2_shape_utilisation_at_m_equals_one() -> None:
    """``docs/CLI.md`` §2.2, and the README's "drop -M to 1" sentence.

    The documented decomposition of the 4.52% is the part that matters, because
    the two halves live at different levels and only the first is what CLAUDE.md's
    ``M=1 -> ~1/rows`` sanity check is about: 1/16 of the array (one row of work
    pays for a whole 16-row instruction tile) times 625/(2*432) of the chip (only
    625 output tiles exist, so 432 cores take two waves and the second is 45%
    full).
    """
    report, _ = _matmul(1, 10_000, 10_000)
    op = report.phases[0].ops[0]

    assert f"{op.utilization:.2%}" == "4.52%"
    assert format_bytes(op.dram_read_bytes) == "139 MB"
    assert format_bytes(op.dram_write_bytes) == "20 kB"
    assert format_time(op.latency_s) == "68.3 µs"
    assert op.bound.value == "DRAM_BW_BOUND"

    array_term = 1 / 16
    chip_term = 625 / (2 * 432)
    assert op.utilization == pytest.approx(array_term * chip_term, rel=1e-9)


@pytest.mark.parametrize(
    ("a", "b", "out", "expected"),
    [
        (DType.INT8, DType.INT8, None, ("16.8 MB", "16.8 MB", "2730.7", "221 µs")),
        (DType.INT8, DType.INT8, DType.INT32, ("33.6 MB", "67.1 MB", "1365.3", "221 µs")),
        (DType.FP16, DType.FP16, DType.FP32, ("67.1 MB", "67.1 MB", "1024.0", "442 µs")),
        (DType.FP16, DType.INT8, None, ("50.3 MB", "33.6 MB", "1638.4", "442 µs")),
    ],
    ids=["int8", "int8-int32", "fp16-fp32", "mixed"],
)
def test_cli_2_3_the_width_table(
    a: DType, b: DType, out: DType | None, expected: tuple[str, str, str, str]
) -> None:
    """``docs/CLI.md`` §2.3's table, row by row. M=N=K=4096 throughout.

    Two rules it exists to show, both still visible in the numbers: the result
    width changes bytes only and never operations (rows 1 and 2 have identical
    latency), and a mixed matmul runs at the *wider* operand (row 4 matches row 3's
    latency while moving fewer bytes, and is 2x rows 1-2, which do the same
    arithmetic at twice the rate).
    """
    report, _ = _matmul(4096, 4096, 4096, dtype=a, b_dtype=b, out_dtype=out)
    op = report.phases[0].ops[0]
    got = (
        format_bytes(op.dram_read_bytes),
        format_bytes(op.dram_write_bytes),
        f"{op.arithmetic_intensity:.1f}",
        format_time(op.latency_s),
    )
    assert got == expected


def test_cli_2_4_the_schedule_steps_through_waves_not_tiles() -> None:
    """``docs/CLI.md`` §2.4: "152 is waves, not tiles".

    The grid is ceil(4096/16)^2 = 65 536 output tiles; A100 has 432 tensor cores,
    so the schedule steps through ceil(65536/432) = 152 waves of them. Drawing one
    bar per tile would show a 432-core chip working through tiles in series (D30).
    """
    from bwz.analysis import build_trace, machine_model
    from bwz.analysis.pipeline import grid_of
    from bwz.graph import GraphPhase, build_graph

    report, spec = _matmul(4096, 4096, 4096, ideal=False)
    machine = machine_model(load_chip("a100_80gb"), spec.operand_dtype)
    graph = build_graph(spec, MATMUL_DEPLOYMENT, GraphPhase.STATIC)
    grid = grid_of(graph.ops[0], machine)
    assert grid is not None and grid.tiles == 65_536

    trace = build_trace(
        graph, report.phases[0], machine, double_buffered=report.memory.double_buffered
    )
    assert trace.tiles == 152, "the trace's step count is waves"


@pytest.mark.parametrize(
    ("splits", "util", "dram", "latency", "waves"),
    [(1, "79.01%", "4.72 MB", "8.71 µs", 3), (8, "99.81%", "13.1 MB", "6.99 µs", 19)],
    ids=["default", "split-k-8"],
)
def test_cli_2_5_1_the_split_k_trade(
    splits: int, util: str, dram: str, latency: str, waves: int
) -> None:
    """``docs/CLI.md`` §2.5.1 and the README's split-K table.

    The trade the flag exists to expose: 512x512x4096 on A100 has only 32x32 =
    1024 output tiles against 432 tensor cores, so occupancy is 79%. Cutting K in
    eight gives 8192 tiles at 99.8% — and writes eight full 512x512 partials to
    DRAM to be read back by CUTLASS's second kernel, which is the 8.4 MB
    difference (D53).
    """
    report, _ = _matmul(512, 512, 4096, split_k=splits)
    op = report.phases[0].ops[0]

    assert f"{op.utilization:.2%}" == util
    assert format_bytes(op.dram_bytes) == dram
    assert format_time(op.latency_s) == latency
    assert -(-(32 * 32 * splits) // 432) == waves

    if splits > 1:
        # The whole difference is the partials' round trip, nothing else.
        plain, _ = _matmul(512, 512, 4096)
        extra = op.dram_bytes - plain.phases[0].ops[0].dram_bytes
        assert extra == pytest.approx(op.dram_reduction_bytes)
        assert op.dram_reduction_bytes == pytest.approx(splits * 512 * 512 * 2.0 * 2.0)


def test_readme_headline_run() -> None:
    """The README's `bwz run` block — the first output a reader ever sees.

    It was a mock-up for a long time: it quoted an energy figure for a model that
    has none (M7), a bottleneck table in a format the CLI never printed, and a
    confidence of "medium" the engine cannot return before Session 5. Replaced
    with the real thing, and pinned here so it stays the real thing.
    """
    deployment = DeploymentSpec.model_validate(
        {"batch": 1, "input_tokens": 2048, "output_tokens": 256}
    )
    report = analyze(load_model("llama3_8b"), load_chip("h100_sxm"), deployment)
    assert report.feasible
    by_phase = {phase.phase.value: phase for phase in report.phases}
    prefill, decode = by_phase["prefill"], by_phase["decode"]
    summary = report.summary
    assert summary is not None

    assert format_time(prefill.latency_s) == "44.6 ms"
    assert f"{prefill.utilization:.2%}" == "67.26%"
    assert prefill.bound.value == "COMPUTE_BOUND"
    assert format_time(decode.latency_s) == "6.05 ms"
    assert f"{decode.utilization:.2%}" == "0.27%"
    assert decode.bound.value == "DRAM_BW_BOUND"
    assert f"{1 / decode.latency_s:.1f}" == "165.4"
    assert format_time(summary.latency_s) == "1.59 s"
    assert format_bytes(report.memory.weight_bytes) == "16.1 GB"
    assert format_bytes(report.memory.kv_cache_bytes) == "302 MB"
    assert report.confidence.value == "low", "the README says so, and says why"


@pytest.mark.parametrize(
    ("m", "intensity", "util", "latency", "margin"),
    [
        (10_000, "3333.3", "99.91%", "6.42 ms", "21.8"),
        (512, "464.4", "98.50%", "333 µs", "3.8"),
        (1, "1.0", "4.52%", "68.3 µs", "4.8"),
    ],
    ids=["compute-bound", "still-compute-bound", "dram-bound"],
)
def test_model_5b_the_same_chip_three_machines(
    m: int, intensity: str, util: str, latency: str, margin: str
) -> None:
    """``docs/MODEL.md`` §5b's shape table. N = K = 10000, A100, fp16, --ideal.

    The point of the table is the last row: the arithmetic fell by 10 000x against
    the first and the traffic did not fall at all, so the workload crossed the
    ridge. Same chip, three different machines.
    """
    report, _ = _matmul(m, 10_000, 10_000)
    op = report.phases[0].ops[0]

    assert f"{op.arithmetic_intensity:.1f}" == intensity
    assert f"{op.utilization:.2%}" == util
    assert format_time(op.latency_s) == latency
    assert f"{report.flip_margins[0].margin:.1f}" == margin


def test_model_6_7_what_the_model_reproduces() -> None:
    """``docs/MODEL.md`` §6.7's table of CLAUDE.md sanity checks."""
    h100 = load_chip("h100_sxm")

    long_run = DeploymentSpec.model_validate(
        {"batch": 1, "input_tokens": 2048, "output_tokens": 128}
    )
    report = analyze(load_model("llama3_8b"), h100, long_run)
    by_phase = {phase.phase.value: phase for phase in report.phases}
    assert by_phase["decode"].bound.value == "DRAM_BW_BOUND"
    assert f"{1 / by_phase['decode'].latency_s:.0f}" == "165"
    assert f"{by_phase['decode'].utilization:.2%}" == "0.27%"
    assert f"{by_phase['prefill'].utilization:.1%}" == "67.3%"

    batched = DeploymentSpec.model_validate(
        {"batch": 128, "input_tokens": 2048, "output_tokens": 128}
    )
    decode = {
        phase.phase.value: phase for phase in analyze(load_model("llama3_8b"), h100, batched).phases
    }["decode"]
    assert decode.bound.value == "COMPUTE_BOUND"
    assert f"{decode.utilization:.0%}" == "22%", "batch amortises the weight traffic"

    small = DeploymentSpec.model_validate({"batch": 1, "input_tokens": 1, "output_tokens": 1})
    assert analyze(load_model("mobilenetv3"), h100, small).phases[0].bound.value == "LATENCY_BOUND"


def test_readme_compare_figures() -> None:
    """The README's and ``docs/CLI.md`` §6.1's head-to-head numbers.

    Both chips run the same workload at the same precision — enforced, since A100
    defaults to fp16 and Metis has no fp16 datapath — so the ratio is a statement
    about the machines rather than about two different amounts of traffic.
    """
    int8 = DeploymentSpec.model_validate(
        {
            "batch": 1,
            "input_tokens": 512,
            "output_tokens": 1,
            "precision": {"weights": "int8", "activations": "int8", "kv_cache": "int8"},
        }
    )
    prefill = {}
    for chip_id in ("a100_80gb", "metis_aipu"):
        report = analyze(load_model("gemma3_4b"), load_chip(chip_id), int8)
        prefill[chip_id] = next(p for p in report.phases if p.phase.value == "prefill")

    assert format_time(prefill["a100_80gb"].latency_s) == "8.95 ms"
    assert format_time(prefill["metis_aipu"].latency_s) == "119 ms"
    ratio = prefill["metis_aipu"].latency_s / prefill["a100_80gb"].latency_s
    assert f"{ratio:.2f}x" == "13.26x"

    # The 8192-cubed INT8 matmul the band headers quote.
    achieved = {}
    for chip_id in ("a100_80gb", "metis_aipu"):
        report, _ = _matmul(8192, 8192, 8192, chip=chip_id, dtype=DType.INT8, ideal=False)
        summary = report.summary
        assert summary is not None
        achieved[chip_id] = (
            f"{summary.achieved_flops_per_s / 1e12:.0f} TOP/s",
            f"{summary.utilization:.0%}",
        )
    assert achieved["a100_80gb"] == ("436 TOP/s", "70%")
    assert achieved["metis_aipu"] == ("186 TOP/s", "89%")


def test_the_validation_suite_is_empty_and_the_docs_say_so() -> None:
    """``docs/CALIBRATION.md`` and the README's Accuracy section both state that
    no reference point has been collected. This asserts the state they describe.

    If someone adds a real reference point, this test fails — and the two
    documents claiming "none collected" have to be updated in the same commit.
    That is the intent: the claim and the fact move together.

    CLAUDE.md is explicit that fabricated validation data would make the project
    worthless, and the README carried an MLPerf error table for months that no
    test had ever computed. An empty table is the correct output until it isn't.
    """
    from pathlib import Path

    validation = Path(__file__).resolve().parents[1] / "validation"
    collected = sorted(p.name for p in validation.glob("*.yaml")) if validation.is_dir() else []
    assert collected == [], (
        f"reference points appeared ({collected}) — update docs/CALIBRATION.md and the "
        f"README's Accuracy section, which both say none have been collected"
    )
