"""The one-op GEMM workload: a hand-checkable probe of the roofline.

Every expected value here is written out arithmetically in the docstring, because
the entire purpose of this family is that its numbers can be verified against a
datasheet without trusting the engine (CLAUDE.md #7).
"""

from __future__ import annotations

import pytest

from bwz.analysis import analyze, idealised
from bwz.graph import GraphPhase, build_graph
from bwz.graph.ops import MatmulAttrs, OpType
from bwz.report import Bound, OpResult, Report
from bwz.spec import DeploymentSpec, GemmSpec, load_chip


def _spec(m: int, n: int, k: int) -> GemmSpec:
    return GemmSpec.model_validate(
        {"id": "t", "name": "t", "family": "gemm", "m": m, "n": n, "k": k}
    )


def _deployment(dtype: str = "fp16") -> DeploymentSpec:
    return DeploymentSpec.model_validate(
        {
            "batch": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "precision": {"weights": dtype, "activations": dtype, "kv_cache": dtype},
        }
    )


def _only_op(report: Report) -> OpResult:
    assert report.feasible, report.infeasibility
    assert len(report.phases) == 1
    ops = report.phases[0].ops
    assert len(ops) == 1
    return ops[0]


def test_graph_is_one_matmul_with_three_operands() -> None:
    """M=8, N=4, K=2: weights [2,4], input [8,2], output [8,4]."""
    graph = build_graph(_spec(8, 4, 2), _deployment(), GraphPhase.STATIC)

    assert len(graph.ops) == 1
    op = graph.ops[0]
    assert op.op_type is OpType.MATMUL
    assert op.attrs == MatmulAttrs(m=8, n=4, k=2)
    assert graph.tensors["gemm.w"].shape == (2, 4)
    assert graph.tensors["gemm.x"].shape == (8, 2)
    assert graph.tensors["gemm.out"].shape == (8, 4)


def test_parameter_count_is_the_weight_operand() -> None:
    """K x N = 2 x 4 = 8. The activation is not a parameter."""
    assert _spec(8, 4, 2).parameter_count() == 8


def test_flops_are_two_mnk() -> None:
    """2 x 10000^3 = 2.0e12."""
    assert _spec(10_000, 10_000, 10_000).flops == pytest.approx(2.0e12)


def test_a100_10k_cube_matches_the_datasheet_by_hand() -> None:
    """The worked example the roofline chart is drawn from.

    arithmetic   = 2 x 10000^3           = 2.000e12 OP
    compulsory   = 3 x 10000^2 x 2 bytes = 6.000e8 bytes  (fp16, three operands)
    intensity    = 2.000e12 / 6.000e8    = 3333 OP/byte
    peak fp16    = 312.0e12 OP/s         (A100 datasheet, tensor core)
    t_compute    = 2.000e12 / (312.0e12 x 0.9984) = 6.42 ms

    The 0.9984 is the systolic tail on a 16x16 array at M=10000
    (10000/10016), which --ideal does not remove because it is geometry.
    An intensity of 3333 against a ridge point of 153 is compute-bound by
    more than 20x, which no plausible calibration constant can flip.
    """
    report = analyze(
        _spec(10_000, 10_000, 10_000), idealised(load_chip("a100_80gb")), _deployment()
    )
    op = _only_op(report)

    assert op.flops == pytest.approx(2.0e12)
    assert op.arithmetic_intensity == pytest.approx(3333.3, rel=1e-3)
    assert op.utilization == pytest.approx(10_000 / 10_016, rel=1e-6)
    assert op.t_compute_s == pytest.approx(6.42e-3, rel=1e-2)
    assert op.bound is Bound.COMPUTE_BOUND


def test_m_equals_one_reproduces_the_tail_effect() -> None:
    """A single row on a 16-row array wastes 15 of them: 1/(1+16) = 5.88%.

    This is the sanity check CLAUDE.md names for the utilisation model, and the
    reason a GEMM is a family rather than a hand-costed custom op — a CustomOp
    carries no shape, so this number would silently come back as 100%.
    """
    report = analyze(_spec(1, 10_000, 10_000), idealised(load_chip("a100_80gb")), _deployment())
    op = _only_op(report)

    assert op.utilization == pytest.approx(1 / 17, rel=1e-6)
    assert op.arithmetic_intensity == pytest.approx(1.0, rel=1e-2)
    assert op.bound is Bound.DRAM_BW_BOUND


def test_int8_is_never_slower_than_fp16() -> None:
    """CLAUDE.md sanity check, on the one workload where nothing else moves."""
    chip = load_chip("a100_80gb")
    spec = _spec(4096, 4096, 4096)
    fp16 = _only_op(analyze(spec, chip, _deployment("fp16")))
    int8 = _only_op(analyze(spec, chip, _deployment("int8")))

    assert int8.latency_s <= fp16.latency_s


def test_ideal_removes_only_the_deratings() -> None:
    """--ideal divides t_compute by 0.70 and t_dram by 0.85, and leaves the
    shape utilisation exactly where it was."""
    spec = _spec(4096, 4096, 4096)
    real = _only_op(analyze(spec, load_chip("a100_80gb"), _deployment()))
    ideal = _only_op(analyze(spec, idealised(load_chip("a100_80gb")), _deployment()))

    assert ideal.utilization == pytest.approx(real.utilization)
    assert ideal.t_compute_s == pytest.approx(real.t_compute_s * 0.70, rel=1e-6)
    assert ideal.t_dram_s == pytest.approx(real.t_dram_s * 0.85, rel=1e-6)
