"""The one-op matmul workload: a hand-checkable probe of the roofline.

Every expected value here is written out arithmetically in the docstring, because
the entire purpose of this family is that its numbers can be verified against a
datasheet without trusting the engine (CLAUDE.md #7).
"""

from __future__ import annotations

import pytest

from bwz.analysis import analyze, idealised
from bwz.analysis.tiling import systolic_utilisation
from bwz.graph import GraphPhase, build_graph
from bwz.graph.ops import MatmulAttrs, OpType
from bwz.report import Bound, OpResult, Report
from bwz.spec import DeploymentSpec, DType, MatmulSpec, load_chip, load_model


def _spec(m: int, n: int, k: int, dtype: str = "fp16", out: str | None = None) -> MatmulSpec:
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
            "out_dtype": out,
        }
    )


def _deployment() -> DeploymentSpec:
    """A matmul reads nothing from the deployment (D18); analyze() still wants one."""
    return DeploymentSpec.model_validate({"batch": 1, "input_tokens": 1, "output_tokens": 1})


def _only_op(report: Report) -> OpResult:
    assert report.feasible, report.infeasibility
    assert len(report.phases) == 1
    ops = report.phases[0].ops
    assert len(ops) == 1
    return ops[0]


def test_graph_is_one_matmul_with_three_operands() -> None:
    """M=8, N=4, K=2: A [8,2], B [2,4], C [8,4]."""
    graph = build_graph(_spec(8, 4, 2), _deployment(), GraphPhase.STATIC)

    assert len(graph.ops) == 1
    op = graph.ops[0]
    assert op.op_type is OpType.MATMUL
    assert op.attrs == MatmulAttrs(m=8, n=4, k=2)
    assert graph.tensors["matmul.a"].shape == (8, 2)
    assert graph.tensors["matmul.b"].shape == (2, 4)
    assert graph.tensors["matmul.c"].shape == (8, 4)


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
    t_compute    = 2.000e12 / (312.0e12 x 0.9984 x 0.99914) = 6.42 ms

    The 0.9984 is the systolic tail on a 16x16 array at M=10000
    (10000/10016), which --ideal does not remove because it is geometry.
    An intensity of 3333 against a ridge point of 153 is compute-bound by
    more than 20x, which no plausible calibration constant can flip.

    The 0.99914 is wave quantisation across the 432 tensor cores (D30):
    ``ceil(10000/16)^2 = 625^2 = 390 625`` tiles over 432 arrays is 905 waves,
    and 390 625 / (905 x 432) leaves the last wave 78% full. Negligible here by
    construction — a 390 625-tile GEMM fills any chip — and dominant on a small
    one, which is the point of carrying it.
    """
    report = analyze(
        _spec(10_000, 10_000, 10_000), idealised(load_chip("a100_80gb")), _deployment()
    )
    op = _only_op(report)
    tail, waves = 10_000 / 10_016, 390_625 / (905 * 432)

    assert op.flops == pytest.approx(2.0e12)
    assert op.arithmetic_intensity == pytest.approx(3333.3, rel=1e-3)
    assert op.utilization == pytest.approx(tail * waves, rel=1e-6)
    assert waves == pytest.approx(0.99914, rel=1e-4)
    assert op.t_compute_s == pytest.approx(6.42e-3, rel=1e-2)
    assert op.bound is Bound.COMPUTE_BOUND


def test_m_equals_one_reproduces_the_tail_effect() -> None:
    """A single row on a 16-row array wastes 15 of them: 1/(1+16) = 5.88%.

    This is the sanity check CLAUDE.md names for the utilisation model, and the
    reason a matmul is a family rather than a hand-costed custom op — a CustomOp
    carries no shape, so this number would silently come back as 100%.

    Since D30 the chip-level figure also carries wave quantisation, so the pure
    ``1/17`` is asserted where it lives — on the array — and the chip's value is
    that times the 390 625-tile wave occupancy. Keeping both on the page is the
    point: the first is geometry of one array, the second is how many arrays the
    work could reach.
    """
    report = analyze(_spec(1, 10_000, 10_000), idealised(load_chip("a100_80gb")), _deployment())
    op = _only_op(report)

    assert systolic_utilisation(1, 10_000, 10_000, 16, 16) == pytest.approx(1 / 17, rel=1e-6)
    assert op.utilization == pytest.approx(1 / 17 * 390_625 / (905 * 432), rel=1e-6)
    assert op.arithmetic_intensity == pytest.approx(1.0, rel=1e-2)
    assert op.bound is Bound.DRAM_BW_BOUND


def test_int8_is_never_slower_than_fp16() -> None:
    """CLAUDE.md sanity check, on the one workload where nothing else moves."""
    chip = load_chip("a100_80gb")
    fp16 = _only_op(analyze(_spec(4096, 4096, 4096, "fp16"), chip, _deployment()))
    int8 = _only_op(analyze(_spec(4096, 4096, 4096, "int8"), chip, _deployment()))

    assert int8.latency_s <= fp16.latency_s


def test_mixed_operands_run_at_the_wider_one() -> None:
    """int8 x fp16 runs at the fp16 rate, not the int8 rate (D18).

    Both operands enter the array through one datapath, so the narrow side is
    widened on the way in. It still saves its bytes — B is half the size — but it
    buys no arithmetic throughput, which is exactly why weight-only quantisation
    speeds up a memory-bound decode and does nothing for a compute-bound prefill.
    """
    chip = idealised(load_chip("a100_80gb"))
    spec = MatmulSpec.model_validate(
        {
            "id": "t",
            "name": "t",
            "family": "matmul",
            "m": 4096,
            "n": 4096,
            "k": 4096,
            "a_dtype": "fp16",
            "b_dtype": "int8",
        }
    )
    assert spec.operand_dtype is DType.FP16

    mixed = _only_op(analyze(spec, chip, _deployment()))
    fp16 = _only_op(analyze(_spec(4096, 4096, 4096, "fp16"), chip, _deployment()))

    assert mixed.t_compute_s == pytest.approx(fp16.t_compute_s, rel=1e-6)
    assert mixed.weight_bytes == pytest.approx(fp16.weight_bytes / 2)


def test_result_width_changes_bytes_not_operations() -> None:
    """int8 x int8 -> int32 does the same 2*M*N*K as int8 x int8 -> int8.

    M=N=K=4096: 137.4 GOP either way. The result grows from 16.8 MB to 67.1 MB,
    a 50.3 MB difference that shows up in traffic and nowhere else.
    """
    chip = idealised(load_chip("a100_80gb"))
    narrow = _only_op(analyze(_spec(4096, 4096, 4096, "int8"), chip, _deployment()))
    wide = _only_op(analyze(_spec(4096, 4096, 4096, "int8", out="int32"), chip, _deployment()))

    assert wide.flops == pytest.approx(narrow.flops)
    assert wide.t_compute_s == pytest.approx(narrow.t_compute_s)
    assert wide.arithmetic_intensity < narrow.arithmetic_intensity


def test_the_result_is_always_written_back() -> None:
    """C has no on-chip consumer, so all of it crosses DRAM (D22).

    10000^3 fp16 on A100: A and C are 400 MB of activations against 60.7 MB of
    on-chip capacity, so activation residency is 15.2% — and applying that
    discount to the *result* would keep 30.4 MB of the answer on a chip nobody
    reads it from. Reads take the discount; the write does not.
    """
    chip = idealised(load_chip("a100_80gb"))
    op = _only_op(analyze(_spec(10_000, 10_000, 10_000), chip, _deployment()))

    assert op.dram_write_bytes == pytest.approx(200e6, rel=1e-6)
    assert op.dram_read_bytes + op.dram_write_bytes == pytest.approx(op.dram_bytes)
    # Reads are discounted, so they are below the 400 MB compulsory figure.
    assert op.dram_read_bytes < 400e6


def test_a_consumed_output_may_stay_on_chip() -> None:
    """The same rule, the other way: a transformer's intermediate activation is
    read by the next operation, so residency applies to it and it need not be
    written at all.
    """
    chip = load_chip("a100_80gb")
    deployment = DeploymentSpec.model_validate(
        {"batch": 1, "input_tokens": 512, "output_tokens": 1}
    )
    report = analyze(load_model("llama3_8b"), chip, deployment)
    decode = report.phase(GraphPhase.DECODE)
    assert decode is not None

    # Almost every tensor in a decode step feeds the next operator; only the
    # logits leave. The write share is therefore tiny next to the read share.
    assert decode.dram_write_bytes < 0.01 * decode.dram_read_bytes


def test_ideal_removes_only_the_deratings() -> None:
    """--ideal divides t_compute by 0.70 and t_dram by 0.85, and leaves the
    shape utilisation exactly where it was."""
    spec = _spec(4096, 4096, 4096)
    real = _only_op(analyze(spec, load_chip("a100_80gb"), _deployment()))
    ideal = _only_op(analyze(spec, idealised(load_chip("a100_80gb")), _deployment()))

    assert ideal.utilization == pytest.approx(real.utilization)
    assert ideal.t_compute_s == pytest.approx(real.t_compute_s * 0.70, rel=1e-6)
    assert ideal.t_dram_s == pytest.approx(real.t_dram_s * 0.85, rel=1e-6)
