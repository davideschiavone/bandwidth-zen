"""Non-matrix work does not run on the matrix engine.

A tensor core performs matrix-multiply-accumulate and nothing else. Norms, GELU,
softmax tails and residuals are elementwise or transcendental work for the vector
units, which on A100 are 16x slower (19.5 against 312 TOP/s). Charging them at
the tensor-core rate overstated them by that factor (docs/CORRECTIONS.md D27).
"""

from __future__ import annotations

import pytest

from bwz.analysis import analyze
from bwz.analysis.roofline import MATRIX_OP_TYPES, machine_model
from bwz.graph.ops import OpType
from bwz.spec import DeploymentSpec, DType, load_chip, load_model


def test_the_two_engines_are_separated() -> None:
    """A100 declares both; the matrix rate is 16x the vector rate."""
    machine = machine_model(load_chip("a100_80gb"), DType.FP16)

    assert machine.unit.name == "tensor_core"
    assert machine.vector_unit.name == "cuda_core"
    assert machine.has_vector_unit
    ratio = machine.effective_flops_per_s / machine.effective_vector_flops_per_s
    assert ratio == pytest.approx(16.0, rel=0.01)

    assert machine.rate_for(OpType.MATMUL) == machine.effective_flops_per_s
    assert machine.rate_for(OpType.NORM) == machine.effective_vector_flops_per_s
    assert OpType.NORM not in MATRIX_OP_TYPES
    assert {OpType.MATMUL, OpType.ATTENTION, OpType.CONV} == set(MATRIX_OP_TYPES)


def test_a_norm_is_charged_at_the_vector_rate() -> None:
    """Llama-3-8B's attn_norm: 8.39 MOP, which is 615 ns of CUDA cores rather
    than the 38.4 ns of tensor cores the engine used to report."""
    deployment = DeploymentSpec.model_validate(
        {"batch": 1, "input_tokens": 512, "output_tokens": 1}
    )
    report = analyze(load_model("llama3_8b"), load_chip("a100_80gb"), deployment)
    machine = machine_model(load_chip("a100_80gb"), DType.FP16)
    norm = next(op for op in report.phases[0].ops if op.op_type is OpType.NORM)

    assert norm.t_compute_s == pytest.approx(
        norm.flops / machine.effective_vector_flops_per_s, rel=1e-6
    )
    assert norm.t_compute_s > norm.flops / machine.effective_flops_per_s


def test_a_profile_with_one_unit_says_so() -> None:
    """chip_a declares only a systolic array, so elementwise work is charged at
    the array's rate — optimistic, and named in the assumptions rather than
    silently assumed."""
    machine = machine_model(load_chip("chip_a"), DType.INT8)
    assert not machine.has_vector_unit
    assert machine.rate_for(OpType.NORM) == machine.effective_flops_per_s

    deployment = DeploymentSpec.model_validate(
        {
            "batch": 1,
            "input_tokens": 128,
            "output_tokens": 1,
            "precision": {"weights": "int8", "activations": "int8", "kv_cache": "int8"},
        }
    )
    report = analyze(load_model("gemma3_4b"), load_chip("chip_a"), deployment)
    assert any("declares no non-systolic compute unit" in a for a in report.assumptions)
