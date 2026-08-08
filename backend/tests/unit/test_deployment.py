"""Deployment validation: intra-spec only, with errors that name the numbers.

Cross-spec checks (does the chip support this dtype? do the weights fit?) are
feasibility questions answered by a Report at M3, not load-time exceptions
(CLAUDE.md #8).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bwz.spec import DeploymentSpec, DType, Mode, Phase, SparsityType
from bwz.spec.deployment import Sparsity


def test_defaults_are_a_valid_single_chip_inference_config() -> None:
    spec = DeploymentSpec()
    assert spec.num_chips == 1
    assert spec.parallelism.total_chips == 1
    assert spec.runs_prefill() and spec.runs_decode()
    assert spec.precision.weights is DType.FP16


def test_parallelism_product_must_equal_num_chips() -> None:
    with pytest.raises(ValidationError, match=r"tp=2 x pp=2 x dp=1 x ep=1 = 4 does not equal"):
        DeploymentSpec(parallelism={"tp": 2, "pp": 2}, num_chips=8)  # type: ignore[arg-type]


def test_matching_parallelism_is_accepted() -> None:
    spec = DeploymentSpec(
        parallelism={"tp": 2, "pp": 2, "microbatches": 4},  # type: ignore[arg-type]
        num_chips=4,
    )
    assert spec.parallelism.total_chips == 4


def test_decode_requires_output_tokens() -> None:
    with pytest.raises(ValidationError, match=r"requires output_tokens > 0"):
        DeploymentSpec(phase=Phase.DECODE, output_tokens=0)


def test_prefill_only_may_generate_nothing() -> None:
    assert DeploymentSpec(phase=Phase.PREFILL, output_tokens=0).runs_decode() is False


def test_training_has_no_phase_split() -> None:
    assert DeploymentSpec(mode=Mode.INFERENCE, phase=Phase.PREFILL).mode is Mode.INFERENCE
    with pytest.raises(ValidationError, match=r"mode='training' has no prefill/decode split"):
        DeploymentSpec(mode=Mode.TRAINING, phase=Phase.PREFILL)


def test_pipeline_needs_at_least_pp_microbatches() -> None:
    with pytest.raises(ValidationError, match=r"leaves the pipeline mostly empty"):
        DeploymentSpec(
            parallelism={"pp": 4, "microbatches": 2},  # type: ignore[arg-type]
            num_chips=4,
        )


def test_context_defaults_to_prompt_plus_generation() -> None:
    spec = DeploymentSpec(input_tokens=2048, output_tokens=256)
    assert spec.context_tokens == 2304


def test_context_override_is_honoured() -> None:
    """The D8 example sizes the KV cache at a fixed C=4096 rather than at end-of-generation."""
    spec = DeploymentSpec(input_tokens=512, output_tokens=256, kv_context_tokens=4096)
    assert spec.context_tokens == 4096


def test_sparsity_ratio_without_a_type_is_rejected() -> None:
    with pytest.raises(ValidationError, match=r"ratio is set but sparsity.type is 'none'"):
        Sparsity(ratio=0.5)


def test_structured_2_4_implies_half() -> None:
    assert Sparsity(type=SparsityType.STRUCTURED_2_4).ratio is None
    with pytest.raises(ValidationError, match=r"implies ratio 0.5"):
        Sparsity(type=SparsityType.STRUCTURED_2_4, ratio=0.75)


def test_bad_enum_names_the_allowed_values() -> None:
    with pytest.raises(ValidationError) as excinfo:
        DeploymentSpec(attention_impl="flashattention3")  # type: ignore[arg-type]
    message = str(excinfo.value)
    assert "flash2" in message and "vanilla" in message
