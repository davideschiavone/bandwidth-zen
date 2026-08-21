"""The two ad-hoc kernel probes: id/name derivation, and defaults that hold.

``docs/CORRECTIONS.md`` D39. ``bwz matmul``/``bwz encoder-layer`` and
``scripts/plot_pipeline.py``'s ``--matmul``/``--encoder`` all go through
:mod:`bwz.kernels` now, so there is one place to test the shape-to-spec mapping
instead of two independently hand-built dicts that can drift.
"""

from __future__ import annotations

import pytest

from bwz.analysis import analyze
from bwz.kernels import encoder_layer_kernel, matmul_kernel
from bwz.spec import DeploymentSpec, DType, FFNType, NormType, load_chip, load_model


def test_matmul_kernel_id_and_name_come_from_the_shape() -> None:
    spec = matmul_kernel(64, 32, 16, a_dtype=DType.FP16, b_dtype=DType.FP16)
    assert spec.id == "matmul_64x32x16"
    assert spec.name == "matmul 64x32x16"
    assert (spec.m, spec.n, spec.k) == (64, 32, 16)


def test_matmul_kernel_threads_out_dtype() -> None:
    spec = matmul_kernel(64, 64, 64, a_dtype=DType.INT8, b_dtype=DType.INT8, out_dtype=DType.INT32)
    assert spec.result_dtype.value == "int32"


def test_encoder_layer_kernel_id_and_name_come_from_the_shape() -> None:
    spec = encoder_layer_kernel(dmodel=8, nheads=2, ffn=16, vocab=16, tokens=4)
    assert spec.id == "encoder_layer_d8_h2_ffn16_s4"
    assert spec.name == "1-layer encoder d=8 h=2 ffn=16"


def test_encoder_layer_kernel_defaults_to_the_plainest_architecture() -> None:
    spec = encoder_layer_kernel(dmodel=8, nheads=2, ffn=16, vocab=16, tokens=4)
    assert spec.params.ffn_type is FFNType.RELU
    assert spec.params.norm is NormType.RMSNORM
    assert spec.params.tie_embeddings is True


def test_encoder_layer_kernel_threads_architecture_overrides() -> None:
    spec = encoder_layer_kernel(
        dmodel=8,
        nheads=2,
        ffn=16,
        vocab=16,
        tokens=4,
        ffn_type=FFNType.SWIGLU,
        norm=NormType.LAYERNORM,
        tie_embeddings=False,
    )
    assert spec.params.ffn_type is FFNType.SWIGLU
    assert spec.params.norm is NormType.LAYERNORM
    assert spec.params.tie_embeddings is False


def test_default_encoder_layer_shape_matches_the_toy_profile() -> None:
    """`bwz encoder-layer` with its own defaults is the shipped toy profile.

    The command exists so a dimension can be changed and its effect read off;
    the profile exists so the hand-derivation has somewhere to live
    (docs/CORRECTIONS.md D24). They must not drift apart — this is the same
    assertion `test_the_cli_shape_matches_the_profile` made before D39 gave
    the CLI's own construction a name (`encoder_layer_kernel`) to call here
    instead of re-typing the dict a third time.
    """
    kernel = encoder_layer_kernel(dmodel=8, nheads=2, ffn=16, vocab=16, tokens=4)
    profile = load_model("single_layer_encoder_toy")
    assert kernel.parameter_count() == profile.parameter_count() == 664

    chip = load_chip("a100_80gb")
    deployment = DeploymentSpec.model_validate(
        {"batch": 1, "input_tokens": 4, "output_tokens": 0, "phase": "prefill"}
    )
    kernel_flops = analyze(kernel, chip, deployment).phases[0].flops
    profile_flops = analyze(profile, chip, deployment).phases[0].flops
    assert kernel_flops == pytest.approx(profile_flops) == pytest.approx(5280)


def test_encoder_layer_kernel_rejects_indivisible_shapes() -> None:
    """D44: the kernel probe has no `head_dim` override — it is always `dmodel
    // nheads` — so a shape that does not divide evenly is rejected outright
    rather than silently floored. Unlike `TransformerParams`'s own, more
    permissive validator (`bwz/spec/model_spec.py`, which lets `head_dim` be
    set independently for real GQA profiles like Gemma-3), this error is
    raised before a spec is ever built, in the caller's own vocabulary
    (`dmodel`/`nheads`), not the schema's (`hidden`/`heads`).
    """
    with pytest.raises(ValueError, match=r"dmodel \(100\) is not divisible by nheads \(6\)"):
        encoder_layer_kernel(dmodel=100, nheads=6, ffn=16, vocab=16, tokens=4)
