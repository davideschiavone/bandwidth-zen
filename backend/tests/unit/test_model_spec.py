"""Golden parameter counts and KV-cache sizes, hand-derived in each docstring."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bwz.spec import TransformerSpec, load_model
from bwz.spec.model_spec import FFNType, TransformerParams


def _params(**overrides: object) -> TransformerParams:
    document: dict[str, object] = {
        "layers": 2,
        "hidden": 64,
        "heads": 4,
        "ffn_hidden": 128,
        "ffn_type": "swiglu",
        "vocab": 100,
        "max_context": 512,
    }
    document.update(overrides)
    return TransformerParams.model_validate(document)


# -- the M1 headline golden -------------------------------------------------


def test_llama3_8b_parameter_count() -> None:
    """Llama-3-8B = 8.03 B (+-0.5%), the M1 definition of done.

    Per layer: attention ``4096x4096 + 4096x1024 + 4096x1024 + 4096x4096 = 41.943 M``;
    FFN (swiglu, three matrices) ``3 x 4096 x 14336 = 176.161 M``; two rmsnorm
    scales ``2 x 4096 = 8192``. Total per layer 218.112 M, x32 layers = 6.980 G.
    Untied embeddings ``2 x 128256 x 4096 = 1.051 G``, final norm 4096.
    **Total 8.030 G.**
    """
    model = load_model("llama3_8b")
    assert isinstance(model, TransformerSpec)
    assert model.parameter_count() == pytest.approx(8.03e9, rel=5e-3)


@pytest.mark.parametrize(
    ("model_id", "expected_params", "tolerance", "why"),
    [
        ("llama3_8b", 8.03e9, 5e-3, "Meta's stated 8.03 B"),
        ("llama2_70b", 68.98e9, 5e-3, "the '70B' name is rounded; the checkpoint is 68.98 B"),
        ("mistral_7b", 7.24e9, 5e-3, "Mistral's stated 7.24 B"),
        ("gemma3_4b", 3.88e9, 5e-3, "text tower only; the 4B checkpoint adds a 417 M vision tower"),
        ("gpt3", 175.0e9, 3e-3, "paper says 175 B; the gap is the learned positional table"),
    ],
)
def test_shipped_transformer_parameter_counts(
    model_id: str, expected_params: float, tolerance: float, why: str
) -> None:
    model = load_model(model_id)
    assert isinstance(model, TransformerSpec)
    assert model.parameter_count() == pytest.approx(expected_params, rel=tolerance), why


def test_gemma3_4b_layer_weight_bytes() -> None:
    """94.4 MB of weights per layer at 1 byte/param (docs/CORRECTIONS.md D8).

    Attention ``2560x2048 + 2 x 2560x1024 + 2048x2560 = 15.729 M``; GEGLU FFN
    ``3 x 2560 x 10240 = 78.643 M``. Total **94.372 M**.

    The idealised variant in D8 used Q/O at ``hidden x hidden`` and K/V at
    ``hidden x 512``, giving ``2 x 2560 x (2560 + 512)`` — identical to this
    config's ``2 x 2560 x (2048 + 1024)``. The real config reproduces the golden.
    """
    model = load_model("gemma3_4b")
    assert isinstance(model, TransformerSpec)
    p = model.params
    per_layer = p.attention_params_per_layer() + p.ffn_params_per_layer()
    assert per_layer == pytest.approx(94.4e6, rel=5e-3)


def test_llama3_8b_kv_cache_at_8k() -> None:
    """``32 layers x 2 (K,V) x 1024 kv_width x 8192 tokens x 2 bytes = 1.0737e9``.

    That is exactly 1.0 GiB. PLAN.md's "1.0 GB +-2%" golden is a binary gigabyte;
    in the decimal convention this module uses throughout it is 1.07 GB.
    """
    model = load_model("llama3_8b")
    assert isinstance(model, TransformerSpec)
    assert model.params.kv_cache_bytes(8192, 2.0) == pytest.approx(2**30, rel=1e-6)


def test_gqa_shrinks_the_kv_cache_by_the_group_size() -> None:
    """Llama-3-8B has 32 query heads over 8 KV heads, so KV is a quarter of MHA."""
    gqa = _params(heads=8, kv_heads=2, hidden=64)
    mha = _params(heads=8, kv_heads=8, hidden=64)
    assert mha.kv_cache_bytes(1024, 2.0) == pytest.approx(4 * gqa.kv_cache_bytes(1024, 2.0))


# -- derived shape properties ----------------------------------------------


def test_head_dim_defaults_to_hidden_over_heads() -> None:
    assert _params(hidden=64, heads=4).effective_head_dim == 16


def test_explicit_head_dim_decouples_q_width_from_hidden() -> None:
    """Gemma-3-4B: 8 heads x 256 = 2048, against a hidden of 2560."""
    model = load_model("gemma3_4b")
    assert isinstance(model, TransformerSpec)
    assert model.params.q_width == 2048
    assert model.params.kv_width == 1024
    assert model.params.hidden == 2560


def test_gated_ffn_carries_three_matrices() -> None:
    assert FFNType.SWIGLU.n_matrices == 3
    assert FFNType.GEGLU.n_matrices == 3
    assert FFNType.GELU.n_matrices == 2
    assert _params(ffn_type="swiglu").ffn_params_per_layer() == 3 * 64 * 128
    assert _params(ffn_type="gelu").ffn_params_per_layer() == 2 * 64 * 128


def test_tied_embeddings_halve_the_embedding_parameters() -> None:
    tied = _params(tie_embeddings=True).embedding_params()
    untied = _params(tie_embeddings=False).embedding_params()
    assert untied == 2 * tied


# -- presets ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_id", "headline", "built", "layers"),
    [("gemma3_preset_1b", 1e9, 0.954e9, 3), ("gemma3_preset_2b", 2e9, 1.993e9, 14)],
)
def test_presets_are_realised_by_scaling_the_layer_count(
    model_id: str, headline: float, built: float, layers: int
) -> None:
    """``declared_params`` is realised by scaling layers, the dimension that
    genuinely varies within a model family.

    The realisation is never exact: layers are integers and the 671 M tied
    embedding table does not scale at all. The 2 B preset lands within 0.4%; the
    1 B preset is 4.6% under and is 70% embedding, which is why the residual is
    reported as an assumption rather than hidden.
    """
    model = load_model(model_id)
    assert isinstance(model, TransformerSpec)
    assert model.headline_parameter_count() == pytest.approx(headline)
    assert model.effective_params.layers == layers
    assert model.parameter_count() == pytest.approx(built, rel=0.01)
    error = model.declared_vs_derived_error()
    assert error is not None and error < 0.06


def test_base_profile_reports_no_declared_vs_derived_gap() -> None:
    model = load_model("gemma3_4b")
    assert isinstance(model, TransformerSpec)
    assert model.declared_vs_derived_error() is None
    assert model.headline_parameter_count() == float(model.parameter_count())


def test_preset_without_declared_params_is_rejected() -> None:
    with pytest.raises(ValidationError, match=r"preset_of is set but declared_params is not"):
        TransformerSpec.model_validate(
            {
                "id": "bad_preset",
                "name": "Bad",
                "family": "transformer_decoder",
                "hypothetical": True,
                "preset_of": "gemma3_4b",
                "params": _params().model_dump(),
            }
        )


# -- validation errors ------------------------------------------------------


def test_indivisible_hidden_is_rejected_with_the_numbers() -> None:
    with pytest.raises(ValidationError, match=r"hidden \(65\) is not divisible by heads \(4\)"):
        _params(hidden=65, heads=4)


def test_indivisible_kv_heads_is_rejected() -> None:
    with pytest.raises(ValidationError, match=r"is not divisible by kv_heads"):
        _params(heads=8, kv_heads=3)


def test_kv_heads_exceeding_heads_is_rejected() -> None:
    with pytest.raises(ValidationError, match=r"must not exceed heads"):
        _params(heads=4, kv_heads=8)


def test_sliding_window_pattern_without_a_window_is_rejected() -> None:
    with pytest.raises(ValidationError, match=r"pattern without a window size"):
        _params(sliding_window_pattern=6)


def test_model_requires_source_url_unless_hypothetical() -> None:
    with pytest.raises(ValidationError, match=r"source_url is required"):
        TransformerSpec.model_validate(
            {
                "id": "nameless",
                "name": "No Source",
                "family": "transformer_decoder",
                "params": _params().model_dump(),
            }
        )


def test_cnn_parameter_count_defers_to_the_graph_builder() -> None:
    model = load_model("mobilenetv3")
    with pytest.raises(NotImplementedError, match=r"channel propagation"):
        model.parameter_count()
