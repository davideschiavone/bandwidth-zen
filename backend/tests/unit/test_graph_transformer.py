"""The M2 golden tests: FLOP rules, parameter cross-checks, prefill vs decode.

The analytic rules and their limits of validity are derived in
``docs/CORRECTIONS.md`` D11 and ``docs/MODEL.md`` §4.
"""

from __future__ import annotations

import pytest

from bwz.graph import (
    GraphPhase,
    MatmulAttrs,
    OpType,
    build_graph,
    build_graphs,
    group_by_layer,
    phases_for,
)
from bwz.operators import cost_of
from bwz.operators.base import OpCost
from bwz.spec import DeploymentSpec, DType, Phase, TransformerSpec, load_model
from bwz.spec.loaders import AnyModelSpec


def _deployment(**overrides: object) -> DeploymentSpec:
    document: dict[str, object] = {"batch": 1, "input_tokens": 2048, "output_tokens": 1}
    document.update(overrides)
    return DeploymentSpec.model_validate(document)


def _total(model_id: str, phase: GraphPhase, **deployment: object) -> OpCost:
    model = load_model(model_id)
    graph = build_graph(model, _deployment(**deployment), phase)
    total = OpCost(0.0)
    for op in graph.ops:
        total = total + cost_of(op, graph.tensors)
    return total


def _non_embedding_params(model: AnyModelSpec) -> int:
    assert isinstance(model, TransformerSpec)
    return model.parameter_count() - model.params.embedding_params()


# -- the graph agrees with the spec -----------------------------------------


@pytest.mark.parametrize("model_id", ["llama3_8b", "llama2_70b", "mistral_7b", "gpt3", "gemma3_4b"])
def test_graph_weight_count_matches_the_closed_form(model_id: str) -> None:
    """Two independent derivations of the same number (M1 promise, kept at M2).

    ``ModelSpec.parameter_count()`` computes it from hyperparameters;
    ``ComputeGraph.parameter_count()`` sums the actual weight tensors the builder
    emitted. Equality means the builder emitted exactly the weights the spec
    describes -- a missing projection or a double-counted tied embedding shows up
    here and nowhere else.
    """
    model = load_model(model_id)
    assert isinstance(model, TransformerSpec)
    graph = build_graph(model, _deployment(), GraphPhase.PREFILL)
    assert graph.parameter_count() == model.parameter_count()


def test_tied_embeddings_are_counted_once() -> None:
    """Gemma-3 ties its 262208x2560 table to the LM head: 671 M, not 1.34 G."""
    model = load_model("gemma3_4b")
    assert isinstance(model, TransformerSpec)
    graph = build_graph(model, _deployment(), GraphPhase.PREFILL)
    head = graph.op("lm_head")
    assert head.weights == ("embed_tokens",)
    assert graph.parameter_count() == model.parameter_count()


def test_untied_embeddings_emit_a_separate_head() -> None:
    model = load_model("llama3_8b")
    graph = build_graph(model, _deployment(), GraphPhase.PREFILL)
    assert graph.op("lm_head").weights == ("lm_head.w",)


# -- the analytic FLOP rules ------------------------------------------------


def test_gpt3_prefill_matches_two_n_d() -> None:
    """GPT-3 175B, S=2048, batch 1: **722.6 TFLOP against 2*N*D = 715.0 TFLOP, +1.1%**.

    PROMPT.md M2 states this rule as ``6*N*D``, which is the *training* figure --
    forward plus a backward pass costing twice as much again. Prefill is
    forward-only. See docs/CORRECTIONS.md D11.

    The full-N form works here only because GPT-3's embeddings are 0.4% of its
    parameters; see the next test for a model where it does not.
    """
    model = load_model("gpt3")
    assert isinstance(model, TransformerSpec)
    analytic = 2.0 * model.parameter_count() * 2048
    actual = _total("gpt3", GraphPhase.PREFILL, input_tokens=2048).flops
    assert actual == pytest.approx(analytic, rel=0.03)
    assert actual != pytest.approx(6.0 * model.parameter_count() * 2048, rel=0.5)


def test_gemma3_forward_matches_two_n_s_over_non_embedding_params() -> None:
    """Gemma-3-4B, S=512, batch 1: **3.33 TFLOP against 2*N_ne*S = 3.29 TFLOP, +1.2%**.

    Against the full ``2*N*S`` it is 16.3% low, because 17.3% of Gemma-3-4B's
    parameters are an embedding table that a forward pass gathers from rather
    than multiplies by. The residual +1.2% is attention, which grows as S^2 --
    at S=2048 the same comparison is +4.5%.
    """
    model = load_model("gemma3_4b")
    analytic = 2.0 * _non_embedding_params(model) * 512
    actual = _total("gemma3_4b", GraphPhase.PREFILL, input_tokens=512).flops
    assert actual == pytest.approx(analytic, rel=0.02)


def test_the_embedding_fraction_is_what_breaks_the_full_n_rule() -> None:
    """Stated as a test so the limit of validity cannot quietly rot (D11)."""
    for model_id, embedding_share, tolerance in [("gpt3", 0.004, 0.03), ("gemma3_4b", 0.173, 0.20)]:
        model = load_model(model_id)
        assert isinstance(model, TransformerSpec)
        share = model.params.embedding_params() / model.parameter_count()
        assert share == pytest.approx(embedding_share, abs=0.01)
        actual = _total(model_id, GraphPhase.PREFILL, input_tokens=2048).flops
        assert actual == pytest.approx(2.0 * model.parameter_count() * 2048, rel=tolerance)


def test_decode_does_two_flops_per_weight_element_streamed() -> None:
    """The D8 rule, stated exactly: a batch-1 GEMM does one MAC per weight element.

    Over Llama-3-8B's decode matmuls the ratio is 2.0 to machine precision. It is
    *not* 2 per *parameter*: the embedding table is gathered rather than
    multiplied, so it contributes weight footprint but no arithmetic, while the
    untied LM head does run and does contribute.

    Attention is the remainder — 1.2 GFLOP of the 16.2 GFLOP total at a 2304-token
    context, growing linearly with context while the matmul term does not.
    """
    model = load_model("llama3_8b")
    graph = build_graph(model, _deployment(input_tokens=2048, output_tokens=256), GraphPhase.DECODE)
    matmuls = [op for op in graph.ops if op.op_type is OpType.MATMUL]
    matmul_flops = sum(cost_of(op, graph.tensors).flops for op in matmuls)
    matmul_weight_elements = sum(
        cost_of(op, graph.tensors).weight_bytes / 2.0
        for op in matmuls  # fp16
    )
    assert matmul_flops / matmul_weight_elements == pytest.approx(2.0, rel=1e-9)

    total = _total("llama3_8b", GraphPhase.DECODE, input_tokens=2048, output_tokens=256).flops
    attention_share = (total - matmul_flops) / total
    assert 0.0 < attention_share < 0.15


# -- prefill and decode are different machines ------------------------------


def test_prefill_and_decode_differ_only_in_row_count() -> None:
    """Same weights, same op count; ``M`` is 2048x larger at prefill."""
    model = load_model("llama3_8b")
    deployment = _deployment(input_tokens=2048, output_tokens=256)
    prefill = build_graph(model, deployment, GraphPhase.PREFILL)
    decode = build_graph(model, deployment, GraphPhase.DECODE)
    assert len(prefill.ops) == len(decode.ops)
    assert prefill.parameter_count() == decode.parameter_count()
    prefill_q = prefill.op("layer0.q_proj").attrs
    decode_q = decode.op("layer0.q_proj").attrs
    assert isinstance(prefill_q, MatmulAttrs) and isinstance(decode_q, MatmulAttrs)
    assert (prefill_q.m, prefill_q.n, prefill_q.k) == (2048, 4096, 4096)
    assert (decode_q.m, decode_q.n, decode_q.k) == (1, 4096, 4096)


def test_decode_is_memory_bound_and_prefill_is_not() -> None:
    """The headline consequence: arithmetic intensity differs by ~700x.

    Decode's ~1 FLOP/byte is below every chip's ridge point; prefill's ~740 is
    above most of them.
    """
    prefill = _total("llama3_8b", GraphPhase.PREFILL, input_tokens=2048, output_tokens=256)
    decode = _total("llama3_8b", GraphPhase.DECODE, input_tokens=2048, output_tokens=256)
    assert decode.arithmetic_intensity == pytest.approx(1.06, rel=0.15)
    assert prefill.arithmetic_intensity > 500
    assert prefill.arithmetic_intensity / decode.arithmetic_intensity > 100


def test_llama3_decode_moves_about_16_gb_per_token() -> None:
    """CLAUDE.md sanity check: ~16 GB of weight traffic per decoded token.

    15.0 GB of it is weights -- 13.96 GB of layer weights plus a 1.05 GB untied
    LM head. The embedding table is gathered, not streamed, so it contributes one
    row rather than its own 1.05 GB.

    (CLAUDE.md pairs this figure with "~35-55 tok/s", which does not follow from
    it: 16 GB at H100's 3.35 TB/s is 4.8 ms, i.e. ~209 tok/s at peak. See
    docs/CORRECTIONS.md D12. Nothing here predicts tok/s; that is M3.)
    """
    decode = _total("llama3_8b", GraphPhase.DECODE, input_tokens=2048, output_tokens=256)
    assert decode.weight_bytes == pytest.approx(15.0e9, rel=0.05)
    assert decode.total_bytes == pytest.approx(16e9, rel=0.10)


def test_decode_weight_traffic_is_independent_of_context() -> None:
    """Weights stream once per token whatever the context; only KV grows."""
    short = _total("llama3_8b", GraphPhase.DECODE, input_tokens=128, output_tokens=1)
    long = _total("llama3_8b", GraphPhase.DECODE, input_tokens=8000, output_tokens=1)
    assert short.weight_bytes == long.weight_bytes
    assert long.input_bytes > short.input_bytes


# -- KV cache ---------------------------------------------------------------


def test_llama3_kv_cache_at_8k_is_one_gibibyte() -> None:
    """``32 x 2 x 1024 x 8192 x 2 bytes = 1.0737e9`` -- exactly 1.0 GiB.

    PLAN.md's "1.0 GB +-2%" golden is a binary gigabyte; in this engine's decimal
    convention it is 1.07 GB.
    """
    model = load_model("llama3_8b")
    graph = build_graph(
        model,
        _deployment(input_tokens=8192, output_tokens=1, kv_context_tokens=8192),
        GraphPhase.DECODE,
    )
    assert graph.kv_cache_bytes() == pytest.approx(2**30, rel=1e-6)


def test_gemma3_kv_cache_is_twice_the_supplied_d8_figure() -> None:
    """``34 x 2 x 1024 x 4096 x 1 byte = 285.2 MB``, against D8's 142.6 MB.

    D8 assumed ``d_kv = 512``; the real config has ``kv_heads x head_dim =
    4 x 256 = 1024``. The attention *weights* coincidence that makes the 94.4
    MB/layer golden hold does not extend to KV, which depends on ``d_kv`` alone
    (docs/CORRECTIONS.md D9).
    """
    model = load_model("gemma3_4b")
    deployment = DeploymentSpec.model_validate(
        {
            "batch": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "kv_context_tokens": 4096,
            "precision": {"weights": "int8", "activations": "int8", "kv_cache": "int8"},
        }
    )
    graph = build_graph(model, deployment, GraphPhase.DECODE)
    assert graph.kv_cache_bytes() == pytest.approx(285.2e6, rel=0.01)


def test_gqa_shrinks_the_kv_cache() -> None:
    """Llama-2-70B: 64 query heads over 8 KV heads, so KV is 1/8 of MHA's."""
    model = load_model("llama2_70b")
    assert isinstance(model, TransformerSpec)
    assert model.params.kv_width * 8 == model.params.q_width


# -- the D8 per-op layer table ----------------------------------------------


def test_gemma3_layer_weight_bytes_and_the_two_ops_per_byte_rule() -> None:
    """docs/CORRECTIONS.md D8: 94.4 MB of INT8 weights per layer, ``ops = 2 x weight_bytes``.

    Q/O at ``2560x2048`` and K/V at ``2560x1024`` give 15.73 MB; the GEGLU FFN's
    three ``2560x10240`` matrices give 78.64 MB. **Total 94.372 MB.** The ratio
    holds exactly over the layer's matmuls, because a batch-1 GEMM does one MAC
    -- two operations -- per weight element.
    """
    model = load_model("gemma3_4b")
    deployment = DeploymentSpec.model_validate(
        {
            "batch": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "kv_context_tokens": 4096,
            "precision": {"weights": "int8", "activations": "int8", "kv_cache": "int8"},
        }
    )
    graph = build_graph(model, deployment, GraphPhase.DECODE)
    matmuls = [op for op in group_by_layer(graph)[0] if op.op_type is OpType.MATMUL]
    weight_bytes = sum(cost_of(op, graph.tensors).weight_bytes for op in matmuls)
    flops = sum(cost_of(op, graph.tensors).flops for op in matmuls)
    assert weight_bytes == pytest.approx(94.4e6, rel=0.01)
    assert flops / weight_bytes == pytest.approx(2.0, rel=1e-9)
    assert len(matmuls) == 7, "Q, K, V, O, gate, up, down"


def test_int8_weights_halve_the_footprint_of_fp16() -> None:
    model = load_model("gemma3_4b")
    base = {"batch": 1, "input_tokens": 1, "output_tokens": 1}
    fp16 = build_graph(model, DeploymentSpec.model_validate(base), GraphPhase.DECODE)
    int8 = build_graph(
        model,
        DeploymentSpec.model_validate({**base, "precision": {"weights": "int8"}}),
        GraphPhase.DECODE,
    )
    assert fp16.weight_footprint_bytes() == pytest.approx(2 * int8.weight_footprint_bytes())
    assert int8.weight_footprint_bytes() == pytest.approx(3.88e9, rel=0.01)


# -- phase selection --------------------------------------------------------


def test_phases_for_follows_the_deployment() -> None:
    model = load_model("llama3_8b")
    assert phases_for(model, _deployment(phase=Phase.PREFILL)) == (GraphPhase.PREFILL,)
    assert phases_for(model, _deployment(phase=Phase.DECODE)) == (GraphPhase.DECODE,)
    assert phases_for(model, _deployment(phase=Phase.BOTH)) == (
        GraphPhase.PREFILL,
        GraphPhase.DECODE,
    )
    assert set(build_graphs(model, _deployment()).keys()) == {
        GraphPhase.PREFILL,
        GraphPhase.DECODE,
    }


def test_a_transformer_refuses_the_static_phase() -> None:
    model = load_model("llama3_8b")
    with pytest.raises(ValueError, match=r"prefill and decode are different machines"):
        build_graph(model, _deployment(), GraphPhase.STATIC)


def test_vanilla_attention_materialises_scores_and_flash_does_not() -> None:
    model = load_model("llama3_8b")
    base = {"batch": 1, "input_tokens": 2048, "output_tokens": 1}
    flash = build_graph(model, DeploymentSpec.model_validate(base), GraphPhase.PREFILL)
    vanilla = build_graph(
        model,
        DeploymentSpec.model_validate({**base, "attention_impl": "vanilla"}),
        GraphPhase.PREFILL,
    )

    def totals(graph: object) -> OpCost:
        total = OpCost(0.0)
        for op in graph.ops:  # type: ignore[attr-defined]
            total = total + cost_of(op, graph.tensors)  # type: ignore[attr-defined]
        return total

    assert totals(flash).flops == pytest.approx(totals(vanilla).flops)
    assert totals(vanilla).total_bytes > totals(flash).total_bytes


def test_precision_is_honoured_throughout() -> None:
    model = load_model("llama3_8b")
    deployment = DeploymentSpec.model_validate(
        {
            "batch": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "precision": {"weights": "int8", "activations": "fp16", "kv_cache": "fp8"},
        }
    )
    graph = build_graph(model, deployment, GraphPhase.DECODE)
    assert graph.tensors["layer0.q_proj.w"].dtype is DType.INT8
    assert graph.tensors["layer0.attn.out"].dtype is DType.FP16
    assert graph.tensors["layer0.k_cache"].dtype is DType.FP8
