"""The hand-countable encoder: every number in one place, derived twice.

This profile exists to be checked with a calculator, so the test writes the
arithmetic out rather than asserting a figure the engine produced. If the engine
and the comment ever disagree, one of them is a bug and this test says which.

Also pins the three ways an encoder differs from a decoder (docs/CORRECTIONS.md
D24): bidirectional attention, one phase, no LM head.
"""

from __future__ import annotations

import pytest

from bwz.analysis import analyze
from bwz.graph import GraphPhase, build_graph, phases_for
from bwz.graph.ops import AttentionAttrs, OpType
from bwz.spec import DeploymentSpec, load_chip, load_model

HIDDEN, HEADS, HEAD_DIM, FFN, VOCAB = 8, 2, 4, 16, 16
TOKENS = 4


def _deployment(tokens: int = TOKENS) -> DeploymentSpec:
    return DeploymentSpec.model_validate(
        {"batch": 1, "input_tokens": tokens, "output_tokens": 0, "phase": "prefill"}
    )


def test_parameter_count_by_hand() -> None:
    """664, and every term is a product of two small integers.

    Q, K, V, O   4 x (8 x 8)        = 256
    FFN          8 x 16 + 16 x 8    = 256
    norms        2 x 8              =  16   rmsnorm: one scale per channel
    per layer                         528
    embeddings   16 x 8             = 128   tied, counted once
    final norm   8                  =   8
    total                             664
    """
    attention = 4 * HIDDEN * HIDDEN
    ffn = 2 * HIDDEN * FFN
    norms = 2 * HIDDEN
    per_layer = attention + ffn + norms
    total = per_layer + VOCAB * HIDDEN + HIDDEN

    assert (attention, ffn, norms, per_layer) == (256, 256, 16, 528)
    assert total == 664
    assert load_model("single_layer_encoder").parameter_count() == total


def test_operation_count_by_hand() -> None:
    """5280 operations for one pass over 4 tokens.

    q,k,v,o_proj  4 x 2·S·8·8                4 x  512 = 2048
    ffn_up/down   2·S·8·16 + 2·S·16·8        2 x 1024 = 2048
    attn          scored = heads·S² = 32
                  2·2·scored·head_dim + 5·scored      =  672
    norms         3 x (S·8 x 4 flops)        3 x  128 =  384
    ffn_act       S·16 x 1 flop (relu)                =   64
    residuals     2 x S·8                    2 x   32 =   64
    embed         a gather                            =    0
    total                                               5280
    """
    projections = 4 * 2 * TOKENS * HIDDEN * HIDDEN
    ffn = 2 * (2 * TOKENS * HIDDEN * FFN)
    scored = HEADS * TOKENS * TOKENS
    attention = 2 * 2 * scored * HEAD_DIM + 5 * scored
    norms = 3 * (TOKENS * HIDDEN * 4)
    activation = TOKENS * FFN
    residuals = 2 * TOKENS * HIDDEN
    total = projections + ffn + attention + norms + activation + residuals

    assert (projections, ffn, attention, norms) == (2048, 2048, 672, 384)
    assert total == 5280

    report = analyze(load_model("single_layer_encoder"), load_chip("a100_80gb"), _deployment())
    assert report.phases[0].flops == pytest.approx(total)


def test_attention_is_bidirectional() -> None:
    """No triangle to halve: every token attends to every other.

    A decoder at S=4 scores 10 positions (the lower triangle); this encoder
    scores all 16. Getting that wrong would understate attention by 37.5% here
    and by nearly half at long sequences.
    """
    graph = build_graph(load_model("single_layer_encoder"), _deployment(), GraphPhase.PREFILL)
    attention = next(op for op in graph.ops if op.op_type is OpType.ATTENTION)
    assert isinstance(attention.attrs, AttentionAttrs)

    assert attention.attrs.causal is False
    scored = HEADS * TOKENS * TOKENS
    assert scored == 32  # a causal decoder would score 2 * 10 = 20


def test_the_cli_shape_matches_the_profile() -> None:
    """`bwz single-layer-encoder` with its defaults is the shipped profile.

    The command exists so a dimension can be changed and its effect read off;
    the profile exists so the derivation has somewhere to live. They must not
    drift apart.
    """
    from bwz.spec import TransformerSpec

    cli = TransformerSpec.model_validate(
        {
            "id": "x",
            "name": "x",
            "family": "transformer_encoder",
            "hypothetical": True,
            "params": {
                "layers": 1,
                "hidden": HIDDEN,
                "heads": HEADS,
                "ffn_hidden": FFN,
                "ffn_type": "relu",
                "vocab": VOCAB,
                "max_context": TOKENS,
                "norm": "rmsnorm",
                "positional": "none",
                "tie_embeddings": True,
            },
        }
    )
    profile = load_model("single_layer_encoder")
    assert cli.parameter_count() == profile.parameter_count() == 664

    chip = load_chip("a100_80gb")
    assert analyze(cli, chip, _deployment()).phases[0].flops == pytest.approx(
        analyze(profile, chip, _deployment()).phases[0].flops
    )


def test_an_encoder_has_no_kv_cache() -> None:
    """K and V are intermediate activations, not a cache.

    A cache exists to be reused by a later step; an encoder has no later step.
    Tagging them as cache reported 128 B of footprint that nothing would ever
    read again, and the planner tracks cache separately from activations, so it
    also skewed the residency waterfall.
    """
    report = analyze(load_model("single_layer_encoder"), load_chip("a100_80gb"), _deployment())
    assert report.memory.kv_cache_bytes == 0.0

    graph = build_graph(load_model("single_layer_encoder"), _deployment(), GraphPhase.PREFILL)
    assert not any("cache" in name for name in graph.tensors)


def test_an_encoder_has_one_phase_and_no_lm_head() -> None:
    """One bidirectional pass, and it stops at hidden states.

    There is no token-by-token phase to separate, and what sits on top of an
    encoder — classifier, MLM head, pooler — is task-specific, so counting one
    would be inventing a layer.
    """
    model = load_model("single_layer_encoder")
    assert phases_for(model, _deployment()) == (GraphPhase.PREFILL,)

    graph = build_graph(model, _deployment(), GraphPhase.PREFILL)
    assert not any(op.id == "lm_head" for op in graph.ops)
    assert len(graph.ops) == 14
