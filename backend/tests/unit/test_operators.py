"""Operator cost goldens. Every expected value is hand-computed in the docstring.

These models are hardware-independent (docs/CORRECTIONS.md D10), so every number
here is checkable with a calculator and nothing else.
"""

from __future__ import annotations

import pytest

from bwz.graph.ops import (
    AttentionAttrs,
    ConvAttrs,
    CustomAttrs,
    ElementwiseAttrs,
    EmbeddingAttrs,
    MatmulAttrs,
    NormAttrs,
    Operation,
    OpType,
    PoolAttrs,
    Tensor,
    TensorKind,
)
from bwz.operators import cost_model_for, cost_of, register_op
from bwz.operators.base import OpCost, OperatorCostModel
from bwz.spec.dtypes import DType


def _tensor(name: str, shape: tuple[int, ...], dtype: DType = DType.FP16) -> Tensor:
    return Tensor(name, shape, dtype, TensorKind.ACTIVATION)


def _weight(name: str, shape: tuple[int, ...], dtype: DType = DType.FP16) -> Tensor:
    return Tensor(name, shape, dtype, TensorKind.WEIGHT)


# -- matmul -----------------------------------------------------------------


def test_matmul_flops_are_two_m_n_k() -> None:
    """``M=128, N=256, K=512`` -> ``2 * 128 * 256 * 512 = 33_554_432`` FLOPs."""
    tensors = {
        "x": _tensor("x", (128, 512)),
        "w": _weight("w", (512, 256)),
        "y": _tensor("y", (128, 256)),
    }
    op = Operation(
        id="mm",
        op_type=OpType.MATMUL,
        attrs=MatmulAttrs(m=128, n=256, k=512),
        inputs=("x",),
        weights=("w",),
        outputs=("y",),
    )
    cost = cost_of(op, tensors)
    assert cost.flops == 2 * 128 * 256 * 512
    # fp16: weights 512*256*2, input 128*512*2, output 128*256*2
    assert cost.weight_bytes == 512 * 256 * 2
    assert cost.input_bytes == 128 * 512 * 2
    assert cost.output_bytes == 128 * 256 * 2


def test_decode_projection_has_intensity_near_one() -> None:
    """Llama-3-8B Q projection at batch 1: ``M=1, K=N=4096``.

    ``2 * 1 * 4096 * 4096 = 33.55 MFLOP`` against ``4096*4096*2 = 33.55 MB`` of
    fp16 weights. One FLOP per byte -- three orders of magnitude below every
    chip's ridge point, which is the whole reason batch-1 decode is memory-bound.
    """
    tensors = {
        "x": _tensor("x", (1, 4096)),
        "w": _weight("w", (4096, 4096)),
        "y": _tensor("y", (1, 4096)),
    }
    op = Operation(
        id="q",
        op_type=OpType.MATMUL,
        attrs=MatmulAttrs(m=1, n=4096, k=4096),
        inputs=("x",),
        weights=("w",),
        outputs=("y",),
    )
    assert cost_of(op, tensors).arithmetic_intensity == pytest.approx(1.0, rel=1e-3)


# -- attention --------------------------------------------------------------


def _attention_op(**overrides: object) -> tuple[Operation, dict[str, Tensor]]:
    attrs_kwargs: dict[str, object] = {
        "batch": 1,
        "heads": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "q_len": 2048,
        "kv_len": 2048,
        "causal": True,
        "materialize_scores": False,
    }
    attrs_kwargs.update(overrides)
    attrs = AttentionAttrs(**attrs_kwargs)  # type: ignore[arg-type]
    tensors = {
        "q": _tensor("q", (attrs.batch * attrs.q_len, attrs.heads * attrs.head_dim)),
        "k": Tensor(
            "k",
            (attrs.batch, attrs.kv_len, attrs.kv_heads * attrs.head_dim),
            DType.FP16,
            TensorKind.KV_CACHE,
        ),
        "v": Tensor(
            "v",
            (attrs.batch, attrs.kv_len, attrs.kv_heads * attrs.head_dim),
            DType.FP16,
            TensorKind.KV_CACHE,
        ),
        "out": _tensor("out", (attrs.batch * attrs.q_len, attrs.heads * attrs.head_dim)),
    }
    op = Operation(
        id="attn",
        op_type=OpType.ATTENTION,
        attrs=attrs,
        inputs=("q", "k", "v"),
        outputs=("out",),
    )
    return op, tensors


def test_causal_masking_halves_prefill_score_flops() -> None:
    """The lower triangle of a square score block is ``S*(S+1)/2`` of ``S*S``.

    At S=2048 that is 2_098_176 of 4_194_304 positions, a factor of 0.50024.
    """
    causal, tensors = _attention_op(causal=True)
    dense, _ = _attention_op(causal=False)
    ratio = cost_of(causal, tensors).flops / cost_of(dense, tensors).flops
    assert ratio == pytest.approx(0.5 + 1 / (2 * 2048), rel=1e-6)


def test_prefill_attention_flops_golden() -> None:
    """Llama-3-8B prefill, S=2048, batch 1, 32 heads, head_dim 128.

    positions = 2048*2049/2 = 2_098_176; scored = 1*32*positions = 67_141_632.
    ``4 * scored * 128`` for the two matmuls = 34.376 GFLOP, plus softmax
    ``5 * scored`` = 0.336 GFLOP. **Total 34.71 GFLOP.**
    """
    op, tensors = _attention_op()
    scored = 32 * (2048 * 2049 / 2)
    expected = 4.0 * scored * 128 + 5.0 * scored
    assert cost_of(op, tensors).flops == pytest.approx(expected)
    assert cost_of(op, tensors).flops == pytest.approx(34.71e9, rel=1e-3)


def test_decode_attention_is_not_halved() -> None:
    """At ``q_len == 1`` every key is visible, so causality removes nothing."""
    causal, tensors = _attention_op(q_len=1, kv_len=4096, causal=True)
    dense, _ = _attention_op(q_len=1, kv_len=4096, causal=False)
    assert cost_of(causal, tensors).flops == cost_of(dense, tensors).flops


def test_flash_changes_bytes_never_flops() -> None:
    """CLAUDE.md sanity check, asserted directly."""
    flash, tensors = _attention_op(materialize_scores=False)
    vanilla, _ = _attention_op(materialize_scores=True)
    flash_cost, vanilla_cost = cost_of(flash, tensors), cost_of(vanilla, tensors)
    assert flash_cost.flops == vanilla_cost.flops
    assert flash_cost.scratch_bytes == 0.0
    assert vanilla_cost.scratch_bytes > 0.0
    assert vanilla_cost.total_bytes > flash_cost.total_bytes


def test_gqa_changes_kv_bytes_never_flops() -> None:
    """8 KV heads instead of 32 quarters the KV traffic and leaves FLOPs alone."""
    gqa, gqa_tensors = _attention_op(kv_heads=8)
    mha, mha_tensors = _attention_op(kv_heads=32)
    assert cost_of(gqa, gqa_tensors).flops == cost_of(mha, mha_tensors).flops
    kv_gqa = gqa_tensors["k"].size_bytes + gqa_tensors["v"].size_bytes
    kv_mha = mha_tensors["k"].size_bytes + mha_tensors["v"].size_bytes
    assert kv_mha == pytest.approx(4 * kv_gqa)


def test_causal_with_existing_context_adds_the_full_block_back() -> None:
    """A 512-token prompt appended to 3584 tokens of context sees all of the
    history and half of itself: ``512*3584 + 512*513/2 = 1_966_336`` positions."""
    op, tensors = _attention_op(q_len=512, kv_len=4096, heads=1, batch=1, head_dim=1)
    expected_positions = 512 * 3584 + 512 * 513 / 2
    assert cost_of(op, tensors).flops == pytest.approx(
        4.0 * expected_positions * 1 + 5.0 * expected_positions
    )


# -- conv -------------------------------------------------------------------


def test_conv_flops() -> None:
    """``2 * 1 * 112*112 * 16 * 3 * 3*3 = 10.838 MFLOP`` for MobileNetV3's stem."""
    tensors = {
        "x": _tensor("x", (1, 3, 224, 224)),
        "w": _weight("w", (16, 3, 3, 3)),
        "y": _tensor("y", (1, 16, 112, 112)),
    }
    op = Operation(
        id="stem",
        op_type=OpType.CONV,
        attrs=ConvAttrs(
            batch=1,
            in_channels=3,
            out_channels=16,
            in_height=224,
            in_width=224,
            out_height=112,
            out_width=112,
            kernel_h=3,
            kernel_w=3,
            groups=1,
        ),
        inputs=("x",),
        weights=("w",),
        outputs=("y",),
    )
    assert cost_of(op, tensors).flops == 2 * 112 * 112 * 16 * 3 * 3 * 3


def test_depthwise_drops_flops_by_the_channel_count_but_not_bytes() -> None:
    """CLAUDE.md sanity check: depthwise layers are memory-bound.

    Same shapes, ``groups=1`` vs ``groups=channels``: FLOPs fall by exactly the
    channel count while activation traffic is identical, so arithmetic intensity
    falls by the same factor.
    """
    channels = 64
    tensors = {
        "x": _tensor("x", (1, channels, 56, 56)),
        "w_dense": _weight("w_dense", (channels, channels, 3, 3)),
        "w_dw": _weight("w_dw", (channels, 1, 3, 3)),
        "y": _tensor("y", (1, channels, 56, 56)),
    }
    common = {
        "batch": 1,
        "in_channels": channels,
        "out_channels": channels,
        "in_height": 56,
        "in_width": 56,
        "out_height": 56,
        "out_width": 56,
        "kernel_h": 3,
        "kernel_w": 3,
    }
    dense = Operation(
        id="d",
        op_type=OpType.CONV,
        attrs=ConvAttrs(**common, groups=1),
        inputs=("x",),
        weights=("w_dense",),
        outputs=("y",),
    )
    depthwise = Operation(
        id="dw",
        op_type=OpType.CONV,
        attrs=ConvAttrs(**common, groups=channels),
        inputs=("x",),
        weights=("w_dw",),
        outputs=("y",),
    )
    dense_cost, dw_cost = cost_of(dense, tensors), cost_of(depthwise, tensors)
    assert dense_cost.flops == channels * dw_cost.flops
    assert dense_cost.input_bytes == dw_cost.input_bytes
    assert dw_cost.arithmetic_intensity < dense_cost.arithmetic_intensity


# -- embedding, norm, elementwise, pool, custom -----------------------------


def test_embedding_charges_only_the_gathered_rows() -> None:
    """One decode token reads one 4096-element row, not the 128256-row table.

    Charging the table would add 1.05 GB of phantom traffic per token.
    """
    tensors = {
        "table": _weight("table", (128256, 4096)),
        "out": _tensor("out", (1, 4096)),
    }
    op = Operation(
        id="embed",
        op_type=OpType.EMBEDDING,
        attrs=EmbeddingAttrs(tokens=1, width=4096),
        weights=("table",),
        outputs=("out",),
    )
    cost = cost_of(op, tensors)
    assert cost.flops == 0.0
    assert cost.weight_bytes == 4096 * 2
    assert tensors["table"].size_bytes == pytest.approx(1.05e9, rel=0.01)


def test_norm_and_elementwise_are_memory_bound() -> None:
    """Both do a handful of operations per element they read and write."""
    tensors = {
        "x": _tensor("x", (2048, 4096)),
        "w": _weight("w", (4096,)),
        "y": _tensor("y", (2048, 4096)),
    }
    norm = Operation(
        id="n",
        op_type=OpType.NORM,
        attrs=NormAttrs(rows=2048, width=4096, flops_per_element=4.0),
        inputs=("x",),
        weights=("w",),
        outputs=("y",),
    )
    add = Operation(
        id="e",
        op_type=OpType.ELEMENTWISE,
        attrs=ElementwiseAttrs(elements=2048 * 4096, n_inputs=1, flops_per_element=1.0),
        inputs=("x",),
        outputs=("y",),
    )
    assert cost_of(norm, tensors).flops == 2048 * 4096 * 4
    assert cost_of(norm, tensors).arithmetic_intensity < 2.0
    assert cost_of(add, tensors).arithmetic_intensity < 1.0


def test_pool_and_custom() -> None:
    tensors = {
        "x": _tensor("x", (1, 64, 14, 14)),
        "y": _tensor("y", (1, 64, 7, 7)),
        "cw": _weight("cw", (1000,)),
    }
    pool = Operation(
        id="p",
        op_type=OpType.POOL,
        attrs=PoolAttrs(batch=1, channels=64, out_height=7, out_width=7, kernel_h=2, kernel_w=2),
        inputs=("x",),
        outputs=("y",),
    )
    assert cost_of(pool, tensors).flops == 1 * 64 * 7 * 7 * 4 * 1.0

    custom = Operation(
        id="c",
        op_type=OpType.CUSTOM,
        attrs=CustomAttrs(flops=1e12, bytes=5e9),
        weights=("cw",),
        outputs=("y",),
    )
    cost = cost_of(custom, tensors)
    assert cost.flops == 1e12
    # The declared figure is the op's *total* traffic; weight bytes are carved out
    # of it so residency still applies, rather than added on top.
    assert cost.total_bytes == pytest.approx(5e9, rel=1e-9)
    assert cost.weight_bytes == tensors["cw"].size_bytes


# -- the registry -----------------------------------------------------------


def test_every_op_type_has_a_cost_model() -> None:
    for op_type in OpType:
        assert cost_model_for(op_type) is not None


def test_unregistered_family_error_lists_what_exists() -> None:
    class Fake:
        value = "quantum_flux"

    with pytest.raises(KeyError, match=r"registered families are"):
        cost_model_for(Fake())  # type: ignore[arg-type]


def test_duplicate_registration_is_refused() -> None:
    """A second model for a family is a bug, not an override."""
    with pytest.raises(RuntimeError, match=r"already registered"):

        @register_op(OpType.MATMUL)
        class Second(OperatorCostModel):
            def cost(self, op: Operation, tensors: dict[str, Tensor]) -> OpCost:  # type: ignore[override]
                return OpCost(flops=0.0)


def test_costs_add() -> None:
    a = OpCost(flops=1.0, weight_bytes=2.0, input_bytes=3.0, output_bytes=4.0, scratch_bytes=5.0)
    assert (a + a).total_bytes == 2 * a.total_bytes
    assert (a + a).flops == 2.0


def test_zero_byte_op_is_compute_bound_not_a_division_error() -> None:
    assert OpCost(flops=1.0).arithmetic_intensity == float("inf")


def test_wrong_attrs_type_names_both() -> None:
    tensors = {"y": _tensor("y", (1, 1))}
    op = Operation(
        id="oops",
        op_type=OpType.MATMUL,
        attrs=CustomAttrs(flops=1.0, bytes=1.0),
        outputs=("y",),
    )
    with pytest.raises(TypeError, match=r"carries CustomAttrs; expected MatmulAttrs"):
        cost_of(op, tensors)
