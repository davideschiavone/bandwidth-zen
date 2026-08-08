"""Attention cost model: vanilla, FlashAttention-2, GQA/MQA. ``docs/MODEL.md`` §3.2."""

from __future__ import annotations

from bwz.calibration import SOFTMAX_FLOPS_PER_SCORE
from bwz.graph.ops import AttentionAttrs, Operation, OpType
from bwz.operators.base import (
    OpCost,
    OperatorCostModel,
    TensorTable,
    _attrs,
    _bytes_of,
    register_op,
)


@register_op(OpType.ATTENTION)
class AttentionCost(OperatorCostModel):
    """Scaled dot-product attention.

    Two matmuls over the score matrix, plus a softmax between them::

        scores  = Q @ K^T     2 * batch * heads * q_len * kv_len * head_dim
        softmax                   c * batch * heads * q_len * kv_len
        out     = A @ V       2 * batch * heads * q_len * kv_len * head_dim

    **Causal masking halves the first and last terms** when ``q_len > 1``: a
    decoder computes the lower triangle only, so roughly ``q_len * (q_len + 1)/2``
    of the ``q_len * kv_len`` positions. The exact triangular fraction is used
    rather than a flat 0.5, so short prompts are right too. At decode ``q_len``
    is 1 and every key is visible, so nothing is halved.

    **GQA affects bytes, not FLOPs.** All ``heads`` query heads participate in
    every score, so the arithmetic is unchanged; only ``kv_heads`` distinct
    key/value heads are read, so the KV traffic shrinks by the group size.

    **FlashAttention changes bytes, never FLOPs** (CLAUDE.md). Vanilla writes the
    ``q_len x kv_len`` score matrix to memory and reads it back; flash tiles it in
    registers. That is the whole difference, and it is expressed here as
    ``scratch_bytes``.

    Worked example (Llama-3-8B prefill, S=2048, batch 1, 32 heads, head_dim 128):
    scores ``2 * 32 * 2048 * 2048 * 128 = 34.36 GFLOP``, halved by causality to
    17.18 GFLOP; the same again for ``A @ V``; softmax
    ``5 * 32 * 2048 * 2048 / 2 = 0.34 GFLOP``. Total 34.7 GFLOP.
    """

    def cost(self, op: Operation, tensors: TensorTable) -> OpCost:
        attrs = _attrs(op, AttentionAttrs)

        positions = float(attrs.q_len) * attrs.kv_len
        if attrs.causal and attrs.q_len > 1:
            # Lower triangle of the q_len x kv_len score block. With kv_len ==
            # q_len this is exactly q_len*(q_len+1)/2; with a prompt appended to
            # existing context the leading full-attention block is added back.
            history = max(attrs.kv_len - attrs.q_len, 0)
            positions = attrs.q_len * history + attrs.q_len * (attrs.q_len + 1) / 2.0

        scored = attrs.batch * attrs.heads * positions
        flops = 2.0 * 2.0 * scored * attrs.head_dim + SOFTMAX_FLOPS_PER_SCORE * scored

        scratch = 0.0
        if attrs.materialize_scores:
            # Written once and read once, at the activation dtype of the output.
            score_dtype_bytes = tensors[op.outputs[0]].size_bytes / max(
                tensors[op.outputs[0]].elements, 1
            )
            scratch = 2.0 * scored * score_dtype_bytes

        return OpCost(
            flops=flops,
            input_bytes=_bytes_of(op.inputs, tensors),
            output_bytes=_bytes_of(op.outputs, tensors),
            scratch_bytes=scratch,
        )
