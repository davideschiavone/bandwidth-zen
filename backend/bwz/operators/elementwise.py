"""Pointwise operations: activations, residual adds, RoPE. ``docs/MODEL.md`` §3.3."""

from __future__ import annotations

from bwz.graph.ops import ElementwiseAttrs, EmbeddingAttrs, Operation, OpType
from bwz.operators.base import (
    OpCost,
    OperatorCostModel,
    TensorTable,
    _attrs,
    _bytes_of,
    register_op,
)


@register_op(OpType.ELEMENTWISE)
class ElementwiseCost(OperatorCostModel):
    """``flops = elements * flops_per_element``; traffic is every operand once.

    The per-element counts are conventions carried on the attributes rather than
    in ``calibration.py``, because they are structural: a residual add is one
    operation per element by definition, and SwiGLU's gate multiply is one more.
    """

    def cost(self, op: Operation, tensors: TensorTable) -> OpCost:
        attrs = _attrs(op, ElementwiseAttrs)
        return OpCost(
            flops=float(attrs.elements) * attrs.flops_per_element,
            weight_bytes=_bytes_of(op.weights, tensors),
            input_bytes=_bytes_of(op.inputs, tensors),
            output_bytes=_bytes_of(op.outputs, tensors),
        )


@register_op(OpType.EMBEDDING)
class EmbeddingCost(OperatorCostModel):
    """Table lookup: no arithmetic, and traffic proportional to *rows gathered*.

    This distinction is load-bearing. A decode step embeds one token, so it reads
    one ``width``-element row — not the whole ``vocab x width`` table. Charging
    the table would make Llama-3-8B decode appear to move an extra 1.05 GB per
    token and triple its predicted latency.

    The table's *footprint* still counts against memory capacity; that is tracked
    on the tensor, not on this cost.
    """

    def cost(self, op: Operation, tensors: TensorTable) -> OpCost:
        attrs = _attrs(op, EmbeddingAttrs)
        table = tensors[op.weights[0]]
        bytes_per_row = table.size_bytes / max(table.shape[0], 1)
        return OpCost(
            flops=0.0,
            weight_bytes=attrs.tokens * bytes_per_row,
            output_bytes=_bytes_of(op.outputs, tensors),
        )
