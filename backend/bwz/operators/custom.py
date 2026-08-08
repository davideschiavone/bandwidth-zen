"""User-supplied ops: the cost is whatever the profile declared."""

from __future__ import annotations

from bwz.graph.ops import CustomAttrs, Operation, OpType
from bwz.operators.base import (
    OpCost,
    OperatorCostModel,
    TensorTable,
    _attrs,
    _bytes_of,
    register_op,
)


@register_op(OpType.CUSTOM)
class CustomCost(OperatorCostModel):
    """Pass-through for ``family: custom`` profiles.

    The declared ``bytes`` figure is treated as input traffic, with weight bytes
    taken from the registered weight tensors so that residency still applies.
    Nothing is derived, nothing is checked — the author owns these numbers, and
    a report built on them says so.
    """

    def cost(self, op: Operation, tensors: TensorTable) -> OpCost:
        attrs = _attrs(op, CustomAttrs)
        weight_bytes = _bytes_of(op.weights, tensors)
        return OpCost(
            flops=attrs.flops,
            weight_bytes=weight_bytes,
            input_bytes=max(attrs.bytes - weight_bytes, 0.0),
        )
