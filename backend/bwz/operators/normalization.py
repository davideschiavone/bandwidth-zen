"""Normalisation cost model. ``docs/MODEL.md`` §3.3."""

from __future__ import annotations

from bwz.graph.ops import NormAttrs, Operation, OpType
from bwz.operators.base import (
    OpCost,
    OperatorCostModel,
    TensorTable,
    _attrs,
    _bytes_of,
    register_op,
)


@register_op(OpType.NORM)
class NormCost(OperatorCostModel):
    """LayerNorm, RMSNorm and BatchNorm, fused or not.

    ``flops = rows * width * flops_per_element``, where the per-element count is a
    modelling convention set in ``calibration.py`` (4 for RMSNorm, 6 for
    LayerNorm) rather than a measurement.

    Normalisation is never the bottleneck by FLOPs and always memory-bound: it
    reads and writes the full activation for a handful of operations per element,
    so its arithmetic intensity is around 1. It earns its place in the graph
    because that traffic is real, not because the arithmetic matters.
    """

    def cost(self, op: Operation, tensors: TensorTable) -> OpCost:
        attrs = _attrs(op, NormAttrs)
        return OpCost(
            flops=float(attrs.rows) * attrs.width * attrs.flops_per_element,
            weight_bytes=_bytes_of(op.weights, tensors),
            input_bytes=_bytes_of(op.inputs, tensors),
            output_bytes=_bytes_of(op.outputs, tensors),
        )
