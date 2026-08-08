"""Convolution and pooling. Direct/im2col only; Winograd and FFT are M8.

``docs/MODEL.md`` §3.4.
"""

from __future__ import annotations

from bwz.calibration import POOL_FLOPS_PER_WINDOW_ELEMENT
from bwz.graph.ops import ConvAttrs, Operation, OpType, PoolAttrs
from bwz.operators.base import (
    OpCost,
    OperatorCostModel,
    TensorTable,
    _attrs,
    _bytes_of,
    register_op,
)


@register_op(OpType.CONV)
class ConvCost(OperatorCostModel):
    """Direct convolution::

        flops = 2 * batch * out_h * out_w * out_channels
                  * (in_channels / groups) * kernel_h * kernel_w

    im2col performs the same arithmetic with a different data layout, so it shares
    this model; the layout cost shows up as activation traffic at M3, not as
    FLOPs. Winograd and FFT genuinely reduce the arithmetic and are deferred to
    M8 (``docs/CORRECTIONS.md`` D5).

    **Depthwise convolution is the interesting case.** With
    ``groups == in_channels == out_channels`` the ``in_channels/groups`` factor
    collapses to 1, so FLOPs fall by a factor of ``out_channels`` while the
    activation traffic is unchanged. Arithmetic intensity drops by the same
    factor, which is exactly why MobileNetV3's depthwise layers are memory-bound
    on a large systolic array (CLAUDE.md sanity checks) — the array has nothing
    to chew on between loads.
    """

    def cost(self, op: Operation, tensors: TensorTable) -> OpCost:
        attrs = _attrs(op, ConvAttrs)
        macs = (
            float(attrs.batch)
            * attrs.out_height
            * attrs.out_width
            * attrs.out_channels
            * (attrs.in_channels / attrs.groups)
            * attrs.kernel_h
            * attrs.kernel_w
        )
        return OpCost(
            flops=2.0 * macs,
            weight_bytes=_bytes_of(op.weights, tensors),
            input_bytes=_bytes_of(op.inputs, tensors),
            output_bytes=_bytes_of(op.outputs, tensors),
        )


@register_op(OpType.POOL)
class PoolCost(OperatorCostModel):
    """Max/average pooling: one comparison or accumulate per window element."""

    def cost(self, op: Operation, tensors: TensorTable) -> OpCost:
        attrs = _attrs(op, PoolAttrs)
        window = float(attrs.kernel_h) * attrs.kernel_w
        elements_out = float(attrs.batch) * attrs.channels * attrs.out_height * attrs.out_width
        return OpCost(
            flops=elements_out * window * POOL_FLOPS_PER_WINDOW_ELEMENT,
            input_bytes=_bytes_of(op.inputs, tensors),
            output_bytes=_bytes_of(op.outputs, tensors),
        )
