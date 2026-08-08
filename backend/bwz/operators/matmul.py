"""GEMM cost model. See ``docs/MODEL.md`` §3.1."""

from __future__ import annotations

from bwz.graph.ops import MatmulAttrs, Operation, OpType
from bwz.operators.base import (
    OpCost,
    OperatorCostModel,
    TensorTable,
    _attrs,
    _bytes_of,
    register_op,
)


@register_op(OpType.MATMUL)
class MatmulCost(OperatorCostModel):
    """``[M, K] x [K, N] -> [M, N]``.

    ``flops = 2 * M * N * K`` — one multiply and one add per MAC, and the factor
    of two appears here because the op *performs* two operations per MAC. That is
    unrelated to ``hardware_spec.peak_flops_per_s``, which doubles a chip's MAC
    *rate*; the two are the same convention applied on opposite sides of the
    roofline (CLAUDE.md #5).

    Compulsory traffic is each operand once:

    - weights ``K x N`` at the weight dtype
    - input ``M x K`` at the activation dtype
    - output ``M x N`` at the activation dtype

    Re-reads forced by tiling are a hardware question and belong to M3.

    Worked example (Llama-3-8B Q projection, decode, batch 1): ``M=1, K=4096,
    N=4096`` gives ``2 * 1 * 4096 * 4096 = 33.55 MFLOP`` against 33.55 MB of fp16
    weights — an arithmetic intensity of almost exactly 1 FLOP/byte, which is why
    batch-1 decode is memory-bound on every chip ever built.
    """

    def cost(self, op: Operation, tensors: TensorTable) -> OpCost:
        attrs = _attrs(op, MatmulAttrs)
        return OpCost(
            flops=2.0 * attrs.m * attrs.n * attrs.k,
            weight_bytes=_bytes_of(op.weights, tensors),
            input_bytes=_bytes_of(op.inputs, tensors),
            output_bytes=_bytes_of(op.outputs, tensors),
        )
