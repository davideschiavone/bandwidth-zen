"""Expand a :class:`GemmSpec` into a one-operation graph.

The whole builder is a single :class:`Operation`, which is the point: a bare GEMM
is the smallest thing that exercises every layer below it — the matmul cost
model, the tail-effect utilisation, the memory planner, the roofline — with
nothing else in the way to explain a number away.

Three tensors, each operand once (``docs/MODEL.md`` §3.1):

- weights ``[K, N]`` at the weight dtype
- input ``[M, K]`` at the activation dtype
- output ``[M, N]`` at the activation dtype

Quantised inference is the reason the operands differ in dtype: a W8A16 GEMM
moves half as many weight bytes as activation bytes per element, and since the
weight operand dominates traffic whenever ``N`` is large, that is where the win
comes from.
"""

from __future__ import annotations

from bwz.graph.ops import (
    ComputeGraph,
    GraphPhase,
    MatmulAttrs,
    Operation,
    OpType,
    Tensor,
    TensorKind,
)
from bwz.spec.deployment import DeploymentSpec
from bwz.spec.model_spec import GemmSpec


def build_gemm_graph(model: GemmSpec, deployment: DeploymentSpec) -> ComputeGraph:
    """One matmul, with its three operands as tensors."""
    w_dtype = deployment.precision.weights
    a_dtype = deployment.precision.activations
    m, n, k = model.m, model.n, model.k

    tensors = {
        "gemm.x": Tensor("gemm.x", (m, k), a_dtype, TensorKind.ACTIVATION),
        "gemm.w": Tensor("gemm.w", (k, n), w_dtype, TensorKind.WEIGHT),
        "gemm.out": Tensor("gemm.out", (m, n), a_dtype, TensorKind.ACTIVATION),
    }
    op = Operation(
        id="gemm",
        op_type=OpType.MATMUL,
        attrs=MatmulAttrs(m=m, n=n, k=k),
        inputs=("gemm.x",),
        weights=("gemm.w",),
        outputs=("gemm.out",),
        layer=0,
    )
    return ComputeGraph(
        name=f"{model.id}.static", phase=GraphPhase.STATIC, ops=(op,), tensors=tensors
    )
