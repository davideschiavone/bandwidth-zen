"""Expand a :class:`MatmulSpec` into a one-operation graph.

The whole builder is a single :class:`Operation`, which is the point: a bare
matmul is the smallest thing that exercises every layer below it — the matmul
cost model, the tail-effect utilisation, the memory planner, the roofline — with
nothing else in the way to explain a number away.

Three tensors, each operand once (``docs/MODEL.md`` §3.1), and each at its own
width:

- ``A`` is ``[M, K]`` at ``a_dtype``
- ``B`` is ``[K, N]`` at ``b_dtype``
- ``C`` is ``[M, N]`` at ``result_dtype`` — the accumulator, routinely wider
  than either operand

The result width is why this builder ignores ``deployment.precision``: that
field names weights and activations, which is transformer vocabulary and has no
meaning for a bare matmul (``docs/CORRECTIONS.md`` D18).

``B`` is tagged ``WEIGHT`` rather than ``ACTIVATION`` only so the residency model
sees it — the memory planner asks "what could stay on chip across the run", and
for a matmul that is the ``K x N`` operand. Nothing about the arithmetic depends
on the tag.
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
from bwz.spec.model_spec import MatmulSpec


def build_matmul_graph(model: MatmulSpec, deployment: DeploymentSpec) -> ComputeGraph:
    """One matmul, with its three operands as tensors."""
    del deployment  # a bare matmul names its own widths; see the module docstring
    m, n, k = model.m, model.n, model.k

    tensors = {
        "matmul.a": Tensor("matmul.a", (m, k), model.a_dtype, TensorKind.ACTIVATION),
        "matmul.b": Tensor("matmul.b", (k, n), model.b_dtype, TensorKind.WEIGHT),
        "matmul.c": Tensor("matmul.c", (m, n), model.result_dtype, TensorKind.ACTIVATION),
    }
    op = Operation(
        id="matmul",
        op_type=OpType.MATMUL,
        attrs=MatmulAttrs(m=m, n=n, k=k),
        inputs=("matmul.a",),
        weights=("matmul.b",),
        outputs=("matmul.c",),
        layer=0,
    )
    return ComputeGraph(
        name=f"{model.id}.static", phase=GraphPhase.STATIC, ops=(op,), tensors=tensors
    )
