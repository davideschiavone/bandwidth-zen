"""Expand a :class:`CustomSpec` — a declared op list — into a graph.

There is nothing to derive: the author supplied FLOPs and bytes. The builder's
only job is to give each op a weight tensor so that residency and capacity still
apply, and to chain them so the DAG utilities have something to walk.
"""

from __future__ import annotations

from bwz.graph.ops import (
    ComputeGraph,
    CustomAttrs,
    GraphPhase,
    Operation,
    OpType,
    Tensor,
    TensorKind,
)
from bwz.spec.deployment import DeploymentSpec
from bwz.spec.dtypes import bytes_per_element
from bwz.spec.model_spec import CustomSpec


def build_custom_graph(model: CustomSpec, deployment: DeploymentSpec) -> ComputeGraph:
    tensors: dict[str, Tensor] = {}
    ops: list[Operation] = []
    w_dtype = deployment.precision.weights
    a_dtype = deployment.precision.activations
    element_bytes = bytes_per_element(w_dtype)

    previous: str | None = None
    for index, spec_op in enumerate(model.ops):
        weights: tuple[str, ...] = ()
        if spec_op.weight_bytes > 0:
            name = f"{spec_op.name}.w"
            elements = int(spec_op.weight_bytes / element_bytes)
            tensors[name] = Tensor(name, (elements,), w_dtype, TensorKind.WEIGHT)
            weights = (name,)
        out = f"{spec_op.name}.out"
        tensors[out] = Tensor(out, (1,), a_dtype, TensorKind.ACTIVATION)
        ops.append(
            Operation(
                id=spec_op.name,
                op_type=OpType.CUSTOM,
                attrs=CustomAttrs(flops=spec_op.flops, bytes=spec_op.bytes),
                inputs=() if previous is None else (previous,),
                weights=weights,
                outputs=(out,),
                layer=index,
            )
        )
        previous = out

    return ComputeGraph(
        name=f"{model.id}.static", phase=GraphPhase.STATIC, ops=tuple(ops), tensors=tensors
    )
