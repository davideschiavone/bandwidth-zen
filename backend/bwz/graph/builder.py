"""``ModelSpec + DeploymentSpec -> ComputeGraph``, dispatched on model family."""

from __future__ import annotations

from bwz.graph.cnn import build_cnn_graph
from bwz.graph.custom import build_custom_graph
from bwz.graph.matmul import build_matmul_graph
from bwz.graph.ops import ComputeGraph, GraphPhase
from bwz.graph.transformer import build_transformer_graph
from bwz.spec.deployment import DeploymentSpec, Phase
from bwz.spec.loaders import AnyModelSpec
from bwz.spec.model_spec import CNNSpec, CustomSpec, MatmulSpec, TransformerSpec


def phases_for(model: AnyModelSpec, deployment: DeploymentSpec) -> tuple[GraphPhase, ...]:
    """Which graphs this configuration needs.

    A transformer with ``phase: both`` needs two; a CNN or custom op list has a
    single static graph regardless of what the deployment asks for.
    """
    if not isinstance(model, TransformerSpec):
        return (GraphPhase.STATIC,)
    if deployment.phase is Phase.PREFILL:
        return (GraphPhase.PREFILL,)
    if deployment.phase is Phase.DECODE:
        return (GraphPhase.DECODE,)
    return (GraphPhase.PREFILL, GraphPhase.DECODE)


def build_graph(
    model: AnyModelSpec, deployment: DeploymentSpec, phase: GraphPhase = GraphPhase.STATIC
) -> ComputeGraph:
    """Build one graph. Use :func:`phases_for` to learn which phases apply."""
    if isinstance(model, TransformerSpec):
        return build_transformer_graph(model, deployment, phase)
    if isinstance(model, CNNSpec):
        return build_cnn_graph(model, deployment)
    if isinstance(model, MatmulSpec):
        return build_matmul_graph(model, deployment)
    if isinstance(model, CustomSpec):
        return build_custom_graph(model, deployment)
    raise TypeError(f"no graph builder for model type {type(model).__name__}")


def build_graphs(model: AnyModelSpec, deployment: DeploymentSpec) -> dict[GraphPhase, ComputeGraph]:
    """Every graph this configuration needs, keyed by phase."""
    return {phase: build_graph(model, deployment, phase) for phase in phases_for(model, deployment)}
