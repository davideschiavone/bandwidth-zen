"""Model spec to DAG of operations.

``graph`` sits between ``spec`` and ``operators`` in the one-way chain
``spec -> graph -> operators -> analysis -> report -> {api, cli}`` (CLAUDE.md).
It knows shapes; it does not know costs, and it must never import from
``analysis``.
"""

from __future__ import annotations

from bwz.graph.builder import build_graph, build_graphs, phases_for
from bwz.graph.cnn import build_cnn_graph
from bwz.graph.custom import build_custom_graph
from bwz.graph.dag import (
    dependencies,
    group_by_layer,
    liveness,
    longest_path,
    peak_activation_bytes,
    topological_order,
)
from bwz.graph.ops import (
    AttentionAttrs,
    ComputeGraph,
    ConvAttrs,
    CustomAttrs,
    ElementwiseAttrs,
    EmbeddingAttrs,
    GraphPhase,
    MatmulAttrs,
    NormAttrs,
    OpAttrs,
    Operation,
    OpType,
    PoolAttrs,
    Tensor,
    TensorKind,
)
from bwz.graph.transformer import build_transformer_graph

__all__ = [
    "AttentionAttrs",
    "ComputeGraph",
    "ConvAttrs",
    "CustomAttrs",
    "ElementwiseAttrs",
    "EmbeddingAttrs",
    "GraphPhase",
    "MatmulAttrs",
    "NormAttrs",
    "OpAttrs",
    "OpType",
    "Operation",
    "PoolAttrs",
    "Tensor",
    "TensorKind",
    "build_cnn_graph",
    "build_custom_graph",
    "build_graph",
    "build_graphs",
    "build_transformer_graph",
    "dependencies",
    "group_by_layer",
    "liveness",
    "longest_path",
    "peak_activation_bytes",
    "phases_for",
    "topological_order",
]
