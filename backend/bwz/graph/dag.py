"""DAG utilities: topological order, critical path, activation liveness.

All three take the graph and, where they need one, a caller-supplied per-op
weight. Keeping cost out of this module is deliberate: at M2 the only weight
available is FLOPs, at M3 it is predicted latency, and the algorithms are the
same either way.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence

from bwz.graph.ops import ComputeGraph, Operation, Tensor, TensorKind

OpWeight = Callable[[Operation], float]


def _producers(graph: ComputeGraph) -> dict[str, str]:
    """Tensor name -> id of the operation that writes it."""
    return {name: op.id for op in graph.ops for name in op.outputs}


def dependencies(graph: ComputeGraph) -> dict[str, tuple[str, ...]]:
    """Operation id -> ids of the operations it consumes output from.

    Weights are excluded: a weight has no producer inside the graph, so it
    creates no edge.
    """
    producer = _producers(graph)
    result: dict[str, tuple[str, ...]] = {}
    for op in graph.ops:
        upstream: list[str] = []
        for name in op.inputs:
            source = producer.get(name)
            if source is not None and source != op.id and source not in upstream:
                upstream.append(source)
        result[op.id] = tuple(upstream)
    return result


def topological_order(graph: ComputeGraph) -> tuple[Operation, ...]:
    """Kahn's algorithm, ties broken by the builder's emission order.

    Deterministic ordering matters: a snapshot test over a schedule must not
    depend on set iteration order.
    """
    deps = dependencies(graph)
    by_id = {op.id: op for op in graph.ops}
    remaining = {op_id: set(upstream) for op_id, upstream in deps.items()}
    consumers: dict[str, list[str]] = defaultdict(list)
    for op_id, upstream in deps.items():
        for source in upstream:
            consumers[source].append(op_id)

    ready = [op.id for op in graph.ops if not remaining[op.id]]
    ordered: list[Operation] = []
    while ready:
        current = ready.pop(0)
        ordered.append(by_id[current])
        for downstream in consumers[current]:
            remaining[downstream].discard(current)
            if not remaining[downstream]:
                ready.append(downstream)
    if len(ordered) != len(graph.ops):
        stuck = sorted(set(by_id) - {op.id for op in ordered})
        raise ValueError(
            f"graph {graph.name!r} contains a cycle; operations that never became ready: {stuck}"
        )
    return tuple(ordered)


def longest_path(graph: ComputeGraph, weight: OpWeight) -> tuple[tuple[str, ...], float]:
    """The critical path and its total weight.

    Standard DAG longest-path over a topological order: the cost of reaching an
    operation is its own weight plus the maximum over its predecessors. With
    ``weight`` = predicted latency this is the serial lower bound on runtime,
    which is what M3's scheduler compares its overlap model against.
    """
    order = topological_order(graph)
    deps = dependencies(graph)
    best: dict[str, float] = {}
    previous: dict[str, str | None] = {}
    for op in order:
        upstream = deps[op.id]
        if upstream:
            source = max(upstream, key=lambda name: best[name])
            best[op.id] = best[source] + weight(op)
            previous[op.id] = source
        else:
            best[op.id] = weight(op)
            previous[op.id] = None
    if not best:
        return (), 0.0
    end = max(best, key=lambda name: best[name])
    path: list[str] = []
    cursor: str | None = end
    while cursor is not None:
        path.append(cursor)
        cursor = previous[cursor]
    return tuple(reversed(path)), best[end]


def liveness(graph: ComputeGraph) -> dict[str, tuple[int, int]]:
    """Activation tensor name -> ``(first_write_index, last_read_index)``.

    Indices are positions in the topological order. Weights and KV cache are
    excluded: both are live for the whole run, so an interval says nothing about
    them.
    """
    order = topological_order(graph)
    position = {op.id: index for index, op in enumerate(order)}
    intervals: dict[str, tuple[int, int]] = {}
    for op in order:
        index = position[op.id]
        for name in op.outputs:
            if graph.tensors[name].kind is not TensorKind.ACTIVATION:
                continue
            start, _ = intervals.get(name, (index, index))
            intervals[name] = (min(start, index), index)
        for name in op.inputs:
            if graph.tensors[name].kind is not TensorKind.ACTIVATION:
                continue
            start, end = intervals.get(name, (index, index))
            intervals[name] = (start, max(end, index))
    return intervals


def peak_activation_bytes(graph: ComputeGraph) -> float:
    """Largest total activation footprint at any point in the topological order.

    This is the number that decides whether a configuration fits, alongside
    weights and KV cache, in M3's memory planner. It assumes an allocator that
    frees a tensor the instant its last reader completes — an optimistic bound,
    recorded as an assumption where it is consumed.
    """
    intervals = liveness(graph)
    if not intervals:
        return 0.0
    order = topological_order(graph)
    live_at: list[float] = []
    for index in range(len(order)):
        total = sum(
            graph.tensors[name].size_bytes
            for name, (start, end) in intervals.items()
            if start <= index <= end
        )
        live_at.append(total)
    return max(live_at)


def activation_tensors(graph: ComputeGraph) -> tuple[Tensor, ...]:
    return tuple(t for t in graph.tensors.values() if t.kind is TensorKind.ACTIVATION)


def total_weight(ops: Sequence[Operation], weight: OpWeight) -> float:
    """Sum of a per-op weight — the fully serial cost, for comparison with the path."""
    return sum(weight(op) for op in ops)


def group_by_layer(graph: ComputeGraph) -> Mapping[int | None, tuple[Operation, ...]]:
    """Operations bucketed by ``layer``, for per-layer reporting tables."""
    grouped: dict[int | None, list[Operation]] = defaultdict(list)
    for op in graph.ops:
        grouped[op.layer].append(op)
    return {layer: tuple(ops) for layer, ops in grouped.items()}
