"""Topological order, critical path and activation liveness."""

from __future__ import annotations

import pytest

from bwz.graph import (
    ComputeGraph,
    ElementwiseAttrs,
    GraphPhase,
    Operation,
    OpType,
    Tensor,
    TensorKind,
    build_graph,
    dependencies,
    liveness,
    longest_path,
    peak_activation_bytes,
    topological_order,
)
from bwz.spec import DeploymentSpec, load_model
from bwz.spec.dtypes import DType


def _chain(n: int, *, cyclic: bool = False) -> ComputeGraph:
    """``op0 -> op1 -> ... -> op{n-1}``, each tensor 1000 fp16 elements."""
    tensors = {
        f"t{i}": Tensor(f"t{i}", (1000,), DType.FP16, TensorKind.ACTIVATION) for i in range(n + 1)
    }
    ops = []
    for i in range(n):
        source = f"t{n}" if cyclic and i == 0 else f"t{i}"
        ops.append(
            Operation(
                id=f"op{i}",
                op_type=OpType.ELEMENTWISE,
                attrs=ElementwiseAttrs(elements=1000, n_inputs=1, flops_per_element=1.0),
                inputs=(source,),
                outputs=(f"t{i + 1}",),
            )
        )
    return ComputeGraph(name="chain", phase=GraphPhase.STATIC, ops=tuple(ops), tensors=tensors)


def test_topological_order_of_a_chain_is_the_chain() -> None:
    graph = _chain(5)
    assert [op.id for op in topological_order(graph)] == ["op0", "op1", "op2", "op3", "op4"]


def test_topological_order_is_deterministic() -> None:
    """Snapshot tests over a schedule must not depend on set iteration order."""
    model = load_model("llama3_8b")
    graph = build_graph(
        model, DeploymentSpec(batch=1, input_tokens=8, output_tokens=1), GraphPhase.PREFILL
    )
    first = [op.id for op in topological_order(graph)]
    assert first == [op.id for op in topological_order(graph)]
    assert len(first) == len(graph.ops)


def test_a_cycle_is_reported_with_the_stuck_operations() -> None:
    graph = _chain(3, cyclic=True)
    with pytest.raises(ValueError, match=r"contains a cycle; operations that never became ready"):
        topological_order(graph)


def test_weights_create_no_edges() -> None:
    """A weight has no producer, so it must not appear as a dependency."""
    model = load_model("llama3_8b")
    graph = build_graph(
        model, DeploymentSpec(batch=1, input_tokens=8, output_tokens=1), GraphPhase.PREFILL
    )
    deps = dependencies(graph)
    assert deps["embed"] == (), "the embedding reads only its table"
    assert "layer0.attn_norm" in deps["layer0.q_proj"]


def test_longest_path_over_a_chain_is_the_whole_chain() -> None:
    graph = _chain(4)
    path, total = longest_path(graph, weight=lambda op: 1.0)
    assert path == ("op0", "op1", "op2", "op3")
    assert total == 4.0


def test_longest_path_picks_the_heavier_branch() -> None:
    """Two parallel branches; the critical path follows the expensive one."""
    tensors = {
        name: Tensor(name, (10,), DType.FP16, TensorKind.ACTIVATION)
        for name in ("x", "cheap", "expensive", "out")
    }
    attrs = ElementwiseAttrs(elements=10, n_inputs=1, flops_per_element=1.0)
    ops = (
        Operation(
            id="cheap", op_type=OpType.ELEMENTWISE, attrs=attrs, inputs=("x",), outputs=("cheap",)
        ),
        Operation(
            id="expensive",
            op_type=OpType.ELEMENTWISE,
            attrs=attrs,
            inputs=("x",),
            outputs=("expensive",),
        ),
        Operation(
            id="join",
            op_type=OpType.ELEMENTWISE,
            attrs=attrs,
            inputs=("cheap", "expensive"),
            outputs=("out",),
        ),
    )
    graph = ComputeGraph(name="diamond", phase=GraphPhase.STATIC, ops=ops, tensors=tensors)
    weights = {"cheap": 1.0, "expensive": 10.0, "join": 1.0}
    path, total = longest_path(graph, weight=lambda op: weights[op.id])
    assert path == ("expensive", "join")
    assert total == 11.0


def test_liveness_intervals_span_producer_to_last_consumer() -> None:
    graph = _chain(3)
    intervals = liveness(graph)
    assert intervals["t1"] == (0, 1), "written by op0, last read by op1"
    assert intervals["t3"] == (2, 2), "written by op2, never read"


def test_peak_activation_is_below_the_sum_of_all_activations() -> None:
    """A chain holds at most two tensors at once, not all six."""
    graph = _chain(5)
    total = sum(t.size_bytes for t in graph.tensors.values())
    peak = peak_activation_bytes(graph)
    assert peak < total
    assert peak == pytest.approx(2 * 1000 * 2, rel=1e-9)


def test_weights_and_kv_are_excluded_from_liveness() -> None:
    """Both live for the whole run, so an interval would say nothing about them."""
    model = load_model("llama3_8b")
    graph = build_graph(
        model,
        DeploymentSpec(batch=1, input_tokens=1, output_tokens=1, kv_context_tokens=1024),
        GraphPhase.DECODE,
    )
    intervals = liveness(graph)
    assert not any(name.endswith(".w") for name in intervals)
    assert not any(name.endswith("_cache") for name in intervals)
    assert peak_activation_bytes(graph) < graph.weight_footprint_bytes()


def test_peak_activation_grows_with_batch() -> None:
    model = load_model("llama3_8b")
    small = build_graph(
        model, DeploymentSpec(batch=1, input_tokens=512, output_tokens=1), GraphPhase.PREFILL
    )
    large = build_graph(
        model, DeploymentSpec(batch=8, input_tokens=512, output_tokens=1), GraphPhase.PREFILL
    )
    assert peak_activation_bytes(large) == pytest.approx(8 * peak_activation_bytes(small))


# -- graph construction invariants ------------------------------------------


def test_unknown_tensor_reference_is_rejected() -> None:
    op = Operation(
        id="bad",
        op_type=OpType.ELEMENTWISE,
        attrs=ElementwiseAttrs(elements=1, n_inputs=1, flops_per_element=1.0),
        inputs=("ghost",),
        outputs=("out",),
    )
    tensors = {"out": Tensor("out", (1,), DType.FP16, TensorKind.ACTIVATION)}
    with pytest.raises(ValueError, match=r"references unknown tensor 'ghost'"):
        ComputeGraph(name="bad", phase=GraphPhase.STATIC, ops=(op,), tensors=tensors)


def test_two_producers_for_one_tensor_is_rejected() -> None:
    attrs = ElementwiseAttrs(elements=1, n_inputs=1, flops_per_element=1.0)
    tensors = {name: Tensor(name, (1,), DType.FP16, TensorKind.ACTIVATION) for name in ("x", "out")}
    ops = (
        Operation(id="a", op_type=OpType.ELEMENTWISE, attrs=attrs, inputs=("x",), outputs=("out",)),
        Operation(id="b", op_type=OpType.ELEMENTWISE, attrs=attrs, inputs=("x",), outputs=("out",)),
    )
    with pytest.raises(ValueError, match=r"produced by both 'a' and 'b'"):
        ComputeGraph(name="dup", phase=GraphPhase.STATIC, ops=ops, tensors=tensors)


def test_weight_tensors_are_deduplicated() -> None:
    """A tied embedding is used twice and must be counted once."""
    model = load_model("gemma3_4b")
    graph = build_graph(
        model, DeploymentSpec(batch=1, input_tokens=8, output_tokens=1), GraphPhase.PREFILL
    )
    names = [t.name for t in graph.weight_tensors()]
    assert len(names) == len(set(names))
    assert names.count("embed_tokens") == 1
