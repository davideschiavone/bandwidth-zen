"""The written arithmetic must agree with the arithmetic the engine used.

`explain` writes its expressions by hand and takes its numbers from `cost_of`,
so the two can drift. `check()` is the guard; this runs it over every operation
of every shipped profile, in both phases.
"""

from __future__ import annotations

import pytest

from bwz.explain import check, explain_graph
from bwz.graph import build_graphs
from bwz.spec import DeploymentSpec, available_models, load_model


@pytest.mark.parametrize("model_id", available_models())
def test_every_explanation_matches_the_cost_model(model_id: str) -> None:
    model = load_model(model_id)
    deployment = DeploymentSpec.model_validate({"batch": 1, "input_tokens": 8, "output_tokens": 1})
    for graph in build_graphs(model, deployment).values():
        assert check(graph) == ()


def test_a_matmul_reads_as_its_formula() -> None:
    """The 4x8x8 projection of the hand-countable encoder, written out."""
    from bwz.graph import GraphPhase, build_graph

    deployment = DeploymentSpec.model_validate(
        {"batch": 1, "input_tokens": 4, "output_tokens": 0, "phase": "prefill"}
    )
    graph = build_graph(load_model("single_layer_encoder"), deployment, GraphPhase.PREFILL)
    q_proj = next(e for e in explain_graph(graph) if e.op_id == "layer0.q_proj")

    assert q_proj.shapes == "A[4,8] x B[8,8] -> C[4,8]"
    assert q_proj.arithmetic == "2·M·N·K = 2·4·8·8 = 512"
    assert q_proj.flops == 512
    # The loop nest carries the real extents, so it can be pasted and run.
    assert "for (k = 0; k < 8; ++k)" in q_proj.code
    assert "/*" in q_proj.code and "/* */" not in q_proj.code


def test_pseudo_c_has_no_nested_comments() -> None:
    """C has no nested block comments, and this is meant to compile if pasted."""
    deployment = DeploymentSpec.model_validate({"batch": 1, "input_tokens": 8, "output_tokens": 1})
    for model_id in available_models():
        for graph in build_graphs(load_model(model_id), deployment).values():
            for explanation in explain_graph(graph):
                body = explanation.code
                depth = 0
                for index in range(len(body) - 1):
                    if body[index : index + 2] == "/*":
                        depth += 1
                        assert depth == 1, f"{explanation.op_id}: nested comment"
                    elif body[index : index + 2] == "*/":
                        depth -= 1
