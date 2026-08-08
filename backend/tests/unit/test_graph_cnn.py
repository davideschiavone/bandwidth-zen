"""CNN expansion: channel propagation, MobileNetV3 goldens, depthwise behaviour."""

from __future__ import annotations

import pytest

from bwz.graph import GraphPhase, OpType, build_graph
from bwz.graph.ops import ConvAttrs
from bwz.operators import cost_of
from bwz.operators.base import OpCost
from bwz.spec import DeploymentSpec, load_model


def _graph(batch: int = 1) -> object:
    model = load_model("mobilenetv3")
    return build_graph(model, DeploymentSpec(batch=batch, input_tokens=1, output_tokens=1))


def _total(graph: object) -> OpCost:
    total = OpCost(0.0)
    for op in graph.ops:  # type: ignore[attr-defined]
        total = total + cost_of(op, graph.tensors)  # type: ignore[attr-defined]
    return total


def test_mobilenetv3_parameter_count() -> None:
    """5.47 M against the paper's 5.4 M (+1.3%).

    The gap is batch-norm scale/shift pairs and the squeeze-excitation FCs, which
    the paper's headline figure does not itemise. Channel propagation through the
    graph is the only way to get this number at all -- ``CNNSpec.parameter_count``
    raises for exactly that reason.
    """
    graph = _graph()
    assert graph.parameter_count() == pytest.approx(5.4e6, rel=0.05)  # type: ignore[attr-defined]


def test_mobilenetv3_multiply_accumulates() -> None:
    """234.8 M MACs against the paper's 219 M (+7.2%).

    ``flops / 2`` because the paper counts MAdds. The excess is the SE branches,
    the classifier head and the batch-norms, whose inclusion the paper's
    accounting does not state. Pinned with a 10% tolerance and a note rather than
    tuned to match a figure whose convention is unknown.
    """
    macs = _total(_graph()).flops / 2.0
    assert macs == pytest.approx(219e6, rel=0.10)


def test_channel_propagation_reaches_the_classifier() -> None:
    """1000-way output from a 3x224x224 input, through 20 declared layers."""
    graph = _graph()
    output = graph.tensors["19.classifier.out"]  # type: ignore[attr-defined]
    assert output.shape == (1, 1000)
    assert graph.tensors["input"].shape == (1, 3, 224, 224)  # type: ignore[attr-defined]


def test_spatial_downsampling_follows_the_strides() -> None:
    """224 -> 112 at the stem, then 56, 28, 14, 7 through the four stride-2 blocks."""
    graph = _graph()
    convs = [
        op.attrs
        for op in graph.ops  # type: ignore[attr-defined]
        if op.op_type is OpType.CONV and isinstance(op.attrs, ConvAttrs)
    ]
    resolutions = sorted({c.out_height for c in convs}, reverse=True)
    assert resolutions == [112, 56, 28, 14, 7]


def test_depthwise_convolutions_are_the_memory_bound_ones() -> None:
    """CLAUDE.md sanity check: MobileNetV3's depthwise layers are memory-bound.

    Two claims, both weaker than "every depthwise beats every dense" -- which is
    false, and worth stating why. Intensity depends on resolution as well as on
    the ``in_channels/groups`` factor, so a 5x5 depthwise over 160 channels at
    7x7 (12.3 FLOP/byte) beats a 1x1 expansion over 16 channels at 112x112
    (8.0 FLOP/byte). The structural claim is about the *distribution*, not about
    every pair:

    1. Depthwise convolutions are markedly less intense on average.
    2. **Every** convolution in the network sits below 65 FLOP/byte — the peak is
       bneck12's 1x1 expansion at 64.4 — against a ridge point of 295 on H100 and
       6168 on chip_a. So the whole network is memory-bound on either, by 4.6x
       and 96x respectively, and the depthwise layers are the worst of it.
    """
    graph = _graph()
    depthwise: list[float] = []
    dense: list[float] = []
    for op in graph.ops:  # type: ignore[attr-defined]
        if op.op_type is not OpType.CONV or not isinstance(op.attrs, ConvAttrs):
            continue
        intensity = cost_of(op, graph.tensors).arithmetic_intensity  # type: ignore[attr-defined]
        (depthwise if op.attrs.is_depthwise else dense).append(intensity)
    assert depthwise and dense
    assert sum(depthwise) / len(depthwise) < sum(dense) / len(dense)
    h100_fp16_ridge = 295.0
    assert max(depthwise + dense) < h100_fp16_ridge / 4.0, "far below any chip's ridge point"
    assert max(depthwise) < 15.0, "depthwise intensity stays in the low teens"


def test_the_first_block_skips_its_no_op_expansion() -> None:
    """bneck1 expands 16 -> 16, so the reference implementation omits the 1x1."""
    graph = _graph()
    ids = {op.id for op in graph.ops}  # type: ignore[attr-defined]
    assert "1.bneck1.expand" not in ids
    assert "2.bneck2.expand" in ids, "bneck2 expands 16 -> 64 and must have one"


def test_residuals_appear_only_where_shape_is_preserved() -> None:
    """A stride-2 block or a channel change cannot carry a residual."""
    graph = _graph()
    residuals = {op.id for op in graph.ops if op.id.endswith(".residual")}  # type: ignore[attr-defined]
    assert "3.bneck3.residual" in residuals, "24 -> 24 at stride 1"
    assert "2.bneck2.residual" not in residuals, "16 -> 24 at stride 2"


def test_squeeze_excite_only_where_declared() -> None:
    graph = _graph()
    ids = {op.id for op in graph.ops}  # type: ignore[attr-defined]
    assert "4.bneck4.se.fc1" in ids
    assert "1.bneck1.se.fc1" not in ids


def test_batch_scales_flops_but_not_parameters() -> None:
    one, eight = _graph(batch=1), _graph(batch=8)
    assert eight.parameter_count() == one.parameter_count()  # type: ignore[attr-defined]
    assert _total(eight).flops == pytest.approx(8 * _total(one).flops)


def test_cnn_graph_is_static() -> None:
    assert _graph().phase is GraphPhase.STATIC  # type: ignore[attr-defined]
