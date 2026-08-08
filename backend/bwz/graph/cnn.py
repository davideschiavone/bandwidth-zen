"""Expand a :class:`CNNSpec` into a graph, propagating channels and spatial dims.

Channel propagation is what a CNN spec cannot do on its own, and it is why
``CNNSpec.parameter_count()`` raises and defers here: layer *n*'s weight shape
depends on layer *n-1*'s output channels.

``docs/MODEL.md`` §5.
"""

from __future__ import annotations

import math

from bwz.calibration import LAYERNORM_FLOPS_PER_ELEMENT, POOL_FLOPS_PER_WINDOW_ELEMENT
from bwz.graph.ops import (
    ComputeGraph,
    ConvAttrs,
    ElementwiseAttrs,
    GraphPhase,
    MatmulAttrs,
    NormAttrs,
    Operation,
    OpType,
    PoolAttrs,
    Tensor,
    TensorKind,
)
from bwz.spec.deployment import DeploymentSpec
from bwz.spec.model_spec import (
    Activation,
    CNNSpec,
    ConvLayer,
    LinearLayer,
    MBConvLayer,
    PoolKind,
    PoolLayer,
)

# Squeeze-and-excitation reduces the channel count by this factor in its
# bottleneck. 4 is the value used throughout MobileNetV3 (paper §5.1).
SE_REDUCTION = 4


class _Shape:
    """Running activation shape as the builder walks the layer list."""

    def __init__(self, batch: int, channels: int, height: int, width: int) -> None:
        self.batch = batch
        self.channels = channels
        self.height = height
        self.width = width

    def as_tuple(self) -> tuple[int, ...]:
        return (self.batch, self.channels, self.height, self.width)


def _strided(extent: int, stride: int) -> int:
    """Output extent under 'same' padding: ``ceil(extent / stride)``."""
    return max(math.ceil(extent / stride), 1)


class _Builder:
    def __init__(self, model: CNNSpec, deployment: DeploymentSpec) -> None:
        self.model = model
        self.deployment = deployment
        self.w_dtype = deployment.precision.weights
        self.a_dtype = deployment.precision.activations
        self.tensors: dict[str, Tensor] = {}
        self.ops: list[Operation] = []
        self.counter = 0

    def weight(self, name: str, shape: tuple[int, ...]) -> str:
        self.tensors.setdefault(name, Tensor(name, shape, self.w_dtype, TensorKind.WEIGHT))
        return name

    def activation(self, name: str, shape: tuple[int, ...]) -> str:
        self.tensors.setdefault(name, Tensor(name, shape, self.a_dtype, TensorKind.ACTIVATION))
        return name

    def build(self) -> ComputeGraph:
        spec = self.model.input
        batch = self.deployment.batch or spec.batch
        shape = _Shape(batch, spec.channels, spec.height, spec.width)
        current = self.activation("input", shape.as_tuple())

        for index, layer in enumerate(self.model.layers):
            if isinstance(layer, ConvLayer):
                current, shape = self.conv_layer(layer, current, shape, index)
            elif isinstance(layer, MBConvLayer):
                current, shape = self.mbconv(layer, current, shape, index)
            elif isinstance(layer, PoolLayer):
                current, shape = self.pool(layer, current, shape, index)
            else:
                current, shape = self.linear(layer, current, shape, index)

        return ComputeGraph(
            name=f"{self.model.id}.static",
            phase=GraphPhase.STATIC,
            ops=tuple(self.ops),
            tensors=self.tensors,
        )

    # -- layer kinds --------------------------------------------------------

    def conv(
        self,
        op_id: str,
        source: str,
        shape: _Shape,
        out_channels: int,
        kernel: tuple[int, int],
        stride: int,
        *,
        depthwise: bool,
        layer: int,
    ) -> tuple[str, _Shape]:
        groups = shape.channels if depthwise else 1
        kh, kw = kernel
        out_shape = _Shape(
            shape.batch, out_channels, _strided(shape.height, stride), _strided(shape.width, stride)
        )
        weight_name = self.weight(f"{op_id}.w", (out_channels, shape.channels // groups, kh, kw))
        out = self.activation(f"{op_id}.out", out_shape.as_tuple())
        self.ops.append(
            Operation(
                id=op_id,
                op_type=OpType.CONV,
                attrs=ConvAttrs(
                    batch=shape.batch,
                    in_channels=shape.channels,
                    out_channels=out_channels,
                    in_height=shape.height,
                    in_width=shape.width,
                    out_height=out_shape.height,
                    out_width=out_shape.width,
                    kernel_h=kh,
                    kernel_w=kw,
                    groups=groups,
                ),
                inputs=(source,),
                weights=(weight_name,),
                outputs=(out,),
                layer=layer,
            )
        )
        return out, out_shape

    def batchnorm(self, op_id: str, source: str, shape: _Shape, layer: int) -> str:
        weight_name = self.weight(f"{op_id}.bn", (2, shape.channels))
        out = self.activation(f"{op_id}.out", shape.as_tuple())
        self.ops.append(
            Operation(
                id=op_id,
                op_type=OpType.NORM,
                attrs=NormAttrs(
                    rows=shape.batch * shape.height * shape.width,
                    width=shape.channels,
                    flops_per_element=LAYERNORM_FLOPS_PER_ELEMENT,
                ),
                inputs=(source,),
                weights=(weight_name,),
                outputs=(out,),
                layer=layer,
            )
        )
        return out

    def activate(
        self, op_id: str, source: str, shape: _Shape, activation: Activation, layer: int
    ) -> str:
        if activation is Activation.NONE:
            return source
        flops_per_element = {
            Activation.RELU: 1.0,
            Activation.RELU6: 2.0,
            Activation.HARDSWISH: 4.0,  # add, clamp, multiply, scale
            Activation.SWISH: 4.0,
        }[activation]
        elements = shape.batch * shape.channels * shape.height * shape.width
        out = self.activation(f"{op_id}.out", shape.as_tuple())
        self.ops.append(
            Operation(
                id=op_id,
                op_type=OpType.ELEMENTWISE,
                attrs=ElementwiseAttrs(
                    elements=elements, n_inputs=1, flops_per_element=flops_per_element
                ),
                inputs=(source,),
                outputs=(out,),
                layer=layer,
            )
        )
        return out

    def conv_layer(
        self, spec: ConvLayer, source: str, shape: _Shape, index: int
    ) -> tuple[str, _Shape]:
        prefix = f"{index}.{spec.name}"
        depthwise = spec.depthwise or (spec.groups is not None and spec.groups == shape.channels)
        current, shape = self.conv(
            prefix,
            source,
            shape,
            spec.out_channels,
            spec.kernel,
            spec.stride,
            depthwise=depthwise,
            layer=index,
        )
        if spec.norm.value != "none":
            current = self.batchnorm(f"{prefix}.norm", current, shape, index)
        current = self.activate(f"{prefix}.act", current, shape, spec.activation, index)
        return current, shape

    def mbconv(
        self, spec: MBConvLayer, source: str, shape: _Shape, index: int
    ) -> tuple[str, _Shape]:
        """Inverted residual: 1x1 expand, depthwise KxK, optional SE, 1x1 project.

        The expand convolution is **omitted when it would be a no-op** — the first
        MobileNetV3 block has ``expand_channels == in_channels``, and the reference
        implementation skips the projection there rather than emitting an identity
        1x1. Emitting it would add ~4k parameters and a spurious memory-bound op.
        """
        prefix = f"{index}.{spec.name}"
        residual_source = source
        residual_shape = _Shape(shape.batch, shape.channels, shape.height, shape.width)
        current = source

        if spec.expand_channels != shape.channels:
            current, shape = self.conv(
                f"{prefix}.expand",
                current,
                shape,
                spec.expand_channels,
                (1, 1),
                1,
                depthwise=False,
                layer=index,
            )
            current = self.batchnorm(f"{prefix}.expand.norm", current, shape, index)
            current = self.activate(f"{prefix}.expand.act", current, shape, spec.activation, index)

        current, shape = self.conv(
            f"{prefix}.depthwise",
            current,
            shape,
            spec.expand_channels,
            spec.kernel,
            spec.stride,
            depthwise=True,
            layer=index,
        )
        current = self.batchnorm(f"{prefix}.depthwise.norm", current, shape, index)
        current = self.activate(f"{prefix}.depthwise.act", current, shape, spec.activation, index)

        if spec.squeeze_excite:
            current = self.squeeze_excite(prefix, current, shape, index)

        current, shape = self.conv(
            f"{prefix}.project",
            current,
            shape,
            spec.out_channels,
            (1, 1),
            1,
            depthwise=False,
            layer=index,
        )
        current = self.batchnorm(f"{prefix}.project.norm", current, shape, index)

        if spec.stride == 1 and residual_shape.channels == spec.out_channels:
            elements = shape.batch * shape.channels * shape.height * shape.width
            out = self.activation(f"{prefix}.residual.out", shape.as_tuple())
            self.ops.append(
                Operation(
                    id=f"{prefix}.residual",
                    op_type=OpType.ELEMENTWISE,
                    attrs=ElementwiseAttrs(elements=elements, n_inputs=2, flops_per_element=1.0),
                    inputs=(residual_source, current),
                    outputs=(out,),
                    layer=index,
                )
            )
            current = out
        return current, shape

    def squeeze_excite(self, prefix: str, source: str, shape: _Shape, index: int) -> str:
        """Global pool, two 1x1 bottleneck FCs, then a per-channel rescale."""
        squeezed = self.activation(f"{prefix}.se.pool.out", (shape.batch, shape.channels, 1, 1))
        self.ops.append(
            Operation(
                id=f"{prefix}.se.pool",
                op_type=OpType.POOL,
                attrs=PoolAttrs(
                    batch=shape.batch,
                    channels=shape.channels,
                    out_height=1,
                    out_width=1,
                    kernel_h=shape.height,
                    kernel_w=shape.width,
                ),
                inputs=(source,),
                outputs=(squeezed,),
                layer=index,
            )
        )
        bottleneck = max(shape.channels // SE_REDUCTION, 1)
        reduced = self.se_fc(
            f"{prefix}.se.fc1", squeezed, shape.batch, shape.channels, bottleneck, index
        )
        excited = self.se_fc(
            f"{prefix}.se.fc2", reduced, shape.batch, bottleneck, shape.channels, index
        )
        elements = shape.batch * shape.channels * shape.height * shape.width
        out = self.activation(f"{prefix}.se.scale.out", shape.as_tuple())
        self.ops.append(
            Operation(
                id=f"{prefix}.se.scale",
                op_type=OpType.ELEMENTWISE,
                attrs=ElementwiseAttrs(elements=elements, n_inputs=2, flops_per_element=1.0),
                inputs=(source, excited),
                outputs=(out,),
                layer=index,
            )
        )
        return out

    def se_fc(
        self, op_id: str, source: str, batch: int, in_features: int, out_features: int, index: int
    ) -> str:
        weight_name = self.weight(f"{op_id}.w", (in_features, out_features))
        out = self.activation(f"{op_id}.out", (batch, out_features, 1, 1))
        self.ops.append(
            Operation(
                id=op_id,
                op_type=OpType.MATMUL,
                attrs=MatmulAttrs(m=batch, n=out_features, k=in_features),
                inputs=(source,),
                weights=(weight_name,),
                outputs=(out,),
                layer=index,
            )
        )
        return out

    def pool(self, spec: PoolLayer, source: str, shape: _Shape, index: int) -> tuple[str, _Shape]:
        prefix = f"{index}.{spec.name}"
        if spec.kind is PoolKind.GLOBAL_AVG:
            kernel_h, kernel_w = shape.height, shape.width
            out_shape = _Shape(shape.batch, shape.channels, 1, 1)
        else:
            kernel_h, kernel_w = spec.kernel or (2, 2)
            out_shape = _Shape(
                shape.batch,
                shape.channels,
                _strided(shape.height, spec.stride),
                _strided(shape.width, spec.stride),
            )
        out = self.activation(f"{prefix}.out", out_shape.as_tuple())
        self.ops.append(
            Operation(
                id=prefix,
                op_type=OpType.POOL,
                attrs=PoolAttrs(
                    batch=shape.batch,
                    channels=shape.channels,
                    out_height=out_shape.height,
                    out_width=out_shape.width,
                    kernel_h=kernel_h,
                    kernel_w=kernel_w,
                ),
                inputs=(source,),
                outputs=(out,),
                layer=index,
            )
        )
        return out, out_shape

    def linear(
        self, spec: LinearLayer, source: str, shape: _Shape, index: int
    ) -> tuple[str, _Shape]:
        prefix = f"{index}.{spec.name}"
        in_features = shape.channels * shape.height * shape.width
        weight_name = self.weight(f"{prefix}.w", (in_features, spec.out_features))
        out_shape = _Shape(shape.batch, spec.out_features, 1, 1)
        out = self.activation(f"{prefix}.out", (shape.batch, spec.out_features))
        self.ops.append(
            Operation(
                id=prefix,
                op_type=OpType.MATMUL,
                attrs=MatmulAttrs(m=shape.batch, n=spec.out_features, k=in_features),
                inputs=(source,),
                weights=(weight_name,),
                outputs=(out,),
                layer=index,
            )
        )
        current = self.activate(f"{prefix}.act", out, out_shape, spec.activation, index)
        return current, out_shape


def build_cnn_graph(model: CNNSpec, deployment: DeploymentSpec) -> ComputeGraph:
    """Expand *model* into a static graph. CNNs have no prefill/decode split."""
    return _Builder(model, deployment).build()


__all__ = ["POOL_FLOPS_PER_WINDOW_ELEMENT", "SE_REDUCTION", "build_cnn_graph"]
