"""Operator cost models: FLOPs and byte counts for one operation.

Formulas are documented in ``docs/MODEL.md`` §3 and each model's docstring states
its own, per CLAUDE.md style.

**Scope boundary.** These models are *hardware-independent*. They report the
arithmetic an operation performs and the compulsory traffic it generates —
each weight read once, each input read once, each output written once. Tiled
re-reads, cache reuse and the systolic tail effect all depend on the chip and
therefore live in ``analysis/`` at M3 (``docs/CORRECTIONS.md`` D10). That keeps
the dependency arrow pointing one way and makes every number here checkable by
hand.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import ClassVar, TypeVar

from bwz.graph.ops import Operation, OpType, Tensor


@dataclass(frozen=True, slots=True)
class OpCost:
    """What one operation costs, before any hardware is considered.

    Bytes are split by role because the M3 roofline treats them differently:
    weight traffic is what residency removes, activation traffic is what tiling
    affects, and scratch is what FlashAttention eliminates.
    """

    flops: float
    weight_bytes: float = 0.0
    input_bytes: float = 0.0
    output_bytes: float = 0.0
    scratch_bytes: float = 0.0
    """Intermediates written to memory and read straight back — the materialised
    attention score matrix, and nothing else in v1."""

    @property
    def total_bytes(self) -> float:
        return self.weight_bytes + self.input_bytes + self.output_bytes + self.scratch_bytes

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs per byte. Compared against the chip's ridge point at M3.

        Zero-byte operations return infinity rather than dividing by zero: an
        operation that touches no memory is compute-bound by definition.
        """
        total = self.total_bytes
        return self.flops / total if total > 0 else float("inf")

    def __add__(self, other: OpCost) -> OpCost:
        return OpCost(
            flops=self.flops + other.flops,
            weight_bytes=self.weight_bytes + other.weight_bytes,
            input_bytes=self.input_bytes + other.input_bytes,
            output_bytes=self.output_bytes + other.output_bytes,
            scratch_bytes=self.scratch_bytes + other.scratch_bytes,
        )


ZERO_COST = OpCost(flops=0.0)

TensorTable = Mapping[str, Tensor]


class OperatorCostModel(ABC):
    """Base for every cost model. One per :class:`OpType`."""

    op_type: ClassVar[OpType]

    @abstractmethod
    def cost(self, op: Operation, tensors: TensorTable) -> OpCost:
        """FLOPs and bytes for *op*, given the graph's tensor table."""


_REGISTRY: dict[OpType, OperatorCostModel] = {}

M = TypeVar("M", bound=OperatorCostModel)
A = TypeVar("A")


def register_op(op_type: OpType) -> Callable[[type[M]], type[M]]:
    """Class decorator registering a cost model for *op_type*.

    Registration is exclusive: two models for one family is a bug, not an
    override, so the second one raises at import time rather than silently
    winning.
    """

    def decorate(model_class: type[M]) -> type[M]:
        if op_type in _REGISTRY:
            existing = type(_REGISTRY[op_type]).__name__
            raise RuntimeError(
                f"cost model for {op_type.value!r} is already registered as {existing}; "
                f"{model_class.__name__} would silently replace it"
            )
        model_class.op_type = op_type
        _REGISTRY[op_type] = model_class()
        return model_class

    return decorate


def cost_model_for(op_type: OpType) -> OperatorCostModel:
    """The registered model for *op_type*, or an actionable error."""
    try:
        return _REGISTRY[op_type]
    except KeyError:
        registered = sorted(t.value for t in _REGISTRY)
        raise KeyError(
            f"no cost model registered for op type {op_type.value!r}; registered families are "
            f"{registered}. Add one in operators/ and decorate it with @register_op."
        ) from None


def cost_of(op: Operation, tensors: TensorTable) -> OpCost:
    """Dispatch *op* to its cost model."""
    return cost_model_for(op.op_type).cost(op, tensors)


def _attrs(op: Operation, attrs_type: type[A]) -> A:
    """Narrow ``op.attrs`` to the type this cost model needs, or say what went wrong.

    Returning the narrowed value rather than just asserting gives every model one
    typed line instead of a check plus an ``assert isinstance`` that mypy needs
    and a reader does not.
    """
    if not isinstance(op.attrs, attrs_type):
        raise TypeError(
            f"operation {op.id!r} is typed {op.op_type.value!r} but carries "
            f"{type(op.attrs).__name__}; expected {attrs_type.__name__}"
        )
    return op.attrs


def _bytes_of(names: tuple[str, ...], tensors: TensorTable) -> float:
    return sum(tensors[name].size_bytes for name in names)
