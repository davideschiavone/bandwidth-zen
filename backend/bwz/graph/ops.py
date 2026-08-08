"""The vocabulary a compute graph is written in: tensors, operations, attributes.

Everything here is a frozen dataclass (CLAUDE.md: frozen dataclasses for internal
value objects). A graph is a value — building one twice from the same spec gives
two equal graphs, which is what makes the analysis core reproducible.

**Shapes carry no cost.** An :class:`Operation` says what is computed and over
what shape; how many FLOPs and bytes that costs is the job of ``operators/``,
which is the only layer that knows the arithmetic. That separation is why a new
operator family is one file plus a registration, not a change to the builder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from bwz.spec.dtypes import DType, bytes_per_element


class OpType(StrEnum):
    """Operation families. Each has exactly one cost model in ``operators/``."""

    MATMUL = "matmul"
    ATTENTION = "attention"
    CONV = "conv"
    POOL = "pool"
    NORM = "norm"
    ELEMENTWISE = "elementwise"
    EMBEDDING = "embedding"
    CUSTOM = "custom"


class TensorKind(StrEnum):
    """What a tensor is, which decides whether it is resident or transient.

    ``WEIGHT`` lives for the whole run and is what the M3 residency model sizes.
    ``ACTIVATION`` lives between its producer and its last consumer, and is what
    the liveness analysis in :mod:`bwz.graph.dag` peaks over. ``KV_CACHE`` grows
    with the context and persists across decode steps, so it is neither.
    """

    ACTIVATION = "activation"
    WEIGHT = "weight"
    KV_CACHE = "kv_cache"


class GraphPhase(StrEnum):
    """Prefill and decode are different machines (CLAUDE.md #6), so they are
    different graphs — not one graph with a flag."""

    PREFILL = "prefill"
    DECODE = "decode"
    STATIC = "static"


@dataclass(frozen=True, slots=True)
class Tensor:
    """A named, shaped, typed buffer."""

    name: str
    shape: tuple[int, ...]
    dtype: DType
    kind: TensorKind

    @property
    def elements(self) -> int:
        return math.prod(self.shape) if self.shape else 0

    @property
    def size_bytes(self) -> float:
        """Storage in bytes. Float because int4 is half a byte per element."""
        return self.elements * bytes_per_element(self.dtype)


# -- per-family attributes --------------------------------------------------
#
# A typed attribute object per family beats a dict: the cost model gets the
# fields it needs with no lookups, and mypy catches a builder that forgets one.


@dataclass(frozen=True, slots=True)
class MatmulAttrs:
    """``[M, K] x [K, N] -> [M, N]``.

    ``M`` folds batch and sequence together: a prefill projection over batch B and
    S tokens has ``M = B*S``; the same projection at decode has ``M = B``. That
    single number is the difference between a compute-bound and a memory-bound
    operation, which is why prefill and decode get separate graphs.
    """

    m: int
    n: int
    k: int


@dataclass(frozen=True, slots=True)
class AttentionAttrs:
    """Scaled dot-product attention over ``q_len`` queries and ``kv_len`` keys.

    ``heads`` is the number of *query* heads and drives FLOPs; ``kv_heads`` is the
    number of distinct key/value heads and drives KV bytes. GQA makes them differ,
    which is the whole point of GQA.
    """

    batch: int
    heads: int
    kv_heads: int
    head_dim: int
    q_len: int
    kv_len: int
    causal: bool
    materialize_scores: bool
    """Vanilla attention writes the ``q_len x kv_len`` score matrix to memory and
    reads it back; FlashAttention tiles it in registers and never does. This flag
    is the entire difference — FLOPs are identical either way (CLAUDE.md)."""


@dataclass(frozen=True, slots=True)
class NormAttrs:
    """Normalisation over ``rows`` vectors of ``width`` elements."""

    rows: int
    width: int
    flops_per_element: float


@dataclass(frozen=True, slots=True)
class ElementwiseAttrs:
    """Pointwise op over ``elements``, reading ``n_inputs`` tensors."""

    elements: int
    n_inputs: int
    flops_per_element: float


@dataclass(frozen=True, slots=True)
class ConvAttrs:
    """A 2-D convolution. ``groups == in_channels == out_channels`` is depthwise."""

    batch: int
    in_channels: int
    out_channels: int
    in_height: int
    in_width: int
    out_height: int
    out_width: int
    kernel_h: int
    kernel_w: int
    groups: int

    @property
    def is_depthwise(self) -> bool:
        return self.groups == self.in_channels and self.groups == self.out_channels


@dataclass(frozen=True, slots=True)
class PoolAttrs:
    """Spatial reduction over a ``kernel_h x kernel_w`` window."""

    batch: int
    channels: int
    out_height: int
    out_width: int
    kernel_h: int
    kernel_w: int


@dataclass(frozen=True, slots=True)
class EmbeddingAttrs:
    """Table lookup: ``tokens`` rows of ``width`` gathered from a ``vocab``-row table.

    Only the gathered rows are *traffic*; the whole table is *footprint*. Keeping
    those apart is what stops decode from appearing to stream a 128k-row
    embedding table for every token.
    """

    tokens: int
    width: int


@dataclass(frozen=True, slots=True)
class CustomAttrs:
    """User-declared cost, straight from a ``family: custom`` profile."""

    flops: float
    bytes: float


OpAttrs = (
    MatmulAttrs
    | AttentionAttrs
    | NormAttrs
    | ElementwiseAttrs
    | ConvAttrs
    | PoolAttrs
    | EmbeddingAttrs
    | CustomAttrs
)


@dataclass(frozen=True, slots=True)
class Operation:
    """One node of the DAG."""

    id: str
    op_type: OpType
    attrs: OpAttrs
    inputs: tuple[str, ...] = ()
    weights: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    layer: int | None = None
    """Which transformer block or CNN stage this belongs to; None for pre/post ops."""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Operation.id must be non-empty")


@dataclass(frozen=True, slots=True)
class ComputeGraph:
    """A DAG of operations over a tensor table.

    Construction validates that every referenced tensor exists and that no tensor
    has two producers — a builder bug otherwise surfaces much later as a wrong
    number rather than as an error.
    """

    name: str
    phase: GraphPhase
    ops: tuple[Operation, ...]
    tensors: dict[str, Tensor]

    def __post_init__(self) -> None:
        producers: dict[str, str] = {}
        for op in self.ops:
            for name in (*op.inputs, *op.weights, *op.outputs):
                if name not in self.tensors:
                    raise ValueError(
                        f"operation {op.id!r} references unknown tensor {name!r}; "
                        f"the builder must register every tensor it names"
                    )
            for name in op.outputs:
                if name in producers:
                    raise ValueError(
                        f"tensor {name!r} is produced by both {producers[name]!r} and "
                        f"{op.id!r}; each activation must have a single producer"
                    )
                producers[name] = op.id

    def op(self, op_id: str) -> Operation:
        for op in self.ops:
            if op.id == op_id:
                return op
        raise KeyError(f"no operation {op_id!r} in graph {self.name!r}")

    def weight_tensors(self) -> tuple[Tensor, ...]:
        """Distinct weight tensors, in first-use order.

        Distinct matters: a weight used by several operations is stored once, so
        summing per-op weight bytes would double-count the footprint.
        """
        seen: dict[str, Tensor] = {}
        for op in self.ops:
            for name in op.weights:
                seen.setdefault(name, self.tensors[name])
        return tuple(seen.values())

    def parameter_count(self) -> int:
        """Total weight elements. Cross-checked against ``ModelSpec.parameter_count()``."""
        return sum(t.elements for t in self.weight_tensors())

    def weight_footprint_bytes(self) -> float:
        """Total weight storage — the ``W`` of the M3 residency model."""
        return sum(t.size_bytes for t in self.weight_tensors())

    def kv_cache_bytes(self) -> float:
        """Total KV-cache storage at this graph's context length."""
        return sum(t.size_bytes for t in self.tensors.values() if t.kind is TensorKind.KV_CACHE)

    def ops_of_type(self, op_type: OpType) -> tuple[Operation, ...]:
        return tuple(op for op in self.ops if op.op_type is op_type)
