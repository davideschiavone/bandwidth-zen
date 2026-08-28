"""Deployment spec: how a model is being run on a chip.

Schema reference: ``docs/SCHEMA.md`` §3; source schema: ``PROMPT.md`` §4.3.

Validation here is **intra-spec only**. Whether the chip actually supports the
requested dtype, or whether the weights fit in its memory, is a *feasibility*
question answered by ``analysis/memory.py`` at M3, which returns a ``Report``
with ``feasible: false`` and the cheapest fixes — never an exception
(CLAUDE.md #8).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from bwz.spec.base import SpecModel
from bwz.spec.dtypes import DType
from bwz.spec.hardware_spec import Dataflow
from bwz.spec.quantities import Bytes, Fraction, Seconds, Watts


class Mode(StrEnum):
    INFERENCE = "inference"
    TRAINING = "training"


class Phase(StrEnum):
    """Prefill and decode are different machines (CLAUDE.md #6) and are costed separately."""

    PREFILL = "prefill"
    DECODE = "decode"
    BOTH = "both"


class AttentionImpl(StrEnum):
    """FlashAttention changes bytes, never FLOPs (CLAUDE.md sanity checks)."""

    VANILLA = "vanilla"
    FLASH2 = "flash2"
    PAGED = "paged"
    SLIDING_WINDOW = "sliding_window"


class SparsityType(StrEnum):
    NONE = "none"
    STRUCTURED_2_4 = "structured_2_4"
    UNSTRUCTURED = "unstructured"


class OptimizeFor(StrEnum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ENERGY = "energy"


class AStrategy(StrEnum):
    """How operand A is loaded for a lone matmul (``docs/CORRECTIONS.md`` D33/D36).

    A byte-amount knob: every tile of a k-slice's group reads the same staged
    slice, so how large that residency is changes how many times A crosses DRAM.
    """

    STAGE = "stage"
    """Fetch each k-slice once, resident for all ``NTILES_PER_KS`` of its tiles
    (D33). A crosses DRAM exactly once."""
    STREAM = "stream"
    """Fetch per tile (D31) — the honest setting for a GPU, whose GEMMs
    genuinely re-read operands: ``NTILES_PER_KS`` times the staged traffic."""
    WHOLE = "whole"
    """Stage all of A before the first tile. Same total bytes as ``stage`` — A
    still crosses DRAM exactly once — only the timing changes: one ramp instead
    of a staging event per k-slice."""


class BDataflow(StrEnum):
    """When B's array write lands relative to compute (``docs/CORRECTIONS.md`` D33).

    A timing knob only: within one pass every B tile is fetched exactly once
    whatever this is set to (D30) — v1 has no on-chip write-bandwidth term to
    charge (D5b), so the choice moves where the write sits in the loop nest, not
    how many bytes cross. ``persistent`` is the exception across *iterations*:
    a weight set that stays resident need not be rewritten on a repeat pass.
    """

    WRITE_AHEAD = "write-ahead"
    """The reload lands in the weight set freed a wave earlier, hidden behind
    that wave's compute (D33, current default)."""
    ON_DEMAND = "on-demand"
    """Written at compute, exposed on the critical path."""
    PERSISTENT = "persistent"
    """Loaded once, never displaced. Needs ``tiles <= units * weight_sets``."""


class Precision(SpecModel):
    """Per-tensor-class numeric formats."""

    weights: DType = DType.FP16
    activations: DType = DType.FP16
    accumulate: DType | None = None
    """``None`` means *derive it from the operands*, which is the honest default:
    the accumulator width is a property of the arithmetic, not a knob, and
    ``spec.dtypes.accumulator_for`` is the one place that mapping lives. It used
    to default to ``fp32`` for every dtype, which said an ``int8 x int8`` matmul
    accumulated in floating point (docs/CORRECTIONS.md D56). Set it to override —
    ``fp16`` is a real MMA mode, just not anyone's default."""
    kv_cache: DType = DType.FP16


class PrecisionOverride(SpecModel):
    """Per-layer override; any field left None inherits the global :class:`Precision`."""

    weights: DType | None = None
    activations: DType | None = None
    accumulate: DType | None = None
    kv_cache: DType | None = None


class Sparsity(SpecModel):
    """Weight sparsity. ``ratio`` is the fraction of weights *removed*."""

    type: SparsityType = SparsityType.NONE
    ratio: Fraction | None = None
    applies_to: list[str] = Field(default_factory=lambda: ["matmul"])

    @model_validator(mode="after")
    def _check_ratio(self) -> Sparsity:
        if self.type is SparsityType.NONE and self.ratio is not None:
            raise ValueError(
                "sparsity.ratio is set but sparsity.type is 'none'; set a type "
                "(structured_2_4, unstructured) or remove the ratio"
            )
        if self.type is SparsityType.STRUCTURED_2_4 and self.ratio not in (None, 0.5):
            raise ValueError(
                f"sparsity.type 'structured_2_4' implies ratio 0.5, got {self.ratio}; "
                f"use 'unstructured' for other ratios"
            )
        return self


class Parallelism(SpecModel):
    """Sharding degrees. Unused until M5, validated here so configs are complete."""

    tp: int = Field(default=1, gt=0, description="tensor parallel")
    pp: int = Field(default=1, gt=0, description="pipeline parallel")
    dp: int = Field(default=1, gt=0, description="data parallel")
    ep: int = Field(default=1, gt=0, description="expert parallel")
    microbatches: int = Field(default=1, gt=0)

    @property
    def total_chips(self) -> int:
        return self.tp * self.pp * self.dp * self.ep


class Constraints(SpecModel):
    """Hard limits a configuration must satisfy to be reported feasible."""

    max_latency_s: Seconds | None = None
    max_power_w: Watts | None = None
    max_memory_bytes: Bytes | None = None


class DeploymentSpec(SpecModel):
    """A single point in the configuration space."""

    mode: Mode = Mode.INFERENCE
    phase: Phase = Phase.BOTH
    batch: int = Field(default=1, gt=0)
    input_tokens: int = Field(default=2048, gt=0)
    output_tokens: int = Field(default=256, ge=0)
    kv_context_tokens: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Context length at which to size the KV cache. Defaults to "
            "input_tokens + output_tokens, i.e. the end-of-generation worst case."
        ),
    )
    precision: Precision = Precision()
    per_layer_precision_overrides: dict[str, PrecisionOverride] = Field(default_factory=dict)
    sparsity: Sparsity = Sparsity()
    attention_impl: AttentionImpl = AttentionImpl.FLASH2
    parallelism: Parallelism = Parallelism()
    num_chips: int = Field(default=1, gt=0)
    optimize_for: OptimizeFor = OptimizeFor.LATENCY
    constraints: Constraints = Constraints()
    stationarity: Dataflow | None = Field(
        default=None,
        description=(
            "Which operand stays resident, deciding the whole decomposition "
            "(D53): input=A, weight=B, output=C. None uses the chip's own "
            "declared dataflow, which for an NVIDIA/AMD matrix core is 'os' — "
            "what cuBLAS does. Asking for one the unit cannot run is REFUSED "
            "with feasible: false, not clamped: a clamp would silently answer a "
            "different question than the one asked."
        ),
    )
    split_k: int = Field(
        default=1,
        gt=0,
        description=(
            "Cut the contraction into this many independent slices, for when the "
            "output grid alone cannot fill the chip. Only meaningful under "
            "stationarity=os, where it is the one thing that makes a reduction "
            "necessary; ws/is already put K on the grid. Costs CUTLASS's second "
            "kernel — see report.assumptions (D53)."
        ),
    )
    a_strategy: AStrategy = AStrategy.STAGE
    b_dataflow: BDataflow = BDataflow.WRITE_AHEAD
    a_residency_tiles: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Override how many of a k-slice's NTILES_PER_KS tiles one A staging "
            "serves, under a_strategy stage/whole. Must be a power-of-2 divisor "
            "of NTILES_PER_KS; a value that is not gets clamped to the largest "
            "one that is (CLAUDE.md #8), noted in report.assumptions. Meaningless "
            "under stream, which always fetches per tile."
        ),
    )
    a_prefetch_depth: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Override the double-buffered staging depth used to schedule A's "
            "k-slice fetches. Schedule-only: changes the drawn pipeline fill/"
            "drain, never a byte count. None keeps today's depth, derived from "
            "whether on-chip capacity fits two tiles."
        ),
    )
    iterations: int = Field(
        default=1,
        gt=0,
        description=(
            "Invocations this report represents. analyze() still costs one "
            "report, not N runs; only b_dataflow=persistent reads this, to "
            "amortise B's write over a resident weight set that a repeat "
            "invocation would not have to rewrite (docs/CORRECTIONS.md D33)."
        ),
    )

    @model_validator(mode="after")
    def _check_chip_count(self) -> DeploymentSpec:
        product = self.parallelism.total_chips
        if product != self.num_chips:
            p = self.parallelism
            raise ValueError(
                f"parallelism tp={p.tp} x pp={p.pp} x dp={p.dp} x ep={p.ep} = {product} does not "
                f"equal num_chips={self.num_chips}; set num_chips to {product} or adjust the "
                f"parallelism degrees"
            )
        return self

    @model_validator(mode="after")
    def _check_phase(self) -> DeploymentSpec:
        if self.phase in (Phase.DECODE, Phase.BOTH) and self.output_tokens == 0:
            raise ValueError(
                f"phase={self.phase.value!r} requires output_tokens > 0, got 0; use "
                f"phase='prefill' to cost the prompt alone"
            )
        if self.mode is Mode.TRAINING and self.phase is not Phase.BOTH:
            raise ValueError(
                f"mode='training' has no prefill/decode split; phase must be 'both', got "
                f"{self.phase.value!r}"
            )
        return self

    @model_validator(mode="after")
    def _check_pipeline_microbatches(self) -> DeploymentSpec:
        if self.parallelism.pp > 1 and self.parallelism.microbatches < self.parallelism.pp:
            raise ValueError(
                f"pipeline parallelism pp={self.parallelism.pp} with only "
                f"{self.parallelism.microbatches} microbatches leaves the pipeline mostly empty; "
                f"microbatches must be at least pp"
            )
        return self

    @property
    def context_tokens(self) -> int:
        """Context length used to size the KV cache."""
        if self.kv_context_tokens is not None:
            return self.kv_context_tokens
        return self.input_tokens + self.output_tokens

    def runs_prefill(self) -> bool:
        return self.phase in (Phase.PREFILL, Phase.BOTH)

    def runs_decode(self) -> bool:
        return self.phase in (Phase.DECODE, Phase.BOTH)
