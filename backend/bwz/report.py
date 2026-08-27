"""The ``Report``: the single contract between engine, CLI and UI.

Schema source: ``PROMPT.md`` §5. Everything downstream — ``bwz run``, the M4
frontend, the validation harness — consumes this and nothing else.

Two fields are mandatory on every report and are the honesty mechanism of the
whole tool (CLAUDE.md #4):

``assumptions``
    Every modelling shortcut taken, and every estimated input the analysis
    actually touched. If a number rests on a guess, the report says which guess.

``flip_margins``
    How far the binding input can move before the verdict changes. A bottleneck
    label with a 1.1x margin means something very different from one with a 260x
    margin, and a reader cannot tell them apart from the label alone.

``meta.generated_at`` is deliberately **not** set by ``analyze()``: the analysis
core reads no wall clock (CLAUDE.md #3), so the same inputs always produce the
same report. The CLI stamps it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from bwz.graph.ops import GraphPhase, OpType


class Bound(StrEnum):
    """What limits a phase. Three terms, per the v1 machine model (D5a)."""

    DRAM_BW_BOUND = "DRAM_BW_BOUND"
    COMPUTE_BOUND = "COMPUTE_BOUND"
    LATENCY_BOUND = "LATENCY_BOUND"


class Confidence(StrEnum):
    """How much weight a number deserves.

    ``LOW`` until Session 5 fits the calibration constants against published
    reference points — every prediction currently rests on documented defaults
    that nobody has checked against a measurement.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class OpResult:
    """One operation's predicted cost."""

    op_id: str
    op_type: OpType
    layer: int | None
    flops: float
    weight_bytes: float
    dram_bytes: float
    arithmetic_intensity: float
    utilization: float
    """Fraction of peak the operand shape allows — the tail effect."""
    t_dram_s: float
    t_compute_s: float
    t_fixed_s: float
    latency_s: float
    bound: Bound
    dram_read_bytes: float = 0.0
    """Operands fetched from DRAM."""
    dram_write_bytes: float = 0.0
    """Results written back. Split from reads because the two are not
    interchangeable: a write happens *after* the arithmetic that produced it, and
    a result nothing on chip consumes must reach DRAM whatever the capacity."""
    dram_weight_read_bytes: float = 0.0
    """The share of :attr:`dram_read_bytes` that is operand B — the stationary
    operand a weight-stationary array holds. Split out because the two operands
    are governed by *different* residency fractions and spill at different
    times: on-chip capacity is granted to activations before weights (D15), so
    B is always the first to stream. A single LOAD figure hides which one is
    crossing the bus, and on a comparison that is the question."""
    dram_activation_read_bytes: float = 0.0
    """The share of :attr:`dram_read_bytes` that is operand A plus scratch — the
    operand that streams *through* the array rather than sitting in it."""
    dram_reduction_bytes: float = 0.0
    """Split-K only: the partial results' round trip, counted in
    :attr:`dram_bytes` and in **neither** :attr:`dram_read_bytes` nor
    :attr:`dram_write_bytes`. Those two are the operands' traffic — A and B in,
    C out — and this is neither: it is one kernel's output read back as the
    next kernel's input, half write and half read (D53). Kept apart so the
    three still sum to :attr:`dram_bytes` and the trace can draw the second
    kernel as the separate thing it is."""
    t_reduce_s: float = 0.0
    """Split-K only: the reduction kernel's additions, on the vector unit. Part
    of :attr:`t_compute_s`, broken out so the trace can draw the second kernel
    as the separate thing it is."""


@dataclass(frozen=True, slots=True)
class FlipMargin:
    """How far an input can move before the bound changes."""

    quantity: str
    bound: Bound
    runner_up: Bound
    margin: float
    """Ratio of the binding term to the next one. 1.0 means a coin flip."""
    description: str
    rests_on_estimate: bool


@dataclass(frozen=True, slots=True)
class MemoryPlan:
    """Capacity waterfall and the residency the on-chip memory buys."""

    weight_bytes: float
    kv_cache_bytes: float
    peak_activation_bytes: float
    total_bytes: float
    dram_capacity_bytes: float
    usable_dram_bytes: float
    on_chip_capacity_bytes: float
    resident_fraction: float
    """``r`` — the weight share that need not be re-streamed from DRAM. Computed
    against the capacity left after the double buffer and the activation working
    set have been served, because both save more traffic per byte than weights."""
    activation_resident_fraction: float
    """Share of the peak activation working set held on chip. At 1.0 activations
    cost no DRAM traffic at all, which is the normal case for decode and the
    usual case for prefill on a chip with tens of MB of SRAM."""
    double_buffered: bool
    """Whether on-chip capacity has room for two tiles, which decides whether the
    phase time is ``max(load, compute)`` or ``load + compute`` (D5a)."""
    fits: bool


@dataclass(frozen=True, slots=True)
class PhaseResult:
    """Aggregate cost of one graph — a prefill pass or one decode step."""

    phase: GraphPhase
    latency_s: float
    flops: float
    dram_bytes: float
    dram_read_bytes: float
    dram_write_bytes: float
    t_dram_s: float
    t_compute_s: float
    t_fixed_s: float
    bound: Bound
    achieved_flops_per_s: float
    utilization: float
    """Achieved throughput as a fraction of the chip's peak for this dtype."""
    n_ops: int
    n_dispatched_ops: int
    ops: tuple[OpResult, ...] = field(default=())
    dram_weight_read_bytes: float = 0.0
    """Operand B's share of ``dram_read_bytes``; see :class:`OpResult`."""
    dram_activation_read_bytes: float = 0.0
    """Operand A's (plus scratch) share of ``dram_read_bytes``."""


@dataclass(frozen=True, slots=True)
class Summary:
    """The numbers a user reads first."""

    latency_s: float
    ttft_s: float | None
    tpot_s: float | None
    tokens_per_s: float | None
    throughput_per_s: float
    peak_flops_per_s: float
    achieved_flops_per_s: float
    utilization: float
    bound: Bound


@dataclass(frozen=True, slots=True)
class Meta:
    model_name: str
    chip_name: str
    config_hash: str
    bwz_version: str
    generated_at: str | None = None
    """Set by the CLI, never by ``analyze()`` — the core reads no wall clock."""


@dataclass(frozen=True, slots=True)
class Report:
    """The whole output of ``analyze(model, hardware, deployment)``."""

    meta: Meta
    feasible: bool
    memory: MemoryPlan
    summary: Summary | None
    phases: tuple[PhaseResult, ...]
    assumptions: tuple[str, ...]
    flip_margins: tuple[FlipMargin, ...]
    confidence: Confidence
    infeasibility: tuple[str, ...] = ()
    """Why the configuration cannot run, and the cheapest fixes. Empty when feasible."""

    def phase(self, phase: GraphPhase) -> PhaseResult | None:
        for result in self.phases:
            if result.phase is phase:
                return result
        return None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict. Enums render as their values."""
        return asdict(self)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


def config_hash(*parts: object) -> str:
    """Stable short hash of the inputs, for cache keys and report identity.

    Deterministic across processes: ``hashlib`` over a sorted JSON rendering,
    never ``hash()``, whose salt changes per interpreter.
    """
    payload = json.dumps([str(part) for part in parts], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
