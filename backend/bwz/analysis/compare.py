"""Chip-versus-chip comparison: head-to-head, prefill curves, crossover points.

``docs/MODEL.md`` §6.6. Pure functions over :func:`bwz.analysis.analyze`, so they
inherit its determinism and can be swept in parallel.

The crossover is the question these exist to answer: below some prompt length one
chip wins and above it the other does, because prefill trades a fixed weight-load
cost against a compute cost that grows with ``S``. Naming that length is more
useful than either chip's number alone.
"""

from __future__ import annotations

from dataclasses import dataclass

from bwz.analysis import analyze
from bwz.graph.ops import GraphPhase
from bwz.report import Bound, Report
from bwz.spec.deployment import DeploymentSpec, Phase
from bwz.spec.hardware_spec import HardwareSpec
from bwz.spec.loaders import AnyModelSpec


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    """One model on one chip, reduced to the figures a comparison table shows."""

    model_id: str
    chip_id: str
    parameter_count: float
    resident_fraction: float
    ttft_s: float | None
    tokens_per_s: float | None
    bound: Bound
    utilization: float
    feasible: bool


@dataclass(frozen=True, slots=True)
class Crossover:
    """Where two chips swap places as prompt length grows."""

    tokens: int | None
    """``None`` when one chip wins across the whole searched range."""
    faster_below: str
    faster_above: str
    searched_lo: int
    searched_hi: int


def _row(model: AnyModelSpec, chip: HardwareSpec, report: Report) -> ComparisonRow:
    summary = report.summary
    decode = report.phase(GraphPhase.DECODE)
    prefill = report.phase(GraphPhase.PREFILL)
    # Decode is the headline regime for a transformer; a CNN has only STATIC.
    headline = decode or prefill or report.phase(GraphPhase.STATIC)
    return ComparisonRow(
        model_id=model.id,
        chip_id=chip.id,
        parameter_count=float(model.parameter_count()) if report.feasible else 0.0,
        resident_fraction=report.memory.resident_fraction,
        ttft_s=prefill.latency_s if prefill else None,
        tokens_per_s=summary.tokens_per_s if summary else None,
        bound=headline.bound if headline is not None else Bound.DRAM_BW_BOUND,
        utilization=headline.utilization if headline is not None else 0.0,
        feasible=report.feasible,
    )


def head_to_head(
    models: list[AnyModelSpec], chips: list[HardwareSpec], deployment: DeploymentSpec
) -> list[ComparisonRow]:
    """Every (model, chip) pair, in the order given."""
    return [
        _row(model, chip, analyze(model, chip, deployment)) for model in models for chip in chips
    ]


def ttft_at(
    model: AnyModelSpec, chip: HardwareSpec, deployment: DeploymentSpec, tokens: int
) -> float:
    """Time to first token for a prompt of *tokens*, in seconds.

    Infeasible configurations return infinity so that a crossover search treats
    them as "loses everywhere" rather than raising.
    """
    probe = deployment.model_copy(
        update={"input_tokens": tokens, "phase": Phase.PREFILL, "output_tokens": 0}
    )
    report = analyze(model, chip, probe)
    prefill = report.phase(GraphPhase.PREFILL)
    return prefill.latency_s if prefill else float("inf")


def prefill_curve(
    model: AnyModelSpec,
    chip_a: HardwareSpec,
    chip_b: HardwareSpec,
    deployment: DeploymentSpec,
    token_counts: list[int],
) -> list[tuple[int, float, float]]:
    """``(tokens, ttft_a, ttft_b)`` at each requested prompt length."""
    return [
        (
            tokens,
            ttft_at(model, chip_a, deployment, tokens),
            ttft_at(model, chip_b, deployment, tokens),
        )
        for tokens in token_counts
    ]


def prefill_crossover(
    model: AnyModelSpec,
    chip_a: HardwareSpec,
    chip_b: HardwareSpec,
    deployment: DeploymentSpec,
    *,
    lo: int = 1,
    hi: int = 100_000,
) -> Crossover:
    """Bisect for the prompt length at which the faster chip changes.

    Bisection is valid because ``ttft_a(S) - ttft_b(S)`` is monotone in ``S`` over
    the regime that matters: both chips pay a fixed weight-load cost and a compute
    cost rising with ``S``, so their difference crosses zero at most once. If the
    sign is the same at both ends there is no crossover and one chip wins
    throughout — reported as ``tokens=None`` rather than a fabricated number.
    """

    def difference(tokens: int) -> float:
        return ttft_at(model, chip_a, deployment, tokens) - ttft_at(
            model, chip_b, deployment, tokens
        )

    low_diff, high_diff = difference(lo), difference(hi)
    winner_low = chip_a.id if low_diff < 0 else chip_b.id
    winner_high = chip_a.id if high_diff < 0 else chip_b.id

    if (low_diff < 0) == (high_diff < 0):
        return Crossover(
            tokens=None,
            faster_below=winner_low,
            faster_above=winner_high,
            searched_lo=lo,
            searched_hi=hi,
        )

    left, right = lo, hi
    while right - left > 1:
        middle = (left + right) // 2
        if (difference(middle) < 0) == (low_diff < 0):
            left = middle
        else:
            right = middle
    return Crossover(
        tokens=right,
        faster_below=winner_low,
        faster_above=winner_high,
        searched_lo=lo,
        searched_hi=hi,
    )


def decode_vs_bandwidth(
    model: AnyModelSpec,
    chip: HardwareSpec,
    deployment: DeploymentSpec,
    bandwidths_bytes_per_s: list[float],
) -> list[tuple[float, float]]:
    """``(bandwidth, tokens_per_s)`` — should come out close to linear when DRAM-bound.

    Departure from linearity is itself informative: it means another term has
    started to bind, and the point where the curve bends is the point where
    buying more bandwidth stops paying.
    """
    out: list[tuple[float, float]] = []
    for bandwidth in bandwidths_bytes_per_s:
        levels = list(chip.memory)
        levels[-1] = levels[-1].model_copy(update={"bandwidth_bytes_per_s": bandwidth})
        variant = chip.model_copy(update={"memory": levels})
        probe = deployment.model_copy(update={"phase": Phase.DECODE})
        summary = analyze(model, variant, probe).summary
        out.append((bandwidth, summary.tokens_per_s if summary and summary.tokens_per_s else 0.0))
    return out
