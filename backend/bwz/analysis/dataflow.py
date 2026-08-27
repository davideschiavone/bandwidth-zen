"""How a lone matmul's operands cross DRAM: A's residency, B's write timing.

``docs/CORRECTIONS.md`` D33/D36. Split out of ``analysis/schedule.py`` so that
:mod:`bwz.analysis.pipeline` — which already supplies :func:`tile_count` and
:func:`tiles_per_a_event`, the two divisors every strategy here is built from —
can depend on :class:`DataflowPlan` for its trace-building signature without a
cycle: this module depends on ``pipeline``, never the reverse.

A byte-amount knob and a timing knob, and they must not be fused into one
(``PROMPT.md`` dataflow strategy discussion): A has reuse — every tile of one
grid row reads the same staged slice, whatever that row is a slice *of* (D53) —
so how often that slice is re-staged changes A's DRAM traffic. B has none:
within one pass every tile is fetched exactly once whatever the choice (D30),
so the choice only moves *when*
the write lands relative to compute, not how many bytes cross — except across
*iterations*, where a resident weight set need not be rewritten on a repeat pass.
"""

from __future__ import annotations

from dataclasses import dataclass

from bwz.analysis.pipeline import grid_of, tile_count, tiles_per_a_event
from bwz.analysis.roofline import MachineModel
from bwz.analysis.stationarity import TileGrid
from bwz.graph.ops import Operation
from bwz.spec.deployment import AStrategy, BDataflow, DeploymentSpec
from bwz.spec.hardware_spec import Dataflow, HardwareSpec
from bwz.units import format_bytes


@dataclass(frozen=True, slots=True)
class DataflowPlan:
    """A and B dataflow for a lone matmul, after clamping to what is feasible
    (CLAUDE.md #8: clamp, never raise). Computed once so the bytes ``run_phase``
    charges, the schedule ``build_trace`` draws and the listing ``deploy.py``
    prints cannot disagree.
    """

    grid: TileGrid | None
    """The decomposition the machine's stationarity implies (D53), or ``None``
    without declared array geometry. Every divisor below is read off it, so a
    strategy cannot be planned against a different grid than the one the trace
    draws and the utilisation model costs."""
    stationarity: Dataflow
    """The effective stationarity — the unit's own unless a deployment asked for
    another it supports. An unsupported request never reaches here: it is
    refused with ``feasible: false`` before any planning (D53)."""
    requested_stationarity: Dataflow | None
    """What the deployment asked for, or ``None`` where it took the chip's own.
    Kept apart from :attr:`stationarity` so the drawer can say which of the two
    a number rests on — unlike the A/B knobs there is never a clamp between
    them, so the pair differ only in whether the user chose."""
    split_k: int
    """``k_partitions``: how many independent pieces the contraction is cut into
    beyond the grid. 1 unless ``--split-k`` asked otherwise."""
    a_strategy: AStrategy
    """Effective, after any clamp: ``whole`` falls back to ``stage`` when the
    scratchpad cannot hold all of A — same bytes either way, D33."""
    a_requested: AStrategy
    residency_tiles: int
    """Tiles served by one A staging event: ``tiles_per_a_event`` under
    stage/whole (the default, or a valid override), 1 under stream."""
    tiles_per_a_event: int
    """Width of one grid row — D33's ``NTILES_PER_KS``, generalised: the tiles
    that share one staged slice of A."""
    a_bytes_multiplier: float
    """1.0 for stage/whole — both cross DRAM exactly once (D33); the byte total
    does not depend on whether the staging is spread across row boundaries or
    ramped upfront. ``tiles_per_a_event / residency_tiles`` for stream (D31)."""
    a_bytes: float
    """A's whole footprint — ``M x K x element size`` — independent of strategy."""
    a_events: int
    """A staging events: one per grid row, per split-K piece. ``ceil(K/rows)``
    k-slices under ``ws`` (D33's own count), ``ceil(M/rows)`` row-bands under
    ``os``; 1 without declared array geometry."""
    group_name: str
    """What one grid row is called — ``"k-slice"`` under ``ws``, ``"row-band"``
    under ``os``. Carried so the assumptions drawer and the listing name A's
    staging after the decomposition actually being run."""
    a_bytes_per_event: float
    """Bytes moved by one A staging event: a full band's share under stage
    (``a_bytes / a_events``) scaled by ``residency_tiles / tiles_per_a_event``
    when an override serves less than the whole band per event."""
    b_dataflow: BDataflow
    """Effective, after any clamp: ``persistent`` falls back to ``write-ahead``
    when B does not fit the array's resident tile capacity."""
    b_requested: BDataflow
    tiles: int
    resident_tile_capacity: int
    iterations: int
    b_write_multiplier: float
    """1.0 unless b_dataflow is persistent, B fits, and iterations > 1: then
    1/iterations — the first invocation writes B, the rest reuse it."""
    a_prefetch_depth: int | None
    """Override for the double-buffered staging depth ``build_trace`` schedules A's
    per-row fetches at. Schedule-only — never changes a byte count. ``None`` keeps
    today's depth, derived from whether on-chip capacity fits two tiles."""
    notes: tuple[str, ...]
    """Clamp and amortisation assumptions, appended verbatim to
    ``report.assumptions``."""


def _largest_power_of_two_divisor_at_most(requested: int, n: int) -> int:
    """Largest power of 2 that divides *n* and is at most *requested*.

    Always returns >= 1: 1 divides every positive integer, so the clamp always
    has something to fall back to (CLAUDE.md #8 — clamp, never raise).
    """
    best = 1
    candidate = 1
    while candidate <= n:
        if n % candidate == 0 and candidate <= requested:
            best = candidate
        candidate *= 2
    return best


def plan_dataflow(
    op: Operation,
    machine: MachineModel,
    chip: HardwareSpec,
    deployment: DeploymentSpec,
    *,
    a_bytes: float,
) -> DataflowPlan:
    """Resolve the requested A/B dataflow into what the schedule can actually do.

    Feasibility is a clamp, never an exception (CLAUDE.md #8): a ``whole`` stage
    that does not fit the scratchpad falls back to ``stage`` (identical bytes,
    only the timing reverts), and a ``persistent`` B that does not fit the
    array's resident tile capacity falls back to ``write-ahead``. Both fallbacks
    are named in :attr:`DataflowPlan.notes` rather than happening silently.
    """
    grid = grid_of(op, machine)
    tiles = tile_count(op, machine)
    per_event = tiles_per_a_event(op, machine)
    unit = machine.unit
    notes: list[str] = []

    a_strategy = deployment.a_strategy
    if a_strategy is AStrategy.WHOLE and a_bytes > chip.on_chip_capacity_bytes:
        notes.append(
            f"a_strategy=whole requested but A ({format_bytes(a_bytes)}) does not fit the "
            f"{format_bytes(chip.on_chip_capacity_bytes)} scratchpad; fell back to stage — the "
            f"same {format_bytes(a_bytes)} total, staged per grid row instead of ramped upfront."
        )
        a_strategy = AStrategy.STAGE

    if a_strategy is AStrategy.STREAM:
        residency_tiles = 1
    else:
        requested = deployment.a_residency_tiles
        if requested is None:
            residency_tiles = per_event
        else:
            residency_tiles = _largest_power_of_two_divisor_at_most(requested, per_event)
            if residency_tiles != requested:
                notes.append(
                    f"a_residency_tiles={requested} is not a power-of-2 divisor of "
                    f"TILES_PER_GROUP={per_event}; clamped to {residency_tiles}, the largest "
                    f"one that is."
                )
    a_bytes_multiplier = (per_event / residency_tiles) if residency_tiles > 0 else 1.0

    # One event per grid row (D53): a k-slice under ws, a band of M rows under
    # os. Both stage A exactly once in total — what changes is what a "row" is.
    a_events = grid.a_events if grid is not None else 1
    group_name = grid.group_name if grid is not None else "k-slice"
    a_bytes_per_event = (a_bytes / a_events) * (residency_tiles / per_event)

    b_dataflow = deployment.b_dataflow
    resident_tile_capacity = unit.resident_tile_capacity()
    if b_dataflow is BDataflow.PERSISTENT and tiles > resident_tile_capacity:
        notes.append(
            f"b_dataflow=persistent requested but B is {tiles} tiles against "
            f"{resident_tile_capacity} resident; fell back to write-ahead."
        )
        b_dataflow = BDataflow.WRITE_AHEAD

    # write-ahead/on-demand/persistent are placements of a weight-bank write
    # (D33) that only exists when the array holds resident weight tiles at
    # all (D30) — a chip with weight_sets=1 reads both operands per
    # instruction, so there is nothing to place ahead of, expose on demand,
    # or keep resident. Every shipped GPU profile is weight_sets=1.
    if b_dataflow is not BDataflow.WRITE_AHEAD and unit.weight_sets <= 1:
        notes.append(
            f"b_dataflow={b_dataflow.value} requested but this array holds no persistent "
            f"weight banks (weight_sets=1); every wave's B tile is loaded and used directly, "
            f"so the schedule and listing are identical to write-ahead's."
        )
        b_dataflow = BDataflow.WRITE_AHEAD

    iterations = deployment.iterations
    if b_dataflow is BDataflow.PERSISTENT and iterations > 1:
        b_write_multiplier = 1.0 / iterations
        notes.append(
            f"b_dataflow=persistent over iterations={iterations}: the first invocation writes "
            f"all {tiles} tiles, the other {iterations - 1} write none (already resident) — "
            f"amortised to a {1.0 / iterations:.1%} share of a full write per invocation, which "
            f"is what this report's DRAM traffic charges."
        )
    else:
        b_write_multiplier = 1.0

    return DataflowPlan(
        grid=grid,
        stationarity=machine.stationarity,
        requested_stationarity=deployment.stationarity,
        split_k=machine.k_partitions,
        a_strategy=a_strategy,
        a_requested=deployment.a_strategy,
        residency_tiles=residency_tiles,
        tiles_per_a_event=per_event,
        a_bytes_multiplier=a_bytes_multiplier,
        a_bytes=a_bytes,
        a_events=a_events,
        group_name=group_name,
        a_bytes_per_event=a_bytes_per_event,
        b_dataflow=b_dataflow,
        b_requested=deployment.b_dataflow,
        a_prefetch_depth=deployment.a_prefetch_depth,
        tiles=tiles,
        resident_tile_capacity=resident_tile_capacity,
        iterations=iterations,
        b_write_multiplier=b_write_multiplier,
        notes=tuple(notes),
    )
