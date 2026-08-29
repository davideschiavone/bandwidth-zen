"""Which operand stays resident, and the tile grid that follows from it.

``docs/CORRECTIONS.md`` D53. One matmul admits several decompositions, and the
choice is not cosmetic: it decides which dimensions run in parallel across the
chip's units, which one each tile sweeps internally, and — the part that costs
real time — whether partial sums have to be summed across units afterwards.

For a matmul the operand names map as **input = A, weight = B, output = C**:

===========  ==========  ======================  ========  =================
stationarity resident    parallel grid           swept     cross-core reduce
===========  ==========  ======================  ========  =================
``os``       C           ceil(M/rows) x ceil(N/cols)  K    no
``ws``       B           ceil(K/rows) x ceil(N/cols)  M    yes, over K
``is``       A           ceil(M/rows) x ceil(K/rows)  N    yes, over K
``rs``       A row       ceil(M/rows) x ceil(N/cols)  K    local to the array
===========  ==========  ======================  ========  =================

**Only the reduction differs in cost, never the arithmetic.** Every grid issues
the same ``M*N*K`` MACs; they differ in how that work is cut up, so any two
stationarities must agree on the flop count and disagree only on utilisation,
traffic and the reduction. :func:`grid_for` is the single place that decides.

``os`` is what cuBLAS/CUTLASS do by default — K accumulates in registers inside
one output tile, so nothing crosses cores. ``ws`` and ``is`` put K on the grid,
so each tile owns a slice of the contraction and the slices must be added.
*Where* they are added is not one answer but four, and :class:`ReductionPlacement`
is the choice: in the unit's own periphery, on chip, or through DRAM. Asking for
**split-K** always lands on the last, because CUTLASS runs it as two kernels
("partitionedK GEMM, and batched reduction") and the partials have nowhere to
live between them — :attr:`TileGrid.materialises_partials` is that case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from bwz.graph.ops import MatmulAttrs
from bwz.report import ReductionPlacement
from bwz.spec.hardware_spec import ComputeUnit, Dataflow


class Dim(StrEnum):
    """A matmul dimension, named so a grid can say which axis is which."""

    M = "M"
    N = "N"
    K = "K"


class Operand(StrEnum):
    """Which matrix is held resident."""

    A = "A"
    B = "B"
    C = "C"


@dataclass(frozen=True, slots=True)
class TileGrid:
    """The parallel tile grid one stationarity implies, and how to read it.

    ``rows``/``cols`` are the *grid* extents in tiles, not the array's own
    ``rows x cols`` geometry — a tile is always array-sized; ``tile_rows`` and
    ``tile_cols`` are that geometry, and this says how many tiles there are and
    along which dimensions.

    Self-describing on purpose: the trace, the listing and both renderers all
    have to name tiles the same way, so they read the decomposition off one
    object rather than each re-deriving it from ``(attrs, rows, cols)`` (D53).
    """

    stationarity: Dataflow
    resident: Operand
    row_dim: Dim
    col_dim: Dim
    swept_dim: Dim
    rows: int
    cols: int
    m: int
    n: int
    k: int
    tile_rows: int
    """Extent of one tile along :attr:`row_dim` — the array's row count."""
    tile_cols: int
    """Extent of one tile along :attr:`col_dim`: the array's column count, except
    where K is the column axis (``is``), K being cut by the array's depth."""
    k_partitions: int = 1
    """Split-K: how many pieces the contraction is cut into *in addition* to the
    grid. 1 for everything but ``os`` with ``--split-k`` (D53)."""

    @property
    def tiles(self) -> int:
        """Independent tiles the chip must get through, split-K included."""
        return self.rows * self.cols * self.k_partitions

    def extent(self, dim: Dim) -> int:
        """The matmul's own size along *dim*."""
        return {Dim.M: self.m, Dim.N: self.n, Dim.K: self.k}[dim]

    @property
    def needs_reduction(self) -> bool:
        """Whether partial sums cross unit boundaries and must be summed.

        True exactly when K is spread across independent tiles — on the grid
        itself (``ws``/``is``) or by split-K. ``rs`` spreads K *within* one
        array, where the wiring sums it locally, so it is False here: the cost
        model has no on-chip interconnect term to charge it against (D5a).
        """
        return self.k_partitions > 1 or Dim.K in (self.row_dim, self.col_dim)

    def decode(self, tile_index: int) -> tuple[int, int]:
        """Split a flat tile index into its ``(row, col)`` grid position.

        Row-major within the grid, which is what makes a coalesced range of
        tile indices decode into whole rows plus a ragged remainder.
        """
        within = tile_index % (self.rows * self.cols)
        return within // self.cols, within % self.cols

    def partition_of(self, tile_index: int) -> int:
        """Which split-K partition a flat tile index belongs to."""
        return tile_index // (self.rows * self.cols)

    @property
    def group_name(self) -> str:
        """What one grid row is called, in the vocabulary of its own row axis.

        A row of the ``ws`` grid is a slice of the contraction — D33's *k-slice*,
        the name the listing and the trace have always used. A row of the ``os``
        grid is a band of output rows instead, and calling that a k-slice would
        describe a decomposition the chip is not running.
        """
        return "k-slice" if self.row_dim is Dim.K else "row-band"

    @property
    def a_events(self) -> int:
        """A staging events the schedule has: one per grid row, per split-K piece.

        Generalises D33's "A is staged once per k-slice". What is invariant is
        not the k-slice — it is that the tiles of one grid *row* all read the
        same slice of A, because A's dimensions are M and K and the column axis
        is never one of those except under ``is``, where a row's tiles between
        them read the row's band exactly once anyway. So A crosses DRAM exactly
        once under every stationarity, in :attr:`a_events` pieces.
        """
        return self.rows * self.k_partitions

    def a_event_elements(self, event: int) -> float:
        """Elements of A one staging event moves.

        Per-row, not a uniform ``M*K / a_events`` average: neither M nor K need
        divide by the array, so the last band is narrower than the rest and
        charging it the average would under-charge every full band. Summed over
        ``range(a_events)`` this is exactly ``M*K`` — A crosses DRAM once.
        """
        row = event % self.rows
        if self.row_dim is Dim.K:
            # ws: the row is a k-slice, and the whole M height reads it.
            elements = float(self.m) * max(0, min(self.tile_rows, self.k - row * self.tile_rows))
        else:
            # os/rs/is: the row is a band of M, which reads all of K.
            elements = float(max(0, min(self.tile_rows, self.m - row * self.tile_rows))) * self.k
        return elements / self.k_partitions

    @property
    def a_event_shape(self) -> tuple[int, int]:
        """``(rows, cols)`` of A one staging event covers, for labelling."""
        if self.row_dim is Dim.K:
            return self.m, self.tile_rows
        return self.tile_rows, self.k // self.k_partitions

    @property
    def resident_tile_elements(self) -> int:
        """Elements of the resident operand one tile holds."""
        return self.tile_rows * self.tile_cols

    @property
    def materialises_partials(self) -> bool:
        """Whether the partials cross DRAM **because there are two kernels** (D53).

        Split-K alone. CUTLASS runs it as "partitionedK GEMM, and batched
        reduction", so the GEMM has ended everywhere before any summing starts
        and the partials have nowhere to live but global memory in between.

        This is a property of the *schedule*, not of capacity, which is why it
        stayed a boolean when :class:`ReductionPlacement` arrived: a K-on-grid
        walk can also end up in DRAM, but for the other reason — an accumulator
        too big to hold (D62). :func:`reduction_placement` is where the two
        meet, and it is the one that decides what anything is charged.
        """
        return self.k_partitions > 1

    @property
    def k_slices(self) -> int:
        """Partial results each output element ends up with — the ``p`` that
        every reduction cost is ``(p - 1) * M * N`` additions of.

        ``k_tiles`` when K is on the grid, since the grid's K axis *is* the cut;
        ``k_partitions`` under split-K, which cuts a K that is swept inside the
        tile. Never both: cutting K twice is refused rather than costed (D62).
        1 when nothing is cut, which is ``os``'s whole advantage.
        """
        if self.row_dim is Dim.K:
            return self.rows
        if self.col_dim is Dim.K:
            return self.cols
        return self.k_partitions

    @property
    def accumulator_elements(self) -> int:
        """Live partial results the grid's own walk keeps in flight.

        The whole ``M x N`` output under a K-on-grid walk: it finishes one slice
        of the contraction across every output cell before starting the next, so
        every cell has an open accumulator throughout. Load-bearing since D62:
        whether these fit on chip is what separates a reduction that costs
        vector time from one that costs a DRAM round trip.
        """
        return self.m * self.n


def grid_for(
    stationarity: Dataflow,
    attrs: MatmulAttrs,
    rows: int,
    cols: int,
    *,
    k_partitions: int = 1,
) -> TileGrid:
    """The tile grid *stationarity* implies for this matmul on this array.

    *rows*/*cols* are the array's own systolic dimensions. K is always cut by
    ``rows`` (it is the dimension that enters the array's depth) and N by
    ``cols``; M is cut by ``rows`` too, since an instruction tile covers that
    many result rows (D52).
    """
    m_tiles = max(1, math.ceil(attrs.m / rows))
    n_tiles = max(1, math.ceil(attrs.n / cols))
    k_tiles = max(1, math.ceil(attrs.k / rows))
    splits = max(1, k_partitions)

    def grid(
        resident: Operand,
        row_dim: Dim,
        col_dim: Dim,
        swept_dim: Dim,
        grid_rows: int,
        grid_cols: int,
        tile_cols: int = cols,
        *,
        k_pieces: int = 1,
    ) -> TileGrid:
        return TileGrid(
            stationarity=stationarity,
            resident=resident,
            row_dim=row_dim,
            col_dim=col_dim,
            swept_dim=swept_dim,
            rows=grid_rows,
            cols=grid_cols,
            m=attrs.m,
            n=attrs.n,
            k=attrs.k,
            tile_rows=rows,
            tile_cols=tile_cols,
            k_partitions=k_pieces,
        )

    if stationarity is Dataflow.WEIGHT_STATIONARY:
        return grid(Operand.B, Dim.K, Dim.N, Dim.M, k_tiles, n_tiles)
    if stationarity is Dataflow.INPUT_STATIONARY:
        # K on the column axis is cut by the array's depth, not its width.
        return grid(Operand.A, Dim.M, Dim.K, Dim.N, m_tiles, k_tiles, rows)
    if stationarity is Dataflow.ROW_STATIONARY:
        # Eyeriss (Chen/Emer/Sze, ISCA 2016) maps a row of the stationary
        # operand to each PE and spreads the contraction across the array's
        # own columns, so the grid is shaped like os but K never leaves one
        # array. Defined for completeness; nothing ships declaring it, so the
        # caller labels its numbers unvalidated (D53).
        return grid(Operand.A, Dim.M, Dim.N, Dim.K, m_tiles, n_tiles)
    # Output-stationary, and the default: C sits in the accumulator while K is
    # swept inside the tile. Split-K cuts that sweep into `splits` independent
    # pieces, which is the only way this grid ever needs a reduction.
    return grid(Operand.C, Dim.M, Dim.N, Dim.K, m_tiles, n_tiles, k_pieces=splits)


K_ON_GRID = frozenset({Dataflow.WEIGHT_STATIONARY, Dataflow.INPUT_STATIONARY})
"""Dataflows whose parallel grid carries K on one of its axes, so that several
units hold partial values of the same output element (D53)."""

UNBOUNDED_ACCUMULATION = math.inf
"""What :func:`accumulation_depth` returns for a unit that runs a K-on-grid
dataflow natively and states no depth: today's claim, unchanged and unfalsifiable
until the profile says how deep the accumulator really is (D62)."""


def accumulation_depth(unit: ComputeUnit) -> float:
    """Contraction inputs *unit* sums without the partial leaving it (D62).

    Three cases, and the middle one is why this is a function rather than a
    field read:

    * a **declared** ``local_accumulation_inputs`` wins outright. Metis's 16384
      is published (ISSCC 2024 11.3, Fig. 11.3.1) and the paper states the
      mechanism: an integer arithmetic unit sums a large MVM's partial products
      "without storing intermediate results back to memory".
    * a unit whose **native** dataflow already puts K on the grid and declares
      no depth is claiming such an accumulator by construction — a profile
      saying "I run ``ws``" while having nowhere to accumulate K would be
      describing a machine that cannot run its own declared dataflow. That is
      exactly the claim this model made for every K-on-grid grid before D62, so
      it is preserved here, unbounded, and named in ``report.assumptions`` as
      the unfalsifiable thing it is.
    * everything else has **none**. An MMA unit's accumulator is
      per-instruction, in one threadblock's registers; k-slices landing on
      different SMs have nothing to meet in, which is the whole reason ``ws`` on
      a tensor core is worth measuring.
    """
    if unit.local_accumulation_inputs > 0:
        return float(unit.local_accumulation_inputs)
    return UNBOUNDED_ACCUMULATION if unit.dataflow in K_ON_GRID else 0.0


@dataclass(frozen=True, slots=True)
class ReductionCost:
    """What summing a grid's partial results costs, and where they are summed.

    Zero on every numeric field when nothing is reduced, so callers can add it
    unconditionally; :attr:`placement` still says *why* it is zero, which is a
    different fact for ``os`` (nothing to sum) than for Metis (summed in the
    array's own periphery).
    """

    placement: ReductionPlacement
    """Where the partials meet — the choice that decides every field below, and
    whether the vector time is added to the matrix time or overlapped with it."""
    partitions: int
    """Partial results per output element — how many pieces the contraction was
    cut into. 1 when nothing is reduced."""
    partial_sums: float
    """Elementwise additions the reduction performs, charged to the *vector*
    unit — a matrix engine does matrix-multiply-accumulate and nothing else
    (D27).

    **Not new arithmetic.** ``2*M*N*K`` already counts them: accumulating K
    products into one output is ``K - 1`` additions however the contraction is
    cut, and ``(K/p - 1)*p + (p - 1) = K - 1`` for every ``p``. What a cut K
    changes is *where* they run — ``(p-1)*M*N`` of them leave the matrix engine's
    own accumulator for the vector unit, which on A100 is 16x slower (D27). So
    this is charged as vector time, and the sliver it double-counts is the same
    count at the matrix rate: 1/16 of what it adds."""
    dram_bytes: float
    """Partials written out and read back, under :attr:`ReductionPlacement.DRAM`
    only. Either the GEMM kernel ended before the reduction kernel started
    (split-K), or the accumulator was too big to hold on chip — both mean the
    partials cannot stay where they were produced."""
    dispatches: int
    """Extra kernel launches — one, for that second kernel."""

    @property
    def is_free(self) -> bool:
        """Whether this costs no time at all — nothing to sum, or summed in the
        unit's own periphery. An ``ON_CHIP`` reduction is *not* free: it costs
        vector time, even where the overlap hides it behind the matrix work."""
        return self.placement in (ReductionPlacement.NONE, ReductionPlacement.LOCAL)


NO_REDUCTION = ReductionCost(ReductionPlacement.NONE, 1, 0.0, 0.0, 0)


def reduction_placement(
    grid: TileGrid,
    accumulator_bytes: float,
    *,
    unit: ComputeUnit,
    on_chip_capacity_bytes: float,
) -> ReductionPlacement:
    """Where this grid's partial sums meet, on this unit (D62).

    Four cases, in the order they are tested:

    ``NONE``
        K is not cut at all — ``os`` without split-K, or ``rs``, which spreads K
        inside one array where the wiring sums it (D5a).
    ``DRAM``
        split-K, always: two kernels, so the partials cross global memory
        between them whatever the capacity (:attr:`TileGrid.materialises_partials`).
    ``LOCAL``
        K is on the grid and ``K`` fits :func:`accumulation_depth` — the partials
        never leave the unit that made them, so nothing is charged.
    ``ON_CHIP``
        K is on the grid, the partials must leave the unit, and the whole
        ``M x N`` accumulator fits in on-chip capacity. This is the tensor-core
        case D62 exists for: partials out to L2, summed by the vector unit,
        overlapped with the matrix work.
    ``DRAM`` again
        K is on the grid and the accumulator does **not** fit on chip. A cliff,
        not a slope, and the report has to say that it flipped.

    Capacity is compared at *accumulator* width — the width the report writes C
    at, which is what :func:`reduction_cost`'s caller passes — rather than at a
    deployment knob, so the two cannot disagree about the same bytes.
    """
    if not grid.needs_reduction:
        return ReductionPlacement.NONE
    if grid.materialises_partials:
        return ReductionPlacement.DRAM
    if grid.k <= accumulation_depth(unit):
        return ReductionPlacement.LOCAL
    live_bytes = grid.accumulator_elements * accumulator_bytes
    if live_bytes <= on_chip_capacity_bytes:
        return ReductionPlacement.ON_CHIP
    return ReductionPlacement.DRAM


def reduction_cost(
    grid: TileGrid,
    accumulator_bytes: float,
    *,
    unit: ComputeUnit,
    on_chip_capacity_bytes: float,
) -> ReductionCost:
    """Cost of summing the ``p`` partial results *grid* leaves per output element.

    ``p`` slices of the contraction each produce a full ``M x N`` partial, so
    summing them is ``(p - 1) * M * N`` additions on the vector unit wherever
    they meet, plus — under :attr:`ReductionPlacement.DRAM` only — a round trip
    of ``p * M * N`` accumulator-width values and the dispatch that reads them.

    What :func:`reduction_placement` decides is not *whether* those additions
    happen but *where*, and the two placements that are not free differ in more
    than their byte count: ``ON_CHIP`` overlaps the adds with the matrix work
    (the caller takes ``max``), ``DRAM`` serialises them behind it. Returns
    :data:`NO_REDUCTION` when nothing is cut, which is why plain
    output-stationary is the cheap default (D53).
    """
    placement = reduction_placement(
        grid, accumulator_bytes, unit=unit, on_chip_capacity_bytes=on_chip_capacity_bytes
    )
    partitions = grid.k_slices
    if placement is ReductionPlacement.NONE or partitions <= 1:
        return NO_REDUCTION
    elements = float(grid.m) * float(grid.n)
    if placement is ReductionPlacement.LOCAL:
        return ReductionCost(placement, partitions, 0.0, 0.0, 0)
    if placement is ReductionPlacement.ON_CHIP:
        # No bytes: the v1 machine has no on-chip bandwidth term (D5a/D5b), so
        # the honest position is that this reduction costs vector time and
        # capacity, and that the L2 traffic it implies is unmodelled. Said in
        # `report.assumptions` rather than silently priced at zero.
        return ReductionCost(placement, partitions, (partitions - 1) * elements, 0.0, 0)
    return ReductionCost(
        placement=placement,
        partitions=partitions,
        partial_sums=(partitions - 1) * elements,
        dram_bytes=partitions * elements * accumulator_bytes * 2.0,
        dispatches=1,
    )


def residency_phrase(grid: TileGrid, unit: ComputeUnit) -> str:
    """What is really held, in the vocabulary of the unit that holds it (D62).

    "Weight-stationary" names a machine whose weights sit in the array while
    activations stream past. Ask a tensor core for that grid and nothing is
    held: it reads every operand from the register file per instruction (D30),
    and what actually changed is that K moved onto the tile grid. Saying "B
    resident" there would describe hardware the caller is not running, so the
    grid — the real content of the choice — is named instead.
    """
    if grid.resident is Operand.B and unit.weight_sets <= 1:
        return "nothing held, K on the grid"
    return f"{grid.resident.value} resident"


def refusal_reason(unit: ComputeUnit, requested: Dataflow) -> str | None:
    """Why *unit* cannot run *requested*, or ``None`` if it can.

    Refused rather than clamped, unlike the A/B strategy knobs: those pick
    between orderings of the same work, so falling back still answers the
    question asked. A stationarity is a different decomposition — quietly
    substituting one would report a number for hardware the caller did not ask
    about (D53). The message names the field, the request, and the declared
    capability, so it is actionable (CLAUDE.md #8).
    """
    supported = unit.dataflows()
    if requested in supported:
        return None
    names = ", ".join(sorted(d.value for d in supported))
    why = ""
    if requested is Dataflow.WEIGHT_STATIONARY and unit.weight_sets <= 1:
        why = (
            f" {unit.name} declares weight_sets=1, i.e. no weight residency at all: both "
            f"operands are re-read per instruction (D30), so there is nothing for a weight "
            f"to stay stationary in."
        )
    elif requested is Dataflow.OUTPUT_STATIONARY and unit.weight_sets > 1:
        why = (
            f" {unit.name} holds {unit.weight_sets} weight sets — for an in-memory array the "
            f"weights ARE the storage, so it cannot instead hold accumulators."
        )
    return (
        f"stationarity={requested.value!r} is not supported by {unit.name}, which declares "
        f"{names}.{why} Drop the flag to use the chip's own dataflow, or pick one it declares."
    )
