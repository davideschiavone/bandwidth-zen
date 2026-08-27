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
so each tile owns a slice of the contraction and the slices must be added; the
same units revisit the same output cell on a later wave, so those partials meet
in an accumulator rather than in DRAM. Asking for **split-K** is the one case
that materialises them, because CUTLASS runs it as two kernels ("partitionedK
GEMM, and batched reduction") — that is what :func:`reduction_cost` charges, and
:attr:`TileGrid.materialises_partials` is where the line is drawn and why.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from bwz.graph.ops import MatmulAttrs
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
        """Whether the partial results are written out and read back (D53).

        Narrower than :attr:`needs_reduction`, and the distinction is the whole
        cost. Split-K is **two kernels** — CUTLASS's "partitionedK GEMM, and
        batched reduction" — so its partials have nowhere to live but global
        memory between them. A grid that merely carries K on its own axis does
        not: the same units revisit the same output cell on a later wave, so the
        partials meet in an accumulator that never leaves the chip, and this
        model has no on-chip bandwidth term to charge that against (D5a).

        What the accumulator costs is *capacity*, and where it does not fit, a
        real compiler re-blocks the output and re-reads A and B rather than
        spilling C. Those re-reads are exactly the traffic ``docs/MODEL.md`` 6.2
        already declines to model — "DRAM traffic is compulsory traffic … so a
        DRAM-bound latency here is a lower bound". Charging a C spill here while
        that stands would price one horn of the dilemma and not the other, on a
        chip nobody has measured; the honest move is to say so in
        ``report.assumptions`` and leave the bound where it is.
        """
        return self.k_partitions > 1

    @property
    def accumulator_elements(self) -> int:
        """Live partial results the grid's own walk keeps in flight.

        The whole ``M x N`` output under a K-on-grid walk: it finishes one slice
        of the contraction across every output cell before starting the next, so
        every cell has an open accumulator throughout. Reported so a reader can
        see when the assumption that they stay on chip stops being plausible.
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


@dataclass(frozen=True, slots=True)
class ReductionCost:
    """What CUTLASS's second kernel costs, when the grid needs one.

    Zero on every field when it does not, so callers can add it unconditionally.
    """

    partitions: int
    """Partial results per output element — how many pieces the contraction was
    cut into. 1 when nothing is reduced."""
    partial_sums: float
    """Elementwise additions the reduction kernel performs, charged to the
    *vector* unit — a matrix engine does matrix-multiply-accumulate and nothing
    else (D27).

    **Not new arithmetic.** ``2*M*N*K`` already counts them: accumulating K
    products into one output is ``K - 1`` additions however the contraction is
    cut, and ``(K/p - 1)*p + (p - 1) = K - 1`` for every ``p``. What split-K
    changes is *where* they run — ``(p-1)*M*N`` of them leave the matrix engine's
    own accumulator for the vector unit, which on A100 is 16x slower (D27). So
    this is charged as vector time on top, and the sliver it double-counts is
    the same count at the matrix rate: 1/16 of what it adds."""
    dram_bytes: float
    """Partials written out and read back. The GEMM kernel ends before the
    reduction kernel starts, so they cannot stay in registers between the two."""
    dispatches: int
    """Extra kernel launches — one, for that second kernel."""

    @property
    def is_free(self) -> bool:
        return self.dispatches == 0


NO_REDUCTION = ReductionCost(1, 0.0, 0.0, 0)


def reduction_cost(grid: TileGrid, accumulator_bytes: float) -> ReductionCost:
    """Cost of summing the partial results *grid* materialises.

    ``partitions`` slices of the contraction each produce a full ``M x N``
    partial, so summing them is ``(partitions - 1) * M * N`` additions on the
    vector unit and a round trip of ``partitions * M * N`` accumulator-width
    values — written by the GEMM kernel, read by the reduction kernel.

    Charged for a grid that :attr:`~TileGrid.materialises_partials` — split-K,
    the two-kernel case — and not for one that merely carries K on an axis,
    where the partials meet in an on-chip accumulator that this model has no
    bandwidth term for (see that property for why pricing it would be worse than
    naming it). Returns :data:`NO_REDUCTION` otherwise, which is why plain
    output-stationary is the cheap default (D53).
    """
    if not grid.materialises_partials:
        return NO_REDUCTION
    partitions = grid.k_partitions
    if partitions <= 1:
        return NO_REDUCTION
    elements = float(grid.m) * float(grid.n)
    return ReductionCost(
        partitions=partitions,
        partial_sums=(partitions - 1) * elements,
        dram_bytes=partitions * elements * accumulator_bytes * 2.0,
        dispatches=1,
    )


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
