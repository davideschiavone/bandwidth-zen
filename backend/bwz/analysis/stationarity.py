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
which is a *split-K* shape: each unit owns a slice of the contraction, and the
slices must be added. CUTLASS runs that as a second kernel ("partitionedK GEMM,
and batched reduction"), which is what :func:`reduction_cost` charges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from bwz.graph.ops import MatmulAttrs
from bwz.spec.hardware_spec import Dataflow


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
    ``rows x cols`` geometry — a tile is always array-sized; this says how many
    of them there are and along which dimensions.
    """

    stationarity: Dataflow
    resident: Operand
    row_dim: Dim
    col_dim: Dim
    swept_dim: Dim
    rows: int
    cols: int
    k_partitions: int = 1
    """Split-K: how many pieces the contraction is cut into *in addition* to the
    grid. 1 for everything but ``os`` with ``--split-k`` (D53)."""

    @property
    def tiles(self) -> int:
        """Independent tiles the chip must get through, split-K included."""
        return self.rows * self.cols * self.k_partitions

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

    if stationarity is Dataflow.WEIGHT_STATIONARY:
        return TileGrid(stationarity, Operand.B, Dim.K, Dim.N, Dim.M, k_tiles, n_tiles)
    if stationarity is Dataflow.INPUT_STATIONARY:
        return TileGrid(stationarity, Operand.A, Dim.M, Dim.K, Dim.N, m_tiles, k_tiles)
    if stationarity is Dataflow.ROW_STATIONARY:
        # Eyeriss (Chen/Emer/Sze, ISCA 2016) maps a row of the stationary
        # operand to each PE and spreads the contraction across the array's
        # own columns, so the grid is shaped like os but K never leaves one
        # array. Defined for completeness; nothing ships declaring it, so the
        # caller labels its numbers unvalidated (D53).
        return TileGrid(stationarity, Operand.A, Dim.M, Dim.N, Dim.K, m_tiles, n_tiles)
    # Output-stationary, and the default: C sits in the accumulator while K is
    # swept inside the tile. Split-K cuts that sweep into `splits` independent
    # pieces, which is the only way this grid ever needs a reduction.
    return TileGrid(stationarity, Operand.C, Dim.M, Dim.N, Dim.K, m_tiles, n_tiles, splits)


@dataclass(frozen=True, slots=True)
class ReductionCost:
    """What summing split partial results costs, when the grid needs it.

    Zero on every field when it does not, so callers can add it unconditionally.
    """

    partial_sums: float
    """Elementwise additions, charged to the *vector* unit — a matrix engine does
    matrix-multiply-accumulate and nothing else (D27)."""
    dram_bytes: float
    """Partials written out and read back: CUTLASS runs the reduction as a second
    kernel, so they cannot stay in registers between the two."""
    dispatches: int
    """Extra kernel launches — one, for that second kernel."""

    @property
    def is_free(self) -> bool:
        return self.dispatches == 0


NO_REDUCTION = ReductionCost(0.0, 0.0, 0)


def reduction_cost(grid: TileGrid, attrs: MatmulAttrs, accumulator_bytes: float) -> ReductionCost:
    """Cost of summing the partial results *grid* leaves behind.

    ``partitions`` slices of the contraction each produce a full ``M x N``
    partial, so summing them is ``(partitions - 1) * M * N`` additions and a
    round trip of ``partitions * M * N`` accumulator-width values — written by
    the GEMM kernel, read by the reduction kernel.

    Returns :data:`NO_REDUCTION` for a grid that keeps K inside one tile, which
    is why output-stationary is the cheap default (D53).
    """
    if not grid.needs_reduction:
        return NO_REDUCTION
    partitions = grid.k_partitions
    if Dim.K is grid.row_dim:
        partitions *= grid.rows
    elif Dim.K is grid.col_dim:
        partitions *= grid.cols
    if partitions <= 1:
        return NO_REDUCTION
    elements = float(attrs.m) * float(attrs.n)
    return ReductionCost(
        partial_sums=(partitions - 1) * elements,
        dram_bytes=partitions * elements * accumulator_bytes * 2.0,
        dispatches=1,
    )
