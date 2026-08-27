"""The tile grid each stationarity implies. ``docs/CORRECTIONS.md`` D53.

The property that holds across every case below: the decomposition changes, the
*arithmetic* does not. Two stationarities may disagree about how many tiles
there are, which dimension each sweeps, and whether a reduction is owed — never
about how many MACs the matmul is.
"""

from __future__ import annotations

import pytest

from bwz.analysis.stationarity import (
    NO_REDUCTION,
    Dim,
    Operand,
    grid_for,
    reduction_cost,
)
from bwz.graph.ops import MatmulAttrs
from bwz.spec.hardware_spec import Dataflow

# 1000x2000x3000 on a 16x16 array: M -> 63 tiles, N -> 125, K -> 188.
ATTRS = MatmulAttrs(m=1000, n=2000, k=3000)
ROWS = COLS = 16


def test_each_stationarity_grids_the_dimensions_it_should() -> None:
    """Hand-counted: ceil(1000/16)=63, ceil(2000/16)=125, ceil(3000/16)=188."""
    os_grid = grid_for(Dataflow.OUTPUT_STATIONARY, ATTRS, ROWS, COLS)
    assert (os_grid.resident, os_grid.rows, os_grid.cols) == (Operand.C, 63, 125)
    assert (os_grid.row_dim, os_grid.col_dim, os_grid.swept_dim) == (Dim.M, Dim.N, Dim.K)

    ws_grid = grid_for(Dataflow.WEIGHT_STATIONARY, ATTRS, ROWS, COLS)
    assert (ws_grid.resident, ws_grid.rows, ws_grid.cols) == (Operand.B, 188, 125)
    assert (ws_grid.row_dim, ws_grid.col_dim, ws_grid.swept_dim) == (Dim.K, Dim.N, Dim.M)

    is_grid = grid_for(Dataflow.INPUT_STATIONARY, ATTRS, ROWS, COLS)
    assert (is_grid.resident, is_grid.rows, is_grid.cols) == (Operand.A, 63, 188)
    assert (is_grid.row_dim, is_grid.col_dim, is_grid.swept_dim) == (Dim.M, Dim.K, Dim.N)

    rs_grid = grid_for(Dataflow.ROW_STATIONARY, ATTRS, ROWS, COLS)
    assert (rs_grid.resident, rs_grid.rows, rs_grid.cols) == (Operand.A, 63, 125)
    assert rs_grid.swept_dim is Dim.K, "K is spread across the array, not across the grid"


def test_only_a_grid_that_splits_k_owes_a_reduction() -> None:
    """The whole reason output-stationary is cuBLAS's default (D53).

    ``os`` keeps K inside one tile's accumulator, so nothing crosses cores.
    ``ws``/``is`` put K on the grid — each unit owns a slice of the contraction
    and the slices must be summed. ``rs`` spreads K inside one array, where the
    wiring sums it locally and this model has no on-chip term to charge (D5a).
    """
    assert not grid_for(Dataflow.OUTPUT_STATIONARY, ATTRS, ROWS, COLS).needs_reduction
    assert grid_for(Dataflow.WEIGHT_STATIONARY, ATTRS, ROWS, COLS).needs_reduction
    assert grid_for(Dataflow.INPUT_STATIONARY, ATTRS, ROWS, COLS).needs_reduction
    assert not grid_for(Dataflow.ROW_STATIONARY, ATTRS, ROWS, COLS).needs_reduction


def test_split_k_adds_tiles_and_turns_the_reduction_on() -> None:
    """Split-K is the one way output-stationary ever owes a reduction.

    It buys parallelism when the output grid alone cannot fill the chip, and
    pays CUTLASS's second kernel for it.
    """
    plain = grid_for(Dataflow.OUTPUT_STATIONARY, ATTRS, ROWS, COLS)
    split = grid_for(Dataflow.OUTPUT_STATIONARY, ATTRS, ROWS, COLS, k_partitions=4)

    assert plain.tiles == 63 * 125
    assert split.tiles == 63 * 125 * 4, "four independent slices of the contraction"
    assert not plain.needs_reduction
    assert split.needs_reduction


def test_reduction_is_partitions_minus_one_adds_over_the_whole_output() -> None:
    """Four partitions leave four M x N partials; summing them is 3 x M x N adds.

    The adds are elementwise, so they belong to the vector unit, not the matrix
    engine (D27). The bytes are a round trip: written by the GEMM kernel, read
    back by the reduction kernel, because the two are separate launches.
    """
    split = grid_for(Dataflow.OUTPUT_STATIONARY, ATTRS, ROWS, COLS, k_partitions=4)
    cost = reduction_cost(split, ATTRS, accumulator_bytes=4.0)

    elements = 1000 * 2000
    assert cost.partial_sums == pytest.approx(3 * elements)
    assert cost.dram_bytes == pytest.approx(4 * elements * 4.0 * 2.0)
    assert cost.dispatches == 1
    assert not cost.is_free


def test_a_grid_that_needs_no_reduction_costs_nothing_to_reduce() -> None:
    plain = grid_for(Dataflow.OUTPUT_STATIONARY, ATTRS, ROWS, COLS)
    assert reduction_cost(plain, ATTRS, accumulator_bytes=4.0) is NO_REDUCTION
    assert NO_REDUCTION.is_free


def test_weight_stationary_reduces_once_per_k_slice_on_the_grid() -> None:
    """``ws`` puts all 188 k-slices on the grid, so all 188 partials must be summed
    — the cost the engine has been silently omitting for every chip (D53)."""
    ws_grid = grid_for(Dataflow.WEIGHT_STATIONARY, ATTRS, ROWS, COLS)
    cost = reduction_cost(ws_grid, ATTRS, accumulator_bytes=4.0)

    elements = 1000 * 2000
    assert cost.partial_sums == pytest.approx(187 * elements), "188 partials, 187 adds"
    assert cost.dispatches == 1


@pytest.mark.parametrize("stationarity", list(Dataflow))
def test_the_decomposition_never_changes_the_arithmetic(stationarity: Dataflow) -> None:
    """Every grid must cover M*N*K MACs exactly once — the invariant that makes
    the choice a scheduling decision rather than a different computation."""
    grid = grid_for(stationarity, ATTRS, ROWS, COLS)
    swept = {Dim.M: ATTRS.m, Dim.N: ATTRS.n, Dim.K: ATTRS.k}[grid.swept_dim]
    # Each tile is array-sized on both grid axes and sweeps the third dimension
    # in full, so tiles x tile-area x swept covers the padded iteration space.
    covered = grid.rows * ROWS * grid.cols * COLS * swept
    exact = ATTRS.m * ATTRS.n * ATTRS.k
    assert covered >= exact, "padding may add work, never remove it"
    assert covered == pytest.approx(exact, rel=0.05), "padding only, not a different computation"


@pytest.mark.parametrize("stationarity", list(Dataflow))
def test_decode_round_trips_every_tile_index(stationarity: Dataflow) -> None:
    """``decode`` is what the renderers and the trace use to name a tile, so a
    flat index must map back to exactly one grid cell, row-major."""
    grid = grid_for(stationarity, ATTRS, ROWS, COLS)
    seen = {grid.decode(t) for t in range(grid.rows * grid.cols)}
    assert len(seen) == grid.rows * grid.cols, "every cell hit exactly once"
    assert grid.decode(0) == (0, 0)
    assert grid.decode(grid.cols - 1) == (0, grid.cols - 1)
    assert grid.decode(grid.cols) == (1, 0), "row-major: the next index starts a new row"


def test_split_k_partitions_are_distinguishable_from_grid_position() -> None:
    """A split-K tile index carries both which output cell and which slice of the
    contraction it computes; the renderers need to tell them apart."""
    grid = grid_for(Dataflow.OUTPUT_STATIONARY, ATTRS, ROWS, COLS, k_partitions=4)
    cells = grid.rows * grid.cols

    assert grid.partition_of(0) == 0
    assert grid.partition_of(cells) == 1, "same output cell, next slice of K"
    assert grid.decode(cells) == grid.decode(0)
