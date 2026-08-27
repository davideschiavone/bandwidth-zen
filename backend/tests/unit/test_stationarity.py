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
    refusal_reason,
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
    cost = reduction_cost(split, accumulator_bytes=4.0)

    elements = 1000 * 2000
    assert cost.partial_sums == pytest.approx(3 * elements)
    assert cost.dram_bytes == pytest.approx(4 * elements * 4.0 * 2.0)
    assert cost.dispatches == 1
    assert not cost.is_free


def test_a_grid_that_needs_no_reduction_costs_nothing_to_reduce() -> None:
    plain = grid_for(Dataflow.OUTPUT_STATIONARY, ATTRS, ROWS, COLS)
    assert reduction_cost(plain, accumulator_bytes=4.0) is NO_REDUCTION
    assert NO_REDUCTION.is_free


def test_k_on_the_grid_owes_a_reduction_but_does_not_materialise_it() -> None:
    """The line D53 draws, and the reason the two properties are separate.

    ``ws`` puts all 188 k-slices on the grid, so partial sums genuinely exist
    and genuinely have to be added — ``needs_reduction`` says so. What it does
    NOT do is write them to DRAM: the same unit comes back to the same output
    cell on a later wave, so they meet in an accumulator, and D5a gives this
    model no on-chip bandwidth term to charge that against. Split-K is the case
    that materialises them, because CUTLASS runs it as two kernels — so only it
    is costed here.
    """
    ws_grid = grid_for(Dataflow.WEIGHT_STATIONARY, ATTRS, ROWS, COLS)

    assert ws_grid.needs_reduction, "188 k-slices of partials do have to be summed"
    assert not ws_grid.materialises_partials, "but in an accumulator, not through DRAM"
    assert reduction_cost(ws_grid, accumulator_bytes=4.0) is NO_REDUCTION
    # What it costs instead is capacity, and the drawer has to be able to say
    # how much: the whole output is live under a K-on-grid walk.
    assert ws_grid.accumulator_elements == 1000 * 2000


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


def test_asking_a_chip_for_a_dataflow_it_cannot_run_is_refused_not_clamped() -> None:
    """D53, and a deliberate departure from the A/B strategy knobs.

    Those clamp: stage/stream/whole are orderings of the same work, so falling
    back still answers the question. A stationarity is a different
    decomposition, so substituting one would report a number for hardware the
    caller never asked about. The message has to name the field, the request
    and the real capability (CLAUDE.md #8).
    """
    from bwz.analysis import analyze
    from bwz.spec import DeploymentSpec, MatmulSpec, load_chip

    # Input-stationary: no shipped profile declares it, so this stays a refusal
    # whichever native dataflow the chips are on.
    tensor_core = load_chip("a100_80gb").compute_units[0]
    why = refusal_reason(tensor_core, Dataflow.INPUT_STATIONARY)
    assert why is not None
    assert "stationarity='is'" in why, "names the field and the request"
    assert "tensor_core" in why, "names the unit that cannot run it"
    assert tensor_core.dataflow.value in why, "names the real capability"

    spec = MatmulSpec.model_validate(
        {"id": "t", "name": "t", "family": "matmul", "m": 512, "n": 512, "k": 4096}
    )
    deployment = DeploymentSpec.model_validate(
        {"batch": 1, "input_tokens": 1, "output_tokens": 1, "stationarity": "is"}
    )
    report = analyze(spec, load_chip("a100_80gb"), deployment)

    assert report.feasible is False, "refused, never raised and never silently substituted"
    assert any("stationarity" in reason for reason in report.infeasibility)


def test_a_chip_running_its_own_declared_dataflow_is_always_accepted() -> None:
    """Whatever a profile declares, it must be able to run — the schema enforces
    that its native dataflow is in its supported set."""
    from bwz.spec import load_chip

    for chip_id in ("a100_80gb", "h100_sxm", "metis_aipu", "chip_a"):
        for unit in load_chip(chip_id).compute_units:
            assert refusal_reason(unit, unit.dataflow) is None, chip_id


def test_the_mma_chips_default_to_the_cublas_dataflow() -> None:
    """The flip D53 exists for: the four matrix-core profiles declare ``os``.

    ``test_asking_a_chip_for_a_dataflow_it_cannot_run_is_refused_not_clamped``
    is deliberately flip-agnostic — it asks for ``is``, which nothing declares.
    This one is not: it pins the default, so a profile silently reverting to
    ``ws`` would fail here rather than quietly re-introducing a decomposition
    with an uncharged reduction in it.

    Metis and the two hypothetical NPUs stay weight-stationary. Their weights
    ARE their memory (D30), so an accumulator-resident dataflow is not a thing
    they could run.
    """
    from bwz.spec import load_chip

    for chip_id in ("a100_80gb", "h100_sxm", "jetson_orin", "mi300x"):
        matrix_unit = load_chip(chip_id).compute_units[0]
        assert matrix_unit.systolic_dims is not None, chip_id
        assert matrix_unit.dataflow is Dataflow.OUTPUT_STATIONARY, chip_id
        assert refusal_reason(matrix_unit, Dataflow.WEIGHT_STATIONARY) is not None, (
            f"{chip_id} must not silently accept the decomposition it moved away from"
        )

    for chip_id in ("metis_aipu", "chip_a", "chip_b"):
        arrays = [u for u in load_chip(chip_id).compute_units if u.systolic_dims is not None]
        assert arrays, chip_id
        assert all(u.dataflow is Dataflow.WEIGHT_STATIONARY for u in arrays), chip_id


def test_split_k_buys_occupancy_and_pays_for_it_in_dram() -> None:
    """The whole trade, end to end on the chip that can now run it.

    512x512x4096 fp16 on A100: the output grid alone is 32x32 = 1024 tiles
    against 432 tensor cores — 3 waves whose last is a third full, so wave
    occupancy is 1024/(3*432) = 0.79. Cutting K in eight gives 8192 tiles and
    19 waves at 0.998, but the eight partials of the whole 512x512 output have
    to be written and read back: 8 x 512 x 512 x 2 B x 2 = 8.4 MB against the
    4.7 MB the un-split GEMM moves in total. That is CUTLASS's second kernel,
    and it is why split-K is the exception rather than the default (D53).
    """
    from bwz.analysis import analyze
    from bwz.spec import DeploymentSpec, MatmulSpec, load_chip

    spec = MatmulSpec.model_validate(
        {"id": "t", "name": "t", "family": "matmul", "m": 512, "n": 512, "k": 4096}
    )

    def run(splits: int) -> tuple[float, float, float]:
        deployment = DeploymentSpec.model_validate(
            {
                "batch": 1,
                "input_tokens": 1,
                "output_tokens": 0,
                "phase": "prefill",
                "split_k": splits,
            }
        )
        report = analyze(spec, load_chip("a100_80gb"), deployment)
        assert report.feasible, report.infeasibility
        op = report.phases[0].ops[0]
        return op.utilization, op.dram_reduction_bytes, op.dram_bytes

    plain_util, plain_reduction, plain_dram = run(1)
    split_util, split_reduction, split_dram = run(8)

    assert plain_reduction == 0.0, "one kernel, nothing materialised"
    assert plain_util == pytest.approx(1024 / (3 * 432), rel=1e-6)
    assert split_util == pytest.approx(8192 / (19 * 432), rel=1e-6)
    assert split_util > plain_util, "more tiles fill the chip better"
    assert split_reduction == pytest.approx(8 * 512 * 512 * 2.0 * 2.0)
    assert split_dram - plain_dram == pytest.approx(split_reduction), (
        "the extra traffic is exactly the partials' round trip, nothing else"
    )


@pytest.mark.parametrize(
    "flow", [Dataflow.ROW_STATIONARY, Dataflow.INPUT_STATIONARY], ids=lambda d: d.value
)
def test_the_unshipped_dataflows_still_run_end_to_end(flow: Dataflow) -> None:
    """``rs`` and ``is`` are implemented and nothing ships declaring them.

    That combination is how a code path rots: it is reachable — a user can write
    a profile — and no golden covers it. So one is written here, against a
    profile forged from A100's, checking the properties that must hold for any
    grid rather than numbers only these two produce: the report is feasible, A
    still crosses DRAM exactly once, the DRAM lane still sums to ``t_dram``, and
    the listing's constants still match the schedule (``deploy.check``).

    ``rs`` additionally has to carry its own honesty label — no such machine has
    been measured here, so its numbers are stated, not claimed (D53).
    """
    from bwz.analysis import analyze, build_trace, machine_model
    from bwz.analysis.pipeline import Stage
    from bwz.deploy import check as check_deployment
    from bwz.deploy import deployment_of
    from bwz.graph import GraphPhase, build_graph
    from bwz.spec import DeploymentSpec, MatmulSpec, load_chip

    base = load_chip("a100_80gb")
    units = [
        unit.model_copy(update={"dataflow": flow, "supported_dataflows": (flow,)})
        if unit.systolic_dims is not None
        else unit
        for unit in base.compute_units
    ]
    chip = base.model_copy(update={"compute_units": units, "hypothetical": True})
    deployment = DeploymentSpec.model_validate(
        {"batch": 1, "input_tokens": 1, "output_tokens": 0, "phase": "prefill"}
    )
    spec = MatmulSpec.model_validate(
        {"id": "t", "name": "t", "family": "matmul", "m": 1000, "n": 2000, "k": 3000}
    )

    report = analyze(spec, chip, deployment)
    assert report.feasible, report.infeasibility
    op = report.phases[0].ops[0]

    graph = build_graph(spec, deployment, GraphPhase.STATIC)
    machine = machine_model(chip, spec.operand_dtype)
    trace = build_trace(
        graph, report.phases[0], machine, double_buffered=report.memory.double_buffered
    )
    listing = deployment_of(
        chip, machine, report.phases[0], trace, workload="t", operation=graph.ops[0]
    )
    check_deployment(listing, trace)

    assert machine.stationarity is flow
    assert sum(s.bytes_moved for s in trace.spans if s.stage is Stage.LOAD_A) == pytest.approx(
        op.dram_activation_read_bytes, rel=1e-9
    )
    assert sum(s.duration_s for s in trace.spans if s.lane.value == "dram") == pytest.approx(
        op.t_dram_s, rel=1e-9
    )
    assert op.dram_reduction_bytes == 0.0, "neither materialises partials without split-K"
    assert (any("UNVALIDATED" in a for a in report.assumptions)) is (
        flow is Dataflow.ROW_STATIONARY
    )
