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
    UNBOUNDED_ACCUMULATION,
    Dim,
    Operand,
    ReductionCost,
    TileGrid,
    accumulation_depth,
    deal,
    deal_for,
    grid_for,
    k_groups,
    reduction_cost,
    refusal_reason,
    sums_locally,
)
from bwz.graph.ops import MatmulAttrs
from bwz.report import ReductionPlacement
from bwz.spec import load_chip
from bwz.spec.hardware_spec import ComputeUnit, Dataflow

# 1000x2000x3000 on a 16x16 array: M -> 63 tiles, N -> 125, K -> 188.
ATTRS = MatmulAttrs(m=1000, n=2000, k=3000)
ROWS = COLS = 16

TENSOR_CORE = load_chip("a100_80gb").compute_units[0]
"""os native, no declared accumulator: the unit D62 exists to measure."""
D_IMC = load_chip("metis_aipu").compute_units[0]
"""ws native, and 16384 inputs of local accumulation the paper states."""
A100_ON_CHIP_BYTES = 2.0736e7 + 4.0e7
"""L1 across 108 SMs plus L2 — 60.7 MB, the capacity the placement is tested against."""


def _cost(
    grid: TileGrid,
    unit: ComputeUnit,
    *,
    capacity: float = A100_ON_CHIP_BYTES,
    vector_adder: bool = True,
) -> ReductionCost:
    """``reduction_cost`` at a 4-byte accumulator, the fp32 case.

    *vector_adder* defaults to True: A100 has its CUDA cores and Metis its DPU.
    """
    return reduction_cost(
        grid,
        accumulator_bytes=4.0,
        unit=unit,
        on_chip_capacity_bytes=capacity,
        vector_adder=vector_adder,
    )


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
    back by the reduction kernel, because the two are separate launches — so
    split-K is ``DRAM`` whatever the capacity, and here it is handed enough
    on-chip capacity (1 GB) to make that unambiguous.
    """
    split = grid_for(Dataflow.OUTPUT_STATIONARY, ATTRS, ROWS, COLS, k_partitions=4)
    cost = reduction_cost(
        split,
        accumulator_bytes=4.0,
        unit=TENSOR_CORE,
        on_chip_capacity_bytes=1e9,
        vector_adder=True,
    )

    elements = 1000 * 2000
    assert cost.placement is ReductionPlacement.DRAM
    assert cost.partial_sums == pytest.approx(3 * elements)
    assert cost.dram_bytes == pytest.approx(4 * elements * 4.0 * 2.0)
    assert cost.dispatches == 1
    assert not cost.is_free


def test_a_grid_that_needs_no_reduction_costs_nothing_to_reduce() -> None:
    plain = grid_for(Dataflow.OUTPUT_STATIONARY, ATTRS, ROWS, COLS)
    assert _cost(plain, TENSOR_CORE) is NO_REDUCTION
    assert NO_REDUCTION.is_free
    assert NO_REDUCTION.placement is ReductionPlacement.NONE


def test_k_on_the_grid_is_reduced_where_the_hardware_can_reduce_it() -> None:
    """The placement table, on one grid, hand-computed (D62).

    ``ws`` on a 16x16 array puts all ``ceil(3000/16) = 188`` k-slices on the
    grid, so every one of the ``1000 x 2000`` output elements ends up with 188
    partial values and summing them is ``187 x 1000 x 2000 = 374,000,000``
    additions. Where they happen is the hardware's answer, not the dataflow's:

    * ``tensor_core`` declares no local accumulator, and 2 M accumulators at 4 B
      is 8 MB against A100's 60.7 MB on chip — so the partials go out to the
      cache and the CUDA cores add them: ``ON_CHIP``, no bytes, no dispatch.
    * give the same unit 1 MB of on-chip capacity and the accumulator no longer
      fits: ``DRAM``, ``188 x 2e6 x 4 B x 2 = 3.008 GB`` of round trip and a
      second dispatch. A cliff, not a slope.
    * ``d_imc`` declares 16384, and K=3000 is inside it, so the partials never
      leave the AI core's own periphery: ``LOCAL``, free — which is what the
      model claimed for *every* K-on-grid grid before D62.
    """
    ws_grid = grid_for(Dataflow.WEIGHT_STATIONARY, ATTRS, ROWS, COLS)
    elements = 1000 * 2000

    assert ws_grid.needs_reduction, "188 k-slices of partials do have to be summed"
    assert not ws_grid.materialises_partials, "but not because a second kernel forces it"
    assert ws_grid.k_slices == 188
    assert ws_grid.accumulator_elements == elements

    on_chip = _cost(ws_grid, TENSOR_CORE)
    assert on_chip.placement is ReductionPlacement.ON_CHIP
    assert on_chip.partial_sums == pytest.approx(187 * elements)
    assert (on_chip.dram_bytes, on_chip.dispatches) == (0.0, 0)
    assert not on_chip.is_free, "no bytes is not no cost: the vector unit pays"

    spilled = _cost(ws_grid, TENSOR_CORE, capacity=1e6)
    assert spilled.placement is ReductionPlacement.DRAM
    assert spilled.partial_sums == pytest.approx(187 * elements)
    assert spilled.dram_bytes == pytest.approx(188 * elements * 4.0 * 2.0)
    assert spilled.dispatches == 1

    metis_grid = grid_for(Dataflow.WEIGHT_STATIONARY, ATTRS, 512, 512)
    local = _cost(metis_grid, D_IMC)
    assert local.placement is ReductionPlacement.LOCAL
    assert local.is_free and local.partial_sums == 0.0
    assert local.partitions == 6, "ceil(3000/512) k-slices, all summed in the periphery"


def test_metis_pays_the_dpu_only_past_the_accumulator_it_declares() -> None:
    """The 16k boundary, from the paper, on both sides (D62, D69).

    ``K = 16384`` is 32 of the array's own 512-input k-slices and exactly the
    depth Fig. 11.3.1 states, so one core sums all of it: ``LOCAL``. At
    ``K = 32768`` (64 slices) one accumulator cannot hold the column, so it is
    split into ``ceil(64 / 32) = 2`` groups, each summed in its own core's
    periphery; only the 2 group partials leave, and the DPU adds
    ``(2 - 1) x M x N`` of them — not the 63 x M x N a round-robin spread of 64
    slices would send (D69).

    Asked without a vector unit so that only the DEPTH reason for grouping is in
    play: N=512 on four cores would otherwise add the occupancy reason.
    """
    inside = grid_for(Dataflow.WEIGHT_STATIONARY, MatmulAttrs(m=64, n=512, k=16384), 512, 512)
    outside = grid_for(Dataflow.WEIGHT_STATIONARY, MatmulAttrs(m=64, n=512, k=32768), 512, 512)

    assert _cost(inside, D_IMC, vector_adder=False).placement is ReductionPlacement.LOCAL
    past = _cost(outside, D_IMC, vector_adder=False)
    assert past.placement is ReductionPlacement.ON_CHIP
    assert past.partitions == 2
    assert past.partial_sums == pytest.approx(1 * 64 * 512)


def test_a_unit_that_runs_a_k_on_grid_dataflow_natively_accumulates_it() -> None:
    """Why the field is read through :func:`accumulation_depth` (D62).

    A profile declaring ``dataflow: ws`` is claiming an accumulator that
    survives across k-slices — without one it could not run the dataflow it says
    is its own. So an undeclared depth means *unbounded* there, which is exactly
    the claim this model made for every K-on-grid grid before D62 and is why
    chip_a's numbers did not move. A unit whose native dataflow is ``os`` has no
    such accumulator and gets 0.
    """
    from bwz.spec import load_chip

    npu_core = load_chip("chip_a").compute_units[0]
    assert npu_core.dataflow is Dataflow.WEIGHT_STATIONARY
    assert npu_core.local_accumulation_inputs == 0, "nothing declared"
    assert accumulation_depth(npu_core) == UNBOUNDED_ACCUMULATION

    assert accumulation_depth(TENSOR_CORE) == 0.0, "os native, nothing declared"
    assert accumulation_depth(D_IMC) == 16384.0, "declared, and it binds"


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

    # Row-stationary: no shipped profile declares it, so this stays a refusal
    # whichever native dataflow the chips are on. (``is`` stopped being one on
    # the MMA units at D62, which declared all three grids on them.)
    tensor_core = load_chip("a100_80gb").compute_units[0]
    why = refusal_reason(tensor_core, Dataflow.ROW_STATIONARY)
    assert why is not None
    assert "stationarity='rs'" in why, "names the field and the request"
    assert "tensor_core" in why, "names the unit that cannot run it"
    assert tensor_core.dataflow.value in why, "names the real capability"

    spec = MatmulSpec.model_validate(
        {"id": "t", "name": "t", "family": "matmul", "m": 512, "n": 512, "k": 4096}
    )
    deployment = DeploymentSpec.model_validate(
        {"batch": 1, "input_tokens": 1, "output_tokens": 1, "stationarity": "rs"}
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
    is deliberately flip-agnostic — it asks for ``rs``, which nothing declares.
    This one is not: it pins the *default*, so a profile silently reverting to
    ``ws`` would fail here rather than quietly re-introducing a decomposition
    whose reduction nobody asked for.

    D62 made ``ws`` and ``is`` *reachable* on these units — the point being to
    measure them against ``os`` on the same silicon — which is why the default
    now has to be pinned separately from the capability. Reachable, not free:
    an MMA unit declares no local accumulator, so a K-on-grid grid there pays
    for its partials.

    Metis and the two hypothetical NPUs stay weight-stationary. Their weights
    ARE their memory (D30), so an accumulator-resident dataflow is not a thing
    they could run.
    """
    from bwz.spec import load_chip

    for chip_id in ("a100_80gb", "h100_sxm", "jetson_orin", "mi300x"):
        matrix_unit = load_chip(chip_id).compute_units[0]
        assert matrix_unit.systolic_dims is not None, chip_id
        assert matrix_unit.dataflow is Dataflow.OUTPUT_STATIONARY, chip_id
        assert refusal_reason(matrix_unit, Dataflow.WEIGHT_STATIONARY) is None, (
            f"{chip_id} declares ws so it can be compared against os (D62)"
        )
        assert matrix_unit.local_accumulation_inputs == 0, (
            f"{chip_id}: declaring an accumulator depth would make that comparison free"
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


def _matmul_report(
    chip_id: str, m: int, n: int, k: int, dtype: str = "fp16", **deployment: object
) -> object:
    from bwz.analysis import analyze
    from bwz.spec import DeploymentSpec, MatmulSpec, load_chip

    spec = MatmulSpec.model_validate(
        {
            "id": "t",
            "name": "t",
            "family": "matmul",
            "m": m,
            "n": n,
            "k": k,
            "a_dtype": dtype,
            "b_dtype": dtype,
        }
    )
    deploy = DeploymentSpec.model_validate(
        {"batch": 1, "input_tokens": 1, "output_tokens": 0, "phase": "prefill", **deployment}
    )
    return analyze(spec, load_chip(chip_id), deploy)


def test_ws_on_a_tensor_core_pays_the_cuda_cores_and_the_overlap_hides_it() -> None:
    """The comparison D62 exists for, hand-computed on A100 at 4096^3 fp16.

    ``ws`` cuts K into ``4096/16 = 256`` k-slices, so every one of the 16.8 M
    output elements ends up with 256 partials and ``255 x 4096 x 4096 =
    4,278,190,080`` additions have to happen on the CUDA cores rather than in
    the tensor cores' accumulators.

    The ratio is the finding, and it is not a coincidence: the CUDA cores are
    ``16x`` slower than the tensor cores, and the reduction is ``2 x 16`` times
    less work than the multiply (``2*M*N*K`` against ``(K/16)*M*N``), so vector
    time is almost exactly HALF matrix time on this array at any large K. The
    overlap therefore hides it completely — ``ws`` and ``os`` report the same
    latency — and what ``ws`` really costs here is the 16.8 M live accumulators,
    which is the thing that eventually bites (see the cliff test below).
    """
    from bwz.report import Bound

    plain = _matmul_report("a100_80gb", 4096, 4096, 4096)
    reduced = _matmul_report("a100_80gb", 4096, 4096, 4096, stationarity="ws")
    assert plain.feasible and reduced.feasible, reduced.infeasibility  # type: ignore[attr-defined]
    a, b = plain.phases[0].ops[0], reduced.phases[0].ops[0]  # type: ignore[attr-defined]

    assert a.reduction_placement is ReductionPlacement.NONE
    assert a.t_reduce_s == 0.0 and a.t_arith_s == a.t_compute_s

    assert b.reduction_placement is ReductionPlacement.ON_CHIP
    assert b.flops == a.flops, "the decomposition never changes the arithmetic (D53)"
    assert b.t_reduce_s == pytest.approx(255 * 4096 * 4096 / (6912 * 1.41e9 * 2 * 0.7))
    assert b.t_reduce_s == pytest.approx(0.5 * b.t_arith_s, rel=0.01), "16x slower, 32x less work"
    assert b.t_compute_s == pytest.approx(b.t_arith_s), "overlapped: the max, not the sum"
    assert b.latency_s == pytest.approx(a.latency_s), "so the reduction costs no latency here"
    assert b.bound is Bound.COMPUTE_BOUND


def test_the_capacity_cliff_flips_the_placement_and_says_so() -> None:
    """Past on-chip capacity the same decomposition costs 20x more (D62).

    A 2048x2048 fp16 output is 8.4 MB of live accumulators against A100's
    60.7 MB, so the partials stay on chip and the reduction is free of latency.
    An 8192x8192 output is 134 MB and does not fit: the partials cross DRAM
    (``256 x 8192 x 8192 x 2 B x 2 = 68.7 GB``), the adds serialise behind the
    matrix work, and a second dispatch is charged. The report has to say the
    placement FLIPPED, not merely report a bigger number.
    """
    fits = _matmul_report("a100_80gb", 2048, 2048, 4096, stationarity="ws")
    spills = _matmul_report("a100_80gb", 8192, 8192, 4096, stationarity="ws")
    small = fits.phases[0].ops[0]  # type: ignore[attr-defined]
    big = spills.phases[0].ops[0]  # type: ignore[attr-defined]

    assert small.reduction_placement is ReductionPlacement.ON_CHIP
    assert small.dram_reduction_bytes == 0.0
    assert small.t_compute_s == pytest.approx(small.t_arith_s)

    assert big.reduction_placement is ReductionPlacement.DRAM
    assert big.dram_reduction_bytes == pytest.approx(256 * 8192 * 8192 * 2.0 * 2.0)
    assert big.t_compute_s == pytest.approx(big.t_arith_s + big.t_reduce_s), "serialised"
    assert big.t_fixed_s > small.t_fixed_s, "and a second dispatch"
    assert any("cliff" in a.lower() for a in spills.assumptions), (  # type: ignore[attr-defined]
        "a discontinuity has to announce itself, not just be bigger"
    )


def test_the_adds_are_never_charged_to_a_matrix_engine() -> None:
    """Trap 8, asserted directly across every chip and dtype (D62).

    ``machine_model`` falls back to the matrix unit when no non-systolic unit
    supports the dtype, so charging the reduction blindly would price
    elementwise adds at the array's rate — free, and the exact opposite of the
    finding. Wherever a K-on-grid stationarity is feasible, the engine paying
    for it must not be a systolic array.
    """
    from bwz.analysis import machine_model
    from bwz.spec import available_chips, load_chip

    seen = 0
    for chip_id in available_chips():
        chip = load_chip(chip_id)
        for dtype in sorted({d for u in chip.compute_units for d in u.supported_dtypes}):
            report = _matmul_report(chip_id, 512, 512, 4096, dtype=dtype.value, stationarity="ws")
            unit = machine_model(chip, dtype).unit
            if Dataflow.WEIGHT_STATIONARY not in unit.dataflows():
                continue
            if not report.feasible:  # type: ignore[attr-defined]
                continue
            op = report.phases[0].ops[0]  # type: ignore[attr-defined]
            if op.reduction_placement in (ReductionPlacement.NONE, ReductionPlacement.LOCAL):
                continue
            seen += 1
            assert machine_model(chip, dtype).vector_unit.systolic_dims is None, (
                f"{chip_id}/{dtype.value} would charge the reduction to a matrix engine"
            )
    assert seen, "the assertion above must actually have been reached"


# ------------------------------------------------------------------ the deal (D68)


def _metis_grid(m: int, n: int, k: int) -> TileGrid:
    return grid_for(Dataflow.WEIGHT_STATIONARY, MatmulAttrs(m=m, n=n, k=k), 512, 512)


def _units_of(grid: TileGrid, ranges: tuple[tuple[int, int], ...]) -> dict[int, set[int]]:
    """Which grid columns each unit is dealt, read off the wave ranges."""
    held: dict[int, set[int]] = {}
    for start, end in ranges:
        for unit, tile in enumerate(range(start, end)):
            held.setdefault(unit, set()).add(tile % grid.cols)
    return held


def test_a_unit_that_sums_locally_keeps_each_column_on_one_core() -> None:
    """N=512, K=8192 with nothing to add partials: 16 k-slices all on ONE core.

    Metis's array asked as a chip WITHOUT a vector unit (like chip_a): the
    column cannot be shared, because its partials would have nowhere to be
    added (D69). With the DPU, see the grouped tests below.

    Hand-computed: grid K x N = 16 x 1. The periphery that sums K belongs to one
    AI core, so the column cannot be split: used cores = min(4, 1) = 1, waves =
    16 rows x ceil(1/1) = 16, occupancy = 16 / (16 x 4) = 0.25. Round-robin
    (the pre-D68 deal) put 4 k-slices on each of 4 cores in 4 waves — 100%
    occupancy for partials that had nowhere to meet.
    """
    grid = _metis_grid(512, 512, 8192)
    assert sums_locally(grid, D_IMC, vector_adder=False)
    dealt = deal_for(grid, D_IMC, vector_adder=False)
    assert (dealt.used_cores, dealt.waves) == (1, 16)
    assert dealt.occupancy == pytest.approx(0.25)
    assert _units_of(grid, dealt.ranges) == {0: {0}}


def test_two_columns_on_four_cores_use_two_and_idle_two() -> None:
    """N=1024, K=4096: 2 columns x 8 k-slices -> 2 cores, 8 waves, 16/(8x4) = 0.5."""
    grid = _metis_grid(512, 1024, 4096)
    dealt = deal_for(grid, D_IMC, vector_adder=False)
    assert (dealt.used_cores, dealt.waves) == (2, 8)
    assert dealt.occupancy == pytest.approx(0.5)
    assert _units_of(grid, dealt.ranges) == {0: {0}, 1: {1}}


def test_a_wave_never_straddles_a_grid_row_under_the_local_deal() -> None:
    """N=2560 (5 columns), K=1024 (2 rows) on 4 cores: rows split 4 + 1.

    Hand-computed: waves = 2 rows x ceil(5/4) = 4, ranges (0,4) (4,5) (5,9)
    (9,10); occupancy 10 / (4 x 4) = 0.625. Every unit sees the same column in
    both rows, which is the claim LOCAL placement makes.
    """
    grid = _metis_grid(512, 2560, 1024)
    dealt = deal_for(grid, D_IMC, vector_adder=True)
    assert dealt.ranges == ((0, 4), (4, 5), (5, 9), (9, 10))
    assert dealt.occupancy == pytest.approx(0.625)
    assert _units_of(grid, dealt.ranges) == {0: {0, 4}, 1: {1}, 2: {2}, 3: {3}}
    for start, end in dealt.ranges:
        assert start // grid.cols == (end - 1) // grid.cols


def test_when_columns_divide_the_cores_both_deals_agree() -> None:
    """N=K=2048: 4 columns on 4 cores — the case round-robin got right by luck."""
    grid = _metis_grid(512, 2048, 2048)
    assert deal_for(grid, D_IMC, vector_adder=True).ranges == (
        deal(grid, 4, keep_k_on_unit=False).ranges
    )


def test_a_unit_whose_partials_leave_anyway_keeps_round_robin() -> None:
    """A100 ws: no local accumulator, so K is spread for occupancy as before (D62).

    512x512x4096 on 16x16: 256 x 32 = 8192 tiles over 432 cores = 19 waves.
    """
    grid = grid_for(Dataflow.WEIGHT_STATIONARY, MatmulAttrs(m=512, n=512, k=4096), 16, 16)
    assert not sums_locally(grid, TENSOR_CORE, vector_adder=True)
    dealt = deal_for(grid, TENSOR_CORE, vector_adder=True)
    assert (dealt.used_cores, dealt.waves) == (432, 19)
    assert dealt.occupancy == pytest.approx(8192 / (19 * 432))


def test_k_on_the_grid_columns_cannot_meet_in_one_periphery() -> None:
    """``is`` puts K on the COLUMN axis: a wave spreads a row's k-slices over units."""
    grid = grid_for(Dataflow.INPUT_STATIONARY, MatmulAttrs(m=512, n=512, k=4096), 512, 512)
    assert not sums_locally(grid, D_IMC, vector_adder=True)


def test_the_round_robin_boundary_is_the_pre_d68_formula() -> None:
    """Coalescing a round-robin run at fractional waves must not move a tile tag."""
    grid = grid_for(Dataflow.OUTPUT_STATIONARY, MatmulAttrs(m=1000, n=2000, k=3000), 16, 16)
    dealt = deal(grid, 432, keep_k_on_unit=False)
    for position in (0.0, 0.4, 1.0, 2.5, 7.25, float(dealt.waves)):
        assert dealt.boundary(position) == min(grid.tiles, int(position * 432))


# ------------------------------------------------------------ K-groups (D69)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((512, 2048, 2048), 1),  # 4 columns fill 4 cores: nothing to gain
        ((512, 1024, 4096), 2),  # 2 columns: 2 cores per column
        ((512, 512, 8192), 4),  # 1 column: 4 cores per column
        ((512, 2048, 32768), 2),  # depth: 64 slices / 32 per accumulator
    ],
)
def test_k_groups_is_the_larger_of_the_depth_and_occupancy_reasons(
    shape: tuple[int, int, int], expected: int
) -> None:
    """The table in :func:`k_groups`'s docstring, hand-computed on Metis."""
    assert k_groups(_metis_grid(*shape), D_IMC, vector_adder=True) == expected


def test_a_narrow_column_is_shared_and_each_core_sums_its_own_group() -> None:
    """Metis 512x512x8192, with the DPU: 4 cores share the one column (D69).

    Hand-computed: K_GROUPS = 4, blocks of 4 grid rows = 4 tiles, one wave each:
    4 waves, occupancy 16 / (4 x 4) = 1.0. Core u gets row-offset u in every
    block, so it sums k-slices u, u+4, u+8, u+12 locally, and only the 4 group
    partials per output leave: 3 x 512 x 512 = 786,432 DPU adds, ON_CHIP.
    """
    grid = _metis_grid(512, 512, 8192)
    dealt = deal_for(grid, D_IMC, vector_adder=True)
    assert (dealt.k_groups, dealt.used_cores, dealt.waves) == (4, 4, 4)
    assert dealt.occupancy == pytest.approx(1.0)
    slices_of: dict[int, list[int]] = {}
    for start, end in dealt.ranges:
        for unit, tile in enumerate(range(start, end)):
            slices_of.setdefault(unit, []).append(tile // grid.cols)
    assert slices_of == {u: [u, u + 4, u + 8, u + 12] for u in range(4)}

    cost = _cost(grid, D_IMC)
    assert cost.placement is ReductionPlacement.ON_CHIP
    assert cost.partitions == 4
    assert cost.partial_sums == pytest.approx(3 * 512 * 512)


def test_without_a_vector_unit_nothing_is_shared() -> None:
    """chip_a's array has no non-systolic unit: its adds would land on the array."""
    npu_core = load_chip("chip_a").compute_units[0]
    grid = _metis_grid(512, 512, 8192)
    assert k_groups(grid, npu_core, vector_adder=False) == 1
    assert _cost(grid, npu_core, vector_adder=False).placement is ReductionPlacement.LOCAL
