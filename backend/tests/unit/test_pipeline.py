"""The tile schedule behind the roofline.

The property that matters throughout: the trace is a *decomposition* of numbers
the report already published, so it can never disagree with them. Every test
below is a form of that statement.
"""

from __future__ import annotations

import math

import pytest

from bwz.analysis import analyze, idealised, machine_model
from bwz.analysis.dataflow import plan_dataflow
from bwz.analysis.pipeline import Lane, PipelineTrace, Stage, build_trace, tile_count
from bwz.analysis.roofline import MATRIX_OP_TYPES, MachineModel
from bwz.graph import GraphPhase, build_graph
from bwz.operators.base import cost_of
from bwz.report import Report
from bwz.spec import BDataflow, DeploymentSpec, DType, MatmulSpec, load_chip, load_model
from bwz.spec.hardware_spec import Dataflow


def _spec(m: int, n: int, k: int, dtype: str = "fp16") -> MatmulSpec:
    return MatmulSpec.model_validate(
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


def _deployment(*, b_dataflow: BDataflow | None = None, iterations: int = 1) -> DeploymentSpec:
    payload: dict[str, object] = {"batch": 1, "input_tokens": 1, "output_tokens": 1}
    if b_dataflow is not None:
        payload["b_dataflow"] = b_dataflow
    if iterations != 1:
        payload["iterations"] = iterations
    return DeploymentSpec.model_validate(payload)


def _trace(
    spec: MatmulSpec,
    chip_id: str,
    *,
    ideal: bool = True,
    max_steps: int = 64,
    b_dataflow: BDataflow | None = None,
    iterations: int = 1,
) -> tuple[PipelineTrace, Report]:
    chip = load_chip(chip_id)
    if ideal:
        chip = idealised(chip)
    deployment = _deployment(b_dataflow=b_dataflow, iterations=iterations)
    report = analyze(spec, chip, deployment)
    assert report.feasible, report.infeasibility
    graph = build_graph(spec, deployment, GraphPhase.STATIC)
    machine: MachineModel = machine_model(chip, spec.operand_dtype)
    dataflow = None
    if b_dataflow is not None:
        dataflow = plan_dataflow(
            graph.ops[0],
            machine,
            chip,
            deployment,
            a_bytes=cost_of(graph.ops[0], graph.tensors).input_bytes,
        )
    return (
        build_trace(
            graph,
            report.phases[0],
            machine,
            double_buffered=report.memory.double_buffered,
            max_steps=max_steps,
            dataflow=dataflow,
        ),
        report,
    )


def test_tile_count_is_the_utilisation_decomposition() -> None:
    """Whatever the grid is, it is the one ``systolic_utilisation`` divides by.

    M=1, N=K=4096 on A100's 16x16 tile. Output-stationary — what the profile now
    declares (D53) — grids the OUTPUT: ceil(1/16) x ceil(4096/16) = 1 x 256, each
    tile sweeping all 4096 of K inside its own accumulator. Weight-stationary
    grids B instead: 256 x 256 = 65 536 tiles, each a slice of the contraction
    that later has to be summed. Both cover the same M*N*K MACs; they disagree
    only about how the work is cut up, and this is the function that decides —
    the same one the utilisation figure calls, which is what keeps the picture
    and the number from telling different stories.
    """
    chip = load_chip("a100_80gb")
    graph = build_graph(_spec(1, 4096, 4096), _deployment(), GraphPhase.STATIC)

    assert machine_model(chip, DType.FP16).stationarity is Dataflow.OUTPUT_STATIONARY
    assert tile_count(graph.ops[0], machine_model(chip, DType.FP16)) == 1 * 256
    weight_stationary = machine_model(chip, DType.FP16, stationarity=Dataflow.WEIGHT_STATIONARY)
    assert tile_count(graph.ops[0], weight_stationary) == 256 * 256


def test_lane_spans_sum_back_to_the_reported_terms() -> None:
    """DRAM busy is t_dram, core busy is t_compute + t_fixed. No new cost."""
    trace, report = _trace(_spec(4096, 4096, 4096, "int8"), "chip_b", ideal=True)
    op = report.phases[0].ops[0]
    busy = trace.busy_s

    assert busy[Lane.DRAM] == pytest.approx(op.t_dram_s, rel=1e-9)
    assert op.dram_read_bytes + op.dram_write_bytes == pytest.approx(op.dram_bytes)
    assert busy[Lane.CORE] == pytest.approx(op.t_compute_s + op.t_fixed_s, rel=1e-9)


def test_double_buffering_holds_exactly_two_tiles() -> None:
    """SRAM occupancy is two tiles deep, never more.

    The buffer being freed is what bounds it: a tile cannot be fetched until the
    one two steps back has been consumed. Without that constraint the schedule
    would quietly assume infinite capacity.
    """
    trace, _ = _trace(_spec(4096, 4096, 4096, "int8"), "chip_b")
    assert trace.double_buffered

    holds = [s for s in trace.spans if s.stage is Stage.HOLD]
    edges = [(s.start_s, 1) for s in holds] + [(s.end_s, -1) for s in holds]
    edges.sort()
    depth = concurrent = 0
    for _at, delta in edges:
        concurrent += delta
        depth = max(depth, concurrent)

    assert depth == 2


def test_totals_match_the_report() -> None:
    """The quantities on the figure are the report's, not a recount.

    A picture that says "39.9 MB moved" while the report says something else is
    worse than no picture, so the bars carry the report's own bytes and
    operations (D19).
    """
    trace, report = _trace(_spec(10_000, 10_000, 10_000), "a100_80gb")
    op = report.phases[0].ops[0]
    totals = trace.totals

    assert totals[Lane.DRAM] == pytest.approx(op.dram_bytes, rel=1e-9)
    assert totals[Lane.CORE] == pytest.approx(op.flops, rel=1e-9)
    # SRAM is a stock, not a flow: what the buffers hold at once, bounded by the
    # capacity the planner granted.
    assert 0 < totals[Lane.SRAM] <= load_chip("a100_80gb").on_chip_capacity_bytes


def test_concurrency_reads_as_a_depth_not_a_duty_cycle() -> None:
    """SRAM occupancy is ~2 because two buffers are held, not because a resource
    was busy 198% of the time.

    DRAM and the array are single serial resources: their spans never overlap, so
    their mean concurrency is a duty cycle and cannot exceed 1. SRAM is *n*
    buffers, so the same ratio counts buffers. Reporting the second as a
    percentage is what made a correct schedule look broken.
    """
    trace, _ = _trace(_spec(10_000, 10_000, 10_000), "a100_80gb")
    concurrency = trace.concurrency

    dram_mean, dram_peak = concurrency[Lane.DRAM]
    core_mean, core_peak = concurrency[Lane.CORE]
    sram_mean, sram_peak = concurrency[Lane.SRAM]

    assert dram_peak == core_peak == 1
    assert dram_mean <= 1.0 and core_mean <= 1.0
    assert sram_peak == 2
    assert 1.9 < sram_mean <= 2.0


def test_fill_drain_is_what_the_roofline_omits() -> None:
    """The schedule costs more than `max(load, compute)` — a load at the head and
    a store at the tail that nothing overlaps.

    The roofline reports the max alone, which is the many-tiles limit. Here it is
    a fraction of a percent; on a decode projection, which is a handful of tiles
    rather than thousands, it is not.
    """
    trace, report = _trace(_spec(10_000, 10_000, 10_000), "a100_80gb")
    op = report.phases[0].ops[0]

    assert op.t_dram_s > 0, "pick a shape that actually touches DRAM"
    assert trace.total_s > trace.reported_latency_s
    assert trace.fill_drain_s == pytest.approx(trace.total_s - trace.reported_latency_s, rel=1e-9)


def test_operation_trace_reproduces_the_reported_latency_exactly() -> None:
    """A network's steps do not pipeline against each other in this model (D5a),
    so the picture must not be faster than the report it illustrates."""
    chip = load_chip("a100_80gb")
    model = load_model("llama3_8b")
    deployment = DeploymentSpec.model_validate(
        {"batch": 1, "input_tokens": 512, "output_tokens": 1}
    )
    report = analyze(model, chip, deployment)
    decode = report.phase(GraphPhase.DECODE)
    assert decode is not None

    trace = build_trace(
        build_graph(model, deployment, GraphPhase.DECODE),
        decode,
        machine_model(chip, DType.FP16),
        double_buffered=report.memory.double_buffered,
    )

    assert trace.fill_drain_s == 0.0
    assert trace.total_s == pytest.approx(decode.latency_s, rel=1e-9)


def test_each_engines_spans_name_only_families_that_engine_can_run() -> None:
    """A matrix family cannot execute on the vector unit, and vice versa.

    D28 split a coalesced group's compute *time* between the two engines but
    named both halves after the group's dominant family — which is a matmul in
    essentially every group of a transformer. The vector lane's spans therefore
    carried ``op_type="matmul"``, and the timeline's hover read
    ``EXEC — matmul on the vector unit``: a claim that contradicts D27, on the
    one lane D27 exists to separate out.

    Llama-3-8B decode on A100 coalesces 451 operations into 32 blocks, and every
    block mixes matmul with norms and elementwise work, so this asserts over a
    case where the naive answer is wrong on every step.
    """
    chip = load_chip("a100_80gb")
    model = load_model("llama3_8b")
    deployment = DeploymentSpec.model_validate(
        {"batch": 1, "input_tokens": 512, "output_tokens": 1}
    )
    report = analyze(model, chip, deployment)
    decode = report.phase(GraphPhase.DECODE)
    assert decode is not None
    trace = build_trace(
        build_graph(model, deployment, GraphPhase.DECODE),
        decode,
        machine_model(chip, DType.FP16),
        double_buffered=report.memory.double_buffered,
    )

    matrix = {t.value for t in MATRIX_OP_TYPES}
    core = {s.op_type for s in trace.spans if s.lane is Lane.CORE and s.stage is Stage.EXEC}
    vector = {s.op_type for s in trace.spans if s.lane is Lane.VECTOR and s.stage is Stage.EXEC}

    assert core and vector, "this workload must exercise both engines for the test to mean anything"
    assert core <= matrix, f"array span named a non-matrix family: {sorted(core - matrix)}"
    assert not (vector & matrix), f"vector span named a matrix family: {sorted(vector & matrix)}"

    # The split is a relabelling, not a re-costing: the two lanes still sum to
    # the phase's compute time and arithmetic (D19).
    busy = trace.busy_s
    totals = trace.totals
    assert busy[Lane.CORE] + busy[Lane.VECTOR] == pytest.approx(
        decode.t_compute_s + decode.t_fixed_s, rel=1e-9
    )
    assert totals[Lane.CORE] + totals[Lane.VECTOR] == pytest.approx(decode.flops, rel=1e-9)


def test_coalescing_preserves_the_span() -> None:
    """Drawing 64 rows instead of every step changes the picture's resolution and
    not its length.

    Since D30 a step is a **wave**, not a tile: 625x625 = 390 625 tiles over
    A100's 432 tensor cores is ``ceil(390625/432)`` = 905 waves, and that is the
    schedule's real step count.
    """
    fine, _ = _trace(_spec(10_000, 10_000, 10_000), "a100_80gb", max_steps=256)
    coarse, _ = _trace(_spec(10_000, 10_000, 10_000), "a100_80gb", max_steps=8)

    assert fine.tiles == coarse.tiles == math.ceil(625 * 625 / 432) == 905
    assert fine.steps == 256
    assert coarse.steps == 8
    # Coarser steps mean a larger fill/drain, because fill/drain is one step.
    assert coarse.total_s > fine.total_s


def test_loads_are_split_by_operand_and_reconcile_with_the_report() -> None:
    """B and A cross the bus separately, and the two bars sum to the one number.

    They obey different residency fractions and spill at different times, so a
    single LOAD figure cannot say which operand moved. At 8192-cubed INT8 on
    Metis both are now 100% non-resident: B does not fit the 16 resident tiles,
    and A is compulsory traffic read exactly once (D33) — the asymmetry D31
    documented is gone, but the reconciliation it exists to test remains.
    """
    chip = idealised(load_chip("metis_aipu"))
    trace, report = _trace(_spec(8192, 8192, 8192, "int8"), "metis_aipu", max_steps=256)
    op = report.phases[0].ops[0]

    weights, activations = trace.operand_bytes
    reads, writes = trace.direction_bytes

    assert weights == pytest.approx(op.dram_weight_read_bytes, rel=1e-9)
    assert activations == pytest.approx(op.dram_activation_read_bytes, rel=1e-9)
    assert weights + activations == pytest.approx(op.dram_read_bytes, rel=1e-9)
    assert reads == pytest.approx(op.dram_read_bytes, rel=1e-9)
    assert writes == pytest.approx(op.dram_write_bytes, rel=1e-9)
    # B does not fit the array and A is read exactly once: both operands are
    # full compulsory traffic now (D33), where A used to get a residency
    # discount — the 2x gap in D31's numbers is gone.
    assert weights == pytest.approx(activations, rel=1e-9)
    assert weights > 0
    assert chip.on_chip_capacity_bytes > 0


def test_a_is_staged_once_per_k_slice_in_the_trace() -> None:
    """D33: the single-matmul trace does not show A streaming every wave.

    8192-cubed INT8 on Metis tiles to ceil(K/512) = 16 k-slices, and the DRAM
    lane carries exactly one LOAD_A event per k-slice — the whole staging, fed
    once to every tile of the group — instead of a per-wave share that reads as
    a re-read.
    """
    trace, report = _trace(_spec(8192, 8192, 8192, "int8"), "metis_aipu", max_steps=64)
    op = report.phases[0].ops[0]
    a_spans = [s for s in trace.spans if s.stage is Stage.LOAD_A]

    assert len(a_spans) == 16, "one staging event per k-slice, ceil(8192/512)"
    assert all(s.a_fetch_mode == "stage" for s in a_spans)
    expected = op.dram_activation_read_bytes / 16
    assert all(s.bytes_moved == pytest.approx(expected, rel=1e-9) for s in a_spans)
    assert sum(s.bytes_moved for s in a_spans) == pytest.approx(
        op.dram_activation_read_bytes, rel=1e-9
    )
    assert all("A k-slice" in s.label for s in a_spans)
    # D48: the label names the tile's own size (M rows x array rows), not just
    # its position — 8192-cubed on Metis's 512x512 array is "8192x512".
    assert all("(8192x512)" in s.label for s in a_spans)

    # The non-matmul path is untouched: a network's activations still stream
    # (inter-op reuse, no k-slice staging to draw).
    chip = load_chip("a100_80gb")
    model = load_model("llama3_8b")
    deployment = DeploymentSpec.model_validate(
        {"batch": 1, "input_tokens": 512, "output_tokens": 1}
    )
    report = analyze(model, chip, deployment)
    decode = report.phase(GraphPhase.DECODE)
    assert decode is not None
    trace_net = build_trace(
        build_graph(model, deployment, GraphPhase.DECODE),
        decode,
        machine_model(chip, DType.FP16),
        double_buffered=report.memory.double_buffered,
    )
    assert not any(s.a_fetch_mode == "stage" for s in trace_net.spans)


def test_a_load_bytes_do_not_inflate_when_the_last_wave_is_underfull() -> None:
    """D46: a 1x1x2 matmul is 1 real tile on A100's 432-array tensor core.

    Before D46, the k-slice-opening count was capped at ``waves * units`` — the
    array's full theoretical capacity, 432 slots here — instead of the real
    tile count (1). Every idle slot in that lone, mostly-empty wave was counted
    as if it, too, opened a fresh A k-slice, so the single LOAD_A span carried
    ``4 B x 432 = 1728 B`` instead of A's true 4 B (``1x2x2 B`` at fp16). That
    inflated byte count fed straight into the span's *duration* too, since
    duration is bytes times a bytes-to-seconds rate: the trace's DRAM busy time
    came out near 288x the report's own ``t_dram``.
    """
    trace, report = _trace(_spec(1, 1, 2, "fp16"), "a100_80gb", max_steps=64)
    op = report.phases[0].ops[0]
    a_spans = [s for s in trace.spans if s.stage is Stage.LOAD_A]

    assert len(a_spans) == 1, "one real tile opens exactly one k-slice"
    assert a_spans[0].bytes_moved == pytest.approx(4.0, rel=1e-9)
    assert a_spans[0].bytes_moved == pytest.approx(op.dram_activation_read_bytes, rel=1e-9)

    dram_busy_s = sum(s.duration_s for s in trace.spans if s.lane is Lane.DRAM)
    assert dram_busy_s == pytest.approx(op.t_dram_s, rel=1e-6)


def test_span_tile_ranges_partition_the_real_tiles_without_gaps_or_overlap() -> None:
    """D48: every span's ``[tile_start, tile_end)`` is the animation panel's only
    source for "which tile is this". An 8192-cubed INT8 matmul on Metis (four
    512x512 arrays) is ceil(8192/512)^2 = 256 tiles over 64 drawn steps, 4 tiles
    each — the same shape ``test_a_tile_step_is_a_wave_not_a_single_tile`` uses.
    Every step's range must be exactly 4 wide, and the 64 ranges must partition
    [0, 256) with no gap and no overlap. A k-slice's own boundary is
    ``n_tiles_per_ks`` wide (``ceil(8192/512) = 16``); the 16 LOAD_A openings
    must land on k-slice indices 0..15 in order once ``tile_start`` is divided
    by that width, one opening per k-slice (D33).
    """
    trace, _report = _trace(_spec(8192, 8192, 8192, "int8"), "metis_aipu", max_steps=64)

    exec_spans = [s for s in trace.spans if s.stage is Stage.EXEC]
    assert all(s.tile_start is not None and s.tile_end is not None for s in exec_spans)
    by_step: dict[int, tuple[int, int]] = {
        s.step: (s.tile_start, s.tile_end)  # type: ignore[misc]  # asserted non-None above
        for s in exec_spans
    }
    assert len(by_step) == 64
    assert all(end - start == 4 for start, end in by_step.values())
    ordered = [by_step[i] for i in range(64)]
    assert ordered[0][0] == 0
    assert ordered[-1][1] == 256
    assert all(ordered[i][1] == ordered[i + 1][0] for i in range(63)), "no gap or overlap"

    # A LOAD_A span's tile_start is the opening step's own tile window, not the
    # k-slice's own boundary — e.g. the k-slice-0 opening fires at tile_start=12
    # (the last 4-tile step before the boundary), not 0. What the animation
    # panel actually needs is the derived k-slice index, tile_start //
    # tiles_per_ks (n_tiles_per_ks = ceil(8192/512) = 16 here) — and that must
    # land on 0, 1, 2, ... 15 in order, one per k-slice.
    tiles_per_ks = 16
    a_spans = [s for s in trace.spans if s.stage is Stage.LOAD_A]
    assert len(a_spans) == 16, "one opening per k-slice, ceil(8192/512)"
    k_slice_indices = [s.tile_start // tiles_per_ks for s in a_spans if s.tile_start is not None]
    assert k_slice_indices == list(range(16))


def test_a_label_names_every_band_a_step_opens_not_just_the_first() -> None:
    """D48, in D53's vocabulary: 1000x1000x2000 fp16 on A100, now
    output-stationary, grids the OUTPUT — ceil(1000/16) = 63 row-bands of M by
    63 n-tiles = 3969 tiles over 10 waves, 432 (one full wave) per drawn step.

    432 tiles is almost 7 bands' worth (432 / 63 ~= 6.86): one step opens
    *several* bands at once, not one. The label used to name only the first
    (``open_tile // grid.cols``), even though the byte total already
    (correctly) charged every band the step actually opens — a step spanning
    bands 1 through 7 read as "A row-band 1/63", silently dropping 2-6. It
    must name the whole span it opens.

    D48's own first fix for this got the *count* wrong the other way — using
    the block the step's *last tile merely touches* (``(end_tile - 1) //
    grid.cols``) rather than the block it *finishes* (``end_tile //
    grid.cols``) named ONE TOO MANY: step 0 read "1-7/63" (7 bands) while its
    own ``bytes_moved`` only ever charged 6 — the block a step's last tile
    lands in is finished by whichever *later* step's own end crosses out of
    it, not by this one, so this step must not claim it. Every assertion below
    is a form of that one invariant: the label's band *count* must equal
    ``bytes_moved / (one band's bytes)`` exactly, for every span, not just a
    hand-checked pair.

    That the bands are bands of M rather than slices of K is exactly what the
    stationarity decides, and nothing else about the shape of this test moves
    with it — which is the point of routing both through the grid.
    """
    trace, report = _trace(_spec(1000, 1000, 2000, "fp16"), "a100_80gb", max_steps=64)
    op = report.phases[0].ops[0]
    a_spans = [s for s in trace.spans if s.stage is Stage.LOAD_A]
    bands = 63
    # Every band is a full 16 rows here: 63 * 16 = 1008 != 1000, so the last is
    # ragged and the fleet-wide average is NOT the right per-band weight. Only
    # the full bands are checked against it; the ragged tail has its own test.
    full_band_bytes = 16 * 2000 * 2.0

    assert a_spans[0].label == "A row-bands 1-6/63 (16x2000) — staged once, feed their tiles"
    assert a_spans[1].label == "A row-bands 7-13/63 (16x2000) — staged once, feed their tiles"
    assert sum(s.bytes_moved for s in a_spans) == pytest.approx(
        op.dram_activation_read_bytes, rel=1e-9
    )
    for span in a_spans[:-1]:
        assert span.tile_start is not None and span.tile_end is not None
        assert span.grid is not None
        first = span.tile_start // span.grid.cols + 1
        last = span.tile_end // span.grid.cols
        assert span.grid.a_events == bands
        assert (
            f"A row-band {first}/" in span.label
            if first == last
            else f"A row-bands {first}-{last}/" in span.label
        )
        # The label's own band count must reconcile with the bytes this exact
        # span carries — the whole point being fixed here.
        assert span.bytes_moved == pytest.approx((last - first + 1) * full_band_bytes, rel=1e-9)


def test_a_band_bytes_are_exact_not_a_fleet_wide_average() -> None:
    """D48: 1000x2000x3000 fp16 on A100, output-stationary, cuts M into
    ceil(1000/16) = 63 row-bands of 16 rows — except the last, which is only
    1000 - 62*16 = 8 rows (63*16 = 1008 != 1000, so M does not divide evenly).

    A uniform total/63 average would charge every band ~95.2 kB regardless —
    under-charging the 62 full-height bands and over-charging the ragged last
    one. A step naming three full bands (1-3) must charge exactly
    3 x 16 x 3000 x 2 B = 288 000 B, not a fraction of the fleet-wide average;
    the step naming the ragged tail alone must charge 8 x 3000 x 2 = 48 000 B.
    """
    trace, report = _trace(_spec(1000, 2000, 3000, "fp16"), "a100_80gb", max_steps=64)
    op = report.phases[0].ops[0]
    a_spans = [s for s in trace.spans if s.stage is Stage.LOAD_A]

    assert a_spans[0].label == "A row-bands 1-3/63 (16x3000) — staged once, feed their tiles"
    assert a_spans[0].bytes_moved == pytest.approx(288_000.0, rel=1e-9)
    assert a_spans[-1].label == "A row-band 63/63 (16x3000) — staged once, feeds its tiles"
    assert a_spans[-1].bytes_moved == pytest.approx(48_000.0, rel=1e-9), "8 rows, not 16"
    assert sum(s.bytes_moved for s in a_spans) == pytest.approx(
        op.dram_activation_read_bytes, rel=1e-9
    )


def test_a_tile_step_is_a_wave_not_a_single_tile() -> None:
    """The bars must not show a 4-core chip working one tile at a time (D30).

    Metis has four 512x512 arrays. An 8192-cubed INT8 matmul is
    ceil(8192/512)^2 = 256 tiles, which is 64 waves of 4 — and the trace draws
    64 steps, not 256, with each step carrying four tiles' worth of arithmetic.
    Drawing one bar per tile said the arrays ran in series, contradicting the
    wave occupancy the same report quotes.
    """
    trace, report = _trace(_spec(8192, 8192, 8192, "int8"), "metis_aipu", max_steps=256)

    assert trace.tiles == 64, "steps are waves: ceil(256 tiles / 4 arrays)"
    assert not trace.coalesced, "64 waves fit under the drawing cap, so nothing is merged"
    assert "4 B tiles 512x512" in trace.spans[0].label
    assert trace.spans[0].label.endswith("all in parallel")

    # And it stays a decomposition: the waves still sum to the reported terms.
    op = report.phases[0].ops[0]
    assert trace.busy_s[Lane.CORE] == pytest.approx(op.t_compute_s + op.t_fixed_s, rel=1e-9)

    # When one bar has to coalesce several waves, the label must stop claiming
    # they are simultaneous and say how many run at a time instead.
    coarse, _ = _trace(_spec(8192, 8192, 8192, "int8"), "metis_aipu", max_steps=8)
    assert coarse.coalesced
    assert coarse.spans[0].label.endswith("4 at a time")
    assert "in parallel" not in coarse.spans[0].label


def test_fill_drain_differs_enough_between_chips_to_distort_a_ratio() -> None:
    """Why the comparison headline must be quoted from the reported latency.

    D33's k-slice staging concentrates A into a few large events instead of a
    per-wave trickle, so much less of it overlaps compute. On an 8192-cubed INT8
    matmul the pipeline fill/drain is a rounding error on A100 and 17.7% of the
    latency on Metis — so a ratio measured off the drawn bars reads 3.93x where
    the report says 3.34x, an 18% error in the one number a comparison figure
    exists to show (D35).

    This asserts the *engine* side of that: the gap is real and asymmetric. The
    figure's job is to quote `reported_latency_s`, which `_subtitle` now does.
    """
    fast, fast_report = _trace(_spec(8192, 8192, 8192, "int8"), "a100_80gb", max_steps=256)
    slow, slow_report = _trace(_spec(8192, 8192, 8192, "int8"), "metis_aipu", max_steps=256)

    def share(trace: PipelineTrace) -> float:
        return trace.fill_drain_s / trace.reported_latency_s

    assert share(fast) < 0.01, "A100 overlaps almost everything"
    assert share(slow) > 0.10, "Metis stages A in lumps, so much less overlaps"

    reported = slow.reported_latency_s / fast.reported_latency_s
    drawn = slow.total_s / fast.total_s
    assert reported == pytest.approx(3.34, rel=0.02)
    assert drawn == pytest.approx(3.93, rel=0.02)
    assert drawn > reported * 1.15, "the two disagree by enough to mislead"

    # Whichever is quoted, the spans still decompose the reported terms (D19).
    for trace, report in ((fast, fast_report), (slow, slow_report)):
        op = report.phases[0].ops[0]
        assert trace.busy_s[Lane.DRAM] == pytest.approx(op.t_dram_s, rel=1e-9)


def test_write_ahead_b_dataflow_is_not_a_new_code_path() -> None:
    """Passing an explicit write-ahead plan must reproduce the ``dataflow=None``
    schedule exactly (D40) — every other test in this file draws that schedule
    without ever naming ``b_dataflow``, so the two paths must never disagree.
    """
    spec = _spec(8192, 8192, 8192, "int8")
    implicit, _ = _trace(spec, "metis_aipu", max_steps=64)
    explicit, _ = _trace(spec, "metis_aipu", max_steps=64, b_dataflow=BDataflow.WRITE_AHEAD)

    assert explicit.total_s == implicit.total_s
    assert explicit.spans == implicit.spans


def test_on_demand_exposes_one_waves_compute_on_the_critical_path() -> None:
    """D40: ``b_dataflow=on-demand`` forces ``load_start(i) >= exec_end(i-1)`` —
    B's load can no longer hide behind the *previous* wave's arithmetic.

    Metis (``weight_sets=4``) at ``M=20000, N=K=2048`` int8 is 16 tiles over 4
    arrays: 4 waves, exactly at the resident-tile-capacity boundary
    (``4 x 4 = 16``). Under write-ahead the schedule is DRAM-port-saturated
    end to end, so the on-demand gate binds once and the delay propagates as a
    constant shift: ``total_s`` grows by exactly one wave's compute time,
    ``t_compute_s / steps = 0.0008196 / 4 = 0.0002049 s``.
    """
    spec = _spec(20_000, 2048, 2048, "int8")
    write_ahead, report = _trace(spec, "metis_aipu", b_dataflow=BDataflow.WRITE_AHEAD)
    on_demand, _ = _trace(spec, "metis_aipu", b_dataflow=BDataflow.ON_DEMAND)
    op = report.phases[0].ops[0]

    assert write_ahead.steps == 4
    delay = op.t_compute_s / write_ahead.steps
    assert delay == pytest.approx(0.0002049, rel=1e-6)
    assert on_demand.total_s == pytest.approx(write_ahead.total_s + delay, rel=1e-9)
    assert on_demand.fill_drain_s == pytest.approx(write_ahead.fill_drain_s + delay, rel=1e-9)


def test_persistent_b_dataflow_matches_write_ahead_within_one_pass() -> None:
    """D30/D36: within one pass every B tile is fetched exactly once whichever
    placement is chosen, and write-ahead's double buffering already achieves
    this model's best-case overlap — any ``depth >= 2`` is provably equivalent
    to today's default (``store_ends`` is monotonically non-decreasing, so
    ``freed = store_ends[i-depth] <= dram_free`` for every such depth; swept
    ``depth_override`` from 2 to 64 on this shape and got a bit-identical
    ``total_s`` every time). ``persistent`` therefore gets no schedule change
    at all (D40): its trace is write-ahead's, unmodified. Its real,
    already-implemented distinguishing effect is the cross-``--iterations``
    ``b_write_multiplier`` amortisation, a byte story, not a same-pass timing
    one.
    """
    spec = _spec(20_000, 2048, 2048, "int8")
    write_ahead, _ = _trace(spec, "metis_aipu", b_dataflow=BDataflow.WRITE_AHEAD)
    persistent, _ = _trace(spec, "metis_aipu", b_dataflow=BDataflow.PERSISTENT)

    assert persistent.total_s == write_ahead.total_s
    assert persistent.spans == write_ahead.spans


def test_b_dataflow_never_moves_a_report_number() -> None:
    """D36's "moves zero bytes against each other in a single invocation" claim,
    pinned as a golden rather than left as prose: t_dram, t_compute and the
    reported latency are decided before ``b_dataflow`` is ever read (only
    ``DataflowPlan.b_write_multiplier`` — 1.0 outside persistent+iterations>1 —
    touches a byte total), so only the *trace's* total_s/fill_drain_s may
    differ across the three placements.
    """
    spec = _spec(20_000, 2048, 2048, "int8")
    baseline_report = None
    for b_dataflow in (BDataflow.WRITE_AHEAD, BDataflow.ON_DEMAND, BDataflow.PERSISTENT):
        _, report = _trace(spec, "metis_aipu", b_dataflow=b_dataflow)
        op = report.phases[0].ops[0]
        if baseline_report is None:
            baseline_report = report
        baseline_op = baseline_report.phases[0].ops[0]
        assert op.t_dram_s == pytest.approx(baseline_op.t_dram_s, rel=1e-12)
        assert op.t_compute_s == pytest.approx(baseline_op.t_compute_s, rel=1e-12)
        assert op.latency_s == pytest.approx(baseline_op.latency_s, rel=1e-12)


def test_weight_sets_le_1_gate_falls_back_b_dataflow_to_write_ahead() -> None:
    """A tensor core reads both operands per instruction and holds no resident
    weight bank (``weight_sets=1``, every shipped GPU profile) — there is
    nothing to place ahead of, expose on demand, or keep resident. ``on-demand``
    and ``persistent`` must therefore resolve to ``write-ahead`` and draw its
    exact schedule, with a note explaining why.

    Picked ``M=N=K=128`` on A100 (64 tiles) specifically because it sits well
    inside ``resident_tile_capacity`` (432 = 432 arrays x 1 weight set) — the
    pre-existing persistent-capacity clamp would not fire here on its own, so
    this isolates the new ``weight_sets<=1`` gate from that older one.
    """
    spec = _spec(128, 128, 128, "fp16")
    write_ahead, _ = _trace(spec, "a100_80gb", b_dataflow=BDataflow.WRITE_AHEAD)

    for b_dataflow in (BDataflow.ON_DEMAND, BDataflow.PERSISTENT):
        gated, report = _trace(spec, "a100_80gb", b_dataflow=b_dataflow)
        assert gated.total_s == write_ahead.total_s
        assert gated.spans == write_ahead.spans
        assert any("weight_sets=1" in a for a in report.assumptions)

    # The latent bug this gate fixes for free: before it, a weight_sets=1 chip
    # (no resident bank to amortise at all) still got persistent's iteration
    # discount purely because its tile count happened to fit the capacity
    # clamp's arithmetic. Now the gate resolves b_dataflow to write-ahead
    # before that amortisation check ever runs.
    _, discounted_report = _trace(spec, "a100_80gb", b_dataflow=BDataflow.PERSISTENT, iterations=4)
    assert not any("amortised" in a for a in discounted_report.assumptions)


def test_an_on_chip_reduction_is_drawn_alongside_the_array_not_behind_it() -> None:
    """D62's overlap, in the picture as well as in the number.

    ``ws`` on A100 puts K on the grid, so the CUDA cores sum partials while the
    tensor cores build the next k-slice. The report charges ``max(matrix,
    vector)`` for that, and a trace that queued the adds after the last wave
    would draw a serialisation the report did not charge — and would push the
    drawn span past the reported latency, which D19 forbids.

    Split-K's reduction *is* a tail, and stays one: two kernels, so nothing can
    be summed until the first has finished everywhere.
    """
    chip = idealised(load_chip("a100_80gb"))
    spec = MatmulSpec.model_validate(
        {"id": "t", "name": "t", "family": "matmul", "m": 2048, "n": 2048, "k": 4096}
    )
    deployment = DeploymentSpec.model_validate(
        {"batch": 1, "input_tokens": 1, "output_tokens": 1, "stationarity": "ws"}
    )
    report = analyze(spec, chip, deployment)
    assert report.feasible, report.infeasibility
    graph = build_graph(spec, deployment, GraphPhase.STATIC)
    machine = machine_model(chip, spec.operand_dtype, stationarity=Dataflow.WEIGHT_STATIONARY)
    trace = build_trace(
        graph, report.phases[0], machine, double_buffered=report.memory.double_buffered
    )

    op = report.phases[0].ops[0]
    reduces = [s for s in trace.spans if s.stage is Stage.REDUCE]
    executes = [s for s in trace.spans if s.stage is Stage.EXEC]
    assert len(reduces) == 1 and reduces[0].lane is Lane.VECTOR
    assert reduces[0].bytes_moved == 0.0, "on-chip partials are charged no DRAM traffic"

    # The overlap: it starts while the array is still working, and the lane
    # still sums to exactly the t_reduce_s the report charged (D19).
    assert reduces[0].start_s < max(s.end_s for s in executes)
    assert reduces[0].start_s == pytest.approx(min(s.end_s for s in executes))
    assert reduces[0].duration_s == pytest.approx(op.t_reduce_s)
    assert trace.total_s == pytest.approx(
        max(s.end_s for s in trace.spans if s.lane is not Lane.VECTOR)
    ), "hidden under the matrix work, so it does not extend the span"
