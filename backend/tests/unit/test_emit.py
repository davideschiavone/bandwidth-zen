"""The emitted program is built from the report, and says so in its own source.

``docs/CORRECTIONS.md`` D54. Two guarantees are checked here and neither needs
the program to run: that ``emit.check`` reconciles the ``PREDICTED`` block with
the ``Report`` it came from, and that the file's structure is the one every other
piece of this feature assumes. Running the programs — where the counts are
actually compared against the model — is
``tests/integration/test_emitted_programs_run.py``; that is the slow half, and
this is the half every change can afford.
"""

from __future__ import annotations

import ast
import re
from dataclasses import replace
from pathlib import Path

import pytest

from bwz.analysis import analyze, machine_model
from bwz.analysis.dataflow import plan_dataflow
from bwz.analysis.pipeline import grid_of
from bwz.emit import check, constants_of, default_filename, emit_matmul, predicted_for
from bwz.graph import GraphPhase, build_graph
from bwz.operators.base import cost_of
from bwz.report import ReductionPlacement
from bwz.spec import DeploymentSpec, DType, MatmulSpec, largest_declared_array, load_chip
from bwz.spec.hardware_spec import Dataflow, HardwareSpec

SHAPE = (64, 64, 128)
"""Small enough to run in pure Python, ragged enough nowhere to hide a clipping
bug: 64 and 128 are whole multiples of both shipped array geometries."""


def _chip_declaring(chip_id: str, flow: Dataflow) -> HardwareSpec:
    """*chip_id* with its array forged to declare *flow*.

    ``is`` and ``rs`` are implemented and nothing ships declaring them (D53), so
    the emitter's branches for them are reachable and otherwise untested. Same
    forging ``test_stationarity`` uses, for the same reason.
    """
    base = load_chip(chip_id)
    units = [
        unit.model_copy(update={"dataflow": flow, "supported_dataflows": (flow,)})
        if unit.systolic_dims is not None
        else unit
        for unit in base.compute_units
    ]
    return base.model_copy(update={"compute_units": units, "hypothetical": True})


def _emit(  # type: ignore[no-untyped-def]
    chip_id: str = "a100_80gb",
    shape: tuple[int, int, int] = SHAPE,
    dtype: str = "fp16",
    *,
    split_k: int = 1,
    a_strategy: str = "stage",
    stationarity: Dataflow | None = None,
    chip: HardwareSpec | None = None,
    accumulate: str | None = None,
):
    """Emit one program, exactly the way ``bwz matmul --emit`` does."""
    m, n, k = shape
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
    deployment = DeploymentSpec.model_validate(
        {
            "batch": 1,
            "input_tokens": 1,
            "output_tokens": 0,
            "phase": "prefill",
            "split_k": split_k,
            "a_strategy": a_strategy,
            "stationarity": stationarity,
            "precision": {"weights": dtype, "activations": dtype, "accumulate": accumulate},
        }
    )
    hardware = chip if chip is not None else load_chip(chip_id)
    report = analyze(spec, hardware, deployment)
    assert report.feasible, report.infeasibility
    machine = machine_model(
        hardware, spec.operand_dtype, stationarity=stationarity, k_partitions=split_k
    )
    graph = build_graph(spec, deployment, GraphPhase.STATIC)
    op = graph.ops[0]
    grid = grid_of(op, machine)
    assert grid is not None
    dataflow = plan_dataflow(
        op, machine, hardware, deployment, a_bytes=cost_of(op, graph.tensors).input_bytes
    )
    program = emit_matmul(
        hardware,
        machine,
        grid,
        dataflow,
        report.phases[0].ops[0],
        a_dtype=spec.a_dtype,
        b_dtype=spec.b_dtype,
        c_dtype=spec.result_dtype,
        acc_dtype=deployment.precision.accumulate,
        double_buffered=report.memory.double_buffered,
        command="bwz matmul (test)",
        version="0.0.0-test",
    )
    return program, report, machine, grid, dataflow


# --------------------------------------------------------------------------- shape


@pytest.mark.parametrize("chip_id", ["a100_80gb", "metis_aipu", "h100_sxm", "chip_a"])
def test_every_shipped_chip_emits_a_program_that_parses(chip_id: str) -> None:
    """Syntactically valid Python, and ``check`` reconciles it with the report."""
    dtype = "int8" if chip_id in ("metis_aipu", "chip_a") else "fp16"
    program, _report, _machine, _grid, _dataflow = _emit(chip_id, dtype=dtype)
    ast.parse(program.source)
    check(program)


@pytest.mark.parametrize("flow", list(Dataflow))
def test_every_stationarity_emits_its_own_loop_nest(flow: Dataflow) -> None:
    """All four decompositions, including the two nothing ships declaring (D53).

    What is asserted is the property that separates the two families: a grid
    carrying K on an axis owes partial sums to an accumulator, so its program has
    a ``Partials``; one that sweeps K inside a tile does not, and stores a
    finished result instead.
    """
    program, _report, _machine, grid, _dataflow = _emit(
        chip=_chip_declaring("a100_80gb", flow), stationarity=flow
    )
    ast.parse(program.source)
    check(program)
    assert program.stationarity is flow
    if grid.needs_reduction:
        assert "partials.accumulate" in program.source
        assert "dram.write_c(m0, n0, acc)" not in program.source
    else:
        assert "Partials(" not in program.source
        assert "dram.write_c(m0, n0, acc)" in program.source


def test_the_swept_dimension_is_the_loop_variable() -> None:
    """The nest sweeps what the grid says it sweeps — the whole point of D53.

    Under ``os`` K is the inner loop; under ``ws`` M is, and K is on the grid.
    A program that swept the wrong dimension would still compute ``A @ B``, which
    is exactly why this is checked in the source rather than by running it.
    """
    output_stationary, *_ = _emit("a100_80gb")
    weight_stationary, *_ = _emit("metis_aipu", dtype="int8")
    assert "for kt in range(kt0, kt1):" in output_stationary.source
    assert "for mt in range(M_TILES):" in weight_stationary.source
    assert "for mt in range(M_TILES):" not in output_stationary.source
    assert "for kt in range(kt0, kt1):" not in weight_stationary.source


def test_split_k_emits_the_second_kernel() -> None:
    """Split-K is two kernels, and the emitted file has two (D53)."""
    plain, *_ = _emit("a100_80gb")
    split, *_ = _emit("a100_80gb", split_k=4)
    assert "KERNEL 2" not in plain.source
    assert "KERNEL 2" in split.source
    assert "dram.write_partial(part, m0, n0, acc)" in split.source
    assert split.predicted["partial_dram_bytes"] > 0
    assert plain.predicted["partial_dram_bytes"] == 0


def test_a_strategies_differ_only_in_the_residency_constant() -> None:
    """``stream`` is ``stage`` with the residency set to one tile (D31/D33).

    Both run the same ``pad.band`` statement; what changes is how many tiles one event
    serves, and therefore how many events there are and how many bytes cross.
    """
    staged, *_ = _emit("a100_80gb")
    streamed, *_ = _emit("a100_80gb", a_strategy="stream")
    assert constants_of(streamed.source)["A_RESIDENCY_TILES"] == 1
    assert constants_of(staged.source)["A_RESIDENCY_TILES"] == 4
    assert streamed.predicted["staging_events"] > staged.predicted["staging_events"]
    assert streamed.predicted["a_dram_bytes"] > staged.predicted["a_dram_bytes"]


def test_whole_ramps_the_staging_in_before_wave_zero() -> None:
    """``whole`` moves the same bytes as ``stage``; only the timing changes (D33)."""
    staged, *_ = _emit("a100_80gb")
    whole, *_ = _emit("a100_80gb", a_strategy="whole")
    assert whole.predicted["a_dram_bytes"] == staged.predicted["a_dram_bytes"]
    assert whole.predicted["staging_events"] == staged.predicted["staging_events"]
    assert "for tile in range(TILES):\n        # == LEVEL 3" in whole.source
    assert "\n        pad.band(\n" in whole.source
    assert "for tile in range(TILES):" not in staged.source
    # One staging statement, run early; the tile's own is unchanged and hits the cache.
    assert whole.source.count("band = pad.band(") == staged.source.count("band = pad.band(") == 1


def test_every_split_is_in_one_function() -> None:
    """``walk()`` holds every split, and the levels come in the same order (D67).

    Waves, the core's tile, the grid position, the sweep and the instruction
    used to be spread over ``run_waves`` in the harness, four helpers and
    ``run_tile``, so seeing how one decomposition differs from another meant
    reading six functions. Now there is one, and nothing it needs is defined
    elsewhere under the old names.
    """
    for flow in (Dataflow.OUTPUT_STATIONARY, Dataflow.WEIGHT_STATIONARY, Dataflow.INPUT_STATIONARY):
        program, *_ = _emit("a100_80gb", stationarity=flow)
        for gone in (
            "def tile_row",
            "def tile_col",
            "def partition_of",
            "def stage_a",
            "def run_tile",
            "def run_waves",
            "def reduce_partials",
        ):
            assert gone not in program.source, (flow, gone)
        body = program.source.split("def walk(", 1)[1].split("\ndef main(", 1)[0]
        body = body.split('"""', 2)[2]  # the code, not the docstring's table of it
        levels = [
            line.strip()[:10] for line in body.splitlines() if line.strip().startswith("# == LEVEL")
        ]
        assert levels == ["# == LEVEL"] * 4, (flow, levels)
        order = [
            body.index(marker)
            for marker in (
                "for wave in range(WAVES):",
                "tile = wave * USED_CORES + core_id",
                "LEVEL 3",
                "LEVEL 4",
                "LEVEL 5",
                "mma(",
            )
        ]
        assert order == sorted(order), flow


def test_walk_draws_its_own_tile_grid() -> None:
    """The docstring draws LEVEL 3 for every stationarity, numbered as ``tile`` is.

    The grid is the stationary operand cut into array-sized blocks, so its axes
    must be that operand's two dimensions: C's M x N, B's K x N, A's M x K.
    """
    expected = {
        Dataflow.OUTPUT_STATIONARY: ("rows:    M", "columns: N", "block of C", "no partial"),
        Dataflow.WEIGHT_STATIONARY: ("rows:    K", "columns: N", "block of B", "PARTIAL"),
        Dataflow.INPUT_STATIONARY: ("rows:    M", "columns: K", "block of A", "PARTIAL"),
    }
    for flow, phrases in expected.items():
        program, *_ = _emit("a100_80gb", stationarity=flow)
        doc = program.source.split("def walk(", 1)[1].split('"""', 2)[1]
        assert "┌" in doc and "│  t0  │" in doc, flow
        for phrase in phrases:
            assert phrase in doc, (flow, phrase)


def test_metis_walks_each_column_on_one_core() -> None:
    """D68: the emitted walk deals column-per-core where the unit sums K locally.

    512x512x1024 INT8: one column, two k-slices -> 1 core, 2 waves. The program
    must say so in its constants and its LEVEL 2, and predict what the report
    charged: 2 x 4 - 2 = 6 idle core-waves.
    """
    program, *_ = _emit("metis_aipu", (512, 512, 1024), "int8")
    constants = constants_of(program.source)
    assert (constants["USED_CORES"], constants["COLUMN_GROUPS"], constants["WAVES"]) == (1, 1, 2)
    assert "kt, column_group = divmod(wave, COLUMN_GROUPS)" in program.source
    assert "nt = column_group * USED_CORES + core_id" in program.source
    assert "weight_set = kt % WEIGHT_SETS" in program.source
    assert program.predicted["idle_core_waves"] == 6
    assert "SAME core's, in its other waves" in program.source


def test_a100_keeps_the_round_robin_deal() -> None:
    """No local accumulator on a tensor core, so nothing about its walk changes."""
    program, *_ = _emit("a100_80gb", stationarity=Dataflow.WEIGHT_STATIONARY)
    assert "tile = wave * USED_CORES + core_id" in program.source
    assert "COLUMN_GROUPS" not in program.source


def test_the_header_draws_which_operand_is_which_shape() -> None:
    """Right after the command: A[M, K] @ B[K, N] -> C[M, N], with this run's sizes."""
    program, *_ = _emit("a100_80gb", shape=(64, 1000, 128), dtype="int8")
    header = program.source.split('"""', 2)[1]
    picture = header.split("The shapes:", 1)[1].split("Not an illustration.", 1)[0]
    assert header.index("$ bwz") < header.index("The shapes:")
    for label in ("M = 64", "N = 1,000", "K = 128", "A · int8", "B · int8", "C · int8"):
        assert label in picture, label
    assert "C has no K" in picture


def test_a_big_grid_is_drawn_with_its_middle_elided() -> None:
    """Past 8 rows or columns the picture keeps both edges and elides the middle."""
    program, *_ = _emit("a100_80gb", shape=(512, 1000, 256))
    doc = program.source.split("def walk(", 1)[1].split('"""', 2)[1]
    assert "32 x 63 = 2,016 tiles" in doc
    for shown in ("mt = 3", "mt = 29", "mt = 31", "nt=62", "t2015"):
        assert shown in doc, shown
    for hidden in ("mt = 4", "mt = 28", "nt=4 "):
        assert hidden not in doc, hidden
    assert "⋮" in doc and "…" in doc
    box = [line for line in doc.splitlines() if line.strip().startswith("│")]
    assert len(box) == 8, "at most 8 rows of boxes"


# ----------------------------------------------------------------------- constants


def test_the_prediction_is_the_report() -> None:
    """Every asserted quantity traces to the ``Report``, not to a second model."""
    program, report, machine, grid, dataflow = _emit("a100_80gb")
    result = report.phases[0].ops[0]
    assert program.predicted == predicted_for(machine, grid, dataflow, result)
    assert program.predicted["a_dram_bytes"] == result.dram_activation_read_bytes
    assert program.predicted["c_dram_bytes"] == result.dram_write_bytes
    assert program.predicted["macs"] == grid.m * grid.n * grid.k
    assert program.predicted["tiles"] == grid.tiles


def test_two_core_counts_and_the_gap_is_wave_occupancy() -> None:
    """``USED_CORES`` is what there is work for; the idle remainder is the loss.

    A100 has 432 cores and a 64x64 fp16 matmul has 16 tiles, so 416 core-waves
    do nothing — which is exactly the 16/432 = 3.7% the report quotes (D30).
    """
    program, report, *_ = _emit("a100_80gb")
    constants = constants_of(program.source)
    assert constants["AVAILABLE_CORES"] == 432
    assert constants["USED_CORES"] == 16
    assert constants["WAVES"] == 1
    assert program.predicted["idle_core_waves"] == 416
    occupancy = 1 - 416 / (1 * 432)
    assert report.phases[0].ops[0].utilization == pytest.approx(occupancy, rel=1e-12)


def test_the_mac_count_is_the_same_under_every_stationarity() -> None:
    """D53's invariant, now written into the file each decomposition emits."""
    programs = [
        _emit(chip=_chip_declaring("a100_80gb", flow), stationarity=flow)[0] for flow in Dataflow
    ]
    assert {program.predicted["macs"] for program in programs} == {64 * 64 * 128}


def test_every_constant_carries_a_comment() -> None:
    """The user's own requirement: no constant without its provenance.

    Checked structurally rather than by eye — a new constant added without one
    fails here, which is the only way a rule like this survives.
    """
    program, *_ = _emit("a100_80gb")
    body = program.source[
        program.source.index("1. the shape") : program.source.index("5. the prediction")
    ]
    assignments = [line for line in body.splitlines() if re.match(r"^[A-Z][A-Z0-9_, ]* = ", line)]
    assert len(assignments) > 15
    for line in assignments:
        assert "#" in line, f"constant without a comment naming its source: {line!r}"


def test_the_filename_names_the_decomposition() -> None:
    assert default_filename("a100_80gb", DType.FP16, Dataflow.OUTPUT_STATIONARY, 1) == (
        "matmul-a100_80gb-fp16-os.py"
    )
    assert default_filename("a100_80gb", DType.FP16, Dataflow.OUTPUT_STATIONARY, 4) == (
        "matmul-a100_80gb-fp16-os-splitk4.py"
    )


# --------------------------------------------------------------------------- check


def test_check_catches_a_prediction_the_source_does_not_carry() -> None:
    """``check`` is the sibling of ``deploy.check``, and it has to bite.

    An emitter that wrote the wrong ``PREDICTED`` block would produce a program
    that passes while checking the wrong thing — the one failure mode this whole
    feature cannot tolerate.
    """
    program, *_ = _emit("a100_80gb")
    tampered = replace(program, predicted={**program.predicted, "macs": 1})
    with pytest.raises(ValueError, match=re.escape("PREDICTED['macs']")):
        check(tampered)


def test_check_catches_a_loop_nest_walking_a_different_grid() -> None:
    """``TILES`` and ``PREDICTED['tiles']`` are written by different code paths."""
    program, *_ = _emit("a100_80gb")
    broken = replace(program, source=program.source.replace("TILES = 16", "TILES = 15", 1))
    with pytest.raises(ValueError, match="the loop nest and the check disagree"):
        check(broken)


def test_check_needs_a_prediction_block() -> None:
    program, *_ = _emit("a100_80gb")
    stripped = replace(program, source=program.source.replace("PREDICTED = {", "OTHER = {", 1))
    with pytest.raises(ValueError, match="no PREDICTED dict literal"):
        check(stripped)


# -------------------------------------------------------------------- the harness


def test_the_inlined_harness_is_the_module_on_disk() -> None:
    """One copy, and it is the copy ruff, mypy and the unit tests see.

    The emitted file's runtime is ``inspect.getsource`` of ``_harness``, so a
    change there reaches every program without a second copy to keep in step.
    """
    from bwz.emit import _harness

    program, *_ = _emit("a100_80gb")
    assert "def run_cores(" in program.source
    assert _harness.run_cores.__doc__ is not None
    assert _harness.run_cores.__doc__.splitlines()[0] in program.source


def test_the_harness_imports_nothing_from_bwz() -> None:
    """An emitted program runs on a machine that has never heard of this repo."""
    import inspect

    from bwz.emit import _harness

    tree = ast.parse(inspect.getsource(_harness))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("bwz")
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("bwz") for alias in node.names)


def test_the_emitted_program_imports_nothing_but_the_standard_library() -> None:
    program, *_ = _emit("a100_80gb")
    tree = ast.parse(program.source)
    modules = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert modules <= {
        "__future__",
        "collections",
        "dataclasses",
        "importlib",
        "math",
        "sys",
        "threading",
        "types",
        "typing",
    }


# ------------------------------------------------------------------ accumulator


def test_an_integer_matmul_accumulates_in_int32() -> None:
    """D56, and the regression test for the bug that made it.

    The harness's own ``mma`` docstring says the widening is "that rule made
    executable" — but the width came from ``precision.accumulate``, a deployment
    field defaulting to ``fp32`` for every dtype, so the emitted int8 program
    accumulated in floating point and said so in a comment. The dyadic operands
    hid it: fp32 holds a sum of small integers exactly, so the numerics check
    passed while the demonstrated format was wrong.
    """
    program, _report, machine, *_ = _emit("metis_aipu", dtype="int8")
    constants = constants_of(program.source)
    assert machine.dtype is DType.INT8
    assert constants["ACC_DTYPE"] == "int32"
    assert "int32, always" in program.source
    assert "CUBLAS" not in program.source, "the float rationale must not follow an int8 run"


def test_a_float_matmul_accumulates_in_fp32_and_says_why() -> None:
    """fp32 for fp16 on a tensor core is right — it is what cuBLAS does by
    default — but the file must not claim fp16 accumulate is impossible, because
    it is a real MMA mode nothing here models (D56)."""
    program, *_ = _emit("a100_80gb")
    assert constants_of(program.source)["ACC_DTYPE"] == "fp32"
    assert "CUBLAS_COMPUTE_32F" in program.source
    assert "fp16 accumulate is a real mode" in program.source


def test_the_accumulator_can_still_be_overridden() -> None:
    """`precision.accumulate` is an override, not the source of truth (D56)."""
    program, *_ = _emit("a100_80gb", accumulate="fp16")
    assert constants_of(program.source)["ACC_DTYPE"] == "fp16"


def test_the_accumulator_follows_the_wider_operand() -> None:
    """Mixed widths run at the wider operand through one datapath (D18), and it
    is that datapath that accumulates — so the accumulator follows it, not A."""
    program, _report, machine, *_ = _emit("a100_80gb", dtype="int8")
    assert machine.dtype is DType.INT8
    assert constants_of(program.source)["ACC_DTYPE"] == "int32"


# ----------------------------------------------------- where a constant came from


def test_the_strategy_constants_name_their_provenance() -> None:
    """Section 3's values come from four different places, and say which (D57).

    The user's question was "is SPLIT_K = 1 hardwired, or calculated?" — which the
    old comment ("one kernel") could not answer, because it described the
    *consequence* of the value rather than its origin. Sections 1, 2 and 4 had
    always named an origin; section 3 was the one that did not.
    """
    program, *_ = _emit("a100_80gb")
    section = program.source[
        program.source.index("3. the strategy") : program.source.index("4. the grid")
    ]
    assert "--split-k, not passed" in section
    assert "NOT computed" in section, "nothing searches for a split factor (D53)"
    assert "OWN declared dataflow" in section, "stationarity is the chip's, not a choice"
    assert "DERIVED, and a yes/no" in section, "DEPTH is the one the engine works out"


def test_a_flag_that_was_passed_reads_differently_from_one_that_was_not() -> None:
    """A flag you passed and one you did not look identical in the value alone."""
    default, *_ = _emit("a100_80gb")
    asked, *_ = _emit("a100_80gb", split_k=4)
    assert "--split-k, not passed" in default.source
    assert "--split-k 4" in asked.source
    assert "Chosen by you, not searched for" in asked.source


def test_a_clamped_knob_says_what_was_asked_for() -> None:
    """A value that is not the one requested must not read as if it were (D36).

    `whole` that does not fit the scratchpad falls back to `stage` — same bytes,
    different timing — and the constant has to name the request, or a reader
    checking why their flag did nothing has no thread to pull.
    """
    program, *_ = _emit("metis_aipu", (8192, 8192, 8192), "int8", a_strategy="whole")
    assert constants_of(program.source)["A_STRATEGY"] == "stage"
    assert "--a-strategy whole did not fit" in program.source
    assert "clamped to stage" in program.source


def test_depth_is_a_yes_no_not_a_capacity_measurement() -> None:
    """`report.memory.double_buffered` is a bool, and DEPTH renders it (D58).

    The comment used to read "on-chip capacity fits two tiles", which invited
    exactly the right question on a 16-cubed matmul: A100's 60.7 MB against a
    512 B tile holds well over a hundred thousand of them. The model only ever
    asks whether it holds *at least* two, because two is what overlapping one
    load with one compute needs and there is no third state (D5a).
    """
    program, *_ = _emit("a100_80gb", (16, 16, 16))
    assert constants_of(program.source)["DEPTH"] == 2
    assert "yes/no dressed as a number" in program.source
    assert "There is no DEPTH 3" in program.source
    assert "capacity fits two tiles" not in program.source, "the wording that caused D58"
    # The headroom quoted must be checkable from constants already on the page —
    # that is the whole reason it is ON_CHIP_BYTES over a tile and not the
    # planner's own spare, which no constant here exposes.
    numbers = {
        name: value
        for name, value in constants_of(program.source).items()
        if isinstance(value, int | float)
    }
    tile = numbers["ROWS"] * numbers["COLS"] * numbers["A_BYTES_PER_ELEMENT"]
    assert f"~{int(numbers['ON_CHIP_BYTES'] / tile):,}" in program.source


def test_depth_one_says_the_loads_serialise() -> None:
    """The other branch, on a chip whose SRAM cannot hold two tiles.

    No shipped profile produces it at any shape, so the levels are shrunk
    directly — the same forging ``test_deploy`` used for this case before D54.
    """
    base = load_chip("a100_80gb")
    chip = base.model_copy(
        update={
            "memory": [
                level.model_copy(update={"capacity_bytes": 300})
                if level.name != base.dram.name
                else level
                for level in base.memory
            ]
        }
    )
    program, report, *_ = _emit(shape=(600, 600, 600), chip=chip)
    assert report.memory.double_buffered is False, "the shrink must actually flip this"
    assert constants_of(program.source)["DEPTH"] == 1
    assert "does NOT hold two" in program.source
    assert "their SUM" in program.source


def test_the_thread_warning_bar_comes_from_the_profiles(tmp_path: Path) -> None:
    """`LARGEST_DECLARED_CORES` is read off the profiles, not picked (D59).

    It replaced a hardcoded 2048, which was 3.9x above anything any profile
    declares and 60x below the host limit that would actually bite — so it fired
    for nothing real and meant nothing when it fired.
    """
    program, *_ = _emit("a100_80gb")
    count, where = largest_declared_array()
    assert constants_of(program.source)["LARGEST_DECLARED_CORES"] == count
    assert where in program.source, "the remark names which profile set the bar"
    assert "2048" not in program.source
    # And it is wired through, not merely declared.
    assert "warn_above=LARGEST_DECLARED_CORES" in program.source


# ------------------------------------------------------------------------ --debug


def test_debug_is_off_unless_asked_for() -> None:
    """The narration is guarded at every call site, not filtered inside `log` (D60).

    Volume is the whole reason it is optional: a 1000x2000x3000 matmul issues
    1.5 million instruction tiles, and building a line for each one that nobody
    reads would dominate the run. `if DEBUG:` costs a bool test.
    """
    program, *_ = _emit("a100_80gb")
    source = program.source
    assert 'DEBUG = "--debug" in sys.argv' in source

    # Every log() call sits inside an `if DEBUG:` block. Checked with ast rather
    # than by looking a few lines back, so a guard further up still counts and a
    # coincidental "if DEBUG:" in a comment does not.
    tree = ast.parse(source)
    guarded = {
        line
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "DEBUG"
        for line in range(node.lineno, (node.end_lineno or node.lineno) + 1)
    }
    calls = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "log"
    ]
    assert calls, "the narration exists at all"
    unguarded = [line for line in calls if line not in guarded]
    assert not unguarded, f"log() outside an `if DEBUG:` block at line(s) {unguarded}"


@pytest.mark.parametrize(
    ("chip_id", "dtype", "expected"),
    [
        ("a100_80gb", "fp16", "K is swept INSIDE"),
        ("metis_aipu", "int8", "resident in weight set {weight_set}, M streams past it"),
    ],
)
def test_debug_narrates_the_decomposition_it_is_of(chip_id: str, dtype: str, expected: str) -> None:
    """What the narration says differs with the stationarity, like the nest does.

    `os` reports an accumulator sweeping K; `ws` reports a resident B tile with M
    streaming past and a PARTIAL landing in the shared accumulator. A single
    generic "processing tile N" line would have been easier and would have taught
    nothing (D60).
    """
    program, *_ = _emit(chip_id, dtype=dtype)
    assert expected in program.source
    marker = "PARTIAL into C" if chip_id == "metis_aipu" else "useful of "
    assert marker in program.source


def test_debug_counts_the_lines_it_will_print_from_a_constant_on_the_page() -> None:
    """The heads-up figure is `mac_slots / SLOTS_PER_MMA` — exactly the mma count.

    Derived rather than re-counted, so it cannot disagree with the walk: every
    instruction tile issues SLOTS_PER_MMA slots, so the quotient IS the number of
    `mma()` calls, and therefore of narrated lines.
    """
    program, *_ = _emit("a100_80gb")
    assert 'issued = PREDICTED["mac_slots"] // SLOTS_PER_MMA' in program.source
    predicted_calls = program.predicted["mac_slots"] / (16 * 16 * 16)
    assert predicted_calls == 4 * 4 * 8, "4x4 grid, 8 k-steps at 64x64x128"


def test_the_program_counts_the_additions_the_report_charges() -> None:
    """A K-on-grid walk's reduction becomes a tier-1 count (D62).

    64x64x128 on a 16x16 array cuts K into 8 k-slices, so every one of the 4096
    output elements ends up with 8 partials and ``7 x 64 x 64 = 28,672``
    additions have to happen. ``os`` cuts nothing and predicts 0; split-K into 4
    predicts ``3 x 64 x 64 = 12,288``. Three decompositions, three counts, one
    matmul.
    """
    plain, *_ = _emit("a100_80gb")
    weight, *_ = _emit("a100_80gb", stationarity=Dataflow.WEIGHT_STATIONARY)
    inputs, *_ = _emit("a100_80gb", stationarity=Dataflow.INPUT_STATIONARY)
    split, *_ = _emit("a100_80gb", split_k=4)

    assert plain.predicted["partial_sum_adds"] == 0
    assert weight.predicted["partial_sum_adds"] == 7 * 64 * 64
    assert inputs.predicted["partial_sum_adds"] == 7 * 64 * 64
    assert split.predicted["partial_sum_adds"] == 3 * 64 * 64
    assert 'Check(\n            "partial-sum additions"' in weight.source
    assert "counters.count_partial_sum_adds" in split.source, "the second kernel counts too"


def test_a_ws_program_on_a_tensor_core_says_nothing_is_held() -> None:
    """Trap 9: "weight-stationary" is a misnomer on a unit with no weight banks.

    The tensor core reads every operand from the register file per instruction
    (D30), so what ``ws`` actually changed is the grid — its rows are slices of
    K. Metis really does hold B in its array, and keeps the original wording.
    """
    tensor_core, *_ = _emit("a100_80gb", stationarity=Dataflow.WEIGHT_STATIONARY)
    imc, *_ = _emit("metis_aipu", dtype="int8")

    assert "Nothing is held — K is on the grid." in tensor_core.source
    assert "misnomer here" in tensor_core.source
    assert "One weight tile. B stays resident" in imc.source
    assert "misnomer" not in imc.source


def test_the_emitted_file_quotes_the_placement_its_report_charged() -> None:
    """Same walk, same count, three different prices — and the file says which.

    Quoting one placement's price in a file emitted for another is exactly the
    kind of adjacent-to-true comment this repository keeps having to correct
    (D57/D58/D61), so the note is built from the ``OpResult``'s own placement.
    """
    on_chip, *_ = _emit("a100_80gb", stationarity=Dataflow.WEIGHT_STATIONARY)
    local, *_ = _emit("metis_aipu", shape=(64, 64, 1024), dtype="int8")

    assert "OVERLAPS them with" in on_chip.source
    assert "max(matrix, vector)" in on_chip.source
    assert "The report charges NOTHING for them" in local.source
    assert "OVERLAPS" not in local.source


def test_a_spilled_accumulator_is_shown_as_tier_2_not_asserted() -> None:
    """The one place the program and the report model different machines (D62).

    Squeeze A100's on-chip capacity to 2 kB per level and the 64x64 accumulator no longer
    fits, so the report charges the partials a DRAM round trip. The walk keeps
    them in ``partials`` regardless — where an accumulator lives is a capacity
    heuristic, not a step of the decomposition — so the row is shown side by
    side and left for a reader to judge, exactly as B's residency discount is.
    Asserting it would make the program fail for modelling a different claim.
    """
    base = load_chip("a100_80gb")
    cramped = base.model_copy(
        update={
            "memory": [
                level.model_copy(update={"capacity_bytes": 2048.0}) if level.level < 3 else level
                for level in base.memory
            ],
            "hypothetical": True,
        }
    )
    program, report, *_ = _emit(chip=cramped, stationarity=Dataflow.WEIGHT_STATIONARY)
    op = report.phases[0].ops[0]

    assert op.reduction_placement is ReductionPlacement.DRAM
    assert program.predicted["partial_dram_bytes"] > 0
    assert "tier 2, and the one place this file and the report model different" in program.source
    assert program.predicted["partial_sum_adds"] == 7 * 64 * 64, "the adds are still asserted"


def test_every_stage_the_trace_plays_has_a_line_in_the_program() -> None:
    """The animation highlights real lines of the real program (D54/D66).

    A K-on-grid walk's `partials.accumulate(...)` is the reduction: the report
    charges it to the vector unit and the trace draws it on the vector lane. It
    was tagged ``exec``, the tag its neighbouring ``mma`` carries, so the
    animation played a REDUCE bar with **no line lit under it** — the bars and
    the highlighting come from different places, and for this one stage they did
    not meet. Asserted here rather than noticed in a browser.
    """
    for flow in (Dataflow.WEIGHT_STATIONARY, Dataflow.INPUT_STATIONARY):
        program, *_ = _emit("a100_80gb", stationarity=flow)
        tags = {tag for tag, _ in program.stage_lines}
        assert "reduce" in tags, f"{flow.value}: the accumulate line is the reduction"
        assert "exec" in tags, f"{flow.value}: the mma line is still the arithmetic"

    # os cuts nothing, so it owes no reduction and must claim no reduce lines.
    plain, *_ = _emit("a100_80gb")
    assert "reduce" not in {tag for tag, _ in plain.stage_lines}
