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

import pytest

from bwz.analysis import analyze, machine_model
from bwz.analysis.dataflow import plan_dataflow
from bwz.analysis.pipeline import grid_of
from bwz.emit import check, constants_of, default_filename, emit_matmul, predicted_for
from bwz.graph import GraphPhase, build_graph
from bwz.operators.base import cost_of
from bwz.spec import DeploymentSpec, DType, MatmulSpec, load_chip
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
    assert "def reduce_partials" not in plain.source
    assert "def reduce_partials" in split.source
    assert "dram.write_partial(part, m0, n0, acc)" in split.source
    assert split.predicted["partial_dram_bytes"] > 0
    assert plain.predicted["partial_dram_bytes"] == 0


def test_a_strategies_differ_only_in_the_residency_constant() -> None:
    """``stream`` is ``stage`` with the residency set to one tile (D31/D33).

    Both call the same ``stage_a``; what changes is how many tiles one event
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
    assert "for tile in range(TILES):\n        stage_a(tile, dram, pad)" in whole.source
    assert "stage_a(tile, dram, pad)\n" in staged.source


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
    assert "def run_waves(" in program.source
    assert _harness.run_waves.__doc__ is not None
    assert _harness.run_waves.__doc__.splitlines()[0] in program.source


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
