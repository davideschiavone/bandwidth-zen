"""Run what was emitted. The programs' own assertions are the test.

``docs/CORRECTIONS.md`` D54. ``tests/unit/test_emit.py`` checks that the emitted
source says what the report says; this checks the half that cannot be checked
statically — that the decomposition, walked, moves the bytes the roofline charged
and computes ``A @ B``. Each program asserts that against the ``PREDICTED`` block
in its own source and exits non-zero if any tier-1 count disagrees, so all this
file has to do is run them and read the exit code.

Tiny shapes throughout: an emitted program must run without numpy (it is not a
``bwz`` dependency), and the pure-Python fallback walks the arithmetic element by
element.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from bwz.analysis import analyze, machine_model
from bwz.analysis.dataflow import plan_dataflow
from bwz.analysis.pipeline import grid_of
from bwz.emit import check, emit_matmul
from bwz.graph import GraphPhase, build_graph
from bwz.operators.base import cost_of
from bwz.spec import DeploymentSpec, MatmulSpec, load_chip
from bwz.spec.hardware_spec import Dataflow, HardwareSpec

TIMEOUT_S = 300
"""Generous: the pure-Python fallback is an interpreted triple loop, and the file
says in its own docstring that it makes no attempt to be fast."""


def _write(
    tmp_path: Path,
    *,
    chip_id: str,
    shape: tuple[int, int, int],
    dtype: str,
    split_k: int = 1,
    a_strategy: str = "stage",
    stationarity: Dataflow | None = None,
    chip: HardwareSpec | None = None,
) -> Path:
    """Emit one program to disk, the way ``bwz matmul --emit`` does."""
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
        }
    )
    chip = chip if chip is not None else load_chip(chip_id)
    report = analyze(spec, chip, deployment)
    assert report.feasible, report.infeasibility
    machine = machine_model(
        chip, spec.operand_dtype, stationarity=stationarity, k_partitions=split_k
    )
    graph = build_graph(spec, deployment, GraphPhase.STATIC)
    op = graph.ops[0]
    grid = grid_of(op, machine)
    assert grid is not None
    program = emit_matmul(
        chip,
        machine,
        grid,
        plan_dataflow(
            op, machine, chip, deployment, a_bytes=cost_of(op, graph.tensors).input_bytes
        ),
        report.phases[0].ops[0],
        a_dtype=spec.a_dtype,
        b_dtype=spec.b_dtype,
        c_dtype=spec.result_dtype,
        acc_dtype=deployment.precision.accumulate,
        double_buffered=report.memory.double_buffered,
        command="bwz matmul (test)",
        version="0.0.0-test",
    )
    check(program)
    path = tmp_path / program.filename
    path.write_text(program.source, encoding="utf-8")
    return path


def _run(path: Path, env: dict[str, str] | None = None, extra_args: list[str] | None = None) -> str:
    """Run an emitted program and return its stdout, failing on a non-zero exit."""
    finished = subprocess.run(
        [sys.executable, str(path), *(extra_args or [])],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        env=env,
        check=False,
    )
    assert finished.returncode == 0, (
        f"{path.name} exited {finished.returncode}. A tier-1 mismatch means the emitter, the "
        f"harness or the cost model is wrong — find out which before weakening anything.\n"
        f"{finished.stdout}\n{finished.stderr}"
    )
    return finished.stdout


@pytest.mark.parametrize(
    ("chip_id", "dtype", "split_k", "a_strategy"),
    [
        ("a100_80gb", "fp16", 1, "stage"),
        ("a100_80gb", "fp16", 4, "stage"),
        ("a100_80gb", "fp16", 1, "stream"),
        ("a100_80gb", "fp16", 1, "whole"),
        ("metis_aipu", "int8", 1, "stage"),
        ("metis_aipu", "int8", 1, "stream"),
    ],
)
def test_every_shipped_decomposition_runs_and_checks_out(
    tmp_path: Path, chip_id: str, dtype: str, split_k: int, a_strategy: str
) -> None:
    """Emit, run, exit 0 — every strategy the shipped chips can be asked for."""
    path = _write(
        tmp_path,
        chip_id=chip_id,
        shape=(64, 64, 128),
        dtype=dtype,
        split_k=split_k,
        a_strategy=a_strategy,
    )
    output = _run(path)
    assert "every tier-1 count matches the report" in output


def test_a_ragged_shape_runs(tmp_path: Path) -> None:
    """Nothing divides the array here, so every clip in the walk is exercised.

    M=100 on a 16-row array is six full bands and a four-row tail; N=70 and K=50
    are ragged too. The useful MAC count must still be exactly ``M*N*K`` — a walk
    that computed the padding would report more, and one that dropped the tail
    would report less (D52/D53).
    """
    path = _write(tmp_path, chip_id="a100_80gb", shape=(100, 70, 50), dtype="fp16")
    output = _run(path)
    assert "MACs" in output
    assert f"{100 * 70 * 50:,}" in output
    assert "every tier-1 count matches the report" in output


def test_output_and_weight_stationary_compute_the_same_c(tmp_path: Path) -> None:
    """Two decompositions, one matmul, one answer.

    The invariant D53 rests on, executable: ``os`` and ``ws`` disagree about how
    many tiles there are, which dimension each sweeps and whether partials are
    owed — never about the result. Each program already asserts ``C == A @ B``
    against its own reference; comparing the checksums says the two references
    were the same matmul.
    """
    shape = (64, 64, 128)
    elsewhere = tmp_path / "ws"
    elsewhere.mkdir()
    # Both halves run on the shipped A100 profile: since D62 its tensor core
    # declares all three grids, precisely so that the two can be compared on one
    # chip. `os` is still its native dataflow and still the default.
    output_stationary = _run(_write(tmp_path, chip_id="a100_80gb", shape=shape, dtype="fp16"))
    weight_stationary = _run(
        _write(
            elsewhere,
            chip_id="a100_80gb",
            shape=shape,
            dtype="fp16",
            stationarity=Dataflow.WEIGHT_STATIONARY,
        )
    )

    def checksum(text: str) -> str:
        (line,) = [line for line in text.splitlines() if line.startswith("C checksum:")]
        return line

    assert checksum(output_stationary) == checksum(weight_stationary)


def test_it_runs_without_numpy(tmp_path: Path) -> None:
    """numpy is optional, so the fallback has to be real and not decorative.

    It is shadowed by an unimportable module rather than uninstalled: the point
    is to prove the emitted file survives a machine without it, and ``make test``
    must not depend on which optional dependency groups were installed.
    """
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "numpy.py").write_text('raise ImportError("shadowed by the test")\n')
    path = _write(tmp_path, chip_id="a100_80gb", shape=(32, 32, 64), dtype="fp16")
    output = _run(path, env={"PYTHONPATH": str(shadow), "PATH": "/usr/bin:/bin"})
    assert "backend: pure Python" in output
    assert "every tier-1 count matches the report" in output


def test_debug_blocks_are_atomic_under_real_concurrency(tmp_path: Path) -> None:
    """Each core's narration prints as one unit, however many cores run (D61).

    The narration is indented — a tile under a core, a k-step under a tile — and
    that nesting is a *claim* about which line belongs to which. Serialising
    individual lines is not enough: with four cores it put ``kt=`` lines under a
    header belonging to a different core, so the indentation said something
    false. This runs 64 cores at once and checks every block is whole.

    A subprocess, and a shape with many tiles, because the failure only appears
    when threads genuinely interleave — an in-process single-core run would pass
    against the broken version.
    """
    path = _write(tmp_path, chip_id="a100_80gb", shape=(128, 128, 32), dtype="fp16")
    output = _run(path, extra_args=["--debug"])

    blocks: list[list[str]] = []
    for line in output.splitlines():
        if line.startswith("core "):
            blocks.append([line])
        elif blocks and line.startswith(" ") and ("kt=" in line or "C[" in line or "stage" in line):
            blocks[-1].append(line)
    assert len(blocks) == 64, f"8x8 grid -> 64 tiles, got {len(blocks)} blocks"

    for block in blocks:
        header, body = block[0], block[1:]
        geometry = [line for line in body if line.strip().startswith("C[")]
        steps = [line for line in body if "kt=" in line]
        assert len(geometry) == 1, f"one tile per block, got {len(geometry)}: {header}"
        assert len(steps) == 2, f"ceil(32/16) = 2 k-steps per tile, got {len(steps)}: {header}"
        # And the k-steps belong to *this* block's tile, not a neighbour's.
        rows = geometry[0].split("C[")[1].split(",")[0]
        assert all(f"A[{rows}," in step for step in steps), (
            "a k-step under the wrong tile header:\n" + "\n".join(block)
        )


def test_debug_changes_no_count(tmp_path: Path) -> None:
    """Narrating the walk must not alter it — the flag is observation only."""
    path = _write(tmp_path, chip_id="a100_80gb", shape=(64, 64, 128), dtype="fp16")
    quiet = _run(path)
    noisy = _run(path, extra_args=["--debug"])

    def table(text: str) -> list[str]:
        return [line for line in text.splitlines() if "OK" in line or "differs" in line]

    assert table(quiet) == table(noisy)
    assert "every tier-1 count matches the report" in noisy


@pytest.mark.parametrize(
    "flow", [Dataflow.WEIGHT_STATIONARY, Dataflow.INPUT_STATIONARY], ids=lambda d: d.value
)
def test_a_k_on_grid_walk_counts_the_additions_it_owes(tmp_path: Path, flow: Dataflow) -> None:
    """The reduction, executed and counted against the report (D62).

    64x64x128 on A100's 16x16 array cuts K into 8 k-slices, so each of the 4096
    output elements collects 8 partials and ``7 x 64 x 64 = 28,672`` additions
    happen in the shared accumulator. The report charges exactly that many to
    the CUDA cores; the program counts them and the check is tier 1, so a
    disagreement fails the run rather than being argued about.
    """
    path = _write(
        tmp_path, chip_id="a100_80gb", shape=(64, 64, 128), dtype="fp16", stationarity=flow
    )
    output = _run(path)
    assert "partial-sum additions" in output
    assert f"{7 * 64 * 64:,}" in output
    assert "every tier-1 count matches the report" in output
