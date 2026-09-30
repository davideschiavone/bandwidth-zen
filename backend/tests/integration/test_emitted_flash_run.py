"""Run the emitted FlashAttention programs. Their own assertions are the test (D70).

``tests/unit/test_emit_flash.py`` checks that the source says what the plan says;
this checks the half that cannot be checked statically — that the plan, walked,
moves the bytes and does the vector work the formula charged, and that the
online softmax computes ``softmax(Q Kᵀ / √d) V``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from bwz.analysis.flash import FlashShape, plan_flash
from bwz.emit.flash import check, emit_flash
from bwz.spec import load_chip
from bwz.spec.dtypes import DType
from bwz.spec.hardware_spec import Dataflow

TIMEOUT_S = 600


def _write(
    tmp_path: Path,
    chip_id: str,
    shape: FlashShape,
    *,
    br: int | None = None,
    bc: int | None = None,
    stationarity: Dataflow | None = None,
) -> Path:
    chip = load_chip(chip_id)
    plan = plan_flash(chip, shape, br=br, bc=bc, stationarity=stationarity)
    program = emit_flash(chip, plan, command="bwz attention ...", version="test")
    check(program)
    path = tmp_path / program.filename
    path.write_text(program.source, encoding="utf-8")
    return path


def _run(path: Path, env: dict[str, str] | None = None) -> str:
    finished = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        env=env,
        check=False,
    )
    assert finished.returncode == 0, (
        f"{path.name} exited {finished.returncode}. A mismatch means the emitter, the harness "
        f"or the cost model is wrong — find out which before weakening anything.\n"
        f"{finished.stdout}\n{finished.stderr}"
    )
    return finished.stdout


A100_SHAPE = FlashShape(batch=1, heads=2, q_len=50, kv_len=50, head_dim=32, dtype=DType.FP16)


@pytest.mark.parametrize(
    "flow",
    [Dataflow.OUTPUT_STATIONARY, Dataflow.WEIGHT_STATIONARY, Dataflow.INPUT_STATIONARY],
)
def test_every_inner_dataflow_runs_and_checks_out(tmp_path: Path, flow: Dataflow) -> None:
    """Ragged blocks on both axes (50 = 3 x 16 + 2) and four kv blocks each, so the
    rescale runs; under ws and is, every k-slice beyond the first is a vector add."""
    output = _run(_write(tmp_path, "a100_80gb", A100_SHAPE, br=16, bc=16, stationarity=flow))
    assert "every count matches the plan" in output


def test_the_formulas_own_choice_runs(tmp_path: Path) -> None:
    output = _run(_write(tmp_path, "a100_80gb", A100_SHAPE))
    assert "every count matches the plan" in output


def test_metis_rereads_k_and_v_for_a_head_that_spans_waves(tmp_path: Path) -> None:
    """9 programs on 4 cores in 3 waves: two of the three heads straddle a wave
    boundary, and the walk fetches their K and V once per wave, as charged."""
    shape = FlashShape(batch=1, heads=3, q_len=1100, kv_len=1100, head_dim=64, dtype=DType.INT8)
    output = _run(_write(tmp_path, "metis_aipu", shape, br=512, bc=512))
    assert "every count matches the plan" in output


def test_it_runs_without_numpy(tmp_path: Path) -> None:
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "numpy.py").write_text('raise ImportError("shadowed by the test")\n')
    shape = FlashShape(batch=1, heads=1, q_len=40, kv_len=40, head_dim=16, dtype=DType.FP16)
    path = _write(
        tmp_path, "a100_80gb", shape, br=16, bc=16, stationarity=Dataflow.WEIGHT_STATIONARY
    )
    output = _run(path, env={"PYTHONPATH": str(shadow), "PATH": "/usr/bin:/bin"})
    assert "backend: pure Python" in output
    assert "every count matches the plan" in output
