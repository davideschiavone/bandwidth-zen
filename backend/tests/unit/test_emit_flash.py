"""The emitted FlashAttention program, checked statically — ``bwz/emit/flash.py``, D70.

Running it is ``tests/integration/test_emitted_flash_run.py``'s job; this checks
that the source says what the plan says, and that it could run anywhere.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import replace

import pytest

from bwz.analysis.flash import FlashShape, plan_flash
from bwz.emit.flash import FlashProgram, check, emit_flash
from bwz.spec import load_chip
from bwz.spec.dtypes import DType
from bwz.spec.hardware_spec import Dataflow

SHAPE = FlashShape(batch=1, heads=2, q_len=40, kv_len=40, head_dim=32, dtype=DType.FP16)


def _program(
    stationarity: Dataflow | None = None, *, br: int | None = None, bc: int | None = None
) -> FlashProgram:
    chip = load_chip("a100_80gb")
    plan = plan_flash(chip, SHAPE, stationarity=stationarity, br=br, bc=bc)
    return emit_flash(chip, plan, command="bwz attention ...", version="test")


def test_the_prediction_is_the_plan() -> None:
    check(_program())


@pytest.mark.parametrize(
    "flow",
    [Dataflow.OUTPUT_STATIONARY, Dataflow.WEIGHT_STATIONARY, Dataflow.INPUT_STATIONARY],
)
def test_every_inner_dataflow_emits_its_own_loop_nest(flow: Dataflow) -> None:
    source = _program(flow).source
    ast.parse(source)
    assert f"def matmul_{flow.value}(" in source
    assert f"matmul_{flow.value}(s, q, transpose(k)" in source
    others = {Dataflow.OUTPUT_STATIONARY, Dataflow.WEIGHT_STATIONARY, Dataflow.INPUT_STATIONARY}
    for other in others - {flow}:
        assert f"def matmul_{other.value}(" not in source


def test_the_online_softmax_is_written_out_in_the_walk() -> None:
    """The rescale is the whole trick, so it belongs where a reader looks, not in the
    harness: it has to be in run_program's own body."""
    source = _program().source
    body = source[source.index("def run_program(") : source.index("def walk(")]
    for step in ("row_max(s)", "exp_shifted(s, m_new", "scale_rows(o, alpha)", "divide_rows(o, l)"):
        assert step in body


def test_the_filename_names_the_plan() -> None:
    assert _program(br=16, bc=16).filename == "flash-a100_80gb-fp16-br16-bc16-os-os.py"


def test_check_catches_a_prediction_the_source_does_not_carry() -> None:
    program = _program()
    tampered = replace(program, predicted={**program.predicted, "scores": 1})
    with pytest.raises(ValueError, match="scores"):
        check(tampered)


def test_the_emitted_program_imports_nothing_but_the_standard_library() -> None:
    tree = ast.parse(_program().source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names = [node.module.split(".")[0]]
        else:
            continue
        for name in names:
            assert name == "__future__" or name in sys.stdlib_module_names, name


def test_an_infeasible_plan_has_nothing_to_emit() -> None:
    chip = load_chip("chip_a")
    shape = FlashShape(batch=1, heads=1, q_len=512, kv_len=512, head_dim=64, dtype=DType.INT8)
    with pytest.raises(ValueError, match="no decomposition to emit"):
        emit_flash(chip, plan_flash(chip, shape), command="x", version="x")
