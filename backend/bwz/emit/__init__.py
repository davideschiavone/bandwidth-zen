"""Runnable kernel programs: the decomposition as code rather than as prose.

``docs/CORRECTIONS.md`` D54. :mod:`bwz.deploy` writes a pseudo-C listing and
``deploy.check`` asserts its *constants* against the schedule, which is as far as
a listing can go: it cannot be run, so nothing verifies that the decomposition it
describes computes a matmul, and nothing verifies that the bytes the roofline
charged are the bytes such a schedule would really move.

:func:`~bwz.emit.matmul.emit_matmul` writes a self-contained Python program that
does both. :func:`check` here is the cheap half of the guarantee — the sibling of
``deploy.check``: it parses the emitted source and asserts that the ``PREDICTED``
block in it is the report's own numbers, without executing anything, so every
unit test can afford it. The expensive half is running the program, which asserts
its measurements against that same block; that lives in
``tests/integration/test_emitted_programs_run.py``.
"""

from __future__ import annotations

import ast

from bwz.emit.matmul import (
    EmittedProgram,
    default_filename,
    emit_matmul,
    predicted_for,
)

__all__ = [
    "EmittedProgram",
    "check",
    "constants_of",
    "default_filename",
    "emit_matmul",
    "predicted_for",
]


def constants_of(source: str) -> dict[str, object]:
    """Every top-level literal assignment in an emitted program.

    ``ast`` rather than importing the module: reading the file must never run it,
    and an emitted program's ``main()`` allocates operands and starts threads.
    Assignments whose value is not a literal (``COUNTERS = Counters()``) are
    skipped rather than guessed at.
    """
    found: dict[str, object] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except ValueError:
            continue
        targets = [target for target in node.targets if isinstance(target, ast.Name)]
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
            names = [n.id for n in node.targets[0].elts if isinstance(n, ast.Name)]
            if isinstance(value, tuple) and len(names) == len(value):
                found.update(dict(zip(names, value, strict=True)))
            continue
        for target in targets:
            found[target.id] = value
    return found


def check(program: EmittedProgram) -> None:
    """Assert the emitted source says what the program object says it says.

    The guarantee :mod:`bwz.explain` and :mod:`bwz.deploy` both give: a listing
    that disagreed with the numbers beside it would be worse than no listing,
    because a reader would believe it. Here the stakes are higher still, since
    the file asserts against its own ``PREDICTED`` block at runtime — an emitter
    that wrote the wrong block would produce a program that passes while checking
    the wrong thing.

    Raises ``ValueError`` naming the field, the emitted value and the expected
    one (CLAUDE.md #8).
    """
    constants = constants_of(program.source)
    predicted = constants.get("PREDICTED")
    if not isinstance(predicted, dict):
        raise ValueError(
            f"{program.filename}: no PREDICTED dict literal at module level; the emitted "
            f"program would have nothing to check itself against."
        )
    for key, expected in program.predicted.items():
        actual = predicted.get(key)
        if actual is None:
            raise ValueError(
                f"{program.filename}: PREDICTED is missing {key!r}, which the report supplies "
                f"as {expected!r}."
            )
        if float(actual) != float(expected):
            raise ValueError(
                f"{program.filename}: PREDICTED[{key!r}] is {actual!r}, but the report says "
                f"{expected!r}."
            )
    # The grid constants the loop nest walks and the prediction it checks itself
    # against are written by two different code paths; a mismatch between them
    # would let a program walk one decomposition while asserting another's counts.
    for name, key in (("TILES", "tiles"), ("WAVES", "waves"), ("USED_CORES", "used_cores")):
        emitted = _as_number(constants.get(name))
        if emitted != float(program.predicted[key]):
            raise ValueError(
                f"{program.filename}: constant {name} is {constants.get(name)!r} but "
                f"PREDICTED[{key!r}] is {program.predicted[key]!r}; the loop nest and the "
                f"check disagree."
            )
    if _as_number(constants.get("AVAILABLE_CORES")) < float(program.predicted["used_cores"]):
        raise ValueError(
            f"{program.filename}: USED_CORES exceeds AVAILABLE_CORES; the chip cannot run more "
            f"tiles at once than it has cores."
        )


def _as_number(value: object) -> float:
    """A parsed literal as a float, or NaN when it is not a number at all.

    NaN rather than an exception so the caller's comparison fails and reports the
    offending literal, which is more useful than a TypeError from inside here.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return float("nan")
    return float(value)
