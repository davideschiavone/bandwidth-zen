"""How the model deploys a *graph* on one chip, written as a loop over operations.

``docs/CORRECTIONS.md`` D32, narrowed by D54. Sibling of :mod:`bwz.explain`, and
the same contract: **pure**, holds no counts of its own, and every constant it
prints is read back out of the schedule the figure draws. ``check()`` asserts
that, so the listing cannot drift from the timeline above it the way prose does.

The difference between the two is what they answer. ``explain`` answers *what
arithmetic is performed* — operand shapes, the algebra, the flop count — and is
the same on every chip. This answers *how that work reaches this particular
silicon*.

**What used to be here.** This module also printed a pseudo-C tile nest for a
lone matmul: how B was cut into array-sized tiles, how many arrays took a wave of
them at once, where the loads and stores sat around it. That listing has been
retired in favour of :mod:`bwz.emit`, which writes the same decomposition as a
program that **runs** — one that computes ``A @ B``, counts what it moves and
asserts those counts against the report (D54). A description of a loop nest and
the loop nest itself are not two useful artifacts; the pseudo-C was the weaker
one, and it could not be checked against anything but its own constants.

What is left is the case ``emit`` has no answer for: a **graph**, which this
model runs as a sequence with no overlap between operations (D5a), so there is no
tile grid to walk and nothing to emit. A matmul on a chip whose fastest unit for
the requested dtype declares no array geometry lands here too, and the listing
says so rather than implying a grid that does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass

from bwz.analysis.pipeline import Lane, PipelineTrace
from bwz.analysis.roofline import MachineModel
from bwz.graph.ops import Operation
from bwz.report import PhaseResult
from bwz.spec.hardware_spec import HardwareSpec
from bwz.units import format_bytes, format_time


@dataclass(frozen=True, slots=True)
class Deployment:
    """One chip's listing, plus the numbers it was built from.

    The numbers are kept alongside the text so :func:`check` can compare them
    against the trace without re-parsing the listing.
    """

    chip_id: str
    title: str
    code: str
    tiles: int
    waves: int
    units: int
    resident_tiles: int
    reloads: int
    kind: str = "operations"
    """``"operations"`` for a graph — the only kind this module still produces.
    A tiled matmul is emitted as a runnable program instead (D54), so the wave
    relation ``waves = ceil(tiles / units)`` no longer has a listing to hold for:
    a network is a sequence of operations with no wave structure to check (D5a)."""
    stage_lines: tuple[tuple[str, tuple[int, ...]], ...] = ()
    """0-indexed line numbers within ``code`` for each animated stage
    (``"load_b"``, ``"load_a"``, ``"exec_core"``, ``"exec_vector"``, ``"store"``
    — the vocabulary ``bwz.figures``'s ``_ANIMATION_STAGE`` already uses).
    A tuple of pairs rather than a ``dict``, matching ``PipelineTrace.work_by_op``'s
    own reason: keep a frozen dataclass hashable-in-substance. Lets a debug view
    highlight the line(s) live at the animation's current time (D41)."""


def _int(value: float) -> str:
    return f"{round(value):,}"


def deployment_of(
    chip: HardwareSpec,
    machine: MachineModel,
    phase: PhaseResult,
    trace: PipelineTrace,
    *,
    workload: str,
    operation: Operation | None = None,
) -> Deployment:
    """Build the listing for *chip* running *workload*.

    *operation* is accepted and unused: it identified the single matmul the
    retired tile branch described, and the signature is kept so callers that
    hand it over need not know the branch is gone (D54).

    The model runs a graph as a sequence with no overlap between operations
    (D5a), so the listing is a loop over operations rather than a tile nest, and
    it says so.
    """
    del operation
    unit = machine.unit
    steps = max(trace.steps, 1)
    tiled_but_untileable = trace.kind == "tiles" and unit.systolic_dims is None
    header_lines = [
        f"/* {chip.name} — how this model deploys the run",
        f" * {workload}",
        " *",
    ]
    if tiled_but_untileable:
        header_lines += [
            f" * {unit.name} declares no array geometry, so there is no tile grid to walk",
            " * and no runnable program to emit for this run (D54). No tail-effect claim is",
            " * made either — inventing one would be worse than omitting it.",
        ]
    else:
        header_lines += [
            " * A network is a SEQUENCE here: no overlap is modelled between one operation's",
            " * prefetch and the previous operation's arithmetic (D5a). Within an operation,",
            " * load and compute overlap when capacity granted a second buffer.",
        ]
    header_lines += [
        " *",
        f" * {trace.tiles} operations, drawn as {steps} step{'s' if steps != 1 else ''}.",
        " */",
    ]
    defines = [
        f"#define OPS   {trace.tiles:<10} /* graph nodes in this phase */",
        f"#define UNITS {unit.count:<10} /* {unit.name} */",
    ]
    # A generic, un-unrolled loop: real operation names (Span.label, e.g.
    # "q_proj") live in the trace and the animation's blocks/hover text, not
    # here — one static "load_B(op[i])" line stands for every operation's load,
    # so tagging it "load_b" highlights it whenever *any* op is loading, not a
    # specific one. The two compute branches get distinct tags (exec_core,
    # exec_vector, D43) so the debug view can glow the matrix and vector
    # stations independently — which one runs depends on op[i], which this
    # text never names, but *which engine* is a real, known distinction the
    # trace already carries per span (D42's original single "exec" tag
    # couldn't make that distinction; splitting it is what let per-engine
    # animation stations exist at all).
    imc_write_line: tuple[str, str | None] = (
        "    imc_write(op[i]);                /* the array's weight set is written before "
        "this op's arithmetic; no per-op write-ahead/on-demand/persistent placement is "
        "modelled for a network trace (D43) */",
        None,
    )
    body: list[tuple[str, str | None]] = [
        ("for (int i = 0; i < OPS; ++i) {", None),
        ("    dispatch(op[i]);                  /* hatched bar, if the op costs one */", None),
        ("", None),
        ("    load_B(op[i]);                    /* weights   — solid bar   */", "load_b"),
        *([imc_write_line] if unit.weight_sets > 1 else []),
        ("    load_A(op[i]);                    /* activations — hatched bar */", "load_a"),
        ("", None),
        ("    if (is_matrix(op[i]))", None),
        ("        parallel_for (int u = 0; u < UNITS; ++u)", None),
        (f"            {unit.name}(u, op[i]);", "exec_core"),
        ("    else", None),
        (
            f"        {machine.vector_unit.name}(op[i]);   /* norms, activations (D27) */",
            "exec_vector",
        ),
        ("", None),
        (
            "    store_C(op[i]);                   /* whatever no later op reads — hollow bar */",
            "store",
        ),
        ("}", None),
    ]
    footer_line = (
        f"/* span {format_time(trace.total_s)}; DRAM {format_bytes(trace.totals[Lane.DRAM])}, "
        f"arithmetic {_int(trace.totals[Lane.CORE] + trace.totals[Lane.VECTOR])} OP */"
    )

    combined: list[tuple[str, str | None]] = [(h, None) for h in header_lines]
    combined.append(("", None))
    combined.extend((d, None) for d in defines)
    combined.append(("", None))
    combined.extend(body)
    combined.append(("", None))
    combined.append((footer_line, None))

    code = "\n".join(text for text, _tag in combined)
    stage_line_map: dict[str, list[int]] = {}
    for index, (_text, tag) in enumerate(combined):
        if tag is not None:
            stage_line_map.setdefault(tag, []).append(index)

    return Deployment(
        chip_id=chip.id,
        title=f"{chip.name} — {unit.name}",
        code=code,
        tiles=trace.tiles,
        waves=trace.tiles,
        units=max(unit.count, 1),
        resident_tiles=unit.resident_tile_capacity(),
        reloads=0,
        stage_lines=tuple((tag, tuple(indices)) for tag, indices in stage_line_map.items()),
    )


def check(deployment: Deployment, trace: PipelineTrace) -> None:
    """Assert the listing's constants are the schedule's.

    The same guarantee :mod:`bwz.explain` gives the arithmetic section: a
    deployment listing that disagreed with the timeline above it would be worse
    than no listing, because a reader would believe it.
    """
    if deployment.waves != trace.tiles:
        raise ValueError(
            f"{deployment.chip_id}: listing says {deployment.waves} waves, schedule has "
            f"{trace.tiles}"
        )
    if deployment.tiles < deployment.waves:
        raise ValueError(
            f"{deployment.chip_id}: {deployment.tiles} tiles cannot fill {deployment.waves} waves"
        )
