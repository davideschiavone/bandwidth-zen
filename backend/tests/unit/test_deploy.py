"""The deployment listing is derived from the schedule, not written alongside it.

Same contract as ``test_explain``: a listing that disagreed with the timeline
above it would be believed, so every constant it prints is checked against the
trace the figure drew (docs/CORRECTIONS.md D32).

**What is no longer here.** This module used to test a pseudo-C *tile nest* for a
lone matmul — its wave counts, its weight-set writes, its bit-serial tax, its
prefetch shape. That listing was retired in D54: the same decomposition is now
emitted as a program that runs, computes ``A @ B`` and asserts its own counts
against the report, and the tests that matter for it live in
``tests/unit/test_emit.py`` and ``tests/integration/test_emitted_programs_run.py``.
Every fact those old tests pinned is still pinned — by a program that would fail
to run if it were wrong, rather than by a substring search over prose.

What remains is the case ``emit`` has no answer for: a graph, which this model
runs as a sequence with no overlap between operations (D5a).
"""

from __future__ import annotations

from bwz.analysis import analyze, idealised, machine_model
from bwz.analysis.pipeline import build_trace
from bwz.deploy import check, deployment_of
from bwz.graph import GraphPhase, build_graph
from bwz.spec import DeploymentSpec, DType, MatmulSpec, load_chip


def test_a_network_gets_a_sequence_listing_rather_than_a_tile_nest() -> None:
    """A graph of operations has no tile nest; the model runs it as a sequence
    with no cross-operation overlap (D5a), and the listing says exactly that."""
    from bwz.spec import load_model

    chip = idealised(load_chip("a100_80gb"))
    model = load_model("gemma3_4b")
    deployment = DeploymentSpec.model_validate(
        {
            "batch": 1,
            "input_tokens": 128,
            "output_tokens": 1,
            "precision": {"weights": "int8", "activations": "int8", "kv_cache": "int8"},
        }
    )
    report = analyze(model, chip, deployment)
    phase = report.phases[0]
    machine = machine_model(chip, DType.INT8)
    trace = build_trace(
        build_graph(model, deployment, phase.phase),
        phase,
        machine,
        double_buffered=report.memory.double_buffered,
    )
    listing = deployment_of(chip, machine, phase, trace, workload="gemma3_4b")

    check(listing, trace)
    assert "#define OPS" in listing.code
    assert "SEQUENCE" in listing.code
    assert "TILES" not in listing.code
    assert listing.kind == "operations"


def _network_listing(chip_id: str, dtype: DType, precision: dict[str, str] | None = None):  # type: ignore[no-untyped-def]
    from bwz.kernels import encoder_layer_kernel

    spec = encoder_layer_kernel(dmodel=8, nheads=2, ffn=16, vocab=16, tokens=4)
    chip = idealised(load_chip(chip_id))
    payload: dict[str, object] = {
        "batch": 1,
        "input_tokens": 4,
        "output_tokens": 0,
        "phase": "prefill",
    }
    if precision is not None:
        payload["precision"] = precision
    deployment = DeploymentSpec.model_validate(payload)
    report = analyze(spec, chip, deployment)
    assert report.feasible, report.infeasibility
    phase = report.phases[0]
    machine = machine_model(chip, dtype)
    trace = build_trace(
        build_graph(spec, deployment, phase.phase),
        phase,
        machine,
        double_buffered=report.memory.double_buffered,
    )
    listing = deployment_of(chip, machine, phase, trace, workload="encoder")
    check(listing, trace)
    return listing, trace


def test_network_listing_tags_stage_lines_for_the_debug_view() -> None:
    """D42/D43: `--animate`'s code pane needs to know which line is which stage
    for a network graph too, not just a lone matmul (D41). The loop is generic —
    `for (i = 0; i < OPS; ++i) { load_B(op[i]); ...; store_C(op[i]); }` — never
    unrolled per named operation (real names like "q_proj" live in the trace's
    spans, not this text), so a tag can only ever mean "a load/store/compute is
    happening right now", not "*this* operation's".

    The two compute branches (`if (is_matrix(op[i])) ... else ...`, D27's
    matrix/vector split) tag *distinct* stages, `exec_core`/`exec_vector`
    (D43) — which one runs for a given `op[i]` is still something this generic
    loop never names, but *which engine* is a real distinction the animation
    can already tell from a span's lane, so each branch gets its own tag
    rather than sharing one (D42's original choice, superseded once per-engine
    animation stations existed to make use of the split).
    """
    listing, _trace = _network_listing("a100_80gb", DType.FP16)

    assert listing.kind == "operations"
    lines = listing.code.split("\n")
    by_stage = dict(listing.stage_lines)
    assert set(by_stage) == {"load_b", "load_a", "exec_core", "exec_vector", "store"}
    assert all(len(idxs) == 1 for idxs in by_stage.values())
    assert "load_B(op[i]);" in lines[by_stage["load_b"][0]]
    assert "load_A(op[i]);" in lines[by_stage["load_a"][0]]
    assert "store_C(op[i]);" in lines[by_stage["store"][0]]
    assert "op[i]);" in lines[by_stage["exec_core"][0]]
    assert "op[i]);" in lines[by_stage["exec_vector"][0]]
    assert "imc_write" not in listing.code, "a100 tensor cores hold no resident weight set (D30)"


def test_network_listing_writes_the_weight_set_when_the_array_has_one() -> None:
    """D43: the tiled matmul path has always shown `imc_write` (untimed, D40)
    for a `weight_sets > 1` array; the network path never mentioned it at all,
    even though the same D-IMC array needs its weight set written before *any*
    operation's arithmetic, not just a lone matmul's. Unlike the tiled path,
    a network trace carries no `b_dataflow` (`_network_deployment` never
    receives one), so the added line makes no write-ahead/on-demand/persistent
    placement claim — just that the write happens, honestly scoped to what a
    generic, un-unrolled loop can actually say.
    """
    listing, _trace = _network_listing(
        "metis_aipu", DType.INT8, {"weights": "int8", "activations": "int8", "kv_cache": "int8"}
    )

    assert "imc_write(op[i]);" in listing.code
    # Untimed, like the tiled path's imc_write lines: no stage tag, no flow
    # event exists to animate it against.
    tagged_lines = {i for _tag, idxs in listing.stage_lines for i in idxs}
    imc_write_line = next(
        i for i, line in enumerate(listing.code.split("\n")) if "imc_write(op[i]);" in line
    )
    assert imc_write_line not in tagged_lines


def test_a_matmul_with_no_array_geometry_says_so_rather_than_implying_a_grid() -> None:
    """The one matmul this module still handles, and why (D54).

    A100 executes fp32 on its CUDA cores, which declare no ``systolic_dims``.
    There is then no tile grid to walk and nothing for ``bwz.emit`` to write, so
    the run falls back to this listing — and it has to say that, rather than
    print a sequence listing whose "a network is a SEQUENCE here" would be about
    a workload that is not a network.
    """
    spec = MatmulSpec.model_validate(
        {
            "id": "t",
            "name": "t",
            "family": "matmul",
            "m": 512,
            "n": 512,
            "k": 512,
            "a_dtype": "fp32",
            "b_dtype": "fp32",
        }
    )
    deployment = DeploymentSpec.model_validate({"batch": 1, "input_tokens": 1, "output_tokens": 1})
    chip = idealised(load_chip("a100_80gb"))
    report = analyze(spec, chip, deployment)
    assert report.feasible, report.infeasibility
    machine = machine_model(chip, DType.FP32)
    assert machine.unit.systolic_dims is None, "fp32 on A100 runs on the CUDA cores"

    graph = build_graph(spec, deployment, GraphPhase.STATIC)
    trace = build_trace(
        graph, report.phases[0], machine, double_buffered=report.memory.double_buffered
    )
    listing = deployment_of(
        chip, machine, report.phases[0], trace, workload="t", operation=graph.ops[0]
    )
    check(listing, trace)

    assert "declares no array geometry" in listing.code
    assert "no runnable program to emit" in listing.code
    assert "A network is a SEQUENCE" not in listing.code
