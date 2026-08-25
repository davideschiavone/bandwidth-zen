"""The deployment listing is derived from the schedule, not written alongside it.

Same contract as ``test_explain``: a listing that disagreed with the timeline
above it would be believed, so every constant it prints is checked against the
trace the figure drew (docs/CORRECTIONS.md D32).
"""

from __future__ import annotations

import math

import pytest

from bwz.analysis import analyze, idealised, machine_model
from bwz.analysis.pipeline import build_trace, tile_count
from bwz.deploy import check, deployment_of
from bwz.graph import GraphPhase, build_graph
from bwz.spec import DeploymentSpec, DType, MatmulSpec, load_chip


def _run(chip_id: str, m: int, n: int, k: int, dtype: str = "int8"):  # type: ignore[no-untyped-def]
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
    deployment = DeploymentSpec.model_validate({"batch": 1, "input_tokens": 1, "output_tokens": 1})
    chip = idealised(load_chip(chip_id))
    report = analyze(spec, chip, deployment)
    assert report.feasible, report.infeasibility
    graph = build_graph(spec, deployment, GraphPhase.STATIC)
    machine = machine_model(chip, DType(dtype))
    trace = build_trace(
        graph,
        report.phases[0],
        machine,
        double_buffered=report.memory.double_buffered,
        max_steps=32,
    )
    listing = deployment_of(
        chip, machine, report.phases[0], trace, workload="t", operation=graph.ops[0]
    )
    return chip, machine, graph, trace, listing


@pytest.mark.parametrize("chip_id", ["a100_80gb", "metis_aipu", "chip_a", "h100_sxm"])
@pytest.mark.parametrize("shape", [(600, 600, 600), (8192, 8192, 8192), (1, 4096, 4096)])
def test_every_listing_agrees_with_the_schedule(chip_id: str, shape: tuple[int, int, int]) -> None:
    """``check`` is the whole point: waves, tiles and units must reconcile."""
    _chip, machine, graph, trace, listing = _run(chip_id, *shape)
    check(listing, trace)

    assert listing.tiles == tile_count(graph.ops[0], machine)
    assert listing.waves == math.ceil(listing.tiles / listing.units)
    assert listing.waves == trace.tiles


def test_the_listing_quotes_the_tile_and_wave_counts_it_computed() -> None:
    """The text and the fields cannot disagree — they are the same numbers.

    8192-cubed INT8 on Metis: ceil(8192/512)^2 = 256 tiles over 4 AI cores is
    64 waves, and 256 tiles against 16 array-resident ones means 240 displace an
    earlier tile — NOT 240 extra writes. Within one pass M is innermost, so each
    tile serves all M rows once and is never revisited; the listing has to say
    that rather than imply a re-write (D30 correction).
    """
    _chip, _machine, _graph, _trace, listing = _run("metis_aipu", 8192, 8192, 8192)

    assert (listing.tiles, listing.waves, listing.units) == (256, 64, 4)
    assert listing.resident_tiles == 16
    assert listing.reloads == 240
    assert "#define TILES        256" in listing.code
    assert "#define WAVES        64" in listing.code
    assert "#define UNITS        4" in listing.code
    assert "256 tiles and only 16 fit, so 240 of them" in listing.code
    assert "Each is still written once in this pass" in listing.code


def test_the_per_tile_byte_share_does_not_shrink_when_the_wave_is_underfull() -> None:
    """D46: 1x1x2 on A100 is 1 real tile against 432 tensor cores, one wave.

    C's DRAM write is 2 B in total (``1x1`` at fp16) and there is exactly one
    real tile, so the per-tile share the listing prints for ``store_C`` must be
    that same 2 B. Before D46 the divisor was ``waves * units`` (432, the
    array's theoretical capacity in this one-wave case) instead of the 1 real
    tile, so the comment printed ``2 B / 432 = 4.63 mB`` — a fractional-byte
    quantity with no physical meaning.
    """
    _chip, _machine, _graph, _trace, listing = _run("a100_80gb", 1, 1, 2, dtype="fp16")

    assert (listing.tiles, listing.waves, listing.units) == (1, 1, 432)
    assert "store_C(u, tile(w, u));       /* 2 B — hollow bar */" in listing.code
    assert "mB" not in listing.code


def test_an_imc_array_gets_weight_sets_and_a_write_and_a_tensor_core_does_not() -> None:
    """The branch is driven by ``weight_sets``, not by the vendor's name.

    A D-IMC weight must be written into a bank before it can join a MAC; an
    NVIDIA tensor core reads both operands per instruction and stores nothing,
    so emitting a write for it would invent a residency it does not have (D30).
    """
    _c, _m, _g, _t, metis = _run("metis_aipu", 8192, 8192, 8192)
    _c2, _m2, _g2, _t2, a100 = _run("a100_80gb", 8192, 8192, 8192)

    assert "WEIGHT_SETS" in metis.code and "imc_write" in metis.code
    assert "weight sets per array" in metis.code
    assert metis.resident_tiles == 16

    assert "WEIGHT_SETS" not in a100.code and "imc_write" not in a100.code
    assert "stores no weights" in a100.code
    assert a100.resident_tiles == a100.units, "no weight residency: capacity is just the arrays"


def test_the_bit_serial_tax_appears_only_where_the_profile_declares_it() -> None:
    """``SUB_CYCLES`` comes from the dtype multiplier, so it is the profile's."""
    _c, _m, _g, _t, metis = _run("metis_aipu", 8192, 8192, 8192)
    _c2, _m2, _g2, _t2, a100 = _run("a100_80gb", 8192, 8192, 8192, dtype="fp16")

    assert "#define SUB_CYCLES   8" in metis.code, "int8 multiplier 0.125 -> 8 cycles"
    assert "SUB_CYCLES" not in a100.code, "fp16 multiplier is 1.0, so there is no sub-cycle loop"


def test_the_listing_stages_a_once_and_rotates_weight_sets() -> None:
    """D33: A k-slices are staged on chip and consumed by every tile of their
    group (A crosses DRAM exactly once), and reloads land in a set freed by
    the previous wave, so the write hides behind arithmetic instead of
    serialising in front of it.
    """
    _c, _m, _g, _t, metis = _run("metis_aipu", 8192, 8192, 8192)
    _c2, _m2, _g2, _t2, a100 = _run("a100_80gb", 8192, 8192, 8192)

    assert "#define NTILES_PER_KS 16" in metis.code, "16 n-tiles per k-slice at N=8192"
    assert "KSLICE" in metis.code
    assert "A crosses DRAM exactly once" in metis.code
    assert "staged once per k-slice" in metis.code
    assert "(w + 1) % WEIGHT_SETS" in metis.code, "write-ahead: next wave's tile, previous set"
    assert "write-ahead" in metis.code
    assert "must land before the array can use it" not in metis.code

    assert "NTILES_PER_KS" in a100.code and "KSLICE" in a100.code
    assert "A crosses DRAM exactly once" in a100.code
    assert "write-ahead" not in a100.code, "a tensor core stores no weights (D30)"


@pytest.mark.parametrize("chip_id", ["a100_80gb", "metis_aipu"])
def test_the_pseudo_c_has_no_nested_comments(chip_id: str) -> None:
    """C forbids them, and the listing is meant to survive being pasted (D26)."""
    _c, _m, _g, _t, listing = _run(chip_id, 600, 600, 600)

    depth = 0
    index = 0
    while index < len(listing.code) - 1:
        pair = listing.code[index : index + 2]
        if pair == "/*":
            depth += 1
            assert depth == 1, f"{chip_id}: nested /* at offset {index}"
            index += 2
            continue
        if pair == "*/":
            depth -= 1
            assert depth == 0, f"{chip_id}: unbalanced */ at offset {index}"
            index += 2
            continue
        index += 1
    assert depth == 0, f"{chip_id}: unterminated comment"


def test_double_buffered_listing_prefetches_wave_w_plus_1_beside_waves_compute() -> None:
    """D41: ``#define DEPTH 2 /* double buffered */`` used to sit above a loop
    that read as fully serial. The real schedule (``_pipelined_tiles``) lets
    wave i's load start once the buffer two waves back has been freed, with no
    dependency on wave i-1's compute at all — genuine overlap — so the listing
    should show a wave-0 prologue and a steady-state loop that prefetches wave
    w+1 alongside wave w's own arithmetic, not one big serial block.

    This also pins a real bug the restructuring fixes: under write-ahead, no
    statement used to write wave 0's own tile into a weight set at all (the
    per-wave line only ever wrote tile(w+1, u), so at w=0 that's tile 1) even
    though the prologue *comment* already claimed it landed somewhere.
    """
    _c, _m, _g, trace, metis = _run("metis_aipu", 8192, 8192, 8192)
    assert trace.double_buffered

    assert "for (int u = 0; u < UNITS; ++u) {" in metis.code
    assert "load_B(u, tile(0, u));" in metis.code, "prologue primes wave 0 before the loop"
    assert "imc_write(u, 0, tile(0, u));" in metis.code, "the wave-0-never-written fix"
    assert "load_B(u, tile(w + 1, u));" in metis.code, "steady state prefetches wave w+1"
    assert "if (w + 1 < WAVES) {" in metis.code
    assert "imc_write(u, (w + 1) % WEIGHT_SETS, tile(w + 1, u));" in metis.code

    # Every stage still points at real, correct lines: not just present text,
    # but the exact statement a debug view would highlight.
    lines = metis.code.split("\n")
    by_stage = dict(metis.stage_lines)
    assert len(by_stage["load_b"]) == 2, "one prologue occurrence, one steady-state occurrence"
    assert all("load_B(u, tile(" in lines[i] for i in by_stage["load_b"])
    assert all("mac(" in lines[i] or "feed(" in lines[i] for i in by_stage["exec"])
    assert all("store_C(" in lines[i] for i in by_stage["store"])

    # Every substring the pre-existing test suite already pins still survives —
    # the restructuring must not be a stealth rewrite of tested behaviour.
    for assertion in (
        "#define NTILES_PER_KS 16",
        "KSLICE",
        "A crosses DRAM exactly once",
        "staged once per k-slice",
        "(w + 1) % WEIGHT_SETS",
        "write-ahead",
    ):
        assert assertion in metis.code


def test_non_double_buffered_listing_stays_serial() -> None:
    """The depth==1 fallback: no shipped profile at any shape actually produces
    ``double_buffered=False`` (SRAM capacity fits two tiles everywhere), so this
    forces it directly — shrink every non-DRAM level below twice one tile's
    bytes, which flips ``report.memory.double_buffered`` end to end (an
    ``on_chip_capacity_bytes`` override alone would silently no-op: it is a
    computed property, not a stored field). Uses a ``weight_sets == 1`` chip
    (A100) deliberately: the weight-set write-ahead/persistent commentary is
    gated only on ``weight_sets > 1``, independent of ``depth``, so testing on
    an IMC chip would trip an unrelated, pre-existing gap this change is not
    trying to close.

    ``_pipelined_tiles`` genuinely serialises at depth==1 (a store is placed
    before the next load's start is even computed), so the listing must show
    no prologue and no reference to a next wave at all — the exact text this
    module always printed, not the new double-buffered shape.
    """
    chip = idealised(load_chip("a100_80gb"))
    shrunk_memory = [
        level.model_copy(update={"capacity_bytes": 300}) if level.name != chip.dram.name else level
        for level in chip.memory
    ]
    chip = chip.model_copy(update={"memory": shrunk_memory})
    spec = MatmulSpec.model_validate(
        {
            "id": "t",
            "name": "t",
            "family": "matmul",
            "m": 600,
            "n": 600,
            "k": 600,
            "a_dtype": "int8",
            "b_dtype": "int8",
        }
    )
    deployment = DeploymentSpec.model_validate({"batch": 1, "input_tokens": 1, "output_tokens": 1})
    report = analyze(spec, chip, deployment)
    assert report.feasible, report.infeasibility
    assert report.memory.double_buffered is False, "the shrink above must actually flip this"
    graph = build_graph(spec, deployment, GraphPhase.STATIC)
    machine = machine_model(chip, DType.INT8)
    trace = build_trace(
        graph,
        report.phases[0],
        machine,
        double_buffered=report.memory.double_buffered,
        max_steps=32,
    )
    listing = deployment_of(
        chip, machine, report.phases[0], trace, workload="t", operation=graph.ops[0]
    )
    check(listing, trace)

    assert "w + 1" not in listing.code, "depth==1: nothing prefetches a wave ahead"
    assert "prologue" not in listing.code
    lines = listing.code.split("\n")
    by_stage = dict(listing.stage_lines)
    assert len(by_stage["load_b"]) == 1, "one load_B statement, referencing wave w"
    assert "tile(w, u)" in lines[by_stage["load_b"][0]]


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
