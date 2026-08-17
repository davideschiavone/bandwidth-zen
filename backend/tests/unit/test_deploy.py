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
