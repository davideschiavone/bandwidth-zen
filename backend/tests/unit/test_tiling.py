"""The systolic tail effect — the test that decides whether an array is modelled."""

from __future__ import annotations

import pytest

from bwz.analysis.tiling import (
    double_buffering_fits,
    operation_utilisation,
    padded,
    systolic_utilisation,
    wave_occupancy,
)
from bwz.graph.ops import MatmulAttrs, Operation, OpType
from bwz.spec import load_chip


def test_padded_rounds_up_the_dimension_not_the_tile_count() -> None:
    """CLAUDE.md gotcha: ``ceil`` on the tile count lets utilisation exceed 100%."""
    assert padded(1, 512) == 512
    assert padded(512, 512) == 512
    assert padded(513, 512) == 1024
    assert padded(100, 128) == 128


def test_batch_one_on_a_128x128_array_is_one_over_128() -> None:
    """CLAUDE.md sanity check, quoted verbatim: "GEMM with M=1 on a 128x128
    systolic array -> utilization approximately 1/128 from the tail effect. If
    your utilization model doesn't reproduce this, it isn't modelling the array."

    The model gives ``M/(M+rows) = 1/129 = 0.00775`` against ``1/128 = 0.00781``:
    the extra row is the pipeline drain, which is real.
    """
    utilisation = systolic_utilisation(m=1, k=1024, n=1024, rows=128, cols=128)
    assert utilisation == pytest.approx(1 / 128, rel=0.01)


def test_batch_one_on_a_512x512_array_is_one_over_512() -> None:
    """chip_a and chip_b. This single factor is the difference between D8's
    ``2*params/tops`` compute term and what a batch-1 GEMM actually achieves."""
    assert systolic_utilisation(m=1, k=2560, n=2048, rows=512, cols=512) == pytest.approx(
        1 / 512, rel=0.01
    )


@pytest.mark.parametrize(
    ("m", "expected"),
    [(1, 1 / 513), (128, 128 / 640), (512, 0.5), (2048, 0.8), (8192, 8192 / 8704)],
)
def test_utilisation_rises_with_batch(m: int, expected: float) -> None:
    """A 512-deep pipeline needs M >> 512 to amortise its fill and drain.

    At M=512 — a 512-token prompt — the array is still only half busy. That is a
    real architectural property of a deep array, not a modelling artefact.
    """
    assert systolic_utilisation(m, 2560, 2560, 512, 512) == pytest.approx(expected, rel=1e-6)


def test_utilisation_never_exceeds_one() -> None:
    for m in (1, 7, 64, 512, 10_000):
        for k in (1, 100, 512, 2560, 4097):
            for n in (1, 100, 512, 4096):
                assert 0.0 < systolic_utilisation(m, k, n, 512, 512) <= 1.0


def test_ragged_dimensions_waste_the_array() -> None:
    """A 100-wide output on a 512-wide array uses 100/512 of it."""
    assert systolic_utilisation(m=100_000, k=512, n=100, rows=512, cols=512) == pytest.approx(
        (100 / 512) * (100_000 / 100_512), rel=1e-6
    )


def test_no_declared_geometry_means_no_tail_effect_claim() -> None:
    """Inventing an array for a profile that does not describe one would be worse
    than omitting the correction."""
    chip = load_chip("h100_sxm")
    cuda_core = next(u for u in chip.compute_units if u.name == "cuda_core")
    assert cuda_core.systolic_dims is None
    op = Operation(
        id="mm", op_type=OpType.MATMUL, attrs=MatmulAttrs(m=1, n=4096, k=4096), outputs=()
    )
    assert operation_utilisation(op, cuda_core) == 1.0


def test_chip_a_decode_projection_utilisation() -> None:
    """Gemma-3-4B's Q projection at batch 1 on chip_a: 2560x2048 on a 512x512 array."""
    chip = load_chip("chip_a")
    unit = chip.compute_units[0]
    op = Operation(
        id="q", op_type=OpType.MATMUL, attrs=MatmulAttrs(m=1, n=2048, k=2560), outputs=()
    )
    assert operation_utilisation(op, unit) == pytest.approx(1 / 513, rel=1e-6)


def test_double_buffering_needs_room_for_two_tiles() -> None:
    assert double_buffering_fits(1_000_000, 0, 400_000) is True
    assert double_buffering_fits(1_000_000, 300_000, 400_000) is False
    assert double_buffering_fits(1_000_000, 999_998, 1) is True
    assert double_buffering_fits(1_000_000, 999_999, 1) is False, "1 spare byte, 2 needed"


def test_wave_occupancy_is_one_when_the_work_fills_the_arrays() -> None:
    """No loss when tiles divide evenly, and none at all on a single-array chip."""
    assert wave_occupancy(tiles=16, units=4) == 1.0
    assert wave_occupancy(tiles=4, units=4) == 1.0
    assert wave_occupancy(tiles=1, units=1) == 1.0
    assert wave_occupancy(tiles=1_000_000, units=1) == 1.0


def test_wave_occupancy_charges_the_empty_slots_of_the_last_wave() -> None:
    """5 tiles on 4 arrays is two waves, the second holding one tile: 5/8."""
    assert wave_occupancy(tiles=5, units=4) == pytest.approx(5 / 8)
    # Metis: one 512x512 weight tile occupies one of four AI cores.
    assert wave_occupancy(tiles=1, units=4) == pytest.approx(0.25)


def test_a_small_matmul_cannot_reach_all_of_metis() -> None:
    """The correction D30 exists for, on the shipped profile.

    A 100x100x100 INT8 matmul is a single 512x512 weight tile. Metis has four AI
    cores; one takes the tile and three have nothing to do, so the chip's real
    ceiling on this shape is a quarter of its 209.7 TOPS. Before D30 the engine
    multiplied one array's throughput by four and reported the whole chip busy.

        shape utilisation  = (100/512)^2 x (100/612) = 0.6233%
        wave occupancy     = 1 tile / (1 wave x 4 arrays) = 25%
        chip utilisation   = 0.1558%
    """
    unit = load_chip("metis_aipu").compute_units[0]
    assert unit.count == 4
    shape = systolic_utilisation(100, 100, 100, 512, 512)
    chip = systolic_utilisation(100, 100, 100, 512, 512, units=unit.count)

    assert shape == pytest.approx(0.006233, rel=1e-3)
    assert chip == pytest.approx(shape * 0.25, rel=1e-9)
    assert chip == pytest.approx(0.0015583, rel=1e-3)


def test_metis_holds_sixteen_weight_tiles_and_runs_four() -> None:
    """Table C: 4 weight sets per AI core, 4 cores.

    A D-IMC weight cannot join a MAC until it has been written into a bank, so
    this is a hard residency limit and not a cache hint: 16 tiles fit, 4 compute
    at once, and a matmul needing more re-writes the array as it goes.
    """
    unit = load_chip("metis_aipu").compute_units[0]
    assert (unit.count, unit.weight_sets) == (4, 4)
    assert unit.resident_tile_capacity() == 16

    # A 4096-cubed INT8 matmul is ceil(4096/512)^2 = 64 tiles: four times what
    # the arrays hold, so 48 tiles must be re-written mid-operation.
    assert 64 - unit.resident_tile_capacity() == 48

    # A tensor core stores no weights at all, so its capacity is just its arrays.
    tensor_core = load_chip("a100_80gb").compute_units[0]
    assert tensor_core.weight_sets == 1
    assert tensor_core.resident_tile_capacity() == 432
