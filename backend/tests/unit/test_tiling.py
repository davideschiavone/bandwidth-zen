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
from bwz.spec import DType, load_chip


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

    The model gives ``M/padded(M, rows) = 1/128`` exactly (D52): one row of
    work occupies one 128-row instruction tile and pays for all of it. Before
    D52 it gave 1/129, treating the loss as a pipeline drain rather than area.
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
    [(1, 1 / 512), (128, 0.25), (512, 1.0), (2048, 1.0), (8192, 1.0)],
)
def test_utilisation_rises_with_batch(m: int, expected: float) -> None:
    """M below one instruction tile wastes the rest of it; a multiple wastes none.

    The MMA branch (D52): M is a spatial dimension sliced into independent
    512-row tiles, so the only M-side loss is the ragged last tile. M=1 pays
    for a whole 512-row tile (1/512); M=512 and every multiple above it pays
    for nothing. Before D52 this asserted a rows-deep pipeline fill that never
    reached 1.0 even on a perfectly-shaped M — 0.5 at M=512, 0.8 at M=2048.
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
        (100 / 512) * (100_000 / 100_352), rel=1e-6
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
    """Gemma-3-4B's Q projection at batch 1 on chip_a: 2560x2048 on a 512x512 array.

    Passes the dtype, as production does (``roofline.py`` has the only call
    site and always supplies ``machine.dtype``): chip_a is bit-serial at int8,
    so this exercises the crossbar branch it actually runs on. Still ~1/512 —
    one active row of 512 — times the sub-cycle fill riding K.
    """
    chip = load_chip("chip_a")
    unit = chip.compute_units[0]
    op = Operation(
        id="q", op_type=OpType.MATMUL, attrs=MatmulAttrs(m=1, n=2048, k=2560), outputs=()
    )
    expected = (1 / 512) * (2560 / (padded(2560, 512) + 1))
    assert operation_utilisation(op, unit, DType.INT8) == pytest.approx(expected, rel=1e-12)
    assert operation_utilisation(op, unit, DType.INT8) == pytest.approx(1 / 512, rel=1e-3)


def test_a_bit_serial_crossbar_has_no_m_serial_pipeline() -> None:
    """D34: the tail effect depends on what the array physically is.

    The Metis D-IMC is a combinational crossbar: M enters in whole 512-row
    chunks (area-only loss), and the bit-serial stream rides K — 8 sub-cycles
    per row plus one sub-cycle row of fill::

        util = M/padded(M, 512) * K/(padded(K, 512) + 1) * N/padded(N, 512)

    An 8192-row stream therefore loses one row's fill, not 512: 8192/8193.
    Without a dtype there is no bit-serial multiplier to detect, so the MMA
    branch applies and a perfectly-shaped 8192-cube loses nothing at all
    (D52). Production never takes that path — ``roofline.py`` is the only
    call site and always passes ``machine.dtype`` — so it is the INT8 line
    below that describes what Metis actually reports.
    """
    unit = load_chip("metis_aipu").compute_units[0]
    op = Operation(
        id="mm", op_type=OpType.MATMUL, attrs=MatmulAttrs(m=8192, n=8192, k=8192), outputs=()
    )
    assert operation_utilisation(op, unit, DType.INT8) == pytest.approx(8192 / 8193, rel=1e-12)
    assert operation_utilisation(op, unit) == pytest.approx(1.0, rel=1e-12)

    # The same chip at batch 1: the crossbar wastes area, not pipeline — the
    # M=1 GEMM still runs at ~1/513 of peak (the 1/512 area loss times the
    # sub-cycle fill), not at 1/9 as an M-side sub-cycle tail would claim.
    decode = Operation(
        id="q", op_type=OpType.MATMUL, attrs=MatmulAttrs(m=1, n=2048, k=2560), outputs=()
    )
    expected = 1 / 512 * 2560 / (padded(2560, 512) + 1) * 2048 / padded(2048, 512)
    assert operation_utilisation(decode, unit, DType.INT8) == pytest.approx(expected, rel=1e-12)


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

    Modelled on the crossbar branch Metis actually runs (fill_cycles=8 at
    int8), not the MMA branch: M and N lose area, K carries the sub-cycle
    fill.

        shape utilisation  = (100/512)^2 x (100/513) = 0.7436%
        wave occupancy     = 1 tile / (1 wave x 4 arrays) = 25%
        chip utilisation   = 0.1859%
    """
    unit = load_chip("metis_aipu").compute_units[0]
    assert unit.count == 4
    shape = systolic_utilisation(100, 100, 100, 512, 512, fill_cycles=8)
    chip = systolic_utilisation(100, 100, 100, 512, 512, units=unit.count, fill_cycles=8)

    assert shape == pytest.approx(0.0074361, rel=1e-3)
    assert chip == pytest.approx(shape * 0.25, rel=1e-9)
    assert chip == pytest.approx(0.0018590, rel=1e-3)


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


def test_operation_tiles_is_what_wave_occupancy_divides_by() -> None:
    """The count factored out of ``systolic_utilisation`` (D30), hand-computed.

    A 1000x2000x3000 matmul on a 16x16 array: ``os`` gives ceil(1000/16) x
    ceil(2000/16) = 63 x 125 = 7875 output tiles; ``ws`` gives ceil(3000/16) x
    125 = 188 x 125 = 23,500 k-slices, which is the parallelism it buys and the
    reason it also owes a reduction. The figures ask this same function how many
    arrays ever receive a tile, so the picture and the utilisation term cannot
    disagree about the decomposition.
    """
    from bwz.analysis.tiling import operation_tiles
    from bwz.graph.ops import MatmulAttrs, Operation, OpType
    from bwz.spec import load_chip
    from bwz.spec.hardware_spec import Dataflow

    unit = load_chip("a100_80gb").compute_units[0]
    op = Operation(
        id="m",
        op_type=OpType.MATMUL,
        inputs=(),
        outputs=(),
        attrs=MatmulAttrs(m=1000, n=2000, k=3000),
    )

    assert operation_tiles(op, unit) == 63 * 125, "os, the unit's own dataflow"
    assert operation_tiles(op, unit, stationarity=Dataflow.WEIGHT_STATIONARY) == 188 * 125
    assert operation_tiles(op, unit, k_partitions=4) == 63 * 125 * 4, "split-K multiplies it"


def test_attention_tiles_count_every_head() -> None:
    """Every (batch, head) pair is a separate GEMM, so they fill the arrays
    alongside each other — counting one head's tiles would report a 32-head
    attention as leaving a 4-core NPU idle (the ``independent`` term, now folded
    into the tile count so there is one definition of it).
    """
    from bwz.analysis.tiling import operation_tiles
    from bwz.graph.ops import AttentionAttrs, Operation, OpType
    from bwz.spec import load_chip

    unit = load_chip("a100_80gb").compute_units[0]
    attrs = AttentionAttrs(
        batch=1,
        heads=32,
        kv_heads=32,
        q_len=1,
        kv_len=2048,
        head_dim=128,
        causal=False,
        materialize_scores=False,
    )
    op = Operation(id="a", op_type=OpType.ATTENTION, inputs=(), outputs=(), attrs=attrs)

    per_head = (128 // 16) * (2048 // 16)
    assert operation_tiles(op, unit) == per_head * 32
