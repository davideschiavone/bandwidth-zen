"""The systolic tail effect — the test that decides whether an array is modelled."""

from __future__ import annotations

import pytest

from bwz.analysis.tiling import (
    double_buffering_fits,
    operation_utilisation,
    padded,
    systolic_utilisation,
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
