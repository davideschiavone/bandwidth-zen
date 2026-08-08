"""Storage widths are exact definitions, so these are equality tests, not tolerances."""

from __future__ import annotations

import pytest

from bwz.spec.dtypes import DType, bytes_per_element, is_integer


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (DType.FP32, 4.0),
        (DType.TF32, 4.0),  # fp32 storage, reduced-mantissa datapath
        (DType.FP16, 2.0),
        (DType.BF16, 2.0),
        (DType.FP8, 1.0),
        (DType.INT8, 1.0),
        (DType.INT4, 0.5),
    ],
)
def test_bytes_per_element(dtype: DType, expected: float) -> None:
    assert bytes_per_element(dtype) == expected


def test_every_dtype_has_a_width() -> None:
    """A new DType member without a width entry would KeyError at analysis time."""
    for dtype in DType:
        assert bytes_per_element(dtype) > 0


def test_integer_dtypes() -> None:
    assert is_integer(DType.INT8)
    assert is_integer(DType.INT4)
    assert not is_integer(DType.FP8)
