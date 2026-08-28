"""Storage widths are exact definitions, so these are equality tests, not tolerances."""

from __future__ import annotations

import pytest

from bwz.spec.dtypes import DType, accumulator_for, bytes_per_element, is_integer


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


@pytest.mark.parametrize(
    ("operand", "expected"),
    [
        (DType.INT8, DType.INT32),
        (DType.INT4, DType.INT32),
        (DType.INT32, DType.INT32),
        (DType.FP16, DType.FP32),
        (DType.BF16, DType.FP32),
        (DType.TF32, DType.FP32),
        (DType.FP8, DType.FP32),
        (DType.FP32, DType.FP32),
    ],
)
def test_the_accumulator_follows_the_operand(operand: DType, expected: DType) -> None:
    """An accumulator width is a definition, not a knob (D56).

    The integer half is the one that bites: ``int8 x int8 -> int32`` is the only
    integer MMA shape these arrays issue, and an ``int8`` accumulator would
    overflow after three or four terms. It used to come from a deployment field
    that defaulted to ``fp32`` for every dtype, so an int8 matmul was described
    as accumulating in floating point.
    """
    assert accumulator_for(operand) is expected


def test_every_dtype_has_an_accumulator() -> None:
    """A new DType member without an entry would KeyError when a program is emitted."""
    for dtype in DType:
        assert accumulator_for(dtype) in (DType.FP32, DType.INT32)


def test_an_integer_operand_never_accumulates_in_a_float() -> None:
    """The property behind the table, stated so a future entry cannot break it."""
    for dtype in DType:
        assert is_integer(accumulator_for(dtype)) == is_integer(dtype)
