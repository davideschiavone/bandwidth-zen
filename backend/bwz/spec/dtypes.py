"""Numeric formats and their storage width.

These are *exact definitions*, not empirical constants: an fp16 element occupies
two bytes by definition. They therefore live here rather than in
``calibration.py`` (CLAUDE.md #2 governs fitted or measured values).

``tf32`` is a compute format, not a storage format — a tf32 tensor is stored as
fp32 and truncated inside the tensor core, so its storage width is 4 bytes while
its throughput multiplier (declared per compute unit) is typically 0.5.
"""

from __future__ import annotations

from enum import StrEnum


class DType(StrEnum):
    """A numeric format the engine understands."""

    FP32 = "fp32"
    TF32 = "tf32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"


_BYTES_PER_ELEMENT: dict[DType, float] = {
    DType.FP32: 4.0,
    DType.TF32: 4.0,  # fp32 storage, reduced-mantissa datapath
    DType.FP16: 2.0,
    DType.BF16: 2.0,
    DType.FP8: 1.0,
    DType.INT8: 1.0,
    DType.INT4: 0.5,
}


def bytes_per_element(dtype: DType) -> float:
    """Storage width of one element of *dtype*, in bytes.

    Returns a float because int4 is half a byte; callers that need a whole
    number of bytes for a tensor must round the *total*, never this.
    """
    return _BYTES_PER_ELEMENT[dtype]


def is_integer(dtype: DType) -> bool:
    """True for integer formats, whose peak throughput is quoted in OP/s not FLOP/s."""
    return dtype in (DType.INT8, DType.INT4)
