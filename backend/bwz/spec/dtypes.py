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
    INT32 = "int32"
    """Accumulator width. No chip declares a compute unit for it -- an int8 x int8
    product accumulates in int32 at the *int8* rate -- so it appears only as the
    result dtype of a matmul, where it doubles or quadruples the output bytes
    without changing a single operation."""


_BYTES_PER_ELEMENT: dict[DType, float] = {
    DType.FP32: 4.0,
    DType.TF32: 4.0,  # fp32 storage, reduced-mantissa datapath
    DType.FP16: 2.0,
    DType.BF16: 2.0,
    DType.FP8: 1.0,
    DType.INT8: 1.0,
    DType.INT4: 0.5,
    DType.INT32: 4.0,
}


def bytes_per_element(dtype: DType) -> float:
    """Storage width of one element of *dtype*, in bytes.

    Returns a float because int4 is half a byte; callers that need a whole
    number of bytes for a tensor must round the *total*, never this.
    """
    return _BYTES_PER_ELEMENT[dtype]


def is_integer(dtype: DType) -> bool:
    """True for integer formats, whose peak throughput is quoted in OP/s not FLOP/s."""
    return dtype in (DType.INT8, DType.INT4, DType.INT32)


_ACCUMULATOR: dict[DType, DType] = {
    DType.FP32: DType.FP32,
    DType.TF32: DType.FP32,
    DType.FP16: DType.FP32,
    DType.BF16: DType.FP32,
    DType.FP8: DType.FP32,
    DType.INT8: DType.INT32,
    DType.INT4: DType.INT32,
    DType.INT32: DType.INT32,
}


def accumulator_for(dtype: DType) -> DType:
    """The format a matmul in *dtype* accumulates into.

    A **definition**, not a free choice, which is why it lives here beside the
    widths rather than in a deployment field. The sum of ``K`` products does not
    fit the operand's own format, so every matrix engine accumulates wider, and
    which format it accumulates into follows from the operand's:

    - **Integers accumulate in ``int32``, always.** ``int8 x int8 -> int32`` is
      the only integer MMA shape any of these chips issues (PTX
      ``mma.sync...s32.s8.s8.s32``, and the same for ``s4``); an ``int8``
      accumulator would overflow after three or four terms. The class docstring
      above already states this rule — this is where it becomes callable.
    - **Floats accumulate in ``fp32``.** That is what cuBLAS computes by default
      (``CUBLAS_COMPUTE_32F``) and the mode a datacenter part's headline fp16
      figure is quoted at. ``bf16`` and ``tf32`` have *no* narrower accumulate on
      NVIDIA hardware at all — the MMA shapes are ``f32``-only.

    **``fp16`` accumulate is a real mode this does not model.** PTX offers
    ``mma.sync...f16.f16.f16.f16`` and cuBLAS ``CUBLAS_COMPUTE_16F``; on GeForce
    parts it is even the *faster* path, because fp32-accumulate is deliberately
    half-rate there. It loses precision over a long contraction, nobody's default
    picks it, and no profile here declares the two rates separately, so modelling
    it would mean inventing a number. Ask for it explicitly with
    ``precision.accumulate`` if you want to see it (docs/CORRECTIONS.md D56).
    """
    return _ACCUMULATOR[dtype]
