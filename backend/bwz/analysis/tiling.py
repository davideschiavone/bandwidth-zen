"""Operand-shape utilisation: the systolic tail effect. ``docs/MODEL.md`` §6.1.

This is the module that decides whether the engine is modelling an *array* or
just dividing FLOPs by a peak number. CLAUDE.md is explicit: a GEMM with M=1 on a
128x128 array must come out at about 1/128 utilisation, and a model that does not
reproduce that is not modelling the hardware.
"""

from __future__ import annotations

import math

from bwz.graph.ops import (
    AttentionAttrs,
    ConvAttrs,
    MatmulAttrs,
    Operation,
)
from bwz.spec.hardware_spec import ComputeUnit


def padded(extent: int, tile: int) -> int:
    """Round *extent* up to a whole number of *tile*-sized steps.

    The ``ceil`` must operate on the padded **dimension**, not on the tile count
    (CLAUDE.md gotchas). ``padded(1, 512) == 512``, so a small-M GEMM reports a
    utilisation below 1 rather than above it.
    """
    if tile <= 0:
        return extent
    return math.ceil(extent / tile) * tile


def systolic_utilisation(m: int, k: int, n: int, rows: int, cols: int) -> float:
    """Fraction of a ``rows x cols`` array a ``[M,K]x[K,N]`` GEMM keeps busy.

    A weight-stationary array holds a ``rows x cols`` slice of the weight matrix
    and streams ``M`` activation rows through it. Each tile costs ``M + rows``
    cycles: ``M`` to push the data and ``rows`` to fill and drain the pipeline.
    Over ``ceil(K/rows) * ceil(N/cols)`` tiles::

        actual_cycles = ceil(K/rows) * ceil(N/cols) * (M + rows)
        ideal_cycles  = M * K * N / (rows * cols)
        utilisation   = ideal / actual
                      = [K / padded(K, rows)] * [N / padded(N, cols)] * [M / (M + rows)]

    Three independent losses, all bounded by 1:

    - **K and N padding** — a 100-wide output on a 512-wide array wastes 80% of it.
    - **The M tail** — with M=1 the pipeline is filled and drained for a single
      row of work. On a 512x512 array that is a factor of 513.

    Worked example (CLAUDE.md sanity check): M=1 on a 128x128 array with K and N
    both multiples of 128 gives ``1 * 1 * 1/129 = 0.0078``, i.e. 1/128 to within
    the fill term. Worked example (prefill, M=2048, 512x512 array):
    ``2048/2560 = 0.80``.

    This is the single largest correction the engine applies at batch 1, and it
    is why a decode step on an edge NPU is nowhere near its TOPS number even
    when the weights are entirely on chip.
    """
    if rows <= 0 or cols <= 0:
        return 1.0
    k_efficiency = k / padded(k, rows)
    n_efficiency = n / padded(n, cols)
    m_efficiency = m / (m + rows)
    return k_efficiency * n_efficiency * m_efficiency


def operation_utilisation(op: Operation, unit: ComputeUnit) -> float:
    """Shape-induced utilisation for *op* on *unit*.

    Returns 1.0 when the profile declares no array geometry — there is then no
    basis for a tail-effect claim, and inventing one would be worse than
    omitting it. Non-GEMM operations also return 1.0: they are memory-bound by
    two orders of magnitude, so their compute term never binds and refining it
    would be effort spent where it cannot matter.
    """
    if unit.systolic_dims is None:
        return 1.0
    rows, cols = unit.systolic_dims

    if isinstance(op.attrs, MatmulAttrs):
        return systolic_utilisation(op.attrs.m, op.attrs.k, op.attrs.n, rows, cols)

    if isinstance(op.attrs, ConvAttrs):
        # im2col view: M is the output pixel count, K the filter volume,
        # N the output channels.
        conv = op.attrs
        m = conv.batch * conv.out_height * conv.out_width
        k = (conv.in_channels // conv.groups) * conv.kernel_h * conv.kernel_w
        return systolic_utilisation(m, k, conv.out_channels, rows, cols)

    if isinstance(op.attrs, AttentionAttrs):
        # Two GEMMs per head: [q_len, head_dim] x [head_dim, kv_len] and back.
        attention = op.attrs
        return systolic_utilisation(
            attention.q_len, attention.head_dim, attention.kv_len, rows, cols
        )

    return 1.0


def double_buffering_fits(
    on_chip_capacity_bytes: float, resident_weight_bytes: float, working_tile_bytes: float
) -> bool:
    """Whether on-chip capacity has room for a second tile alongside the resident set.

    This is where SRAM *capacity* earns the ``max(load, compute)`` rule instead of
    it being asserted (D5a). Overlapping the next tile's load with the current
    tile's compute needs both resident simultaneously; when capacity does not
    allow it the loads serialise behind compute and the phase time is a sum.
    """
    spare = on_chip_capacity_bytes - resident_weight_bytes
    return spare >= 2.0 * working_tile_bytes


def tile_bytes(unit: ComputeUnit, bytes_per_element: float) -> float:
    """Bytes in one weight tile of the declared array geometry.

    Falls back to a 128x128 tile when no geometry is declared — a conventional
    size that keeps the double-buffering test meaningful for chips whose profile
    does not describe an array.
    """
    rows, cols = unit.systolic_dims if unit.systolic_dims is not None else (128, 128)
    return rows * cols * bytes_per_element
