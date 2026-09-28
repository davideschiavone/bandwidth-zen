"""Operand-shape utilisation: the systolic tail effect. ``docs/MODEL.md`` §6.1.

This is the module that decides whether the engine is modelling an *array* or
just dividing FLOPs by a peak number. CLAUDE.md is explicit: a GEMM with M=1 on a
128x128 array must come out at about 1/128 utilisation, and a model that does not
reproduce that is not modelling the hardware.
"""

from __future__ import annotations

import math

from bwz.analysis.stationarity import Deal, TileGrid, deal_for, grid_for
from bwz.graph.ops import (
    AttentionAttrs,
    ConvAttrs,
    MatmulAttrs,
    Operation,
)
from bwz.spec.dtypes import DType
from bwz.spec.hardware_spec import ComputeUnit, Dataflow


def padded(extent: int, tile: int) -> int:
    """Round *extent* up to a whole number of *tile*-sized steps.

    The ``ceil`` must operate on the padded **dimension**, not on the tile count
    (CLAUDE.md gotchas). ``padded(1, 512) == 512``, so a small-M GEMM reports a
    utilisation below 1 rather than above it.
    """
    if tile <= 0:
        return extent
    return math.ceil(extent / tile) * tile


def wave_occupancy(tiles: int, units: int) -> float:
    """Fraction of the chip's arrays kept busy, averaged over the run.

    A chip with *units* arrays runs *units* weight tiles at once and no more, so
    ``tiles`` tiles take ``ceil(tiles / units)`` waves and the last wave is
    partly empty::

        waves     = ceil(tiles / units)
        occupancy = tiles / (waves * units)

    This is the loss the aggregate peak hides. ``peak_flops_per_s`` multiplies one
    array's throughput by ``count``, which silently assumes every array always has
    a tile to work on — true for a 4096-cubed GEMM, false for a small one. Metis
    has four AI cores; a matmul with a single 512x512 weight tile occupies **one**
    of them and the other three idle, so its real ceiling is a quarter of the
    209.7 TOPS the datasheet quotes (docs/CORRECTIONS.md D30).

    Returns 1.0 for a single-unit chip, which is what makes this a refinement
    rather than a rewrite: the term only exists where a profile declares more
    than one array.
    """
    if units <= 1 or tiles <= 0:
        return 1.0
    waves = math.ceil(tiles / units)
    return tiles / (waves * units)


def systolic_utilisation(
    m: int,
    k: int,
    n: int,
    rows: int,
    cols: int,
    *,
    units: int = 1,
    fill_cycles: int | None = None,
    tiles: int | None = None,
    occupancy: float | None = None,
) -> float:
    """Fraction of a ``rows x cols`` array a ``[M,K]x[K,N]`` GEMM keeps busy.

    A weight-stationary array holds a ``rows x cols`` slice of the weight matrix
    and streams ``M`` activation rows through it. Each tile costs ``M + fill``
    cycles: ``M`` to push the data and ``fill`` to fill and drain the pipeline.
    Over ``ceil(K/rows) * ceil(N/cols)`` tiles::

        actual_cycles = ceil(K/rows) * ceil(N/cols) * (M + fill)
        ideal_cycles  = M * K * N / (rows * cols)
        utilisation   = ideal / actual
                      = [K / padded(K, rows)] * [N / padded(N, cols)] * [M / (M + fill)]

    Three independent losses, all bounded by 1:

    - **K and N padding** — a 100-wide output on a 512-wide array wastes 80% of it.
    - **The M tail** — with M=1 the pipeline is filled and drained for a single
      row of work. On a 512x512 array that is a factor of 513.

    **What ``fill`` is depends on what the array physically is** (D34). For a
    conventional systolic pump the M dimension streams serially through a
    ``rows``-deep pipeline: M rows cost ``M + rows`` cycles, so the tail is the
    row count. A bit-serial crossbar is combinational: M enters in whole
    ``rows``-sized chunks with no serial pipeline at all, so the M side loses
    only area on ragged chunks, and the serial cost moves to K — each row takes
    ``1 / multiplier`` sub-cycles, plus one sub-cycle row of fill per chunk::

        m_efficiency = M / padded(M, rows)          # area only
        k_efficiency = K / (padded(K, rows) + 1)    # K * s / (padded(K,s) * s + s)

    ``fill_cycles`` selects the crossbar branch (it is derived from the
    profile's dtype multiplier); ``None`` keeps the conservative systolic
    depth ``rows``.

    Worked example (CLAUDE.md sanity check): M=1 on a 128x128 array with K and N
    both multiples of 128 gives ``1 * 1 * 1/129 = 0.0078``, i.e. 1/128 to within
    the fill term. Worked example (prefill, M=2048, 512x512 array):
    ``2048/2560 = 0.80``.

    This is the single largest correction the engine applies at batch 1, and it
    is why a decode step on an edge NPU is nowhere near its TOPS number even
    when the weights are entirely on chip.

    A **fourth** loss applies when the chip has more than one array: ``units`` is
    how many run concurrently and ``tiles`` how many independent pieces of work
    there are to spread across them — :func:`operation_tiles` counts those, and
    for attention it is ``batch x heads`` times one head's grid, since the heads
    are genuinely separate GEMMs. See :func:`wave_occupancy`. Both default to
    the single-array case, so a profile that declares one unit gets exactly the
    three-term result above.
    """
    if rows <= 0 or cols <= 0:
        return 1.0
    n_efficiency = n / padded(n, cols)
    if fill_cycles is not None:
        # Bit-serial crossbar (D34): M enters in whole row-chunks in parallel,
        # so the M side is area-only; the serial sub-cycle stream rides K with
        # one sub-cycle row of fill: K*s / ((padded K)*s + s) = K / (Kpad + 1).
        m_efficiency = m / padded(m, rows)
        k_efficiency = k / (padded(k, rows) + 1)
    else:
        # MMA unit — a tensor/matrix core, not a systolic pump (D52). It
        # issues fixed instruction tiles (Ampere .f16: m8n8k4, m16n8k8,
        # m16n8k16 — PTX ISA 9.7.15), so M and N slice into INDEPENDENT
        # output tiles dispatched to different cores in parallel. M is a
        # spatial dimension here, not a time one: there is no M-serial
        # pipeline to fill or drain, and the only M-side loss is the ragged
        # last tile. All three axes are therefore rule-of-multiples padding,
        # finishing what D34 started when it called the M=1 loss "area — one
        # active row of 512 — not a pipeline drain".
        m_efficiency = m / padded(m, rows)
        k_efficiency = k / padded(k, rows)
    # How many independent tiles the chip's units have to get through. The
    # default is the weight-stationary grid this function has always assumed;
    # a caller that knows the machine's stationarity passes the real count so
    # this and `pipeline.tile_count` cannot disagree about the decomposition
    # (D53). The three efficiency terms above are unaffected: they measure how
    # each dimension fits an array-sized tile, which is true whichever operand
    # stays resident.
    grid_tiles = (
        tiles if tiles is not None else (padded(k, rows) // rows) * (padded(n, cols) // cols)
    )
    # A caller that knows how the tiles are dealt passes the deal's occupancy
    # (D68): a unit that sums K locally keeps each output column on one unit, so
    # its waves cannot be packed as tightly as tiles/units says.
    waves_term = occupancy if occupancy is not None else wave_occupancy(grid_tiles, units)
    return k_efficiency * n_efficiency * m_efficiency * waves_term


def operation_tiles(
    op: Operation,
    unit: ComputeUnit,
    *,
    stationarity: Dataflow | None = None,
    k_partitions: int = 1,
) -> int:
    """Independent tiles *op* gives *unit*'s arrays to spread across.

    The count :func:`wave_occupancy` divides by ``units``, factored out so that
    the figures can ask a second question of it: **how many arrays ever receive
    a tile at all**. A matmul with two tiles occupies two of A100's 432 tensor
    cores and the other 430 never start, which is a different statement from
    "occupancy is 0.5%" and the one a resource row has to make (D30).

    Attention multiplies by ``batch x heads`` because every (batch, head) pair
    is a genuinely separate GEMM that fills the arrays alongside the others.

    1 where there is no array geometry to tile against, and for work the arrays
    do not run at all — norms and elementwise operations go to the vector unit
    (D27), which this model does not decompose into tiles.
    """
    if unit.systolic_dims is None:
        return 1
    rows, cols = unit.systolic_dims
    if isinstance(op.attrs, MatmulAttrs):
        flow = stationarity if stationarity is not None else unit.dataflow
        return grid_for(flow, op.attrs, rows, cols, k_partitions=k_partitions).tiles
    if isinstance(op.attrs, ConvAttrs):
        # im2col view, as in `operation_utilisation`: K is the filter volume and
        # N the output channels, so the grid is the same K x N one `ws` implies.
        conv = op.attrs
        k = (conv.in_channels // conv.groups) * conv.kernel_h * conv.kernel_w
        return (padded(k, rows) // rows) * (padded(conv.out_channels, cols) // cols)
    if isinstance(op.attrs, AttentionAttrs):
        attention = op.attrs
        per_head = (padded(attention.head_dim, rows) // rows) * (
            padded(attention.kv_len, cols) // cols
        )
        return per_head * max(attention.batch * attention.heads, 1)
    return 1


def operation_utilisation(
    op: Operation,
    unit: ComputeUnit,
    dtype: DType | None = None,
    *,
    stationarity: Dataflow | None = None,
    k_partitions: int = 1,
) -> float:
    """Shape-induced utilisation for *op* on *unit*.

    Returns 1.0 when the profile declares no array geometry — there is then no
    basis for a tail-effect claim, and inventing one would be worse than
    omitting it. Non-GEMM operations also return 1.0: they are memory-bound by
    two orders of magnitude, so their compute term never binds and refining it
    would be effort spent where it cannot matter.

    *dtype* selects the physical model (D34, D52): a dtype the unit
    bit-serialises (multiplier < 1) marks the array as a combinational
    crossbar — area-only on M, sub-cycle stream on K — while ``None`` (or a
    full-rate dtype) marks it an MMA unit, which pads on every axis because
    it issues fixed instruction tiles and has no serial pipeline at all.
    Neither model has an M-serial pipeline any more; both reproduce the
    batch-1 ~1/rows golden, and they differ only in whether K carries a
    sub-cycle fill.

    *stationarity* decides which dimensions form the parallel grid, and so how
    many tiles the units must get through — the wave-occupancy term (D53). It
    defaults to the unit's own declared dataflow. The three padding terms do
    not depend on it: they measure how each dimension fits an array-sized
    tile, which holds whichever operand stays resident.
    """
    if unit.systolic_dims is None:
        return 1.0
    rows, cols = unit.systolic_dims
    multiplier = unit.dtype_multipliers.get(dtype, 1.0) if dtype is not None else 1.0
    # Only a bit-serial (multiplier < 1) array is a crossbar: it gets the D34
    # branch. A full-rate unit passes fill_cycles=None and keeps the systolic
    # tail, even when dtype would over-ride the multiplier (int8 on A100 is
    # 2x, not serial — both operands still enter per instruction).
    fill = round(1 / multiplier) if 0 < multiplier < 1 else None

    if isinstance(op.attrs, MatmulAttrs):
        return systolic_utilisation(
            op.attrs.m,
            op.attrs.k,
            op.attrs.n,
            rows,
            cols,
            units=unit.count,
            fill_cycles=fill,
            tiles=operation_tiles(op, unit, stationarity=stationarity, k_partitions=k_partitions),
            occupancy=_occupancy(op, unit, stationarity=stationarity, k_partitions=k_partitions),
        )

    if isinstance(op.attrs, ConvAttrs):
        # im2col view: M is the output pixel count, K the filter volume,
        # N the output channels.
        conv = op.attrs
        m = conv.batch * conv.out_height * conv.out_width
        k = (conv.in_channels // conv.groups) * conv.kernel_h * conv.kernel_w
        return systolic_utilisation(
            m,
            k,
            conv.out_channels,
            rows,
            cols,
            units=unit.count,
            fill_cycles=fill,
            tiles=operation_tiles(op, unit),
            occupancy=_occupancy(op, unit),
        )

    if isinstance(op.attrs, AttentionAttrs):
        # Two GEMMs per head: [q_len, head_dim] x [head_dim, kv_len] and back.
        # Every (batch, head) pair is an independent GEMM, so they fill the
        # arrays alongside each other — counting only one head's tiles would
        # report a 64-head attention as leaving a 4-core NPU idle.
        attention = op.attrs
        return systolic_utilisation(
            attention.q_len,
            attention.head_dim,
            attention.kv_len,
            rows,
            cols,
            units=unit.count,
            fill_cycles=fill,
            tiles=operation_tiles(op, unit),
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


def operation_grid(
    op: Operation,
    unit: ComputeUnit,
    *,
    stationarity: Dataflow | None = None,
    k_partitions: int = 1,
) -> TileGrid | None:
    """The tile grid *op* is decomposed into on *unit*, or None where there is none.

    A matmul gets the grid its stationarity implies. A convolution gets the
    im2col ``K x N`` weight grid :func:`operation_tiles` has always counted —
    ``M`` output pixels, ``K`` the filter volume, ``N`` the output channels.
    Attention is ``batch x heads`` separate grids, not one, and has none here.
    """
    if unit.systolic_dims is None:
        return None
    rows, cols = unit.systolic_dims
    if isinstance(op.attrs, MatmulAttrs):
        flow = stationarity if stationarity is not None else unit.dataflow
        return grid_for(flow, op.attrs, rows, cols, k_partitions=k_partitions)
    if isinstance(op.attrs, ConvAttrs):
        conv = op.attrs
        k = (conv.in_channels // conv.groups) * conv.kernel_h * conv.kernel_w
        m = conv.batch * conv.out_height * conv.out_width
        attrs = MatmulAttrs(m=m, n=conv.out_channels, k=k)
        return grid_for(Dataflow.WEIGHT_STATIONARY, attrs, rows, cols)
    return None


def operation_deal(
    op: Operation,
    unit: ComputeUnit,
    *,
    stationarity: Dataflow | None = None,
    k_partitions: int = 1,
) -> Deal | None:
    """How *op*'s tiles are dealt to *unit*'s arrays (D68), or None with no grid."""
    grid = operation_grid(op, unit, stationarity=stationarity, k_partitions=k_partitions)
    return deal_for(grid, unit) if grid is not None else None


def _occupancy(
    op: Operation,
    unit: ComputeUnit,
    *,
    stationarity: Dataflow | None = None,
    k_partitions: int = 1,
) -> float | None:
    dealt = operation_deal(op, unit, stationarity=stationarity, k_partitions=k_partitions)
    return dealt.occupancy if dealt is not None else None


def operation_cores(
    op: Operation,
    unit: ComputeUnit,
    *,
    stationarity: Dataflow | None = None,
    k_partitions: int = 1,
) -> int:
    """Arrays *op* ever gives a tile to — the deal's, where there is one (D68).

    ``min(count, tiles)`` for everything the round-robin deal covers; fewer
    when a unit that sums K locally keeps each output column on one array and
    there are fewer columns than arrays.
    """
    dealt = operation_deal(op, unit, stationarity=stationarity, k_partitions=k_partitions)
    if dealt is not None:
        return dealt.used_cores
    return min(
        unit.count,
        operation_tiles(op, unit, stationarity=stationarity, k_partitions=k_partitions),
    )
