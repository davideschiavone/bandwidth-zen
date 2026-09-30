"""FlashAttention as a decomposition — ``bwz/analysis/flash.py``, D70.

Every golden number is worked by hand in its test's docstring from the profile's
own fields, so a failure says which term moved.
"""

from __future__ import annotations

import pytest

from bwz.analysis import machine_model
from bwz.analysis.flash import FlashShape, inner_cost, local_group_slices, plan_flash
from bwz.report import Bound, ReductionPlacement
from bwz.spec import load_chip
from bwz.spec.dtypes import DType
from bwz.spec.hardware_spec import Dataflow


def _a100_toy() -> FlashShape:
    return FlashShape(batch=1, heads=2, q_len=64, kv_len=64, head_dim=16, dtype=DType.FP16)


def test_the_a100_toy_is_countable_by_hand() -> None:
    """A100 fp16 --ideal, 2 heads, S=64, d=16, pinned Br=16, Bc=32, os.

    Rates, from the profile:
        tensor core   256 MACs x 1.41 GHz x 2       = 7.2192e11 OP/s per unit
        CUDA cores    6912 x 1.41 GHz x 2 / 432     = 4.512e10  OP/s per unit
        HBM2e                                          2.039e12  B/s

    Programs: 2 heads x ceil(64/16) = 8, on 8 of 432 units — one wave.
    One program streams ceil(64/32) = 2 kv blocks. Per block:
        S = Q K^T   [16,16]x[16,32]   2*16*32*16 = 16 384 OP   22.695 ns
        O += P V    [16,32]x[32,16]   16 384 OP                22.695 ns
    so t_matrix = 2 blocks x 45.39 ns = 90.78 ns (nothing pads: all multiples of 16).
    Vector, per program:
        scores      16 x 64 x 5 = 5120
        rescale     16 x 16 x (2 - 1) = 256     block 1 only
        normalise   16 x 16 = 256
        5632 OP / 4.512e10 = 124.82 ns
    t_compute = 215.60 ns.
    DRAM: Q and O 2 x 64 x 16 x 2 B = 4096 B each; K and V once per head (every
    head's programs share the one wave) 4096 B each; 16 384 B / 2.039e12 = 8.035 ns.
    latency = max(8.035 ns, 215.60 ns) = 215.60 ns, COMPUTE_BOUND.
    """
    plan = plan_flash(load_chip("a100_80gb"), _a100_toy(), br=16, bc=32, ideal=True)
    chosen = plan.chosen
    assert chosen is not None
    assert plan.per_unit_matrix_flops_per_s == pytest.approx(7.2192e11)
    assert plan.per_unit_vector_flops_per_s == pytest.approx(4.512e10)
    assert (chosen.programs, chosen.waves, chosen.used_cores) == (8, 1, 8)
    assert chosen.t_matrix_s == pytest.approx(4 * 16_384 / 7.2192e11)
    assert chosen.t_vector_s == pytest.approx(5632 / 4.512e10)
    assert chosen.t_dram_s == pytest.approx(16_384 / 2.039e12)
    assert chosen.latency_s == pytest.approx(215.60e-9, rel=1e-4)
    assert chosen.bound is Bound.COMPUTE_BOUND
    assert (chosen.scores, chosen.rescaled, chosen.normalised) == (8192, 2048, 2048)
    assert chosen.mac_slots == 8 * 2 * 2 * 8192


def test_the_working_set_is_the_worst_wave() -> None:
    """Per program ``Br x (d x 2 B [Q] + d x 4 B [O] + Bc x 4 B [S] + 2 x 4 B [m, l])``
    = 16 x (32 + 64 + 128 + 8) = 3712 B, x 8 programs = 29 696 B. K and V blocks
    ``2 x 32 x 16 x 2 B = 2048 B`` per head, x 2 heads, x 2 buffers = 8192 B.
    37 888 B in all — and it fits a second buffer, so the plan double-buffers."""
    chosen = plan_flash(load_chip("a100_80gb"), _a100_toy(), br=16, bc=32, ideal=True).chosen
    assert chosen is not None
    assert chosen.working_set_bytes == 37_888
    assert chosen.double_buffered


def test_a_head_spanning_two_waves_reads_its_k_and_v_twice() -> None:
    """Metis, int8, 3 heads, S=1100, d=64, pinned Br=Bc=512.

    3 programs per head (ceil(1100/512)), 9 in all, on 4 cores: waves of programs
    0-3, 4-7, 8. Head 0 (0-2) sits in wave 0; head 1 (3-5) spans waves 0 and 1;
    head 2 (6-8) spans waves 1 and 2. So K crosses DRAM 1 + 2 + 2 = 5 times, not 3:
    5 x 1100 x 64 x 1 B = 352 000 B, and the same again for V. On A100 the same
    heads fit one wave and each is read once — a difference between the chips, not
    the kernel.
    """
    shape = FlashShape(batch=1, heads=3, q_len=1100, kv_len=1100, head_dim=64, dtype=DType.INT8)
    chosen = plan_flash(load_chip("metis_aipu"), shape, br=512, bc=512, ideal=True).chosen
    assert chosen is not None
    assert (chosen.programs, chosen.waves, chosen.kv_streams) == (9, 3, 5)
    assert chosen.k_bytes == chosen.v_bytes == 352_000
    assert chosen.q_bytes == chosen.o_bytes == 3 * 1100 * 64


def test_head_dim_64_fills_an_eighth_of_a_512_deep_crossbar() -> None:
    """Metis ``S = Q·Kᵀ`` at [512,64]x[64,512]: M and N fill the array, K = 64 does
    not — a bit-serial crossbar's K term is ``K / (padded(K) + 1) = 64 / 513``
    (D34), 12.48%. At 5.24288e13 OP/s per core that is
    ``2 x 512 x 512 x 64 / (5.24288e13 x 64/513) = 5.130 µs``. K is one slice,
    so there is nothing to reduce."""
    machine = machine_model(load_chip("metis_aipu"), DType.INT8)
    cost = inner_cost(machine, 512, 512, 64, Dataflow.WEIGHT_STATIONARY)
    assert cost.utilisation == pytest.approx(64 / 513)
    assert cost.t_s == pytest.approx(2 * 512 * 512 * 64 / (5.24288e13 * 64 / 513))
    assert cost.placement is ReductionPlacement.NONE


def test_ws_on_a_tensor_core_owes_the_vector_unit_every_slice() -> None:
    """A100 ``S = Q·Kᵀ`` at [64,128]x[128,64] under ws: K = 128 is 8 slices of 16
    and no periphery holds them, so ``(8 - 1) x 64 x 64 = 28 672`` adds. At
    4.512e10 per unit that is 0.635 µs against 1.453 µs of matrix work — hidden,
    so ws ties os on time (D62), and the tie goes to os, the unit's own."""
    machine = machine_model(load_chip("a100_80gb"), DType.FP16)
    ws = inner_cost(machine, 64, 64, 128, Dataflow.WEIGHT_STATIONARY)
    os_ = inner_cost(machine, 64, 64, 128, Dataflow.OUTPUT_STATIONARY)
    assert (ws.partials, ws.vector_adds) == (8, 28_672)
    assert ws.placement is ReductionPlacement.ON_CHIP
    assert ws.t_s == os_.t_s
    shape = FlashShape(batch=1, heads=1, q_len=64, kv_len=64, head_dim=128, dtype=DType.FP16)
    chosen = plan_flash(load_chip("a100_80gb"), shape, br=64, bc=64).chosen
    assert chosen is not None
    assert chosen.qk.chosen.stationarity is Dataflow.OUTPUT_STATIONARY


def test_metis_sums_a_long_kv_block_in_its_periphery() -> None:
    """``O += P·V`` with Bc = 2048 cuts K into 4 slices of 512; Metis's periphery
    holds 16384 / 512 = 32 of them, so all four meet locally, for free (D62)."""
    machine = machine_model(load_chip("metis_aipu"), DType.INT8)
    cost = inner_cost(machine, 512, 64, 2048, Dataflow.WEIGHT_STATIONARY)
    assert (cost.k_slices, cost.partials, cost.vector_adds) == (4, 1, 0)
    assert cost.placement is ReductionPlacement.LOCAL
    assert local_group_slices(machine.unit, Dataflow.WEIGHT_STATIONARY, 2048) == 32


def test_the_chosen_plan_is_the_fastest_that_fits() -> None:
    shape = FlashShape(batch=1, heads=32, q_len=4096, kv_len=4096, head_dim=128, dtype=DType.FP16)
    plan = plan_flash(load_chip("a100_80gb"), shape)
    assert plan.chosen is not None
    fitting = [c for c in plan.candidates if c.fits]
    assert all(plan.chosen.latency_s <= c.latency_s * (1 + 1e-6) for c in fitting)


def test_blocks_change_vector_work_never_the_scores() -> None:
    """Every block size scores every (query, key) pair once — ``B·H·Sq·Skv`` — and
    normalises every output once. What Bc changes is how often O is rescaled:
    ``B·H·Sq·d·(kv_blocks - 1)``. The matrix work never moves (CLAUDE.md:
    FlashAttention changes bytes, never FLOPs — here, never MACs)."""
    shape = FlashShape(batch=1, heads=2, q_len=256, kv_len=256, head_dim=64, dtype=DType.FP16)
    plan = plan_flash(load_chip("a100_80gb"), shape)
    for candidate in plan.candidates:
        assert candidate.scores == 2 * 256 * 256
        assert candidate.normalised == 2 * 256 * 64
        assert candidate.rescaled == 2 * 256 * 64 * (candidate.kv_blocks - 1)


def test_doubling_dram_bandwidth_never_increases_latency() -> None:
    chip = load_chip("metis_aipu")
    faster = chip.model_copy(
        update={
            "memory": [
                *chip.memory[:-1],
                chip.dram.model_copy(
                    update={"bandwidth_bytes_per_s": 2 * chip.dram.bandwidth_bytes_per_s}
                ),
            ]
        }
    )
    shape = FlashShape(batch=1, heads=8, q_len=2048, kv_len=2048, head_dim=64, dtype=DType.INT8)
    slow, fast = plan_flash(chip, shape).chosen, plan_flash(faster, shape).chosen
    assert slow is not None and fast is not None
    assert fast.latency_s <= slow.latency_s


def test_a_chip_without_a_vector_unit_is_refused() -> None:
    """chip_a declares only its array: the softmax's exp and max would run on a unit
    that only multiplies and accumulates, and are refused rather than costed (D62)."""
    shape = FlashShape(batch=1, heads=1, q_len=512, kv_len=512, head_dim=64, dtype=DType.INT8)
    plan = plan_flash(load_chip("chip_a"), shape)
    assert not plan.feasible
    assert "D62" in plan.infeasibility[0]


def test_an_undeclared_stationarity_is_refused_not_clamped() -> None:
    shape = FlashShape(batch=1, heads=1, q_len=512, kv_len=512, head_dim=64, dtype=DType.INT8)
    plan = plan_flash(load_chip("metis_aipu"), shape, stationarity=Dataflow.OUTPUT_STATIONARY)
    assert not plan.feasible
    assert "stationarity='os'" in plan.infeasibility[0]


def test_a_block_that_is_not_whole_tiles_is_refused() -> None:
    plan = plan_flash(load_chip("a100_80gb"), _a100_toy(), br=24)
    assert not plan.feasible
    assert "br=24" in plan.infeasibility[0]
    assert "multiple of 16" in plan.infeasibility[0]


def test_the_assumptions_name_the_imc_write_on_an_in_memory_array() -> None:
    shape = FlashShape(batch=1, heads=1, q_len=512, kv_len=512, head_dim=64, dtype=DType.INT8)
    plan = plan_flash(load_chip("metis_aipu"), shape)
    assert any("WRITTEN into the array" in note for note in plan.assumptions)
    a100 = plan_flash(load_chip("a100_80gb"), _a100_toy())
    assert not any("WRITTEN into the array" in note for note in a100.assumptions)
