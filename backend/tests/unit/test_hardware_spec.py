"""Golden tests for peak throughput — the one place MACs become operations.

Every expected value below is hand-computed in the test docstring, per CLAUDE.md #7.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bwz.spec import DType, load_chip
from bwz.spec.hardware_spec import ComputeUnit, HardwareSpec, MemoryLevel


def _minimal_chip(**overrides: object) -> dict[str, object]:
    document: dict[str, object] = {
        "id": "test_chip",
        "name": "Test Chip",
        "vendor": "Test",
        "source_url": "https://example.com/datasheet",
        "clock_ghz": 1.0,
        "compute_units": [
            {
                "name": "tc",
                "count": 1,
                "ops_per_cycle_per_unit": 1000,
                "supported_dtypes": ["fp16"],
            }
        ],
        "memory": [
            {"name": "SRAM", "level": 1, "capacity_bytes": 1e6, "bandwidth_bytes_per_s": 1e12},
            {"name": "DRAM", "level": 2, "capacity_bytes": 1e9, "bandwidth_bytes_per_s": 1e11},
        ],
    }
    document.update(overrides)
    return document


# -- the MAC x 2 rule -------------------------------------------------------


def test_mac_to_op_doubling_happens_exactly_once() -> None:
    """1 unit x 1000 MACs/cycle x 1 GHz x 1.0 = 1e12 MAC/s = **2e12** OP/s."""
    unit = ComputeUnit.model_validate(
        {"name": "tc", "count": 1, "ops_per_cycle_per_unit": 1000, "supported_dtypes": ["fp16"]}
    )
    assert unit.peak_flops_per_s(1e9, DType.FP16) == pytest.approx(2e12)


def test_dtype_multiplier_and_sparsity_compose() -> None:
    """1000 MACs x 1 GHz x 2.0 (int8) x 2.0 (2:4 sparsity) x 2 = 8e12 OP/s."""
    unit = ComputeUnit.model_validate(
        {
            "name": "tc",
            "count": 1,
            "ops_per_cycle_per_unit": 1000,
            "supported_dtypes": ["fp16", "int8"],
            "dtype_multipliers": {"int8": 2.0},
            "structured_sparsity_speedup": 2.0,
        }
    )
    assert unit.peak_flops_per_s(1e9, DType.INT8, sparsity=True) == pytest.approx(8e12)
    assert unit.peak_flops_per_s(1e9, DType.INT8) == pytest.approx(4e12)


def test_unsupported_dtype_on_a_unit_contributes_nothing() -> None:
    unit = ComputeUnit.model_validate(
        {"name": "tc", "count": 1, "ops_per_cycle_per_unit": 1000, "supported_dtypes": ["fp16"]}
    )
    assert unit.peak_flops_per_s(1e9, DType.INT8) == 0.0


# -- shipped-profile goldens ------------------------------------------------


@pytest.mark.parametrize(
    ("chip_id", "dtype", "expected_flops_per_s", "derivation"),
    [
        # 132 SMs x 4 tensor cores = 528, x 512 MACs/cycle x 1.83 GHz x 2 = 9.894e14
        ("h100_sxm", DType.FP16, 9.894e14, "528 x 512 x 1.83e9 x 2"),
        # 108 SMs x 4 = 432 tensor cores x 256 MACs x 1.41 GHz x 2 = 3.120e14
        ("a100_80gb", DType.FP16, 3.120e14, "432 x 256 x 1.41e9 x 2"),
        # 304 CUs x 1024 MACs x 2.1 GHz x 2 = 1.3075e15
        ("mi300x", DType.FP16, 1.3074e15, "304 x 1024 x 2.1e9 x 2"),
        # 64 tensor cores x 256 MACs x 1.3 GHz x 2 = 4.26e13 FP16 dense
        ("jetson_orin", DType.FP16, 4.260e13, "64 x 256 x 1.3e9 x 2"),
        # 4 cores x 512x512 MACs x 0.8 GHz x 0.125 (bit-serial) x 2 = 2.097e14
        ("chip_a", DType.INT8, 2.0972e14, "4 x 262144 x 0.8e9 x 0.125 x 2"),
        # one quarter of chip_a
        ("chip_b", DType.INT8, 5.2429e13, "1 x 262144 x 0.8e9 x 0.125 x 2"),
    ],
)
def test_profile_peak_matches_datasheet(
    chip_id: str, dtype: DType, expected_flops_per_s: float, derivation: str
) -> None:
    chip = load_chip(chip_id)
    assert chip.peak_flops_per_s(dtype) == pytest.approx(expected_flops_per_s, rel=1e-3), derivation


def test_int8_is_never_slower_than_fp16() -> None:
    """CLAUDE.md sanity check: a chip supporting both must not regress on int8."""
    for chip_id in ("h100_sxm", "a100_80gb", "mi300x", "jetson_orin"):
        chip = load_chip(chip_id)
        assert chip.peak_flops_per_s(DType.INT8) >= chip.peak_flops_per_s(DType.FP16)


@pytest.mark.parametrize(
    ("chip_id", "dtype", "expected_ridge"),
    [
        # 209.6 TOPS / 34 GB/s = 6165 OP/byte (docs/CORRECTIONS.md D8)
        ("chip_a", DType.INT8, 6165.0),
        # 52.4 TOPS / 34 GB/s = 1541 OP/byte
        ("chip_b", DType.INT8, 1541.0),
    ],
)
def test_ridge_point(chip_id: str, dtype: DType, expected_ridge: float) -> None:
    chip = load_chip(chip_id)
    assert chip.ridge_point_flops_per_byte(dtype) == pytest.approx(expected_ridge, rel=2e-3)


def test_peak_takes_the_max_unit_not_the_sum() -> None:
    """H100's CUDA cores also do fp16; the headline figure is the tensor cores alone."""
    chip = load_chip("h100_sxm")
    tensor = next(u for u in chip.compute_units if u.name == "tensor_core")
    assert chip.peak_flops_per_s(DType.FP16) == pytest.approx(
        tensor.peak_flops_per_s(chip.clock_hz, DType.FP16)
    )


def test_unsupported_dtype_on_a_chip_is_an_actionable_error() -> None:
    chip = load_chip("chip_a")
    with pytest.raises(ValueError, match=r"no compute unit supporting dtype 'fp16'"):
        chip.peak_flops_per_s(DType.FP16)


# -- memory accessors -------------------------------------------------------


def test_dram_is_deepest_and_on_chip_is_shallowest() -> None:
    chip = load_chip("h100_sxm")
    assert chip.dram.name == "HBM3"
    assert chip.on_chip.name == "L1"
    # 33.792 MB L1 + 50 MB L2, DRAM excluded
    assert chip.on_chip_capacity_bytes == pytest.approx(8.3792e7)


def test_single_level_chip_reports_no_on_chip_capacity() -> None:
    """A chip with only DRAM must report 0 on-chip bytes, not count DRAM twice."""
    document = _minimal_chip(
        memory=[{"name": "DRAM", "level": 1, "capacity_bytes": 1e9, "bandwidth_bytes_per_s": 1e11}]
    )
    chip = HardwareSpec.model_validate(document)
    assert chip.on_chip_capacity_bytes == 0.0
    assert chip.dram is chip.on_chip


# -- validation errors name the field, the value and the allowed set --------


def test_memory_levels_must_be_ordered() -> None:
    document = _minimal_chip(
        memory=[
            {"name": "DRAM", "level": 2, "capacity_bytes": 1e9, "bandwidth_bytes_per_s": 1e11},
            {"name": "SRAM", "level": 1, "capacity_bytes": 1e6, "bandwidth_bytes_per_s": 1e12},
        ]
    )
    with pytest.raises(ValidationError, match=r"innermost-first, got \[2, 1\]"):
        HardwareSpec.model_validate(document)


def test_duplicate_memory_levels_rejected() -> None:
    document = _minimal_chip(
        memory=[
            {"name": "A", "level": 1, "capacity_bytes": 1e6, "bandwidth_bytes_per_s": 1e12},
            {"name": "B", "level": 1, "capacity_bytes": 1e9, "bandwidth_bytes_per_s": 1e11},
        ]
    )
    with pytest.raises(ValidationError, match=r"duplicate memory level"):
        HardwareSpec.model_validate(document)


def test_dtype_multiplier_for_unsupported_dtype_rejected() -> None:
    document = _minimal_chip(
        compute_units=[
            {
                "name": "tc",
                "count": 1,
                "ops_per_cycle_per_unit": 1000,
                "supported_dtypes": ["fp16"],
                "dtype_multipliers": {"int8": 2.0},
            }
        ]
    )
    with pytest.raises(ValidationError, match=r"not in supported_dtypes"):
        HardwareSpec.model_validate(document)


def test_real_product_requires_source_url() -> None:
    document = _minimal_chip()
    del document["source_url"]
    with pytest.raises(ValidationError, match=r"source_url is required"):
        HardwareSpec.model_validate(document)


def test_hypothetical_profile_may_omit_source_url() -> None:
    document = _minimal_chip(hypothetical=True)
    del document["source_url"]
    assert HardwareSpec.model_validate(document).hypothetical


def test_derived_from_requires_hypothetical() -> None:
    document = _minimal_chip(derived_from="other_chip")
    with pytest.raises(ValidationError, match=r"only meaningful on a hypothetical profile"):
        HardwareSpec.model_validate(document)


def test_estimates_must_name_real_fields() -> None:
    document = _minimal_chip(estimates={"bandwidth": "typo for a real field"})
    pattern = r"estimates names unknown field\(s\) \['bandwidth'\]"
    with pytest.raises(ValidationError, match=pattern):
        HardwareSpec.model_validate(document)


def test_estimates_accept_dotted_paths() -> None:
    document = _minimal_chip(estimates={"memory.0.bandwidth_bytes_per_s": "unpublished"})
    assert HardwareSpec.model_validate(document).estimates


def test_bandwidth_fields_are_float_not_int() -> None:
    """CLAUDE.md gotcha: an int bandwidth silently integer-divides downstream."""
    level = MemoryLevel.model_validate(
        {
            "name": "DRAM",
            "level": 1,
            "capacity_bytes": 1_000_000,
            "bandwidth_bytes_per_s": 1_000_000,
        }
    )
    assert isinstance(level.bandwidth_bytes_per_s, float)
    assert isinstance(level.capacity_bytes, float)


def test_human_units_are_accepted() -> None:
    level = MemoryLevel.model_validate(
        {
            "name": "HBM3",
            "level": 3,
            "capacity_bytes": "80 GB",
            "bandwidth_bytes_per_s": "3.35 TB/s",
        }
    )
    assert level.capacity_bytes == pytest.approx(8e10)
    assert level.bandwidth_bytes_per_s == pytest.approx(3.35e12)


def test_unsigned_exponent_string_is_read_as_a_number() -> None:
    """YAML 1.1 loads ``3.35e12`` as a string; treating it as a unit-less SI value
    is the only reading that is not a silent bug."""
    level = MemoryLevel.model_validate(
        {"name": "HBM3", "level": 3, "capacity_bytes": "8.0e10", "bandwidth_bytes_per_s": "3.35e12"}
    )
    assert level.bandwidth_bytes_per_s == pytest.approx(3.35e12)


def test_unknown_key_is_rejected_by_name() -> None:
    document = _minimal_chip(clock_ghz_typo=1.0)
    with pytest.raises(ValidationError, match=r"clock_ghz_typo"):
        HardwareSpec.model_validate(document)
