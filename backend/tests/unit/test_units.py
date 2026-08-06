"""Golden tests for bwz.units — every expected value hand-computed.

E.g. "3.35 TB/s" = 3.35 × 1e12 = 3.35e12 bytes/s; "80 GiB" = 80 × 2**30
= 85 899 345 920 bytes; "2 us" = 2e-6 s.
"""

from __future__ import annotations

import pytest

from bwz.units import (
    UnitParseError,
    format_bandwidth,
    format_bytes,
    format_flops_per_s,
    format_time,
    parse_or_pass,
    parse_quantity,
)


class TestParseQuantity:
    def test_hbm3_bandwidth(self) -> None:
        assert parse_quantity("3.35 TB/s", "B/s") == pytest.approx(3.35e12)

    def test_capacity_decimal(self) -> None:
        assert parse_quantity("80 GB", "B") == pytest.approx(80e9)

    def test_capacity_binary(self) -> None:
        assert parse_quantity("80 GiB", "B") == 80 * 2**30  # 85_899_345_920

    def test_tensor_flops(self) -> None:
        assert parse_quantity("989 TFLOP/s", "FLOP/s") == pytest.approx(9.89e14)

    def test_no_prefix(self) -> None:
        assert parse_quantity("700 W", "W") == 700.0

    def test_kernel_launch_micro(self) -> None:
        assert parse_quantity("3 us", "s") == pytest.approx(3e-6)
        assert parse_quantity("3 µs", "s") == pytest.approx(3e-6)

    def test_latency_nano(self) -> None:
        assert parse_quantity("600 ns", "s") == pytest.approx(6e-7)

    def test_milli_vs_mega_case_sensitive(self) -> None:
        assert parse_quantity("5 ms", "s") == pytest.approx(5e-3)
        assert parse_quantity("1.755 GHz", "Hz") == pytest.approx(1.755e9)

    def test_scientific_notation_number(self) -> None:
        assert parse_quantity("1.2e13 B/s", "B/s") == pytest.approx(1.2e13)

    def test_no_space(self) -> None:
        assert parse_quantity("30GB", "B") == pytest.approx(30e9)

    def test_wrong_base_unit_rejected(self) -> None:
        with pytest.raises(UnitParseError, match="does not end with"):
            parse_quantity("3.35 TB/s", "B")

    def test_unknown_prefix_rejected(self) -> None:
        with pytest.raises(UnitParseError, match="unknown prefix"):
            parse_quantity("3.35 QB/s", "B/s")

    def test_garbage_rejected(self) -> None:
        with pytest.raises(UnitParseError, match="cannot parse"):
            parse_quantity("fast", "B/s")

    def test_unknown_expected_unit_rejected(self) -> None:
        with pytest.raises(UnitParseError, match="unknown base unit"):
            parse_quantity("3 furlong", "furlong")


class TestParseOrPass:
    def test_bare_float_passes_through(self) -> None:
        assert parse_or_pass(3.35e12, "B/s") == 3.35e12

    def test_bare_int_becomes_float(self) -> None:
        result = parse_or_pass(1_000_000, "B")
        assert result == 1e6
        assert isinstance(result, float)  # guards the int-division gotcha (CLAUDE.md)

    def test_string_is_parsed(self) -> None:
        assert parse_or_pass("3.35 TB/s", "B/s") == pytest.approx(3.35e12)


class TestFormatting:
    def test_bandwidth(self) -> None:
        assert format_bandwidth(3.35e12) == "3.35 TB/s"

    def test_bytes(self) -> None:
        assert format_bytes(17.4e9) == "17.4 GB"

    def test_flops(self) -> None:
        assert format_flops_per_s(9.89e14) == "989 TFLOP/s"

    def test_time_ms(self) -> None:
        assert format_time(0.0217) == "21.7 ms"

    def test_time_us(self) -> None:
        assert format_time(3e-6) == "3 µs"

    def test_zero(self) -> None:
        assert format_bytes(0.0) == "0 B"

    def test_unit_range_no_prefix(self) -> None:
        assert format_time(5.62) == "5.62 s"

    def test_roundtrip(self) -> None:
        value = 4.5e11
        assert parse_quantity(format_bandwidth(value), "B/s") == pytest.approx(value)
