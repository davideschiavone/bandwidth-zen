"""SI unit helpers: parse human-readable quantities, format SI values.

Internal convention (CLAUDE.md #1): the engine works exclusively in SI base
units — FLOPs, bytes, bytes/s, seconds, joules, watts. Human-readable strings
("3.35 TB/s", "80 GB", "2 us") exist only in user-authored YAML and in
formatted output, and this module is the only place conversions happen.

Decimal SI prefixes follow datasheet convention (1 GB = 1e9 bytes); binary
prefixes (GiB = 2**30) are accepted on parse for user convenience.
"""

from __future__ import annotations

import re

# Multiplier prefixes, case-sensitive where SI is (m = milli, M = mega).
_PREFIXES: dict[str, float] = {
    "": 1.0,
    "k": 1e3,
    "K": 1e3,  # non-SI but ubiquitous in datasheets
    "M": 1e6,
    "G": 1e9,
    "T": 1e12,
    "P": 1e15,
    "E": 1e18,
    "m": 1e-3,
    "u": 1e-6,
    "µ": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
    "Ki": 2.0**10,
    "Mi": 2.0**20,
    "Gi": 2.0**30,
    "Ti": 2.0**40,
    "Pi": 2.0**50,
}

# Base units the engine understands. Values are the canonical SI symbol used
# in error messages and formatting.
_BASE_UNITS: frozenset[str] = frozenset(
    {"B", "B/s", "bit", "bit/s", "FLOP", "FLOP/s", "OP", "OP/s", "s", "W", "J", "Hz"}
)

_QUANTITY_RE = re.compile(r"^\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*([^\s]+)\s*$")


class UnitParseError(ValueError):
    """Raised when a quantity string cannot be parsed; message names the input."""


def parse_quantity(text: str, expected_base_unit: str) -> float:
    """Parse ``"3.35 TB/s"`` (with ``expected_base_unit="B/s"``) into ``3.35e12``.

    The unit in *text* must end with *expected_base_unit*; anything before it
    is interpreted as an SI or binary prefix. Raises :class:`UnitParseError`
    naming the input, the offending part, and the accepted values.
    """
    if expected_base_unit not in _BASE_UNITS:
        raise UnitParseError(
            f"unknown base unit {expected_base_unit!r}; expected one of {sorted(_BASE_UNITS)}"
        )
    match = _QUANTITY_RE.match(text)
    if match is None:
        raise UnitParseError(
            f"cannot parse quantity {text!r}: expected '<number> <unit>', "
            f"e.g. '3.35 T{expected_base_unit}'"
        )
    value = float(match.group(1))
    unit = match.group(2)
    if not unit.endswith(expected_base_unit):
        raise UnitParseError(
            f"cannot parse {text!r}: unit {unit!r} does not end with the expected "
            f"base unit {expected_base_unit!r}"
        )
    prefix = unit[: len(unit) - len(expected_base_unit)]
    if prefix not in _PREFIXES:
        raise UnitParseError(
            f"cannot parse {text!r}: unknown prefix {prefix!r}; accepted prefixes are "
            f"{sorted(p for p in _PREFIXES if p)}"
        )
    return value * _PREFIXES[prefix]


def parse_or_pass(value: str | float | int, expected_base_unit: str) -> float:
    """Accept either a bare number (already SI) or a quantity string.

    YAML profiles may write ``bandwidth_bytes_per_s: 3.35e12`` or
    ``bandwidth_bytes_per_s: "3.35 TB/s"``; both mean the same thing.
    """
    if isinstance(value, str):
        return parse_quantity(value, expected_base_unit)
    return float(value)


# Formatting ---------------------------------------------------------------

_DECIMAL_STEPS: list[tuple[float, str]] = [
    (1e18, "E"),
    (1e15, "P"),
    (1e12, "T"),
    (1e9, "G"),
    (1e6, "M"),
    (1e3, "k"),
]

_SUB_STEPS: list[tuple[float, str]] = [
    (1.0, ""),
    (1e-3, "m"),
    (1e-6, "µ"),
    (1e-9, "n"),
    (1e-12, "p"),
]


def format_quantity(value_si: float, base_unit: str, precision: int = 3) -> str:
    """Format an SI value with the nearest sensible prefix: ``3.35e12 -> '3.35 TB/s'``."""
    if value_si == 0.0:
        return f"0 {base_unit}"
    magnitude = abs(value_si)
    if magnitude >= 1e3:
        for factor, prefix in _DECIMAL_STEPS:
            if magnitude >= factor:
                return f"{value_si / factor:.{precision}g} {prefix}{base_unit}"
    if magnitude < 1.0:
        for factor, prefix in _SUB_STEPS:
            if magnitude >= factor:
                return f"{value_si / factor:.{precision}g} {prefix}{base_unit}"
        factor, prefix = _SUB_STEPS[-1]
        return f"{value_si / factor:.{precision}g} {prefix}{base_unit}"
    return f"{value_si:.{precision}g} {base_unit}"


def format_bytes(n_bytes: float, precision: int = 3) -> str:
    """``17.4e9 -> '17.4 GB'`` (decimal, datasheet convention)."""
    return format_quantity(n_bytes, "B", precision)


def format_bandwidth(bytes_per_s: float, precision: int = 3) -> str:
    """``3.35e12 -> '3.35 TB/s'``."""
    return format_quantity(bytes_per_s, "B/s", precision)


def format_flops_per_s(flops_per_s: float, precision: int = 3) -> str:
    """``9.89e14 -> '989 TFLOP/s'``."""
    return format_quantity(flops_per_s, "FLOP/s", precision)


def format_time(latency_s: float, precision: int = 3) -> str:
    """``0.0217 -> '21.7 ms'``."""
    return format_quantity(latency_s, "s", precision)


def format_energy(energy_j: float, precision: int = 3) -> str:
    """``0.42 -> '420 mJ'``."""
    return format_quantity(energy_j, "J", precision)
