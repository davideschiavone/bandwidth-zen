"""Annotated SI field types for pydantic specs.

Every physical quantity in a spec is stored as a plain SI ``float`` (CLAUDE.md
#1) but may be *written* by the user either as a bare number or as a human
string: ``bandwidth_bytes_per_s: 3.35e12`` and ``bandwidth_bytes_per_s:
"3.35 TB/s"`` mean the same thing. The ``BeforeValidator``s here are the only
place that conversion happens on the spec boundary.

Declaring these as ``float`` (never ``int``) is load-bearing: pydantic v2
coerces ``1_000_000`` in YAML to ``int``, and an ``int`` bandwidth silently
integer-divides in later arithmetic (CLAUDE.md gotchas).
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BeforeValidator, Field

from bwz.units import parse_or_pass


def _as_bytes(value: object) -> object:
    return parse_or_pass(value, "B") if isinstance(value, str) else value


def _as_bytes_per_s(value: object) -> object:
    return parse_or_pass(value, "B/s") if isinstance(value, str) else value


def _as_seconds(value: object) -> object:
    return parse_or_pass(value, "s") if isinstance(value, str) else value


def _as_watts(value: object) -> object:
    return parse_or_pass(value, "W") if isinstance(value, str) else value


def _as_hertz(value: object) -> object:
    return parse_or_pass(value, "Hz") if isinstance(value, str) else value


def _as_flops(value: object) -> object:
    return parse_or_pass(value, "FLOP") if isinstance(value, str) else value


Bytes = Annotated[float, BeforeValidator(_as_bytes), Field(gt=0)]
"""A positive size in bytes. Accepts ``80e9``, ``"80 GB"``, ``"75 GiB"``."""

BytesPerSecond = Annotated[float, BeforeValidator(_as_bytes_per_s), Field(gt=0)]
"""A positive bandwidth in bytes/second. Accepts ``3.35e12`` or ``"3.35 TB/s"``."""

Seconds = Annotated[float, BeforeValidator(_as_seconds), Field(ge=0)]
"""A non-negative duration in seconds. Accepts ``3e-6`` or ``"3 us"``."""

Watts = Annotated[float, BeforeValidator(_as_watts), Field(gt=0)]
"""A positive power in watts. Accepts ``700`` or ``"700 W"``."""

Hertz = Annotated[float, BeforeValidator(_as_hertz), Field(gt=0)]
"""A positive frequency in hertz. Accepts ``1.755e9`` or ``"1.755 GHz"``."""

Flops = Annotated[float, BeforeValidator(_as_flops), Field(ge=0)]
"""A non-negative operation count. Accepts ``4.2e12`` or ``"4.2 TFLOP"``."""

Fraction = Annotated[float, Field(gt=0.0, le=1.0)]
"""A fraction in (0, 1] — efficiencies, usable-capacity ratios, sparsity ratios."""
