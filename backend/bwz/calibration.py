"""All empirical constants used by the engine, in one place.

Every constant in this module MUST carry a comment citing its source: a vendor
datasheet, a published paper, or a fitted dataset documented in
``docs/CALIBRATION.md``. Inline fudge factors anywhere else in the codebase are
a review-rejectable offence (see CLAUDE.md, non-negotiable convention #2).

This module is intentionally empty at M0. Constants land here together with the
analysis code that consumes them, starting at M3.
"""

from __future__ import annotations
