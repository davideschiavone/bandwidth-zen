"""All empirical constants used by the engine, in one place.

Every constant in this module MUST carry a comment citing its source: a vendor
datasheet, a published paper, or a fitted dataset documented in
``docs/CALIBRATION.md``. Inline fudge factors anywhere else in the codebase are
a review-rejectable offence (see CLAUDE.md, non-negotiable convention #2).

**Nothing here has been fitted yet.** These are documented starting defaults;
Session 5 of ``docs/PLAN.md`` fits them against published reference points on
H100/A100/MI300X and rewrites this module with the fitted values and their
dataset. Until then every number below is a placeholder that reports must
disclose, and no result derived from them should be quoted as a prediction.

Per ``docs/CORRECTIONS.md`` D6, a chip profile may override any of these with a
per-chip value, but only when that value carries a ``source_url`` or an entry in
the profile's ``estimates`` block.
"""

from __future__ import annotations

# Fraction of a device's nominal memory capacity available to model weights,
# activations and KV cache. The remainder is runtime context, allocator
# fragmentation and workspace. 0.90 is the figure PROMPT.md 4.1 carries in its
# worked chip profile; it is not a measurement.
# TO BE FITTED: docs/PLAN.md Session 5.
DEFAULT_USABLE_MEMORY_FRACTION: float = 0.90

# Non-overlappable fixed cost per dispatched operation: command submission,
# descriptor setup, and pipeline fill/drain that double buffering cannot hide.
# 3 us is PROMPT.md 4.1's worked value for a datacenter GPU. It is the term that
# makes small-batch decode latency-bound (CLAUDE.md sanity checks), and on a
# fully-resident model it is the *only* term left once DRAM traffic goes to zero
# (docs/CORRECTIONS.md D5a) - so results in that regime are as uncertain as this
# number is.
# TO BE FITTED: docs/PLAN.md Session 5.
DEFAULT_KERNEL_LAUNCH_OVERHEAD_S: float = 3.0e-6
