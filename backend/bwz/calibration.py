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


# Fraction of nominal DRAM bandwidth a real access stream achieves: refresh,
# bank conflicts, read/write turnaround and imperfect access patterns. 0.85 sits
# in the 0.8-0.9 band that PLAN.md Session 4 names as the documented starting
# point for HBM. It is a datacenter-GPU figure and has no business being applied
# to an LPDDR estimate that is itself a guess - profiles that would be
# double-derated override it to 1.0 with a note (docs/CORRECTIONS.md D6).
# TO BE FITTED: docs/PLAN.md Session 5.
DEFAULT_DRAM_BANDWIDTH_EFFICIENCY: float = 0.85

# Fraction of peak arithmetic a real kernel sustains once instruction issue,
# scheduling and thermal limits are accounted for - but NOT operand shape, which
# the tail-effect model in analysis/tiling.py handles separately. The two are
# multiplicative and must not be conflated: 0.7 x a 1/512 tail effect is 0.14%,
# and that is the correct reading for a batch-1 GEMM on a 512x512 array.
# TO BE FITTED: docs/PLAN.md Session 5.
DEFAULT_ACHIEVED_FLOPS_FRACTION: float = 0.70


# -- op-count conventions ---------------------------------------------------
#
# These are not measurements and never will be: they are how many operations we
# agree to charge for arithmetic whose exact count depends on the kernel. They
# live here rather than inline because CLAUDE.md #2 bans magic numbers in
# operators/, and because a reader deserves to find every debatable constant in
# one place. None of them changes a bottleneck verdict - a softmax charged at 4
# instead of 5 FLOPs per score moves total prefill FLOPs by well under 1%.

# exp, running max subtract, sum accumulate, and the final divide.
# These per-element counts are *algebraic* operations, not machine instructions.
# A GELU is 8 arithmetic operations on paper; on hardware with no special-function
# unit its erf is a polynomial approximation costing many more ALU ops, and on
# hardware with one it runs at a fraction of the ALU rate. Neither figure is
# declared by any shipped profile, so every non-linear cost below is a lower
# bound (docs/CORRECTIONS.md D27).
SOFTMAX_FLOPS_PER_SCORE: float = 5.0

# square, sum-accumulate, rsqrt (amortised), scale multiply.
RMSNORM_FLOPS_PER_ELEMENT: float = 4.0

# RMSNorm's four plus a mean accumulate and a subtract.
LAYERNORM_FLOPS_PER_ELEMENT: float = 6.0

# One compare (max pool) or one accumulate (average pool) per window element.
POOL_FLOPS_PER_WINDOW_ELEMENT: float = 1.0

# sigmoid (exp, add, reciprocal) plus the gating multiply.
SILU_FLOPS_PER_ELEMENT: float = 4.0

# tanh approximation: two multiplies, a cube, a tanh, an add and two scales.
GELU_FLOPS_PER_ELEMENT: float = 8.0

# Rotary embedding: a cos multiply, a sin multiply and an add per element.
ROPE_FLOPS_PER_ELEMENT: float = 3.0
