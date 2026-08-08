"""Operator cost models, one file per family, registered on import.

Importing this package registers every shipped model, so ``cost_of`` works
without the caller knowing which module defines what. A new family is a new file
plus a ``@register_op`` decorator plus an import line here.

These models are hardware-independent by construction: they report arithmetic and
compulsory traffic. Anything that depends on the chip — tile re-reads, cache
reuse, the systolic tail effect — is ``analysis/``'s job at M3
(``docs/CORRECTIONS.md`` D10).
"""

from __future__ import annotations

from bwz.operators import attention, conv, custom, elementwise, matmul, normalization
from bwz.operators.base import (
    ZERO_COST,
    OpCost,
    OperatorCostModel,
    cost_model_for,
    cost_of,
    register_op,
)

__all__ = [
    "ZERO_COST",
    "OpCost",
    "OperatorCostModel",
    "attention",
    "conv",
    "cost_model_for",
    "cost_of",
    "custom",
    "elementwise",
    "matmul",
    "normalization",
    "register_op",
]
