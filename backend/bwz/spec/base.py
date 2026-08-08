"""Shared pydantic configuration and validation-error formatting for all specs.

Every spec model is frozen (specs are value objects — the analysis core is pure,
CLAUDE.md #3) and forbids unknown keys, so a typo in a YAML profile is an error
that names the offending key rather than a silently ignored field.
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

_URL_RE = re.compile(r"^https?://\S+$")

SourceUrl = Annotated[str, Field(pattern=_URL_RE.pattern)]
"""An ``http(s)://`` citation. Stored as ``str`` rather than pydantic's ``HttpUrl``
so that YAML round-trips byte-for-byte — ``HttpUrl`` normalises trailing slashes."""

Identifier = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_]*$", max_length=64)]
"""A profile id: lowercase, digits and underscores. Matches the YAML filename stem."""


class SpecModel(BaseModel):
    """Base for every spec type: frozen, strict about unknown keys."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        use_enum_values=False,
        populate_by_name=True,
    )
