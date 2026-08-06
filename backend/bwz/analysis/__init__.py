"""Analysis core: roofline, tiling, memory planning, scheduling, bottlenecks.

Pure module: no I/O, no web framework, no global mutable state, no wall-clock
reads (CLAUDE.md #3). Same inputs -> identical output.
"""

from __future__ import annotations
