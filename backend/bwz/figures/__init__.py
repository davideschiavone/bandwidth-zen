"""Self-contained HTML views of a run: where the time went, and how it moved.

``docs/CORRECTIONS.md`` D55. These live inside the package rather than in
``backend/scripts/`` because they are reached from ``bwz matmul``/``bwz run``/
``bwz encoder-layer`` — one command per workload, the same one that prints the
report — and a module ``bwz.cli`` imports cannot sit in a directory that is not
a package and does not ship in a wheel.

**The plotting-library rule still holds.** CLAUDE.md's "a figure goes in
``scripts/``" exists so the engine never imports matplotlib and ``make test``
never pulls it in. Nothing here imports a plotting library: the timeline and the
flow animation are hand-written SVG and JavaScript in one file each, stdlib only.
``scripts/plot_roofline.py`` — which *does* need matplotlib, and emits PNGs —
stays exactly where it was, in its own optional dependency group.

Both pages are pure functions of a ``Report`` and the schedule it came from:
``timeline_html.render`` and ``dataflow_html.render`` return text and write
nothing, and :mod:`bwz.figures.timeline` is the only thing here that touches a
file.
"""

from __future__ import annotations

from bwz.figures.timeline import (
    Panel,
    Workload,
    build_matmul,
    build_phases,
    shared_dtype,
    write_animation,
    write_timeline,
)

__all__ = [
    "Panel",
    "Workload",
    "build_matmul",
    "build_phases",
    "shared_dtype",
    "write_animation",
    "write_timeline",
]
