"""Runtime for the runnable kernel programs :mod:`bwz.emit` writes.

``docs/CORRECTIONS.md`` D54. This module is **inlined by source** into every
emitted program, so there is exactly one copy of it and it is the copy ruff,
``mypy --strict`` and ``tests/unit/test_emit_harness.py`` all see. Writing it as
a string constant would put the part of the feature most likely to be subtly
wrong outside all three.

What it provides is the machinery an emitted program needs and nothing about any
particular decomposition: counted memory (:class:`Dram`), the shared on-chip
staging buffer (:class:`Scratchpad`), the accumulator where a K-on-the-grid
walk's partial sums meet (:class:`Partials`), the thread-per-core starter
(:func:`run_cores`), and one instruction tile (:func:`mma`). **Which tiles exist,
which dimension each sweeps, the waves, and where the result goes are the emitted
``walk()``'s business** — that is the part a reader is meant to read, so none of it is hidden
in here.

Two constraints shape everything below:

* **Nothing imports from ``bwz``.** An emitted program has to run on a machine
  that has never heard of this repository. A unit test asserts it.
* **numpy is optional.** It is not a ``bwz`` dependency — it arrives with
  matplotlib in the ``plots`` group — so every operation has a pure-Python
  fallback and the emitted program prints which backend it took.

It is not a benchmark. Every count here is a count of *bytes and MACs*, never of
seconds; the wall clock of a program built on this has no relationship to any
latency the model predicts.
"""

from __future__ import annotations

import math
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType
from typing import Any


def _load_numpy() -> ModuleType | None:
    """numpy if this machine has it, ``None`` otherwise.

    ``find_spec`` rather than a bare ``import numpy`` in a ``try``: numpy is not
    a dependency of this project, so a plain import at module scope would make
    ``mypy`` fail on a checkout that never installed the optional plotting
    group. The emitted program must type-check and run either way.
    """
    if find_spec("numpy") is None:
        return None
    try:
        return import_module("numpy")
    except ImportError:
        # Present but unimportable: a broken install, or a shadowed one — which
        # is how the pure-Python path gets tested on a machine that has numpy.
        return None


NUMPY = _load_numpy()
BACKEND = "numpy" if NUMPY is not None else "pure Python"

DEBUG = "--debug" in sys.argv
"""Whether to narrate the walk: which core takes which tile, when A is staged,
and every instruction tile issued.

Off by default and guarded at each call site rather than filtered inside
:func:`log`, because the volume is the whole problem: a 1000x2000x3000 matmul
issues 1.5 million instruction tiles, and formatting a line for each one that
nobody reads would dominate the run. ``if DEBUG:`` costs one bool test."""

_LOG_LOCK = threading.Lock()
_LOG = threading.local()


def log(message: str) -> None:
    """One line of ``--debug`` output, held until this core's block is flushed.

    Serialising individual lines is not enough. The narration is *indented* —
    a tile under a core, a k-step under a tile — and that nesting is a claim
    about which line belongs to which. Four cores printing line by line put a
    ``kt=`` under a ``tile`` header belonging to a different core, and the
    indentation then says something false (D61).

    So a line goes into the current core's buffer and the whole block is printed
    at once. Blocks still appear in completion order, which is honest: they
    genuinely run at the same time. Lines within one no longer do.
    """
    lines = getattr(_LOG, "lines", None)
    if lines is None:
        with _LOG_LOCK:
            print(message)
    else:
        lines.append(message)


def log_block(header: str) -> None:
    """Open a block on this thread; every :func:`log` from it joins the block."""
    _LOG.lines = [header]


def log_flush() -> None:
    """Print this thread's block as one unit, and close it."""
    lines = getattr(_LOG, "lines", None)
    _LOG.lines = None
    if lines:
        with _LOG_LOCK:
            print("\n".join(lines))


Tile = Any
"""One block of numbers: a 2-D numpy array where numpy is present, a list of
rows where it is not.

Deliberately ``Any`` — the one place outside ``spec/loaders.py`` this repository
allows it (CLAUDE.md style). There is no static type covering both backends, and
a ``cast`` to ``list[list[float]]`` over a numpy array would type-check while
being false. Both backends are real: an emitted program runs on machines that
have never installed numpy."""

HOST_DTYPE: dict[str, str] = {
    "fp32": "float32",
    "tf32": "float32",
    "fp16": "float16",
    "bf16": "float32",
    "fp8": "float16",
    "int8": "int8",
    "int4": "int8",
    "int32": "int32",
}
"""How a modelled dtype is *held on the host*. **Not** the modelled width: every
byte count in an emitted program comes from the width constants at the top of
the file, which carry the chip's own numbers. The host has no bfloat16, fp8 or
int4, so those are held in the narrowest type that represents their values
exactly, and nothing downstream depends on the host width."""


def is_integer_dtype(dtype: str) -> bool:
    """True for the integer formats, whose arithmetic here is exact."""
    return dtype.startswith("int")


def zeros(rows: int, cols: int, dtype: str) -> Tile:
    """A ``rows x cols`` block of zeros held at *dtype*'s host representation."""
    host = HOST_DTYPE[dtype]
    if NUMPY is not None:
        return NUMPY.zeros((rows, cols), dtype=host)
    fill: float | int = 0 if is_integer_dtype(dtype) else 0.0
    return [[fill] * cols for _ in range(rows)]


def operand(rows: int, cols: int, dtype: str, seed: int) -> Tile:
    """A deterministic ``rows x cols`` operand.

    Values are drawn from ``{-7 .. 7}`` — divided by 8 for a floating dtype, so
    every element is a **dyadic** number both the operand format and the
    accumulator hold exactly. That is on purpose: with exact operands, every
    product and every partial sum is exact too, so a mismatch at the end of the
    run is a *walk* error and never a rounding one. This program exists to check
    the decomposition; rounding is a different question, and mixing the two would
    make a failure ambiguous.

    The seeded values differ between the two backends (numpy's generator against
    the small LCG below), which nothing depends on: each run checks its own
    operands against its own reference.
    """
    integer = is_integer_dtype(dtype)
    if NUMPY is not None:
        draw = NUMPY.random.default_rng(seed).integers(-7, 8, size=(rows, cols))
        values = draw if integer else draw / 8.0
        return values.astype(HOST_DTYPE[dtype])
    state = (seed * 2654435761 + 1) & 0x7FFFFFFF
    block: list[list[Any]] = []
    for _ in range(rows):
        row: list[Any] = []
        for _ in range(cols):
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            value = (state >> 13) % 15 - 7
            row.append(value if integer else value / 8.0)
        block.append(row)
    return block


def sub(tile: Tile, r0: int, r1: int, c0: int, c1: int) -> Tile:
    """``tile[r0:r1, c0:c1]``. Free: this never crosses DRAM."""
    if NUMPY is not None:
        return tile[r0:r1, c0:c1]
    return [row[c0:c1] for row in tile[r0:r1]]


def shape_of(tile: Tile) -> tuple[int, int]:
    """``(rows, cols)`` of a block, in either backend."""
    if NUMPY is not None:
        rows, cols = tile.shape
        return int(rows), int(cols)
    return len(tile), (len(tile[0]) if tile else 0)


def copy_of(tile: Tile) -> Tile:
    """A block that can be written without disturbing what it came from."""
    if NUMPY is not None:
        return tile.copy()
    return [list(row) for row in tile]


def add_into(acc: Tile, block: Tile) -> None:
    """``acc += block``, elementwise and in place."""
    if NUMPY is not None:
        acc += block
        return
    for acc_row, block_row in zip(acc, block, strict=True):
        for index, value in enumerate(block_row):
            acc_row[index] += value


def mma(acc: Tile, a: Tile, b: Tile, counters: Counters, slots: int) -> None:
    """``acc += a @ b`` — one instruction tile, at the accumulator's width.

    The accumulation is explicitly widened before the multiply: an ``int8``
    product does not fit an ``int8`` accumulator, which is why the schema
    declares ``precision.accumulate`` separately, and this is that rule made
    executable.

    *slots* is the MAC positions the array issues for one instruction whatever
    the operands' real extents — ``ROWS * ROWS * COLS``. The useful MACs are the
    operands' own product. The ratio of the two is the shape-padding loss the
    report quotes as utilisation: **area, not a pipeline drain** (D52).
    """
    rows, inner = shape_of(a)
    cols = shape_of(b)[1]
    counters.count_mma(useful=rows * inner * cols, slots=slots)
    if rows == 0 or inner == 0 or cols == 0:
        return
    if NUMPY is not None:
        acc += a.astype(acc.dtype) @ b.astype(acc.dtype)
        return
    for i in range(rows):
        acc_row = acc[i]
        a_row = a[i]
        for kk in range(inner):
            value = a_row[kk]
            if value:
                b_row = b[kk]
                for j in range(cols):
                    acc_row[j] += value * b_row[j]


def reference(a: Tile, b: Tile, integer: bool) -> Tile:
    """``a @ b`` computed once, at full width, as the answer to check against."""
    if NUMPY is not None:
        wide = "int64" if integer else "float64"
        return a.astype(wide) @ b.astype(wide)
    rows, inner = shape_of(a)
    cols = shape_of(b)[1]
    out = [[0 if integer else 0.0] * cols for _ in range(rows)]
    for i in range(rows):
        out_row = out[i]
        a_row = a[i]
        for kk in range(inner):
            value = a_row[kk]
            if value:
                b_row = b[kk]
                for j in range(cols):
                    out_row[j] += value * b_row[j]
    return out


def checksum(tile: Tile) -> float:
    """Sum of every element, exactly.

    A one-number fingerprint of the result, so two programs walking the *same*
    matmul by different decompositions can be shown to agree on more than "both
    exited 0". Exact by construction: the operands are dyadic, so nothing here
    rounds.
    """
    if NUMPY is not None:
        return float(NUMPY.sum(tile.astype("float64")))
    return float(sum(sum(row) for row in tile))


def max_abs_diff(x: Tile, y: Tile) -> float:
    """Largest elementwise ``|x - y|``, at full width in both backends."""
    if NUMPY is not None:
        return float(NUMPY.max(NUMPY.abs(x.astype("float64") - y.astype("float64"))))
    worst = 0.0
    for x_row, y_row in zip(x, y, strict=True):
        for left, right in zip(x_row, y_row, strict=True):
            worst = max(worst, abs(float(left) - float(right)))
    return worst


@dataclass
class Counters:
    """What the run actually moved.

    Every field is **measured** — nothing here is copied from the prediction,
    which is the whole point: the program and the report are two independent
    accounts of the same schedule, and the file's job is to make them meet.

    One lock rather than per-core tallies merged at the end. It costs time and
    buys clarity, and this is not a benchmark.
    """

    macs: int = 0
    """Useful MACs. ``M*N*K`` under every stationarity, or the walk is wrong
    (D53) — the same arithmetic, cut up differently."""
    mac_slots: int = 0
    """MAC positions the array issued, padding included."""
    a_dram_bytes: float = 0.0
    """Every byte of A that crossed the bus, re-reads included."""
    b_dram_bytes: float = 0.0
    """Every byte of B that crossed the bus, re-reads included."""
    c_dram_bytes: float = 0.0
    partial_dram_bytes: float = 0.0
    """Split-K's partials, out of kernel 1 and back into kernel 2 (D53)."""
    partial_sum_adds: int = 0
    """Elementwise additions that summed partial results — the ``(p-1)*M*N`` the
    report charges to the VECTOR unit (D62). Counted where they happen: in the
    shared accumulator under a K-on-grid walk, in the second kernel under
    split-K. A first touch is a copy into a zeroed accumulator and is not one of
    them, which is exactly why the count is ``p-1`` and not ``p``."""
    a_compulsory_bytes: float = 0.0
    """A bytes on their **first** touch only."""
    b_compulsory_bytes: float = 0.0
    """B bytes on their first touch only — what "each operand crosses the bus
    once" charges (``docs/MODEL.md`` §6.2). The gap against
    :attr:`b_dram_bytes` is the tiling re-read that model declines to charge,
    measured rather than argued about."""
    staging_events: int = 0
    tiles_run: int = 0
    idle_core_waves: int = 0
    """Core-waves with no tile to run. This is wave occupancy, executable (D30)."""
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def count_mma(self, *, useful: int, slots: int) -> None:
        with self.lock:
            self.macs += useful
            self.mac_slots += slots

    def count_read(self, operand_name: str, byte_count: float, *, first_touch: bool) -> None:
        with self.lock:
            if operand_name == "A":
                self.a_dram_bytes += byte_count
                if first_touch:
                    self.a_compulsory_bytes += byte_count
            else:
                self.b_dram_bytes += byte_count
                if first_touch:
                    self.b_compulsory_bytes += byte_count

    def count_c_write(self, byte_count: float) -> None:
        with self.lock:
            self.c_dram_bytes += byte_count

    def count_partial(self, byte_count: float) -> None:
        with self.lock:
            self.partial_dram_bytes += byte_count

    def count_partial_sum_adds(self, adds: int) -> None:
        with self.lock:
            self.partial_sum_adds += adds

    def count_staging_event(self) -> None:
        with self.lock:
            self.staging_events += 1

    def count_tile(self) -> None:
        with self.lock:
            self.tiles_run += 1

    def count_idle_core_wave(self) -> None:
        with self.lock:
            self.idle_core_waves += 1

    def occupancy(self, waves: int, used_cores: int, available_cores: int) -> float:
        """Fraction of core-waves that had a tile. Measured, not assumed.

        Against **available** cores, not the used ones: a core the schedule never
        gave a tile to is idle in every wave, and pretending it does not exist
        would report a chip with one busy array out of four as fully occupied.
        That difference is the whole of D30, and it is why the emitted file
        carries two core counts rather than one.
        """
        slots = waves * available_cores
        never_used = (available_cores - used_cores) * waves
        return 1.0 - ((self.idle_core_waves + never_used) / slots) if slots > 0 else 1.0

    def padding_efficiency(self) -> float:
        """Useful MACs over issued MAC slots — the shape-padding term (D52)."""
        return self.macs / self.mac_slots if self.mac_slots > 0 else 1.0


class Dram:
    """Off-chip memory, and the only place bytes cross the bus.

    Every read and write goes through here and is counted, and nothing else in
    an emitted program touches A, B or C directly. That is what lets the file
    assert its own traffic against the roofline's rather than assert a comment.

    First touches are tracked per byte-range so the program can report both what
    the walk *fetched* and what it would have fetched if every operand crossed
    the bus once — the difference being exactly the tiling re-read
    ``docs/MODEL.md`` §6.2 declines to model.
    """

    def __init__(
        self,
        a: Tile,
        b: Tile,
        c: Tile,
        counters: Counters,
        *,
        a_bytes_per_element: float,
        b_bytes_per_element: float,
        c_bytes_per_element: float,
        acc_bytes_per_element: float,
        partitions: int = 1,
        acc_dtype: str = "fp32",
    ) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.counters = counters
        self.a_bytes_per_element = a_bytes_per_element
        self.b_bytes_per_element = b_bytes_per_element
        self.c_bytes_per_element = c_bytes_per_element
        self.acc_bytes_per_element = acc_bytes_per_element
        rows, cols = shape_of(c)
        self.partials: list[Tile] = [zeros(rows, cols, acc_dtype) for _ in range(partitions)]
        """Split-K only: one full-size partial result per partition. They exist
        because split-K is **two kernels** and a partial cannot stay in a
        register across a kernel boundary (D53)."""
        self._touched: set[tuple[str, int, int, int, int]] = set()
        self._lock = threading.Lock()

    def _first_touch(self, key: tuple[str, int, int, int, int]) -> bool:
        with self._lock:
            if key in self._touched:
                return False
            self._touched.add(key)
            return True

    def read_a(self, r0: int, r1: int, c0: int, c1: int) -> Tile:
        """``A[r0:r1, c0:c1]``, charged at A's width."""
        elements = max(0, r1 - r0) * max(0, c1 - c0)
        first = self._first_touch(("A", r0, r1, c0, c1))
        self.counters.count_read("A", elements * self.a_bytes_per_element, first_touch=first)
        return sub(self.a, r0, r1, c0, c1)

    def read_b(self, r0: int, r1: int, c0: int, c1: int) -> Tile:
        """``B[r0:r1, c0:c1]``, charged at B's width."""
        elements = max(0, r1 - r0) * max(0, c1 - c0)
        first = self._first_touch(("B", r0, r1, c0, c1))
        self.counters.count_read("B", elements * self.b_bytes_per_element, first_touch=first)
        return sub(self.b, r0, r1, c0, c1)

    def write_c(self, r0: int, c0: int, block: Tile) -> None:
        """Store a finished block of C. Charged at C's width."""
        rows, cols = shape_of(block)
        self.counters.count_c_write(rows * cols * self.c_bytes_per_element)
        self._store(self.c, r0, c0, block)

    def write_partial(self, partition: int, r0: int, c0: int, block: Tile) -> None:
        """Store one partition's partial result — kernel 1's output (D53)."""
        rows, cols = shape_of(block)
        self.counters.count_partial(rows * cols * self.acc_bytes_per_element)
        self._store(self.partials[partition], r0, c0, block)

    def read_partial(self, partition: int, r0: int, r1: int, c0: int, c1: int) -> Tile:
        """Read one partition's partial result back — kernel 2's input (D53)."""
        elements = max(0, r1 - r0) * max(0, c1 - c0)
        self.counters.count_partial(elements * self.acc_bytes_per_element)
        return sub(self.partials[partition], r0, r1, c0, c1)

    @staticmethod
    def _store(target: Tile, r0: int, c0: int, block: Tile) -> None:
        rows, cols = shape_of(block)
        if NUMPY is not None:
            target[r0 : r0 + rows, c0 : c0 + cols] = block
            return
        for i in range(rows):
            target[r0 + i][c0 : c0 + cols] = list(block[i])


class Scratchpad:
    """On-chip staging for A, shared by every core (D33).

    A staging event fetches one band of A; every tile that shares that band then
    reads it from here for nothing. The band is keyed by the grid row it belongs
    to *and* by which ``A_RESIDENCY_TILES``-sized chunk of that row's tiles is
    being served, which is what makes the three A strategies one mechanism:
    ``stage``/``whole`` serve a whole row per event (A crosses DRAM exactly
    once), ``stream`` sets the residency to one tile, so every tile becomes its
    own key and re-fetches (D31).

    Shared, not per-core: consecutive tiles of a row land on *different* cores in
    the same wave, so "A is staged once per row" is a claim about a chip-wide
    buffer. Modelling it per-core would multiply A's traffic by the core count.
    """

    def __init__(self, counters: Counters) -> None:
        self._counters = counters
        self._bands: dict[tuple[int, ...], Tile] = {}
        self._lock = threading.Lock()

    def band(self, key: tuple[int, ...], fetch: Callable[[], Tile]) -> Tile:
        """The staged band for *key*, fetching it through DRAM on first ask."""
        with self._lock:
            band = self._bands.get(key)
            if band is None:
                band = fetch()
                self._bands[key] = band
                self._counters.count_staging_event()
                if DEBUG:
                    log(f"  stage A for key {key} -- crosses DRAM, once per key (D33)")
            return band


class Partials:
    """The accumulator where a K-on-the-grid walk's partial sums meet (D53).

    Under ``ws`` and ``is`` each tile owns a *slice* of the contraction, so two
    tiles in the same wave can land on the same output block from different
    cores. One lock per output block resolves that, and the choice is a claim
    about the model: a lock is an atomic on-chip accumulate and moves no bytes,
    which is what the cost model charges (nothing). Per-core private copies
    merged at the end would silently *be* split-K, which it charges for.

    That this class is needed at all is the difference between the two families
    of decomposition. Under ``os`` the question never arises: K is swept inside
    one tile, in one core's own accumulator.

    It also **counts the additions** it performs (D62), which is the number the
    report charges to the vector unit. What it does not model is *time*: the
    report overlaps those adds with the matrix work, and this program measures
    counts, not rates.
    """

    def __init__(
        self, m: int, n: int, tile_rows: int, tile_cols: int, dtype: str, counters: Counters
    ) -> None:
        self.acc = zeros(m, n, dtype)
        self.tile_rows = tile_rows
        self.tile_cols = tile_cols
        self.counters = counters
        self._block_cols = max(1, math.ceil(n / tile_cols))
        blocks = max(1, math.ceil(m / tile_rows)) * self._block_cols
        self._locks = [threading.Lock() for _ in range(blocks)]
        self._touched = [False] * blocks

    def accumulate(self, r0: int, c0: int, block: Tile) -> None:
        """Add one tile's partial into the output block that owns it.

        The **first** partial to reach a block lands in a zeroed accumulator, so
        it is a copy rather than an addition and is not counted: over ``p``
        k-slices that leaves ``p-1`` additions per element, which is what the
        model charges (D62).
        """
        rows, cols = shape_of(block)
        index = (r0 // self.tile_rows) * self._block_cols + (c0 // self.tile_cols)
        with self._locks[index]:
            if self._touched[index]:
                self.counters.count_partial_sum_adds(rows * cols)
            else:
                self._touched[index] = True
            if NUMPY is not None:
                self.acc[r0 : r0 + rows, c0 : c0 + cols] += block
                return
            for i in range(rows):
                acc_row = self.acc[r0 + i]
                block_row = block[i]
                for j in range(cols):
                    acc_row[c0 + j] += block_row[j]

    def drain(self, dram: Dram, m: int, n: int) -> None:
        """Write the finished accumulator out as C, one output block at a time.

        The bytes are C's own compulsory write — the accumulator itself never
        crossed DRAM, which is the claim ``materialises_partials`` makes.
        """
        for r0 in range(0, m, self.tile_rows):
            for c0 in range(0, n, self.tile_cols):
                r1, c1 = min(r0 + self.tile_rows, m), min(c0 + self.tile_cols, n)
                dram.write_c(r0, c0, sub(self.acc, r0, r1, c0, c1))


def run_cores(
    used_cores: int,
    core: Callable[[int, Callable[[], None]], None],
    *,
    warn_above: int = 0,
    warn_source: str = "",
) -> None:
    """Start one thread per modelled core, each running ``core(core_id, end_of_wave)``.

    Plumbing only. The waves, the tile each core takes, and the idle slot when
    the last wave is partly empty are all in the emitted ``walk()``, where a
    reader looks for them (D67); this starts the threads and hands each one the
    barrier that makes the waves lockstep.

    Not a thread pool. ``waves = ceil(tiles / units)`` is *lockstep*: a core is
    a persistent thing that takes one tile per wave, and ``end_of_wave`` is the
    barrier that keeps any core from starting wave w+1 before every core has
    finished wave w (D30). A task queue would deliver the same answer while
    erasing the structure the report costed.

    One OS thread per modelled core, deliberately not capped to the host's CPU
    count: this program is about structure, not speed, and a
    modelled-core-to-host-thread mapping would be a third concept with no
    counterpart in the model.

    *warn_above* is the core count past which that choice is worth remarking on,
    and the emitter derives it from the profiles rather than inventing a round
    number: it is the largest array any of them declares (D59). 0 disables the
    remark. It is not a limit — nothing here caps anything.
    """
    if warn_above and used_cores > warn_above:
        # The bar is the largest array any bundled profile declares, passed in
        # by the emitter — not a round number picked here (D59). Above it you
        # are past anything the repository describes, which is worth saying
        # once; it is not a limit, and nothing is capped.
        print(
            f"note: starting {used_cores:,} threads, one per modelled core — more "
            f"than the largest array any profile here declares ({warn_above:,}, "
            f"{warn_source}). Not capped to this host's CPUs on purpose; see the "
            f"docstring above."
        )
    barrier = threading.Barrier(used_cores)
    failures: list[BaseException] = []
    failure_lock = threading.Lock()

    def end_of_wave() -> None:
        barrier.wait()

    def run(core_id: int) -> None:
        try:
            core(core_id, end_of_wave)
        # Caught broadly and re-raised on the main thread: a core that dies while
        # its peers are inside barrier.wait() would hang the whole run, so the
        # barrier is aborted and the first failure surfaces from run_cores.
        except BaseException as exc:
            # Flush whatever this core had narrated: on a failure that partial
            # block is the most useful thing on screen.
            if DEBUG:
                log_flush()
            with failure_lock:
                failures.append(exc)
            barrier.abort()

    threads = [
        threading.Thread(target=run, args=(core_id,), name=f"core-{core_id}")
        for core_id in range(used_cores)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise failures[0]


@dataclass(frozen=True)
class Check:
    """One measured quantity beside the one the report predicted."""

    name: str
    predicted: float
    measured: float
    asserted: bool
    """Tier 1 (``True``) is a quantity the model computes *structurally*, so a
    mismatch is a genuine bug on one side or the other. Tier 2 is printed and
    not asserted — see the file's own header for which is which and why."""
    note: str = ""


def report_checks(checks: Sequence[Check]) -> list[Check]:
    """Print ``measured vs predicted`` and return the asserted rows that failed."""
    width = max((len(check.name) for check in checks), default=4)
    print(f"\n{'quantity'.ljust(width)}  {'predicted':>20}  {'measured':>20}  status")
    print("-" * (width + 54))
    failed: list[Check] = []
    for check in checks:
        ok = check.predicted == check.measured
        if check.asserted:
            status = "OK" if ok else "MISMATCH"
            if not ok:
                failed.append(check)
        else:
            status = "same" if ok else "differs"
        print(
            f"{check.name.ljust(width)}  {_number(check.predicted):>20}  "
            f"{_number(check.measured):>20}  {status}"
        )
        for line in check.note.splitlines():
            print(f"{' ' * 4}{line}")
    return failed


def _number(value: float) -> str:
    """Integers with thousands separators, fractions to six places."""
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:.6f}"
