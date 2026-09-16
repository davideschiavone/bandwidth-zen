# Command reference

Every command the tool has, every flag, and the exact invocations that reproduce the figures quoted
in `README.md`, `docs/MODEL.md` and `docs/CORRECTIONS.md`.

**Every output below was produced by running the command shown**, at the commit that added this
file. If one of them stops matching, the engine changed and one of the two is now a bug — say which
in `docs/CORRECTIONS.md` rather than quietly editing the number here.

---

## 0. Setup

```bash
git clone <repo> && cd bandwidth-zen
make venv                      # uv venv --python 3.11 + uv sync --all-groups
```

Then run everything from `backend/` with `uv run`, which needs no activation:

```bash
cd backend
uv run bwz --help
```

`make venv` installs three dependency groups: runtime, `dev` (pytest, ruff, mypy) and `plots`
(matplotlib). The engine itself never imports matplotlib.

---

## 1. `bwz list`

```bash
uv run bwz list                # both tables
uv run bwz list --help
```

Prints the bundled chip and model profiles with their headline numbers, shapes and provenance
(`sourced` / `N est.` / `hypothetical`). The `ridge` column is the arithmetic intensity above which
that chip is compute-bound — 6168 OP/byte for `chip_a`, 153 for A100 at fp16.

```
  id            name                       peak   dtype     DRAM BW     DRAM   on-chip   ridge   provenance
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
  a100_80gb     NVIDIA A100 SXM4 80GB   312 TFLOP/s  fp16   2.04 TB/s   80 GB   60.7 MB     153   1 est.
  chip_a        chip_a (edge INT8 NPU)    210 TOP/s  int8     34 GB/s    8 GB     55 MB    6168   hypothetical
  chip_b        chip_b (hypothetical)    52.4 TOP/s  int8     34 GB/s    8 GB      1 GB    1542   hypothetical
  h100_sxm      NVIDIA H100 SXM5 80GB   989 TFLOP/s  fp16   3.35 TB/s   80 GB   83.8 MB     295   2 est.
  jetson_orin   NVIDIA Jetson AGX Orin 42.6 TFLOP/s  fp16    205 GB/s   64 GB   7.07 MB     208   3 est.
  metis_aipu    Axelera Metis AIPU        210 TOP/s  int8   34.1 GB/s    8 GB   54.5 MB    6145   6 est.
  mi300x        AMD Instinct MI300X     1.31 PFLOP/s fp16    5.3 TB/s  192 GB    275 MB     247   2 est.
```

(Names are elided here to fit the page; the real table wraps them.) `metis_aipu` is the Axelera
Metis AIPU from ISSCC 2024 — the first profile whose numbers come from a paper rather than a
datasheet, and the one with the most `est.` fields, because the paper publishes compute and
capacity in full and DRAM bandwidth not at all. Read
[`../backend/profiles/chips/metis_aipu.yaml`](../backend/profiles/chips/metis_aipu.yaml) before
quoting anything derived from it.

---

## 2. `bwz matmul` — one `A[M,K] × B[K,N] → C[M,N]`

The smallest probe of a machine. Matmul vocabulary throughout: operands **A** and **B** and a
result **C**, no weights or activations, and no batch/context/phase knobs — `M` folds the batch in,
so a batch of 128 is `-M 128`.

| Flag | Meaning |
|---|---|
| `-M`, `-N`, `-K` | the three dimensions (required) |
| `-c`, `--chip` | chip profile id or path to YAML (required) |
| `-d`, `--dtype` | width of both operands, and of the result unless `--c` (default `fp16`) |
| `--a`, `--b` | per-operand widths, when they differ |
| `--c` | result width — the **accumulator**. Defaults to the wider operand. (`--out` is the figure directory, §3) |
| `--ideal` | set both efficiency de-ratings to 1.0: a datasheet ceiling, not a prediction |
| `--pipeline` / `--no-pipeline` | lane occupancy table (default on) |
| `--stationarity` | `os` \| `ws` \| `is` \| `rs` — which operand stays resident, §2.5.1. Default: the chip's own |
| `--split-k` | cut K into N pieces so one output tile is N units of work (`os` only); the N partials cost CUTLASS's second kernel, §2.5.1 |
| `--a-strategy` | `stage` (default, D33) \| `stream` (D31) \| `whole` — how A is loaded, §2.5.2 |
| `--b-dataflow` | `write-ahead` (default, D33) \| `on-demand` \| `persistent` — when B's write lands |
| `--a-residency-tiles` | override tiles served per A staging event; power-of-2 divisor of `TILES_PER_GROUP` |
| `--a-prefetch-depth` | override A's double-buffered staging depth (schedule-only) |
| `--iterations` | invocations this report represents; only `persistent` reads it |
| `--emit` | write this decomposition as a **runnable Python program** into `--out`, §2.6 |
| `--emit-stdout` | print that program instead, and nothing else, so `… \| python -` runs it |
| `--json` | the raw `Report` as JSON |

### 2.1 The datasheet check

```bash
uv run bwz matmul -M 10000 -N 10000 -K 10000 -c a100_80gb -d fp16 --ideal
```

```
  DRAM reads                   400 MB
  DRAM writes                  200 MB
  intensity            3333.3 OP/byte
  ridge point           153.0 OP/byte
  stationarity                     os
  shape utilisation            99.91%
  t_dram                       294 µs
  t_compute                   6.42 ms
  latency                     6.42 ms
  verdict               COMPUTE_BOUND
```

`2·10000³ = 2.000e12` OP at A100's 312 TFLOP/s is 6.41 ms; the extra 0.09% is quantisation — every
dimension is a multiple of 16 here, so the only loss is wave occupancy (390 625 tiles over 432
tensor cores is 905 waves, the last one 89% full). Every number can be checked against the datasheet
by hand — that is what `--ideal` is for.

Drop `--ideal` and the same command gives **9.17 ms**: the difference is `÷0.70`, the unfitted
`DEFAULT_ACHIEVED_FLOPS_FRACTION`. The verdict does not move, which is what the flip margin
measures.

### 2.2 Shape utilisation at M=1

```bash
uv run bwz matmul -M 1 -N 10000 -K 10000 -c a100_80gb -d fp16 --ideal
```

```
  stationarity                    os   C resident, 1 x 625 tiles (M x N) each
                                       sweeping K — tensor_core's own
  shape utilisation            4.52%
  DRAM reads                  139 MB
  DRAM writes                  20 kB
  latency                    68.3 µs
  verdict              DRAM_BW_BOUND
```

The arithmetic fell by 10 000× against §2.1 and the traffic did not fall at all, so the workload
crossed the ridge. `--ideal` does **not** remove the 4.52%: it is geometry, not a derating.

The figure is two losses multiplied, and they live at different levels — only the first is what
CLAUDE.md's `M=1 → ≈1/rows` sanity check is about:

| | | |
|---|---|---|
| **array** | `1 / 16 = 6.25%` | one row of work pays for a whole 16-row instruction tile. Area, not a pipeline drain — a tensor core has no M-serial pipeline to fill (`docs/CORRECTIONS.md` D52) |
| **chip** | `625 / (2 × 432) = 72.3%` | only 625 output tiles exist, so 432 tensor cores take two waves and the second is 45% full (D30) |
| **reported** | `6.25% × 72.3% = 4.52%` | |

### 2.3 Widths — all integer, all float, and mixed

```bash
uv run bwz matmul -M 4096 -N 4096 -K 4096 -c a100_80gb -d int8            --ideal   # all int8
uv run bwz matmul -M 4096 -N 4096 -K 4096 -c a100_80gb -d int8 --c int32 --ideal   # int32 result
uv run bwz matmul -M 4096 -N 4096 -K 4096 -c a100_80gb -d fp16 --c fp32   --ideal  # all float
uv run bwz matmul -M 4096 -N 4096 -K 4096 -c a100_80gb --a fp16 --b int8   --ideal  # mixed
```

| | operations | result C | DRAM read / written | intensity | latency |
|---|---|---|---|---|---|
| `int8 × int8 → int8` | 137 GOP | 16.8 MB | 16.8 MB / 16.8 MB | 2730.7 OP/byte | 221 µs |
| `int8 × int8 → int32` | 137 GOP | **67.1 MB** | 33.6 MB / 67.1 MB | **1365.3 OP/byte** | 221 µs |
| `fp16 × fp16 → fp32` | 137 GOP | 67.1 MB | 67.1 MB / 67.1 MB | 1024.0 OP/byte | 442 µs |
| `fp16 × int8 → fp16` | 137 GOP | 33.6 MB | 50.3 MB / 33.6 MB | 1638.4 OP/byte | **442 µs** |

The read column shrinks as **B** becomes resident; the write column never does. On the first row
A100 reads no B at all — 16.8 MB of int8 weights fit in its 60.7 MB of on-chip capacity — and the
16.8 MB it does read is **A**, which a lone matmul always pays for in full: the residency discount
is inter-operation reuse, and a one-operation graph has no producer to reuse from (D33). `C` is the
answer, so all 16.8 MB of it is written whatever the capacity (D22).

Two rules, both visible above:

- **The result width changes bytes only, never operations.** `int32` quadruples C and halves the
  intensity; `t_compute` is identical.
- **A mixed matmul runs at the wider operand.** The last row halves B's bytes against a plain fp16
  matmul and takes exactly as long, because both operands enter the array through one datapath. It
  is 2× the int8 rows, which do the same arithmetic at twice the rate.

### 2.4 The tile schedule

`--pipeline` is on by default:

```bash
uv run bwz matmul -M 4096 -N 4096 -K 4096 -c a100_80gb -d fp16
```

```
  lane       busy       occupancy   what it was doing
  dram    58.1 µs      9% of span   operand tiles in, results out, across the one modelled link
  sram    1.25 ms   1.97 of 2 buf   tile buffers held from fetch to use
  core     634 µs    100% of span   matrix arithmetic, plus the dispatches
  vector      0 s      0% of span   norms, activations, residuals — not the matrix engine

  64 steps drawn, coalesced from 152 tiles, double buffered. Span 635 µs against a
  reported 634 µs: the extra 908 ns (0.1%) is pipeline fill/drain, which
  max(load, compute) omits.
```

**"152" is waves, not tiles.** The grid here is `⌈4096/16⌉ × ⌈4096/16⌉ = 65 536` output tiles, and
A100 has 432 tensor cores, so the schedule steps through `⌈65536/432⌉ = 152` waves of them — a step
is a wave, because drawing one bar per tile would show a 432-core chip working through tiles in
series when it does 432 at a time (D30). The 64 rows drawn coalesce those 152 and say so.

**The occupancy column carries a different unit per row, on purpose.** DRAM and the array are
single serial resources — their spans never overlap — so theirs is a duty cycle and cannot exceed
100%. SRAM is not a resource that is busy or idle; it is *n* buffers, and the same arithmetic
counts how many were occupied: `1.97 of 2` means both halves of the double buffer were in use
almost all the time. That is capacity, not bandwidth, which is the whole of what SRAM contributes
in this model.

### 2.5 Dataflow — which operand stays resident, and how the other two move

Three questions, in the order they have to be answered. **Stationarity** picks the decomposition:
which dimensions form the tile grid and which one each tile sweeps. **A's strategy** decides how
often A crosses DRAM within that grid, and **B's dataflow** decides when B's write lands relative to
compute. The defaults reproduce every number in §2.1–2.4 exactly.

#### 2.5.1 Stationarity — `--stationarity`, `--split-k`

**Which operand stays resident decides the whole decomposition** (D53): which dimensions form the
parallel tile grid, which one each tile sweeps, and whether partial sums are owed. `--stationarity`
selects it and defaults to the chip's own — `os` for every matrix core, what cuBLAS and CUTLASS do,
and `ws` for the in-memory-compute profiles, whose weights *are* their memory.

The effective choice is a table row, not just an assumptions line:

```bash
uv run bwz matmul -M 512 -N 512 -K 4096 -c a100_80gb --ideal --no-pipeline
```

```
  stationarity                    os   C resident, 32 x 32 tiles (M x N) each
                                       sweeping K — tensor_core's own
  shape utilisation           79.01%   shape padded to the 16x16 tile —
                                       geometry, not a derating
  t_compute                  8.71 µs   operations / (effective peak x util)
  latency                    8.71 µs
  verdict              COMPUTE_BOUND
```

79% is wave occupancy, not padding: 32 × 32 = 1024 output tiles over 432 tensor cores is 3 waves
whose last is a third full.

**What `--split-k` is.** Under `os` one core owns an output tile and sweeps the *whole* contraction
inside it, so the amount of parallelism available is the number of output tiles and nothing else —
`M·N/(rows·cols)`. When that is smaller than the chip, most of the chip idles however long K is.
Split-K cuts K into `p` chunks and makes `(output tile, chunk)` the unit of work, so there are `p`×
as many pieces and the chip fills. Nothing about the arithmetic changes: each chunk does `1/p` of
the same MACs.

What changes is that the `p` chunks of one output tile each end up holding a **partial** value of
it, and partials have to be added. CUTLASS does that in a *second kernel* — *"partitionedK GEMM, and
batched reduction"* — so the first kernel writes `p` full `M × N` partials to DRAM and the second
reads them back, adds `(p−1)·M·N` of them on the vector unit, and costs one more dispatch. That is
the whole trade, and it is exactly what CUTLASS documents split-K for: when *"there are too few
threadblocks to efficiently occupy the entire GPU"*.

```bash
uv run bwz matmul -M 512 -N 512 -K 4096 -c a100_80gb --ideal --no-pipeline --split-k 8
```

```
  DRAM traffic               13.1 MB   reads + writes
  stationarity                    os   C resident, 32 x 32 tiles (M x N) each
                                       sweeping K — tensor_core's own
  split-K                          8   CUTLASS's two kernels: partials out to
                                       DRAM and back, summed on cuda_core
  shape utilisation           99.81%   shape padded to the 16x16 tile —
                                       geometry, not a derating
  t_dram                     6.43 µs   traffic / effective bandwidth
  t_compute                  6.99 µs   operations / (effective peak x util)
  latency                    6.99 µs
  verdict              COMPUTE_BOUND
```

Utilisation goes 79.01% → 99.81% and latency 8.71 → 6.99 µs, but DRAM traffic goes 4.72 → 13.1 MB:
the extra 8.4 MB is eight full `512×512` partials written by the first kernel and read back by the
second. Push the factor higher and `t_dram` overtakes `t_compute` — the flag lets you find where.

**`ws` buys the same occupancy without the traffic** — and pays a different engine for it (D62).
A100's tensor core declares all three grids so this comparison can be made on one chip:

```bash
uv run bwz matmul -M 512 -N 512 -K 4096 -c a100_80gb --ideal --no-pipeline --stationarity ws
```

```
  DRAM traffic                   4.72 MB   reads + writes
  stationarity                        ws   nothing held, K on the grid, 256 x
                                           32 tiles (K x N) each sweeping M —
                                           requested with --stationarity
  reduction            on chip — 3.43 µs   66,846,720 adds on cuda_core at
                                           19.5 TOP/s, overlapped with the
                                           matrix work (compute is the max of
                                           the two, not the sum): hidden under
                                           6.9 µs of matrix work (50% of it),
                                           so it costs capacity, not latency
  shape utilisation               99.81%   shape padded to the 16x16 tile —
                                           geometry, not a derating
  t_compute                       6.9 µs   operations / (effective peak x
                                           util), overlapped with the
                                           reduction: max of the two engines
  latency                         6.9 µs
  verdict                  COMPUTE_BOUND
```

Same 99.81% occupancy as `--split-k 8`, same 4.72 MB of traffic as plain `os` — because K is on the
tile grid rather than cut twice, the partials never leave the chip. What they cost instead is
**vector time and capacity**: 256 k-slices leave `255 × 512 × 512 = 66.8 M` additions for the CUDA
cores, which run 16× below the tensor cores. The array builds the next k-slice while they sum the
last, so compute is `max(6.9, 3.43) µs` — and 3.43 is almost exactly half of 6.9 for a reason that
does not depend on the shape: the vector unit is 16× slower at `2 × 16 =` 32× less work.

The reduction is therefore **free of latency here and is still the thing to watch**, because its
other cost is a live `M × N` accumulator. Past on-chip capacity the placement flips from `ON_CHIP`
to `DRAM` and the same decomposition falls off a cliff — at `-M 8192 -N 8192 -K 4096` it is 39.8 ms
against `os`'s 2.52 ms, the report naming the flip:

```
  reduction            through DRAM — 1.25 ms   17,112,760,320 adds on
                                                cuda_core at 13.6 TOP/s plus
                                                68.7 GB of round trip,
                                                serialised after the matrix
                                                work: the 134 MB of live
                                                accumulators do NOT fit the
                                                60.7 MB on chip, so the
                                                placement flipped from on-chip
                                                to DRAM — a cliff, not a slope
```

`docs/MODEL.md` §6.1 has the full placement table, including the `LOCAL` case: a unit that declares
`local_accumulation_inputs` deep enough for K sums its partials in its own periphery and is charged
nothing. Metis declares 16384 of them and the paper says why.

**A chip that cannot run the choice is refused, not clamped** (CLAUDE.md #8). Unlike
`--a-strategy`/`--b-dataflow`, which pick between orderings of the same work, a stationarity is a
different decomposition, so a silent fallback would report a number for hardware you did not ask
about:

```bash
uv run bwz matmul -M 512 -N 512 -K 4096 -c a100_80gb --stationarity rs
```

```
Infeasible.
  • stationarity='rs' is not supported by tensor_core, which declares is, os,
ws. Drop the flag to use the chip's own dataflow, or pick one it declares.
```

Two more refusals guard the reduction itself, and both are about pricing it to the wrong engine or
cutting K twice:

```bash
uv run bwz matmul -M 4096 -N 4096 -K 4096 -c a100_80gb -d int8 --stationarity ws
uv run bwz matmul -M 4096 -N 4096 -K 4096 -c a100_80gb --stationarity ws --split-k 4
```

```
  • stationarity='ws' puts K on the tile grid, so partial sums leave tensor_core
and something has to add them — but NVIDIA A100 SXM4 80GB declares no
non-systolic compute unit supporting 'int8', so the only engine available is
tensor_core itself. A matrix engine does matrix-multiply-accumulate and nothing
else (D27), and charging elementwise adds at its rate would report this
decomposition as nearly free. Refused rather than mispriced (D62). Non-systolic
units here support ['fp16', 'fp32']; run at one of those dtypes, or use the
chip's own dataflow.

  • --split-k 4 cannot be combined with stationarity='ws': that grid already
carries K on one of its axes, so the contraction would be cut twice and its
partials summed twice. Split-K is an output-stationary knob (D53) — drop one of
the two flags.
```

Exit code 2, the same as any other infeasible report. `rs` is implemented but no profile declares
it; if one ever does, `report.assumptions` labels its numbers unvalidated.

Both flags exist on `bwz run` too, and reach its figures — and unlike the A/B knobs below they
are **not** inert on a network, since every matmul in the graph is decomposed the same way. In the
figures the effective grid appears in the page banner, and the animation's geometry panel draws the
resident operand's grid (`docs/plots/README.md`).

#### 2.5.2 A and B — `--a-strategy`, `--b-dataflow`

A is a byte-amount knob; B is a timing knob (`docs/MODEL.md` §6.3a, `docs/CORRECTIONS.md` D36).
Both operate *within* the grid §2.5.1 chose: "one staging event per grid row" is a k-slice under
`ws` and a band of M rows under `os`, and A crosses DRAM exactly once under `stage` either way.

```bash
uv run bwz matmul -M 8192 -N 8192 -K 8192 -c metis_aipu -d int8 --ideal \
  --a-strategy stream --no-pipeline
```

```
  operand A                   67.1 MB   8192 x 8192 x 1 B
  operand B                   67.1 MB   8192 x 8192 x 1 B
  DRAM reads                  1.14 GB   A and B, less whatever stays on chip
  DRAM traffic                1.21 GB   reads + writes
  t_dram                      35.4 ms   traffic / effective bandwidth
  latency                     35.4 ms
  verdict               DRAM_BW_BOUND
```

`stream` re-reads A once per tile instead of once per grid row — `TILES_PER_GROUP = 16` here, so A's
share of DRAM reads is 16× `stage`'s: 1.07 GB against 67.1 MB, and the assumptions drawer says so:

```
A: streamed per tile (D31), 16x the staged total — 67.1 MB would cross DRAM
once under stage/whole, 1.07 GB crosses it under stream
```

**`whole` and `persistent` clamp rather than raise when they do not fit** (CLAUDE.md #8), and this
shape hits both fallbacks:

```bash
uv run bwz matmul -M 8192 -N 8192 -K 8192 -c metis_aipu -d int8 --ideal \
  --a-strategy whole --no-pipeline
```

```
a_strategy=whole requested but A (67.1 MB) does not fit the 54.5 MB
scratchpad; fell back to stage — the same 67.1 MB total, staged per grid row
instead of ramped upfront.
```

```bash
uv run bwz matmul -M 8192 -N 8192 -K 8192 -c metis_aipu -d int8 --ideal \
  --b-dataflow persistent --no-pipeline
```

```
b_dataflow=persistent requested but B is 256 tiles against 16 resident; fell
back to write-ahead.
```

`whole` needs the scratchpad to hold all of A; `persistent` needs `tiles <= units * weight_sets`
(16 here — Metis's 4 AI cores × 4 weight sets). Both fallbacks are named, with the requested value
and the value used, never silent. `write-ahead`/`on-demand`/`persistent` never move a *byte* within
one pass — B is fetched exactly once whichever is chosen (D30) — so the dataflow line always names
the byte-carrying side plainly:

```
A: staged 4.19 MB per k-slice (16 k-slices) — crosses DRAM exactly once
(D33) · B: write-ahead depth 4 (D33).
```

`--iterations` only changes anything when paired with a `persistent` B that fits: the first
invocation writes B in full, every later one reuses it, and the report — which is still one
invocation's numbers, not `N` of them — charges the amortised share `1/iterations` of that write.

**The two escape hatches, each with a real effect.** `--a-residency-tiles` overrides how many of a
grid row's `TILES_PER_GROUP` tiles one A staging serves, clamped down to the largest power-of-2 divisor
when the requested value is not one:

```bash
uv run bwz matmul -M 8192 -N 8192 -K 8192 -c metis_aipu -d int8 --ideal \
  --a-residency-tiles 5 --no-pipeline
```

```
a_residency_tiles=5 is not a power-of-2 divisor of TILES_PER_GROUP=16; clamped
to 4, the largest one that is.
```

`--a-prefetch-depth` overrides the double-buffered staging depth the schedule uses — schedule-only,
so it never moves a report number, only how much pipeline fill/drain the drawn trace shows:

```bash
uv run bwz matmul -M 4096 -N 4096 -K 4096 -c a100_80gb -d fp16 --a-prefetch-depth 1
```

```
  sram      691 µs   1.00 of 1 buf   tile buffers held from fetch to use
  64 steps drawn, coalesced from 152 tiles, double buffered. Span 694 µs against
  a reported 636 µs: the extra 58.1 µs (8.4%) is pipeline fill/drain, which
  max(load, compute) omits.
```

Depth 1 holds one buffer at a time instead of two and pays for it in fill/drain — 8.4% here against
under 1% at the default depth on the same shape (`sram` reads `1.00 of 1 buf` against `1.97 of 2`
buffers) — while `latency` itself is untouched: `636 µs` either way. The reported latency always
comes from the roofline's formula, never from the drawn schedule (D19, D35).

**`--b-dataflow on-demand` is the same kind of schedule-only cost, on the other operand (D40).**
`write-ahead` (the default) already prefetches B's load behind the previous wave's compute;
`on-demand` forces the load to wait instead, exposing the same, already-costed load duration on the
critical path:

```bash
uv run bwz matmul -M 20000 -N 2048 -K 2048 -c metis_aipu -d int8 --ideal --b-dataflow on-demand
```

```
  sram     4.62 ms   1.69 of 2 buf   tile buffers held from fetch to use
  4 tile steps, double buffered. Span 2.73 ms against a reported 2.52 ms: the
extra 205 µs (7.5%) is pipeline fill/drain, which max(load, compute) omits.
```

205 µs is exactly one wave's compute time (`t_compute / steps = 820 µs / 4`) — the DRAM port is
saturated end to end under `write-ahead` on this shape, so the gate binds once and the delay
propagates as a constant shift. `t_dram`, `t_compute` and `latency` are unchanged from `write-ahead`
(2.52 ms either way) — only the drawn span moves, the same D19 rule `--a-prefetch-depth` follows.

`--b-dataflow persistent` on the same shape draws **byte-for-byte and timing-for-timing identical**
to `write-ahead` — not a rounding-error difference, an exact one: `write-ahead`'s existing double
buffering already achieves this model's best-case overlap, so there is no reordering of B's fixed
loads that makes a same-pass `persistent` trace faster (D40). `persistent`'s real effect stays the
cross-`--iterations` amortisation §2.5 already documents above, a byte story, not a schedule one.

### 2.6 `--emit` — the decomposition as a program you can run

Everything above §2.5 describes a schedule. `--emit` writes it as **real Python** that walks the
same tile grid, stages A on the same events, hands tiles to cores the same way, counts what it
moves, and asserts those counts against the report it came from (D54). This is what a chip's
deployment listing *is* now: the pseudo-C loop nest the figures used to print was retired for it in
the same change, and `--timeline`'s page and `--animate`'s code pane both show this file.

```bash
uv run bwz matmul -M 64 -N 64 -K 128 --chip a100_80gb --emit --out ~/k
python ~/k/matmul-a100_80gb-fp16-os.py
```

```
NVIDIA A100 SXM4 80GB · fp16 · output-stationary
backend: numpy;  16 threads, one per modelled core in use

quantity                          predicted              measured  status
---------------------------------------------------------------------------
tiles                                    16                    16  OK
waves                                     1                     1  OK
MACs                                524,288               524,288  OK
MAC slots issued                    524,288               524,288  OK
idle core-waves                         416                   416  OK
    a core with no tile this wave. This IS wave occupancy (D30).
A staging events                          4                     4  OK
A bytes                              16,384                16,384  OK
C bytes                               8,192                 8,192  OK
partial bytes                             0                     0  OK
partial-sum additions                     0                     0  OK
    (p-1) x M x N, where p is how many k-slices each output element ends
    up with. Not new arithmetic — 2*M*N*K already counts them — but the
    report charges them to the VECTOR unit, because they have left the
    matrix engine's own accumulator (D27/D62).
B bytes fetched                           0                65,536  differs
    tier 2. The report charges compulsory traffic and then discounts it by a
    residency fraction a capacity heuristic supplies; this walk fetches what
    the tile order asks for. The gap above it is the tiling re-read that
    docs/MODEL.md 6.2 declines to model, measured rather than argued about.
B bytes, first touch                      0                16,384  differs
    tier 2. What B costs if every byte of it crosses the bus exactly once.

wave occupancy   0.0370   1 - idle core-waves / (WAVES * AVAILABLE_CORES)
shape padding    1.0000   useful MACs / MAC slots issued (D52)
utilisation      0.0370   against the report's 0.0370

numerics: max |C - A@B| = 0   (tolerance 0)
C checksum: 38

every tier-1 count matches the report, and C == A @ B.
```

#### `--debug` — narrate the walk

The emitted program takes one flag of its own. It prints which core takes which tile in which wave,
when A is staged, and **every instruction tile** with its operand ranges:

```bash
uv run bwz matmul --m 16 --n 16 --k 17 --chip a100_80gb --ideal --emit --out ~/k
python ~/k/matmul-a100_80gb-fp16-os.py --debug
```

```
--debug: 1 tile(s) over 1 core(s) in 1 wave(s), issuing 2 instruction tile(s).
         core/wave assignment, A staging events, and every
         instruction tile follow. Expect one line each.
core 0     wave 0    tile 0
  C[0:16, 0:16]  piece 0, sweeping kt 0..2
  stage A for key (0, 0, 0) -- crosses DRAM, once per key (D33)
    kt=0    A[0:16, 0:16] @ B[0:16, 0:16]      4,096 useful of 4,096 slots
    kt=1    A[0:16, 16:17] @ B[16:17, 0:16]        256 useful of 4,096 slots
```

`kt=1` is `K=17`'s ragged step: a 16x**1** slice of A against a **1**x16 slice of B, issuing a whole
4 096-slot instruction tile for 256 useful MACs. That is where `shape utilisation` comes from, one
line at a time.

**One core's lines print as one block.** Cores run concurrently, and the narration is *indented* — a
tile under a core, a k-step under a tile — so that nesting is a claim about which line belongs to
which. Each core buffers its block and prints it whole (D61). Blocks still appear in **completion
order**, not tile order, which is honest: they genuinely run at the same time. Within a block
nothing interleaves.

**It says what the decomposition is.** The same flag on Metis reports the other family — a resident
B tile with M streaming past it, and a partial rather than a result:

```
core 0     wave 0    tile 0
  B[0:512, 0:512] resident, M streams past it
  stage A for key (0, 0, 0) -- crosses DRAM, once per key (D33)
    mt=0    A[0:512, 0:512] @ B[0:512, 0:512] -> PARTIAL into C[0:512, 0:512]
...
core 3     wave 6    idle -- no tile left (D30)
```

Those `idle` lines are the wave-occupancy term, printed: three of them here, against the
`idle core-waves 3` the tier-1 table asserts.

**One line per instruction tile**, so it is for small shapes — the header says up front how many to
expect, from `PREDICTED["mac_slots"] // SLOTS_PER_MMA`, which *is* the `mma()` call count. At
1000x2000x3000 that is 1.48 million lines. Without the flag nothing changes: every narration site is
guarded by `if DEBUG:`, so an unread line is never even formatted.

`--emit-stdout` writes to stdout and suppresses everything else, so the whole loop is one line:

```bash
uv run bwz matmul -M 64 -N 64 -K 128 --chip a100_80gb --emit-stdout | python -
```

The file is named `matmul-<chip>-<dtype>-<stationarity>[-splitk<N>].py` and lands in `--out`,
beside the figures — the page and the program are two views of one decomposition, so they share a
destination and an ending (D65). There is no separate path to give.

**What the file contains, in order.** The constants first — the shape from the command line, the
chip from its profile, the strategy, then the grid those imply — each with a comment naming where
its value came from and nothing anywhere that is a free parameter. Then a `PREDICTED` block, which
is the report's own numbers. Then the loop nest, which is the part to read. The runtime that makes
it run — counted DRAM, the shared staging buffer, the lockstep wave loop — is at the **bottom**,
against convention and deliberately: 400 lines of machinery between the constants and the walk
would bury the thing you came for.

```python
def run_tile(tile: int, dram: Dram, pad: Scratchpad) -> None:
    """One output tile. C stays in the accumulator; K is swept INSIDE it.

    The whole contraction for this output block happens in one core's own
    accumulator, which is exactly why output-stationary owes no reduction: no
    partial sum ever leaves this function (D53).
    """
    mt, nt, part = tile_row(tile), tile_col(tile), partition_of(tile)
    ...
    acc = zeros(m1 - m0, n1 - n0, ACC_DTYPE)     # the accumulator that stays put
    for kt in range(kt0, kt1):                   # K is swept INSIDE this tile
        k0, k1 = kt * ROWS, min(kt * ROWS + ROWS, K)
        a = sub(band, 0, m1 - m0, k0 - k_lo, k1 - k_lo)   # already on chip
        b = dram.read_b(k0, k1, n0, n1)                   # crosses DRAM
        mma(acc, a, b, COUNTERS, SLOTS_PER_MMA)

    dram.write_c(m0, n0, acc)                    # finished, not a partial
```

Ask the same shape for weight-stationary and the nest is a different shape, because the
decomposition is:

```python
    b = dram.read_b(k0, k1, n0, n1)              # the operand that stays put
    band = stage_a(tile, dram, pad)

    for mt in range(M_TILES):                    # M streams past the resident tile
        ...
        partials.accumulate(m0, n0, product)     # a PARTIAL over K; nothing is stored
```

That `partials` object has to exist under `ws` and does not under `os`, which is D53's claim in one
line of code. Two cores can own `(kt, nt)` and `(kt', nt)` in the same wave, so it takes **a lock
per output block** — an atomic on-chip accumulate, costing no bytes. Private per-core copies merged
at the end would silently *be* split-K, which is a different decomposition and is charged like one.

`partials` also **counts what it adds**, and that count is tier 1 (D62). A first touch lands in a
zeroed accumulator and is a copy, so `p` k-slices leave `p−1` additions per output element: 28,672
at 64×64×128 under `ws` and `is`, 12,288 under `--split-k 4`, 0 under `os`. The report charges
exactly those to the vector unit, so a disagreement fails the run.

What the file does **not** execute is the *overlap*: that the vector unit sums slice *n* while the
array builds slice *n+1* is a claim about time, and this program counts. Its own fidelity table says
so, and its docstring quotes the placement its report actually charged — free in one AI core's
periphery on Metis, on the CUDA cores and overlapped on A100, through DRAM when the accumulator does
not fit.

**It is not a benchmark, and the file says so before anything else.** It validates counts, not
time: it runs one OS thread per modelled core whatever the host has, makes no attempt to be fast,
and its wall clock has no relationship to the predicted latency.

#### Two tiers, and why the B rows are not asserted

Tier 1 is asserted and must match exactly: `tiles`, `waves`, the useful MAC count, the MAC slots
issued, the idle core-waves, A's staging events, A's bytes, C's bytes, split-K's partial round trip,
the partial-sum additions, and `C == A @ B`. These are quantities the model computes *structurally*, so a mismatch is a
real bug on one side or the other.

Wave occupancy and shape padding are pinned as **integers** — `idle_core_waves` and `mac_slots` —
rather than as floats, and the program prints the ratios from the counts it asserted. Same claim,
no float comparison. The `if tile < TILES` inside `run_waves` *is* wave occupancy (D30), executable.

Tier 2 is printed and not asserted, and the B rows above are it — with one more that appears only
when the report's accumulator did not fit on chip: the partial *bytes*. There the report charges a
DRAM round trip on a capacity judgement and the walk keeps its accumulator in `partials` whatever
its size, because where an accumulator lives is a heuristic about a machine and not a step of the
decomposition (D62). Same treatment, same reason. The report charges compulsory
traffic — each operand crosses the bus once — and then discounts it by a residency fraction that
comes from a capacity heuristic this file deliberately does not imitate; on a lone matmul whose B
fits on chip, that discount is total and the charged figure is **zero**. The walk fetches what the
tile order asks for. At 1000×2000×3000 on A100 the three numbers are far apart, and the gap is the
point:

```bash
uv run bwz matmul -M 1000 -N 2000 -K 3000 --chip a100_80gb --emit --out ~/k \
  && python ~/k/matmul-a100_80gb-fp16-os.py
```

```
(excerpt)
A bytes                          6,000,000             6,000,000  OK
C bytes                          4,000,000             4,000,000  OK
B bytes fetched                          0           756,000,000  differs
B bytes, first touch                     0            12,000,000  differs

wave occupancy   0.9594   1 - idle core-waves / (WAVES * AVAILABLE_CORES)
shape padding    0.9894   useful MACs / MAC slots issued (D52)
utilisation      0.9493   against the report's 0.9493
```

12 MB is B crossing the bus once. 756 MB is what an `os` walk actually fetches, because each of the
63 row-bands of M re-reads the whole of B — the tiling re-read `docs/MODEL.md` §6.2 declines to
model and names as the reason a DRAM-bound latency there is a lower bound. 0 is what the report
charges. Promoting these rows to tier 1 needs the model changed, not the assertion loosened; D54
records the finding.

#### What is executed and what is only written down

| choice | in the emitted program |
|---|---|
| `--stationarity` | **real** — a different loop nest, a different resident buffer, different counters |
| `--split-k` | **real** — per-partition partials and a second reduction kernel |
| `--a-strategy` | **real** — changes how often A is staged, and A's measured bytes |
| `--a-residency-tiles` | **real** — the staging buffer serves that many tiles before refill |
| `--b-dataflow` | annotation — a placement in *time*, moving no byte within one pass (D30/D33) |
| double buffering | annotation — "latency is `max(load, compute)`" is a claim about time |
| sub-cycles (bit-serial) | annotation — a rate, not a structure |

The last three are named in the file's own header rather than left to be inferred. On a bit-serial
array the program prints why its utilisation figure is *higher* than the report's: D34 charges a
sub-cycle row of fill on K, which is a rate, and this program measures no rates.

**numpy is optional.** It is not a `bwz` dependency, so every operation has a pure-Python fallback
and the emitted file prints which backend it took. That is also what makes the artifact portable:
it runs on a machine that has never heard of this repository.

**Operands are dyadic on purpose.** Values are drawn from `{-7..7}`, divided by 8 for a float
format, so every product and every partial sum is exact in the accumulator and the tolerance is
`0`. A difference at the end is then a *walk* error and never a rounding one — this program checks
a decomposition, and letting rounding share the same tolerance would make a failure ambiguous.

---

## 3. `bwz encoder-layer` — one encoder layer, sized from the command line

The transformer counterpart of `bwz matmul` (§2): the shape comes from the command line, not a
profile. Bidirectional attention over all `S` tokens, no KV cache, no LM head — one layer, always.
For anything deeper, load a profile and use `bwz run` (§4).

| Flag | Meaning |
|---|---|
| `-c`, `--chip` | chip profile id or path (required) |
| `-d`, `--dmodel` | model width (default 8) |
| `--nheads` | attention heads (default 2) — `dmodel` must divide evenly by this; `head_dim` is always `dmodel // nheads`, never a separate input, and a non-divisible pair is rejected rather than floored |
| `--ffn` | FFN inner width (default 16, or 4x `--dmodel` when `--dmodel` is explicitly set — D45) |
| `--vocab` | vocabulary size (default 16) |
| `-S`, `--tokens` | sequence length (default 4) |
| `-b`, `--batch` | batch size (default 1) |
| `--ffn-type` | `relu` (default) \| `gelu` \| `swiglu` \| `geglu` |
| `--norm` | `layernorm` \| `rmsnorm` (default) \| `batchnorm` \| `none` |
| `--tie` / `--untie` | tie the embedding and output tables (default tie) |
| `--weights` | precision (default `fp16`) |
| `--ideal`, `--json` | as for `matmul` |

### 3.1 The default shape — small enough to count by hand

```bash
uv run bwz encoder-layer --chip a100_80gb --ideal
```

```
  what         parameters   derivation
 ───────────────────────────────────────────────
  Q, K, V, O          256   8x8 + 2 x 8x8 + 8x8
  FFN                 256   2 x 8 x 16
  norms                16   2 x rmsnorm over 8 channels
  per layer           528
  embeddings          128   16 x 8, tied
  final norm            8
  total               664

  phase     latency   bound            util    t_dram   t_compute   t_fixed
 ─────────────────────────────────────────────────────────────────────────
  prefill   97.9 ns   COMPUTE_BOUND   0.02%   31.4 ps     97.9 ns       0 s

  TTFT      97.9 ns
  total     97.9 ns
  achieved  53.9 GOP/s of 312 TOP/s (0.02%)
```

664 parameters, 5280 operations at the defaults — the golden numbers that `tests/unit/test_kernels.py`
and the bundled `single_layer_encoder_toy` profile both pin to (`docs/CORRECTIONS.md` D39): the CLI
probe and the frozen teaching profile agree exactly because they run through the same
`encoder_layer_kernel` factory (`bwz/kernels.py`), the way `bwz matmul` runs through `matmul_kernel`.

Shape utilisation bottoms out at 0.02% — 8-wide operands do not begin to fill a 16×16 array — which
is the point of the default: every number here is checkable by hand, not representative of a real
workload.

### 3.2 Widening one dimension at a time

```bash
uv run bwz encoder-layer --chip a100_80gb --ideal --ffn 32   # 920 params, 7392 ops
uv run bwz encoder-layer --chip a100_80gb --ideal -S 16      # 664 params, 29184 ops
```

`--ffn` only moves the FFN block's own count (256 → 512 params; per-layer 528 → 784; total 664 →
920). `-S` moves no parameter count at all — sequence length is a deployment quantity, not a weight
— but every op count grows with it, and not linearly: 4 → 16 tokens is 4× the sequence but 5280 →
29184 is 5.5× the operations, because attention's `Q·Kᵀ` and `A·V` scale with `S²` while the
projections and FFN stay linear in `S`.

Both variants keep the same shape-derived `id`/`name` construction as §3.1 — `encoder_layer_d8_h2_ffn32_s4`,
never a placeholder.

---

## 4. `bwz run` — a network on a chip

| Flag | Meaning |
|---|---|
| `-m`, `--model` / `-c`, `--chip` | profile ids or paths (required) |
| `-b`, `--batch` | batch size (default 1) |
| `--input-tokens` / `--output-tokens` | prompt and generation length |
| `--context` | KV context; defaults to in + out |
| `--weights` | precision (sets weights, activations and KV alike) |
| `--phase` | `prefill` \| `decode` \| `both` |
| `--attention` | `vanilla` \| `flash2` \| `paged` \| `sliding_window` |
| `--show-ops N` | the N most expensive operations |
| `--ideal`, `--json` | as for `matmul` |
| `--stationarity`, `--split-k` | as for `matmul` (§2.5.1), and **not** inert here: every matmul in the graph is decomposed the same way |
| `--a-strategy`, `--b-dataflow`, `--a-residency-tiles`, `--a-prefetch-depth`, `--iterations` | accepted for parity with `matmul` (§2.5.2), but inert here — a network's graph is never one bare matmul, so `analysis/schedule.py` never reaches the single-matmul branch these read |

```bash
uv run bwz run -m llama3_8b -c a100_80gb --input-tokens 2048 --output-tokens 128
```

```
  TTFT      138 ms
  TPOT      9.49 ms   105.4 tok/s
  total     1.35 s
  achieved  22 TOP/s of 312 TOP/s (7.05%)
```

Prefill compute-bound, decode DRAM-bound — two different machines out of one model, which is why
they are costed as separate graphs.

The edge case from `docs/CORRECTIONS.md` D8:

```bash
uv run bwz run -m gemma3_4b -c chip_a --weights int8 --input-tokens 512 --output-tokens 1
```

```
  TPOT      115 ms   8.7 tok/s
  weight residency         1.40%
  double buffered            yes
```

55 MB of SRAM against a 3.88 G-parameter model holds 1.40% of the weights, so 98.6% of them cross
34 GB/s of LPDDR4x every token.

---

## 5. `bwz compare` — chips head to head

```bash
uv run bwz compare --chips chip_a,chip_b --models gemma3_4b
```

```
  model       chip     params   resident     TTFT   tok/s   bound            util
  gemma3_4b   chip_a   3.88 G      1.40%   114 ms     8.7   DRAM_BW_BOUND   0.03%
  gemma3_4b   chip_b   3.88 G     25.75%    87 ms    11.0   DRAM_BW_BOUND   0.17%

Prefill crossover
  gemma3_4b: no crossover in [1, 100000] — chip_b is faster throughout
```

Flags: `--chips`, `--models` (comma-separated), `--batch`, `--input-tokens`, `--output-tokens`,
`--context`, `--weights`, `--crossover/--no-crossover`, `--ideal`.

---

## 6. Figures

The figures come from **the same commands that print the numbers** — one command per workload, so
there is no second spelling of `-M`/`-N`/`-K` and no second set of defaults to keep in step (D55).
Add `--timeline` for the zoomable page, `--animate` for the playback, `--out` for where they go:

```bash
uv run bwz matmul -M 16 -N 16 -K 16 -c a100_80gb --ideal --timeline --animate --out ~/figs
uv run bwz run --model llama3_8b -c a100_80gb --input-tokens 512 --timeline --out ~/figs
uv run bwz encoder-layer -c a100_80gb -S 512 --timeline --animate --out ~/figs
```

```
wrote ~/figs/timeline-a100_80gb-fp16-os.html
wrote ~/figs/animate-a100_80gb-fp16-os.html
```

**The name ends with the decomposition the page draws** — the same
`-<stationarity>[-splitk<N>]` the emitted program carries, so `--stationarity is` cannot land on
top of `os`'s page and a directory of them reads as a comparison (D65). A `--compare-with` page
whose chips run *different* native dataflows has no single decomposition to name and keeps the
plain stem, which is the same condition that leaves its stationarity banner blank.

| flag | on | meaning |
|---|---|---|
| `--timeline` | `matmul`, `run`, `encoder-layer` | the zoomable page: where the time went, per hardware resource, with the roofline and the runnable loop nest below it |
| `--animate` | `matmul`, `encoder-layer` | a second page playing the same schedule back as DRAM → SRAM → Accelerator motion. Not on `run`: a full model's per-operation trace coalesces hundreds of operations, past what a station diagram can usefully show (D42) |
| `--compare-with CHIP` | all three | draw that chip alongside `--chip` on **one** page with a shared, absolute time axis. Repeatable. Refused with `--animate`, which plays one chip back |
| `--out DIR` | all three | where the pages go (default `.`) |
| `--steps N` | all three | resolution of the drawn trace (default 256) — the only such knob, since the page zooms rather than needing a second, coarser register (D37) |
| `-q`, `--quiet` | all three | drop the report table; the `wrote …` lines still print |

Everything else is the command's own: the shape flags, the dataflow strategy flags of §2.5, and
`--ideal`. That is the point — `bwz matmul --stationarity ws --timeline` draws the decomposition
the table above it was computed from, because there is only one analysis (D55).

**`--quiet` never silences an error.** An infeasible configuration and a refused flag combination
print regardless, since they are the answer rather than a table (CLAUDE.md #8). That is what makes
it safe for `make plots`, which passes it to avoid a wall of tables per chip.

**`--compare-with` insists on one precision every chip supports**, and says which they share when
they do not — per-chip defaults would put a run that moved half the bytes on the same time axis as
one that did not (D29, §6.1):

```bash
uv run bwz matmul -M 512 -N 512 -K 512 -c a100_80gb -d fp16 --compare-with metis_aipu --timeline
```

```
bwz: metis_aipu has no fp16 datapath, so --compare-with cannot run the same workload on every
chip. Supported by all: int8
```

### The PNG figures still have a script

`scripts/plot_roofline.py` is the one thing left in `scripts/`, because it is the one thing that
needs matplotlib — kept in its own `plots` dependency group so `make test` never pulls it in, and
so nothing in `bwz/` ever imports a plotting library. The HTML pages import nothing beyond the
standard library, which is why they could move into the package at all (D55).

```bash
uv run --group plots python scripts/plot_roofline.py --chip a100_80gb --model llama3_8b
uv run --group plots python scripts/plot_roofline.py --chip chip_a --weights int8 \
  --model gemma3_4b --tokens 512 \
  --matmul 512,4096,4096 --matmul 128,4096,4096 --matmul 1,4096,4096
```

`plot_roofline.py`: `--chip`, `--weights`, `--matmul M,N,K` (repeatable), `--model ID`
(repeatable), `--tokens`, `--out`.

```bash
make plots        # from the repo root — regenerates all of docs/plots/
```

### 6.1 `--compare-with` — two chips, one workload, one page

Run the command twice and each chip gets its own page, x normalised to that chip's own span. Add
`--compare-with` and they land in **one** page on a **shared, absolute** axis:

```bash
uv run bwz run --model gemma3_4b -c a100_80gb --compare-with metis_aipu \
  --weights int8 --input-tokens 512 --output-tokens 1 --timeline -q --out ../docs/plots
```

```
wrote ../docs/plots/timeline-compare-a100_80gb-vs-metis_aipu-gemma3_4b-prefill-int8.html
wrote ../docs/plots/timeline-compare-a100_80gb-vs-metis_aipu-gemma3_4b-decode-int8.html
```

Both views are kept on purpose (`docs/CORRECTIONS.md` D29). Absolute is the axis for "which is
faster and by how much"; normalised is the axis for "how is *this* machine's time distributed", and
at 13x the faster chip's whole run is 7% of a shared axis.

It works on all three commands and for more than two chips — the flag repeats:

```bash
bwz matmul -M 4096 -N 4096 -K 4096 -c a100_80gb -d int8 --compare-with metis_aipu --timeline
bwz encoder-layer -c a100_80gb --dmodel 4096 --nheads 64 --ffn 16384 -S 1024 \
  --compare-with metis_aipu --weights int8 --timeline
bwz matmul -M 2048 -N 2048 -K 2048 -c a100_80gb -d int8 \
  --compare-with metis_aipu --compare-with jetson_orin --timeline
```

A `bwz run` comparison pairs prefill against prefill and decode against decode, never across.

**Every chip runs the same workload at the same precision**, which has to be enforced rather than
assumed — A100 defaults to fp16 and Metis has no fp16 datapath, so per-chip defaults would compare
two different amounts of traffic. `--compare-with` refuses a precision any chip on the page cannot run:

```bash
uv run bwz matmul -M 512 -N 512 -K 512 -c a100_80gb -d fp16 --compare-with metis_aipu --timeline
```

```
bwz: metis_aipu has no fp16 datapath, so --compare-with cannot run the same workload on every
chip. Supported by all: int8
```

`--animate` is refused alongside it, since a playback follows one chip's schedule:

```
bwz: --animate plays back one chip; drop --compare-with or drop --animate.
```

Rows are **banded by chip**, not aligned across them: A100 declares 3 memory levels and 2 compute
units, Metis 4 and 2, `chip_a` 2 and 1, and no correspondence between `cuda_core` and `dpu` exists
to draw. Each band opens with a header row carrying that machine's peak, DRAM bandwidth, on-chip
capacity, its row counts, its total span **and its achieved throughput** — `436 TOP/s achieved ·
70% of peak` on A100 against `186 TOP/s achieved · 89% of peak` on Metis for an 8192³ INT8 matmul,
the same 2.34x the latency ratio is (2.52 ms against 5.9 ms), inverted, because both come from the
one reported latency (D35). Note which chip is nearer *its own* ceiling: A100 is at 70% because the
unfitted `DEFAULT_ACHIEVED_FLOPS_FRACTION` says so, Metis at 89% because its shape utilisation says
so — the same figure meaning two different things is exactly why the band states both. Everything the single-chip page does survives inside the band — grey rows for
declared-and-unused resources, matrix and vector lanes on separate rows, filled loads against
hollow stores, named bars, and the three info boxes, now one set per chip.

Below the two registers the comparison adds a **roofline register**: both chips' ceilings, ridge
points and M=1 lines on one chart, with each chip's workload point on it. Inside that panel colour
means *chip* rather than *resource*, which the panel says on itself. The arithmetic section is
rendered **once** — it is a property of the workload, and the workload is the same on both
machines — but the **deployment section is rendered once per chip**, because how the work reaches
the silicon is exactly what differs between them (D32/D54) — two loop nests, not one.

Each command draws its own workload, so the shape flags are the ones that command already has:
`bwz encoder-layer --dmodel …`, `bwz run --model …`. There is no way left to pass an encoder's
dimensions to a profile, because they are different commands.

`bwz run --timeline` draws a network, one page per phase — which is where the
`COMPUTED — matmul 98% · attention 2%` breakdown earns itself:

```bash
uv run bwz run --model llama3_8b -c a100_80gb --input-tokens 512 --timeline --out ~/figs
# timeline-a100_80gb-llama3_8b-{prefill,decode}-fp16.html
```

`--out` is relative to where you run the command, so from `backend/` a bare `--out ..` lands in the
repo root and from the repo root it lands *outside* the repo — with a `wrote ../timeline-….html`
line that looks right either way. Pass an absolute path when it matters.

### The zoomable page

```bash
cd backend
uv run bwz matmul -M 10000 -N 10000 -K 10000 -c a100_80gb --ideal --timeline \
  --out /absolute/path/you/want

xdg-open /absolute/path/you/want/timeline-a100_80gb-fp16-os.html
```

The time axis **zooms** (wheel, about the cursor), **pans** (drag) and **resets** (double-click),
and every bar names its transaction on hover — `LOAD — operands in`, `STORE — result written
back`, `EXEC — matmul` — with its bytes or operations and the rate. Below the timeline the page
carries the **roofline** for the same run: both ceilings, the ridge point, the M=1 tail, and the
workload as a labelled point. One self-contained file: no server, no port, no download, no CDN —
`file://` is enough.

`xdg-open` prints nothing and hands the file to a browser that may already be running, so look for
a new **tab in an existing window**. `google-chrome <file>` or `firefox <file>` work too.

`--steps` (default 256) is the trace's only resolution knob — there is no separate register to keep
legible at a fixed scale, because the page zooms instead.

The timeline gives every declared memory level and compute unit its own row, with the bytes moved,
the achieved bandwidth and the operations retired written beside it — and draws grey the resources
this model never uses. See [`plots/README.md`](plots/README.md) for how to read it.

---

## 7. Development

```bash
make test          # pytest + vitest
make test-fast     # pytest -m "not validation and not slow"
make lint          # ruff check + ruff format --check + mypy
make fmt           # ruff format + ruff check --fix
make validate      # predicted-vs-published suite (nothing calibrated yet)
```

Single test:

```bash
cd backend && uv run pytest tests/unit/test_matmul_workload.py -q
cd backend && uv run pytest tests/unit/test_pipeline.py::test_double_buffering_holds_exactly_two_tiles -q
```

---

## 8. Where each claim is derived

| Claim | Command | Derivation |
|---|---|---|
| 6.42 ms, compute-bound by 21.8× | §2.1 | `docs/MODEL.md` §5b, §6.3 |
| 4.52% at M=1 (`1/16` array x `72.3%` occupancy) | §2.2 | `docs/MODEL.md` §6.1 |
| result width changes bytes only | §2.3 | `docs/CORRECTIONS.md` D18 |
| mixed operands run at the wider | §2.3 | `docs/CORRECTIONS.md` D18 |
| SRAM depth 1.97 of 2 buffers | §2.4 | `docs/MODEL.md` §6.5, D19 |
| the result is always written back | §2.1, §2.3 | `docs/CORRECTIONS.md` D22 |
| grey rows = the model's boundary | §6 | `docs/CORRECTIONS.md` D20 |
| no Konata, no Kanata | — | `docs/CORRECTIONS.md` D21 |
| 1.40% residency on chip_a | §4 | `docs/CORRECTIONS.md` D8, D15 |
| traffic is a lower bound when the working set does not fit | every report's assumptions | `docs/MODEL.md` §6.2 |
| stream is 16x the staged total | §2.5.2 | `docs/MODEL.md` §6.3a, `docs/CORRECTIONS.md` D36 |
| whole/persistent clamp rather than raise | §2.5.2 | `docs/CORRECTIONS.md` D36, CLAUDE.md #8 |
| the matrix cores decompose output-stationary | §2.5.1 | `docs/MODEL.md` §6.1, `docs/CORRECTIONS.md` D53 |
| an unsupported stationarity is refused, not clamped | §2.5.1 | `docs/CORRECTIONS.md` D53, CLAUDE.md #8 |
| split-K trades DRAM traffic for wave occupancy | §2.5.1 | `docs/MODEL.md` §6.1, `docs/CORRECTIONS.md` D53 |
| 436/186 TOP/s achieved, 2.34x inverted | §6.1 | `docs/CORRECTIONS.md` D35, D37 |
| the emitted program's tier-1 counts | §2.6 | `docs/MODEL.md` §6.8, `docs/CORRECTIONS.md` D54 |
| B fetched 756 MB against 12 MB compulsory and 0 charged | §2.6 | `docs/MODEL.md` §6.2, `docs/CORRECTIONS.md` D54 |
