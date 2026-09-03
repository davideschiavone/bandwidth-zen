# bandwidth-zen

**Can I run this model on this chip — and if so, how fast?**

`bandwidth-zen` is an analytical performance model for neural-network inference. Give it a model
(transformer LLM or CNN) and a chip (or a multi-chip system), and it predicts latency, throughput,
utilization, memory footprint, and energy — then tells you **what the bottleneck is and what to do
about it**, with the numbers behind every claim.

It is fast enough to explore hundreds of configurations interactively, and honest enough to show you
every assumption it made.

> ⚠️ **These are estimates, and they have not been validated yet.** Every prediction rests on
> calibration constants that are documented defaults, not values fitted against any measurement —
> so no report claims better than `low` confidence, and the error against real hardware is
> currently **unknown rather than bounded**. See [Accuracy](#accuracy) and `docs/CALIBRATION.md`.

---

## What it does

- **Decomposes** a model into a DAG of operations with FLOP and byte counts per memory level
- **Maps** each operation onto the chip: which compute unit, which tile grid the chip's dataflow
  implies, what fits in SRAM
- **Predicts** latency via a flat roofline (compute ridge vs DRAM ridge; the on-chip SRAM enters
  as a tile-buffer capacity) plus tail-effect and pipeline-fill utilization modelling
- **Separates prefill from decode** for LLMs — they are different machines, and the tool shows why
- **Declares parallelism**: tensor / pipeline / data / expert sharding degrees are validated in the
  deployment spec today; multi-chip collective costs (alpha-beta over NVLink/InfiniBand topologies)
  land at M5 — every number this README quotes is a single-chip result
- **Plans memory**: weights + KV cache + peak live activations + workspace vs. device capacity,
  including the context length at which you hit the memory wall
- **Explains**: ranked bottlenecks, roofline position per op, resource timeline, and quantified
  suggestions ("quantize to INT8: −38% latency" — re-simulated, not guessed)
- **Compares**: chips head-to-head on the same model and precision, with the prefill-crossover
  point where a slower-but-cheaper chip catches up

## What it is not

Not a cycle-accurate simulator, not a compiler, not a benchmark harness. It doesn't run your model.
It won't capture kernel-selection quirks, thermal throttling, or your framework's overhead. Use it
for design-space exploration and sanity-checking, then measure on real hardware.

---

## Quickstart

**Environment setup (do this first after cloning).** The backend requires Python ≥ 3.11 and is
managed with [uv](https://docs.astral.sh/uv/) — do not use `python -m venv`; uv downloads its own
Python 3.11 interpreter if the system one is older:

```bash
git clone https://github.com/<you>/bandwidth-zen && cd bandwidth-zen
make venv         # creates backend/.venv (uv venv --python 3.11 + uv sync --all-groups)
```

Then either prefix commands with `uv run` from `backend/` (recommended — no activation needed),
or activate classically with `source backend/.venv/bin/activate`.

**Every command documents itself — this is the fastest way to see current flags**, grouped and
validated, rather than trusting this README's snapshot of them:

```bash
uv run bwz --help                                       # the subcommands, in typical order
uv run bwz matmul --help                                 # every matmul flag, incl. dataflow strategy
uv run bwz matmul --help                                # add --timeline / --animate to any report command
```

**[`docs/CLI.md`](docs/CLI.md) is the full command reference** — every flag, and the exact
invocation behind every number quoted in this README.

```bash
make dev          # backend on :8000, UI on :5173  (no-op until M4 — backend-only phase)
```

Or from the CLI:

```bash
uv run bwz run \
  --model llama3_8b --chip h100_sxm \
  --batch 1 --input-tokens 2048 --output-tokens 256 \
  --weights fp16 --attention flash2
```

```
                     Llama-3-8B on NVIDIA H100 SXM5 80GB

  phase     latency   bound             util    t_dram   t_compute   t_fixed
 ────────────────────────────────────────────────────────────────────────────
  prefill   44.6 ms   COMPUTE_BOUND   67.26%   5.25 ms     43.5 ms    774 µs
  decode    6.05 ms   DRAM_BW_BOUND    0.27%   5.24 ms      642 µs    774 µs

  TTFT      44.6 ms
  TPOT      6.05 ms   165.4 tok/s
  total     1.59 s
  achieved  18.7 TOP/s of 989 TOP/s (1.89%)

              Memory
  item                     bytes
 ────────────────────────────────
  weights                16.1 GB
  KV cache                302 MB
  peak activations        265 kB
  total                  16.4 GB
  usable DRAM              72 GB
  on-chip                83.8 MB
  weight residency         0.52%
  double buffered            yes

Why  (confidence: low)
  • Compute-bound by 8.29x over DRAM_BW_BOUND; DRAM takes over above 5.74 POP/s of
    effective throughput. (rests on an estimated input)
  • DRAM-bound by 6.78x over LATENCY_BOUND; latency bound takes over above 19.3 TB/s
    of effective bandwidth. (rests on an estimated input)
  → Shape utilisation bottoms out at 5.93%: operands do not fill the (16, 16) array.
    Larger batches or fused projections help.
  → Reduce bytes moved: quantise the weights further, or raise the batch size so each
    weight read serves more tokens.
  → Utilisation is 0.27% of peak — the compute array is nearly idle. A cheaper chip
    with the same bandwidth would perform identically.
```

**Two phases, two different machines.** Prefill is compute-bound at 67% of an H100; decode is
DRAM-bound at 0.27%, because a batch-1 token has to drag all 16.1 GB of weights across the bus to
produce one row of arithmetic. The `Why` block is not commentary — each line is a **flip margin**,
the factor by which the binding input would have to move before the verdict changes, and it says
which of them rest on a calibration constant nobody has fitted. That is why the confidence is
`low` and stays `low` until `docs/CALIBRATION.md` has reference points in it.

Add `--show-ops N` for the N most expensive operations, `--json` for the raw `Report`.

Interrogate the machine rather than a network — one matrix multiply, in matmul vocabulary
(operands A and B, result C — no weights, no activations, no batch or context knobs; `M` folds the
batch in):

```bash
uv run bwz matmul -M 10000 -N 10000 -K 10000 --chip a100_80gb --dtype fp16 --ideal
```

```
       A 10000x10000 fp16  x  B 10000x10000 fp16  ->  C 10000x10000 fp16
                          on NVIDIA A100 SXM4 80GB

  operations                2 TOP    2 x 10000 x 10000 x 10000 — unchanged by the result width
  arithmetic runs at        fp16     tensor_core peak 312 TOP/s
  intensity        3333.3 OP/byte    operations / compulsory traffic
  ridge point       153.0 OP/byte    above it the chip is compute-bound
  stationarity                os     C resident, 625 x 625 tiles (M x N) each sweeping K
  shape utilisation        99.91%    shape padded to the 16x16 tile — geometry, not a derating
  DRAM reads               400 MB    A and B, less whatever stays on chip
  DRAM writes              200 MB    C in full — nothing on chip consumes it
  t_dram                   294 µs
  t_compute               6.42 ms
  latency                 6.42 ms
  verdict           COMPUTE_BOUND
```

`--ideal` sets both efficiency de-ratings to 1.0, so every number above can be checked against the
datasheet by hand. Drop `-M` to 1 and the same matmul reports **4.52%** utilisation and flips to
DRAM-bound: one row of work pays for a whole 16-row instruction tile (`1/16` of the array), and the
625 output tiles that remain cannot fill 432 tensor cores evenly. Both losses are geometry, not a
fudge factor, and `--ideal` does not remove either.

**To see that same run instead of reading it** — one row per hardware resource, zoomable:

```bash
uv run bwz matmul -M 10000 -N 10000 -K 10000 --chip a100_80gb --ideal --timeline --out ..
xdg-open ../timeline-a100_80gb-fp16.html
```

More in [Figures](#figures).

Widths are per operand, and the result width is the **accumulator** width — it changes bytes only,
never operations:

```bash
uv run bwz matmul -M 4096 -N 4096 -K 4096 -c a100_80gb --dtype int8               # all int8
uv run bwz matmul -M 4096 -N 4096 -K 4096 -c a100_80gb --dtype int8 --c int32     # int32 accumulate
uv run bwz matmul -M 4096 -N 4096 -K 4096 -c a100_80gb --dtype fp16 --c fp32      # all float
uv run bwz matmul -M 4096 -N 4096 -K 4096 -c a100_80gb --a fp16 --b int8          # mixed operands
```

All four do the same 137.4 GOP. The int32 result quadruples C from 16.8 MB to 67.1 MB and halves
the arithmetic intensity; the mixed-operand case runs at the **fp16** rate, because both operands
share one datapath — the narrow side saves bytes and buys no throughput.

Put two chips head to head on the same model:

```bash
uv run bwz compare --chips chip_a,chip_b --models gemma3_4b
```

```
                      Head to head @ S=512, batch 1, int8

  model       chip     params   resident     TTFT   tok/s   bound            util
  gemma3_4b   chip_a   3.88 G      1.40%   114 ms     8.7   DRAM_BW_BOUND   0.03%
  gemma3_4b   chip_b   3.88 G     25.75%    87 ms    11.0   DRAM_BW_BOUND   0.17%

Prefill crossover
  gemma3_4b: no crossover in [1, 100000] — chip_b is faster throughout
```

Docker:

```bash
docker compose up      # http://localhost:5173
```

---

## Commands

Every command, from `backend/`. `bwz --help` and `bwz <command> --help` list the flags; this is
what each one is *for*.

| | |
|---|---|
| `bwz list` | the bundled chip and model profiles, with peaks, shapes and provenance |
| `bwz matmul` | one `A[M,K] × B[K,N] → C[M,N]` — the smallest probe of a machine |
| `bwz encoder-layer` | one encoder layer, sized from the command line |
| `bwz run` | a model profile on a chip |
| `bwz compare` | chips head to head, with the prefill crossover |
| `bwz version` | print the installed `bwz` version |

### `bwz matmul` — the shape is M, N, K

```bash
uv run bwz matmul -M 10000 -N 10000 -K 10000 --chip a100_80gb --dtype fp16 --ideal
```

`-M -N -K` the dimensions · `-c/--chip` · `-d/--dtype` both operands · `--a`/`--b` per-operand
widths · `--out` the **result** width, i.e. the accumulator · `--ideal` · `--pipeline/--no-pipeline`
· `--stationarity`/`--split-k` and `--a-strategy`/`--b-dataflow` with
`--a-residency-tiles`/`--a-prefetch-depth`/`--iterations` — dataflow, next two sections · `--json`.

`M` folds the batch in — a batch of 128 rows is `-M 128`. There is no `--batch`, no context and no
phase, because one matmul has none of those.

```bash
uv run bwz matmul -M 4096 -N 4096 -K 4096 -c a100_80gb --dtype int8               # all int8
uv run bwz matmul -M 4096 -N 4096 -K 4096 -c a100_80gb --dtype int8 --c int32     # int32 accumulate
uv run bwz matmul -M 4096 -N 4096 -K 4096 -c a100_80gb --dtype fp16 --c fp32      # all float
uv run bwz matmul -M 4096 -N 4096 -K 4096 -c a100_80gb --a fp16 --b int8          # mixed operands
```

All four do the same 137.4 GOP. The result width changes bytes only, never operations; a mixed
matmul runs at the **wider** operand, because both share one datapath.

### Stationarity: which operand stays resident

**The first dataflow question, and the one the other two live inside.** Which operand stays put
decides the whole decomposition: which dimensions form the parallel tile grid, which one each tile
sweeps, and whether partial sums have to be reduced afterwards. For a matmul the operands map as
**input = A, weight = B, output = C**.

| `--stationarity` | resident | parallel grid | swept per tile | reduction |
|---|---|---|---|---|
| `os` output-stationary | C's accumulator | `⌈M/rows⌉ × ⌈N/cols⌉` | K | none |
| `ws` weight-stationary | a B tile | `⌈K/rows⌉ × ⌈N/cols⌉` | M | over K |
| `is` input-stationary | an A tile | `⌈M/rows⌉ × ⌈K/rows⌉` | N | over K |
| `rs` row-stationary | one A row per PE | `⌈M/rows⌉ × ⌈N/cols⌉` | K, spread spatially | inside the array |

**The default is the chip's own.** Every matrix core here runs `os` natively, because that is what
cuBLAS and CUTLASS do — K accumulates in registers inside one output tile, so no partial sum ever
leaves a core. All four also *declare* `ws` and `is` so the three can be compared on one chip, which
moves no default. The in-memory-compute profiles declare `ws` alone: their weights *are* their
memory, so an accumulator-resident dataflow is not something they could run. `rs` is implemented
after Eyeriss and labelled unvalidated, since no shipped profile declares it.

**Every stationarity issues the same `2·M·N·K` operations.** They differ in how the work is cut up —
so in quantisation loss, in how the operands are staged, and in whether a reduction is owed.

**Where the partials meet is the hardware's answer, not the dataflow's.** A K-on-the-grid grid owes
`(p−1)·M·N` additions, and what they cost depends on whether the unit declares an accumulator deep
enough to sum them itself (Metis: 16 384 inputs, free), whether they fit on chip (the vector unit
adds them, *overlapped* with the matrix work), or neither (a DRAM round trip, serialised). On A100
at 4096³ that overlap hides the reduction entirely — `ws` costs the same latency as `os` — and what
it really costs is 16.8 M live accumulators, which at 8192×8192 stop fitting and turn 2.52 ms into
39.8 ms. `docs/MODEL.md` §6.1 has the table.

**A chip asked for one it cannot run is refused, not clamped** — the one place this tool returns
`feasible: false` for a *strategy* rather than a capacity problem:

```bash
$ uv run bwz matmul -M 512 -N 512 -K 4096 -c a100_80gb --stationarity rs
Infeasible.
  • stationarity='rs' is not supported by tensor_core, which declares is, os, ws. Drop the flag
    to use the chip's own dataflow, or pick one it declares.
```

The A/B knobs below clamp instead, and the difference is deliberate: `stage`/`stream`/`whole` are
orderings of the same work, so falling back still answers the question you asked. A stationarity is
a *different decomposition*, so substituting one would quietly report a number for hardware you
never asked about.

**`--split-k` cuts K so that one output tile becomes several units of work.** Under `os` a core owns
an output tile and sweeps the whole contraction inside it, so the parallelism available is the
number of output tiles and nothing else — when that is smaller than the chip, most of the chip idles
however long K is. Cutting K into `p` chunks makes `(tile, chunk)` the unit of work and gives `p`×
as many. The arithmetic is unchanged; what changes is that the `p` chunks each hold a *partial*
value of the same output, and those must be added. That is the one thing that makes `os` owe a
reduction: CUTLASS runs it as two kernels — a partitioned GEMM and a batched reduction — and the
model charges both.

```bash
uv run bwz matmul -M 512 -N 512 -K 4096 -c a100_80gb --ideal --split-k 8
```

|  | tiles | waves | occupancy | DRAM traffic | latency |
|---|---|---|---|---|---|
| default | 1 024 | 3 | 79.0% | 4.72 MB | 8.71 µs |
| `--split-k 8` | 8 192 | 19 | 99.8% | **13.1 MB** | 6.99 µs |

The extra 8.4 MB is eight full `512×512` partials written by the first kernel and read back by the
second. Push the factor higher and `t_dram` overtakes `t_compute` — the flag is there to find where.
Full derivation in `docs/CLI.md` §2.5.

### A and B: how a matmul's operands move

Two more flags on `bwz matmul` (and `bwz run`), answering two different
questions about a lone matmul's DRAM traffic *within* the grid the stationarity chose — full
derivation in `docs/CLI.md` §2.5:

```bash
uv run bwz matmul -M 8192 -N 8192 -K 8192 -c metis_aipu -d int8 --ideal --a-strategy stream
```

**`--a-strategy` is a byte-amount knob** — how often A crosses DRAM. `stage` (default) reads one
*grid row's* slice of A once and feeds every tile of that row; `stream` re-reads A per tile instead
— the honest picture for a GPU, whose GEMMs genuinely refetch operands — which is **16x** the
traffic on this shape (1.07 GB against 67.1 MB staged); `whole` stages all of A before the first
tile, same bytes as `stage`, only the timing changes. (What a "row" is follows from the
stationarity: a slice of K under `ws`, a band of M rows under `os`. Either way A crosses DRAM
exactly once under `stage`.)

**`--b-dataflow` is a timing knob** — B is fetched exactly once either way (D30), so
`write-ahead`/`on-demand`/`persistent` only move *when* the write lands relative to compute, never
how many bytes cross. The one exception is `--iterations N > 1` with `persistent`: it amortises B's
write over a resident weight set that a repeat invocation would not have to rewrite.

Both `whole` and `persistent` **clamp rather than error** when they don't fit — `whole` needs the
scratchpad to hold all of A, `persistent` needs the array to hold all of B — falling back to
`stage`/`write-ahead` and naming the fallback in the assumptions drawer, never silently.

### `bwz encoder-layer` — the shape is the dimensions

The transformer counterpart: arguments rather than a profile, so one term can be changed and its
effect read off.

```bash
uv run bwz encoder-layer --chip a100_80gb --ideal             # 664 params, 5280 ops
uv run bwz encoder-layer --chip a100_80gb --ideal --ffn 32    # 920 params, 7392 ops
uv run bwz encoder-layer --chip a100_80gb --ideal -S 16       # 664 params, 29184 ops
```

`head_dim` is never a separate input — always `dmodel // nheads` — so a pair that doesn't divide
evenly is rejected outright rather than silently floored:

```bash
$ uv run bwz encoder-layer --chip a100_80gb --ideal --dmodel 100 --nheads 6
bwz: dmodel (100) is not divisible by nheads (6); choose a head count dividing 100 evenly
```

`--ffn`'s own default is conditional: 16 only when `--dmodel` is also left at its default of 8;
pass `--dmodel` explicitly with no `--ffn` and it becomes `4 x --dmodel` instead (an explicit
`--ffn` always overrides either way) — the bare command still has to print the hand-countable
664 params / 5280 ops above, so the 4x convention only kicks in once `--dmodel` was itself a choice.

`--dmodel -d` · `--nheads` · `--ffn` · `--vocab` · `-S/--tokens` · `-b/--batch` ·
`--ffn-type` · `--norm` · `--tie/--untie` · `--weights` · `--ideal` · `--show-ops` · `--json`.

It leads with a table of *where* the parameters are, not just the total. One layer always — that is
the name; for anything deeper, write a profile and use `bwz run`. No KV cache and no LM head: an
encoder has no later step to reuse a cache for.

### `bwz run` — a profile on a chip

```bash
uv run bwz run --model llama3_8b --chip a100_80gb --input-tokens 2048 --output-tokens 128
uv run bwz run --model single_layer_encoder_toy --chip a100_80gb --input-tokens 4 --show-ops 20 --ideal
```

`-m/--model` · `-c/--chip` · `-b/--batch` · `--input-tokens` · `--output-tokens` · `--context` ·
`--weights` · `--phase` · `--attention` · `--show-ops N` · `--ideal` · `--json` ·
[`--stationarity`/`--split-k`](#stationarity-which-operand-stays-resident), which **do** apply here
— every matmul in the graph is decomposed the same way — plus the [A/B strategy
flags](#a-and-b-how-a-matmuls-operands-move), accepted for parity with `bwz matmul` but inert on a
network, whose graph is never one bare matmul.

`--show-ops N` lists the N most expensive operations with their arithmetic, DRAM bytes and time —
`20` simply asks for more lines than the encoder's 14 operations.

### `bwz compare` — chips head to head

```bash
uv run bwz compare --chips chip_a,chip_b --models gemma3_4b
```

```
                      Head to head @ S=512, batch 1, int8

  model       chip     params   resident     TTFT   tok/s   bound            util
  gemma3_4b   chip_a   3.88 G      1.40%   114 ms     8.7   DRAM_BW_BOUND   0.03%
  gemma3_4b   chip_b   3.88 G     25.75%    87 ms    11.0   DRAM_BW_BOUND   0.17%

Prefill crossover
  gemma3_4b: no crossover in [1, 100000] — chip_b is faster throughout
```

`--chips` · `--models` (both comma-separated) · `-b/--batch` · `--input-tokens` · `--output-tokens`
· `--context` · `--weights` · `--crossover/--no-crossover` · `--ideal`.

Every chip runs the same model at the same precision, so the comparison is apples to apples. The
crossover search bisects for the prompt length at which prefill (TTFT) swaps which chip is faster:
each chip pays a fixed weight-load cost plus compute that grows with prompt length, so if the two
curves cross it happens at most once, reported as a token count rather than a fabricated one when
they don't. `chip_b` wins throughout here, so there is none to report.

### `--ideal`

Zeroes every unfitted calibration constant: both efficiency de-ratings **and** the per-dispatch
overhead. What is left follows from published quantities — MACs, clock, bandwidth, capacity — so
the numbers can be checked against a datasheet by hand. Shape utilisation is *not* disabled: a
batch-1 GEMM on a 512×512 array still runs at 1/513 of peak, because that is geometry rather than a
fudge factor.

---

## Figures

**The same run, as numbers and then as a picture.** Three lines, from `backend/`:

```bash
# 1. the numbers
uv run bwz matmul -M 10000 -N 10000 -K 10000 --chip a100_80gb --dtype fp16 --ideal

# 2. the picture of that same run — same command, one more flag
uv run bwz matmul -M 10000 -N 10000 -K 10000 --chip a100_80gb --ideal --timeline --out ..

# 3. open it
xdg-open ../timeline-a100_80gb-fp16.html
```

`--out ..` puts them in the repo root, where they are gitignored. `xdg-open` prints nothing and
hands the file to a browser that may already be running, so look for a **new tab in an existing
window**.

### `--timeline` — where the time went

It is a flag on the command that already prints the numbers, so there is one place to say what to
run and one set of defaults (D55):

```bash
# a matmul
bwz matmul -M 10000 -N 10000 -K 10000 -c a100_80gb --ideal --timeline --out ..

# a profile — one page per phase
bwz run --model llama3_8b -c a100_80gb --input-tokens 512 --timeline --out ..

# an ad-hoc single-layer encoder
bwz encoder-layer -c a100_80gb --dmodel 4096 --nheads 64 --ffn 16384 -S 4096 --ideal \
    --timeline --out ..
```

The shape and dataflow flags are the ones that command already had — `--stationarity`,
`--split-k`, `--a-strategy` and the rest ([what they
mean](#stationarity-which-operand-stays-resident), `docs/CLI.md` §2.5) draw the decomposition the
table above them was computed from, because there is only one analysis. `--animate` adds the
playback page (matmul and encoder; a full model's per-operation trace is past what a station
diagram can show). `--out` · `--steps` (default 256 — the only resolution knob; the page zooms) ·
`-q/--quiet` (drop the table, keep the `wrote …` lines) apply to all three.

### `--compare-with` — two chips, one workload, one picture

```bash
bwz run --model gemma3_4b -c a100_80gb --compare-with metis_aipu \
    --weights int8 --input-tokens 512 --output-tokens 1 --timeline
```

The time axis is **shared and absolute**, so a bar three times as long took three times as long —
here A100 8.95 ms against Metis 119 ms, stated as 13.26x rather than left to be measured off the
ticks. Rows are **banded by chip** rather than aligned, because the two profiles declare different
resources (A100: 3 memory levels, 2 engines; Metis: 4 and 2) and no correspondence between
`cuda_core` and `dpu` exists to draw. Each band header also states its chip's **achieved
throughput** — `436 TOP/s achieved · 70% of peak` against `186 TOP/s achieved · 89% of peak` on an
8192³ INT8 matmul, the same 2.34x the latency ratio is, inverted. Everything the per-chip figure
does survives inside each band.

Below the timeline, both rooflines on one chart. Both chips do the same 3.33 TOP over the same
~3.8 GB — ~870 OP/byte — and land on **opposite sides of their own ridge point**: A100's ridge is
252 OP/byte so it is compute-bound, Metis's is 6145 so it is DRAM-bandwidth-bound. Identical
arithmetic, identical traffic, different limiter. That is what the head-to-head table in
`bwz compare` cannot show.

Works with `--matmul`, `--model` and `--encoder`, and with more than two chips. Every chip runs the
same precision — enforced, not assumed, since A100 defaults to fp16 and Metis has no fp16 datapath
at all. Without `--compare` each chip still gets its own figure with x normalised to its own span;
both views are kept because they answer different questions.

### `plot_roofline.py` — where the workload sits

```bash
… plot_roofline.py --chip a100_80gb --model llama3_8b
… plot_roofline.py --chip chip_a --weights int8 --model gemma3_4b --tokens 512 \
    --matmul 512,4096,4096 --matmul 128,4096,4096 --matmul 1,4096,4096
```

`--chip` · `--weights` · `--matmul M,N,K` (repeatable) · `--model ID` (repeatable) · `--tokens` ·
`--out`.

### What gets written

```bash
make plots        # → docs/plots/
```

All computed by calling `analyze()` rather than drawn by hand, each carrying the command that
produced it and the commit it came from:

| | |
|---|---|
| `roofline-<chip>-<dtype>.png` | the two ceilings, the ridge point, the M=1 line, and workloads placed on them |
| `machine-<chip>.png` | the three-element machine — which link carries a bandwidth number and which does not |
| `timeline-<chip>-<dtype>.html` | where the time went, one row per hardware resource, **zoomable**, with the roofline and the arithmetic below it |
| `timeline-compare-<a>-vs-<b>-….html` | two chips, one workload, one shared absolute axis, both rooflines |
| `animate-<chip>-<dtype>.html` | the same schedule played back — blocks moving between stations, the loop nest lighting up debugger-style, and the A/B/C tile geometry (`--animate`) |

**None of them is committed.** They are outputs of the engine, regenerable in one command, and a
750 kB PNG per run is churn nobody can review (the timeline dropped its PNG form entirely for this
reason — it could not zoom, so it needed a second "first N steps" figure just to stay legible, and
the HTML page needs none of that, D37). `make plots` writes the set above into `docs/plots/`, which
`.gitignore` covers; the README that documents the set stays tracked.

**Rows are hardware resources**, read off the chip profile — every memory level and every compute
unit it declares, each with its own quantity: bytes moved and at what rate, operations retired and
at what fraction of peak, how much the buffers hold. `tensor_core` carries the matrix work and
`cuda_core` the norms and activations, because a tensor core does matrix-multiply-accumulate and
nothing else.

Grey rows are declared and unused by this model, which puts its boundary on the page instead of
hiding it: A100 has 40 MB of L2 the v1 roofline never spends. Put that next to `chip_a`, whose DRAM
row reads `LOAD 0 B · STORE 16.8 MB` — it fetches nothing because 55 MB of SRAM holds all of
operand B, and still writes the answer out — and the contrast is the architecture comparison.

The spans are a decomposition of the reported latency, not a second model: DRAM busy sums to
`t_dram`, the compute rows to `t_compute + t_fixed`.

### The zoomable page

One self-contained file — no server, no download, no CDN. Wheel zooms about the cursor, drag pans,
double-click resets. Hovering a bar names the transaction: `LOAD — operands in`, `STORE — result
written back`, `EXEC — matmul`, `HOLD — on chip`, each with its bytes or operations and the rate.

Below the timeline it carries the **roofline** for the same run; below that **how it is deployed on
the chip**, which for a matmul is the decomposition written out as a **runnable Python program**
(`bwz matmul --emit`, D54) that walks the same grid, counts what it moves and asserts those counts
against the page's own numbers; and below that **the arithmetic operation by operation** — operand
shapes, the algebra, the flop count as an expression (`2·M·N·K = 2·4·8·8 = 512`) and a pseudo-C loop
nest with the real extents, so the model can be back-tested against code rather than trusted.

**The figures are not the report.** `bwz matmul`, `bwz run` and `bwz encoder-layer` print the
numbers, their derivations and the assumptions drawer; the plot scripts write files and print only
`wrote …`. [`docs/plots/README.md`](docs/plots/README.md) covers how to read each figure and
[`docs/CLI.md`](docs/CLI.md) every flag with its real output.

---

## The model, briefly

Full derivations in [`docs/MODEL.md`](docs/MODEL.md). The core is a roofline with a compute
ceiling and a DRAM ceiling; the on-chip SRAM enters as a tile-buffer capacity that constrains
tiling:

```
AI          = FLOPs / bytes_moved_at_level
t_compute   = FLOPs / (peak_flops_per_s × utilization_efficiency)
t_memory    = bytes_moved / (bandwidth × bandwidth_efficiency)
t_op        = max(t_compute, t_memory)            # or sum, if the chip can't overlap
```

Five things would make it more than a textbook roofline. **Three are implemented and two are
not** — `docs/MODEL.md` says which formula is live:

1. **`utilization_efficiency` is derived, not assumed.** *(implemented)* Two independent losses,
   multiplied. **Shape**, the rule of multiples — `[K/padded(K)]·[N/padded(N)]·[M/padded(M)]` — is
   why a matmul with M=1 on a 128×128 array gets ~1/128 of peak, and why LLM decode looks the way it
   does. It is *area*: an instruction tile is issued whether or not its rows carry work, so one row
   pays for a whole tile. (It is **not** a pipeline fill — a tensor core has no M-serial pipeline to
   drain, which cost this project a correction, `docs/CORRECTIONS.md` D52.) **Wave occupancy** is
   the second: a chip with `count` arrays runs that many tiles at once, so an operation with fewer
   tiles than that leaves the rest idle and the last wave of any operation is partly empty. Pipeline
   fill/drain is folded into neither — the tile schedule reports it separately (§6.5), because the
   roofline's `max(load, compute)` is the many-tiles limit and hiding the difference inside a
   utilisation figure would make it unfalsifiable.
2. **The tile grid follows the chip's declared dataflow.** *(implemented)* Which operand stays
   resident decides which dimensions are cut, how many tiles there are, and whether partial sums
   need reducing — so it decides the wave-occupancy term above and the operand staging below.
   `analysis/stationarity.py` is the single place that decides it, and every site that needs a tile
   index goes through the grid it returns rather than re-deriving one. Selectable per run, refused
   rather than clamped when a chip cannot run the choice (`docs/CORRECTIONS.md` D53).
3. **On-chip capacity decides overlap and residency.** *(implemented)* Capacity is allocated
   double buffer → activations → weights, and whether two tiles fit is what earns
   `max(load, compute)` instead of `load + compute`.
4. **`bytes_moved` from a tile-reuse search.** *(not implemented — v1 charges compulsory traffic.)*
   The target is `M·K·⌈N/Tn⌉ + K·N·⌈M/Tm⌉ + M·N` over a search of tile sizes; v1 charges each
   operand once, which is the traffic of an ideal schedule and therefore a **lower bound** once the
   working set stops fitting on chip. Quantified in `docs/MODEL.md` §6.2 — 1.5–2× at 16384³ — and
   stated in every report's assumptions drawer. The same gap is why a weight-stationary grid's
   partial sums are assumed to meet in an accumulator rather than spilling: pricing one horn of that
   dilemma and not the other would be worse than naming both (D53).
5. **Communication with alpha-beta costs on the real topology.** *(not implemented — M5.)* Ring
   allreduce as `2(N−1)α + 2(N−1)/N·S·β`, separate `(α, β)` per link class.

Energy (M7) is not implemented either. Nothing in this repo has been calibrated against a published
measurement yet, which is why no report claims better than medium confidence.

---

## Built-in profiles

**Chips:** NVIDIA H100 SXM5 / A100 80GB / Jetson AGX Orin, AMD MI300X, the **Axelera Metis AIPU**,
and `chip_a` / `chip_b`, a generic edge NPU and a hypothetical variant of it. Every profile
describing a real product cites its source in `source_url`; the two that do not declare
`hypothetical: true`, and any field that is an engineering estimate rather than a published figure
names itself in `estimates:` and is propagated into the report's assumptions drawer.

`metis_aipu` is the odd one out and worth reading as the worked example of that policy: its numbers
come from an ISSCC 2024 paper rather than a datasheet, so compute and on-chip capacity are exact —
`512x512 MACs / 8 bit-serial cycles x 0.8 GHz x 2 x 4 cores = 209.7 TOPS` against the paper's
209.6, and 4x1 MiB + 4x4 MiB + 32 MiB = exactly the 52 MiB the paper states — while DRAM bandwidth
is not published at all and is an inference from a 64-bit LPDDR4x bus at the JEDEC maximum. Which
of its fields you can trust is written on the profile itself.

**Models:** GPT-3, BERT-base, Llama-3-8B, Llama-2-70B, Mistral-7B, Mixtral-8x7B, Gemma-4,
MobileNetV3, ViT-B/16, Stable Diffusion U-Net.

Plus **`single_layer_encoder_toy`** — a one-layer encoder sized so every number can be checked with a
calculator: 664 parameters, 5280 operations over 4 tokens, with the derivation in the profile's own
header.

```bash
uv run bwz run --model single_layer_encoder_toy --chip a100_80gb --input-tokens 4 --show-ops 20 --ideal
```

Or size one from the command line, the way `bwz matmul` takes M/N/K — the shape is arguments, so a
dimension can be changed and its effect read straight off:

```bash
uv run bwz encoder-layer --chip a100_80gb --ideal          # 664 params, 5280 ops
uv run bwz encoder-layer --chip a100_80gb --ideal --ffn 32 # 920 params, 7392 ops
uv run bwz encoder-layer --chip a100_80gb --ideal -S 16    # 664 params, 29184 ops
```

Flags: `--dmodel --nheads --ffn --vocab --tokens --batch --ffn-type --norm --tie/--untie
--weights --ideal --show-ops`. One layer always — that is the point; for anything deeper write a
profile.

`--show-ops 20` just asks for more lines than the 14 operations there are. `--output-tokens` and
`--phase` are not needed: an encoder has one phase, and the report says so in its assumptions if
you ask for generation anyway.

Add your own — chips and models are plain YAML:

```yaml
name: My NPU
clock_ghz: 1.2
compute_units:
  # `dataflow` is read, not decorative: it picks the tile grid every number below
  # is computed against. `ws` says this array holds a weight tile and streams M
  # past it; a matrix core that accumulates in registers declares `os` instead.
  - {name: systolic, count: 4, ops_per_cycle_per_unit: 16384,
     supported_dtypes: [int8, int4], systolic_dims: [128, 128], dataflow: ws}
memory:
  - {name: SRAM, level: 1, capacity_bytes: 67108864, bandwidth_bytes_per_s: 2.0e12, latency_ns: 5}
  - {name: LPDDR5, level: 3, capacity_bytes: 17179869184, bandwidth_bytes_per_s: 6.8e10, latency_ns: 120}
tdp_w: 60
```

See [`docs/SCHEMA.md`](docs/SCHEMA.md) for the full schema.

---

## Accuracy

**Nothing here has been checked against a measurement yet.** `make validate` runs, and currently
selects zero reference points:

```bash
$ make validate
459 deselected in 0.33s
```

That is the honest state, and it is why every report comes back `confidence: low` regardless of how
clean the arithmetic looks. Three calibration constants — DRAM bandwidth efficiency, achieved-FLOPs
fraction, per-dispatch overhead — are documented defaults rather than fitted values, and `--ideal`
exists precisely so you can see what a number looks like with all three removed. What remains under
`--ideal` follows from published quantities (MACs, clock, bandwidth, capacity) and can be checked
against a datasheet by hand; what `--ideal` removes is the part nobody has earned yet.

The **targets** for Session 5, against which the harness will be judged (`docs/PLAN.md`):

| Workload class | Target error | Status |
|---|---|---|
| LLM decode, single chip | ±15% | no reference point collected |
| LLM prefill, single chip | ±20% | no reference point collected |
| CNN inference, large batch | ±15% | no reference point collected |
| CNN inference, batch 1 | ±35% — launch overhead dominates | no reference point collected |
| Multi-chip TP | ±25% | not implemented (M5) |
| Energy | ±50% | not implemented (M7) |

What the engine *is* checked against today is **itself and its own physics**: 459 tests, including
the sanity checks in `CLAUDE.md` (doubling DRAM bandwidth never raises latency; INT8 never slower
than FP16; `M=1` on a 128×128 array is `1/128` of the array; every dataflow issues the same MAC
count), hand-computed goldens with the arithmetic written out in each test's docstring, and
cross-checks that two independent derivations agree — the graph builder's summed weight tensors
against `ModelSpec.parameter_count()`, the drawn schedule against the reported latency it
decomposes. That is self-consistency, not accuracy, and the difference matters.

Every report carries a `confidence` field, an assumptions drawer listing every shortcut taken, and
a **flip margin** per verdict: how far the binding input can move before the answer changes. A
bottleneck label with a 1.1× margin means something very different from one with a 260× margin, and
the label alone cannot tell you which you have.

---

## Project layout

```
backend/bwz/
  spec/         pydantic schemas + YAML loaders
  graph/        ModelSpec → operation DAG (transformer, CNN builders)
  operators/    per-family cost models (matmul, conv, attention, norm, elementwise)
  analysis/     roofline, stationarity (the tile grid), tiling, memory, schedule,
                parallelism, collectives, power, bottleneck
  emit/         one matmul's decomposition as a runnable, self-checking program
  api/, cli.py  thin shells over analyze(model, hardware, deployment) -> Report
  profiles/     chip and model YAML
backend/scripts/  figure generation (imports the engine; the engine never imports it)
frontend/src/   React + TS dashboard (roofline plot, Gantt, Pareto explorer)
docs/           CLI.md · MODEL.md · SCHEMA.md · CALIBRATION.md · CORRECTIONS.md · PLAN.md · plots/
```

The analysis core is pure and dependency-free: `analyze()` is a deterministic function of its
inputs, which is what makes sweeps parallelizable and results reproducible.

---

## Contributing

Contributions especially welcome for: new chip profiles (with datasheet citations), reference
measurements for `docs/CALIBRATION.md`, and operator cost models (state-space/Mamba, MoE routing,
diffusion attention).

```bash
make lint test        # ruff + mypy --strict + pytest + vitest, all must pass
```

House rules, in short: SI units internally with unit-suffixed names; every empirical constant lives
in `calibration.py` with a source comment; physics code ships with a golden test in the same commit;
every profile YAML carries a `source_url`; and no fabricated measurements, ever — "no reference
point available" is a perfectly good entry in the calibration table.

Details in [`CLAUDE.md`](CLAUDE.md) (which is also the operating manual if you use Claude Code here).

---

## License

Apache-2.0
