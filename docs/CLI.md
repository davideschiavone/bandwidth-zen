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
| `-d`, `--dtype` | width of both operands, and of the result unless `--out` (default `fp16`) |
| `--a`, `--b` | per-operand widths, when they differ |
| `--out` | result width — the **accumulator**. Defaults to the wider operand |
| `--ideal` | set both efficiency de-ratings to 1.0: a datasheet ceiling, not a prediction |
| `--pipeline` / `--no-pipeline` | lane occupancy table (default on) |
| `--a-strategy` | `stage` (default, D33) \| `stream` (D31) \| `whole` — how A is loaded, §2.5 |
| `--b-dataflow` | `write-ahead` (default, D33) \| `on-demand` \| `persistent` — when B's write lands |
| `--a-residency-tiles` | override tiles served per A staging event; power-of-2 divisor of `NTILES_PER_KS` |
| `--a-prefetch-depth` | override A's double-buffered staging depth (schedule-only) |
| `--iterations` | invocations this report represents; only `persistent` reads it |
| `--json` | the raw `Report` as JSON |

### 2.1 The datasheet check

```bash
uv run bwz matmul -M 10000 -N 10000 -K 10000 -c a100_80gb -d fp16 --ideal
```

```
  DRAM reads                   370 MB
  DRAM writes                  200 MB
  intensity            3333.3 OP/byte
  ridge point           153.0 OP/byte
  shape utilisation            99.84%
  t_dram                       279 µs
  t_compute                   6.42 ms
  latency                     6.43 ms
  verdict               COMPUTE_BOUND
```

`2·10000³ = 2.000e12` OP at A100's 312 TFLOP/s is 6.41 ms; the extra 0.16% is the systolic tail.
Every number here can be checked against the datasheet by hand — that is what `--ideal` is for.

Drop `--ideal` and the same command gives **9.18 ms**: the difference is `÷0.70`, the unfitted
`DEFAULT_ACHIEVED_FLOPS_FRACTION`. The verdict does not move, which is what the flip margin
measures.

### 2.2 The systolic tail

```bash
uv run bwz matmul -M 1 -N 10000 -K 10000 -c a100_80gb -d fp16 --ideal
```

```
  shape utilisation             5.88%
  DRAM reads                   139 MB
  DRAM writes                   20 kB
  latency                    71.3 µs
  verdict              DRAM_BW_BOUND
```

`1/17` of peak on a 16×16 array. The arithmetic fell by 10 000× against §2.1 and the traffic did
not fall at all, so the workload crossed the ridge. `--ideal` does **not** remove this: it is
geometry, not a derating.

### 2.3 Widths — all integer, all float, and mixed

```bash
uv run bwz matmul -M 4096 -N 4096 -K 4096 -c a100_80gb -d int8            --ideal   # all int8
uv run bwz matmul -M 4096 -N 4096 -K 4096 -c a100_80gb -d int8 --out int32 --ideal  # int32 result
uv run bwz matmul -M 4096 -N 4096 -K 4096 -c a100_80gb -d fp16 --out fp32  --ideal  # all float
uv run bwz matmul -M 4096 -N 4096 -K 4096 -c a100_80gb --a fp16 --b int8   --ideal  # mixed
```

| | operations | result C | DRAM read / written | intensity | latency |
|---|---|---|---|---|---|
| `int8 × int8 → int8` | 137 GOP | 16.8 MB | 0 B / 16.8 MB | 2730.7 OP/byte | 224 µs |
| `int8 × int8 → int32` | 137 GOP | **67.1 MB** | 21.4 MB / 67.1 MB | **1365.3 OP/byte** | 224 µs |
| `fp16 × fp16 → fp32` | 137 GOP | 67.1 MB | 46.9 MB / 67.1 MB | 1024.0 OP/byte | 445 µs |
| `fp16 × int8 → fp16` | 137 GOP | 33.6 MB | 20 MB / 33.6 MB | 1638.4 OP/byte | **445 µs** |

The read column shrinks as operands become resident; the write column never does. `C` is the
answer, and on the first row A100 fetches **nothing** — both operands fit on chip — yet still
writes all 16.8 MB of it (D22).

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
  lane      busy       occupancy   what it was doing
  dram     23 µs      4% of span   streaming operand tiles across the one modelled link
  sram   1.25 ms   1.97 of 2 buf   tile buffers held from fetch to use
  core    635 µs    100% of span   arithmetic, plus one dispatch

  64 steps drawn, coalesced from 65536 tiles, double buffered.
```

**The occupancy column carries a different unit per row, on purpose.** DRAM and the array are
single serial resources — their spans never overlap — so theirs is a duty cycle and cannot exceed
100%. SRAM is not a resource that is busy or idle; it is *n* buffers, and the same arithmetic
counts how many were occupied: `1.97 of 2` means both halves of the double buffer were in use
almost all the time. That is capacity, not bandwidth, which is the whole of what SRAM contributes
in this model.

### 2.5 Dataflow strategies — `--a-strategy`, `--b-dataflow`

A is a byte-amount knob; B is a timing knob (`docs/MODEL.md` §6.3a, `docs/CORRECTIONS.md` D36).
Defaults reproduce every number above exactly — `stage` and `write-ahead` are what §2.1–2.4 already
ran.

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

`stream` re-reads A once per tile instead of once per k-slice — `NTILES_PER_KS = 16` here, so A's
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
scratchpad; fell back to stage — the same 67.1 MB total, staged per k-slice
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
k-slice's `NTILES_PER_KS` tiles one A staging serves, clamped down to the largest power-of-2 divisor
when the requested value is not one:

```bash
uv run bwz matmul -M 8192 -N 8192 -K 8192 -c metis_aipu -d int8 --ideal \
  --a-residency-tiles 5 --no-pipeline
```

```
a_residency_tiles=5 is not a power-of-2 divisor of NTILES_PER_KS=16; clamped
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

---

## 3. `bwz encoder-layer` — one encoder layer, sized from the command line

The transformer counterpart of `bwz matmul` (§2): the shape comes from the command line, not a
profile. Bidirectional attention over all `S` tokens, no KV cache, no LM head — one layer, always.
For anything deeper, load a profile and use `bwz run` (§4).

| Flag | Meaning |
|---|---|
| `-c`, `--chip` | chip profile id or path (required) |
| `-d`, `--hidden` | model width (default 8) |
| `--heads` | attention heads (default 2) |
| `--head-dim` | defaults to `hidden // heads` |
| `--ffn` | FFN inner width (default 16) |
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
  prefill    122 ns   COMPUTE_BOUND   0.01%   31.4 ps      122 ns       0 s

  TTFT      122 ns
  total     122 ns
  achieved  43.1 GOP/s of 312 TOP/s (0.01%)
```

664 parameters, 5280 operations at the defaults — the golden numbers that `tests/unit/test_kernels.py`
and the bundled `single_layer_encoder_toy` profile both pin to (`docs/CORRECTIONS.md` D39): the CLI
probe and the frozen teaching profile agree exactly because they run through the same
`encoder_layer_kernel` factory (`bwz/kernels.py`), the way `bwz matmul` runs through `matmul_kernel`.

Shape utilisation bottoms out at 0.01% — 8-wide operands do not begin to fill a 16×16 array — which
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
| `--a-strategy`, `--b-dataflow`, `--a-residency-tiles`, `--a-prefetch-depth`, `--iterations` | accepted for parity with `matmul` (§2.5), but inert here — a network's graph is never one bare matmul, so `analysis/schedule.py` never reaches the single-matmul branch these read |

```bash
uv run bwz run -m llama3_8b -c a100_80gb --input-tokens 2048 --output-tokens 128
```

```
  TTFT      139 ms
  TPOT      9.49 ms   105.4 tok/s
  total     1.35 s
  achieved  22 TOP/s of 312 TOP/s (7.04%)
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
  gemma3_4b   chip_b   3.88 G     25.75%   144 ms    11.0   DRAM_BW_BOUND   0.17%

Prefill crossover
  gemma3_4b: no crossover in [1, 100000] — chip_b is faster throughout
```

Flags: `--chips`, `--models` (comma-separated), `--batch`, `--input-tokens`, `--output-tokens`,
`--context`, `--weights`, `--crossover/--no-crossover`, `--ideal`.

---

## 6. Figures

**The figures are not the report.** `bwz matmul` and `bwz run` print the numbers, their derivations
and the assumptions drawer (§2, §4); the plot scripts write files and print only `wrote …`. Two
commands, on purpose — `make plots` runs the scripts several times and a wall of tables per chip
would drown it.

```bash
make plots        # from the repo root — regenerates all of docs/plots/
```

or individually, from `backend/`:

```bash
# roofline + the three-element machine diagram (roofline-*.png, machine-*.png) — needs matplotlib
uv run --group plots python scripts/plot_roofline.py --chip a100_80gb --model llama3_8b
uv run --group plots python scripts/plot_roofline.py --chip chip_a --weights int8 \
  --model gemma3_4b --tokens 512 \
  --matmul 512,4096,4096 --matmul 128,4096,4096 --matmul 1,4096,4096

# resource timeline, one self-contained zoomable HTML page per chip — pure stdlib,
# no --group plots needed: nothing here imports a plotting library at all
uv run python scripts/plot_pipeline.py
uv run python scripts/plot_pipeline.py --chip h100_sxm --matmul 8192,8192,8192
```

`plot_roofline.py`: `--chip`, `--weights`, `--matmul M,N,K` (repeatable), `--model ID`
(repeatable), `--tokens`, `--out`.
`plot_pipeline.py` draws one of three things, and the flags mirror the report commands:

| what | flags |
|---|---|
| a matmul (default) | `--matmul M,N,K`, plus the dataflow strategy flags of §2.5 |
| a profile | `--model ID`, `--tokens/-S` — one page per phase |
| an ad-hoc single-layer encoder | `--encoder --hidden --heads --head-dim --ffn --vocab --tokens/-S` |

plus `--chip` (repeatable), `--compare`, `--weights`, `--ideal`, `--steps` (default 256 — the only
resolution knob; the page zooms, so there is no separate static-figure register to keep legible),
`--out`, and `--animate` — matmul only, opt-in, rejected alongside `--model`/`--encoder`/`--compare`;
writes a second self-contained page playing the same schedule back as DRAM -> SRAM -> Accelerator
motion instead of a static strip (`docs/plots/README.md` "Playing the flow animation", D40).

### 6.1 `--compare` — two chips, one workload, one page

Without it, `--chip A --chip B` writes one page per chip, each with x normalised to that chip's own
span. With it, they land in **one** page on a **shared, absolute** axis:

```bash
uv run python scripts/plot_pipeline.py \
  --chip a100_80gb --chip metis_aipu --compare --model gemma3_4b -S 512
```

```
wrote ../docs/plots/timeline-compare-a100_80gb-vs-metis_aipu-gemma3_4b-prefill-int8.html
wrote ../docs/plots/timeline-compare-a100_80gb-vs-metis_aipu-gemma3_4b-decode-int8.html
```

Both views are kept on purpose (`docs/CORRECTIONS.md` D29). Absolute is the axis for "which is
faster and by how much"; normalised is the axis for "how is *this* machine's time distributed", and
at 13x the faster chip's whole run is 7% of a shared axis.

It works for all three workload kinds and for more than two chips:

```bash
… --chip a100_80gb --chip metis_aipu --compare --matmul 4096,4096,4096
… --chip a100_80gb --chip metis_aipu --compare --encoder --hidden 4096 --heads 64 --ffn 16384 -S 1024
… --chip a100_80gb --chip metis_aipu --chip jetson_orin --compare --matmul 2048,2048,2048
```

A `--model` comparison pairs prefill against prefill and decode against decode, never across.

**Every chip runs the same workload at the same precision**, which has to be enforced rather than
assumed — A100 defaults to fp16 and Metis has no fp16 datapath, so per-chip defaults would compare
two different amounts of traffic. `--compare` picks one dtype every chip supports and otherwise
refuses:

```bash
uv run python scripts/plot_pipeline.py \
  --chip metis_aipu --chip a100_80gb --compare --weights fp16
```

```
bwz: metis_aipu has no fp16 datapath, so --compare cannot run the same workload on every chip.
Supported by all: int8
```

```bash
uv run python scripts/plot_pipeline.py --chip a100_80gb --compare
```

```
bwz: --compare puts two or more chips in one figure and got 1; pass --chip twice, or drop
--compare for the per-chip view
```

Rows are **banded by chip**, not aligned across them: A100 declares 3 memory levels and 2 compute
units, Metis 4 and 2, `chip_a` 2 and 1, and no correspondence between `cuda_core` and `dpu` exists
to draw. Each band opens with a header row carrying that machine's peak, DRAM bandwidth, on-chip
capacity, its row counts, its total span **and its achieved throughput** — `622 TOP/s achieved ·
100% of peak` on A100 against `186 TOP/s achieved · 89% of peak` on Metis for an 8192³ INT8 matmul,
the same 3.34x the latency ratio is, inverted, because both come from the one reported latency
(D35). Everything the single-chip page does survives inside the band — grey rows for
declared-and-unused resources, matrix and vector lanes on separate rows, filled loads against
hollow stores, named bars, and the three info boxes, now one set per chip.

Below the two registers the comparison adds a **roofline register**: both chips' ceilings, ridge
points and M=1 lines on one chart, with each chip's workload point on it. Inside that panel colour
means *chip* rather than *resource*, which the panel says on itself. The arithmetic section is
rendered **once** — it is a property of the workload, and the workload is the same on both
machines — but the **deployment listing is rendered once per chip**, because how the work reaches
the silicon is exactly what differs between them (D32).

The shape flags belong to `--encoder`; passing them with `--model` is an error, because a profile
already carries its dimensions:

```bash
uv run python scripts/plot_pipeline.py --chip a100_80gb \
  --encoder --hidden 4096 --heads 64 --ffn 16384 -S 4096 --ideal --out ..
```

`--model` draws a network instead of a matmul, one page per phase — which is where the
`COMPUTED — matmul 98% · attention 2%` breakdown earns itself:

```bash
uv run python scripts/plot_pipeline.py \
  --chip a100_80gb --model llama3_8b --tokens 512 --out ..
# timeline-a100_80gb-llama3_8b-{prefill,decode}-fp16.html
```

`--out` is relative to where you run the script, so from `backend/` a bare `--out ..` lands in the
repo root and from the repo root it lands *outside* the repo — with a `wrote ../timeline-….html`
line that looks right either way. Pass an absolute path when it matters.

### The zoomable page

```bash
cd backend
uv run python scripts/plot_pipeline.py \
  --chip a100_80gb --matmul 10000,10000,10000 --ideal \
  --out /absolute/path/you/want

xdg-open /absolute/path/you/want/timeline-a100_80gb-fp16.html
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
| 6.42 ms, compute-bound by 24× | §2.1 | `docs/MODEL.md` §5b, §6.3 |
| 5.88% = 1/17 at M=1 | §2.2 | `docs/MODEL.md` §6.1 |
| result width changes bytes only | §2.3 | `docs/CORRECTIONS.md` D18 |
| mixed operands run at the wider | §2.3 | `docs/CORRECTIONS.md` D18 |
| SRAM depth 1.97 of 2 buffers | §2.4 | `docs/MODEL.md` §6.5, D19 |
| the result is always written back | §2.1, §2.3 | `docs/CORRECTIONS.md` D22 |
| grey rows = the model's boundary | §6 | `docs/CORRECTIONS.md` D20 |
| no Konata, no Kanata | — | `docs/CORRECTIONS.md` D21 |
| 1.40% residency on chip_a | §4 | `docs/CORRECTIONS.md` D8, D15 |
| traffic is a lower bound when the working set does not fit | every report's assumptions | `docs/MODEL.md` §6.2 |
| stream is 16x the staged total | §2.5 | `docs/MODEL.md` §6.3a, `docs/CORRECTIONS.md` D36 |
| whole/persistent clamp rather than raise | §2.5 | `docs/CORRECTIONS.md` D36, CLAUDE.md #8 |
| 622/186 TOP/s achieved, 3.34x inverted | §6.1 | `docs/CORRECTIONS.md` D35, D37 |
