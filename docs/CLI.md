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
| `--json` | the raw `Report` as JSON |

### 2.1 The datasheet check

```bash
uv run bwz matmul -M 10000 -N 10000 -K 10000 -c a100_80gb -d fp16 --ideal
```

```
  intensity            3333.3 OP/byte
  ridge point           153.0 OP/byte
  shape utilisation            99.84%
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

| | operations | result C | intensity | latency |
|---|---|---|---|---|
| `int8 × int8 → int8` | 137 GOP | 16.8 MB | 2730.7 OP/byte | 224 µs |
| `int8 × int8 → int32` | 137 GOP | **67.1 MB** | **1365.3 OP/byte** | 224 µs |
| `fp16 × fp16 → fp32` | 137 GOP | 67.1 MB | 1024.0 OP/byte | 445 µs |
| `fp16 × int8 → fp16` | 137 GOP | 33.6 MB | 1638.4 OP/byte | **445 µs** |

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

---

## 3. `bwz run` — a network on a chip

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

## 4. `bwz compare` — chips head to head

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

## 5. Figures

```bash
make plots        # from the repo root — regenerates all of docs/plots/
```

or individually, from `backend/`:

```bash
# roofline + the three-element machine diagram (roofline-*.png, machine-*.png)
uv run --group plots python scripts/plot_roofline.py --chip a100_80gb --model llama3_8b
uv run --group plots python scripts/plot_roofline.py --chip chip_a --weights int8 \
  --model gemma3_4b --tokens 512 \
  --matmul 512,4096,4096 --matmul 128,4096,4096 --matmul 1,4096,4096

# resource timeline, one figure per chip (timeline-*.png) plus a zoomable page
uv run --group plots python scripts/plot_pipeline.py --html
uv run --group plots python scripts/plot_pipeline.py --chip h100_sxm --matmul 8192,8192,8192 --html
```

`plot_roofline.py`: `--chip`, `--weights`, `--matmul M,N,K` (repeatable), `--model ID`
(repeatable), `--tokens`, `--out`.
`plot_pipeline.py`: `--chip` (repeatable), `--matmul M,N,K`, `--weights`, `--ideal`, `--steps`,
`--zoom`, `--html`, `--html-steps`, `--out`.

The `.html` is the one to open when you want to *look around* rather than glance: rows are the same
hardware resources, but the time axis zooms (wheel), pans (drag), resets (double-click), and every
bar reports its own bytes, rate and duration on hover. One self-contained file, no server and no
viewer to install.

The pipeline figure gives every declared memory level and compute unit its own row, with the bytes
moved, the achieved bandwidth and the operations retired written beside it — and draws grey the
resources this model never uses. See [`plots/README.md`](plots/README.md) for how to read it.

---

## 6. Development

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

## 7. Where each claim is derived

| Claim | Command | Derivation |
|---|---|---|
| 6.42 ms, compute-bound by 24× | §2.1 | `docs/MODEL.md` §5b, §6.3 |
| 5.88% = 1/17 at M=1 | §2.2 | `docs/MODEL.md` §6.1 |
| result width changes bytes only | §2.3 | `docs/CORRECTIONS.md` D18 |
| mixed operands run at the wider | §2.3 | `docs/CORRECTIONS.md` D18 |
| SRAM depth 1.97 of 2 buffers | §2.4 | `docs/MODEL.md` §6.5, D19 |
| 1.40% residency on chip_a | §3 | `docs/CORRECTIONS.md` D8, D15 |
| traffic is a lower bound when the working set does not fit | every report's assumptions | `docs/MODEL.md` §6.2 |
