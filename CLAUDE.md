# CLAUDE.md

Operating manual for Claude Code working in this repository. Read this before every task.
The full build specification lives in `PROMPT.md`; the physics lives in `docs/MODEL.md`; every
command and the invocation behind every published number lives in `docs/CLI.md`.

---

## What this project is

`bandwidth-zen` is an **analytical performance model** that predicts how a neural network (transformer LLM
or CNN) will run on a given chip or multi-chip system: latency, throughput, utilization, memory
footprint, energy, and — most importantly — **what the limiter is and why**.

It is a fast, explainable estimator (roofline + tiling + collective-cost models), not a simulator
and not a compiler. Every number must be traceable to a formula and its inputs.

**In scope:** analytical cost models, hardware/model spec schemas, sweep + Pareto exploration,
visualization, validation against published benchmarks.

**Out of scope:** cycle-accurate simulation, kernel codegen, running real models, training loops
beyond a memory/FLOP multiplier, auth, databases, multi-tenancy.

---

## Commands

```bash
make dev            # backend :8000 (uvicorn --reload) + frontend :5173 (vite)
make test           # pytest + vitest
make test-fast      # pytest -m "not validation and not slow"
make lint           # ruff check + mypy --strict + eslint
make fmt            # ruff format + prettier
make types          # regenerate frontend/src/api/types.ts from OpenAPI
make validate       # run the predicted-vs-published validation suite, print the table
make plots          # regenerate docs/plots/ (roofline + tile schedule, per chip)
make docker         # docker compose build && up
```

Single test: `pytest backend/tests/unit/test_roofline.py::test_ridge_point -q`
Backend only: `cd backend && uv run uvicorn bwz.api.app:app --reload`

---

## Architecture in one paragraph

`spec/` parses YAML/JSON into validated pydantic objects (`ModelSpec`, `HardwareSpec`,
`DeploymentSpec`). `graph/` expands a `ModelSpec` into a DAG of `Operation`s. `operators/` attaches
a cost model to each op, producing FLOPs and per-memory-level byte counts. `analysis/` runs the
roofline, the stationarity that decides the tile grid, the tiling search, memory planner, scheduler,
parallelism rewrites, and collective cost model, then classifies bottlenecks and emits a `Report`. `api/` and `cli.py` are thin shells over
`analyze(model, hardware, deployment) -> Report`. The frontend only ever consumes `Report`.

**The dependency arrow points one way:** `spec → graph → operators → analysis → report → {api, cli}`.
Nothing in `analysis/` may import from `api/`. Nothing in `graph/` may import from `analysis/`.

---

## Non-negotiable conventions

1. **SI units internally, always.** FLOPs, bytes, bytes/s, seconds, joules, watts. Variable names
   carry the unit: `bandwidth_bytes_per_s`, `latency_s`, `energy_j`. Human units (GB/s, TFLOP/s, ms)
   exist only in `units.py` formatters and in YAML the user writes. A bare `bandwidth` or `time`
   variable is a bug.

2. **All empirical constants live in `bwz/calibration.py`.** Each one needs a comment with its
   source (datasheet, paper, or the fitted dataset in `docs/CALIBRATION.md`). An inline `* 0.85`
   anywhere else must be rejected in review — including by you, on your own code.

3. **The analysis core is pure.** `bwz/analysis/` and `bwz/operators/` import no web
   framework, no file I/O, no global mutable state, no wall-clock reads. Same inputs → identical
   output. This is what makes sweeps parallelizable and tests reliable.

4. **Every output carries its assumptions.** If you add a modelling shortcut, append a
   human-readable string to `report.assumptions` at the point where you take the shortcut. Users
   read this drawer; it is the honesty mechanism of the whole tool.

5. **Matmul FLOPs are `2·M·N·K`.** MACs are `M·N·K`. Never mix the two. Chip profiles specify
   MACs/cycle; the engine multiplies by 2 exactly once, in `hardware_spec.peak_flops_per_s()`.

6. **Prefill and decode are different machines.** Any transformer code path that doesn't distinguish
   them is wrong. Prefill is compute-bound with `S` tokens; decode is memory-bound with `S=1` and a
   growing context. Utilization differs by two orders of magnitude between them.

7. **Physics code ships with its test.** Anything under `analysis/` or `operators/` gets a golden
   test in the same commit, with the hand-computed expected value written out in the docstring.

8. **Errors are actionable.** A validation failure names the field, the bad value, and the allowed
   range. An infeasible config returns a `Report` with `feasible: false` and the cheapest fixes —
   never an exception, never a bare 500.

---

## Where things go

| Adding... | Goes in | Also update |
|---|---|---|
| A new operator cost model | `operators/<family>.py`, registered via `@register_op` | `docs/MODEL.md`, unit test |
| A new chip | `profiles/chips/<id>.yaml` with `source_url` | `tests/unit/test_profiles.py` |
| A new model | `profiles/models/<id>.yaml` with `source_url` | golden param-count test |
| An ad-hoc kernel-probe factory (`matmul_kernel`, `encoder_layer_kernel`) | `bwz/kernels.py` | `cli.py`, `scripts/plot_pipeline.py`, `docs/CLI.md` |
| A new empirical constant | `calibration.py` only | `docs/CALIBRATION.md` |
| A new report field | `report.py` | `docs/report.schema.json`, TS types, snapshot tests |
| A new parallelism strategy | `analysis/parallelism.py` + `collectives.py` | `docs/MODEL.md` |
| A new dataflow / stationarity | `analysis/stationarity.py` — the one place a tile grid is decided | `docs/MODEL.md` §6.1, `docs/SCHEMA.md`, unit test |
| A new model family | `spec/model_spec.py` + `graph/<family>.py`, dispatched in `graph/builder.py` | `docs/SCHEMA.md`, `docs/MODEL.md`, golden test |
| A new dtype | `spec/dtypes.py` only — widths are definitions, not calibration | `docs/SCHEMA.md` dtype lists |
| A figure | `backend/scripts/plot_*.py`, never inside `bwz/` | `docs/plots/README.md`, `make plots` |
| An interactive view | `backend/scripts/timeline_html.py` / `dataflow_html.py` — one self-contained file each, no CDN, no server, no third-party viewer | `docs/plots/README.md` |
| A CLI command or flag | `cli.py` | `docs/CLI.md` — with real output, not a description |
| A new UI panel | `frontend/src/components/` | `Dashboard.tsx`, vitest |

---

## Sanity checks — run these mentally before claiming a change works

These are known-good behaviours. If a change breaks one, the change is wrong.

- Llama-3-8B, fp16, batch 1, H100, decode → **memory-bound**, roughly 16 GB of weights moved per
  token, so ~35–55 tok/s. If you predict 500 tok/s, you've forgotten weight traffic.
- Same model, batch 128, decode → utilization rises sharply; still memory-bound until batch is large
  enough that weight traffic amortizes.
- Llama-3-8B prefill, 2048 tokens → **compute-bound**, utilization 40–70%.
- Gemma-4, batch 1, H100 → **latency/launch-bound**, single-digit % utilization. Batch 128 → good
  utilization. A model that shows 80% utilization at batch 1 is broken.
- MobileNetV3 depthwise layers → **memory-bound**, poor utilization on a large systolic array.
- GEMM with M=1 on a 128×128 array → **array-level** utilization ≈ 1/128, from padding M to a whole
  instruction tile (D52 — it is area, not a pipeline drain). If your utilization model doesn't
  reproduce this, it isn't modelling the array. **The chip-level figure is this times wave
  occupancy** and is legitimately lower: at M=1 there are few output tiles, so most of a
  many-core chip idles. Don't "fix" the second by breaking the first.
- **Every stationarity issues the same `M*N*K` MACs.** `os` and `ws` disagree about how many tiles
  there are, which dimension each sweeps, and whether partial sums are owed — never about the flop
  count. A change that moves the arithmetic when only the dataflow changed is wrong (D53).
- A chip asked for a dataflow it does not declare returns `feasible: false` naming the field and the
  capability — **refused, not clamped**, unlike the A/B strategy knobs. A clamp there would answer a
  different question than the one asked.
- A more accurate decomposition may report *lower* utilization. Flipping the matrix cores from `ws`
  to `os` (D53) cut M=1 chip utilization on A100 from 6.24% to 4.52%, because the parallelism `ws`
  claimed only existed as 390 625 partial sums nothing was reducing. Lower is not automatically a
  regression — check what the tiles *were*.
- FlashAttention changes bytes, never FLOPs. So does GQA, and so does a matmul's result width: an
  `int8 x int8 -> int32` matmul does the same `2*M*N*K` as `int8 x int8 -> int8` and writes four
  times the bytes.
- A mixed-width matmul runs at the **wider** operand — both enter the array through one datapath.
  `fp16 x int8` is 312 TOP/s on A100, not 624. (The transformer path still uses the older
  weight-dtype rule and is wrong for W8A16; see `docs/CORRECTIONS.md` D18.)
- A pipeline trace must never be faster than the report it illustrates. Spans are slices of
  `t_dram`/`t_compute`/`t_fixed` and sum back to them (D19).
- Figure rows are **hardware resources** read off the chip profile, and a resource the model does
  not use is drawn grey with the reason rather than omitted (D20).
- TP=8 across NVLink on a 7B model at batch 1 → comms is a large fraction of the critical path;
  speedup is well below 8×.
- Doubling DRAM bandwidth never increases predicted latency. INT8 is never slower than FP16 on
  hardware that supports both.

---

## Working style in this repo

- **One milestone at a time.** `PROMPT.md` §8 defines them. Finish, test, lint, commit, summarize,
  then stop for review before starting the next.
- **Commit is a standing authorization, not something to ask permission for each time.** Once a
  change is tested and linted clean, commit it as the last step of finishing that unit of work —
  do not wait for the user to separately say "commit". This line *is* the durable, in-repo
  authorization Claude Code's own safety default asks for before it will act on standing
  permission for a consequential action. Scope: local commits only. Pushing to a remote,
  force-pushing, or any destructive git operation still requires the user to ask in the moment —
  this bullet does not extend to those.
- **Plan before large changes.** For anything touching more than ~3 files, state the plan first.
- **Prefer editing over rewriting.** Don't restructure modules that already have passing tests.
- **Small commits, conventional messages:** `feat(analysis): flat roofline with tiled DRAM traffic`,
  `fix(attention): halve prefill score FLOPs for causal mask`, `docs(model): derive PP bubble`.
- **Never commit** failing tests, `# type: ignore` without a reason comment, `TODO` without an
  issue reference, generated `types.ts` edited by hand, or profile YAML without a `source_url`.
- **Push back on the spec when it's wrong.** `PROMPT.md` and the original design doc contain known
  errors (see `docs/CORRECTIONS.md`). If a formula looks wrong, say so, explain why, propose the
  fix, and record the decision — don't silently diverge, and don't implement something you believe
  to be incorrect.
- **Don't invent measurements.** Numbers in `docs/CALIBRATION.md` must come from a citable published
  source. If you don't have one, write "no reference point available" rather than a plausible
  figure. Fabricated validation data would make the entire project worthless.

---

## Style

**Python 3.11+**: ruff (line length 100), `mypy --strict`, pydantic v2 for all boundary types,
frozen dataclasses for internal value objects, `from __future__ import annotations`, typed
`Enum`s not string literals, no `Any` outside `loaders.py`. Docstrings on every public function
in `analysis/` state the formula and cite `docs/MODEL.md`.

**TypeScript/React**: strict mode, no `any`, functional components with hooks, zustand for the
config store, Tailwind for layout, D3 for scales/axes and canvas for anything drawing >500 marks,
`recharts` only for simple charts. API types are generated — never hand-written.

---

## Gotchas discovered so far

- Pydantic v2 coerces `1e12` in YAML to `float` but `1_000_000` to `int`; bandwidth fields must be
  declared `float` or comparisons silently integer-divide.
- PyYAML implements YAML **1.1**, which only recognises an exponent when it carries a sign:
  `3.35e12` loads as the *string* `"3.35e12"`, while `3.35e+12` loads as a float. `parse_or_pass`
  rescues the unsigned form by reading it as a unitless SI value, but profile YAML should still
  write the signed exponent — that is what the format means.
- `ceil` in the shape-utilisation model must operate on the *padded* dimension, not the tile count,
  or small-M GEMMs report >100% utilization.
- `analysis/stationarity.py` must stay the only place a tile grid is computed. It was duplicated in
  `pipeline.tile_count` and `tiling.systolic_utilisation` once, and the two drifted — the schedule
  drew a decomposition the utilisation figure was not costing (D53).
- The sweep process pool must receive plain dicts, not pydantic objects — pickling validated models
  across processes is measurably slower than re-validating in the worker.
- Vite dev server needs `server.proxy['/api'] = 'http://localhost:8000'`; do not hardcode the
  backend origin in `client.ts`.
