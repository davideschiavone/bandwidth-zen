# bandwidth-zen — Phase 1 plan: engine bring-up + single-chip calibration

## Context

The repo currently contains only docs (CLAUDE.md, PROMPT.md, README.md) — no code. The goal of
this phase is to get from zero to a **calibrated, backtested single-chip analytical engine**:
a Python environment, the core `analyze()` pipeline, chip profiles for well-documented hardware
(NVIDIA H100/A100, AMD MI300X), and a validation harness that compares predictions against
published benchmarks (MLPerf Inference, vendor blogs).

Decisions taken with the user:
- **Samsung deferred** — no datacenter chip with public benchmarks; revisit once the harness exists.
- **Single-chip first** — DeepSeek-V3/R1 on H100 clusters needs multi-chip (M5); it is the named
  next phase, not this one.
- **Backend + CLI only** — no frontend in this phase; `bwz run` and `make validate` are the interfaces.
- **Edge-NPU worked example is additive** (2026-08-07) — a supplied two-chip INT8 comparison
  (`chip_a`, `chip_b`, Gemma-3-4B at 1B/2B/4B) joins the roster at M1 and becomes M3's acceptance
  demo. Milestone structure unchanged. See `docs/CORRECTIONS.md` D8.
- **The v1 machine is three elements** (2026-08-07) — external DRAM/HBM provides *bandwidth*,
  on-chip SRAM provides *capacity only* (residency, and the headroom that makes double buffering
  possible), the compute engine provides *TOPS*. No on-chip bandwidth term. See D5a and D5b.

The plan follows PROMPT.md milestones M0→M3 plus an early slice of M7 (validation), one session
each. Each session ends green (`make lint test`), committed, per CLAUDE.md working style.

---

## Session 1 — M0: environment + scaffold (backend only)

- Python env with **uv** (PROMPT.md's chosen tool): `uv venv` + `backend/pyproject.toml`
  (hatchling, `py>=3.11`, deps: pydantic v2, typer, pyyaml, rich; dev: pytest, hypothesis,
  ruff, mypy strict).
- Repo skeleton per PROMPT.md §2, backend paths only: `backend/bwz/{units.py, calibration.py,
  spec/, graph/, operators/, analysis/, report.py, cli.py}`, `backend/profiles/{chips,models}/`,
  `backend/tests/{unit,integration,validation}/`.
- `Makefile` (dev/test/test-fast/lint/fmt/validate targets — frontend targets stubbed as no-ops),
  `.github/workflows/ci.yml`, `LICENSE` (Apache-2.0), empty `docs/{MODEL,CALIBRATION,CORRECTIONS,SCHEMA}.md`.
- `units.py` first real code: SI helpers + prefix parsing ("3.35 TB/s" → float), with tests.

**Done when:** `make test` and `make lint` pass; `uv run bwz --help` works.

## Session 2 — M1: specs, loaders, profiles

- Pydantic v2 specs: `spec/{model_spec,hardware_spec,deployment,loaders}.py`. Bandwidth/capacity
  fields declared `float` (CLAUDE.md gotcha). Validation errors name field + bad value + allowed range.
- `hardware_spec.peak_flops_per_s()` — the single place MACs×2 happens. Must reproduce both
  H100 fp16 dense (≈989 TFLOP/s) and `chip_a` INT8 (512×512 MACs × 4 cores × 0.8 GHz × 2 ÷ 8
  bit-serial tax = 209.6 TOPS), the latter via a `dtype_multipliers: {int8: 0.125}` entry.
- Three schema additions the edge example forces (D6, D7, D5a):
  - `estimates: {<field>: <rationale>}` — per-field provenance; touched entries propagate into
    `report.assumptions`.
  - `hypothetical: true` + `derived_from:` — for `chip_b`, which has no datasheet. `source_url`
    stays mandatory for anything claiming to be a real product.
  - `params` override + preset family (1B/2B/4B), emitting an assumption line when the declared
    figure diverges from the derived count by >1%.
- Chip profiles (6) with `source_url` from vendor datasheets: `h100_sxm.yaml`, `a100_80gb.yaml`,
  `mi300x.yaml` (calibration targets), `jetson_orin.yaml` (edge sanity case), plus `chip_a.yaml`
  and `chip_b.yaml` for the D8 example.
- Model profiles (7) with `source_url` from HF configs: `llama3_8b.yaml`, `llama2_70b.yaml`,
  `mistral_7b.yaml`, `gpt3.yaml`, `gemma3_4b.yaml` (+ 1B/2B presets), `mobilenetv3.yaml` — the
  last exercising the CNN/layer-list branch of the spec union, which is otherwise shipped untested.
- `bwz list` CLI; round-trip tests for every profile.

**Done when:** profiles round-trip spec→YAML→spec (not textually — comments, key order, and
`"3.35 TB/s"` → `3.35e12` normalization are all lost by design); golden test: Llama-3-8B param
count = 8.03 B ±0.5%, via a closed-form `ModelSpec.parameter_count()` that M2's graph builder
later cross-checks against its summed weight tensors.

## Session 3 — M2: graph + operator cost models

- `graph/`: `ops.py`, `builder.py`, `transformer.py` (separate **prefill and decode** graphs),
  `cnn.py`, `dag.py` (topo sort, critical path, liveness).
- `operators/`: matmul (2·M·N·K, tiled traffic, tail effect), attention (vanilla + FlashAttention-2
  + GQA; flash changes bytes never FLOPs), conv (direct/im2col only; Winograd/FFT in the
  refinement backlog), norm, elementwise.
  FLOP/byte formulas exactly per PROMPT.md §3.2, documented in `docs/MODEL.md` as implemented.
- Golden tests in the same commits: GPT-3 prefill FLOPs ≈ **`2·N·D`** within 3% (PROMPT.md says
  `6·N·D`, which is the training rule — see D11); Gemma-3-4B forward FLOPs (batch 1) ≈
  `2·N_non-embedding·S` at S=512 within 2%; Llama-3-8B KV cache @ 8k fp16 = 1.0 **GiB** ±2%.
- Per-op layer golden from D8 — the decode graph's seven projections (Q/O at `d_model²`, K/V at
  `d_model·d_kv`, three FFN at `d_model·d_ff`) sum to 94.4 MB/layer, 2.78 ms load @ 34 GB/s,
  0.90 µs compute, ≈3080× memory-bound. `ops = 2·weight_bytes` falls out of `2·M·N·K` at M=1.
- KV cache `n_layers·2·d_kv·C·w_bytes`; prefill score FLOPs **halved for causal masking**, a
  deliberate deviation from the supplied formulas (D8).

**Done when:** all M2 golden tests pass, and the graph builder's summed weight tensors equal
`ModelSpec.parameter_count()` for every shipped transformer — two independent derivations agreeing.

## Session 4 — M3: single-chip analysis → Report

- `analysis/`: `roofline.py` (two ridges only — compute and DRAM, per D5a), `tiling.py`,
  `memory.py` (waterfall + feasibility with cheapest fixes, **plus the residency fraction
  `r = min(1, sram_bytes/W)`** — SRAM capacity earning its keep by removing DRAM traffic),
  `schedule.py` (double buffering: `max(load, compute)` when SRAM capacity has room for two tiles,
  `load + compute` when it does not — so the "max, never the sum" rule is *derived* from capacity
  rather than asserted), `bottleneck.py` (three-way label: DRAM / COMPUTE / LATENCY). All
  constants in `calibration.py` with source comments (start with documented defaults: ~0.8–0.9
  achievable HBM efficiency and one ~0.5–0.7 achieved-flops fraction for tensor cores — fitted in
  Session 5). The multi-level hierarchy, ws/os/rs loop-order search, and Winograd/FFT are not in
  this session — see the refinement backlog below.
- `report.py` per PROMPT.md §5 (confidence + assumptions fields mandatory), plus a **flip-margin**
  field: for any result whose binding term came from an `estimates:` field, how far that input can
  move before the `argmax` changes ("DRAM-bound; on-chip takes over below 68 GB/s"). This is what
  makes D5b's deferred refinements safe to defer — it turns an unquantified caveat into a number,
  and it is a report annotation, not a physics change.
- `bwz run` with rich terminal table (README shows the target output); `bwz compare` for the
  two-chip head-to-head, prefill curve, and crossover `S*` (bisection over S ∈ [1, 1e5]).
- Property tests (hypothesis): bandwidth↑ never latency↑; INT8 never slower than FP16; monotonic in
  batch; `decode_vs_bandwidth` comes out ~linear in `bw_dram`.

**Done when:** the CLAUDE.md sanity checks hold, and the D8 acceptance demo is reproduced.
Achieved — see `docs/MODEL.md` §6.7 for the full table. Llama-3-8B fp16 decode on H100 →
DRAM_BW_BOUND at **165 tok/s** (not the 35–55 CLAUDE.md pairs with its own 16 GB figure: see D12);
prefill @2k → COMPUTE_BOUND at 67.7%; Gemma-3-4B batch 1 → 0.24% utilisation but **DRAM-bound**,
with MobileNetV3 on H100 as the LATENCY_BOUND case instead (D13). D8: ridge points 6165 / 1541;
residency 1.40 / 2.73 / 5.71% and 25.76 / 50.15 / 100%; chip_a decode 8.4 / 16.8 / 37.0 tok/s;
chip_a TTFT @512 114.6 ms. All self-consistency goldens in `tests/unit` and
`tests/integration/test_analyze.py` — **not** in `tests/validation/`, which is reserved for
published reference points (D8).

Two D8 figures move, both because the systolic tail effect is modelled (D14): `chip_b` is
COMPUTE_BOUND at batch 1, so its decode rates are 9.3 / 20.6 / 48.2 rather than 11.7 / 34 / 128,
and the head-to-head margin over chip_a collapses from 33% to ~11%.

## Session 4b — Unplanned, user-driven (2026-08-15/16)

Not in the original plan; added because the questions that came up while reading M3's output could
not be answered with what M3 shipped. All of it single-chip, all of it inside the M3 dependency
arrow, no milestone re-ordering.

- **`family: matmul`** (`spec/model_spec.py`, `graph/matmul.py`, `bwz matmul`). A one-operation
  workload: the smallest thing that exercises the matmul cost model, the systolic tail, the memory
  planner and the roofline with no network around it to explain a number away. A family rather than
  a hand-costed `custom` op because `CustomOp` carries no shape, so the tail effect would silently
  come back as 100% (D17).
- **Per-operand widths and an accumulator** — `a_dtype`, `b_dtype`, `out_dtype`, plus `DType.INT32`
  as a storage-only width (D18). Two rules fell out: the result width changes bytes only and never
  operations, and a mixed-width matmul runs at the *wider* operand. The latter contradicts
  `compute_dtype()`, which is right for W8A8 and wrong for W8A16; the transformer path was left on
  the old rule deliberately, since changing it moves every mixed-precision number in the repo.
  **Open for Session 5.**
- **`analysis/pipeline.py` + `kanata.py`** — the tile schedule behind `max(load, compute)`,
  as a Kanata log for Konata and as a per-chip figure. Constrained to be a *decomposition* of the
  reported numbers, never a second model (D19). It surfaced that the roofline's `max()` omits
  pipeline fill/drain, `min(t_dram, t_compute)/tiles`, now reported separately rather than folded
  into a latency. **Folding it in is Session 5 work**, since it moves every double-buffered result.
- **`--ideal`** (both efficiency de-ratings to 1.0), so a number can be checked against a datasheet
  by hand and the unfitted part of a prediction is visible rather than baked in.
- **`scripts/plot_roofline.py`, `scripts/plot_pipeline.py`, `make plots`** — figures computed by
  calling `analyze()`, outside the package, in their own dependency group.
- **Docs**: `docs/CLI.md` (every command, with real output), `docs/plots/README.md`, SCHEMA §2.4.

Two documentation bugs found and fixed rather than left: `precision.accumulate` is declared and
read by nothing, and README's model summary advertised a tile-reuse search and alpha-beta
collectives as though implemented.

Also fixed: `backend/pyproject.toml` carried `readme = "../README.md"`, which hatchling rejects as
outside the project directory — `make venv` failed on a fresh clone.

## Session 5 — Calibration + backtest (early M7 slice)

- Collect **published** reference points into `tests/validation/reference_points.yaml`, each with
  URL + exact config. Candidates (verify at collection time; never fabricate — "no reference point
  available" is acceptable):
  - MLPerf Inference v3.1/v4.0 Llama-2-70B on H100 and MI300X (server + offline)
  - NVIDIA/AMD vendor blog numbers for Llama-3-8B / Mistral-7B on H100 and MI300X
  - Community single-GPU throughput measurements (vLLM benchmark reports) as secondary points
- `make validate`: predicted-vs-published table with per-point error; `@pytest.mark.validation`.
- Fit calibration constants (bandwidth efficiency, launch overhead, util derating) against these
  points; every fitted value documented in `docs/CALIBRATION.md` with the dataset.
- Write up error analysis + outlier explanations in `docs/CALIBRATION.md`.

- `chip_a` and `chip_b` have no published benchmarks and contribute **no** reference points. They
  inherit the constants fitted on H100/A100/MI300X and stay explicitly uncalibrated; every report
  they produce says so via the `estimates:` propagation (D7) and the flip-margin field.

**Done when:** validation table prints; single-chip LLM decode MAPE ≤ ~20% (README's stated target
is ±15% for this class); every outlier has a written explanation.

---

## Deferred refinement backlog (M8, on user demand)

Not built in the sessions above; preserved, not deleted. Landed explicitly when a user asks for it,
and each item is kept only if it improves `make validate` MAPE over the flat model:

- **Hierarchical multi-level roofline** — per-level (L1/SRAM, L2, DRAM) byte accounting and
  "which level binds" reporting. Needed only if the flat model mispredicts working-set-in-L2 cases
  (activation-heavy CNNs, small models with big batches).
- **ws/os/rs loop-order search** — derive which operand stays put per op instead of honouring the
  profile's declared dataflow.
- **Winograd F(2×2,3×3) / F(4×4,3×3) and FFT convolution** — FLOP-reduced algorithm selection.
- **L2 reuse effects** refinement for CNN activations.
- **SRAM/NoC split** (D5b) — separate `sram_read_bytes_per_s` (local weight feed) from
  `noc_bytes_per_s` (inter-core), instead of one `bw_onchip` standing for both. The supplied
  128 GB/s is a NoC bisection figure and sits below the array's structural weight-feed floor
  (≈205 GB/s SoC).
- **Full-W SRAM traffic accounting** (D5b) — streamed weights land in SRAM and are read out from
  there, so SRAM carries `W` reads + `(1-r)·W` writes, not `r·W`. Shrinks the on-chip margins from
  260× / 11× / 3.8× to 1.9× / 1.6× / 1.26× and moves several decode figures by ~25%.

Trigger: users need a per-level breakdown, or Session 5 validation shows a systematic error the
flat model cannot explain. For the two SRAM items specifically: a published SRAM organization
(banks × width × clock) or a measured figure for either edge chip.

---

## Named next phase (not in this plan)

**Multi-chip + DeepSeek backtest (M5 slice):** TP/PP/EP graph rewrites, alpha-beta collectives with
NVLink/IB hierarchy, MoE support (Mixtral first, then DeepSeek-V3 MLA + fine-grained experts —
MLA is a new attention cost model), then backtest against DeepSeek-V3 tech-report inference
figures and SGLang/vLLM published H100-cluster throughput. Samsung/edge profiles also revisit here.

## Verification (end of phase)

```bash
make lint test        # ruff + mypy --strict + pytest, all green
uv run bwz run --model llama3_8b --chip h100_sxm --batch 1 \
  --input-tokens 2048 --output-tokens 256   # sane report, DRAM_BW_BOUND decode
uv run bwz compare --chips chip_a,chip_b \
  --models gemma3_preset_1b,gemma3_preset_2b,gemma3_4b \
  --input-tokens 512 --context 4096         # D8 head-to-head + crossover S*
make plots            # figures regenerate and match what is committed
make validate         # prints predicted-vs-published table, MAPE within target
```

Every command in this repo, with the output it actually produces, is in [`CLI.md`](CLI.md).
