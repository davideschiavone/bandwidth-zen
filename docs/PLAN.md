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
- `hardware_spec.peak_flops_per_s()` — the single place MACs×2 happens.
- Chip profiles with `source_url` from vendor datasheets: `h100_sxm.yaml`, `a100_80gb.yaml`,
  `mi300x.yaml` (calibration targets), plus `jetson_orin.yaml` (edge sanity case).
- Model profiles with `source_url` from HF configs: `llama3_8b.yaml`, `llama2_70b.yaml`,
  `mistral_7b.yaml`, `gpt3.yaml`, `gemma4.yaml`.
- `bwz list` CLI; round-trip tests for every profile.

**Done when:** profiles round-trip YAML→spec→YAML; golden test: Llama-3-8B param count = 8.03 B ±0.5%.

## Session 3 — M2: graph + operator cost models

- `graph/`: `ops.py`, `builder.py`, `transformer.py` (separate **prefill and decode** graphs),
  `cnn.py`, `dag.py` (topo sort, critical path, liveness).
- `operators/`: matmul (2·M·N·K, tiled traffic, tail effect), attention (vanilla + FlashAttention-2
  + GQA; flash changes bytes never FLOPs), conv (direct/im2col only; Winograd/FFT in the
  refinement backlog), norm, elementwise.
  FLOP/byte formulas exactly per PROMPT.md §3.2, documented in `docs/MODEL.md` as implemented.
- Golden tests in the same commits: GPT-3 prefill FLOPs ≈ `6·N·D` within 3%; Gemma-4 forward
  FLOPs (batch 1) ≈ `2·N·S` within 2%; Llama-3-8B KV cache @ 8k fp16 = 1.0 GB ±2%.

**Done when:** all M2 golden tests pass.

## Session 4 — M3: single-chip analysis → Report

- `analysis/`: `roofline.py` (flat: compute ridge vs DRAM ridge; the single SRAM tile buffer's real
  capacity sets the tiling constraint, and double buffering trades against it), `tiling.py`,
  `memory.py` (waterfall + feasibility with cheapest fixes), `schedule.py`, `bottleneck.py`. All
  constants in `calibration.py` with source comments (start with documented defaults: ~0.8–0.9
  achievable HBM efficiency and one ~0.5–0.7 achieved-flops fraction for tensor cores — fitted in
  Session 5). The multi-level hierarchy, ws/os/rs loop-order search, and Winograd/FFT are not in
  this session — see the refinement backlog below.
- `report.py` per PROMPT.md §5 (confidence + assumptions fields mandatory).
- `bwz run` with rich terminal table (README shows the target output).
- Property tests (hypothesis): bandwidth↑ never latency↑; INT8 never slower than FP16; monotonic in batch.

**Done when:** the CLAUDE.md sanity checks hold — Llama-3-8B fp16 decode on H100 → DRAM_BW_BOUND,
~35–55 tok/s; prefill @2k → COMPUTE_BOUND; Gemma-4 batch 1 → LATENCY_BOUND. Snapshot tests for all three.

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

Trigger: users need a per-level breakdown, or Session 5 validation shows a systematic error the
flat model cannot explain.

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
make validate         # prints predicted-vs-published table, MAPE within target
```
