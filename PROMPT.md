# Claude Code Build Prompt — `bandwidth-zen`: LLM/CNN → Chip Performance Estimator

> **How to use this file:** put it at the repo root as `PROMPT.md`, open Claude Code in an empty
> directory, and paste the "Kickoff" block at the bottom. Claude Code should read this file first,
> then `CLAUDE.md`, then start at Milestone 0. Work milestone-by-milestone; do not skip ahead.

---

## 0. Role and objective

You are building **`bandwidth-zen`**, a full-stack analytical performance model that answers one question:

> *"Can I run this model on this chip, and if so, how fast — and what is the limiter?"*

It is **not** a cycle-accurate RTL simulator and **not** a compiler. It is a **fast analytical
model** (roofline + tiling + collective-communication cost) that evaluates 100+ configurations in
under a minute and is explainable at every step: every number the UI shows must be traceable to a
formula and its inputs.

**Non-goals (do not build these):** kernel autotuning against real silicon, ONNX graph execution,
training-loop simulation beyond a memory/FLOP multiplier, CUDA code generation, user accounts,
databases, cloud deployment.

**Primary users:** ML systems researchers, hardware architects, inference engineers doing
model/hardware co-design.

---

## 1. Ground rules (these override anything else in this document)

1. **Accuracy honesty.** Every prediction returns a `confidence` field and the assumptions that
   produced it. Never present a modelled number as measured. The UI must label outputs as
   *estimates*.
2. **No magic constants.** Every derating factor, efficiency coefficient, or fudge factor lives in
   `backend/bwz/calibration.py` with a comment citing its source or the fitted dataset. No
   `* 0.85` inline anywhere else.
3. **Units are explicit.** Suffix every variable: `_flops`, `_bytes`, `_bytes_per_s`, `_s`, `_ns`,
   `_j`. Internally the engine uses **SI base units only** (FLOPs, bytes, bytes/s, seconds, joules).
   Conversion to GB/s, TFLOP/s, ms happens *only* at the serialization boundary.
4. **Pure core.** `bwz/analysis/` must be importable without FastAPI, without I/O, without
   globals. The engine is a pure function: `(ModelSpec, HardwareSpec, DeploymentSpec) -> Report`.
5. **Determinism.** Same inputs → byte-identical report (except a `generated_at` timestamp). Sort
   all dict iteration; no set ordering in outputs.
6. **Test-first for the physics.** Any file under `analysis/` gets its test written before or with
   the implementation. Analytical closed forms get golden-value tests.
7. **Correct the source spec where it is wrong.** The originating design doc contains hardware
   errors (e.g. "H100: 1456 TFLOPS FP32, 192 GB/s"). Real H100 SXM5 ≈ 67 TFLOP/s FP32 vector,
   ~989 TFLOP/s dense FP16/BF16 tensor, 3.35 TB/s HBM3, 80 GB. Use vendor datasheets, put the
   citation in the YAML profile's `source_url`, and note the correction in `docs/CORRECTIONS.md`.

---

## 2. Repository layout (create exactly this)

```
bandwidth-zen/
├── CLAUDE.md                      # agent operating manual (already provided)
├── README.md                      # public docs (already provided)
├── PROMPT.md                      # this file
├── LICENSE                        # Apache-2.0
├── Makefile                       # dev, test, lint, fmt, docker
├── docker-compose.yml
├── .github/workflows/ci.yml
├── docs/
│   ├── MODEL.md                   # the analytical model, with every formula + derivation
│   ├── CALIBRATION.md             # measured-vs-predicted tables, error analysis
│   ├── CORRECTIONS.md             # deviations from the original design doc, with reasons
│   └── SCHEMA.md                  # YAML/JSON spec reference for models & chips
├── backend/
│   ├── pyproject.toml             # uv / hatchling, py>=3.11, ruff + mypy strict
│   ├── bwz/
│   │   ├── __init__.py
│   │   ├── units.py               # SI helpers, formatting, prefix parsing ("3.35 TB/s")
│   │   ├── calibration.py         # ALL empirical constants, one place
│   │   ├── spec/
│   │   │   ├── model_spec.py      # pydantic: ModelSpec, LayerSpec, TransformerSpec, CNNSpec
│   │   │   ├── hardware_spec.py   # pydantic: Chip, ComputeUnit, MemoryLevel, Interconnect
│   │   │   ├── deployment.py      # pydantic: batch, seq, precision, parallelism, mode
│   │   │   └── loaders.py         # YAML/JSON → spec, with helpful validation errors
│   │   ├── graph/
│   │   │   ├── ops.py             # Operation, Tensor, OpType enum
│   │   │   ├── builder.py         # ModelSpec → ComputeGraph (DAG)
│   │   │   ├── transformer.py     # decoder/encoder expansion, prefill + decode graphs
│   │   │   ├── cnn.py             # conv/pool/bn/residual expansion
│   │   │   └── dag.py             # topo sort, critical path, liveness/peak-activation
│   │   ├── operators/             # pluggable cost models, one file per family
│   │   │   ├── base.py            # OperatorCostModel ABC + registry decorator
│   │   │   ├── matmul.py          # GEMM: flops, tiled DRAM traffic, tail-effect utilization
│   │   │   ├── conv.py            # direct / im2col (Winograd/FFT deferred to M8)
│   │   │   ├── attention.py       # vanilla, FlashAttention-2, GQA/MQA, sliding-window, paged KV
│   │   │   ├── normalization.py   # layernorm, rmsnorm, softmax, batchnorm (fused/unfused)
│   │   │   ├── elementwise.py     # activations, residual adds, rope
│   │   │   └── custom.py          # user-supplied {flops, bytes} ops
│   │   ├── analysis/
│   │   │   ├── roofline.py        # per-op roofline: compute ridge vs DRAM ridge (hierarchy at M8)
│   │   │   ├── tiling.py          # tile-size search, reuse distance, DRAM traffic model
│   │   │   ├── memory.py          # capacity planning: weights, KV cache, activations, fit/spill
│   │   │   ├── schedule.py        # timeline construction, overlap, list scheduling
│   │   │   ├── parallelism.py     # TP/PP/DP/EP sharding transforms on the graph
│   │   │   ├── collectives.py     # alpha-beta cost for allreduce/allgather/a2a/p2p, topologies
│   │   │   ├── power.py           # energy-per-op table → power & efficiency estimates
│   │   │   ├── bottleneck.py      # ranking, classification, optimization suggestions
│   │   │   └── sweep.py           # config space enumeration + Pareto frontier
│   │   ├── report.py              # Report/LayerResult/Timeline dataclasses → JSON schema
│   │   ├── cli.py                 # typer: `bwz run|sweep|validate|list`
│   │   └── api/
│   │       ├── app.py             # FastAPI app + CORS + error handlers
│   │       ├── routes.py          # endpoints (see §6)
│   │       └── schemas.py         # request/response models (re-export spec + report)
│   ├── profiles/
│   │   ├── chips/                 # h100_sxm.yaml, a100_80gb.yaml, tpu_v4.yaml, mi300x.yaml,
│   │   │                          # jetson_orin.yaml, generic_npu.yaml, cpu_xeon.yaml
│   │   └── models/                # gpt3.yaml, llama3_8b.yaml, llama2_70b.yaml, mistral_7b.yaml,
│   │                              # bert_base.yaml, gemma4.yaml, mobilenetv3.yaml, vit_b16.yaml,
│   │                              # sd_unet.yaml, mixtral_8x7b.yaml
│   └── tests/
│       ├── unit/                  # per-module, golden values
│       ├── integration/           # full-report snapshots
│       └── validation/            # predicted vs published MLPerf/vendor numbers
└── frontend/
    ├── package.json               # vite + react 18 + typescript + tailwind
    ├── src/
    │   ├── api/client.ts          # typed fetch, types generated from OpenAPI
    │   ├── state/                 # zustand store: config ⟷ URL query string (shareable links)
    │   ├── components/
    │   │   ├── ModelPicker.tsx
    │   │   ├── ChipPicker.tsx     # preset dropdown + editable custom fields
    │   │   ├── DeploymentPanel.tsx
    │   │   ├── SummaryCards.tsx
    │   │   ├── RooflinePlot.tsx   # log-log, one dot per op, ridge point, hover detail
    │   │   ├── GanttTimeline.tsx  # canvas/d3, resource lanes, zoom + pan
    │   │   ├── MemoryChart.tsx    # capacity waterfall + KV-cache growth over tokens
    │   │   ├── BottleneckTable.tsx
    │   │   ├── ParetoExplorer.tsx # sweep scatter, brush to filter, click to load config
    │   │   └── AssumptionsDrawer.tsx  # every assumption behind the current number
    │   └── pages/Dashboard.tsx
    └── tests/                     # vitest + testing-library
```

---

## 3. The analytical model — implement exactly this, document it in `docs/MODEL.md`

### 3.1 Roofline (per operation, per memory level)

```
AI            = flops / bytes_moved_at_level          [FLOP/byte]
peak_flops_s  = compute units × ops/cycle × clock × dtype_throughput_multiplier
bw_ceiling    = bandwidth_bytes_per_s × AI
achievable    = min(peak_flops_s × util_eff, bw_ceiling × bw_eff)
t_compute_s   = flops / (peak_flops_s × util_eff)
t_memory_s    = bytes_moved / (bandwidth_bytes_per_s × bw_eff)
t_op_s        = overlap ? max(t_compute_s, t_memory_s)
                        : t_compute_s + t_memory_s
t_op_s        = max(t_op_s, kernel_launch_overhead_s + memory_latency_s)   # floor for tiny ops
```

For v1 do a **flat roofline**: one compute ridge (`peak_flops_s × util_eff`) and one memory
ridge at the DRAM level. The on-chip memory is a single tile buffer whose **real capacity**
constrains tile sizes (§3.3) — capacity drives the HBM turnaround, the double-buffering depth,
and the memory-vs-compute verdict. Its bandwidth is assumed sufficient: typical GEMM tile AI is
10–30 FLOP/byte vs a 100–500 FLOP/byte SRAM ridge, and datasheets rarely publish it anyway.
`bytes_moved` at DRAM comes from the tiling model (§3.3), not from naive tensor sizes — this is
the single biggest fidelity win over a textbook roofline. A **hierarchical roofline** (separate
L1/SRAM, L2, DRAM ceilings, report which level binds) is deferred to M8, on user demand and only
if it improves validation accuracy.

`util_eff` is **not** a constant. Compute it from:
- **Tail/quantization effect:** `prod(ceil(dim_i / tile_i) * tile_i) / prod(dim_i)` over the
  systolic-array-mapped dims. A GEMM with M=1 on a 128×128 array gets ~1/128 utilization — this is
  exactly why LLM decode is catastrophic on big arrays, and the model must show it.
- **Pipeline fill/drain:** `steps / (steps + array_depth - 1)`.
- **dtype support:** if the op's dtype isn't in the compute unit's `supported_dtypes`, fall back to
  the next-highest supported dtype and record a warning in the report.

### 3.2 FLOP and byte counts (all matmuls counted as `2·M·N·K`)

**Transformer decoder layer** — hidden `d`, heads `h`, KV heads `h_kv`, head dim `dh = d/h`, FFN
`d_ff`, batch `B`, tokens processed `S`, context length `C`, layers `L`, vocab `V`:

| Op | FLOPs | Notes |
|---|---|---|
| QKV projection | `2·B·S·d·(d + 2·h_kv·dh)` | GQA-aware |
| RoPE | `~6·B·S·d` | elementwise |
| Attention scores `QKᵀ` | `2·B·h·S·C·dh` | prefill causal → halve |
| Softmax | `~5·B·h·S·C` | memory-bound |
| `AV` | `2·B·h·S·C·dh` | prefill causal → halve |
| Output projection | `2·B·S·d·d` | |
| FFN (SwiGLU, 3 mats) | `6·B·S·d·d_ff` | GELU/2-mat variant: `4·B·S·d·d_ff` |
| RMSNorm ×2 | `~8·B·S·d` | |
| LM head | `2·B·S·d·V` | once, not per layer |

**Phases — model them separately, they have opposite characters:**
- **Prefill** (`S = S_in`, `C = S_in`, causal masking halves attention): compute-bound, high AI.
- **Decode** (`S = 1`, `C` grows from `S_in` to `S_in + T`): memory-bound. Per step, bytes moved ≈
  `weight_bytes + kv_cache_bytes(C)`; latency floor is `bytes / DRAM_BW`. Integrate over `T` steps
  (closed form for linear KV growth; do not loop 4096 times).

**KV cache:** `kv_bytes = 2 · L · B · C · h_kv · dh · sizeof(kv_dtype)` (×`num_pages/page_util` if
paged). Report the **memory wall**: the max context that fits given weights + activations, and the
context at which KV-cache traffic exceeds weight traffic per decode step.

**FlashAttention:** does not change FLOPs; it changes bytes — the `B·h·S·C` score matrix is never
materialized to DRAM. Model as: DRAM traffic = Q + K + V + O tiles only, with SRAM traffic
`O(S·C·dh / block)`. Show the AI improvement explicitly (this is a headline feature).

**Convolution:**
```
flops   = 2 · B · H_out · W_out · C_out · (C_in / groups) · K_h · K_w
bytes   = weight_bytes + input_act_bytes·reload_factor + output_act_bytes
```
Algorithm selection (v1: **direct** and **im2col+GEMM** only — the latter adds
`K_h·K_w×` input expansion traffic; model it. **Winograd F(2×2,3×3)** (2.25× FLOP reduction,
+transform overhead, 3×3 stride-1 only, FP16+ only) and **FFT** (large kernels) are deferred to
M8.) Pick per-layer by minimizing modelled latency and report the choice.

**Depthwise conv is memory-bound** — AI ≈ `2·K_h·K_w / (2 + bytes_per_elem)`. MobileNet on a big
systolic array must show terrible utilization. If it doesn't, the model is wrong.

### 3.3 Tiling and DRAM traffic

For `C[M,N] = A[M,K] · B[K,N]` with tiles `(Tm, Tn, Tk)` and on-chip capacity `S_bytes`:

```
constraint:  (Tm·Tk + Tk·Tn + Tm·Tn) · elem_bytes · double_buffer ≤ S_bytes
dram_bytes = M·K·ceil(N/Tn)          # A re-read per N-tile
           + K·N·ceil(M/Tm)          # B re-read per M-tile
           + M·N·(write + accum_reads)
```
Search tiles by enumerating powers of two ≥ the systolic dimension, filter by the capacity
constraint, minimize `max(t_compute, t_dram)`. Cache the search by `(shape, dtype, chip_id)`.
Report the chosen tile sizes and the resulting **reuse factor** — users want to see this.

The chip profile declares `dataflow: ws|os|rs`; v1 honours it directly — LLM weights are the
dominant reused operand (ws), CNN activations (os). A full ws/os/rs trade-off search with per-op
loop-order modelling to find which operand stays put is deferred to M8.

### 3.4 Multi-chip: sharding and collectives

Parallelism transforms on the graph (implement in `parallelism.py` as graph rewrites, not as
after-the-fact multipliers):

| Strategy | Shard | Comms per transformer layer |
|---|---|---|
| **Tensor (TP=N)** | heads across N; FFN `d_ff/N` | 2 × AllReduce of `B·S·d·dtype` (attn out + FFN out) |
| **Pipeline (PP=P)** | layers into P stages | P2P send of `B·S_micro·d` per stage boundary; bubble = `(P−1)/(M+P−1)` for M microbatches |
| **Data (DP=N)** | batch across N | inference: none. training: AllReduce of param grads |
| **Expert (EP=N)** | MoE experts across N | 2 × All-to-All of `B·S·d·top_k/N` |
| **Sequence/Ring** | sequence across N | ring P2P of K,V blocks |

Alpha-beta collective cost (`α` = link latency, `β` = 1/bandwidth per link, `N` ranks, `S` bytes):
```
ring allreduce   t = 2(N−1)·α + 2·(N−1)/N · S·β
ring allgather   t = (N−1)·α + (N−1)/N · S·β
tree allreduce   t = 2·log2(N)·α + 2·log2(N)·S·β/ (bw-limited variant: 2S·β)
all-to-all       t = (N−1)·α + (N−1)/N · S·β        # on full-bisection fabric
```
Choose ring for large `S`, tree for small `S` — implement the crossover, don't hardcode.
Model **hierarchical topologies**: intra-node NVLink vs inter-node InfiniBand have different
`(α, β)`; a 16-way allreduce across 2 nodes of 8 = intra-node reduce-scatter → inter-node allreduce
→ intra-node allgather. Get this right; it's where naive tools are most wrong.

**Overlap:** `t_layer = max(t_compute, t_comm)` only if the chip profile declares
`async_copy_engines > 0` and the strategy permits it (TP allreduce is a hard dependency and only
partially overlappable — model `overlap_fraction` from calibration, default 0.0 for TP allreduce,
0.8 for PP p2p). Report **comms as % of critical path**.

### 3.5 Memory capacity planning (`memory.py`)

Compute and report a waterfall: `weights + KV cache + peak activations + workspace + framework
overhead ≤ device HBM × usable_fraction`. Peak activations come from **liveness analysis** on the
DAG (interval graph, max overlap), not the sum of all tensors. If it doesn't fit, return a
`FEASIBILITY_FAIL` report that says by how much and lists the cheapest fixes (lower precision,
more TP, shorter context, smaller batch) with the numbers for each.

### 3.6 Power/energy (rough, clearly labelled)

Energy table in `calibration.py` (pJ/op at 7nm/5nm, cite Horowitz ISSCC 2014 + updates):
INT8 MAC ~0.2 pJ, FP16 MAC ~1.5 pJ, SRAM read (8KB) ~10 pJ/word, DRAM/HBM access ~15–40 pJ/bit.
`E = Σ(flops·e_mac) + Σ(bytes_level·e_level) + comm_bytes·e_link + static_power·t`.
Report joules/token, joules/image, TOPS/W. Label it `±50%`.

### 3.7 Bottleneck classification and suggestions

Classify each op: `COMPUTE_BOUND`, `DRAM_BW_BOUND`, `SRAM_BW_BOUND`, `LATENCY_BOUND`
(launch/dependency dominated), `CAPACITY_BOUND` (spilling), `COMM_BOUND`, `UNDERUTILIZED`
(tail effect). Rank by contribution to critical path.

Suggestions must be **quantified**, generated from a rule table, each carrying the predicted delta
computed by re-running the engine on the modified config:
- `DRAM_BW_BOUND` + FP16 weights → "quantize to INT8: −38% latency (re-simulated)"
- attention dominant + no flash → "enable FlashAttention: AI 2.1 → 47, −61% attention time"
- decode + `util < 5%` → "increase batch to 32: throughput 14 → 380 tok/s, latency +12%"
- `COMM_BOUND` with TP → "reduce TP to 2, add PP 2: comms 41% → 9% of critical path"
- `CAPACITY_BOUND` → "KV cache is 62 GB at C=32k; use GQA/paged KV/INT8 KV: fits at 15.5 GB"

---

## 4. Data schemas (write these to `docs/SCHEMA.md`, validate with pydantic v2)

### 4.1 Chip profile YAML

```yaml
name: NVIDIA H100 SXM5
vendor: NVIDIA
source_url: https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-datasheet
process_nm: 4
clock_ghz: 1.755
compute_units:
  - name: tensor_core
    count: 528
    ops_per_cycle_per_unit: 1024          # MACs; flops = 2 × MACs
    supported_dtypes: [fp16, bf16, fp8, int8, tf32]
    dtype_multipliers: {fp16: 1.0, bf16: 1.0, fp8: 2.0, int8: 2.0, tf32: 0.5}
    structured_sparsity_speedup: 2.0
    systolic_dims: [16, 16]               # drives the tail-effect model
    dataflow: ws
  - name: cuda_core
    count: 16896
    ops_per_cycle_per_unit: 2
    supported_dtypes: [fp32, fp16]
memory:
  - {name: L1, level: 1, capacity_bytes: 33554432, bandwidth_bytes_per_s: 1.3e14, latency_ns: 30}
  - {name: L2, level: 2, capacity_bytes: 52428800, bandwidth_bytes_per_s: 1.2e13, latency_ns: 200}
  - {name: HBM3, level: 3, capacity_bytes: 85899345920, bandwidth_bytes_per_s: 3.35e12, latency_ns: 600}
usable_memory_fraction: 0.90
async_copy_engines: 4
kernel_launch_overhead_s: 3.0e-6
tdp_w: 700
static_power_w: 120
interconnect:
  intra_node: {name: NVLink4, bandwidth_bytes_per_s: 4.5e11, latency_s: 2.0e-6, topology: fully_connected}
  inter_node: {name: IB NDR, bandwidth_bytes_per_s: 5.0e10, latency_s: 5.0e-6, topology: fat_tree}
cost_usd: 30000
```
**v1 consumption note:** analysis reads only the top on-chip level (tile-buffer capacity) and the
deepest level (DRAM bandwidth); intermediate levels are schema-only until the hierarchical
roofline lands at M8.

### 4.2 Model spec YAML (two flavours: parametric and explicit-layer)

```yaml
name: Llama-3-8B
family: transformer_decoder
source_url: https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json
params:
  layers: 32
  hidden: 4096
  heads: 32
  kv_heads: 8            # GQA
  ffn_hidden: 14336
  ffn_type: swiglu
  vocab: 128256
  max_context: 8192
  norm: rmsnorm
  positional: rope
  tie_embeddings: false
```
```yaml
name: MobileNetV3
family: cnn
input: {batch: 1, channels: 3, height: 224, width: 224}
layers:
  - {type: conv, name: conv1, out_channels: 16, kernel: [3,3], stride: 2, padding: 1}
  - {type: bottleneck_block, name: block0, repeat: 1, width: 16, stride: 1}
  - {type: bottleneck_block, name: block1, repeat: 2, width: 24, stride: 2}
  ...
```
Also accept an **explicit op list** (`type: custom, flops:, bytes:, parallel_dims:`) so users can
model anything, and an **ONNX import path** (optional, `onnx` extra) that lowers to the same op list.

### 4.3 Deployment spec

```yaml
mode: inference            # inference | training
phase: both                # prefill | decode | both  (transformer only)
batch: 1
input_tokens: 2048
output_tokens: 256
precision: {weights: int8, activations: fp16, accumulate: fp32, kv_cache: fp16}
per_layer_precision_overrides: {"layer.0.attn.qkv": {weights: fp16}}
sparsity: {type: structured_2_4, ratio: 0.5, applies_to: [matmul]}
attention_impl: flash2     # vanilla | flash2 | paged | sliding_window
parallelism: {tp: 2, pp: 1, dp: 1, ep: 1, microbatches: 8}
num_chips: 2
optimize_for: latency      # latency | throughput | energy
constraints: {max_latency_s: 0.05, max_power_w: 700, max_memory_bytes: 8.0e10}
```

---

## 5. Report schema (the single contract between engine, CLI, and UI)

```python
Report:
  meta: {bwz_version, generated_at, model_name, chip_name, config_hash}
  feasible: bool
  infeasibility: list[str] | None
  summary:
    latency_s, ttft_s, tpot_s, throughput_per_s, tokens_per_s
    compute_utilization, dram_bandwidth_utilization, achieved_tflops
    peak_memory_bytes, memory_headroom_bytes
    energy_j, avg_power_w, tokens_per_joule
    confidence: {level: high|medium|low, expected_error_pct: float, reasons: [str]}
  memory_breakdown: {weights_bytes, kv_cache_bytes, activations_bytes, workspace_bytes, ...}
  layers: [LayerResult]        # per-op: flops, bytes_per_level, ai_per_level, t_compute_s,
                               # t_memory_s, t_comm_s, t_total_s, bound_by, utilization,
                               # tile_plan, algorithm_chosen, share_of_critical_path
  timeline: [TimelineEvent]    # {resource, op_name, start_s, end_s, kind: compute|memory|comm|idle}
  critical_path: [op_name]
  bottlenecks: [{rank, op_name, bound_by, cost_share, explanation}]
  suggestions: [{id, title, rationale, predicted_delta: {latency_pct, memory_pct, ...}, patch: {}}]
  assumptions: [str]           # every modelling assumption used, human-readable
```

Emit the JSON Schema to `docs/report.schema.json` and generate TypeScript types from it in CI.

---

## 6. HTTP API

```
GET  /api/health
GET  /api/chips                     → list of chip profiles (id, name, headline specs)
GET  /api/chips/{id}
POST /api/chips/validate            → validate a custom chip YAML/JSON
GET  /api/models
GET  /api/models/{id}
POST /api/analyze                   → {model, hardware, deployment} → Report
POST /api/sweep                     → {base_config, knobs: {batch: [1,2,...], tp: [1,2,4], ...}}
                                      → {points: [{config, summary}], pareto: [idx], frontier_axes}
POST /api/compare                   → N configs → aligned side-by-side summaries
GET  /api/openapi.json
```
`/api/sweep` must cap the cartesian product (default 512 points), run in a process pool, and stream
progress via SSE (`/api/sweep/stream`) so the UI can render the Pareto plot incrementally.

---

## 7. Frontend requirements

Single dashboard page, three columns: **config (left) → summary + plots (center) → assumptions and
suggestions (right)**. Requirements:

- Config state serializes into the URL query string so any analysis is a shareable link.
- Debounced auto-analyze (300 ms) — the engine is fast enough for live updates.
- **Roofline plot**: log-log, memory-bound diagonal and compute-bound plateau, ridge point marked,
  one dot per op sized by latency contribution, colored by bound type, hover shows the full
  LayerResult.
- **Gantt timeline**: canvas-rendered (thousands of events), lanes for compute / DRAM / L2 /
  each interconnect link, zoom+pan, brushing selects ops in the table.
- **KV-cache growth chart**: memory vs generated tokens, with the capacity wall drawn as a red line
  and the OOM token index annotated.
- **Pareto explorer**: scatter with selectable axes, frontier highlighted, click a point to load
  that config into the left panel.
- **Assumptions drawer**: always reachable in one click; lists `report.assumptions` verbatim.
- Accessibility: keyboard-navigable, no color-only encoding (use shape/pattern too), WCAG AA.
- Zero backend calls needed for presets — hydrate from `/api/chips` and `/api/models` on load.

---

## 8. Milestones — implement in order, each ends green and committed

**M0 — Scaffold.** Repo layout, `pyproject.toml` (ruff, mypy strict, pytest), `Makefile`,
pre-commit, CI workflow, `LICENSE`, empty docs. `make test` passes on an empty suite.
*DoD:* `make dev` starts backend on :8000 and frontend on :5173; `/api/health` returns 200.

**M1 — Specs and loaders.** Pydantic models, YAML loaders with actionable error messages,
5 chip profiles + 5 model profiles, `bwz list` CLI.
*DoD:* every shipped profile round-trips YAML→spec→YAML; validation errors name the field and the
allowed values.

**M2 — Graph + operator costs.** Op/Tensor/ComputeGraph, transformer and CNN builders, matmul /
conv / attention / norm / elementwise cost models, DAG utilities (topo sort, critical path,
liveness).
*DoD:* golden tests — GPT-3 prefill FLOPs match the analytic `6·N·D` rule within 3%; Gemma-4
forward FLOPs (batch 1) match the analytic `2·N·S` rule within 2%; Llama-3-8B parameter count =
8.03 B ±0.5%; KV-cache size for Llama-3-8B @ 8k context, fp16 = 1.0 GB ±2%.

**M3 — Single-chip analysis.** Roofline (flat: compute ridge vs DRAM ridge, tile-buffer capacity
from SRAM; multi-level hierarchy deferred to M8), tiling search, memory capacity planning,
scheduling/timeline, bottleneck classification, `Report` emission, `bwz run` CLI with a rich
terminal table.
*DoD:* Llama-3-8B fp16 decode on H100 predicts ~35–55 tok/s single-stream and is classified
`DRAM_BW_BOUND`; prefill @2k tokens is classified `COMPUTE_BOUND`; Gemma-4 batch-1 on H100 is
classified `LATENCY_BOUND`/`UNDERUTILIZED`. Snapshot tests for all three.

**M4 — API + minimal UI.** FastAPI endpoints, OpenAPI → TS types, React dashboard with pickers,
summary cards, roofline plot, bottleneck table.
*DoD:* end-to-end click-through produces a report in <500 ms; Playwright smoke test passes.

**M5 — Multi-chip.** Sharding graph rewrites for TP/PP/DP/EP, alpha-beta collectives with
hierarchical topology, overlap modelling, comms lanes in the timeline.
*DoD:* TP scaling of Llama-3-70B from 1→8 H100s shows sub-linear speedup with comms share rising;
PP bubble matches `(P−1)/(M+P−1)` analytically in a unit test; 8-way allreduce cost crosses over
from tree to ring at the modelled message size.

**M6 — Precision, sparsity, trade-off explorer.** Per-layer precision, 2:4 structured sparsity,
sweep engine with process pool + SSE, Pareto computation, ParetoExplorer UI, quantified suggestions
(re-simulated deltas).
*DoD:* 256-point sweep completes in <60 s on 8 cores; Pareto set verified by brute force in a test.

**M7 — Validation and polish.** `docs/CALIBRATION.md` with predicted-vs-published tables for at
least 8 (model, chip, config) points from MLPerf Inference and vendor blogs; fit the calibration
constants; energy model; KV-cache chart; assumptions drawer; README screenshots/GIF.
*DoD:* mean absolute percentage error ≤ 20% on the validation set with ≥ 6 of 8 points inside ±15%;
every remaining outlier has a written explanation.

**M8 — Refinement backlog (on user demand).** Full-fidelity modelling deferred from M2/M3:
hierarchical multi-level roofline with per-level byte accounting, full ws/os/rs dataflow
loop-order search, Winograd/FFT convolution selection, L2 reuse effects for activation-heavy CNNs.
These land only when users ask for them.
*DoD:* each refinement ships with golden tests and re-runs `make validate`; a refinement is kept
only if it improves (or does not regress) MAPE on the published reference set — otherwise the flat
model remains the default.

---

## 9. Testing requirements

- **Unit:** every formula in §3 gets a hand-computed golden value in the test docstring.
- **Property (hypothesis):** latency is monotonic non-decreasing in batch and sequence length;
  doubling bandwidth never increases predicted latency; INT8 never slower than FP16 on a chip that
  supports both; total layer time ≥ critical path ≥ max single-layer time.
- **Invariant:** `sum(layer.share_of_critical_path) ≈ 1.0`; memory breakdown sums to peak;
  timeline events never overlap on the same resource.
- **Snapshot:** full JSON reports for 6 canonical configs, reviewed on diff.
- **Validation:** `tests/validation/reference_points.yaml` holds published measurements; the test
  asserts MAPE thresholds and prints a table. Mark `@pytest.mark.validation`, run in CI nightly.
- Coverage gate: ≥ 90% on `bwz/analysis/` and `bwz/operators/`.

---

## 10. Deliverables checklist

- [ ] Working `make dev` (backend + frontend) and `docker compose up`
- [ ] `bwz run --model llama3_8b --chip h100_sxm --batch 1 --input-tokens 2048 --output-tokens 128`
- [ ] `bwz sweep --config sweeps/batch_precision.yaml --out results.json`
- [ ] `docs/MODEL.md` with every formula, derivation, and its limits of validity
- [ ] `docs/CALIBRATION.md` with the validation table and error analysis
- [ ] `docs/CORRECTIONS.md` listing every place this build deviates from the original design doc
- [ ] README with quickstart, screenshot, accuracy statement, and contribution guide
- [ ] CI green: ruff, mypy strict, pytest, vitest, build

---

## Kickoff (paste this into Claude Code)

```
Read PROMPT.md, then CLAUDE.md, in full before writing any code.

Then:
1. Restate the plan for Milestone 0 in ≤10 bullets and list any decisions you're making that
   PROMPT.md leaves open. Ask me only about decisions that are genuinely ambiguous — otherwise
   pick a sane default, note it in docs/CORRECTIONS.md, and proceed.
2. Implement Milestone 0 only. Run `make test` and `make lint`. Commit.
3. Stop and show me a summary + the diff stat before starting Milestone 1.

Rules for the whole build:
- Follow the Ground Rules in PROMPT.md §1 without exception, especially SI units, the single
  calibration module, and the pure-core constraint.
- Write the test for any physics/formula code in the same commit as the code.
- Never leave a milestone with failing tests, type errors, or lint errors.
- If a formula in PROMPT.md looks wrong to you, say so, explain why, propose the fix, and record
  the decision in docs/MODEL.md rather than silently diverging.
```
