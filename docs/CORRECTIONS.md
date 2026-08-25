# Corrections and deviations

Every place this build deviates from the original design doc / `PROMPT.md`, with the reason
and where the decision was taken. Per PROMPT.md ground rule #7 and CLAUDE.md working style:
we don't silently diverge.

## D1 — `CLAUDE_CODE_PROMPT.md` renamed to `PROMPT.md` (2026-08-06)

The build prompt shipped as `CLAUDE_CODE_PROMPT.md`; its own header and all references in
`CLAUDE.md` expect it at `PROMPT.md`. Renamed, no content change.

## D2 — Frontend and HTTP API deferred past Phase 1 (2026-08-06)

PROMPT.md M0 scaffolds backend + frontend together and its DoD includes `/api/health`.
Decision (user, plan approval 2026-08-06): Phase 1 targets calibration/backtesting, for which
`bwz` CLI + `make validate` suffice. The frontend and FastAPI layer land at M4 as specified.
`make dev`, `make types`, and `make docker` are no-op stubs until then.

## D3 — Milestone order: early validation slice before M4–M6 (2026-08-06)

PROMPT.md places validation at M7. Decision (user, plan approval): after M3 (single-chip
analysis) we immediately build the validation harness and fit calibration constants against
published single-chip reference points (H100, A100, MI300X). Multi-chip (M5) and the
DeepSeek-cluster backtest follow as the next phase. Rationale: backtesting accuracy is the
project's first deliverable of value; calibrating early catches modelling errors before they
propagate into the multi-chip layer.

## D4 — Samsung chip profiles deferred (2026-08-06)

Requested target vendors included Samsung. No Samsung datacenter accelerator has public
benchmarks rich enough to backtest against (mobile Exynos NPU data is too thin). Decision
(user): start with NVIDIA H100/A100 and AMD MI300X, revisit Samsung once the validation
harness exists.

## D5 — Flat roofline for v1; multi-level hierarchy deferred to M8 (2026-08-06)

M3 originally specified a hierarchical roofline (separate L1/SRAM, L2, DRAM ceilings, report which
level binds), and §3.2/§3.3 described Winograd/FFT conv selection and a ws/os/rs loop-order search.
Decision (user, 2026-08-06): v1 models one compute ridge (TOPS × achieved fraction) and one DRAM
ridge. The on-chip memory is a single tile buffer whose **capacity** constrains tile sizes — the
turnaround to HBM, the double-buffering depth, and the memory-vs-compute verdict all flow from
capacity, not from SRAM bandwidth. SRAM bandwidth is rarely published and rarely binds for
GEMM-shaped ops (tile AI 10–30 FLOP/byte vs a 100–500 FLOP/byte SRAM ridge). The deferred items are
preserved as an M8 refinement backlog, landed only on user demand and only if they improve
validation MAPE over the flat model.

## D5a — v1 machine model: DRAM + SRAM-as-capacity + compute (2026-08-07)

Reaffirms and sharpens D5 after the alternative was considered and rejected. Decision
(user, 2026-08-07): the first model is deliberately the simplest thing that can be right — three
elements, no more.

| element | what it contributes | what it does **not** contribute |
|---|---|---|
| External memory (DRAM/HBM) | **bandwidth** — the only bandwidth ceiling in v1 | — |
| On-chip SRAM | **capacity** — sets the resident fraction and the double-buffering headroom | no bandwidth term |
| Compute engine | **TOPS** — the compute ceiling | — |

SRAM earns its place through capacity in two distinct ways, and both are real:

1. **Residency.** `r = min(1, sram_bytes / W)` is the weight fraction that need not be re-streamed
   from DRAM each token, so it *removes DRAM traffic*: `t_dram = (1 - r) · W / bw_dram`.
2. **Double buffering.** Overlapping the next tile's load with the current tile's compute requires
   room for both. When capacity allows it, phase time is the `max` of the load and compute paths;
   when it does not, the loads serialize behind compute and it is the `sum`. This is where the
   "max, never the sum" rule earns its keep instead of being asserted.

```
t_dram    = (1 - r) * W / bw_dram
t_compute = 2 * params * S / tops          # S = 1 for decode
t_fixed   = n_ops * per_op_overhead_s
t_phase   = max(t_dram + t_kv, t_compute) + t_fixed      # double buffering feasible
          = t_dram + t_kv + t_compute      + t_fixed      # otherwise
bound     ∈ {DRAM, COMPUTE, LATENCY}
```

The rest of D5 stands: one compute ridge, no L1/L2/L3 hierarchy, no loop-order search, no
Winograd/FFT. Chip profiles still carry `bandwidth_bytes_per_s` on every memory level per
PROMPT.md §4.1 — v1 analysis simply does not read it above the DRAM level, and says so in
`report.assumptions`.

**Consequence for the D8 goldens:** with no on-chip bandwidth term, the supplied
`chip_b` 1B decode figure of **128 tok/s has no term that produces it** — it was
`r·W/bw_onchip` and nothing else. Fully resident means `t_dram = 0`.

*Updated at M3:* the term that takes over is **compute**, not the fixed per-op cost as predicted
here. `t_compute = 2·params/tops` is 38 µs only if the 512×512 array is fully utilised; at batch 1
it runs at 1/513 of peak (see D14), giving 20.8 ms and **48 tok/s**. That is a physically grounded
answer rather than a placeholder, so the cell is `COMPUTE_BOUND` and the golden is 48 tok/s, not
128 and not a latency artefact.

## D5b — Why on-chip bandwidth is absent, and what it would take to add it (2026-08-07)

An on-chip bandwidth ceiling (`t_onchip = r·W/bw_onchip`) was proposed, analysed, and **deliberately
not taken** (user, 2026-08-07: "skip the SRAM BW and keep the model simple"; D5a). The analysis is
kept because it is the justification for the omission, not merely a rejected alternative:

1. **`bw_onchip` conflates two physical paths.** Local SRAM → PE array (weight feed) never crosses
   the NoC; the NoC carries inter-core activations and reductions. The supplied estimate of
   128 GB/s is a NoC bisection figure (1 Tbit/s ÷ 8), not an SRAM read bandwidth. A structural
   check on the array — 512×512 MACs at 0.8 GHz with the ÷8 bit-serial tax needs ≈51 GB/s per core,
   ≈205 GB/s SoC, merely not to starve — puts the true weight-feed path *above* the figure in use.
   Equivalently: at 128 GB/s the on-chip ridge is 1637 ops/byte, implying batch ≥ 819 to reach peak
   TOPS from SRAM, which is not a plausible design point.
2. **Streamed weights also pass through SRAM.** Non-resident weights land in an SRAM staging buffer
   and are then read by the array, so SRAM carries `W` reads + `(1-r)·W` writes, not `r·W`.

Consequence, quantified so the omission is bounded rather than hand-waved. Margin by which the
binding term beats a hypothetical `t_onchip` at the supplied 128 GB/s:

| config | binding term | margin |
|---|---|---|
| chip_a 4B / 2B / 1B decode | DRAM | 260× |
| chip_b 4B decode | DRAM | 11× |
| chip_b 2B decode | DRAM | 3.8× |
| **chip_b 1B decode** | COMPUTE (D14) | on-chip would bind |
| all prefill TTFT @ S=512, crossover S\* | DRAM / compute | large |

So every result except one is 3.8×–260× away from an on-chip ceiling even at the pessimistic
128 GB/s: omitting the term costs nothing there. The single exception is the fully-resident
`chip_b` 1B cell, and that is exactly the cell whose input was least defensible — which is why
D5a and D14 route it to the array's batch-1 utilisation instead. Under refinement (2) the margins shrink to
1.9× / 1.6× / 1.26×, which would move several decode figures by ~25%, so the omission is material
enough to keep visible via the flip-margin field (`docs/PLAN.md` Session 4). Trigger to revisit: a
published SRAM organization (banks × width × clock), or a measured figure for either chip.

## D6 — Per-chip empirical fields live in the profile; defaults in `calibration.py` (2026-08-07)

CLAUDE.md #2 requires every empirical constant to live in `calibration.py`, but PROMPT.md §4.1 puts
`usable_memory_fraction` and `kernel_launch_overhead_s` in the chip profile. Both are right for
different reasons: the constant is per-chip, but an unsourced one is exactly what CLAUDE.md #2
exists to prevent. Rule adopted (build decision): a profile **may** carry a per-chip value **only**
with a source or an `estimates:` entry; when omitted, the value comes from `calibration.py`. The
"no inline magic numbers" ban is unchanged — a bare literal in `analysis/` or `operators/` is still
a review failure.

## D7 — Hypothetical and estimated profile fields declare themselves (2026-08-07)

CLAUDE.md forbids committing profile YAML without a `source_url`. Two shipped profiles cannot
satisfy it as written: `chip_b` is a hypothetical variant with no datasheet, and several `chip_a`
fields are engineering estimates rather than published figures. Rather than weaken the rule, the
schema gains two declarations (build decision):

- `hypothetical: true` + `derived_from: <chip_id>` — profile is not a product. Flagged in
  `bwz list` and in every report it appears in.
- `estimates: {<field>: <rationale>}` — per-field provenance for values that are estimated rather
  than sourced. The loader propagates every *touched* estimate into `report.assumptions`, so a
  report states which of its inputs were guessed and why.

`source_url` stays mandatory for any profile claiming to describe a real product.

## D8 — Edge-NPU worked example adopted as the M3 acceptance demo (2026-08-07)

The user supplied a complete two-chip INT8 edge-NPU comparison (residency-weighted weight traffic,
prefill/decode split, per-op layer table, chip-vs-chip crossover). Decision (user, 2026-08-07):
**additive** — the milestone structure is unchanged, and this example becomes M3's integration
snapshot. `chip_a` and `chip_b` join the M1 chip roster; H100/A100/MI300X still carry the Session 5
calibration, because they are the only profiles with published numbers to fit shared constants
against. `chip_a`/`chip_b` inherit those constants and are explicitly uncalibrated.

Deviations from the spec as supplied, with reasons:

- **`chip_a.sram_bytes` corrected to ≈55 MB.** The supplied 5210241024 (5.21 GB) yields `r = 100%`
  for a 3.9 GB model, contradicting the stated residencies. 55 MB reproduces all three
  (1.4% / 2.7% / 5.5% at 4B / 2B / 1B) and all six decode figures, plus the KV@4k case
  (117.3 ms, 8.5 tok/s) to within 1%. Carried as an `estimates:` entry.
- **Prefill attention FLOPs halved for causal masking.** The supplied
  `n_layers·4·d_model·S²` counts the full score matrix; a causal decoder computes half of it.
  Effect on TTFT is small (0.2 ms of 19.5 at S=512) but it shifts the crossover, which is where a
  golden sits.
- **Crossover goldens re-derived from the implementation.** The supplied `S* ≈ 759` (4B) does not
  follow from the supplied inputs; they give ≈735, or ≈745 with the causal fix. Both are outside
  the stated ±2%. The implementation's value is pinned instead of asserting a number the formulas
  do not produce.
- **Prefill ignores KV writes to DRAM** (≈4 ms at S=4096 for the 4B). Small, but stated as an
  assumption rather than silently absent.

The supplied ~2% assertions are **self-consistency goldens** derived from `[EST]` inputs, not
measurements. They live in `tests/unit/` and an integration snapshot — never in
`tests/validation/` or `docs/CALIBRATION.md`, which are reserved for published, citable reference
points. Keeping that line sharp is what makes the Session 5 calibration mean anything.

## D10 — Hardware-dependent effects stay in `analysis/`, not `operators/` (2026-08-08)

`docs/PLAN.md` Session 3 lists "tiled traffic, tail effect" under `operators/matmul.py`. Both
depend on the chip — tiling on SRAM capacity, the tail effect on `systolic_dims` — so putting them
in `operators/` would make a cost model take a `HardwareSpec` and blur the layering that CLAUDE.md's
architecture paragraph draws: `operators/` produces FLOPs and byte counts, `analysis/` runs "the
roofline, tiling search, memory planner".

Decision (build, 2026-08-08): operator cost models are **hardware-independent**. They report the
arithmetic performed and the *compulsory* traffic — each weight read once, each input read once,
each output written once. Tile re-reads, cache reuse and the tail effect land in `analysis/` at M3.

Practical benefit beyond tidiness: every M2 number is hand-checkable without reference to a chip,
which is what makes the golden tests in `tests/unit/test_operators.py` meaningful.

## D11 — Analytic prefill FLOPs are `2·N·D`, not `6·N·D` (2026-08-08)

`PROMPT.md` §8 M2 gives as a definition of done: "GPT-3 prefill FLOPs match the analytic `6·N·D`
rule within 3%". `6·N·D` is the *training* compute rule (Kaplan et al.): roughly `2·N·D` forward
plus `4·N·D` for the backward pass. Prefill is forward-only, so the correct rule is **`2·N·D`**,
and the DoD as written is wrong by a factor of three. PROMPT.md's own next line — "Gemma forward
FLOPs match `2·N·S` within 2%" — uses the forward rule, so the two cannot both be right.

Implemented as `2·N·D`. Measured at S=2048, batch 1: **722.6 TFLOP against `2·N·D` = 715.0 TFLOP,
+1.1%**, inside the stated 3%.

### The rule's limit of validity, which PROMPT.md does not state

`2·N·D` counts every parameter as doing two FLOPs per token. Embedding parameters do not: a
forward pass *gathers* rows from the table rather than multiplying by it, and the LM head runs on
one token, not `D`. So the rule is only accurate when embeddings are a small fraction of `N`:

| model | embeddings as % of N | actual vs `2·N·D` | actual vs `2·N_non-embedding·D` |
|---|---|---|---|
| GPT-3 175B | 0.4% | **+1.1%** | +1.4% |
| Llama-3-8B | 13.1% | −9.7% | +3.9% |
| Gemma-3-4B | 17.3% | −13.5% | +4.5% |

The general form is `2·N_non-embedding·D`, and the residual above it is attention, which grows as
`D²` and so is itself only negligible at short context — at S=512 Gemma-3-4B is +1.2% over that
form, at S=2048 it is +4.5%. Goldens are therefore pinned as: GPT-3 against `2·N·D` at S=2048
(±3%), Gemma-3-4B against `2·N_non-embedding·S` at S=512 (±2%).

## D12 — CLAUDE.md's Llama-3-8B decode sanity check looks ~3x too slow (2026-08-08)

CLAUDE.md's sanity checks say: "Llama-3-8B, fp16, batch 1, H100, decode → memory-bound, roughly
16 GB of weights moved per token, so ~35–55 tok/s."

The weight-traffic half is confirmed by M2: the decode graph moves **15.3 GB** per token (13.96 GB
of layer weights, 1.05 GB of untied LM head, ~0 for the embedding gather, ~0.3 GB of activations
and KV). But 16 GB at H100's 3.35 TB/s is **4.8 ms, i.e. ~209 tok/s** at peak and ~167 tok/s at a
plausible 80% achieved bandwidth. To arrive at 35–55 tok/s you would need to move 60–95 GB per
token, which contradicts the same sentence's own 16 GB figure.

Published single-GPU numbers for this configuration land around 100–140 tok/s, consistent with the
bandwidth argument and not with 35–55. **No change made at M2** — nothing here predicts tok/s yet.
Flagged so that M3 does not "fix" a correct model to hit an incorrect target, and so that Session 5
resolves the range against a citable measurement rather than against this line. The 16 GB figure
stands and is the one worth keeping.

## D13 — CLAUDE.md's Gemma latency-bound sanity check needs a much smaller model (2026-08-08)

CLAUDE.md: "Gemma-4, batch 1, H100 → latency/launch-bound, single-digit % utilization. Batch 128 →
good utilization."

The **utilisation** half is confirmed exactly: 0.24% at batch 1, 19.5% at batch 128. The **label**
is not. At fp16 a 3.88 G model moves 7.8 GB of weights per token, so on H100:

| term | value |
|---|---|
| `t_dram` | **2.70 ms** |
| `t_fixed` (274 dispatches × 3 µs) | 0.82 ms |
| `t_compute` | 0.21 ms |

DRAM wins by 3.3×. Latency *is* the runner-up — it beats compute by 4× — so "launch-bound" is
half-right, but the verdict is `DRAM_BW_BOUND`. Reaching a genuinely launch-bound regime takes a
model three orders of magnitude smaller: **MobileNetV3 (5.4 M parameters) on H100 comes out
`LATENCY_BOUND`**, and is now the sanity check that covers this case.

No constant was tuned to force the stated label. Doing so would have meant raising
`per_op_overhead_s` past 10 µs — well outside any defensible range for a GPU — to make a 4 B model
launch-bound, and that would have corrupted every other prediction to satisfy one line of prose.

## D14 — The systolic tail effect changes the D8 conclusion (2026-08-08)

CLAUDE.md requires it: "GEMM with M=1 on a 128×128 systolic array → utilization ≈ 1/128 from the
tail effect. If your utilization model doesn't reproduce this, it isn't modelling the array."

Implemented in `analysis/tiling.py` as three multiplicative losses:

```
utilisation = [K / padded(K, rows)] · [N / padded(N, cols)] · [M / (M + rows)]
```

The `M/(M+rows)` term is a weight-stationary array's pipeline fill and drain: one row of work still
costs `rows` cycles of latency. **This is the largest single correction the engine applies**, and it
falsifies a D8 assumption.

D8 computes `t_compute = 2·params/tops`, i.e. at 100% array utilisation. On a 512×512 array at
batch 1 the real figure is 1/513 of that — a factor of **513**. Consequences:

| D8 claim | with the tail effect |
|---|---|
| per-op layer: compute 0.9 µs, "≈3080× memory-bound" | 0.72 ms, **3.8× memory-bound** |
| chip_b 4B decode 11.7 tok/s, DRAM-bound | **9.3 tok/s, COMPUTE_BOUND** |
| chip_b 1B decode 128 tok/s | **48 tok/s, COMPUTE_BOUND** |
| chip_b prefill @S=512 85 ms, DRAM-bound | **144 ms, COMPUTE_BOUND** |

**The head-to-head conclusion changes.** D8 has chip_b — quarter the compute, 18× the SRAM —
winning decode by 33%. Once the array's batch-1 utilisation is modelled, chip_b's quarter of the
TOPS becomes its binding term and the margin collapses to ~11% (9.3 against 8.4). At 1B it wins by
30%, not 3.5×. chip_a is unaffected throughout: with 4× the compute its array never binds, which is
why every chip_a figure still reproduces D8 within 5%.

Note also that even at prefill the effect is not small: `M/(M+512)` at a 512-token prompt is 0.5,
so a 512-deep array is *half idle* on a prompt exactly as long as itself. Amortising it needs
S ≫ 512.

## D15 — On-chip capacity is allocated to activations before weights (2026-08-08)

D5a says SRAM contributes capacity. It does not say what the capacity is *spent on*, and the answer
turns out to matter more than the residency formula.

Charging every activation to DRAM gave chip_a a prefill TTFT of 178 ms against D8's 113 ms — a 58%
error — because a 15 MB activation working set was being streamed from DRAM by a chip with 55 MB of
SRAM. Allocation order adopted (build decision), which is what an NPU compiler does:

1. **The double buffer** (two tiles). Filling SRAM with resident weights leaves no staging room and
   forfeits the `max(load, compute)` overlap entirely — a far worse trade than giving up two tiles
   of residency. chip_b at 1 GB is the case that surfaces it.
2. **The activation working set.** On chip_a at prefill, 15 MB of activations held on chip removes
   ~2 GB of DRAM traffic (58 ms); the same 15 MB spent on weight residency removes 15 MB (1.6 ms).
3. **Weights**, with whatever remains. This is what `resident_fraction` reports.

With this order the D8 residency table is reproduced essentially exactly — 1.40 / 2.73 / 5.71% on
chip_a and 25.76 / 50.15 / 100% on chip_b against the stated 1.4 / 2.7 / 5.5 and 25.6 / 50 / 100 —
and chip_a's TTFT lands at 114.6 ms against 113.

## D16 — Size presets are realised by scaling the layer count (2026-08-08)

D8 asks for 1B/2B/4B presets driven by `params`, "keep the other dims or scale them — document the
choice". At M1 `declared_params` set only the headline figure; the graph still built the full 4B, so
all three presets produced identical predictions. Fixed at M3.

Choice taken: **scale the layer count**, the dimension that genuinely varies within a model family.
Widths and vocabulary stay at the family's values. The realisation is never exact — layers are
integers and the embedding table does not scale at all:

| preset | layers | built | vs declared |
|---|---|---|---|
| 2B | 14 | 1.993 G | −0.4% |
| 1B | 3 | 0.954 G | −4.6% |

The 1B preset is **70% embedding table** (Gemma-3-4B's tied 262208×2560 is 671 M) and has room for
three blocks. It reproduces D8's residency and decode figures, but as a model it is degenerate, and
a real `gemma3_1b` profile (26 layers, hidden 1152, 1.0 B) would be the better 1 B point if that
sweep is ever used for anything beyond checking D8. The residual is reported in
`report.assumptions` rather than hidden.

## D9 — `gemma4` profile replaced by `gemma3_4b` (2026-08-07)

Commit 5218046 named `gemma4.yaml` in the Phase 1 plan (no profile was written — `profiles/models/`
is still empty). No public `config.json` exists under that name, and CLAUDE.md forbids inventing
hyperparameters. Decision (user, 2026-08-07): ship Gemma-3-4B, which has a published config and is
the basis of the D8 worked example.

**Resolved at M1 (weights) and M2 (KV).** The supplied figures are an idealised Gemma-3-4B
(Q projection at `d_model×d_model`, `d_kv=512`, vocab 256000); the real config has 8 query heads
× 256 head-dim, `kv_width = 4×256 = 1024`, and vocab 262208. Only one profile ships — the faithful
one — because:

- **Attention weights per layer are identical.** `2·2560·(2560+512)` equals `2·2560·(2048+1024)`,
  so both give 15.73 MB, and with the 78.64 MB GEGLU FFN both give the 94.4 MB/layer that D8's
  per-op golden asserts. Verified at M1: 94.372 MB, and `ops = 2 × weight_bytes` holds to 4
  decimal places over the layer's matmuls.
- **The KV cache is not identical, and the coincidence does not extend to it.** KV depends on
  `d_kv` alone, not on the sum, so the real `kv_width = 1024` gives exactly **twice** the supplied
  figure: `34 × 2 × 1024 × 4096 × 1 = 285.2 MB` at C=4k, against D8's 142.6 MB.

The D8 decode-with-KV golden moves accordingly, and the corrected figures are what M3 will assert:

| quantity | as supplied | with the real config |
|---|---|---|
| KV(4096), INT8 | 142.6 MB | **285.2 MB** |
| `t_kv` @ 34 GB/s | 4.19 ms | **8.39 ms** |
| chip_a 4B decode with KV@4k | 117.3 ms, 8.53 tok/s | **121.5 ms, 8.23 tok/s** |

Every KV-free D8 golden is untouched — the weight-traffic figures dominate at 113 ms and the KV
term only ever adds to them.

---

## D17 — A bare GEMM is a model family, not a `custom` op (2026-08-15)

The engine could describe an 8 B transformer but not a single matrix multiply. `family: custom`
came close — it takes an op list with hand-written FLOPs and bytes — but `CustomOp` carries **no
shape**, and `analysis.tiling.operation_utilisation` dispatches on the attribute type. A GEMM
expressed as a custom op therefore falls through to `MatmulAttrs`-less `CustomAttrs` and returns a
utilisation of **1.0**: the systolic tail effect silently disappears.

That effect is the single most interesting thing about a small GEMM. `M=1` on A100's 16×16 tensor
core is `1/17 = 5.88%` of peak; on chip_a's 512×512 array it is `1/513 = 0.19%`. A tool that
reported 100% there would be wrong in exactly the way CLAUDE.md's sanity checks exist to catch.

Decision (user, 2026-08-15): add `ModelFamily.GEMM` with `m`, `n`, `k`, and a 25-line builder
emitting one `MATMUL` operation with real `MatmulAttrs`. Nothing in `operators/`, `analysis/` or
`report.py` changed — `MatmulCost` and the roofline picked it up unmodified, which is the useful
test of whether the layering is honest.

Two consequences worth stating:

- **`M` folds batch in**, exactly as `MatmulAttrs` documents. There is no `--batch` on `bwz gemm`,
  because at the level of one operation nothing distinguishes a batch of 128 rows from 128 rows.
- **`hypothetical` defaults to `true`** for this family. A synthetic shape has no `source_url` to
  cite, and the provenance validator would otherwise reject every invocation.

Golden, and the figure the roofline plot is drawn from: 10000³ fp16 on A100 under `--ideal` is
`2.000e12` OP against `6.000e8` bytes, intensity 3333 against a ridge of 153, **6.42 ms**,
`COMPUTE_BOUND` by 24×.

> **Superseded in part by D18**, one day later: the family, the command, the module and the test
> file were all renamed from `gemm` to `matmul`, and the transformer precision vocabulary this
> entry inherited was replaced by per-operand widths. The reasoning above — why it is a family and
> not a `custom` op — stands unchanged.

---

## D18 — A matmul speaks matmul, not transformer (2026-08-16)

D17 shipped the family with `--weights` and `--activations` on the CLI, inherited from
`DeploymentSpec.precision`. That vocabulary is wrong here and the user said so: in a network one
operand is a parameter that lives on the chip and the other is data flowing through it, but a bare
matmul has no such asymmetry. It has operand `A`, operand `B`, and a result `C`.

Decision (user, 2026-08-16): rename the family to `matmul`, give it three widths of its own, and
have the builder ignore `deployment.precision` entirely.

```
bwz matmul -M 4096 -N 4096 -K 4096 --chip a100_80gb --dtype int8              # all int8
bwz matmul -M 4096 -N 4096 -K 4096 --chip a100_80gb --dtype int8 --out int32  # int32 accumulator
bwz matmul -M 4096 -N 4096 -K 4096 --chip a100_80gb --dtype fp16 --out fp32   # all float
bwz matmul -M 4096 -N 4096 -K 4096 --chip a100_80gb --a fp16 --b int8         # mixed operands
```

Three consequences, each a modelling statement rather than a naming one:

**1. The result width is an accumulator width, and changes bytes only.** `int8 x int8 -> int32`
performs exactly the same `2·M·N·K` integer operations as `int8 x int8 -> int8`; it writes four
bytes per result instead of one. At M=N=K=4096 that is 67.1 MB of result against 16.8 MB — the
compulsory intensity halves, from 2731 to 1365 OP/byte. `DType.INT32` was added for this and is a
storage width only: no chip declares a compute unit for it, which is correct, because an int8
product accumulates in int32 *at the int8 rate*.

**2. Mixed-width operands run at the wider one.** Both operands enter the array through the same
datapath, so `int8 × fp16` is computed by widening the int8 side — there is no int8 rate available
for it. The narrow operand still saves its bytes; it buys no throughput.

This **reverses** what the engine did before. `compute_dtype()` returns the weight dtype whenever
the chip supports it, on the reasoning that "quantised inference runs the GEMM at the weight dtype
— that is the entire point of quantising". That is true for W8A8 and **false for W8A16**, where
real kernels dequantise and run the fp16 tensor cores; the speedup of weight-only quantisation is
entirely in the bytes, which is why it helps a memory-bound decode and does nothing for a
compute-bound prefill. `MatmulSpec` now uses the wider-operand rule. The transformer path still
uses `compute_dtype()` and is therefore still wrong for W8A16 — **left open deliberately**, since
changing it moves every mixed-precision transformer number in the repo and deserves its own
decision.

**3. The deployment is unused for this family.** `analyze()` still takes one, because its signature
is fixed, but `build_matmul_graph` discards it. Batch is folded into `M`; there is no context, no
phase, no attention implementation.

Renames: `ModelFamily.GEMM` → `MATMUL`, `GemmSpec` → `MatmulSpec`, `graph/gemm.py` →
`graph/matmul.py`, `bwz gemm` → `bwz matmul`, `--gemm` → `--matmul` on the plot script. One commit
old and no profile YAML used it, so no compatibility shim.

---

## D19 — The pipeline trace decomposes the roofline; it does not re-model it (2026-08-16)

Asked for a Konata-style view of what happens on DRAM, SRAM and the cores over the run, one per
chip, tailored to the matmul.

The design constraint that decided everything: **a picture that disagrees with the report is worse
than no picture.** So `analysis/pipeline.py` emits spans that are slices of quantities
`op_roofline` already produced — DRAM busy sums to `t_dram`, core busy to `t_compute + t_fixed` —
and the tests assert exactly that. The tile count is the same `ceil(K/rows)·ceil(N/cols)` that
`systolic_utilisation` divides by, so the picture and the utilisation figure cannot tell different
stories.

Three things fell out of drawing it, each a modelling statement rather than a rendering one:

**1. The roofline's `max()` omits pipeline fill/drain.** A double-buffered schedule costs
`max(t_dram, t_compute) + min(t_dram, t_compute)/tiles` — one extra step of whichever resource is
*not* binding. `max()` alone is the `tiles → ∞` limit. Reported as `fill_drain_s`, not folded into
the latency: 0.1% on a 65 536-tile matmul, not negligible on a decode projection of a dozen tiles.
Fixing the latency itself would move every number in the repo and belongs with calibration.

**2. Double buffering means depth two, and the schedule has to say so.** The first draft let a
tile be fetched as soon as DRAM was free, which put all 64 tiles on chip at once — SRAM occupancy
came out at 3237% of the span. The missing constraint is the buffer being freed:
`load_start(i) >= compute_end(i - depth)`. With it, occupancy is two tiles throughout, which is
what "double buffered" means and what capacity planning granted.

> **Follow-up (2026-08-17):** it was reported as "196%", and the user rightly asked how SRAM can be
> 198% occupied. It cannot — that figure was a *depth* printed with a duty cycle's unit. DRAM and
> the array are single serial resources whose spans never overlap, so `busy/span` is a fraction of
> time for them; SRAM is *n* buffers, so the same ratio counts buffers. `PipelineTrace.concurrency`
> now returns `(mean, peak)` per lane and the CLI and figure label each row in its own unit —
> `100% of span` for the array, `1.97 of 2 buf` for SRAM. No number changed; the schedule was
> right and its presentation was not.

**3. Coalescing must not re-schedule.** 390 625 tiles cannot be drawn, and 451 graph nodes should
not be. Grouping them, a group's duration is the **sum of its members' latencies**, not the latency
of their summed terms — `max(ΣL, ΣC) ≤ Σ max(L, C)`, so the natural-looking version drew a network
1.0% faster than the report it illustrated. Caught by the test asserting the operation trace
reproduces the reported latency exactly.

Output is a Kanata log (`--kanata`, opens in Konata) and a two-register PNG per chip
(`scripts/plot_pipeline.py`): the whole run at total scale, and the first few steps zoomed, since
at total scale one step of a 65 536-tile matmul is a hairline. The x axis is normalised to the
total time in both, which is what makes two chips comparable when their absolute times differ by
orders of magnitude. Kanata has no cycle here — a tick is `total/resolution`, stated in the file
header.


---

## D20 — The pipeline figure is resource-centric, not instruction-centric (2026-08-17)

D19 shipped the trace with a Kanata log as its headline output. Konata draws one row per
instruction with stages inside it, which is right for a CPU pipeline and wrong here: it answers
"what happened to this tile" when the question a comparison asks is "what was the memory system
doing while the array worked". User verdict (2026-08-17): *"it doesn't give us the insights … you
must explicitly dedicate a DRAM row for the copy … with the BW and number of BYTES being copied as
an info box"*. Correct.

Decision: `scripts/plot_pipeline.py` is now the primary output and its **rows are hardware
resources read off the chip profile** — every memory level and every compute unit — with the
quantity written beside each: bytes and achieved bandwidth on the DRAM row, bytes and buffer count
on the on-chip row, operations and achieved rate on the compute row, plus three headline boxes
above. `Span` gained `bytes_moved`, `flops` and `resident_bytes` so the figure reads the report's
own quantities rather than recomputing them; a test asserts they match.

**Resources the model does not use are drawn grey rather than omitted.** A100 declares L1, L2, HBM,
432 tensor cores and 6912 CUDA cores; v1 spends HBM's bandwidth, L1's capacity and the tensor
cores, and nothing else. Two of five rows are therefore idle with the reason on them — "declared,
not modelled — the roofline is flat (D5)" and "idle — peak is the max over units, not the sum".
Putting the model's boundary on the page beats hiding it, and against `chip_a` — three rows, DRAM
reading `0 B — nothing crossed` because 55 MB of SRAM holds all of B — the contrast is the
comparison the figure exists for.

The Kanata log stays, demoted to `--kanata`: Konata remains a good viewer for step-level detail,
and `make konata` still opens it. It is no longer what the figure is *for*.

Layout note, since it cost two attempts: row counts vary by chip (five for A100, three for
`chip_a`), so every vertical position is now derived in **inches** from the row count and converted
once. Guessing figure fractions put the second register on top of the footer.

---

## D21 — Konata removed; the timeline is our own zoomable page (2026-08-17)

D20 kept the Kanata log as a secondary output. Two things then became clear from use.

**First, it could not answer the question.** Opening `ciao.kan` in Konata shows sixty-four rows
labelled `6104 B tiles 16x16 [n/64]` and no DRAM row — because the Kanata format has no notion of
one. Rows are instructions; the lanes inside a row are stage tracks, not hardware. "Where is my
DRAM row" has no answer in that format, and dedicating one is not a matter of emitting different
records.

**Second, the reason to want Konata was interactivity**, not its layout: *"the plots you produce
are not interactive … nor can they be zoomed"*. Correct, and the PNG cannot become so.

Decision (user, 2026-08-17): delete Konata and the Kanata emitter entirely — `bwz/kanata.py`,
`scripts/konata.sh`, the `--kanata` flags, the `make konata` target, the committed `.kanata` files
and every mention outside this log — and put the interactivity in an artifact we own.
`scripts/timeline_html.py` renders the trace as one self-contained HTML page: same
resource rows as the PNG, with wheel-zoom about the cursor, drag-pan, double-click reset, and a
hover tooltip carrying each span's bytes, rate and duration. No server, no port, no download, no
CDN — `file://` is enough. It ships at 256 steps against the PNG's 32, because a static figure has
to stay legible at one scale and a zoomable one does not.

Renamed at the same time, since it was the direct cause of "I still see the old one": two different
figures were both called `pipeline-*`. Now `machine-<chip>.png` is the block diagram and
`timeline-<chip>-<dtype>.{png,html}` is the schedule.

What was actually lost with Konata: nothing this project needs. What was gained by fetching it: a
pinned-release script, a port allocator and a cache validator, all now deleted. The lesson worth
keeping is that reaching for a third-party viewer imported its data model along with its features,
and the data model was the part that did not fit.

---

## D22 — DRAM traffic is split by direction, and the result is always written back (2026-08-17)

Asked where the write-back transactions were in the timeline. They were nowhere: every DRAM span
was a `Ld`, drawn *before* the arithmetic, and `C`'s bytes were folded into those load bars. Three
faults behind one question.

**1. The result was discounted by activation residency.** `op_roofline` lumped inputs, outputs and
scratch into one "activation" pool and applied `activation_resident_fraction` to all of it. For a
transformer that is right — an intermediate activation really can stay on chip for the next
operator to read. For a standalone matmul it is not: `C` is the answer, and nothing on chip
consumes it. The 10000³ fp16 case on A100 kept 30.4 MB of the result on a chip nobody reads it
from, understating traffic by 5.6%.

Now `schedule.py` computes, per operation, the share of its outputs that **no later operation
reads** — the only place the shape of the graph enters the traffic model — and `op_roofline`
charges that share in full:

```
read  = (1-r)·weight + (1-r_a)·(input + scratch)
write = terminal_output + (1-r_a)·consumed_output
```

The 10000³ goes from 539 MB to 570 MB (370 read + 200 written). The sharpest case is chip_a's
4096³ INT8: it fetches **nothing**, because 55 MB of SRAM holds all of B, and still writes 16.8 MB.
Its DRAM row read `0 B — nothing crossed` before, which was simply false.

**2. The schedule had no store stage.** `Stage.STORE` now follows the arithmetic that produced it,
on the same DRAM lane, and `OpResult`/`PhaseResult` carry `dram_read_bytes` and `dram_write_bytes`.
The figures draw loads filled and stores hollow, so the direction is readable without a legend.

**3. Ordering on the one port matters, and got it wrong first.** Making loads queue behind stores
dropped SRAM occupancy from two buffers to one — the write-back of tile *i* delayed the fetch of
tile *i+1*, cancelling the double buffer the capacity planner had granted. The fix is the policy a
memory controller actually uses: **the next tile's fetch outranks this tile's write-back**, with
the store draining afterwards. A buffer is also held until its result is written, not merely until
the arithmetic ends, so the reuse test now keys on `store_end(i-depth)`.

`fill_drain_s` is now measured from the schedule rather than derived from a formula, since with
stores in it the head-and-tail cost is no longer one clean term.

Not fixed, and still open: the capacity check counts only the **B tile** (`rows × cols × bytes` —
512 B for A100 fp16). The A tile streaming through it and the C tile accumulating out of it are not
counted, so "do the three tiles fit together" is not actually verified. At M=10000 the real working
set is ~640 KB against the 512 B reserved. Nothing moves on the shipped profiles, but on a small
buffer `double_buffered` could come back true when three tiles do not fit.

---

## D23 — The timeline names its transactions, its operators, and carries the roofline (2026-08-17)

Three additions, all from the same complaint: a bar with no label is a shape, not a measurement.

**Direction is named, not implied.** The DRAM box and row read `LOAD 15 GB · STORE 322 kB`, and a
hover gives `LOAD — operands in` or `STORE — result written back`. Filled bars are loads, hollow
ones stores. D22 made the two quantities exist; this makes them readable.

**The compute box names the operators.** `COMPUTED — matmul 98% · attention 2%`. "15.3 GOP" does
not say whether that was one matmul or a decode step's worth of matmul, attention, norms and
elementwise, and for a comparison the mixture is the point. `PipelineTrace.work_by_op` is built
from the operation results rather than the drawn spans — coalescing 451 nodes into 57 blocks and
taking each block's dominant family reported *only* matmul, which is exactly the information the
box exists to avoid losing.

**`--model` draws a network**, one figure per phase, which is what gives the breakdown something to
break down. Llama-3-8B decode on A100: DRAM 92% busy against compute at 21%, `matmul 98% ·
attention 2%`, `LOAD 15 GB · STORE 322 kB` — memory-bound, visibly, in one picture.

**The HTML carries the roofline too.** Same page, below the timeline: both ceilings, the ridge
point, the M=1 tail line, and the run as a labelled point with its intensity and achieved rate on
hover. Two views of one run in one file — where the time went, and why it had to.

---

## D24 — An encoder is not a decoder with the same code path (2026-08-17)

`ModelFamily.TRANSFORMER_ENCODER` existed in the spec from M1 and was never exercised. Building
one for teaching showed the graph builder treated it as a decoder in three ways, each of which
changes the count:

**Attention was causal.** `causal=True` was hardcoded, so an encoder's scores were halved by a
triangle it does not have. At S=4 with 2 heads that is 20 scored positions instead of 32 — a 37.5%
understatement, approaching 50% at long sequences. Now `causal = family is TRANSFORMER_DECODER`.

**It was given a decode phase.** `phases_for` returned prefill *and* decode for any
`TransformerSpec`. An encoder runs one bidirectional pass over the whole sequence; there is no
token-by-token phase to separate. Now encoders return `(PREFILL,)`.

**It was given an LM head.** An encoder emits hidden states; what sits on top — a classifier, an
MLM head, a pooler — is task-specific. Counting one here would be inventing a layer, so the head is
built for decoders only.

Ships with `profiles/models/single_layer_encoder_toy.yaml`: one layer, hidden 8, 2 heads of 4, FFN 16,
vocab 16, tied embeddings, ReLU FFN. Small enough that every figure is a product of two small
integers — **664 parameters** and **5280 operations** over 4 tokens — with the full derivation in
the profile's own header and in `tests/unit/test_single_layer_encoder_toy.py`, which recomputes both from the
dimensions rather than asserting what the engine happened to produce.

**It was given a KV cache.** K and V were tagged `KV_CACHE`, so the memory plan reported a
footprint — 128 B on the tiny profile — that nothing would ever read again, and the planner tracks
cache separately from activations, which also skewed the residency waterfall. A cache exists to be
reused by a later step and an encoder has no later step, so K and V are now plain activations
consumed within the pass.

A deployment may still ask an encoder for `output_tokens`; the request is ignored, and the
assumptions drawer says so by name rather than dropping it silently.

`rank_operations` (the `--show-ops` lines) now carries each operation's arithmetic and DRAM traffic
alongside its time. On a model this small they tell different stories: seven operations cost 3 µs
each and are all `LATENCY_BOUND`, while one of them does 1.02 kOP and another does none at all.


---

## D26 — The page carries the arithmetic it claims (2026-08-17)

A performance model nobody has calibrated is only worth what its arithmetic can be checked against.
The HTML timeline now ends with every operation written out four ways: operand shapes, the algebra,
the flop count as an expression, and a pseudo-C loop nest carrying the real extents.

```
layer0.q_proj   A[4,8] x B[8,8] -> C[4,8]                      512 OP
                C[m,n] = Σ_k A[m,k] · B[k,n]
                2·M·N·K = 2·4·8·8 = 512
                for (m = 0; m < 4; ++m) …
```

`bwz/explain.py` is pure and holds no counts of its own: the numbers come from
`operators.base.cost_of`, and `check()` asserts every printed expression evaluates to the cost the
engine actually used. A test runs it over every operation of every shipped profile in both phases,
so the prose cannot drift from the model — which is the failure mode that would make the whole
feature worse than useless.

The pseudo-C is meant to compile if pasted, so it carries no nested `/* */` (C forbids them) and a
test enforces that too. Indices and extents are real; types and memory layout are not.

---

## D27 — A tensor core does matrix-multiply-accumulate and nothing else (2026-08-17)

Asked whether a GELU can run on the tensor core. It cannot, and the engine was charging it as if it
could.

`machine_model` picked one unit — the highest-throughput one supporting the dtype — and
`op_roofline` costed **every** operation against it. On A100 that meant norms, GELU, softmax tails
and residuals were charged at the tensor cores' 312 TOP/s rather than the CUDA cores' 19.5 TOP/s:
**16x too fast**. The error compounded, because `operation_utilisation` returns 1.0 for non-GEMM
operations, so a norm received 100% of a peak it can never reach.

Now `MachineModel` carries two engines. `MATRIX_OP_TYPES` — matmul, attention, conv — run on the
array; everything else runs on the fastest **non-systolic** unit the profile declares.
Llama-3-8B's `attn_norm` goes from 38.4 ns to 615 ns, which is the honest figure.

A profile declaring only a systolic array — `chip_a` — has nowhere to put elementwise work, so it
is still charged at the array's rate and the assumptions drawer says exactly that. Adding a vector
unit to those profiles needs a published figure nobody has yet.

**Decided, not left open (user, 2026-08-17): the per-element constants stay as they are** — GELU 8,
softmax 5, RMSNorm 4. They are *algebraic* operation counts, not machine instructions: hardware
without a special-function unit evaluates GELU's erf by polynomial approximation over many more ALU
ops, and hardware with one typically runs it at a fraction of the ALU rate. No shipped profile
declares either number, and rather than invent one the model keeps the simple count and says what
it costs.

The bound on that approximation: all non-matrix work is 1.4% of Llama-3-8B prefill, so even a 4x
transcendental penalty would move the phase by 4%. That is why the simplification is affordable
here. It would not be on a model whose arithmetic is mostly activations, or on an NPU with no
vector unit — and the assumptions drawer says so on every report, so the reader can tell which case
they are in. Closing it properly would need a published SFU throughput per profile
(`transcendental_flops_ratio` on `ComputeUnit`), fitted in Session 5 against a measurement.

---

## D28 — Each engine gets its own lane, and the handoff between them is not modelled (2026-08-17)

D27 split the *costing* between the matrix and vector engines. The figure still showed one busy
row and drew `cuda_core` grey, so the correction was invisible where it mattered most.

`Lane.VECTOR` now exists alongside `Lane.CORE`, and a step emits one span per engine that did work
in it. Llama-3-8B prefill on A100:

| row | | |
|---|---|---|
| `tensor_core` 432 x 16x16 | matrix work — peak 312 TOP/s | 7.22 TOP @ 207 TOP/s · matmul 99% · attention 1% |
| `cuda_core` 6912 x 1 MAC/cycle | norms, activations — peak 19.5 TOP/s | 2.11 GOP @ 13.6 TOP/s · elementwise 74% · norm 26% |

Splitting by *dominant family per coalesced group* was the first attempt and produced an empty
vector lane: 451 operations collapse into 31 blocks, and matmul dominates every one. The group's
compute time is now split by engine instead, which is the only version that survives coalescing.

**How the two exchange data, and what this model says about it.** On real silicon they are inside
the same SM: tensor cores read operands from the register file (or shared memory) and write results
back to registers, and the CUDA cores use the same register file and the same shared memory/L1.
A GEMM's result reaches the activation that follows it **in registers — never through DRAM**, which
is exactly why fused epilogues are the default in every serving kernel.

This model has no register file, no shared memory and no fusion. The two lanes are drawn adjacent
and exchange nothing: each operation is charged its own compulsory traffic, with residency applied,
and each costs its own dispatch. So an unfused chain is charged roughly right, and a *fused* one —
GEMM + bias + GELU as a single kernel — is overcharged here by a dispatch and by whatever traffic
residency failed to absorb. Stated in every report's assumptions rather than left for the reader to
discover. Modelling it properly means a register/shared-memory level and a fusion pass, which is
M8 work (D5).

---

## D29 — Two chips in one figure: the axis is shared, the rows are not (2026-08-17)

`bwz compare` prints a head-to-head table. Asked for it visually — "the same workload on each,
resource rows grouped per chip, so the difference is readable at a glance" — with the time axis
**shared and absolute**, because seeing that one span is 13x the other is the whole reason to draw
it. `scripts/plot_pipeline.py --chip A --chip B --compare`.

Four decisions the comparison forces, none of which arise for a single chip.

**1. The rows cannot line up, so they are banded rather than aligned.** Row counts are read off the
profile and profiles disagree:

| chip | memory levels | compute units | rows |
|---|---|---|---|
| `chip_a` | 2 | 1 | 3 |
| `a100_80gb` | 3 | 2 | 5 |
| `metis_aipu` | 4 | 2 | 6 |

There is no honest correspondence to draw between A100's `L2` and Metis's `L2`, still less between
`cuda_core` and `dpu`. Forcing one — a canonical DRAM/SRAM/COMPUTE triple, say — would throw away
exactly what a comparison is for, which is that the machines are shaped differently. So each chip
keeps **its own band**, introduced by a header row naming it and carrying its peak, its DRAM
bandwidth, its on-chip capacity, its row counts and its total span. The shared thing is the **time
axis**; the rows are deliberately not shared. Adding a third chip adds a third band and nothing else
has to change.

**2. Both chips must run the same workload at the same precision, and that has to be enforced.**
Per-chip dtype defaults would silently compare different amounts of traffic: A100 defaults to fp16,
Metis has no fp16 datapath at all. `--compare` therefore resolves **one** dtype that every chip
supports (preferring fp16, then int8) and refuses with the intersection named when there is none:

```
bwz: metis_aipu has no fp16 datapath, so --compare cannot run the same workload on
every chip. Supported by all: int8
```

The single-chip path keeps its per-chip default, which is right there — nothing is being compared.

**3. The normalised view is kept, not replaced.** Shared-and-absolute is the right axis for "which
machine is faster and by how much" and the *wrong* one for "how is this machine's time distributed":
at 13x, the faster chip's whole run is 7% of the axis and its internal structure is a smudge. Both
questions are real, so `--compare` is a flag and the default remains one normalised figure per chip.

**4. The roofline gets a register of its own, and colour changes meaning inside it.** The timeline
answers *what happened*; two chips landing on opposite sides of their own ridge point is *why*, and
that is invisible in a table of latencies. On the shipped comparison — Gemma-3-4B prefill, S=512,
int8 — both chips do 3.33 TOP over ~3.8 GB, i.e. ~870 OP/byte, and:

| | ridge | this run | verdict |
|---|---|---|---|
| `a100_80gb` | 306 OP/byte | 870.4 | **compute bound** |
| `metis_aipu` | 6145 OP/byte | 868.9 | **DRAM-bandwidth bound** |

Identical arithmetic, identical traffic, opposite limiters. Inside that panel colour encodes
**chip**, where everywhere else in the figure it encodes **resource** — so the panel gets its own
two hues (slots 4 and 5) rather than reusing the lane blue/grey/orange/green, its own stated legend,
and a direct label on every mark.

**What is unchanged, and tested to be.** The comparison composes per-chip traces; it does not build
a new one. Each panel's spans are the same `build_trace` output the single-chip figure draws, so the
D19 invariant survives untouched — measured on the shipped figure, DRAM busy reproduces `t_dram` to
7e-18 s on A100 and 1e-16 s on Metis, the compute rows reproduce `t_compute + t_fixed` to the same
tolerance, and each band's span equals its chip's reported latency exactly.

### Four pre-existing bugs this surfaced, all on the vector lane

D28 added `Lane.VECTOR` and taught the PNG's bar drawing and lane totals about it. **Four other
places that switch on lane were never updated**, and every one of them fails in the same direction:
the vector engine renders as something it is not. They are grouped here because the lesson is one
lesson — adding an enum member is not the same as handling it — and because a matmul-only figure
set hides all four, which is why a year of committed figures never showed them.

Each was confirmed against a *rendered* artefact from the unmodified script (a screenshot, or the
JSON payload of the emitted page), never from reading the source.

**1. No colour.** The page's JS colour map still had three entries. A Llama-3-8B prefill page
emitted 130 vector spans, every one rendering `fill="undefined"`; browsers fall back to black, so
the vector row read as a solid dark bar rather than the green D28 assigned it.

**2. No denominator on the rate.** The `COMPUTED` box read
`array 3.32 TOP @ 381 TOP/s of 624 TOP/s` and then `vector 2.22 GOP @ 437 TOP/s` — a rate with
nothing to measure it against, on every chip. It now carries the vector unit's peak, and when the
profile declares no vector unit for that dtype it says so rather than printing the array's peak
twice as though it were a second engine. That makes D27's open problem legible on the face of the
figure: **A100 at int8 has no vector datapath at all** (`cuda_core` is fp32/fp16 only), so its norms
and activations are charged at the tensor cores' 437 TOP/s and the box now says so. Metis, which
declares a DPU, reads `vector 2.22 GOP @ 410 GOP/s of 410 GOP/s`.

**3. The hover called it a buffer.** `_tip()` tested `Lane.DRAM`, then `Lane.CORE`, then fell
through to the SRAM branch — so hovering any vector bar gave
`HOLD — elementwise on chip … holding 0 B`: the arithmetic lane described as an occupancy, with its
operations and its rate, the two numbers a reader most wants there, absent entirely. Now
`EXEC — norm on the vector unit … 33.6 MOP @ 410 GOP/s`.

**4. The vector spans were named after matrix operators.** This one is in the engine, not the
script. `_operation_trace` splits a coalesced group's compute time between the two engines (D28) but
named *both* halves after the group's dominant family — and a transformer block's dominant family by
FLOPs is a matmul in essentially every group. So the vector lane's spans carried
`op_type="matmul"`, and once bug 3 was fixed the hover read **`EXEC — matmul on the vector unit`**,
a claim that directly contradicts the correction D27 exists to make. 136 of 274 vector spans on the
shipped comparison said it.

`_engine_work(group, matrix=...)` now picks each half's dominant family from the operations *that
engine actually ran*. It is a relabelling and not a re-costing, and `test_pipeline.py`
asserts both halves of that: no vector span names a family in `MATRIX_OP_TYPES`, no array span names
one outside it, and the two lanes still sum to `t_compute + t_fixed` and to the phase's arithmetic.

**Open, not fixed: `HELD ON CHIP` can exceed the chip's capacity.** The same figure reports
`671 MB` held `of 60.7 MB capacity` on A100 — because `_operation_trace` sets a step's
`resident_bytes` to `max(weight_bytes)` over the operations it coalesces, and Gemma-3-4B's LM head
alone is 671 MB of int8 weights. That is the operation's whole weight footprint, not what the
capacity planner granted, so the box is measuring one thing and labelling it another. It reproduces
on the unmodified script for any `--model` whose largest operator exceeds SRAM, and fixing it is a
semantics change in `analysis/pipeline.py` (either charge the tiled working set or rename the
quantity), which deserves its own decision rather than being folded into a figure change.

---

## D30 — A chip is *n* arrays, not one big one, and an IMC array owns its weights (2026-08-17)

Two related errors, both surfaced by reading a `--compare` figure and asking why a 100-cubed matmul
looked the way it did. Both come from one shortcut: `peak_flops_per_s` multiplies **one** array's
throughput by `count`, and everything downstream then reasons as if the chip were a single array of
`count x rows x cols` PEs. It is not.

### 1. Work reaches the arrays in waves, and the last wave is partly empty

`systolic_utilisation` modelled one array's shape efficiency exactly and then let the aggregate peak
assume every array always had a tile to work on. For a 4096-cubed GEMM that is true. For a small one
it is badly false:

```
waves     = ceil(tiles / units)
occupancy = tiles / (waves * units)
```

Metis has **four** AI cores. A 100x100x100 INT8 matmul is `ceil(100/512)^2` = **one** 512x512 weight
tile, so one core takes it and three idle — the chip's real ceiling on that shape is a quarter of
the 209.7 TOPS the datasheet quotes, on top of the 0.62% shape utilisation it already paid.

The correction is large where it should be and vanishes where it should:

| shape | a100 tiles / units | a100 util | metis tiles / units | metis util |
|---|---|---|---|---|
| 100^3 | 49 / 432 | 68.72% -> **7.80%** | 1 / 4 | 0.623% -> **0.156%** |
| 600^3 | 1444 / 432 | 94.86% -> **79.27%** | 4 / 4 | 18.52% unchanged |
| 4096^3 | 65536 / 432 | 99.61% -> **99.42%** | 64 / 4 | 88.89% unchanged |

It also **changes the comparison it was found in**. A100 against Metis on a 100-cubed matmul went
from 312x to **74x**, because A100 — 432 tensor cores that 49 tiles cannot fill — loses more to wave
quantisation than Metis does, and A100's verdict on that shape flips `DRAM_BW_BOUND` ->
`COMPUTE_BOUND`. At 4096-cubed the ratio is 1.7x, finally in the neighbourhood of the 3.0x
peak-TOPS ratio rather than two orders away from it. That is the sense in which the model could not
compare real performance before.

Attention passes `independent = batch x heads`, because every head is a genuinely separate GEMM and
they fill the arrays alongside each other. Counting one head's tiles would have reported a 64-head
attention as leaving a 4-core NPU idle.

Two goldens moved, both A100 at 10000-cubed and both by 0.086%: `390 625 / (905 x 432) = 0.99914`.
The CLAUDE.md sanity check "M=1 on a 128x128 array is about 1/128" is now asserted where it belongs,
on `systolic_utilisation` itself, with the chip-level figure asserted as that times occupancy. Both
numbers stay on the page, because they are different claims.

### 2. A D-IMC array can only compute on weights that are inside it

`ComputeUnit.weight_sets` (Metis: 4 — Table C's "512 x 512 x 4 weight sets", which is exactly the
1 MiB of IMC per core) and `resident_tile_capacity() = count x weight_sets` = **16 tiles held, 4
computing**. A hard constraint, not a cache hint: a digital in-memory-compute weight cannot take
part in a MAC until it has been written into a bank.

**NVIDIA is not the same, and the datasheet says so.** The A100 whitepaper puts tensor-core operands
in the register file, fed from shared memory — 192 KB combined L1/SMEM per SM, SMEM configurable to
164 KB, RF 256 KB/SM — and `cp.async` exists to stage global into SMEM "eliminating the need for
intermediate register file (RF) usage". There is no persistent weight store in the array at all,
which is exactly why a GPU can stream 16 GB of weights per token and Metis cannot. So `weight_sets`
stays 1 on every GPU profile and the term is inert there by physics, not by omission.

**Modelled: the capacity. Not modelled: the write time.** A 4096-cubed INT8 matmul on Metis is 64
tiles against 16 resident, so 48 of them displace an earlier tile. Charging the writes needs a
bandwidth on the L1-to-IMC path, and v1 has no on-chip bandwidth term by explicit decision
(D5a, D5b). The paper publishes the *activation* feed (512 bits/cycle/core from L1) but not the
weight-write port. So the capacity is disclosed in `report.assumptions` and the time is not charged.
Trigger to revisit: a published weight-write bandwidth, at which point D5b reopens with a sourced
number instead of an estimate.

> **Correction (2026-08-17), from a user's question that I answered wrongly first.** This paragraph
> originally said those 48 tiles "must be **re**-written … mid-operation", and D32's listing printed
> "240 of them are re-written". Both are false. The question that exposed it was the obvious one:
> *why load B every wave if three weight sets are still holding tiles?*
>
> **Within one pass every tile is written exactly once, whether or not it fits.** `M` is the
> innermost loop, so a B tile serves all `M` rows and is then never revisited — walking the wave
> schedule confirms 256 distinct tiles, 256 total writes, none written twice. Exceeding the array's
> capacity costs **nothing** in a single pass; the tiles simply displace one another as they go.
>
> What the capacity actually decides is the cost of the **next invocation on the same weights**:
>
> | B | fits the 16 slots? | first run | second run |
> |---|---|---|---|
> | ≤ 16 tiles (≤ 2048x2048 int8) | yes | write once | **0 writes** |
> | 17 tiles | no | write 17 | write 17 again |
> | 256 tiles (8192-cubed) | no | write 256 | write 256 again |
>
> That is the real significance of 4 MiB of D-IMC, and it is an **inference** threshold rather than
> a matmul one: a CNN layer whose weights fit is written into the arrays once and every subsequent
> frame streams only activations. Which is exactly the design point the paper's 2502 FPS ResNet-50
> sits at, and exactly what a one-shot `bwz matmul` cannot show. The engine still models neither
> side — same missing on-chip bandwidth — but it now says the right thing about which one matters.

**Why the activation stream is not a separate term.** The input feeder takes 512 one-bit activations
per cycle per core and accumulates over 8 cycles — 512 INT8 activations per 8 cycles, which is
exactly one MVM. The feed is rate-matched to the array by construction, so streaming A adds no time
beyond the arithmetic it feeds; charging it would be double counting. What A *does* cost is DRAM
traffic when it does not fit on chip, which `op_roofline` already charges as `input_bytes`.

### Still open

`double_buffering_fits` is now known to be vacuous. It tests two array-sized tiles against the whole
on-chip capacity (52 MB on Metis, 60.7 MB on A100), and tiles are array-sized by construction, so it
cannot fail on a shipped profile. Tested against the store the array can really compute from, it
also passes — 164 KiB of SMEM holds 328 A100 tiles, 1 MiB of D-IMC holds exactly 4. The verdicts
were right for the wrong reason. Replacing it with the reload model above is the real fix, and waits
on the same missing bandwidth.

---

## D31 — The DRAM row names its operand, and a bar is a wave (2026-08-17)

Two follow-ons from D30, both asked for by a user staring at a `--compare` figure and finding it
said less than the report behind it.

### 1. `LOAD 107 MB` does not say *which* operand crossed the bus

The DRAM row split by direction (D22) but not by operand, so A and B were one number. They are not
interchangeable: they obey **different residency fractions** and spill at different times, because
on-chip capacity is granted to activations before weights (D15). On an 8192-cubed INT8 matmul that
asymmetry is the whole story:

| | operand B | operand A | result C |
|---|---|---|---|
| `a100_80gb` | 67.1 MB — **100% of B** | 36.7 MB — 55% of A | 67.1 MB |
| `metis_aipu` | 67.1 MB — **100% of B** | 40.4 MB — 60% of A | 67.1 MB |

B streams in full on both chips while A is more than half resident, and one merged `LOAD` figure
hid exactly that. `OpResult`/`PhaseResult` now carry `dram_weight_read_bytes` and
`dram_activation_read_bytes`, `Stage.LOAD_A` joins `Stage.LOAD`, and the figure gives the DRAM row
**three looks** rather than two: operand B solid, operand A hatched, the result hollow. The box
reads `LOAD B 67.1 MB · A 36.7 MB` / `STORE C 67.1 MB`, and the hover names which operand a bar is.

The split is a decomposition, not a recount — `weights + activations` reproduces the report's
`dram_read_bytes` exactly, and a test asserts it.

### 2. A tile step is a wave, not a tile

D30 taught the *utilisation* that a chip is `count` arrays running a wave at a time. The *drawing*
had not been told: `_tile_trace` laid one bar per tile, end to end, so a four-core NPU appeared to
chew through four tiles in series while the same report quoted a wave occupancy of 1.0. The picture
contradicted the number beside it, which is the one thing D19 says a figure may never do.

A step is now `ceil(tiles / units)` waves. An 8192-cubed INT8 matmul on Metis is 256 tiles and draws
**64 bars**, each labelled `8 B tiles 512x512 [n/64]  all in parallel`. When one bar has to coalesce
several waves the qualifier changes to `4 at a time`, because "in parallel" would then overstate it.
`PipelineTrace.tiles` — documented as "steps the schedule really has" — is the wave count, so
`coalesced` compares like with like.

Two consequences worth stating. The bar label carries its qualifier after a double space, which is
the existing `_short()` convention the plot script splits on, so the bar text stays legible and the
hover keeps the whole thing. And the D19 invariant is untouched: waves still sum to
`t_compute + t_fixed` and DRAM busy to `t_dram`, measured to 1e-18 s on both shipped chips.

### Still open, and now more visible

Charging the reload of weight tiles beyond `resident_tile_capacity()` still waits on an L1-to-IMC
bandwidth (D30). With B drawn as its own bar, a reader can now *see* 67.1 MB of operand B crossing
DRAM on every invocation of a matmul whose 16 array-resident tiles could have held a quarter of it —
which makes the missing term harder to forget, and is an argument for closing it.

---

## D32 — The page says how the workload reaches the silicon, per chip (2026-08-17)

The HTML carried two derived sections: the timeline (what happened when) and the arithmetic (D26 —
what was computed). Neither answers *how the work is mapped onto this particular chip*, which after
D30/D31 is where the two machines actually differ. Asked for a pseudo-code section reflecting the
pipeline, and it belongs next to the figure rather than in prose someone has to keep in step.

`bwz/deploy.py` emits a loop nest per chip: how B is cut into array-sized tiles, how many arrays
take a wave of them at once, whether a tile must be **written into** the array before it can
compute, and where the loads and stores sit around it — with the real extents and the real
per-wave byte counts.

**Same contract as `explain.py`, and for the same reason.** Pure, holds no counts of its own, and
`check()` asserts the listing's constants are the schedule's before it reaches the page: `waves ==
trace.tiles`, `waves == ceil(tiles / units)`, and `tiles` from the same `tile_count()` the schedule
and the utilisation model both divide by. A listing that disagreed with the timeline above it would
be worse than no listing, because a reader would believe it.

**Rendered once per chip, unlike the arithmetic.** D29 renders the operator list once on a
comparison because it is a property of the workload and identical on both machines. This is the
exact opposite: the mapping is what differs, so merging it would delete the content.

**No vendor knowledge in the emitter.** Every branch is driven by a field the profile declares, so
the module never learns what an NVIDIA or an Axelera part is:

| declared | emitted |
|---|---|
| `weight_sets > 1` | `WEIGHT_SETS`, a set rotation, and an `imc_write` write-ahead — the next wave's tile lands in a set freed by the wave before it |
| `weight_sets == 1` | no write at all, and "the array stores no weights: both operands are re-read per instruction" |
| dtype multiplier < 1 | `SUB_CYCLES` and an inner loop — 8 for Metis's INT8 0.125, absent at fp16 |
| `systolic_dims` | `ROWS`/`COLS` and the tile nest; a profile without one falls through to the sequence listing |

On 8192-cubed INT8 the two listings say, from the same emitter: Metis **256 tiles over 4 arrays ->
64 waves**, 16 chip-resident, so **240 re-written**; A100 **262 144 tiles over 432 arrays -> 607
waves**, no residency limit. That contrast is the deployment difference in six lines.

A graph of operations gets a different listing — a sequence with no cross-operation overlap, which
is what the model actually does (D5a) — and says so rather than pretending to a tile nest it does
not have. `Deployment.kind` records which, because the wave relation `check()` enforces only holds
for the tiled form.

The pseudo-C carries no nested `/* */`, and a test enforces it on every shipped chip, on the same
grounds as D26: it is meant to survive being pasted.


## D33 — A crosses DRAM exactly once, and reloads land ahead of the arithmetic (2026-08-19)

The original roofline gave the activation operand a residency discount: whatever did not fit on
chip was re-read, scaled by the SRAM capacity (D15). For a **single-matmul graph** that discount
was nonsense, and a user staring at the Metis deploy listing caught it: with M as the innermost
loop, no tile is ever revisited in the same pass — so nothing is re-read, and "how much of A fits"
answers nothing. A is *compulsory traffic, period*, the same way B is.

**The rule is structural, not arithmetic.** The k-slice staging makes it literal: each k-slice of A
is consumed by every output tile of its group (`ceil(N/COLS)` n-tiles) while it sits on chip, so A
crosses DRAM exactly once. `run_phase()` therefore passes `activation_resident_fraction = 0.0` —
no residency benefit — for graphs where no other operation reads the tensor (a lone matmul, which
is also every matmul-benchmark the figures show). The memory *planner* still computes its D15
fraction for multi-op graphs; the two agree there, and only there. An assumption line is appended
in the same breath, so the report never carries the rule silently.

**The same user question exposed a second fiction in the deploy listing.** The old comment said a
reload "must land before the array can use it", which serialised the write in front of the
arithmetic. It is a weight-set *rotation*: the next wave's tile lands in the set freed three waves
ago, hiding behind that wave's MACs. The listing now prints that write-ahead (`(w + 1) % WEIGHT_SETS`)
with the staging prologue in the same loop, and `check()`'s byte labels still reconcile to the
report because the listing sums the same per-wave shares it always did.

**And a third fiction, in the figures.** A user running `--ideal` on 8192-cubed INT8 still saw A
"read multiple times": the timeline hovered every DRAM bar as "operand A, streaming through the
array", and the listing printed a `load_A` on every wave — the D31 per-step share, which is only a
decomposition of the charged traffic. For a lone matmul the physical event is the k-slice
**staging**: one read per k-slice, feeding all `ceil(N/COLS)` tiles of the group. Both renderings
now show the event, not the share. The timeline carries exactly one `LOAD_A` span per k-slice
(16 for Metis at 8192-cubed, each 4.19 MB — the `Span.staged_once` flag keeps the "read once"
wording off network traces, whose activations genuinely stream); the listing stages at k-slice
boundaries (`tile(w, u) % NTILES_PER_KS == 0`). A100's K-tiles are a k-slice per tensor-core step
almost continuously, so its trace barely changes — which is correct, because on A100 the staging
really does interleave with the fetches.

Numbers that moved on 8192-cubed INT8:

- Metis: DRAM traffic 40.4 MB → **67.1 MB** of A, so `t_dram` 5.12 ms → **5.90 ms**, and the
  verdict flips **compute-bound → DRAM-bound**: a reported latency of **5.90 ms**, against a drawn
  span of 6.94 ms — the difference being the fill/drain the roofline omits, which this staging makes
  large enough to matter and which D35 then had to keep out of the comparison headline. The 80 µs the figure called "imc_write" was never a write: it was the pipeline fill/drain
  term `min(t_dram, t_compute) / tiles`.
- A100: the same A-once rule applies (the benchmark has no consumer of A either) — 36.7 → 67.1 MB
  of reads, but 2.04 TB/s of bandwidth makes it a 32 µs change against 1.77 ms of compute. The
  verdict (compute-bound) and the headline latency barely move.

## D34 — The M-tail is a pipeline property, not a shape constant (2026-08-19)

The tail-effect model was `M/(M+rows)` for every array, and the "rows-deep pipeline" reading of it
is only true of a systolic pump. A bit-serial array like the Metis D-IMC is a *combinational
crossbar*: the 512 M-channels enter one bit at a time, but in parallel — there is no M-serial
pipeline to fill. Modelling it with `M/(M+512)` invented a 512-cycle fill that costs 5.9% of a
large-M stream that physically loses nothing; modelling it with `M/(M+8)` (one INT8 row's
sub-cycles) fixed the large-M case but broke the small-M one, where the M=1 GEMM must still run at
~1/513 of peak (CLAUDE.md: a batch-1 GEMM on a 512x512 array at 1/513 is the sanity check that
decides whether the array is modelled at all). The 1/513 is *area* — one active row of 512 — not a
pipeline drain, and conflating the two is the bug.

**The dtype multiplier now selects the physical model**, and the two branches are hand-verifiable:

- systolic pump (`fill_cycles=None`): `M/(M+rows)` — unchanged, all existing goldens hold;
- bit-serial crossbar (`fill_cycles = 1/multiplier`): `M/padded(M,rows)` area only, and the
  sub-cycle stream rides K with one sub-cycle row of fill:
  `K·s / (K_pad·s + s) = K/(K_pad + 1)`.

At M=8192 the crossbar branch is 8192/8193 = 99.99%, where the old model claimed 94.1%. At M=1
it is 1/512 × K/(K+1) ≈ 1/513 — the same 1/513 the systolic branch produces, which is why the
batch-1 sanity checks and `chip_b`'s compute-bound-at-batch-1 golden (D8) still pass: a bit-serial
array is *slower*, not more pipeline-bound. The branch is keyed on `0 < multiplier < 1` only: A100
at INT8 has multiplier 2.0 (wider, not serial — D18) and must not take the crossbar branch, even
though a dtype is present.

## D35 — A comparison quotes the reported latency, not the length of its bars (2026-08-19)

D33's k-slice staging concentrates A into a few large events instead of a per-wave trickle, so far
less of it overlaps compute and the pipeline fill/drain grows. That is correct — the D19 invariant
still holds exactly, DRAM busy reproducing `t_dram` to 1e-18 s — but it collided with a defect
`--compare` had carried since D29 and made it material.

`_subtitle()` computed its headline "Nx faster" from `trace.total_s`, the **drawn** span. The drawn
span is the reported latency **plus** the fill/drain that the roofline's `max()` leaves out, and
that term is wildly asymmetric between machines:

| | reported latency | drawn span | fill/drain |
|---|---|---|---|
| `a100_80gb` | 1.7668 ms | 1.7671 ms | **0.0%** |
| `metis_aipu` | 5.8992 ms | 6.9437 ms | **17.7%** |

So the figure announced **3.93x** where the report says **3.34x** — an 18% error in the single
number a comparison figure exists to state, on the side that flatters neither chip honestly. A
picture that disagrees with the report is the failure D19 was written to prevent, and this was that
failure in the headline rather than in a bar.

Fixed by quoting `reported_latency_s` everywhere a figure states a *duration*: the subtitle's spans
and ratio, each band header's right-hand quantity, and the info-box band labels. The **bars**
deliberately still show the drawn span — they are a schedule, and the fill/drain is really there —
so the subtitle now names the gap instead of hiding it:

> Shared absolute time axis, reported latency. a100_80gb 1.77 ms · metis_aipu 5.9 ms. a100_80gb is
> 3.34x faster than metis_aipu on this workload. Bars run past it by the pipeline fill/drain the
> roofline omits (metis_aipu +1.04 ms).

That restores, in the comparison, the disclosure the single-chip subtitle has always carried and
which D29 dropped when it replaced `_trace_subtitle` with the shared-axis text. It fires only where
the gap is ≥1% of the latency, so a figure with nothing to disclose says nothing.

`deploy.py`'s listing footer had the same ambiguity — it printed `span 6.94 ms = max(...) +
fill/drain` with no mention of the prediction — and now prints the reported latency first and the
drawn span as the thing that adds to it.

A test pins the asymmetry itself (`test_fill_drain_differs_enough_between_chips_to_distort_a_ratio`)
rather than the wording: A100 under 1%, Metis over 10%, reported ratio 3.34x, drawn 3.93x, and the
D19 decomposition still exact on both.

**Housekeeping in the same commit.** `make lint` was failing — `ruff format --check` on the four
files D33/D34 touched. And D33's removal of `docs/plots/*` from history left two dead image embeds
in `README.md`, which rendered locally after `make plots` and were broken on GitHub; they are now a
sentence saying the figures are regenerable and deliberately untracked.

## D36 — A's residency and B's write timing are command-line strategies, not fixed rules (2026-08-19)

D33 hard-coded the single-matmul roofline to "A crosses DRAM exactly once" and D30/D33's deploy
listing hard-coded B's write to "write-ahead". Both were the *right default* and the *wrong ceiling*
— a user should be able to ask "what if this chip's GEMMs genuinely re-fetch A, the way a GPU's do"
or "what if B is small enough to stay resident across calls" without editing the engine. `bwz matmul
--a-strategy {stage,stream,whole}` and `--b-dataflow {write-ahead,on-demand,persistent}` generalise
D31/D33/D30 into one parametric rule each, with the existing behaviour as the default so every
number in D33/D34/D35 is unchanged.

**A is a byte-amount knob.** `A_traffic = |A| · NTILES_PER_KS / residency_tiles`: `stage` (default)
and `whole` serve a whole k-slice per staging event (`residency_tiles = NTILES_PER_KS`), so both
reproduce D33's "crosses DRAM exactly once" exactly and differ only in *when* the events land —
`stage` at each k-slice boundary, `whole` all of them ramped in before wave 0, at zero extra bytes.
`stream` serves one tile per event (`residency_tiles = 1`) and re-reads A `NTILES_PER_KS` times —
the D31 picture, restored as an explicit choice rather than a bug to fix. `--a-residency-tiles`
generalises further to any power-of-2 divisor of `NTILES_PER_KS` in between, clamped down (never
raised as an error, CLAUDE.md #8) when the requested value is not one.

**B is a timing knob, provably.** D30's own correction note proved it: within one pass every B tile
is fetched exactly once whichever placement is chosen, because M is the innermost loop and a tile is
never revisited. `write-ahead`, `on-demand` and `persistent` therefore move zero bytes against each
other in a single invocation — v1 has no on-chip write-bandwidth term to charge one placement more
than another (D5b, D30) — and the deploy.py listing renders each as a different loop-nest placement
with the identical `#define` numbers underneath, `check()`-verified as ever.

**The one place bytes genuinely move is across *iterations*, and that needed a real design
decision — asked of the user rather than picked.** `persistent` promised "0 B on a repeat run" but
`analyze()` costs one report, not N runs, so "0 B" meant nothing until the deployment could say how
many invocations it was being asked about. The chosen answer: `DeploymentSpec.iterations: int = 1`,
read only by `persistent`. When it fits (`tiles <= units * weight_sets`) and `iterations > 1`, the
first invocation writes B in full and the other `iterations - 1` write nothing, so the *report* —
still one invocation's numbers — charges the amortised share `1/iterations` of one write. This is
the regime D30's correction identified as the real significance of Metis's 4 MiB of D-IMC: a CNN
layer whose weights fit is written once and every later frame streams only activations, which is
what the paper's 2502 FPS ResNet-50 point is actually about, and which a one-shot `bwz matmul`
could not express before this.

**Both `whole` and `persistent` clamp on the same shape that exposed D33's original numbers.**
8192³ INT8 on Metis: `whole` needs the scratchpad to hold all 67.1 MB of A against 54.5 MB on chip,
and `persistent` needs the 256 tiles B tiles to fit the array's 16 resident ones — neither does, so
both fall back (to `stage` and `write-ahead` respectively) with the requested value, the value used
and why, named in `report.assumptions` rather than happening silently. `analysis/dataflow.py` is the
one place this resolution happens, so `run_phase`'s bytes, `build_trace`'s schedule and
`deploy.py`'s listing read the same `DataflowPlan` and cannot disagree on which strategy actually
ran.

`Span.staged_once: bool` becomes `Span.a_fetch_mode: str` (`"stage"` / `"stream"` / `"whole"`),
defaulting to `"stream"` — a network's inter-operation activation traffic has no k-slice structure
to stage, so it is the same physical picture `a_strategy=stream` deliberately reproduces for a lone
matmul, and the two now share one hover wording rather than a special-cased "streaming" string.

## D37 — The PNG timeline is gone; the HTML page was never a duplicate (2026-08-19)

`scripts/plot_pipeline.py` wrote a matplotlib PNG *and* the zoomable HTML page from the same trace,
and the PNG existed first (D19–D22 predate the HTML). Once the HTML shipped, the PNG became strictly
worse at everything the two shared: it cannot zoom, so it needed a second "first N steps" register
and a `--zoom` flag purely to work around that, and it is a large regenerated binary nobody reviews
in a diff. Removed: `draw()`, `_bars`, `_row_axis`, `_ticks`, `_info_boxes`, `_draw_roofline`,
`_past`, `_short`, the `BOX_H`/`BAND_LABEL_H`/`BAND_GAP` layout constants, and the matplotlib import
and `Agg` backend selection. `--html` and `--html-steps` lose their reason to exist — `--steps`
(default now 256, the old HTML-only default, since there is no longer a separately-legible static
figure to keep at 32) is the only resolution knob — and `--zoom` goes entirely: the page zooms.

**`plot_pipeline.py` no longer imports a plotting library at all**, confirmed by parsing its own
import list and by running it without `--group plots` — `uv run python scripts/plot_pipeline.py`
writes the same `timeline-*.html` it always did, using nothing but the standard library, `bwz`
itself, and the sibling `timeline_html.py` module (already pure stdlib). `scripts/plot_roofline.py`
is untouched and still needs `--group plots`: it has no HTML equivalent, `roofline-*.png` and
`machine-*.png` are not duplicates of anything, and whether *it* should also grow an HTML form is a
separate decision this commit does not make. `make plots`, `docs/CLI.md` §5 and `docs/plots/README.md`
are updated with real executed output; the stray `.png` timeline files `make plots` used to leave in
`docs/plots/` (gitignored, never committed) are deleted, since the target no longer writes them.

## D38 — A comparison states the rate, not just the duration (2026-08-19)

`bwz compare`'s table and the HTML `COMPUTED` box both stated durations and byte counts; asked for
the throughput explicitly, because "A100 takes 1.77 ms, Metis takes 5.9 ms" is a workload-size-
dependent way to say what "622 TOP/s against 186 TOP/s" says directly, and the rate is what makes
two chips comparable independently of how big the probed matmul happened to be. `PhaseResult`
already carried it — `achieved_flops_per_s` and `utilization` — so this surfaces an existing report
field rather than computing a new one in the plot script.

**Re-derived, not trusted, per CLAUDE.md's standing warning about this session's own prior
mistakes.** `bwz matmul -M 8192 -N 8192 -K 8192 --ideal` on int8:

```
a100_80gb   1.1 TOP / 1.77 ms =  622 TOP/s   (99.77% of 624 TOP/s peak, COMPUTE_BOUND)
metis_aipu  1.1 TOP / 5.90 ms =  186 TOP/s   (88.88% of 210 TOP/s peak, DRAM_BW_BOUND)
```

622 / 186 = 3.34x, the inverse of Metis's 5.90 / 1.77 = 3.34x latency ratio — the same check D35
already runs on the duration side, now run on the rate side too, and it agrees.

**It must come from `phase.achieved_flops_per_s`, derived from the reported latency, never from the
drawn span (D19, D35).** Placed in the `COMPUTED` box first line and in each comparison band header,
next to the *existing* `array … @ 622 TOP/s while busy, of 624 TOP/s` line — `totals / busy`, the
rate while the array specifically was busy — which is a different and also true number: on a
DRAM-bound run the array can sit near its own peak whenever it does get a tile, while the chip's
delivered rate is far below it because the array is idle most of the span. The two are labelled
apart (`achieved` vs `while busy`) so a reader cannot mistake "the array ran at 622 TOP/s while it
was running" for "the chip delivered 622 TOP/s on this workload" — confusing them was the failure
this correction exists to prevent.

## D39 — The two kernel probes get one factory each, not two hand-typed dicts (2026-08-20)

`bwz matmul`/`bwz single-layer-encoder` and `scripts/plot_pipeline.py`'s `--matmul`/`--encoder`
each build a spec straight from CLI shape arguments — a "kernel probe", the smallest unit the
engine costs, as opposed to a model loaded from a profile. The CLI's report and the script's figure
are meant to describe the same probe; each independently hand-built the same dict a second time,
and they had already drifted.

**What had drifted, found by a user question asking why the encoder profile and the encoder command
looked like the same thing.** `plot_pipeline.py`'s `build_matmul()` used a placeholder id (`"p"`)
where `bwz matmul` derives one from the shape, and dropped per-operand width support entirely — no
`--out`/`--a`/`--b` exist on the script, so a plotted matmul can never show a widening accumulator
or mixed operands, only `bwz matmul` can. `build_encoder()` hardcoded three architecture knobs
(`ffn_type`, `norm`, `tie_embeddings`) that `bwz single-layer-encoder` exposes as `--ffn-type`/
`--norm`/`--tie`, with no way to override them from the script. And `build_encoder()`'s `name`
baked in `S={tokens}`, which the shared `_workloads_for()` helper then appended a second time —
the HTML timeline title for `--encoder` read `...ffn=16 S=512 prefill S=512`.

**Fix: `bwz/kernels.py`**, two pure factory functions — `matmul_kernel(m, n, k, *, a_dtype, b_dtype,
out_dtype=None)` and `encoder_layer_kernel(*, hidden, heads, head_dim=None, ffn, vocab, tokens,
ffn_type=RELU, norm=RMSNORM, tie_embeddings=True)` — both deriving id and name from the shape
(`matmul_4096x4096x4096`, `encoder_layer_d8_h2_ffn16_s4`) rather than a placeholder or a constant.
`cli.py`'s `matmul`/`encoder-layer` commands and `plot_pipeline.py`'s `build_matmul`/`build_encoder`
now call these instead of each writing the dict by hand. The `S=` double-embed is gone as a side
effect: the factory's `name` doesn't carry it, so `_workloads_for()`'s own append is the only place
it appears. **Scope, decided deliberately:** the factories accept the full parameter set `cli.py`
already had (so nothing there loses capability), but no new flags were added to `plot_pipeline.py`
— `build_matmul`/`build_encoder` still pass through only what the script already exposes, defaulting
the rest exactly as before. Closing the `--out`/`--a`/`--b`/`--ffn-type`/`--norm`/`--tie` gap on the
script, if wanted, is a separate decision.

**The command is renamed** `bwz single-layer-encoder` → `bwz encoder-layer`, to read as a kernel
probe the same way `bwz matmul` does — one word naming the kernel (`matmul`, `encoder-layer`), not a
sentence describing it.

**The bundled profile is renamed** `single_layer_encoder` → `single_layer_encoder_toy`
(`profiles/models/single_layer_encoder_toy.yaml`, D24's hand-countable reference model — 664
parameters, 5280 operations, unchanged by this rename). Its old id read as the same thing as the
`bwz single-layer-encoder` command even though they were unrelated: one a frozen teaching profile
loaded by id, the other an ad-hoc probe built from flags. `_check_id_matches_filename()`
(`bwz/spec/loaders.py`) keeps the filename and `id:` field in lockstep automatically.

`tests/unit/test_single_layer_encoder.py` splits along the same line the two concepts now make
explicit: the profile-only tests stay in the renamed `test_single_layer_encoder_toy.py`, and
`test_the_cli_shape_matches_the_profile` — which asserts the kernel probe's default shape and the
toy profile agree — moves to the new `test_kernels.py`, now calling `encoder_layer_kernel()` instead
of re-typing the dict a third time.

## D40 — B-dataflow gets real schedule timing, and persistent provably can't buy more than write-ahead already does (2026-08-20)

A request for a DRAM -> SRAM -> Accelerator flow animation (`plot_pipeline.py --animate`) needed
`--b-dataflow` to look different in motion, not just in a text listing. Before this entry,
`--b-dataflow` (D33) only changed the pseudo-C text `deploy.py` prints — `_pipelined_tiles`
(`analysis/pipeline.py`), the function that actually times the schedule, never read
`dataflow.b_dataflow` at all. Animating it honestly meant giving it real timing first.

**The constraint that ruled out the obvious approach.** `deploy.py`'s pseudo-C shows
`write-ahead`/`on-demand`/`persistent` placing an `imc_write` (the on-chip-buffer -> array-register
write) differently, and the naive fix is to cost that write. It has no cost anywhere in this
codebase — no on-chip write-bandwidth term exists in `calibration.py`, and D5b/D30/D36 already
establish that as a deliberate v1 boundary. Costing it would mean fabricating an uncited constant,
which CLAUDE.md forbids outright. The fix instead reuses only the one already-costed quantity that
exists — B's per-wave DRAM load duration, a real bandwidth-based number — and changes only *when*
that existing cost is allowed to overlap compute. No new constants anywhere.

**`on-demand` (D40 fix).** `_pipelined_tiles`'s recurrence schedules B's load as
`load_start = max(dram_free, freed)` — never gated on the previous wave's compute, which is
`write-ahead`'s definition (hidden behind an earlier wave's arithmetic) and was already every
chip's unconditional behaviour. `on-demand` adds one more lower bound,
`max(dram_free, freed, exec_end)`, forcing the load to wait for the previous wave's compute to
finish — no prefetch-ahead — which exposes the same, already-modelled load duration on the critical
path. Verified against the real engine (`bwz matmul -M 20000 -N 2048 -K 2048 -c metis_aipu -d int8`):
`on-demand`'s `trace.total_s` is `write-ahead`'s plus exactly one wave's compute time
(`t_compute_s / steps = 0.0008196 / 4 = 0.0002049 s`), because the DRAM port is saturated end to
end under `write-ahead` on this shape, so the gate binds once and the delay is a constant shift.

**`persistent` — no schedule change, and this is the load-bearing finding, not a shortfall.** Two
tempting fixes were tried and are both wrong. Zeroing the load for waves after wave 0 breaks D36's
own invariant ("moves zero bytes against each other in a single invocation") and desyncs a span's
bytes from its timing, double-counting the amortisation `DataflowPlan.b_write_multiplier` already
applies one layer up, before `_tile_trace` ever runs. Letting a `depth_override` (mirroring
`--a-prefetch-depth`) skip further ahead is a no-op: `store_ends` is monotonically non-decreasing by
construction, so `freed = store_ends[i-depth] <= dram_free` for *every* `depth >= 2` — swept
`depth_override` from 2 to 64 on a real shape and got a bit-identical `trace.total_s` every time,
which `docs/CLI.md` §2.5's own worked `--a-prefetch-depth` example independently corroborates (it
only ever shows `depth=1` differing from the default). The honest conclusion: `write-ahead`'s
existing double buffering already achieves this model's best-case overlap, so there is no
byte-conserving reordering of B's fixed loads that makes a single-pass `persistent` trace faster.
`persistent` therefore gets no schedule change at all — its trace is `write-ahead`'s, span for
span. Its real, already-correctly-implemented effect stays `DataflowPlan.b_write_multiplier`, a
byte story across `--iterations`, not a same-pass timing one.

**The `weight_sets <= 1` gate, and a bug it fixes for free.** `deploy.py` already gates every
write-ahead/on-demand/persistent-specific line behind `unit.weight_sets > 1` — a chip with no
resident weight bank (`weight_sets = 1`, every shipped GPU profile) has nothing to place ahead of,
expose, or persist. `plan_dataflow` (`analysis/dataflow.py`) needed the same gate, added right after
the existing persistent-capacity clamp: `on-demand`/`persistent` requested on a `weight_sets <= 1`
chip now resolve to `write-ahead`, with a note. This closes a real, verified latent bug as a side
effect: before the gate, `--b-dataflow persistent --iterations 4` on `a100_80gb` (`weight_sets=1`)
gave `b_write_multiplier=0.25` — a 25% DRAM-write discount for a chip whose tensor cores have no
resident weight bank at all to amortise across calls, contradicting the chip's own declared physics.

**What this does and does not change.** `t_dram`, `t_compute` and the reported `latency` are
identical across all three placements for the same shape — pinned as a golden
(`test_b_dataflow_never_moves_a_report_number`, `tests/unit/test_pipeline.py`) — because they are
decided before `b_dataflow` is ever read; only `trace.total_s`/`fill_drain_s`, the *drawn* schedule,
differ. This is D19's asymmetry rule again: a trace may run slower than the report it illustrates,
never faster, and the report itself never moves — the same rule `--a-prefetch-depth` already
established for A. Five golden tests in `test_pipeline.py` cover: `write-ahead` is not a new code
path (regression guard — every other test in the file assumes it), `on-demand`'s exact derived
delay, `persistent` matching `write-ahead` span for span, the byte/latency invariant across all
three, and the `weight_sets <= 1` gate plus the `--iterations` bug fix.

## D41 — The pseudo-C now shows double buffering instead of just claiming it, and `--animate` highlights it live (2026-08-20)

A user watching the new `--animate` page asked a sharp question: does the pseudo-C loop nest
`bwz/deploy.py` generates actually *show* double buffering, or does it just say so? It didn't.
`#define DEPTH 2 /* double buffered: capacity fits two tiles */` sat above a loop that read as
fully serial — load wave *w*, compute wave *w*, store wave *w*, then move to *w+1* — for every chip
except the Metis-style weight-set case, where only a comment hinted at overlap. The real schedule
(`_pipelined_tiles`) genuinely overlaps wave *i*'s load with wave *i-1*'s compute whenever
`depth==2` (`load_start(i) = max(dram_free, store_ends[i-2])`, never gated on the previous wave's
compute) — the listing should read that way too, since its own docstring's whole point is showing
"how that arithmetic reaches this particular silicon."

**Fix.** `deployment_of()` now builds a genuine prologue (wave 0's tile loads before the loop
starts) plus a steady-state loop that prefetches wave *w+1* alongside wave *w*'s own compute —
but only when `trace.double_buffered` is `True`. When it is `False` (`depth==1`), the listing keeps
the exact serial shape it always had, because that *is* the faithful rendering of a genuinely
serialized schedule (`_pipelined_tiles`'s depth==1 branch really does place a wave's store before
computing the next wave's load start). No shipped profile at any shape actually produces
`depth==1` — verified across every chip from 600³ to 32768³ — so the fallback path is exercised in
`test_deploy.py` by forcing it directly (shrinking every non-DRAM `MemoryLevel.capacity_bytes`
below twice one tile's bytes, which flips `report.memory.double_buffered` end to end; a bare
`on_chip_capacity_bytes` override alone silently no-ops, since it is a computed property, not a
stored field).

**A real, pre-existing bug fell out of writing the prologue honestly.** Under `write-ahead`, no
statement anywhere wrote wave 0's own B tile into a weight set — the per-wave line only ever wrote
`tile(w+1, u)`, so at `w=0` that's tile 1, never tile 0 — even though the old prologue *comment*
already claimed wave 0's tiles landed somewhere. The new prologue adds the actual statement,
`imc_write(u, 0, tile(0, u))`.

Every existing `test_deploy.py` assertion (19 tests, all substring-based, none checking line order)
passes unmodified against the restructured text — checked line by line before writing the change,
not discovered by running the suite afterward. Two new tests: one pins the double-buffered shape
(the prologue exists, the steady-state loop references `tile(w + 1, u)`, the wave-0 fix is present,
every pre-existing substring still survives), one pins the forced depth==1 fallback staying exactly
serial (no prologue, no `w + 1` anywhere).

**`Deployment` gains `stage_lines`** — a `tuple[tuple[str, tuple[int, ...]], ...]` (matching
`PipelineTrace.work_by_op`'s own reason for avoiding a plain `dict` on a frozen dataclass) mapping
each animated stage (`load_b`, `load_a`, `exec`, `store` — the same vocabulary
`plot_pipeline._ANIMATION_STAGE` already uses) to the line number(s) in `Deployment.code` where
that stage's statement appears. Built by constructing the *entire* listing — header, `#define`s,
body, footer — as one flat list of `(line_text, stage_tag | None)` pairs and enumerating it once,
rather than computing an offset from separate lists' lengths, which breaks the moment a header
branch grows or shrinks by a line. `imc_write` and buffer-hold ("the tile sits resident") are
deliberately left untagged: neither has a `Stage`/`Span` that ever marks it "live" — `imc_write`
has no cost/timing model at all (D40), and residency is implicit in a C variable's lifetime, not an
explicit statement — so pointing a line at either would be decoration, not data.

**`plot_pipeline.py --animate` shows the listing live, debugger-style.** `write_animation_html` now
calls `deployment_of()` the same way the timeline page's `_deployments()` always has, and passes the
split lines plus `stage_lines` into `dataflow_html.render()`. A new scrollable code pane sits beside
the station diagram; every animation frame reads the same `activeAt(t)` events the moving blocks
already use, unions the line numbers for whichever stages are currently live, and highlights exactly
those lines — lane-coloured (DRAM blue for loads/stores, core orange for the arithmetic line) so a
highlighted `load_B` line and the block sliding toward SRAM read as the same event. When double
buffering means a `load_B` line for wave *w+1* and the `mac`/`feed` line for wave *w* are both live
at once, **both light up together** — verified by sampling the generated trace directly (not just
by eye): at a real moment in an 8192³ INT8 Metis trace, `load_b` and `exec` are simultaneously
active and the union correctly resolves to both the prologue/steady-state load line and the exec
line. Auto-scroll only fires when the *set* of hot lines changes frame to frame, not every frame, so
it doesn't fight a reader who scrolled up to read the header.

Out of scope, deliberately: highlighting `imc_write` (no event models its timing — a physics
extension, not a rendering one, and a separate decision if ever wanted); highlighting buffer
occupancy (no statement to point at); bringing the same hover-to-highlight treatment to the
timeline page's own static `<pre class="deploy">` block (a cheap, obvious follow-on, not what was
asked, not built now).

## D42 — `--animate` learns the encoder, and the debug pane's real limit for a network (2026-08-20)

D40/D41 built `--animate` and its code-highlight against the tiled matmul path only. `bwz/kernels.py`
names exactly two kernel probes — `matmul_kernel` and `encoder_layer_kernel` — and the original
request for this feature was explicit that both should eventually animate ("the matmul (now) and the
other kernels (later)"). Extending to `--encoder` meant going through a genuinely different code
path first: a multi-operation graph builds its trace via `_operation_trace`/`_serial_steps`, not
`_tile_trace`, and its pseudo-C via `_network_deployment`, not the tiled `deployment_of` body — both
already existed, neither had ever been asked to feed an animation before.

**What the research found, before any code changed.** Cross-operation overlap is exactly zero —
`_operation_trace`'s own comment: "Operations do not pipeline against each other in this model...
the trace reproduces the reported latency exactly" — operations run in strict sequence (D5a).
*Within* one operation, load and compute genuinely can overlap when double buffered
(`_serial_steps`'s `exec_start` formula), so the debug highlight still sometimes shows two lines lit
together — just never two different named operations at once. Every `Span` a network trace produces
already carries the real operation's name (`Span.label = OpResult.op_id`, e.g. `"q_proj"`,
`"ffn_up"`) — the animation's blocks and hover text were already correct per-operation with zero
changes needed. What is *not* per-operation is `_network_deployment`'s listing text: one generic
`for (i = 0; i < OPS; ++i) { load_B(op[i]); ...; store_C(op[i]); }` loop, never unrolled per named
op, so the pane can say "a load is happening" but never "*q_proj's* load is happening" — a real,
structural limit of a generic loop nest, not a bug to chase here. EXEC can also land on
`Lane.VECTOR` as well as `Lane.CORE` (D27's matrix/vector split) — genuinely new territory, since a
lone matmul only ever produces `Lane.CORE` EXEC spans.

**Fix, scoped to match what was actually found.** `_network_deployment` gained the same
`stage_lines` tagging the tiled path got in D41 (`load_B`→`load_b`, `load_A`→`load_a`, `store_C`→
`store`) — mechanical, since the loop body was already a fixed shape, just never tracked line
numbers. Both compute branches (`if (is_matrix(op[i])) ... else ...`) tag the same plain `"exec"`,
deliberately, rather than adding a `exec_core`/`exec_vector` split: which branch runs for a given
`op[i]` is exactly what this generic text never names either, so lighting both whenever *either*
engine is active is an honest answer, not an imprecise one — the annotation panel and hover tooltip
already disambiguate engine and operation. `plot_pipeline.py`'s `--animate` gate now only rejects
`--model` (hundreds of operations, well past what a three-station diagram or a debug pane usefully
shows) and `--compare`; `write_animation_html` no longer asserts a matmul-only `dataflow` plan —
when there isn't one (every `--encoder` workload), `deployment_of` is called without
`a_strategy`/`b_dataflow` (the same conditional-kwarg pattern the timeline page's `_deployments()`
already used), and the animation banner names the workload as "a network graph — operations run in
sequence (D5a)" instead of quoting a strategy that does not apply.

No JS changed at all — `updateCodeHighlight`/`draw` were already stage-generic, and a `Lane.VECTOR`
EXEC event still just carries `stage: "exec"`. Verified against a real encoder-layer trace: EXEC
spans land on both `core` and `vector` lanes as predicted, and a real playback moment shows `exec`
(vector) and `store` concurrently active — the within-operation-overlap claim above, confirmed, not
assumed. One new golden test (`test_network_listing_tags_stage_lines_for_the_debug_view`,
`tests/unit/test_deploy.py`) pins the tagging on the encoder-layer kernel fixture; the pre-existing
`test_a_network_gets_a_sequence_listing_rather_than_a_tile_nest` passes unmodified.

## D43 — The animation draws every declared resource, not three fixed stations; the array's weight write gets text on the network path too (2026-08-20)

A user question ("Metis needs to move data from SRAM to the IMC — I don't see this in the
animation... and doesn't `d_imc` only do matmul, where's the rest of the encoder's code?") led to
two findings, one about physics, one about the animation's own fidelity to what the timeline
already does.

**On-chip movement is uncosted for every chip, not an NVIDIA-specific exemption.** D5a/D5b already
establish this — the flat v1 machine has no on-chip bandwidth term for *any* chip, a deliberate,
user-approved, quantified simplification, not a consequence of NVIDIA's tensor cores having
"implicit" caching. Checking the actual profile data found a real asymmetry worth recording: A100's
L1/L2 `bandwidth_bytes_per_s` fields are self-labelled "order-of-magnitude estimates; NVIDIA
publishes neither" (`profiles/chips/a100_80gb.yaml`) — not proof-grade by this project's own
standard. Metis's profile is more specific: its `estimates.memory` note *derives* an L1→D-IMC
activation-feed figure (204.8 GB/s) directly from the paper's own stated architecture ("each cycle
processing 512 single bit activations through the input feeder from the L1 memory"), and a
D-IMC operand-delivery rate (838 TB/s) from the array's own MAC geometry — closer to real proof
than anything NVIDIA has here, the opposite of what the existing `imc_write` asymmetry (Metis gets
text, NVIDIA doesn't need it) might suggest. Not acted on here — D5b's own quantified margin
analysis (3.8×–260× headroom before the omission would matter) was run against `chip_a`/`chip_b`
before Metis joined the roster and has never been redone for Metis specifically, so whether costing
this would actually change anything for Metis is still an open question, not one this entry closes.
Trigger to revisit stays D5b's own: a published SRAM organisation or a measured figure — which
Metis's paper-derived numbers arguably already clear and NVIDIA's do not.

**The animation's own station list didn't match the timeline's.** `rows_for` (the timeline's row
list) already draws one row per declared memory level and compute unit, grey for what v1 doesn't
cost, with the reason (D20). `_stations_for` (the animation's, until this entry) hardcoded exactly
three conceptual stations regardless of what a chip declares — its own docstring said so. Verified
directly rather than assumed: under `rows_for`'s existing deepest/shallowest/matrix/vector/other
rule, **Metis gets 6 stations** (LPDDR4x, L2, L1, D-IMC, `d_imc`, `dpu`) and **A100 gets 5** (HBM2e,
L2, L1, `tensor_core`, `cuda_core`) — confirming `cuda_core` really is A100's `vector_unit`
(`machine_model(...).vector_unit.name == "cuda_core"`), catching a wrong claim from first-pass
research before it reached the design. Both chips need a second, distinct compute station — not a
Metis-specific gap, matching what the user flagged.

**Fix.** `_stations_for` is gone; `write_animation_html` calls `rows_for` directly, the same
function the timeline already uses, converting each `Row` into a station dict (`name`, `detail`,
`note`, `lane`). The animation's station layout is now dynamic (`stationLayout()` spaces N stations
evenly, shrinking width with a floor rather than assuming 3), stations are found by **lane**, not a
fixed index, and grey stations (`lane === null`) render `var(--idle)`-styled with no glow and no
motion target, their hover tip showing `row.note` (e.g. *"declared, not modelled — the roofline is
flat (D5)"*) — the same "grey with the reason" convention D20 established for the timeline, now
shared rather than reinvented. **Every station with a lane glows independently** on its own
matching-lane `exec` event, replacing the single hardcoded "accelerator glows on any exec" — this
is the direct fix for "where's the dpu work": the vector station now visibly glows only when a
vector-lane operation is genuinely running, never together with the matrix station (verified: 0
moments of simultaneous core+vector glow in a real encoder trace, matching D5a's strict-sequence
rule for a network).

Two real, latent JS bugs were found and fixed as a side effect of keying station colour by lane
instead of a station "key" string: the accelerator's stroke/label colour referenced `var(--acc)`,
a CSS variable that was never defined (only `--dram`/`--sram`/`--core`/`--vector`/`--idle` exist);
and the block-fill `COLOUR` map had no `"vector"` entry, so any `Lane.VECTOR` flow event (only
possible since D42 added `--encoder` support) silently rendered grey instead of green. Neither was
ever exercised by a test, since nothing checked animation SVG output directly.

**The code pane's `"exec"` tag needed the same precision the stations just gained.**
`_network_deployment` tagged both the matrix-branch and vector-branch lines with plain `"exec"`
(D42's deliberate simplification, made before per-engine stations existed to use a finer split) —
both would light up together even when only one station glows. Split into `"exec_core"`/
`"exec_vector"`; the tiled matmul path is untouched (its `exec` line is always `Lane.CORE`, nothing
to disambiguate, and touching it would only churn its own passing tests for no benefit). The JS
lookup tries `stage + "_" + lane` first, falling back to the plain stage name — so the tiled path's
still-plain `"exec"` keeps resolving with zero changes there. This is D42's stated reasoning
partially superseded, recorded here rather than editing D42 (append-only).

**The network path never mentioned `imc_write` at all**, even textually — unlike the tiled matmul
path, which has shown it (untimed, D40) for `weight_sets > 1` chips since it existed. A real,
additional inconsistency, not just "less detailed": the same D-IMC array needs its weight set
written before *any* operation's arithmetic, not only a lone matmul's. Added one unconditional,
untagged `imc_write(op[i]);` line when `unit.weight_sets > 1`, worded to claim nothing about
placement — `_network_deployment` never receives a `b_dataflow` at all (a network trace has no
per-operation dataflow strategy to name, D5a), so it cannot honestly say write-ahead, on-demand, or
persistent the way the tiled path's three placements do.

Two new/updated golden tests in `test_deploy.py`: the D42 stage-lines test updated for the
`exec_core`/`exec_vector` split (23 total tests, all green, nothing else in the file touched); a
new test confirming `imc_write` appears for Metis and is absent for A100, mirroring the tiled
path's own `test_an_imc_array_gets_weight_sets_and_a_write_and_a_tensor_core_does_not`.

## D44 — The encoder kernel probe takes `dmodel`/`nheads`, not `head_dim` (2026-08-21)

`bwz encoder-layer`/`plot_pipeline.py --encoder` took `--hidden`/`--heads` with an optional
`--head-dim` override that silently floor-divided (`hidden // heads`) when omitted, with no
validation at all if an explicit override left the three inconsistent. User request (2026-08-21):
rename to the standard transformer literature's own terms (`dmodel`/`nheads`), and stop
`head_dim` being a separate input at all — always `dmodel // nheads`, erroring rather than
flooring when it doesn't divide evenly.

**Scope, deliberately narrow — the kernel probe only, not the schema every loaded model uses.**
`TransformerParams` (`bwz/spec/model_spec.py`) already has near-identical validation
(`_check_shapes`: errors when `head_dim is None and hidden % heads != 0`) but deliberately keeps
`head_dim` independently settable, because real profiles need it — Gemma-3 is `8 heads x 256
head_dim = 2048`, against a `hidden` of `2560`, not `2560/8`. Removing that flexibility from the
schema would break every GQA-style profile. `bwz/kernels.py`'s `encoder_layer_kernel` is
different: it exists so a shape can be typed on the command line and its effect read straight off
(§3.1's "small enough to count by hand"), and for that purpose one head-count knob that must
divide cleanly is the right amount of flexibility, not a missing feature. The two renamed CLI
surfaces (`bwz encoder-layer`, `plot_pipeline.py --encoder`) call this factory; nothing about
`--model`, loaded profiles, or the schema itself changed.

`encoder_layer_kernel` raises a plain `ValueError` — in the caller's own vocabulary (`dmodel`/
`nheads`), not the schema's (`hidden`/`heads`), which would otherwise print terms nobody typed —
before ever constructing a spec, so `TransformerParams`'s own validator becomes an
unreachable-but-harmless backstop from this path (it still applies in full to `--model`). `bwz
encoder-layer`'s existing `except (SpecLoadError, ValidationError)` block gained `ValueError` to
catch it cleanly; `plot_pipeline.py`'s `build_encoder()` (which has no typer-style handling to
inherit) converts it to `SystemExit`, the same idiom `build_matmul()` already uses for a
user-facing construction failure.

## D45 — `--ffn`'s default follows `--dmodel`, but only when `--dmodel` was actually typed (2026-08-21)

User request (2026-08-21): make `--ffn`'s default `4 x dmodel`, matching the standard transformer
FFN-expansion ratio, instead of the fixed `16` it inherited from `single_layer_encoder_toy.yaml`'s
own toy numbers.

**Why not unconditionally `4 x dmodel`.** `bwz encoder-layer`'s bare invocation (no flags beyond
`--chip`) is the hand-countable example §3.1 and D24 pin: `dmodel=8, ffn=16` producing exactly 664
parameters / 5280 operations, quoted in `docs/CLI.md`, `README.md`, and the profile's own header.
`4 x 8 = 32`, not 16 — so making the rule unconditional would silently move the one number this
whole example exists to keep fixed, breaking every doc and test that quotes it. Raised via
`AskUserQuestion` rather than picked silently; user chose **conditional on `--dmodel` being
explicit** over recomputing the golden numbers.

**The rule:** `--ffn` defaults to `16` when `--dmodel` is also left at its own default (8) — the
bare command is unchanged, byte for byte. The moment `--dmodel` is passed explicitly, an omitted
`--ffn` becomes `4 x --dmodel` instead. An explicit `--ffn` always wins over either default. This
needed a "was `dmodel` actually typed" boolean, not just "what did `dmodel` resolve to" — the two
CLI layers implement it with the same `None`-sentinel pattern already used elsewhere for
order-dependent defaults: `bwz/cli.py`'s `encoder_layer` command takes `dmodel: int | None` and
`ffn: int | None` (`typer.Option(None, ...)`), resolves `dmodel_given = dmodel is not None` before
filling in `dmodel`'s own default, then fills `ffn` conditionally; `plot_pipeline.py`'s
`_reject_flags_for_the_wrong_workload` does the equivalent with `argparse`'s existing
`default=None` fields. Both call sites share the identical three-line resolution shape.

Verified unchanged: `bwz encoder-layer --chip a100_80gb --ideal` still prints 664 params / 5280 ops.
Verified new behaviour: `--dmodel 32` alone gives `ffn=128` (4x32, 8,192 FFN params); `--dmodel 32
--ffn 64` gives `ffn=64`, the explicit value, not 128. `plot_pipeline.py --encoder --dmodel 32`
mirrors the same `ffn=128` in its generated timeline title and pseudo-C.

## D46 — A's k-slice byte share no longer inflates on an underfull wave (2026-08-21)

Found while explaining the `--animate` output for `matmul 1,1,2 --chip a100_80gb`: the DRAM lane's
lone `LOAD_A` event showed A staged at 1.73 kB, but the report's own numbers say operand A is 4 B
in total (`1x2` at fp16), staged once, crossing DRAM exactly once (D33).

**Root cause**, in `analysis/pipeline.py`'s `_tile_trace`: the per-step count of "how many k-slices
opened this step" was computed against `total_tiles = waves * units` — the array's full theoretical
capacity (432 tensor-core slots on A100, one wave) — not the real tile count (`tile_count`, 1 tile
here). Every idle slot in that mostly-empty wave was counted as if it, too, opened a fresh A
k-slice, so the byte share came out `4 B x 432 = 1728 B`. This is not merely a cosmetic label:
`load_a`'s *duration* is that byte count times a bytes-to-seconds rate (`scale`), so the inflated
bytes inflated the span's time too — the trace's DRAM busy time came out ~288x the report's own
`t_dram` (848 ps drawn against a true 2.94 ps), which is exactly the "extra 848 ps of pipeline
fill/drain" the report printed for that shape. Not unique to this pathological small case either:
any matmul whose tile count doesn't divide evenly into the array count leaves its last wave
partially empty, and the old formula counted every empty slot in that wave as an opening — just
imperceptible when the real tile count dwarfs the array count, dramatic when it doesn't (M=1 is
this repo's own standing tail-effect example, CLAUDE.md's sanity checks).

**The fix:** cap `total_tiles` at the real `tiles` count instead of `waves * units`, so the
`open_tile`/`end_tile` window used to count k-slice openings tracks occupied tiles, not theoretical
array capacity. One line in `_tile_trace`; every other formula downstream (`a_bytes_step`, `load_a`,
`ks_opened`) already divided by `k_slices` and multiplied by the real op's `dram_activation_read_bytes`
correctly — only the occupancy count feeding those was wrong.

Added `test_a_load_bytes_do_not_inflate_when_the_last_wave_is_underfull`
(`tests/unit/test_pipeline.py`) as the golden regression: `matmul 1,1,2` on `a100_80gb` now reports
exactly 1 `LOAD_A` span carrying A's true 4 B, and the trace's total DRAM busy time matches the
report's `t_dram` to within `1e-6` relative — confirmed to fail against the pre-fix code
(`1728.0 == 4.0` assertion failure) before the one-line change, and to pass after. Re-verified the
large multi-wave case (`4096,4096,4096` on `a100_80gb`) is unaffected: DRAM busy still matches
`t_dram` exactly, as it did before — that shape's tile count already divided evenly into the array
count, so `waves * units` and `tiles` coincided there and the bug never showed.

## D47 — The pseudo-C's per-tile B/C byte annotations had the same D46 bug (2026-08-21)

Found while answering "is the generated pseudo-code correct too?" right after D46. `deploy.py`'s
`deployment_of` prints `load_B`/`store_C` comments annotated with one representative tile's byte
share (`b_bytes`, `c_bytes`), computed as `result.dram_weight_read_bytes / per` and
`result.dram_write_bytes / per` with `per = max(waves, 1) * units` — the exact same
theoretical-capacity divisor D46 had just removed from `_tile_trace`, in a different call site
computing a different annotation.

For `matmul 1,1,2` on `a100_80gb` this printed `store_C(...)  /* 4.63 mB — hollow bar */` for what
is actually one whole 2 B write — a fractional-byte quantity with no physical meaning, the same
symptom D46 fixed, just visible here as a comment shrinking below one byte instead of a span
duration inflating. On `metis_aipu` the same shape printed `250 mB`. A's own `k_slice_bytes`
annotation was never affected — it already divided by `k_slices`, a real geometric property
(`ceil(K/rows)`), not by unit count.

**The fix**, mirroring D46 exactly: `per = max(tiles, 1)` — the real tile count — instead of
`waves * units`. For any shape whose tile count already divides evenly into the array count the two
coincide, so nothing changes there (confirmed: `4096,4096,4096` on `a100_80gb`, itself off by 128
of 65664 slots, moved `store_C`'s annotation by 0.2%, an unchanged display value). For an underfull
wave, `store_C` now reads `2 B` on `a100_80gb` and `1 B` on `metis_aipu` — the true, whole write —
instead of a fraction of one.

Added `test_the_per_tile_byte_share_does_not_shrink_when_the_wave_is_underfull`
(`tests/unit/test_deploy.py`), same `matmul 1,1,2` shape D46's own test uses, asserting the exact
`store_C` comment text and that no `mB` string appears in the listing at all — confirmed to fail
against the pre-fix code (`4.63 mB` in the listing) before the change, and to pass after. No
existing test in `test_deploy.py` pinned exact byte values in these comments — only line structure
and presence — so nothing else needed updating.

## D48 — A tile-geometry panel for `--animate`: A/B/C's shapes, sizes, and index range (2026-08-21)

User request (2026-08-21): the animation's only clue to which tile is moving was a text label like
"A k-slice 1/125" — no picture of how big a tile actually is, which axis got cut to produce it, or
where it sits in the whole operand. Educational project; the ask was to make the cut visible, not
just narrated: three rectangles for M/N/K, the current A/B tile locations, tile size as "rows x
cols" for both operands, and index labels like `A(0,0)` to `A(4,8)`.

**One correction made transparent, not silently absorbed.** In this model A is never tiled along
M — "M streams; it never tiles" (the pseudo-C's own existing comment, `deploy.py`). A is cut only
along K into k-slices; the whole M height reads one staged slice. So A's honest index is
one-dimensional (`A(:,0) … A(:,{k_slices-1})`), while B genuinely has a 2-D grid (`B(0,0) …
B({k_slices-1},{tiles_per_ks-1})`, matching the user's `(0,0)` format exactly). The panel shows this
directly — A's rectangle is cut with vertical lines only, B's with a full grid.

**Layout**: the classic GEMM diagram (A bottom-left, B top-right, C bottom-right, blank top-left
corner) rather than three disconnected boxes — it makes the shared axes visible for free: A's width
and B's height are both K, drawn to one shared log-compressed scale (same principle `blockSize`
already uses for byte-sized blocks, not a new one); B's width and C's width are both N; A's height
and C's height are both M. Grid lines capped at ~40/axis with a coarser stride and an explicit note
past that — never a silent truncation.

**Data plumbing, no formula changes.** Everything the panel needs was already computed and
discarded upstream — this just plumbs it through:
- `analysis/pipeline.py`: `Span` gained `tile_start`/`tile_end` — the global, k-major
  `[start, end)` tile-index range one drawn step covers, the same numbering `deploy.py`'s
  `KSLICE`/`tile()` already use. Computed once in `_tile_trace` (reusing D46's `total_tiles = tiles`
  fix, not duplicating it) and threaded through `_pipelined_tiles` onto every span a step produces —
  one range per step, shared by LOAD/LOAD_A/HOLD/EXEC/STORE alike, since they all cover the same
  real tiles. `None` for a network's per-operation trace, which has no tile grid.
- `deploy.py`: `Deployment` gained `array_rows`/`array_cols`/`k_slices`/`tiles_per_ks` — the
  tile-grid numbers `deployment_of` already computes locally, kept alongside the listing the same
  way `stage_lines` was (D41). All 0 for a network's `"operations"` listing.
- `plot_pipeline.py`: `_flow_spans` passes `tile_start`/`tile_end` straight off each `Span`;
  `write_animation_html` builds a `geometry = {m, n, k, rows, cols, k_slices, tiles_per_ks}` object
  (`None` when `listing.kind != "tiles"`) and threads it into `render_animation`.
- `dataflow_html.py`: a new SVG panel + persistent dark caption (reusing `#tip`'s look, but
  always-visible and updated every frame rather than hover-only, like the D41 code pane). A lights
  on a `load_a` event, B on `exec`, C on `store` — each tied to whichever operand that instant
  genuinely touches. A frame whose step coalesces many real tiles (e.g. the 65k-tile
  `4096,4096,4096` example) shows the true multi-cell range it covers, never a fake single index.

Verified: `matmul 4,4,32` on `a100_80gb` (2 k-slices, 1 n-tile-per-slice, hand-countable) produces
the expected `geometry` object and per-frame ranges; `matmul 4096,4096,4096` shows real multi-tile
ranges and the grid-cap note firing (256 k-slices/n-tiles, stride 7); `--encoder` gets
`geometry: null` and the panel hides itself, unaffected. New golden tests:
`test_span_tile_ranges_partition_the_real_tiles_without_gaps_or_overlap`
(`tests/unit/test_pipeline.py`, an 8192-cubed Metis shape checking every step's range is 4 tiles
wide, the 64 ranges partition `[0, 256)` with no gap or overlap, and each k-slice opening's derived
index lands on 0..15 in order) and an extension of
`test_the_listing_quotes_the_tile_and_wave_counts_it_computed` (`tests/unit/test_deploy.py`)
asserting the new `Deployment` geometry fields. The generated pages were also executed end-to-end
in a headless DOM (jsdom) — not just syntax-checked — confirming no runtime errors and sane caption
output across the hand-countable, coalesced, and network cases, since a rendered SVG can't be
eyeballed headlessly otherwise.

### D48 addendum — two follow-up fixes from user review (2026-08-25)

User feedback on the shipped panel, tested across `--a-strategy`/`--b-dataflow`:

1. **The highlight looked "off grid."** A coalesced step's B highlight often spans more than one
   k-slice row (e.g. `4096,4096,4096` on `a100_80gb`: `tiles_per_ks = 256 < units = 432`, so even a
   *single* wave's 432 arrays routinely cross a k-slice boundary) — correct, but the two segments'
   edges fell between the coarse, stride-capped grid lines with nothing marking where they actually
   were, reading as a jagged, unexplained blob rather than "two rows, each partially filled."
   Fixed by drawing explicit boundary lines at the *exact* edges of whatever is currently
   highlighted, regardless of the display stride, and giving each highlight segment a
   `var(--surface)` border so adjacent segments read as distinct cells with a visible seam instead
   of one merged shape. A second bug surfaced while testing `--a-strategy whole`: the WHOLE ramp
   legitimately lights every one of 256 k-slices at once, and the caption spelled out all 256
   indices verbatim (`A(:,0,1,2,...,255)`) — added `formatIndexRanges`, collapsing consecutive
   indices into `start..end` (`A(:,0..255)`), reused for every operand's caption text.

2. **The dark info box "still refers to tiles without sizes."** B's label already carried its own
   size (`"B tile 16x16 [...]"`), but A's k-slice label did not — `"A k-slice 1/125"` said which
   slice, never how big one is. This tooltip is shared verbatim between the timeline and the
   animation (`plot_pipeline.py`'s `_tip`, which only echoes `Span.label`), so the fix belongs in
   `_tile_trace` (`analysis/pipeline.py`) where the label is built: both the per-k-slice `stage`
   label and the `whole` ramp's label now name A's tile size — `"A k-slice 1/256 (4096x16) —
   staged once..."` and `"A staged whole  256 k-slices before wave 0 (4096x16 each)"` — using
   `attrs.m`/`rows`, already in scope, no new plumbing.

Verified across all six strategy combinations (`--a-strategy stage/stream/whole`,
`--b-dataflow write-ahead/on-demand/persistent`) via a jsdom harness driving real playback, not
just a syntax check: no runtime errors, and each generated caption inspected by hand — including
confirming `stream`'s LOAD_A intentionally falls back to B's own label (no k-slice structure to
name under a per-tile re-fetch, D31), which is pre-existing behavior, not a regression. Extended
`test_a_is_staged_once_per_k_slice_in_the_trace` with the new label-size assertion.
