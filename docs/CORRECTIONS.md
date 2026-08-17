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
