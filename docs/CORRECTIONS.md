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

Ships with `profiles/models/single_layer_encoder.yaml`: one layer, hidden 8, 2 heads of 4, FFN 16,
vocab 16, tied embeddings, ReLU FFN. Small enough that every figure is a product of two small
integers — **664 parameters** and **5280 operations** over 4 tokens — with the full derivation in
the profile's own header and in `tests/unit/test_single_layer_encoder.py`, which recomputes both from the
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
