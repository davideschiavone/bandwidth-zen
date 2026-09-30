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
bwz matmul -M 4096 -N 4096 -K 4096 --chip a100_80gb --dtype int8 --c int32    # int32 accumulator
bwz matmul -M 4096 -N 4096 -K 4096 --chip a100_80gb --dtype fp16 --c fp32     # all float
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

### D48 addendum 2 — highlight legibility, and the shared hover learns tile-grid notation (2026-08-25)

User feedback on a real, non-toy shape (`1000,1000,2000` on `a100_80gb`, 125 k-slices, 63 n-tiles
per slice — 432 real tiles per step against those 63): the highlight was barely visible, and the
timeline/animation hover still had no `A(i,j)`/`B(i,j)` notation despite the geometry panel's own
caption already using it.

**Highlight legibility — root cause was a stroke thicker than the cell it outlined.** The previous
fix (D48 addendum 1) gave each highlight segment a `var(--surface)` border to separate stacked rows.
For this shape one k-slice row is `kPx/125 ≈ 1.6` viewBox units tall — the `1.5`-unit stroke was
*thicker than the fill it bordered*, visually replacing the orange with a washed-out pale sliver.
Worse, drawing a full top+bottom boundary line for *every* individual touched row (rather than once
per shared edge between adjacent rows) piled near-identical grey lines on top of each other, reading
as a dense cluster of hairlines rather than a highlighted block. Fixed: removed the highlight's own
stroke entirely (opacity raised to 0.55 to compensate), and deduplicated boundary lines so two
touched rows share one edge instead of drawing it twice.

**The shared hover (`_tip`) gets `A(:,g)`/`B(row,col)`/`C(:,col)` notation.** `Span` gained
`tiles_per_ks` alongside D48's `tile_start`/`tile_end` — B's own geometry (`ceil(N/cols)`), true
under every `a_strategy`, hoisted out of the stage-only branch in `_tile_trace` so every span
carries it, not just A's. `plot_pipeline.py` gained `_tile_segments`/`_format_index_ranges`/
`_index_notation` (Python ports of the geometry panel's own `geoSegments`/`formatIndexRanges` JS —
same decomposition, so the hover and the panel never name a tile differently), spliced into `_tip`
right after `span.label`.

**This surfaced a real, pre-existing labeling gap, not just a missing feature.** Whenever a step's
real-tile window exceeds `tiles_per_ks` — the common case, not an edge case (432 tiles/step here
against 63/k-slice) — a single step opens *several* k-slices, but the label named only the first
(`ks_opened` was a single int). The byte total already (correctly) charged every k-slice opened;
only the text lied by omission — "A k-slice 1/125" when the step actually opened slices 1 through
7. `ks_opened` is now `(first, last)` per step, and the label reads "A k-slices 1-7/125 … feed
their tiles" when they differ, the unchanged singular form when they don't. Also fixed a leftover
in the geometry panel's own JS caption: C's notation still used the pre-fix joined-string pattern
(`C(:,0..1; 0..1)`) after B's had already been split into one `B(row,col)` per segment — corrected
to match B's `C(:,0..1) · C(:,0..1)`.

Verified via the same six-combination jsdom sweep as addendum 1, now also inspecting the geocaption
text at each scrub point for every combination: no runtime errors, no more raw multi-value strings
inside one paren pair. New golden test
`test_a_label_names_every_k_slice_a_step_opens_not_just_the_first` (`tests/unit/test_pipeline.py`)
pins both the exact multi-slice label text and that it agrees with `tile_start`/`tile_end` via
`tiles_per_ks` — never a second, potentially-diverging computation.

### D48 addendum 3 — C's index notation had no k-slice-row dimension to repeat (2026-08-25)

User question on a real hover: `STORE — result C written back` for a 432-tile wave (`1000,1000,2000`
on `a100_80gb`) listed `C(:,0..62); C(:,0..62); C(:,0..62); C(:,0..62); C(:,0..62); C(:,0..62);
C(:,0..53)` — the same column range named six times, plus a subset of it once more.

**Root cause**: addendum 2's `_index_notation`/`geoCaptionHtml` built C's notation by reusing the
same per-k-slice-row segmentation that is *correct* for B (B genuinely has a 2-D grid, row and
column both real), but wrong for C — a result tile is the full M height x one n-tile's width, with
no k-slice-row dimension at all. Several of a wave's tiles landing on the same C columns from
*different* k-slice rows read as "6 things happening" when there is one C region receiving 6
partial-sum contributions.

**Fix**: added `_merge_column_ranges` (`plot_pipeline.py`) / `mergeColumnRanges` (`dataflow_html.py`
— same merge-overlapping-intervals logic in both, keeping the hover and the geometry panel
consistent) that collapses C's touched columns into their union before formatting or drawing,
instead of one entry/rect per underlying row segment. `0..53 ⊂ 0..62`, so the true answer is the
single range `C(:,0..62)` — also fixes a real (if minor) rendering defect: the geometry panel used
to stack up to 7 identical semi-transparent rects at the same position, alpha-blending into a
darker patch that implied emphasis nothing in the model actually assigns there.

Verified directly (`_index_notation` on the exact reported span now returns `"C(:,0..62)"`) and via
the same jsdom sweep as addendum 2, scanning every 25 scrub-ticks across `--a-strategy`
stage/stream/whole and Metis `--b-dataflow on-demand`: no runtime errors, and never more than one
`C(:,...)` entry active at once in any case. A/B are unaffected — their row index is real
information, not an artifact of the segmentation.

### D48 addendum 4 — B's own size label was drawing off the top of the panel (2026-08-25)

User report on `matmul 1000,2000,3000` (`a100_80gb`): A and C's size labels ("A 1000 x 3000",
"C 1000 x 2000") were visible, B's was not.

**Root cause**: A's and C's labels sit in the gap between B and A/C (`y = aY - 5`/`cY - 5`, both
comfortably positive), but B sits at the very top of the diagram with nothing above it
(`bY = 0`), so its own label at `y = bY - 5 = -5` drew *above* the viewBox's top edge — off-canvas,
never rendered, not merely small or faint. Confirmed directly: the SVG held all three `<text>`
elements, but B's had a negative `y`.

**Fix**: added a small `TOP_MARGIN` (16 units) above B, shifting every other y-coordinate down by
the same amount so the whole diagram's relative geometry is unchanged — B's label now draws at
`y = 11`, comfortably inside the panel, the same way A's and C's already did.

Verified across the reported shape and two others already used as regression cases (`4,4,32`, the
hand-countable example, and `4096,4096,4096`, the equal-dimension case): all three labels render
with a positive `y` in every case.

### D48 addendum 5 — the multi-k-slice label was off by one, in the direction it can't sanity-check itself (2026-08-25)

User caught it by hand: for `matmul 1000,2000,3000` on `a100_80gb`, one hover read "A k-slices
1-4/188 … 95.7 kB". One A k-slice is `1000 x 16 x 2 B = 32 kB`; 4 of them should be ~128 kB, not
95.7 kB. The bytes were right — 95.7 kB is exactly *3* slices' worth — the label (addendum 2's own
fix) was wrong, and wrong in the one direction a byte-total sanity check would catch, which is
exactly how the user found it.

**Root cause**: addendum 2 computed the label's `last` k-slice as the block the step's *last tile
merely lands in* — `(end_tile - 1) // tiles_per_ks`. But `openings = end_tile // tiles_per_ks -
open_tile // tiles_per_ks` (the quantity the byte total, unchanged and always correct, actually
uses) counts blocks this step *finishes*, not blocks it *touches*: consecutive steps' `[start,
end)` tile windows are contiguous, so `start//w` telescopes exactly — step *i*'s window
`[start_i//w, end_i//w)` and step *i+1*'s `[end_i//w, end_{i+1}//w)` share the boundary `end_i//w`
with no gap and no overlap. The block a step's last tile lands in straddles into whichever *later*
step's own end actually crosses out of it — that later step is who finishes it, and whose bytes
are charged for it. Naming it in *this* step's label too was double-naming a block across two
consecutive steps, while the bytes were only ever charged once (correctly) — the fix is `last =
end_tile // tiles_per_ks` (drop both the `-1` on `end_tile` and the `+1` at the end; they don't
cancel the way addendum 2 assumed).

**The same wrong (`_tile_segments`-based) decomposition had leaked into the `A(:,g)` index
notation too** — both `plot_pipeline.py`'s `_index_notation` and the geometry panel's own `aRows`
collection (`dataflow_html.py`) computed A's touched k-slices from the *raw* tile range, the same
"touches vs. finishes" conflation, silently agreeing with the *old* wrong label (both said "4")
while *disagreeing* with the correct bytes. Both now compute A's range directly as `[start //
tiles_per_ks, end // tiles_per_ks)` — the same window the label and the bytes already agree on —
bypassing the generic segmentation entirely for `Stage.LOAD_A` (that segmentation remains correct
for B/EXEC and C/STORE, which genuinely span multiple rows/columns in true parallel within one
step — a different physical claim, not subject to this single-owner byte-attribution rule at all).

Rewrote `test_a_label_names_every_k_slice_a_step_opens_not_just_the_first` to assert the
*invariant*, not just two hand-picked corrected strings: for every `LOAD_A` span, the label's own
slice count (`last - first + 1`) must reconcile with `bytes_moved / (one slice's bytes)` exactly.
Confirmed this version fails against the (already-committed) addendum-2 code — `assert
a_spans[0].label == "...1-6/125..."` raised on `"...1-7/125..."` — and passes after. Re-verified
across `--a-strategy stage/stream/whole` and Metis `--b-dataflow on-demand` via the same jsdom
sweep: no runtime errors, and the label/index/bytes triple now agrees in every hover checked by
hand, including the `whole`-strategy ramp (still correctly `A(:,0..187)`, all 188 slices at once).

### D48 addendum 6 — A's per-k-slice bytes were a fleet-wide average, not the true per-slice cost (2026-08-25)

User caught it by hand a second time, this time in the physics itself, not just the label: for
`matmul 1000,2000,3000` on `a100_80gb`, a hover named 3 full k-slices (1-3/188) and charged 95.7 kB
— but `1000 x 16 x 2 B = 32 kB` per slice, and 3 of them is 96 kB (93.75 KiB), not 95.7 kB.

**Root cause**: `ceil(3000/16) = 188` k-slices of 16 rows would need `188 x 16 = 3008` rows of K,
but K is only 3000 — the *last* k-slice is narrower (8 rows, not 16). `_tile_trace` charged every
step `result.dram_activation_read_bytes * openings / k_slices` — a **uniform fleet-wide average**
(`6 000 000 B / 188 ≈ 31.9 kB` per slice) applied to however many slices a step's `openings` count
said it finished, regardless of which *specific* slices those were. That average silently
under-charges the 187 full-width slices and over-charges the one ragged slice — small in total
(the sum across all steps is still exactly right, since the average telescopes to the true total),
but wrong for any *individual* step, and it's individual steps the hover names.

**Fix**: replaced the uniform average with an exact per-slice weight. `bytes_per_element =
result.dram_activation_read_bytes / (attrs.m * attrs.k)` (dtype-agnostic — works out to 2 for fp16,
1 for int8, etc., without a lookup table), and slice `g`'s true width is `min(rows, attrs.k - g *
rows)` — `rows` for every slice except the last, whatever remains for that one. A step's charge is
now the sum of its own specific slices' true byte counts, computed directly from the `(first, last)`
window addendum 5 already derived (correctly) for the label — one source of truth for both, not two
formulas that can drift apart the way the label and the old byte average just did.

Verified: `matmul 1000,2000,3000` on `a100_80gb` — the first LOAD_A event (k-slices 1-3, all
full-width) now charges exactly `96 000 B` (`93.75 KiB`, matching the user's own hand calculation
exactly); the *last* event (k-slices 187-188, the ragged pair) charges `48 000 B` = one full slice
(`32 000 B`) plus the narrow 8-row slice (`16 000 B`). The trace-wide total is unchanged and still
exact: `sum(bytes_moved) == 6 000 000 B == op.dram_activation_read_bytes`. Re-verified a
well-fitting shape (`8192,8192,8192` on Metis, `512 | 8192` evenly) is bit-for-bit unaffected: every
one of its 16 k-slices still charges exactly `8192 x 512 x 1 B = 32 768 B`, since the exact and
uniform-average formulas coincide when nothing is ragged. New golden test
`test_a_k_slice_bytes_are_exact_not_a_fleet_wide_average` (`tests/unit/test_pipeline.py`) pins both
exact byte counts by hand; confirmed it fails against the pre-fix code
(`95744.68... != 96000.0`) before the change and passes after.

**Aside, not acted on**: `format_bytes` (`units.py`) is deliberately decimal/SI (`kB = 1000 B`),
documented as "the datasheet convention" — matching how chip vendors publish GB/s and TB/s figures
throughout this tool. The apparent "93.75 vs 95.7" gap the user first saw was *two* things at once:
this real byte-accounting bug (now fixed), and simply computing by hand in binary KiB
(`96 000 / 1024 = 93.75`) against a tool that displays decimal kB (`96 000 / 1000 = 96.0`) — once
the underlying byte count is exact, both conventions agree it's `96 000` bytes; they just print it
differently. Not a bug, and not changed.

## D49 — a "Next" button steps the animation through transitions one at a time (2026-08-25)

User request: read the schedule by hand, one event at a time, instead of only continuous playback
or manual scrubbing.

Added a `Next` button (`dataflow_html.py`) beside `Play`/`Reset`. Steps to the next moment the
active-event set actually *changes* — the next distinct span start or end across `DATA.flow`
(`BOUNDARIES`, computed once, sorted and deduplicated) — not merely the next individual event's own
start, so a wave's simultaneous `load_b`/`load_a` starting together (double buffering) is one step,
not two. Pauses playback if running; stays at the trace's end once no boundary remains ahead. No
new data — pure client-side navigation over the same `start_s`/`end_s` values play/scrub already
use.

## D50 — the tile-address notation gets its own explanation in the page (2026-08-25)

User question: `432 B tiles ... B(0,0..124); B(1,0..124); B(2,0..124); B(3,0..56)` reads as only 4
tiles named, not 432 — because nothing in the page ever explained that a comma-range names *every*
tile in a row, or that B's addressing is row-major with `tiles_per_ks` tiles per row (so 432
consecutive tiles span only 4 rows, not 432). The math was already right (D48/its addenda); only
the page never said what the notation itself means.

Added a paragraph to both pages' existing hint text (`dataflow_html.py`'s and
`timeline_html.py`'s — both share `_tip`'s hover text, so both need it): `B(row,col)`/`C(:,col)`
number tiles row-major within B's tile grid, `A(:,g)` has no column at all (A is 1-D, one k-slice
wide, the whole M height), and a comma-range like `0..124` names every tile in that row, not one —
worked through the reported example by hand (`125 + 125 + 125 + 57 = 432`) so a reader can verify
it themselves rather than take it on faith.

## D51 — the geometry panel's own rectangle labels didn't say which number is M, N, or K (2026-08-25)

User report: labels like "A 1000 x 3000" don't say which of the two numbers is M and which is K —
a reader has to already know the convention (A is M x K) to map the label back onto the M/N/K
vocabulary the rest of the tool uses.

Fixed in `dataflow_html.py`'s `drawGeometry`: the three rectangle labels now read "A M=1000 x
K=3000", "B K=3000 x N=2000", "C M=1000 x N=2000" — naming the dimension inline rather than relying
on the reader to remember which axis is which from position alone.

## D52 — The M-tail was a systolic pipeline the tensor cores do not have (2026-08-27)

User challenge, after I claimed in an example script that an NVIDIA tensor core streams one row of
A at a time: *"is the Tensor Core from nvidia processing only 1 ROW of matrix A?"* It is not, and
the claim was wrong — but the same wrong picture was baked into the cost model, which is the part
that matters.

**What real tensor cores do.** They issue fixed matrix-matrix instruction tiles — for `.f16` on
Ampere the shapes are `m8n8k4`, `m16n8k8`, `m16n8k16` ([PTX ISA §9.7.15][ptx]), NVIDIA describing
3rd-gen tensor cores as having "a larger base matrix size" ([Ampere Tuning Guide][amp]). M and N
slice into *independent* output tiles dispatched to different cores in parallel; only K is a
sequential accumulation. So **M is a spatial dimension**, and the only M-side loss is the ragged
last tile — the rule of multiples, not a pipeline fill.

**What the model charged instead.** `systolic_utilisation`'s non-bit-serial branch used
`M/(M+rows)`, documented as "a weight-stationary array's pipeline fill and drain" (D24). D34 had
already found this reading wrong for the Metis crossbar and split the function in two — and stated
the principle outright: *"The 1/513 is area — one active row of 512 — not a pipeline drain, and
conflating the two is the bug."* But D34's selector was the **dtype multiplier**, a proxy for "is
it bit-serial", and a tensor core is neither bit-serial nor a systolic pump, so the proxy misrouted
it to the pump branch. D52 finishes what D34 started.

**Why no third branch** (the user rejected one, correctly). Every non-bit-serial unit in
`profiles/chips/` is a 16×16 MMA core — A100, H100, Jetson Orin, MI300X. The repo ships **no**
conventional systolic pump, so the pump branch had no chip to serve: making it the MMA branch adds
nothing and removes a model of hardware that is not here. The two branches are now MMA (pad every
axis) and bit-serial crossbar (pad M and N, sub-cycle fill on K), differing only on K.

**Scope.** `roofline.py:224` is the only production call site and always passes `machine.dtype`, so
bit-serial chips keep the crossbar branch untouched; the change lands exactly on the MMA chips. The
no-dtype path is test-only, and the tests that used it to model Metis/chip_a now pass the dtype and
exercise the branch those chips really run.

**Verified per chip, not argued.** Different chips genuinely have different rules, so which ones
moved was measured (`bwz matmul -M 16 -N 4096 -K 4096 --ideal`, shape utilisation, D52 reverted vs
applied):

| chip | dtype | before | after | branch |
|---|---|---:|---:|---|
| `a100_80gb` | fp16 | 49.90% | **99.81%** | MMA — NVIDIA tensor core ([PTX ISA][ptx]) |
| `h100_sxm` | fp16 | 49.65% | **99.30%** | MMA — NVIDIA tensor core |
| `jetson_orin` | fp16 | 50.00% | **100.00%** | MMA — NVIDIA (Ampere) tensor core |
| `mi300x` | fp16 | 49.90% | **99.81%** | MMA — AMD Matrix Core, MFMA ([AMD][mfma]) |
| `chip_a` | int8 | 3.12% | 3.12% | crossbar — unchanged |
| `chip_b` | int8 | 3.12% | 3.12% | crossbar — unchanged |
| `metis_aipu` | int8 | 3.12% | 3.12% | crossbar — unchanged |

The change is therefore **not NVIDIA-only**: it also moves AMD's MI300X, and that is correct rather
than incidental — AMD Matrix Cores issue fixed-shape `MxNxK` MFMA instructions
(`__builtin_amdgcn_mfma_CDFmt_MxNxKABFmt`, "`M`, `N` and `K` are matrix dimensions"), the same
instruction-tile picture as NVIDIA's MMA, so the rule of multiples applies identically. No
bit-serial chip moved by a single digit.

**Effect** — worst in mid-M, which is where real decode batches sit:

| M (rows=16) | before `m/(m+16)` | after `m/pad(m,16)` |
|---:|---:|---:|
| 1 | 0.059 | 0.063 |
| 16 | 0.500 | **1.000** |
| 128 | 0.889 | **1.000** |
| 2048 | 0.992 | 1.000 |

Llama-3-8B on H100 barely moves (prefill 44.7 → 44.4 ms, util 67.1% → 67.7%, still inside
CLAUDE.md's 40–70% band; decode unchanged at 165 tok/s, being DRAM-bound). Both `M=1` sanity checks
now hold *exactly* rather than approximately: 1/128 on a 128×128, 1/512 on a 512×512. The
`bwz matmul` derivation line no longer says "systolic tail", which would now name the wrong
mechanism. `docs/MODEL.md` §6.1 rewritten; goldens in `test_tiling.py` and `test_matmul_workload.py`
updated, including one whose prose ("wastes 15 of them") had always described `1/16` while the
formula returned `1/17`.

**Not done, flagged only:** instruction-tile granularity is dtype-dependent on real hardware (INT8
uses `m16n8k32`, so K quantises to 32, not 16), and `HardwareSpec.dataflow` is declared in every
profile and read by nothing — it would be the honest selector if a genuine systolic array is ever
added.

[ptx]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
[amp]: https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
[mfma]: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores/README.html

### D52 addendum — the loop nest narrated what the cost model had stopped believing (2026-08-27)

User check after D52 landed: *"did anything on performance or scheduling change? cause I don't see
it in the animation."* It had — but the shape under inspection hides it, and probing that surfaced
a real leftover.

**Why it looked unchanged.** D52 alters compute rate, not tile geometry, so span *counts* are
identical and only *durations* scale (verified: 152 exec spans before and after). And at
`1000x2000x3000` M pads 1000 → 1008 instead of 1016, a 0.8% shift: 39.6 → 39.3 µs, invisible. At
M=16 the same pipeline halves, 3.450 → 1.725 µs in the animation's own span data, so the change
does propagate — the shape was simply the wrong place to look.

**The real leftover.** `deploy.py` still printed, for every A100 listing:

```
 * B is cut into 16x16 tiles and held by the array; M streams past it.
       for (int m = 0; m < 1000; ++m)   /* M streams; it never tiles */
```

Both claims are the weight-stationary picture D52 removed from the cost model, and the residency
claim already contradicted D30 ("the array stores no weights: both operands are re-read per
instruction"). The numbers were right and the narrative was not.

**Fix.** An MMA unit — full-rate *and* holding no resident weights, the same partition
`analysis/tiling.py` makes — now gets a listing built from instruction tiles: a `MTILES =
ceil(M/ROWS)` define, `for (int mt = 0; mt < MTILES; ++mt)`, and `mma(u, &A[mt * ROWS][...])`
naming the `16x16x16` tile, with the residency sentence dropped. Metis keeps the streaming form
verbatim — its weight sets *are* resident, so "held by the array; M streams past it" is literally
what happens there. Guarded by
`test_an_mma_unit_issues_instruction_tiles_and_a_resident_array_streams_m`, confirmed to fail
against the previous listing.

**Still open, deliberately not changed here.** `tile_count` remains `ceil(K/rows) · ceil(N/cols)`
— a split-K decomposition, one unit per (k-slice, n-tile) — so partial sums across k-slices need a
cross-core reduction the model does not cost. A real cuBLAS GEMM instead tiles the *output*
`ceil(M/rows) · ceil(N/cols)` and keeps K sequential inside each tile, needing no reduction.
Changing that moves tile counts, wave occupancy and every trace built on them, so it is a separate
question from making the listing honest about the tiling the model actually performs.

## D53 — Dataflow stationarity becomes a real, per-chip, selectable strategy (2026-08-27)

D52's closing paragraph named what was still wrong: `tile_count` was
`ceil(K/rows) · ceil(N/cols)`, one unit per (k-slice, n-tile), M streaming past a resident B tile.
That is **weight-stationary split-K**, and it is the wrong shape for every matrix core. cuBLAS and
CUTLASS accumulate K **in registers inside one output tile** — the accumulator elements *"typically
occupy at least half a thread's total register budget"* — so nothing crosses cores. Split-K is the
documented *exception*, used when *"there are too few threadblocks to efficiently occupy the entire
GPU"*, and it costs **two kernels**: *"partitionedK GEMM, and batched reduction"*. So the engine was
always split-K, the wrong shape in the common case, and never charged the reduction it implied.

`Dataflow` had declared `ws`/`os`/`rs` since M1 and its docstring promised *"v1 honours the
profile's declaration"*. Nothing read the field.

### The abstraction

One place decides the decomposition: `analysis/stationarity.py`, whose `grid_for(stationarity,
attrs, rows, cols, k_partitions)` returns a frozen `TileGrid`. For a matmul the operands map as
**input = A, weight = B, output = C**.

| stationarity | resident | parallel grid | swept per tile | partials cross cores | partials reach DRAM |
|---|---|---|---|---|---|
| `os` output-stationary | C (accumulator) | `ceil(M/rows) × ceil(N/cols)` | K | no | no |
| `os` + `--split-k p` | C | that grid, `× p` | K/p | yes | **yes** |
| `ws` weight-stationary | B tile | `ceil(K/rows) × ceil(N/cols)` | M | yes | no |
| `is` input-stationary | A tile | `ceil(M/rows) × ceil(K/rows)` | N | yes | no |
| `rs` row-stationary | one A row per PE | `ceil(M/rows) × ceil(N/cols)` | K, spread spatially | local to the array | no |

The grid carries the matmul's own extents and the array's, so it can answer every question the
trace, the listing and both renderers used to answer for themselves: how many tiles there are,
which of M/N/K each axis is, which one a tile sweeps, where one flat tile index sits (`decode`),
how many A staging events there are and how big each is (`a_events`, `a_event_elements`).

**What is invariant is not the k-slice.** D33's "A is staged once per k-slice" is a special case of
"the tiles of one grid **row** all read the same slice of A" — which holds under every
stationarity, because A's dimensions are M and K and the grid's column axis is never one of those
except under `is`, where a row's tiles between them read the row's band exactly once anyway. So A
crosses DRAM exactly once under every grid, in `a_events` pieces; what changes is whether a piece
is a slice of K (`ws`) or a band of M rows (`os`). Same for the hover notation: an operand's index
along a dimension is the row number if the grid carries it on rows, the column range if on columns,
and `:` if the tiles sweep it. That one rule replaced every hardcoded `A(:,g)`/`B(row,col)`/`C(:,col)`
in `plot_pipeline.py` and in the animation's geometry-panel JS.

`rs` is implemented and **explicitly unvalidated**: defined for GEMM after Eyeriss (Chen/Emer/Sze,
ISCA 2016) as one A row per PE with K spread across the array's own columns, so partial sums reduce
inside the array. No shipped profile declares it, and `report.assumptions` says its numbers are
unvalidated rather than pretending otherwise.

### Refusing, not clamping

`ComputeUnit.supported_dataflows` defaults to `(dataflow,)` — the honest default, since a tensor
core has no weight storage (D30) and an in-memory array's weights *are* its memory. A request
outside the set returns `Report(feasible=False)` naming the field, the request and the chip's real
capability. This is a deliberate departure from `plan_dataflow`'s clamp convention: `stage`/`stream`/
`whole` are orderings of the same work, so falling back still answers the question asked, while a
stationarity is a different decomposition and substituting one would report a number for hardware
the caller never asked about.

### The reduction, and where this diverges from the approved plan

The plan proposed charging a DRAM round trip wherever `needs_reduction` — including `ws`, with the
spilled portion discounted by on-chip capacity. **That is not implemented, on purpose**, and the
reasoning is worth recording because it cuts against a plan the user had already approved.

Two things turned out to be true.

**The additions are not new arithmetic.** Accumulating K products into one output is `K − 1`
additions however the contraction is cut: `(K/p − 1)·p + (p − 1) = K − 1` for every `p`. `2·M·N·K`
already counts them (CLAUDE.md #5). What split-K changes is *where* they run — `(p−1)·M·N` of them
leave the matrix engine's accumulator for the vector unit, 16× slower on A100 (D27). So the cost is
the relocation, not the count, and that is what `partial_sums` is charged as: vector time on top,
double-counting only the same adds at the matrix rate, 1/16 of what it adds.

**Only split-K materialises the partials.** Under a K-on-grid grid the same unit comes back to the
same output cell on a later wave, so the partials meet in an accumulator, and D5a gives this model
no on-chip bandwidth term to charge that against. What it costs is *capacity* — the whole `M×N`
output is live under a K-outer walk — and where that does not fit, a real compiler re-blocks the
output and re-reads A and B rather than spilling C. Those re-reads are exactly the traffic
`docs/MODEL.md` §6.2 already declines to model: *"DRAM traffic is compulsory traffic… so a DRAM-bound
latency here is a lower bound."* Charging a C spill while that stands would price one horn of the
dilemma and not the other. On Metis at 8192-cubed it would have added ~8.5 GB against the GEMM's
own 201 MB — a 40× move on a chip nobody has measured, decided by a loop order the model does not
even claim to choose. So `TileGrid.materialises_partials` is `k_partitions > 1` alone, and every
K-on-grid grid gets an assumption naming its accumulator working set and stating that neither its
spill nor the re-reads that avoid it are charged.

Split-K itself is charged in full, because CUTLASS genuinely runs two kernels and its partials
genuinely live in global memory between them: `p · M · N · acc_bytes` written and read back on
`OpResult.dram_reduction_bytes` (kept out of `dram_read_bytes`/`dram_write_bytes`, which are the
*operands'* traffic, so the three still sum to `dram_bytes`), `(p−1)·M·N` adds on the vector unit as
`t_reduce_s`, and one extra dispatch. The trace draws it as its own `Stage.REDUCE` spans after the
tile schedule ends, and `deploy.py` prints it as a second listing block.

### What moved

| check | before (`ws`) | after (`os`) |
|---|---|---|
| A100, M=1, N=K=10000, tiles | 390 625 | 625 |
| … wave occupancy over 432 cores | 0.99923 (905 waves) | 0.7234 (2 waves) |
| … chip utilisation | 6.24% | 4.52% |
| A100, 1000×2000×3000, tiles | 23 500 | 7 875 |
| … waves | 55 | 19 |
| Llama-3-8B prefill on H100, util | 67.65% | 67.26% |
| … decode t_compute | 375 µs | 641 µs |
| … TPOT | 6.04 ms (165.4 tok/s) | unchanged |

**Utilisation gets *worse* on the flip, and that is correct.** Total MMA instructions are identical
— the same `M·N·K` MACs, only quantised differently — so `os` reports slightly lower utilisation
while being the more accurate model *and* avoiding a reduction the old model never charged. The
parallelism `ws` claimed was not free; it was 390 625 partial sums.

`--split-k` now does something on these chips. At 512×512×4096 on A100 it takes wave occupancy from
0.79 (1024 tiles, 3 waves) to 0.998 (8192 tiles, 19 waves) and charges 8.4 MB of partials against
the 4.7 MB the un-split GEMM moves in total. Exposing that trade is the point of the flag.

The array-level goldens are untouched: `systolic_utilisation(1, …, 16, 16)` is still exactly `1/16`,
and CLAUDE.md's `M=1 → ≈1/rows` check is a statement about one array's geometry, which no
stationarity changes. INT8 remains exactly 2× FP16.

### Vocabulary that changed

`NTILES_PER_KS` → `TILES_PER_GROUP` and `KSLICE(w, u)` → `GROUP(w, u)` / `COL(w, u)` in the listing;
`Span.tiles_per_ks` → `Span.grid`; `pipeline.ntiles_per_kslice` → `tiles_per_a_event`;
`DataflowPlan.k_slices`/`ntiles_per_ks` → `a_events`/`tiles_per_a_event`;
`Deployment.k_slices`/`tiles_per_ks` → `Deployment.grid`. `deploy.py`'s `is_mma` became
`issues_instruction_tiles`, which is what it always meant: whether the *swept* dimension is walked
in whole instruction tiles (D52) or element by element — a question about the unit, now cleanly
separate from which dimension is swept, which is a question about the stationarity.

**Not done, flagged only:** instruction-tile granularity is still dtype-independent (D52's own open
item); the loop *order* within a grid is not modelled, so the accumulator working set quoted in the
assumptions is the K-outer worst case; and auto-selecting the best stationarity or split-K factor
remains M8 work — the flags select, they do not search.

[cutlass]: https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md
[eyeriss]: https://people.csail.mit.edu/emer/papers/2016.06.isca.eyeriss_architecture.pdf

---

## D54 — The pseudo-C becomes a program that runs, counts and checks itself (2026-08-28)

`deploy.py` prints a pseudo-C loop nest, and `deploy.check` asserts its constants against the
schedule the figure drew — as far as a *listing* can go. It cannot be run, so nothing verified that
the decomposition it narrates computes a matmul at all, and nothing verified that the bytes the
roofline charged are the bytes such a schedule would really move. The user's verdict on it: *"the
pseudo code that is useless and too high level"*.

`bwz matmul --emit PATH` (and `plot_pipeline.py --emit`) now writes a self-contained Python program
that walks the same tile grid, stages A on the same events, hands tiles to cores by the same
`wave · used_cores + core_id` rule, **counts what it moves**, and asserts those counts against a
`PREDICTED` block that is the report's own numbers. `docs/CLI.md` §2.6 has the output;
`docs/MODEL.md` §6.8 has what it does and does not model.

Two things follow that a listing could not give. It is a better teaching artifact — the goal of this
project is that a reader can see how a workload maps onto an architecture, and a program they can
run, edit and break shows that better than a description of one. And it is an **executable
specification of the cost model's own bookkeeping**: when the program and the report disagree, one
of them is wrong, and until now there was no way to find that out.

### The design decisions, and why

**Real Python, kernels only.** A whole model would be a program nobody can run on this machine, so
`--emit` is refused for `--model`/`--encoder` with a message saying that.

**Constants first, each with its provenance.** The shape from the command line, the chip from its
profile, the strategy, then the grid those imply. A unit test asserts that no constant in that
region is written without a trailing comment naming where its value came from — the only way a rule
like that survives.

**Two core counts, and one OS thread per modelled core.** `AVAILABLE_CORES` is `unit.count`;
`USED_CORES` is `min(AVAILABLE_CORES, TILES)`. The gap between them is the whole of wave occupancy.
An earlier draft capped the thread count at `os.cpu_count()`; that was rejected, and the reason
generalises — it optimises for a speedup nobody asked for, invents a modelled-core-to-host-thread
mapping with no counterpart in the model, and demotes `USED_CORES` to decoration. 432 threads is
unremarkable on Linux and the largest shipped profile is H100's 528; the harness warns above a few
thousand and does not cap.

**Persistent threads and a `threading.Barrier`, not a pool.** A pool is a task queue;
`waves = ceil(tiles / units)` is *lockstep*, and a core is a persistent thing that takes one tile
per wave. The `if tile < TILES` inside `run_waves` **is** wave occupancy (D30), executable.

**`ws` needs a lock, and `os` does not — which is the finding.** Under `ws` two tiles `(kt, nt)` and
`(kt', nt)` accumulate into the same output block and the wave scheduler can hand them to two cores
at once. The emitted program takes **a lock per output block**: an atomic on-chip accumulate,
costing no bytes, which is exactly what the cost model charges for it. Per-core private partials
merged at the end would silently *be* split-K, which the model does charge for. That the program
needs a lock at all is evidence about where `ws`'s partial sums live; under `os` the question never
arises, and that absence is D53's claim in one line of code.

**Operands are dyadic, so the tolerance is zero.** Values are drawn from `{-7..7}`, divided by 8 for
a float format, so every product and every partial sum is exact in the accumulator. A difference is
then a *walk* error and never a rounding one. This deviates from the plan's "fp16 needs a tolerance
against an fp32 reference" deliberately: the program checks a decomposition, and letting rounding
share the same tolerance would make a failure ambiguous. The achieved error is printed either way,
and `_tolerance` still derives a real bound for shapes large enough to need one.

**The harness is a real module, inlined by source.** `bwz/emit/_harness.py` is ruff-clean,
mypy-strict and unit-tested, and the emitter inlines `inspect.getsource` of it, so there is one copy
and it is the checked one. A test asserts it imports nothing from `bwz`; another asserts an emitted
program imports nothing but the standard library. numpy is optional — it is not a `bwz` dependency —
so every operation has a pure-Python fallback, tested by shadowing numpy with an unimportable
module.

### Two tiers, and what tier 2 found

Tier 1 is asserted and matches today on every decomposition the shipped chips declare: `tiles`,
`waves`, useful MACs (`M·N·K` under all four stationarities), MAC slots issued, idle core-waves, A's
staging events, A's bytes, C's bytes, split-K's partial round trip, and `C == A @ B`. Occupancy and
shape padding are pinned as **integers** (`idle_core_waves`, `mac_slots`) rather than as floats, and
the ratios are printed from counts already checked — the same claim without a float comparison.

Tier 2 is printed and not asserted, and it is where the interesting disagreement lives. **B's
traffic has three defensible values and no two of them agree.** At 1000×2000×3000 fp16 on A100:

| | bytes | what it is |
|---|---|---|
| the report charges | **0** | compulsory traffic, discounted by §6.2's residency fraction — B fits on chip |
| first touch, measured | **12 MB** | B crossing the bus exactly once |
| fetched, measured | **756 MB** | what an `os` walk asks for: each of the 63 row-bands of M re-reads all of B |

The 756 MB against 12 MB is the tiling re-read `docs/MODEL.md` §6.2 already declines to model and
names as the reason a DRAM-bound latency there is a lower bound. The 0 is new information: the
weight-residency discount (D8, D15) was written for a network, where a weight held on chip across
many operations genuinely need not be re-streamed, and applying it to a **one-operation graph**
says B never crosses the bus at all — which cannot be true of an operand that starts in DRAM.

That is a model bug, not an emitter bug, and it is deliberately not fixed here: it would move
documented figures in four files and belongs in its own change with its own goldens, exactly as D53
argued. What this change does is make it *visible* — printed side by side on every emitted run,
rather than an unexamined term in a residency formula.

### Not done, flagged only

- **`b_dataflow`, double buffering and sub-cycles are annotated, not executed.** All three are
  claims about *time* — a write's placement, an overlap, a rate — and this program measures counts.
  D54-PLAN proposed a real prefetch queue of `DEPTH`; it would move no byte and change no result,
  so it is written down in the file's own header instead of simulated.
- **`--used-cores N` is descriptive, not a knob.** `USED_CORES` is `min(AVAILABLE, TILES)` and the
  file shows the idle remainder. Making it a user-settable *cap* would change reported latency
  (fewer cores → more waves → lower occupancy) and needs its own goldens.
- **Other kernels.** Attention, convolution and the encoder come later; the harness is shaped so
  they slot in, but one kernel done properly beats three sketched.

### The pseudo-C is retired, in the same change

The plan proposed keeping it and re-targeting the animation later. The user asked for it now, and
the argument for waiting was weak once the Python could carry the same stage tags: two listings of
the same loop nest are not two useful artifacts, and the weaker one could be checked against nothing
but its own constants.

So `EmittedProgram` now carries `stage_lines` — 0-indexed line numbers for `load_a`, `load_b`,
`exec`, `store` and `reduce`, in the vocabulary `plot_pipeline.py` already used — and the animation
highlights **those**. The lines it lights are statements that perform the transfer:
`dram.read_b(k0, k1, n0, n1)`, `mma(acc, a, b, counters, SLOTS_PER_MMA)`,
`dram.write_c(m0, n0, acc)`. `emit.check` asserts the map points at real, non-blank lines and that
all four required stages are present, so an off-by-one is caught statically rather than found by eye
in a browser.

`deploy.py` keeps only the case `emit` has no answer for: a **graph**, which this model runs as a
sequence with no overlap between operations (D5a), so there is no tile grid to walk and nothing to
emit. A matmul on a chip whose fastest unit for the requested dtype declares no array geometry
(fp32 on A100 runs on the CUDA cores) lands there too, and the listing now says *that* rather than
printing "a network is a SEQUENCE here" about a workload that is not a network.

Two consequences worth naming. The emitted file's runtime moved **below** `main()`, against
convention and on purpose: the loop nest is what a reader is here for, and 400 lines of machinery
between the constants and the walk would bury it — which is the mirror of the failure the pseudo-C
had. And the `COUNTERS` global went with it; counters are threaded through `run_tile` explicitly,
which the reordering forced and which is better anyway.

`--emit` consequently changed meaning on `plot_pipeline.py`: the page **always** carries the
program now, and the flag decides only whether the `.py` is also saved beside it.

`tests/unit/test_deploy.py` lost the tile-nest half. Every fact it pinned — the wave counts, the
staged-once-per-group claim, the swept dimension, the bit-serial tax — is still pinned, by a program
that would fail to run if it were wrong rather than by a substring search over prose. Four
pre-existing `mypy` errors in `scripts/plot_pipeline.py` went with the retired `a_strategy`/
`b_dataflow` parameters.


---

## D55 — One command per workload: the figures move onto `bwz` (2026-08-28)

Two entry points drew one workload. `bwz matmul -M 16 -N 16 -K 16 -c a100_80gb` printed the table;
`python scripts/plot_pipeline.py --matmul 16,16,16 --chip a100_80gb` drew the picture. The user's
verdict: *"I want to use `uv run bwz matmul`, else it's confusing"* — and they were right about more
than the spelling. The two had different flag names for the same inputs, different defaults (the
script's `--encoder` defaulted to `-S 512` against the command's `-S 4`, a mismatch `docs/CLI.md`
had to carve out and explain), and **two separate `analyze()` calls** whose `DeploymentSpec`s
differed (`output_tokens: 1` and no phase against `0` and prefill). Identical for a matmul today,
by luck rather than by construction.

Now `--timeline`, `--animate`, `--compare-with`, `--out`, `--steps` and `-q/--quiet` are flags on
`bwz matmul`, `bwz run` and `bwz encoder-layer`, each drawing its own workload from the report it
just printed.

### Where the code went, and the rule that had to change

`scripts/plot_pipeline.py` → `bwz/figures/timeline.py`, with `timeline_html.py` and
`dataflow_html.py` beside it. `git mv`, so the history follows.

CLAUDE.md said *"a figure goes in `backend/scripts/plot_*.py`, never inside `bwz/`"*. The reason,
stated in `pyproject.toml`, is that the engine must never import a plotting library and `make test`
must not pull one in. That reason does not apply to these three files: the timeline and the flow
animation are hand-written SVG and JavaScript, standard library only — D37 removed the last
matplotlib import from them. The rule is now split by what it was actually protecting: **matplotlib
figures** stay in `scripts/`, **self-contained HTML views** live in `bwz/figures/`. The dependency
arrow gains `figures` beside `api`, `cli` and `emit`, all consumers, none of them imported by
`analysis/`.

Keeping the code in `scripts/` and importing it from `cli.py` was considered and is not possible:
`scripts/` is not a package and does not ship in a wheel.

### What this bought beyond the spelling

- **One analysis per run.** `build_matmul` and `build_phases` take the caller's own `Report` and
  `DeploymentSpec` instead of rebuilding them, so the picture and the table above it cannot describe
  different runs. `bwz matmul --stationarity ws --timeline` draws the decomposition the numbers came
  from because there is only one.
- **Tests, for the first time.** Nothing imported `plot_pipeline.py`, so 3200 lines had no coverage
  at all: a broken page was found by opening one in a browser, or not at all.
  `tests/unit/test_figures.py` now pins self-containment (no fetchable reference, no network call in
  any `<script>` — checked over the script blocks alone, because the page also *displays* the
  emitted program, whose `Scratchpad.band(key, fetch)` callback is a false positive for a
  whole-page search), the sections the docs promise, the no-tile-grid fallback, and that the drawn
  trace sums to the reported latency.
- **`mypy`'s configured `files = ["bwz", "tests"]` now covers them permanently**, rather than only
  when someone typed `mypy .`.
- **A pre-existing crash, found by the move.** `bwz matmul -d fp32 -c a100_80gb` raised
  `ZeroDivisionError`: the tile grid was built unconditionally with `*(dims or (0, 0))`, and A100
  runs fp32 on its CUDA cores, which declare no array geometry. It had never worked. The grid is now
  built only where there is an array to tile against — an exception is never an acceptable output
  (CLAUDE.md #8).

### The decisions

**`--compare-with CHIP`, not a repeatable `--chip`.** `plot_pipeline --compare` drew one workload on
N chips; `bwz compare` is a different command entirely, a chips × models table with the prefill
crossover. Rather than overload the word or make `--chip` repeatable on every report command — which
raises the question of what *table* to print for two chips — the primary chip stays `--chip` and the
figure gains the others. The table stays single-chip; only the page compares.

**`--quiet` rather than silence by default.** A report command reports; the figure is extra. But
`make plots` would then print a wall of tables, which is what the two-command split was avoiding, so
`-q` drops the report. It never silences an error: infeasibility and a refused flag combination
print regardless, since they are the answer rather than a table.

**Filenames are unchanged** — `timeline-<chip>-<dtype>.html` and the rest — so links to
`docs/plots/` still resolve and `make plots` produces the same pages. The phase is in the name only
where there is more than one to tell apart, which is why `bwz run` has it and the encoder does not.

### Verified by byte-for-byte reproduction

Ten reference pages were generated from the script before the move — matmul, encoder, model,
comparison, animation, fp32, `--a-strategy stream`, the emitted program — and reproduce **byte for
byte** through the new CLI, modulo the invocation each page records in its own footer. That is the
whole argument that a 3200-line move of untested code changed nothing.

### Not done, flagged only

- **`scripts/plot_roofline.py` stays a script.** It is the one thing that genuinely needs
  matplotlib. Folding it in would mean either an optional import inside the package or a hard
  dependency, and neither is worth it for a PNG.
- **`bwz compare` still prints only a table.** Teaching it to draw the shared-axis page would need a
  whole workload surface added to a command that takes `--models`; `--compare-with` covers the same
  ground from the command that already has one.


---

## D56 — The accumulator width followed a knob, not the arithmetic (2026-08-28)

The user asked: *"are you sure `ACC_DTYPE = "fp32"` in the NVIDIA chips for tensor cores?"*

For **fp16 on a tensor core, yes.** That is what cuBLAS computes by default
(`CUBLAS_COMPUTE_32F`) and the mode a datacenter part's headline fp16 figure is quoted at. fp16
accumulate is a real MMA shape — PTX `mma.sync...f16.f16.f16.f16`, cuBLAS `CUBLAS_COMPUTE_16F`, and
on GeForce parts it is the *faster* path because fp32-accumulate is deliberately half-rate there —
but it is nobody's default, and no profile here declares the two rates separately, so modelling it
would mean inventing a number.

**But the question found a bug, because the emitter printed `fp32` for every dtype.** An `int8`
matmul emitted `ACC_DTYPE = "fp32"` and accumulated in floating point:

```
ACC_DTYPE = "fp32"        # deployment.precision.accumulate — wider than the
                          # operands on purpose: a product of two int8s
                          # does not fit one of them
```

`int8 x int8 -> int32` is the only integer MMA shape any of these arrays issues. `spec/dtypes.py`
has said so since M1 — *"an int8 x int8 product accumulates in int32 at the int8 rate"* — and D54's
own harness docstring claims the emitted code is **"that rule made executable"**. It was not.

### Why nothing caught it

`Precision.accumulate` defaulted to `DType.FP32` unconditionally, and nothing derived it. Until D54
nothing *read* it either — `docs/SCHEMA.md` said in as many words *"`precision.accumulate` is not
read by anything"* — so a field that had never mattered acquired a consumer without anyone checking
that its default was right for the consumer.

The emitted program's own self-check could not see it. D54 draws operands from the dyadic grid
`{-7..7}` precisely so every partial sum is exact, and fp32 holds a sum of small integers exactly —
so `max |C - A@B| = 0` and every tier-1 count matched while the demonstrated *format* was wrong.
The design that made the walk unambiguous also hid a format error, which is worth remembering: an
exactness check is not a fidelity check.

### The fix

`accumulator_for(dtype)` in `spec/dtypes.py`, beside the widths, because it is the same kind of
thing — a definition, not calibration (CLAUDE.md's "a new dtype goes in `spec/dtypes.py` only"):

| operand | accumulates in | why |
|---|---|---|
| `int8`, `int4`, `int32` | `int32` | the only integer MMA shape these arrays issue; an 8-bit accumulator overflows after three or four terms |
| `fp16`, `bf16`, `tf32`, `fp8`, `fp32` | `fp32` | cuBLAS's default; `bf16` and `tf32` have no narrower accumulate on NVIDIA hardware at all |

`Precision.accumulate` becomes `DType | None = None` — *derive it* — and stays available as an
override. `emit_matmul` resolves it from `machine.dtype`, the dtype the arithmetic actually runs at,
which for mixed widths is the wider operand (D18): both enter one datapath and it is that datapath
that accumulates.

The emitted comment now says something different per format, because the two claims have different
force. Saying "a product does not fit one operand" of `fp16` implies fp16 accumulate is impossible,
and it is not.

### What did not change

No byte count and no reported number. Split-K's partials are still sized at **C's** width rather
than the accumulator's — a separate, deliberate choice `roofline._reduction_for` makes and the
emitted file already flags on its own `PARTIAL_BYTES_PER_ELEMENT` line. Whether *that* is right is
its own question; it is not this one.

`tests/unit/test_dtypes.py` pins the table and the property behind it (an integer operand never
accumulates in a float, so a future entry cannot break the rule); `tests/unit/test_emit.py` pins
that an int8 program says `int32` and carries the integer rationale, that a float one says `fp32`
and does *not* claim fp16 accumulate is impossible, and that the override still works.


---

## D57 — Section 3 said what a constant meant, not where it came from (2026-08-28)

The user, reading an emitted program: *"is `SPLIT_K = 1  # one kernel` hardwired when launching
`bwz matmul --m 16 --n 16 --k 16 --chip a100_80gb --ideal --pipeline --emit .`? is this calculated?
can you write in the comment next to it how?"*

**It is not calculated.** It is the default of `--split-k`, a flag that was not passed. Nothing in
this engine searches for a good split factor — D53's own closing list says so: *"auto-selecting the
best stationarity or split-K factor remains M8 work — the flags select, they do not search."* The
old comment, `# one kernel`, was true and answered a different question.

**The inconsistency behind it.** D54 required every constant in the emitted file to carry a comment
naming where its value came from, and `test_every_constant_carries_a_comment` enforces that a comment
*exists*. Sections 1, 2 and 4 named an origin — `-M / -N / -K`, `systolic_dims`,
`count: tensor_core`, `ceil(M / ROWS)`. **Section 3 named a meaning instead**, and section 3 is the
one where origin matters most, because its seven constants come from four genuinely different
places:

| | example | what it is |
|---|---|---|
| a flag you passed | `--split-k 4` | your choice |
| the default of one you did not | `--split-k, not passed` | nobody's choice |
| the chip's own declaration | `tensor_core's OWN declared dataflow` | not yours to pick, only to override |
| a **clamp** of what you asked | `--a-strategy whole did not fit` | your choice, overruled |
| genuinely computed | `DEPTH` | from on-chip capacity |

Reading `SPLIT_K = 1` and `DEPTH = 2` side by side, nothing said that one was an unset flag and the
other a capacity calculation.

### What the section says now

```
SPLIT_K = 1               # --split-k, not passed. NOT computed: nothing here looks
                          # for a good split factor — the flag selects, it does not
                          # optimise (D53). 1 means one GEMM kernel and no reduction
...
DEPTH = 2                 # COMPUTED, not asked for: on-chip capacity fits two
                          # tiles, so the report's latency is max(load, compute)
                          # rather than their sum (D5a). Annotated here, not executed.
```

with a header over the block naming the four origins, so the distinction is stated once rather than
inferred seven times. A clamp names the request it overruled — `--a-strategy whole did not fit and
was clamped to stage` — because a reader whose flag did nothing otherwise has no thread to pull;
the full reason is already in the file's own header, from `DataflowPlan.notes`.

`"--iterations's default"` was the phrasing that killed the possessive form: every default now reads
`<flag>, not passed`, which is both uniform and more direct about the fact that nobody chose it.

### Scope

Comment text only — no value, no count and no assertion changed, and the D54 output block
`docs/CLI.md` §2.6 quotes verbatim is unaffected because it is the *program's* output, not its
source. Three tests were added: that section 3 names its four origins, that a passed flag reads
differently from an unpassed one, and that a clamped knob names what was asked for.


---

## D58 — `DEPTH = 2` is a yes/no, and the comment called it a measurement (2026-08-28)

The user, on a 16x16x16 matmul: *"`DEPTH = 2  # ... on-chip capacity fits two tiles` — why does it
fit only two tiles for such a small matmul?"*

It does not fit two. It fits **118 624**. A100 declares 60.7 MB of L1 and a 16x16 fp16 working tile
is 512 B:

```
on-chip capacity           60,736,000 B
resident weights                  512 B
spare                      60,735,488 B
one working tile                  512 B   (ROWS*COLS*2)
tiles that fit                118,624
```

`report.memory.double_buffered` is a **bool**, and `double_buffering_fits` asks
`spare >= 2 * working_tile` — *at least* two, because two is what overlapping one tile's load with
the previous tile's compute requires. The cost model has exactly two states,
`max(load, compute)` or `load + compute` (D5a); there is no DEPTH 3 to report however much capacity
there is. Rendering the bool as the number `2` and captioning it "capacity fits two tiles" turned a
threshold into a measurement, and invited precisely the question it should have answered.

### What it says now

```
DEPTH = 2   # DERIVED, and a yes/no dressed as a number: the model asks
            # only whether capacity holds TWO 16x16 tiles at once —
            # what overlapping one load with one compute needs — not how
            # many it would really hold, which here is ~118,625 before
            # anything else is resident. There is no DEPTH 3: latency is
            # max(load, compute) or their sum, nothing between (D5a).
            # Not read below; this program counts bytes, not time.
```

The headroom quoted is `ON_CHIP_BYTES` over one `ROWS x COLS` tile — **all three already constants
on the page**, so a reader can check the claim without leaving the file. It is deliberately *not*
the planner's own spare, which subtracts resident weights and activations and which no constant here
exposes; recomputing that in the emitter would be a second copy of `analysis/memory.py`'s arithmetic,
free to drift, which is the failure D53 spent a session on. The wording says "before anything else is
resident" rather than implying otherwise.

`DEPTH` is also declared and never read — the fidelity table already lists double buffering as
written-down-only, since it is a claim about *time* and this program counts bytes — so the comment
now says so on the line itself.

### The pattern behind three questions in a row

D56, D57 and D58 all came from the user reading one emitted file and asking where a constant came
from. All three found a comment describing something *adjacent* to the value: `ACC_DTYPE` named a
deployment field whose default was wrong for the dtype, `SPLIT_K` named a consequence instead of an
origin, `DEPTH` named a measurement instead of a threshold. One was a real model bug (D56); two were
comments alone.

The common cause is that D54 required every constant to *carry* a comment, and
`test_every_constant_carries_a_comment` enforces that one exists — but nothing enforced that it
describes where the value came from. The tests added across D57 and D58 check specific phrasings,
which is weaker than a general rule and is the honest state of it: there is no way to assert that
prose is true, only that the particular claims a reader challenged are still there.

Comment text only; no value, count or assertion moved.


---

## D59 — The thread-count remark fired at a number nobody chose (2026-08-28)

The user, on the emitted harness: *"why is there a specific check on 2048?"*

There was no reason. D54-CONTINUE said *"432 threads is unremarkable on Linux; the largest shipped
profile is H100's 528. Warn, do not cap, above a few thousand"*, and "a few thousand" got rounded to
a power of two. Measured against anything real it is meaningless in both directions:

```
array 'count' across every shipped profile:
    528  h100_sxm.tensor_core      <- the largest that can ever become USED_CORES
    432  a100_80gb.tensor_core
    304  mi300x.matrix_core
     64  jetson_orin.tensor_core
      4  metis_aipu.d_imc / chip_a.npu_core
      1  chip_b.npu_core

this host:  ulimit -u 126,394   ·   threads-max 252,789
```

2048 is **3.9x above** anything a bundled profile can ask for and **62x below** the limit that would
actually bite. It could only fire for a hand-written profile, and when it fired it said nothing
useful — not "you are near a limit", just "this is a lot, by a standard nobody set".

CLAUDE.md #2 does not literally cover it: it is not a calibration constant, it touches no predicted
number, and it *cannot* live in `calibration.py` because `_harness.py` imports nothing from `bwz`
(it is inlined into standalone programs). The spirit applies anyway — an unexplained magic number
that looks derived and is not.

### Derived from the profiles instead

`spec.loaders.largest_declared_array()` returns the biggest array any bundled profile declares,
`(528, "h100_sxm.tensor_core")`. The emitter reads it and writes it into the file as a constant with
its provenance, and `run_waves` takes it as `warn_above` rather than hardcoding a bar:

```
LARGEST_DECLARED_CORES = 528            # the biggest array any bundled profile declares
                                        # (h100_sxm.tensor_core). Only used to decide whether
                                        # starting one thread per core is worth remarking on
```

```
note: starting 4,096 threads, one per modelled core — more than the largest array any
profile here declares (528, h100_sxm.tensor_core). Not capped to this host's CPUs on
purpose; see the docstring above.
```

The remark now says something true and specific: you are past everything this repository describes.
It is still **not a limit** — nothing is capped, which was the whole point of D54's trap 2.

**Array units only.** MI300X declares 19 456 vector lanes and H100 16 896 CUDA cores, but a unit
with no `systolic_dims` has no tile grid and `bwz.emit` refuses it, so those can never become
`USED_CORES`. Counting them would set the bar against a population that cannot occur and the remark
would never fire for anything.

**Pinned, so it cannot go stale the way 2048 did.** `test_profiles.py` asserts the pair is
`(528, "h100_sxm.tensor_core")`, so adding a profile with a bigger array fails there and the author
re-reads the emitted constant rather than absorbing the change silently. A second test asserts the
figure ignores geometry-less units, since that is the part a future reader is most likely to
"simplify".


---

## D60 — The emitted program can narrate its own walk (2026-08-28)

The user, after adding `print`s by hand to trace a `K=17` run: *"can you add in the python
generation print to debug those iterations (k-steps) and what each unit process? only if --debug is
passed to the python script, else for bigger models or matmuls becomes too wide."*

Reasonable, and the hand-editing was evidence: the questions D57 and D58 answered were both reached
by instrumenting the emitted file, and the one place a `print` is most wanted — inside the K sweep —
took reading two functions to find, because `run_waves` narrates cores and `run_tile` narrates
k-steps and neither says anything by default.

`python <emitted file> --debug` now prints:

```
--debug: 1 tile(s) over 1 core(s) in 1 wave(s), issuing 2 instruction tile(s).
core 0     wave 0    takes tile 0
  stage A for key (0, 0, 0) -- crosses DRAM, once per key (D33)
  tile 0      C[0:16, 0:16]  piece 0, kt 0..2
    kt=0    A[0:16, 0:16] @ B[0:16, 0:16]      4,096 useful of 4,096 slots
    kt=1    A[0:16, 16:17] @ B[16:17, 0:16]        256 useful of 4,096 slots
```

### Decisions

**Guarded at each call site, not filtered inside `log`.** Volume is the entire reason the flag
exists: a 1000x2000x3000 matmul issues 1.48 million instruction tiles, and formatting a line for
each one that nobody reads would dominate a run whose docstring already insists it is not a
benchmark. `if DEBUG:` costs one bool test. A test asserts, with `ast`, that every `log()` call sits
inside an `if DEBUG:` block — checked structurally rather than by eye, since the cost of missing one
is silent and only shows up on a large shape.

**The narration is per-decomposition, not generic.** `os` reports an accumulator sweeping K; `ws`
reports a resident B tile with M streaming past and a PARTIAL landing in the shared accumulator;
`is` reports N streaming past a resident A tile. A single "processing tile N" line would have been
less code and would have taught nothing — the same reason D54 emits different loop nests rather than
one parameterised walk.

**The line count is derived, not re-counted.** The header says how many lines to expect from
`PREDICTED["mac_slots"] // SLOTS_PER_MMA`. Every instruction tile issues `SLOTS_PER_MMA` slots, so
that quotient *is* the number of `mma()` calls and therefore of narrated lines — one expression,
taken from a constant already on the page, rather than a second count free to drift.

**Output is serialised through a lock.** Cores run concurrently and unsynchronised `print`
interleaves mid-line. The lock orders whole lines and nothing else: two cores' lines may still
appear in either order, which is honest, because in this model they genuinely run at the same time.

### What it makes visible

The staging structure D33 asserts, for one: on a 48x32x32 run, core 0 stages the band for grid row
0, core 1 takes the next tile of that row and stages **nothing**, core 2 stages row 1. "A is staged
once per row-band, shared across cores" stops being a claim in a docstring.

And the idle tail: `core 3 wave 6 idle -- no tile left (D30)`, three of them on a 25-tile run over
4 cores, against the `idle core-waves 3` the tier-1 table asserts two screens further down. The
same number, once as a line per occurrence and once as a total.

`sys` joins the emitted program's import list, which the standard-library allowlist test now
includes. No count, assertion or byte moved; `docs/CLI.md` §2.6's verbatim output block is unchanged
because that run passes no flag.


---

## D61 — Indented narration needs whole blocks, not whole lines (2026-08-28)

D60 shipped `--debug` a few minutes earlier and the user ran it on a 17-cubed matmul — four tiles,
four cores — and got this:

```
core 0     wave 0    takes tile 0
  stage A for key (0, 0, 0) -- crosses DRAM, once per key (D33)
  tile 0      C[0:16, 0:16]  piece 0, kt 0..2
core 2     wave 0    takes tile 2
  stage A for key (0, 1, 0) -- crosses DRAM, once per key (D33)
core 3     wave 0    takes tile 3
    kt=0    A[0:16, 0:16] @ B[0:16, 0:16]      4,096 useful of 4,096 slots
    kt=1    A[0:16, 16:17] @ B[16:17, 0:16]        256 useful of 4,096 slots
  tile 2      C[16:17, 0:16]  piece 0, kt 0..2
```

*"the debug log is not very clear, there is some overlap or something is not writing in order"*.

Those two `kt=` lines are **core 0's**, printed after core 3's header. The indentation says they
belong to tile 3. They do not.

### The reasoning that was wrong

D60 said, and this entry retracts it:

> *Output is serialised through a lock. ... The lock orders whole lines and nothing else: two cores'
> lines may still appear in either order, which is honest, because in this model they genuinely run
> at the same time.*

The honesty argument does not survive contact with the format. Unordered output would be honest if
the lines were independent — but they are **indented**, and indentation is a claim that a line
belongs under the one above it. Line-level serialisation preserves that claim's *typography* while
destroying its *truth*, which is worse than no structure at all. It is the same failure D56–D58 kept
finding in comments, arrived at from the other direction: a presentation that says something the
data does not.

### The fix

`log()` appends to a **thread-local** buffer; `run_waves` opens a block before `run_tile` and
flushes it after, so one core's tile prints as one unit:

```
core 0     wave 0    tile 0
  C[0:16, 0:16]  piece 0, sweeping kt 0..2
  stage A for key (0, 0, 0) -- crosses DRAM, once per key (D33)
    kt=0    A[0:16, 0:16] @ B[0:16, 0:16]      4,096 useful of 4,096 slots
    kt=1    A[0:16, 16:17] @ B[16:17, 0:16]        256 useful of 4,096 slots
```

Blocks still appear in **completion order** rather than tile order, and that genuinely is honest —
they run at the same time, and nothing in the block claims otherwise. A per-wave ordered flush was
considered and rejected: it needs a second barrier and a designated printing core, which is real
plumbing in `run_waves`, the one function whose shape is the teaching point.

The tile's geometry line also moved *above* `stage_a`, so a block reads in the order things happen —
identify the tile, stage what it needs, then issue its instruction tiles — instead of announcing the
staging before saying what was being staged for.

A failing core flushes its partial block before recording the failure: on a crash that block is the
most useful thing on screen.

### Tested against the bug, not around it

`test_debug_blocks_are_atomic_under_real_concurrency` runs an 8x8-grid program — **64 cores at
once** — in a subprocess and asserts every block has exactly one geometry line, exactly two k-steps,
and k-steps whose row range matches *its own* header. Reverting the buffering makes it fail; an
in-process single-core check would have passed against the broken version, which is why it is an
integration test with a shape big enough to interleave.

A second test asserts `--debug` changes no counted quantity: the flag is observation, and the
tier-1/tier-2 table must be identical with and without it.

## D62 — `ws` and `is` on a tensor core, and where a cut contraction's partials meet (2026-08-29)

The user, having watched `os` and `ws` produce different tile grids on paper: *"the tensor cores need
to stream out to L1 or L2 the partial C[x,y] and the GPU threads will need to know when those are
there and make the additions — this is fine, I know performance wise it's gonna be worse — what I
want is that we don't count for synchronization overheads, but we do double buffering, i.e. while
the tensor core makes the next sub-C, the cuda cores do the previous additions."*

Two things had to change for that to be answerable. An NVIDIA tensor core declared `os` alone, so
`--stationarity ws` was refused before any of it was reached; and the model charged **nothing** for a
K-on-the-grid reduction on any chip, on this reasoning:

> the same units revisit the same output cell on a later wave, so the partials meet in an
> accumulator that never leaves the chip, and this model has no on-chip bandwidth term to charge
> that against (D5a)

That is shape-dependent and unenforced. Tiles are handed out round-robin as `tile = k_slice ·
n_tiles + column`, so the k-slices of one output column land on one unit only when `n_tiles ≡ 0 (mod
units)` — true for Metis at N=8192, false at N=2560, and never on A100's 432 tensor cores:

```
does round-robin keep one output column on one core?
  metis ws   units=4    k_tiles=16   n_tiles=16   -> True     <- by accident: 16 % 4 == 0
  metis ws   units=4    k_tiles=16   n_tiles=5    -> False
  a100 ws    units=432  k_tiles=256  n_tiles=256  -> False
```

### The placement, and why the rule is a declared capability rather than a dataflow

Where partials meet is a property of the **hardware**, so it is now a `ReductionPlacement` decided by
what the unit declares, not by the dataflow's name (`docs/MODEL.md` §6.1 has the full table):
`NONE` when K is not cut, `LOCAL` when the unit declares an accumulator deep enough for K, `ON_CHIP`
when they fit in on-chip capacity and a vector unit sums them, `DRAM` for split-K's two kernels or an
accumulator too big to hold.

`ComputeUnit.local_accumulation_inputs` is that declaration, and the Metis paper — the DOI the
profile already cited — settles the case it was written for. ISSCC 2024 11.3, Fig. 11.3.1: *"local
accumulation up to 16k input channels"*, and the MVM section says what that means: *"the integer
arithmetic unit accumulates the partial products from a large MVM operation **without storing
intermediate results back to memory**"*. Fig. 11.3.4(a)'s datapath is bit-serial feeder → 16 IMC
banks → 512×26-bit accumulators → output serializer → integer arithmetic unit (across k-slices) →
64×32-bit → DPU. The accumulation happens in **one AI core's periphery** and never reaches L1.

One correction to the intuition that reached this: it is not "B static, A swapped" at the level that
matters. Within one k-slice, yes — B sits in the bank and M streams past it, which is the model's
`ws`. To accumulate *over* K the weights must **change**; what stays put is the accumulator. The four
weight sets hide that reload, which the model already calls `b_dataflow: write-ahead` (D33).

An MMA unit has no counterpart. Its accumulator is per-instruction, in one threadblock's registers,
and nothing persists across k-slices landing on different SMs. That asymmetry is the whole finding,
and it is why `local_accumulation_inputs` is 0 on all four MMA profiles.

**A unit whose native dataflow already carries K on the grid and declares no depth keeps the old,
unbounded claim.** `chip_a` and `chip_b` rest on that, so no documented number moved; the drawer says
plainly that it is unfalsifiable until a depth is declared. The alternative — treating an undeclared
depth as zero — would have charged every `chip_a` matmul a reduction at its own matrix rate, which
is worse than the thing being fixed.

### The overlap, and what is deliberately not charged

`ON_CHIP` costs `(p−1)·M·N` additions on `machine.vector_unit`, **overlapped** with the matrix work,
so an operation's compute is `max(t_matrix, t_vector)` and not their sum. That is the steady state of
the pipeline the request describes; fill and drain are omitted exactly as `max(t_dram, t_compute)`
omits them a level up (D19). Called **reduction overlap** everywhere, never "double buffering" —
that name is already spoken for, between DRAM and compute, and D5a's story becomes unreadable if the
two share a word.

Not charged, and both named in `report.assumptions` at the point they are skipped: **synchronisation**
between the two engines, per the request, and the **on-chip traffic** the partials imply — v1 has no
on-chip bandwidth term (D5a/D5b), so pricing one direction of it and not the other would be a guess
dressed as a measurement.

### What it shows, which is not what was expected

On A100 at 4096³ fp16, `ws` reports **the same latency as `os`**. The CUDA cores owe 4.28 G additions
at 19.5 TOP/s = 219 µs against 442 µs of matrix work, and the overlap hides them completely. The
ratio is not shape-dependent: the vector unit is 16× slower at `2 · rows` = 32× less work, so vector
time tends to *half* matrix time on a 16-row array at any large K.

So the cost of `ws` on a tensor core is not throughput — it is **capacity**, a live `M × N`
accumulator. At 512×512×4096 that is a bargain: `ws` gets the same 99.81% occupancy `--split-k 8`
buys and keeps `os`'s 4.72 MB of traffic instead of split-K's 13.1 MB, because the partials never
leave the chip. At 8192×8192 the accumulator is 134 MB against 60.7 MB on chip, the placement flips
to `DRAM`, and the same decomposition costs 39.8 ms against `os`'s 2.52 ms. The report says the
placement *flipped* rather than merely reporting a bigger number: it is a cliff, and a reader who
cannot see the discontinuity cannot act on it.

### Two refusals rather than a plausible number

`machine_model` falls back to the *matrix* unit when no non-systolic unit supports the dtype, so on
A100 at int8 the "vector unit" **is** the tensor core. Charging elementwise adds at 437 TOP/s would
have made `ws` look nearly as good as `os` — the exact opposite of the finding — so a requested
K-on-grid stationarity at such a dtype returns `feasible: false` naming the chip, the dtype and the
missing capability. Declaring int8 on the CUDA cores (DP4A) would unlock those cases, but
`vector_unit` also prices every non-matrix op (D27), so it moves documented int8 figures and belongs
in its own change. `--split-k` on a grid that already carries K is refused too: cutting the
contraction twice is incoherent, and the flag is an output-stationary knob (D53).

### The name is a misnomer, and the report says so

"Weight-stationary" on a unit that holds no weights describes hardware the caller is not running: an
MMA unit reads every operand from the register file per instruction (D30). The table, the drawer and
the emitted program's docstring all say *nothing held, K on the grid* there, and keep "B stays
resident" for Metis, where it is true.

### Executable, as D54 requires

`Partials.accumulate` counts what it adds — a first touch lands in a zeroed accumulator and is a
copy, so `p` slices leave `p−1` additions per element — and `partial-sum additions` is a **tier-1**
check: 28,672 at 64×64×128 under `ws` and `is`, 12,288 under `--split-k 4`, 0 under `os`. The
emitted file quotes its own report's placement rather than one placement's price in a file emitted
for another, and the fidelity table lists the overlap as *written down only*: it is a claim about
time, and the program counts.

One row is tier 2, for the one case where the program and the report model different machines: an
accumulator the report found too big for on-chip capacity. Where an accumulator lives is a capacity
heuristic and not a step of the walk, so the gap is shown rather than asserted — the same treatment
B's residency discount already gets.

## D63 — A resource row names the units the run uses, not the datasheet's count (2026-09-03)

The user, on the HTML pipeline and animation pages: *"update the HTML pipeline and animation so that
they show the max number of units USED (not available, but USED) … I am talking about the variable
`USED_CORES` in the pipelines and so on"*.

The compute row read `432 x 16x16 array` — A100's declared tensor cores. A 17-cubed matmul runs on
**two** of them. The emitted program has said so since D54 (`USED_CORES = min(AVAILABLE_CORES,
TILES)`) and the report's wave-occupancy term has charged it since D30, but the picture beside them
quoted the datasheet, which is the number a reader would divide by.

Rows now read `2 of 432 x 16x16 array`, with the note saying why: *"the grid has 2 tiles at its
widest, so 430 of these arrays never start (D30)"*. Both numbers, always, including when they are
equal — a row that only names the hardware is answering a question nobody asked of a *resource*
row.

**`used` is not occupancy, and that is why it needed its own term.** Occupancy averages over the
run: 0.5% could mean one array busy throughout or all 432 busy 0.5% of the time. `min(units, tiles)`
says which. For a phase it is the **max** over the phase's operations — the widest the workload ever
gets — so a graph whose largest GEMM fills the chip has reached all of it even if a projection later
occupies four cores.

`analysis/tiling.operation_tiles` now owns that count. It was inline in `operation_utilisation`,
where the attention case reached it as a separate `independent` multiplier; factoring it out let the
figures ask the same function the utilisation term divides by, rather than re-deriving a tile count
beside it — the failure mode D53 had already been through once. `systolic_utilisation` loses its
`independent` parameter, which no caller now passes, and the multiplication happens once inside
`operation_tiles`.

**The vector row gets the same treatment, and an admission.** It reads `0 of 6912` when nothing
elementwise reaches it — true of a lone output-stationary matmul, and the honest thing to draw — and
its full count when something does. *How much* of a vector unit runs that work is not modelled: the
cost is charged at the whole unit's rate, so the note says all of it is assumed engaged rather than
implying a measurement. Under `--stationarity ws` the row goes from `0 of 6912` to `6912 of 6912`,
which is D62 in one line of the picture.

### Two things found while doing it

**The stationarity banner had never rendered.** `_stationarity_banner` read `spans[0].grid`, and span
0 is the kernel dispatch, which carries no tile to address — so the guard saw `None` and returned
empty on every matmul page ever generated. It now takes the first span that *has* a grid. The page
had been missing the line that says which decomposition its tile addresses index into, which is what
makes them readable at all.

**The banner said "B stays resident" on a tensor core.** Same misnomer D62 fixed in the table, the
drawer and the emitted program, missed here. It now uses the same `residency_phrase`, and — where
every panel agrees — adds where the partials meet: local and free, on chip and overlapped, or
through DRAM and serialised. Silent when two chips disagree, by the rule that function already had:
Metis sums a contraction in its own periphery where A100 pays the CUDA cores, and one line cannot be
right about both.

## D64 — `--out` was declared twice on `bwz matmul`, and the dtype half never ran (2026-09-03)

Found while assembling a set of commands to demonstrate D62/D63: `bwz matmul` declared `--out`
**twice** — once as the result width (`--out int32`, the widening accumulator, §2.3) and once as the
figure directory (`--out ~/figs`, §3). Click keeps one. The directory won, so

```bash
uv run bwz matmul -M 512 -N 512 -K 512 -c a100_80gb -d int8 --out int32
```

parsed `int32` as a **path**, silently ignored the accumulator width, and printed `result C 262 kB`
— 512×512×1 B, the int8 result — where the documentation promised 1.05 MB. No error, no warning: a
documented flag that had not worked since the figure flag was added, quoted in `docs/CLI.md` §2.3,
the README's widths block and D18's own example.

The result width is now **`--c`**, which pairs with the `--a` and `--b` it belongs with: three
operands, three flags. `--out` keeps the meaning it actually had — the figure directory, on `matmul`,
`run` and `compare` alike, so one word does not mean two things across three commands.

No number moved: the tests that cover widening accumulators (`test_documented_numbers` §2.3,
`test_matmul_workload`) call `matmul_kernel`/`analyze` directly, which is exactly why they stayed
green through a broken CLI. That is the gap worth naming — a CLI flag is only covered by a test that
goes through the CLI, and `test_cli_smoke` did not try this one.

## D65 — A page is named for the decomposition it draws, and `--emit` loses its path (2026-09-16)

The user, on `bwz matmul … --stationarity is --ideal --emit . --timeline --out .`:

> *"so the html miss the -is in the name - plus, emit and timeline should both makes their output to
> out field, emit should not have its own path"*

Two bugs in one command, and the second is worse than it looks.

### The page did not say which decomposition it was of

`_figure_stem` was `<chip>-<dtype>`, deliberately so per D55 — *"unchanged from the names
docs/plots/README.md documents … so a moved script does not orphan a figure a reader has a link
to"*. That reasoning predates D53: when every chip had exactly one dataflow, the shape and the chip
*were* the decomposition. Since `--stationarity` exists, they are not, and three runs of one shape
wrote three different pages to **one filename** — the last one silently winning.

The stem now ends with the same `-<stationarity>[-splitk<N>]` the emitted program has carried since
D54, from a single `emit.decomposition_suffix`. A page and the program beside it are two views of
one decomposition, so they are named alike; `matmul-a100_80gb-fp16-is.py` now sits next to
`timeline-a100_80gb-fp16-is.html`.

**Omitted where the page draws more than one.** A `--compare-with` page of A100 (`os` natively) and
Metis (`ws`) has no single decomposition to name, so it keeps the plain stem — the same condition
that already leaves its stationarity banner blank, now enforced in one place and read by both.
Nothing tracked was renamed: `docs/plots/*.html` is gitignored and `make plots` regenerates it, so
the churn was four links in three markdown files.

### `--emit` carried a destination that `--out` already owned

`--emit PATH` and `--out DIR` were two destinations for one run's artifacts. The failure that
exposes it is not the duplication but the parse: **`--emit --timeline` consumed `--timeline` as the
path** and wrote the program to a file named `--timeline`, with no error and no page. That file was
sitting in `backend/` when this was found, which is how long it had gone unnoticed.

`--emit` is now a flag and the program lands in `--out`. The stdout mode — the documented
`| python -` pipeline, which is not a path but read like one — became its own flag,
`--emit-stdout`, so nothing in the command line accepts a second destination.

### The gap this pair shares with D64

Both were CLI-shaped, and both survived a green suite for the same reason: every test that covered
emission or figures called `emit_matmul`/`write_timeline` **directly**, where a filename is an
argument rather than a parse. `test_cli_smoke` ran `--emit` only with an explicit path, which is the
one form that worked. Four tests now go through the CLI and assert the *set of files on disk* —
which is the actual contract, and the thing neither bug could have passed.

## D66 — A partial cannot be summed before it exists (2026-09-16)

The user, looking at `timeline-a100_80gb-fp16-is.html` for 512³:

> *"why is the reduction happening in the middle and nothing happens at the end? aren't we reducing
> the final part?"*

They are. D62 drew an `ON_CHIP` reduction as **one span** starting when the first wave ended and
running for the whole `t_reduce_s`. On three waves that put the vector lane at 468→885 ns against a
core lane running to 1194 ns — the CUDA cores finishing their additions **309 ns before the array
produced the last partials they were adding**.

The overlap D62 argued for is real; the shape drawn for it was not. Both of these have to hold at
once, and the single block only had the first:

- every bar but the last **overlaps** the core lane — the array builds wave *n+1* while the vector
  unit sums wave *n*, which is what `max(t_matrix, t_vector)` means;
- the last bar **follows** the last wave, because the partials it sums do not exist until then.

So the lane is now one bar per compute step, each starting where its own step ended. The final bar
is the pipeline **drain**, and the drawn span consequently runs past the reported latency — 1.33 µs
against 1.09 µs, with the 243 ns named in `fill_drain_s` and printed under the figure. That is
exactly how the DRAM/compute overlap has always worked (D19): `max` is the steady state, and the
picture is where fill and drain become visible. Each lane still sums to the term it decomposes.

### The root cause is a duplicated schedule, and the user named it

> *"if you base the timeline code on the generated python code, it should be much more trivial"*

Right about the diagnosis. `analysis/pipeline.py` and `bwz/emit/matmul.py` both encode the same
walk — the grid, the wave assignment, the staging events, where the partials meet — and the emitted
program has had the correct structure since D62: `partials.accumulate(...)` sits *inside*
`run_tile`, so it trails each tile by construction and cannot be drawn in the wrong place. The trace
re-derived that ordering by hand and got it wrong. This is the failure mode CLAUDE.md already
records once, about the tile grid: *"it was duplicated in `pipeline.tile_count` and
`tiling.systolic_utilisation` once, and the two drifted."*

What it does **not** imply is that the timeline should be generated from the program. Three reasons,
recorded so the idea is not re-proposed without them:

1. **The dependency arrow.** `analysis → report → {api, cli, emit, figures}`. `analysis/pipeline.py`
   importing `bwz/emit/` inverts it.
2. **The program models counts, not time**, and says so before anything else — *"NOT A BENCHMARK …
   its wall clock has no relationship to the latency the report predicts"*. The trace is entirely
   about time.
3. **A network has no program.** `bwz run` draws a per-operation trace where there is no tile grid
   to walk (D5a), so the trace needs a path the emitter does not have.

The fix the diagnosis actually points to is to hoist the shared thing **down** into `analysis/`: an
ordered walk of `(wave, core, tile, stage)` events that `build_trace` assigns durations to and
`emit` writes as loops, the way `analysis/stationarity.py` already owns the grid both of them read.
Queued, not done here — it is a refactor across two thousand-line modules and wants its own change.

### A second instance, found by asking the question

Checking *how much* the pages already depend on the program turned one up. `figures/` does import
`bwz.emit` — the code pane **is** the emitted program, and the animation lights its lines as the
schedule plays, mapping each `Stage` onto a tag the emitter wrote. But `analysis/pipeline.py`, which
decides the bars and their timing, imports nothing from it. So the bars and the highlighting come
from two places, and for one stage they did not meet:

```
stage_lines tags in the emitted program: ['exec', 'load_a', 'load_b', 'store']
stages the trace plays back:             ['exec', 'hold', 'load_a', 'reduce', 'store']
```

A K-on-grid walk's `partials.accumulate(...)` was tagged `exec` — the tag its neighbouring `mma`
carries — so a `ws`/`is` animation played a REDUCE bar on the vector lane with **no line lit under
it**. Only split-K's second kernel had ever claimed the `reduce` tag. That line *is* the reduction:
the report charges it to the vector unit and the trace draws it there, so it is tagged `reduce` now,
and a test asserts every stage the trace plays has lines in the program rather than leaving it to be
noticed in a browser.

## D67 — Every split of the decomposition lives in one emitted function (2026-09-28)

### What was wrong

The emitted program was correct and hard to read for the question it exists to answer — *how
does this decomposition differ from that one?* The hierarchy was spread over six functions: the
waves and the core-to-tile assignment in the harness's `run_waves`, the tile's grid position in
`tile_row`/`tile_col`/`partition_of`, A's staging in `stage_a`, the sweep in `run_tile`, and
split-K's second kernel in `reduce_partials`. Four of those had bodies that changed with the
stationarity, so a `diff` of the `os` and `ws` programs landed in five places, and a reader asking
"what is a wave, and where do M, N and K go?" had to assemble the answer from all of them.

### The fix

One function, `walk()`, holds every split, outermost first, each marked `# == LEVEL n`:

```
LEVEL 1  waves        for wave in range(WAVES)
LEVEL 2  cores        tile = wave * USED_CORES + core_id; idle if tile >= TILES (D30)
LEVEL 3  tile grid    the tile's (split-K piece,) grid row and column -> two of M, N, K
         A            pad.band(...) — staged once per group into the shared buffer (D33)
         stays put    acc / b / a, by stationarity
LEVEL 4  the sweep    the third dimension, one instruction tile at a time
LEVEL 5  one mma()
after    partials.drain, or split-K's KERNEL 2 inline
```

Its docstring is that table filled in with the run's own numbers, so two files differ first in
the table. The code of levels 1, 2 and 5 is identical across stationarities; levels 3 and 4 are
where they differ, which is D53's claim made visible.

The harness keeps only what is the same for every decomposition. `run_waves` became `run_cores`:
it starts one thread per core and hands each the barrier as `end_of_wave`, and nothing else. D54's
design is unchanged — persistent threads, one per modelled core, lockstep through a barrier, the
`if tile >= TILES` branch *being* wave occupancy — it has only moved to where it is read.
`a_strategy=whole`'s prologue is generated by the same code as the per-tile staging, so the two
cannot drift.

### What it cost

Repetition, deliberately: the generated file restates the grid arithmetic inline instead of
calling helpers, and the prologue repeats the grid-position lines. Counts, checks and animation
stage tags are unchanged; every emitted program still asserts its tier-1 counts against the report.

### Open: split-K's vocabulary (`TODO(D67-open)`)

Two different cuts of K share overlapping names. Split-K's cut is a "partition" in
`TileGrid.k_partitions`/`partition_of`, a "piece" or `part` in the emitted program, and
`K_TILE_BOUNDS` in its constants; K on the grid under `ws`/`is` makes "k-slices", and
`TileGrid.k_slices` counts *both*. `walk()`'s docstring now draws the grid and says which is which,
but the names still invite the confusion. Deferred on purpose until the Axelera work, so the emitted
files stay diffable against today's; the sites carry `TODO(D67-open)`.

`walk()`'s docstring also draws LEVEL 3: the grid as numbered boxes (at most 8 x 8, the middle
elided past that so both edges stay visible), then one tile followed through its sweep and where its
result goes.

## D68 — A column's k-slices share one core, or they do not sum locally (2026-09-28)

### What was wrong

`LOCAL` placement (D62) charges nothing for a K-on-grid reduction because Metis's periphery sums
the k-slices "without storing intermediate results back to memory". That periphery belongs to one
AI core. The deal that put tiles on cores was round-robin — tile `t` to core `t mod units`, across
grid rows — so a column's k-slices only shared a core when the column count happened to divide the
core count. The report even said so: *"That assignment is assumed, not enforced (D62)."*

It mattered exactly where it was assumed away. Metis, `512 x 512 x 8192`: one output column, 16
k-slices, dealt four to each of four cores in four waves at 100% occupancy — and the drawer said
all 16 were summed in one periphery. They were in four.

### The fix

`analysis/stationarity.py` now decides the deal as well as the grid. `sums_locally(grid, unit)` is
the placement's condition — K cut on the grid's **row** axis, not by split-K, within
`accumulation_depth` — and when it holds `deal()` keeps each column on one unit: a wave never
straddles a grid row, unit `u` takes column `round · used + u` in every row, `used = min(units,
cols)`, `waves = rows · ceil(cols / used)`. Otherwise the round-robin deal is unchanged.

`Deal` is read by the utilisation's wave-occupancy term (`tiling.operation_utilisation`, via
`operation_deal`), the pipeline's wave count and tile ranges, the timeline's "arrays reached", and
the emitted program, whose LEVEL 2 now reads `kt, round_ = divmod(wave, ROUNDS)` /
`nt = round_ * USED_CORES + core_id` — the column a core keeps, visibly the same for every kt.
`is` grids no longer qualify for `LOCAL` on any unit: their K is the column axis, and a wave
spreads a row's k-slices across units.

### What moved

Only what was wrong. Metis `512 x 512 x 8192`: utilisation 99.99% → 25.00%, `t_compute` 20.5 →
81.9 µs (latency unchanged at 131 µs — DRAM-bound). `512 x 1024 x 4096`: → 49.99%, 41 µs.
MobileNetV3 on Metis and `chip_a` +3.4% (convolutions whose output channels give fewer 512-wide
columns than cores). Every A100 and GPU figure, every transformer figure in the documented tables,
Metis at N = K = 2048 and 8192³: identical.

### What it does not do

Spreading one column's k-slices over idle cores *and* summing their partials on the DPU is a real
alternative when N is narrow — more occupancy, paid for in cross-core traffic and DPU time. It is a
different decomposition, not modelled; the drawer names it where cores sit idle. Choosing between
the two would need the DPU's throughput, which the paper does not publish.

## D69 — K-groups: idle cores share a narrow column's K, and only the groups' partials leave (2026-09-29)

### What was wrong

D68 kept a column's k-slices on one core so that the periphery could sum them, and paid for it in
occupancy: Metis at N = 512, K = 8192 ran on one core of four. That is the right answer for a chip
with nothing to add partials off the array, and the wrong one for a chip that has the DPU. And past
Metis's 16k-input accumulator the model went back to a round-robin spread, sending one partial per
k-slice to the DPU — 63 · M · N adds at K = 32768 where the periphery could have summed all but one
group's worth.

### The fix

`k_groups(grid, unit, vector_adder=...)` in `analysis/stationarity.py`: the number of units sharing
one column's K. The larger of a depth reason (`ceil(k_slices / slices per accumulator)`) and an
occupancy reason (`floor(units / cols)` when N is narrow **and** the chip has a vector unit), capped
at `k_slices`. Each unit keeps one column and one group (`kt mod g`) and sums it locally; only the
`g` group partials per output leave, so `partials_per_output` is `g` and the reduction is
`(g − 1) · M · N` on the vector unit, `ON_CHIP`/`DRAM` by the usual capacity test. `deal()` walks
blocks of `g` grid rows, so every wave is still one contiguous tile range. `vector_adder` is a
required argument everywhere the deal is decided — the utilisation, the reduction, the trace, the
timeline, the emitter, the report — so no path can take the occupancy and skip the price.

The emitted program shows it: `K_GROUPS`, `CHUNKS`, and a LEVEL 2 of `block, chunk =
divmod(wave, CHUNKS)` / `group, nt = divmod(item, GRID_COLS)` / `kt = block * K_GROUPS + group`.
`Partials` holds one accumulator per group and `drain()` adds them, counting those adds as
`cross_core_adds` — a new tier-1 check against the report's `(g − 1) · M · N`.

Convolutions whose k-slices are grouped are charged the same reduction (`roofline._reduction_for`);
before this only matmuls were, so a grouped convolution would have taken the occupancy for free.

### What moved

Metis `512 × 512 × 8192`: 25% → 99.99%, `t_compute` 81.9 → 20.5 µs, plus 1.92 µs of hidden DPU adds.
`512 × 1024 × 4096`: 50% → 99.98%. K = 32768: 63 → 1 group-partial per output. MobileNetV3 on
Metis back to 1.370 ms (its convolutions' DPU adds are charged, and hidden). chip_a unchanged from
D68 (no vector unit). Every A100 figure unchanged.

Also fixed on the way: `reduction_cost` returned `NO_REDUCTION` for a `LOCAL` grid once the count of
partials *leaving* the unit became 1, which hid the `local — free` row; and every on-chip REDUCE span
in the timeline was labelled as split-K's second kernel.

### What it does not do

The group partials' L1/L2 traffic and the synchronisation between cores are not charged (no on-chip
bandwidth term, D5b); the DPU rate is the profile's estimate. The paper does not describe splitting
one matmul's K across AI cores — only that cores can jointly tackle a workload — so this is the
model's mapping, and the drawer says so. A convolution spread across MMA units (A100) is still
charged no reduction: a gap older than D69, recorded here rather than closed, since closing it moves
every GPU CNN figure.


## D70 — FlashAttention as a decomposition the formula chooses, and a program that runs it (2026-09-30)

### What was missing

`bwz matmul` shows how one GEMM maps onto a chip — the grid, the deal, the reduction — and `--emit`
makes that mapping a program that checks itself (D54). Attention had nothing comparable. The
operator (`operators/attention.py`) prices FlashAttention as the score matrix *not* crossing DRAM,
and that is all: no blocks, no programs, no inner dataflow, no online softmax, and so nothing to
learn about why the same attention runs at 86% of peak on one chip and 11% on another.

### The change

`analysis/flash.py` models FlashAttention-2 as programs — one `(batch, head, Br-row block)` each,
pinned to one matrix unit, dealt round-robin in lockstep waves (D30) — whose two inner matmuls,
`S = Q·Kᵀ` and `O += P·V`, are costed by the lone-matmul formula on one unit: shape padding (D52)
and the reduction their stationarity owes (D62), with no occupancy term. The online softmax is
charged to the vector unit's per-unit share. K and V are staged chip-wide per wave, D33's rule, so a
head straddling a wave boundary is read once per wave it touches.

**The formula chooses the plan.** `Br`, `Bc` and each inner matmul's dataflow are searched, every
candidate is costed, and the fastest that fits on chip is kept — all candidates are returned, so
`bwz attention` shows what the choice beat and by how much, and each inner matmul's table shows
every dataflow the unit declares with its reduction. `--br`, `--bc` and `--stationarity` pin
instead of search; an undeclared stationarity is refused (D53), as is a chip with no vector unit to
run the exp (D62's rule — chip_a).

`emit/flash.py` writes the chosen plan as a runnable program on the same `_harness.py` runtime,
extended with generic tile primitives (transpose, row max/sum, a shifted exp, row scaling), a named
`Ledger` and a counted `DramTensor`. It asserts sixteen counts against the plan and checks `O`
against the textbook softmax in float64.

### A rule refined, not broken

CLAUDE.md says FlashAttention changes bytes, never FLOPs. That holds for the matrix engine — every
plan issues exactly `2·Sq·Skv·d` MACs per head, and the tests pin it. It does not hold for the
vector unit: each kv block after a program's first rescales `O` by `exp(m − m')`, `Br·d` more
operations per block, and the divide moves from the scores onto `O`. This is small on A100 and not
small on Metis, whose DPU is already the bottleneck, so it is charged and the rule is reworded.

### What it does not do

- It is a probe, like `bwz matmul`: `encoder-layer` and `run` still cost attention through the
  operator. Wiring the planned cost into the graph (a Report change: schema, TS types, snapshots)
  is the next step.
- No causal mask, no vanilla-attention comparison, no flash-decoding (split over `kv_len`, the
  attention form of K-groups, D69). All three are named follow-ups.
- Capacity is judged against every on-chip level together, as residency is everywhere in v1. A real
  kernel's `S` and `O` live in registers and shared memory, which caps `Bc` far below what this
  allows; the drawer says so.
- Softmax within a program is serial with the matmuls (FA2). FlashAttention-3's overlap is not
  modelled.
- On an in-memory array (Metis) every K and V block must be written into the banks as the
  stationary operand. That write is capacity-modelled only — it needs the on-chip bandwidth term v1
  lacks (D5a, D30) — and the drawer says so.

## D71 — The FlashAttention plan, drawn: one step per (wave, kv block) (2026-09-30)

### What was missing

D70 gave attention a plan and a program, but `--timeline` existed only on `matmul`, `run` and
`encoder-layer`, all drawn from `analysis/pipeline.py`'s matmul and graph schedules. The most
teachable thing about FlashAttention — the serial chain of array, vector unit, array inside every
block, and K/V fetches hiding behind it — could only be read off a table.

### The change

`analysis/flash.flash_trace` turns the chosen plan into a `PipelineTrace` of the same `Span`s every
other page draws. One step is one kv block of one lockstep wave; its DRAM work is the block of K and
V for every head in the wave (Q on the wave's first block, O after its last), its compute the wave's
slowest program running that block. Loads follow §6.5's double-buffer rule. The per-block costs were
factored out of the planner into `program_steps`, which both now read — so the DRAM, array and vector
lanes sum to `t_dram`, `t_matrix` and `t_vector` exactly, and the span is never shorter than the
reported latency (D19). Past `--steps` steps, consecutive ones are coalesced with every total kept.

The page itself needed three things generalised rather than special-cased. A `Workload` now carries
its own achieved rate when it has no `Report` phase, its operand names (`K,V`/`Q`/`O` rather than
`B`/`A`/`C`), a banner, a legend and a program introduction — the matmul defaults are unchanged byte
for byte. And the vector row names the lanes a workload engages when it is not the whole unit: each
FlashAttention program gets `1/units` of the vector unit (D70), so eight programs on A100 engage
128 of 6912 CUDA cores. Saying "6912 of 6912, assumed engaged" there would have contradicted the
rate the plan charged.

### What it does not do

`--animate` is not on `bwz attention` yet: the playback's stations and geometry panel are built
around a matmul's tile grid, and the flash walk needs its own. `--compare-with` works, with one
catch the model states rather than hides: A100's CUDA cores declare no int8 datapath, so A100 cannot
run int8 attention (no vector unit for the softmax, D62) and cannot share an int8 page with Metis.
