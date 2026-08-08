# The analytical model

Every formula the engine uses, with derivation and limits of validity.
Filled in milestone by milestone; the authoritative build spec is `PROMPT.md` §3.

## Status

| Section | Milestone | Status |
|---|---|---|
| Peak throughput and the ridge point | M1 | **implemented** |
| Parameter and KV-cache counts | M1 | **implemented** |
| FLOP/byte counts (transformer, CNN) | M2 | **implemented** |
| Roofline (flat: compute ridge vs DRAM ridge) | M3 | not yet implemented |
| Roofline (hierarchical, multi-level) | M8, on demand | not yet implemented |
| Tiling and DRAM traffic | M3 | not yet implemented |
| Multi-chip sharding and collectives | M5 | not yet implemented |
| Memory capacity planning | M3 | not yet implemented |
| Power/energy | M7 | not yet implemented |
| Bottleneck classification | M3 | not yet implemented |

## The v1 machine

Three elements, and no more (`docs/CORRECTIONS.md` D5a):

- **External memory (DRAM/HBM)** — the only bandwidth ceiling.
- **On-chip SRAM** — **capacity only**. It buys two things: the fraction of weights that need not
  be re-streamed from DRAM, and the headroom that makes double buffering possible. It contributes
  no bandwidth term. D5b records why, and what a published SRAM organization would let us add.
- **Compute** — the TOPS ceiling.

The multi-level hierarchy, the ws/os/rs loop-order search and Winograd/FFT are deferred to M8
(D5).

---

## 1. Peak throughput (M1, `spec/hardware_spec.py`)

A compute unit is described by its MAC rate; operations are twice that. **The doubling happens in
`ComputeUnit.peak_flops_per_s` and nowhere else in the codebase** (CLAUDE.md #5):

```
peak_ops_per_s = count x macs_per_cycle_per_unit x clock_hz x dtype_multiplier
                       x sparsity_speedup x 2
```

A chip's peak is the **max** over its compute units, not the sum: a GEMM runs on the tensor cores
or on the vector cores, not both at once, and vendor headline figures quote the dominant unit.

Worked examples, each pinned by a golden test:

| chip | derivation | peak |
|---|---|---|
| H100 SXM5 fp16 | `528 x 512 x 1.83e9 x 2` | 989.4 TFLOP/s |
| A100 80GB fp16 | `432 x 256 x 1.41e9 x 2` | 312.0 TFLOP/s |
| MI300X fp16 | `304 x 1024 x 2.1e9 x 2` | 1307.4 TFLOP/s |
| Jetson Orin fp16 | `64 x 256 x 1.3e9 x 2` | 42.6 TFLOP/s |
| chip_a int8 | `4 x 262144 x 0.8e9 x 0.125 x 2` | 209.7 TOP/s |

`chip_a`'s 0.125 multiplier is a bit-serial INT8 datapath taking 8 cycles per MAC. Expressing it
as a dtype multiplier rather than a divided clock keeps the MAC count architectural and the
doubling in one place.

### The ridge point

```
ridge_point = peak_ops_per_s / dram_bandwidth_bytes_per_s      [ops/byte]
```

An operation whose arithmetic intensity is below the ridge is DRAM-bound; above it,
compute-bound. `chip_a`: `209.6e12 / 34e9 = 6165` ops/byte. `chip_b`: `52.4e12 / 34e9 = 1541`.
Both are enormous, which is the whole story of batch-1 decode: at an intensity of 2 ops/byte
(one MAC pair per INT8 weight byte) you are three orders of magnitude to the left of the ridge,
and nothing about the compute array matters.

---

## 2. Parameter counts (M1, `spec/model_spec.py`)

Derived closed-form from the hyperparameters, independently of the graph. M2's builder sums its
weight tensors and cross-checks against this — two independent derivations agreeing is a stronger
test than either alone.

```
q_width    = heads    x head_dim         # NOT hidden, when head_dim is explicit
kv_width   = kv_heads x head_dim         # GQA: kv_heads < heads

attn/layer = hidden x q_width + 2 x hidden x kv_width + q_width x hidden
ffn/layer  = n_matrices x hidden x ffn_hidden      # 3 for swiglu/geglu, 2 for gelu/relu
norm/layer = 2 x (hidden for rmsnorm, 2 x hidden for layernorm)

embeddings = vocab x hidden, doubled unless tie_embeddings
params     = layers x (attn + ffn + norm) + embeddings + final_norm
```

Worked example (Llama-3-8B): attention `4096x4096 + 2x4096x1024 + 4096x4096 = 41.94 M`; FFN
`3 x 4096 x 14336 = 176.16 M`; norms `2 x 4096`. Per layer 218.11 M, times 32 layers = 6.980 G.
Untied embeddings `2 x 128256 x 4096 = 1.051 G`. **Total 8.030 G**, against Meta's stated 8.03 B.

The `q_width != hidden` case is not a curiosity: Gemma-3-4B has 8 heads of 256 against a hidden
of 2560, so `q_width = 2048`. Assuming `q_width == hidden` would overstate its attention weights
by 25%.

### KV cache

```
kv_bytes(C) = layers x 2 x kv_width x C x bytes_per_element x batch
```

The 2 is K and V. Worked example (Llama-3-8B, 8k context, fp16, batch 1):
`32 x 2 x 1024 x 8192 x 2 = 1.0737e9` bytes — exactly 1.0 GiB, or 1.07 GB in the decimal
convention this engine uses throughout. GQA is what makes this affordable: with 32 KV heads
instead of 8 it would be 4.3 GB.

---

## 3. Operator cost models (M2, `operators/`)

Each model reports the arithmetic an operation performs and its **compulsory** traffic — every
weight read once, every input read once, every output written once. Nothing here knows about a
chip: tile re-reads, cache reuse and the systolic tail effect are `analysis/`'s job at M3
(`docs/CORRECTIONS.md` D10). That is what makes every number below checkable with a calculator.

`OpCost` splits bytes by role because M3 treats them differently — weight traffic is what
residency removes, activation traffic is what tiling affects, and `scratch` is what
FlashAttention eliminates.

### 3.1 Matmul

```
flops        = 2 · M · N · K
weight_bytes = K · N · sizeof(w_dtype)
input_bytes  = M · K · sizeof(a_dtype)
output_bytes = M · N · sizeof(a_dtype)
```

`M` folds batch and sequence: `batch·S` at prefill, `batch` at decode. That one number is the
difference between the two regimes. Llama-3-8B's Q projection at batch 1 is `M=1, K=N=4096`:
33.55 MFLOP against 33.55 MB of fp16 weights — **1 FLOP per byte**, against a ridge point of 295
on H100. Nothing about the compute array matters at that intensity.

### 3.2 Attention

```
positions = q_len · kv_len,  or the causal lower triangle when q_len > 1:
            q_len · (kv_len − q_len) + q_len · (q_len + 1)/2
scored    = batch · heads · positions
flops     = 2 · 2 · scored · head_dim  +  5 · scored
```

Three properties the tests assert directly:

- **Causal masking halves prefill scores.** The exact triangular fraction is used, not a flat 0.5,
  so short prompts and prompts appended to existing context are both right. At decode `q_len = 1`
  and every key is visible, so nothing is halved. PROMPT.md's `4·d_model·S²` omits this.
- **GQA changes bytes, never FLOPs.** All `heads` query heads participate in every score; only
  `kv_heads` distinct K/V heads are read.
- **FlashAttention changes bytes, never FLOPs.** Vanilla writes the `q_len × kv_len` score matrix
  and reads it back (`scratch_bytes = 2 · scored · sizeof(a_dtype)`); flash tiles it in registers.

Worked example (Llama-3-8B prefill, S=2048, batch 1, 32 heads, head_dim 128): positions
`2048·2049/2 = 2 098 176`, scored `67 141 632`, so `4 · scored · 128 = 34.38 GFLOP` plus
`5 · scored = 0.34 GFLOP` — **34.71 GFLOP**.

### 3.3 Norm, elementwise, embedding

`flops = elements × flops_per_element`, with the per-element counts in `calibration.py` as
declared conventions (RMSNorm 4, LayerNorm 6, softmax 5 per score, SiLU 4, GELU 8, RoPE 3). None
of them changes a bottleneck verdict; all of them are memory-bound at intensity ≈ 1.

**Embedding is a gather, and this matters.** Only the rows touched are traffic; the table is
footprint. Charging the table would add 1.05 GB of phantom traffic to every Llama-3-8B decode step
and roughly double its predicted latency.

### 3.4 Convolution

```
flops = 2 · batch · out_h · out_w · out_channels · (in_channels / groups) · kernel_h · kernel_w
```

im2col shares this model — same arithmetic, different layout. Winograd and FFT genuinely reduce
the arithmetic and are M8.

**Depthwise** sets `groups = in_channels = out_channels`, collapsing the `in_channels/groups`
factor to 1: FLOPs fall by the channel count while activation traffic is unchanged, so arithmetic
intensity falls by the same factor. That is why MobileNetV3's depthwise layers are memory-bound.

---

## 4. Transformer expansion (M2, `graph/transformer.py`)

Prefill and decode are built as **separate graphs**, not one graph with a flag (CLAUDE.md #6).
Same weights, same op count, same structure; the only difference is `M` and the KV wiring:

| | prefill | decode |
|---|---|---|
| `M` (projection rows) | `batch · S` | `batch` |
| `q_len` / `kv_len` | `S` / `S` | `1` / context |
| KV traffic | K/V just computed, written as the cache | the whole cache streamed |
| arithmetic intensity (Llama-3-8B) | **744** FLOP/byte | **1.06** FLOP/byte |

Per block: norm → Q/K/V projections → RoPE → attention → O projection → residual → norm → FFN
(gate/up/down for a gated FFN, up/down otherwise) → residual. Around them: an embedding gather,
a final norm, and the LM head.

Two modelling choices worth stating:

- **The LM head runs on the last token only**, in both phases. Inference needs one distribution per
  *generated* token, and every serving stack slices before the projection. Computing all `S` would
  add `2·S·hidden·vocab` — for Gemma-3-4B at S=2048 that is 2.7 TFLOP against a 13.7 TFLOP forward
  pass, a 20% error.
- **Tied embeddings are one tensor**, so the LM head reuses `embed_tokens` and the footprint counts
  it once.

### The analytic FLOP rules, and where they stop working

Prefill is forward-only, so the rule is **`2·N·D`** — `6·N·D` is the *training* figure and
PROMPT.md's M2 DoD states it in error (`docs/CORRECTIONS.md` D11). But `2·N·D` assumes every
parameter does two FLOPs per token, and embedding parameters do not — they are gathered, and the
head runs once:

| model | embeddings as % of N | vs `2·N·D` | vs `2·N_non-emb·D` |
|---|---|---|---|
| GPT-3 175B | 0.4% | **+1.1%** | +1.4% |
| Llama-3-8B | 13.1% | −9.7% | +3.9% |
| Gemma-3-4B | 17.3% | −13.5% | +4.5% |

The general form is `2·N_non-embedding·D`, and the residual above it is attention, which grows as
`D²` — Gemma-3-4B is +1.2% over that form at S=512 and +4.5% at S=2048. Quote either rule with its
context length and its embedding share, or not at all.

---

## 5. CNN expansion (M2, `graph/cnn.py`)

Channel propagation is the whole job: layer *n*'s weight shape depends on layer *n−1*'s output
channels, which is why `CNNSpec.parameter_count()` raises and defers to the graph. Spatial extents
follow `ceil(extent / stride)` ('same' padding).

An MBConv block expands 1×1, convolves depthwise `k×k`, optionally squeeze-excites, and projects
1×1, with a residual when stride is 1 and channels are unchanged. **The expansion is omitted when
`expand_channels == in_channels`** — MobileNetV3's first block expands 16→16, and the reference
implementation skips it rather than emitting an identity 1×1.

MobileNetV3-Large 1.0 at 224², batch 1: **5.47 M parameters** (paper: 5.4 M, +1.3% — batch-norm
pairs and SE bottlenecks, which the headline figure does not itemise) and **234.8 M MACs**
(paper: 219 M, +7.2%). Every convolution in the network sits below 65 FLOP/byte against H100's
ridge of 295 and chip_a's of 6168, so the whole network is memory-bound on either.
