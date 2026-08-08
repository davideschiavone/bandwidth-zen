# The analytical model

Every formula the engine uses, with derivation and limits of validity.
Filled in milestone by milestone; the authoritative build spec is `PROMPT.md` §3.

## Status

| Section | Milestone | Status |
|---|---|---|
| Peak throughput and the ridge point | M1 | **implemented** |
| Parameter and KV-cache counts | M1 | **implemented** |
| FLOP/byte counts (transformer, CNN) | M2 | not yet implemented |
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
