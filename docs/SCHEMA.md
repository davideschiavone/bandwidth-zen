# Spec schema reference

YAML/JSON reference for chip profiles, model specs, and deployment specs. For the commands that
consume them, see [`CLI.md`](CLI.md).
**Authoritative source: the pydantic models in `backend/bwz/spec/`** — this document describes them;
where the two disagree, the code wins and this file is the bug.

Conventions that apply everywhere:

- **SI units, always** (CLAUDE.md #1). Field names carry the unit: `bandwidth_bytes_per_s`,
  `capacity_bytes`, `latency_s`. The two exceptions are `clock_ghz` and `latency_ns`, which are
  human-facing YAML fields converted on read.
- **Human strings are accepted** wherever a quantity is expected: `capacity_bytes: "80 GB"` and
  `capacity_bytes: 8.0e+10` are identical. Binary prefixes (`GiB`) parse too.
- **Write exponents with a sign** — `1.0e+11`, not `1.0e11`. YAML 1.1 only recognises a signed
  exponent as a float; the loader rescues the unsigned form by treating it as a unitless SI value,
  but the signed form is what YAML means.
- **Unknown keys are errors.** A typo names itself rather than being silently dropped.
- Every profile needs a `source_url`, or must declare `hypothetical: true` (see *Provenance*).

---

## 1. Chip profile — `backend/profiles/chips/<id>.yaml`

```yaml
id: h100_sxm                    # must equal the filename stem; auto-filled if omitted
name: NVIDIA H100 SXM5 80GB
vendor: NVIDIA
source_url: https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-datasheet
process_nm: 4                   # optional
clock_ghz: 1.83

compute_units:                  # at least one
  - name: tensor_core
    count: 528
    ops_per_cycle_per_unit: 512         # MACs. x2 -> OPs happens once, in peak_flops_per_s()
    supported_dtypes: [fp16, bf16, fp8, int8, tf32]
    dtype_multipliers: {fp8: 2.0, int8: 2.0, tf32: 0.5}   # keys must be in supported_dtypes
    structured_sparsity_speedup: 2.0    # default 1.0
    systolic_dims: [16, 16]             # optional; drives the M3 tail-effect model
    weight_sets: 1                      # array-sized weight tiles ONE unit holds; default 1
    dataflow: ws                        # ws | os | rs, default ws

memory:                         # innermost first, level numbers ascending, no duplicates
  - {name: L1,   level: 1, capacity_bytes: 3.3792e+7, bandwidth_bytes_per_s: 1.3e+14, latency_ns: 30}
  - {name: L2,   level: 2, capacity_bytes: 5.0e+7,    bandwidth_bytes_per_s: 1.2e+13, latency_ns: 200}
  - {name: HBM3, level: 3, capacity_bytes: 8.0e+10,   bandwidth_bytes_per_s: 3.35e+12, latency_ns: 600}

usable_memory_fraction: 0.90    # optional; defaults to calibration.py
async_copy_engines: 4           # default 1
kernel_launch_overhead_s: 3.0e-6  # optional; defaults to calibration.py
tdp_w: 700
static_power_w: 120             # optional
cost_usd: 30000                 # optional

interconnect:                   # optional; unused until M5
  intra_node: {name: NVLink 4, bandwidth_bytes_per_s: 4.5e+11, latency_s: 2.0e-6, topology: fully_connected}
  inter_node: {name: IB NDR,   bandwidth_bytes_per_s: 5.0e+10, latency_s: 5.0e-6, topology: fat_tree}
```

**`weight_sets`** is how many array-sized weight tiles **one** unit holds at once, and it is a
statement about whether the array stores weights at all. Leave it at 1 for anything that reads its
operands per instruction — every NVIDIA and AMD profile here does, because a tensor core has no
persistent weight store. Set it only for in-memory compute, where a weight cannot join a MAC until
it has been written into a bank: `metis_aipu` declares 4, so its 4 AI cores hold 16 tiles and run
4 (`docs/CORRECTIONS.md` D30).

**Enums.** `dataflow`: `ws | os | rs`. `topology`: `fully_connected | ring | mesh | fat_tree |
switched`. `supported_dtypes` and `dtype_multipliers` keys: `fp32 | tf32 | fp16 | bf16 | fp8 |
int8 | int4`. **Not `int32`**: that is an accumulator width with no compute unit behind it (an
int8 product accumulates in int32 *at the int8 rate*), so it is legal only as a matmul's
`out_dtype`. No bundled profile declares it and nothing validates against it yet.

### What v1 actually reads

The machine model is three elements (`docs/CORRECTIONS.md` D5a):

| element | field | contributes |
|---|---|---|
| External memory | deepest `memory` entry | **bandwidth** — the only bandwidth ceiling |
| On-chip SRAM | all `memory` entries above the deepest | **capacity** — residency and double-buffering headroom |
| Compute | `compute_units`, `clock_ghz` | **TOPS** |

So `bandwidth_bytes_per_s` on the on-chip levels is schema-only in v1. It is required because
PROMPT.md §4.1 specifies it and the M8 hierarchical roofline will consume it, but no v1 result
depends on it. Say so in `estimates` if the value is a guess.

### Derived quantities

| method | formula |
|---|---|
| `peak_flops_per_s(dtype)` | `max` over units of `count x ops_per_cycle_per_unit x clock x multiplier x 2` |
| `ridge_point_flops_per_byte(dtype)` | `peak_flops_per_s / dram.bandwidth_bytes_per_s` |
| `on_chip_capacity_bytes` | sum of every level except the deepest; `0` for a single-level chip |
| `dram` / `on_chip` | deepest / shallowest memory level |

`peak_flops_per_s` takes the **max** over compute units, not the sum: a GEMM runs on the tensor
cores or the vector cores, not both, and vendor headline figures quote the dominant unit.

---

## 2. Model spec — `backend/profiles/models/<id>.yaml`

A discriminated union on `family`: `transformer_decoder | transformer_encoder | cnn | custom`.

### 2.1 Parametric transformer

```yaml
id: llama3_8b
name: Llama-3-8B
family: transformer_decoder
source_url: https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json
params:
  layers: 32
  hidden: 4096
  heads: 32
  kv_heads: 8              # optional, defaults to heads (i.e. MHA); must divide heads
  head_dim: 128            # optional, defaults to hidden // heads
  ffn_hidden: 14336
  ffn_type: swiglu         # relu | gelu | swiglu | geglu
  vocab: 128256
  max_context: 8192
  norm: rmsnorm            # layernorm | rmsnorm | batchnorm | none
  positional: rope         # learned | rope | alibi | none
  tie_embeddings: false
  sliding_window: 4096            # optional
  sliding_window_pattern: 6       # optional; requires sliding_window
declared_params: 1.0e+9    # optional, see Size presets
preset_of: gemma3_4b       # optional, requires declared_params
```

Field names match a HuggingFace `config.json` where one exists, so transcribing a model is
mechanical.

**Derived counts** (`docs/MODEL.md`):

```
q_width  = heads    x head_dim          # NOT hidden, when head_dim is explicit
kv_width = kv_heads x head_dim
attn/layer = hidden x q_width  +  2 x hidden x kv_width  +  q_width x hidden
ffn/layer  = n_matrices x hidden x ffn_hidden        # 3 for swiglu/geglu, 2 otherwise
norm/layer = 2 x (hidden for rmsnorm, 2 x hidden for layernorm)
params     = layers x (attn + ffn + norm) + embeddings + final_norm
embeddings = vocab x hidden, doubled when tie_embeddings is false
kv_bytes(C)= layers x 2 x kv_width x C x bytes_per_element x batch
```

**Size presets.** `declared_params` overrides the headline count when a profile is a size variant
whose hyperparameters do not sum to it. `headline_parameter_count()` returns it and drives weight
bytes `W = params x w_bytes`; `parameter_count()` always returns the derived figure, and M3 emits
an assumption when the two differ by more than 1% (`docs/CORRECTIONS.md` D8).

### 2.2 CNN

```yaml
id: mobilenetv3
family: cnn
source_url: https://arxiv.org/abs/1905.02244
input: {batch: 1, channels: 3, height: 224, width: 224}
layers:
  - {type: conv,   name: stem,   out_channels: 16, kernel: [3, 3], stride: 2, padding: 1, activation: hardswish}
  - {type: mbconv, name: bneck1, expand_channels: 16, out_channels: 16, kernel: [3, 3], stride: 1, squeeze_excite: false}
  - {type: pool,   name: avgpool, kind: global_avg}
  - {type: linear, name: classifier, out_features: 1000}
```

Layer `type` discriminates: `conv | mbconv | pool | linear`. `pool.kind`: `max | avg | global_avg`.
`activation`: `relu | relu6 | hardswish | swish | none`. A `conv` with `depthwise: true` (or
`groups` equal to the channel count) is a depthwise convolution.

`parameter_count()` **raises** on a CNN: the count depends on each layer's *input* channels, which
only exist once the M2 graph builder has propagated shapes. Returning a wrong number would be worse.

### 2.3 Custom op list

```yaml
id: my_workload
family: custom
source_url: https://example.com/where-these-came-from
ops:
  - {name: big_gemm, flops: 4.2e+12, bytes: 8.0e+9, weight_bytes: 4.0e+9, parallel_dims: {m: 4096}}
```

The escape hatch of PROMPT.md §4.2 — models anything the parametric flavours cannot express.

**A bare matmul is not this.** `CustomOp` carries hand-written FLOPs and bytes but no shape, so the
systolic tail effect cannot be computed for it and comes back as 100%. Use `family: matmul`
(§2.4), which carries `m`, `n`, `k` (`docs/CORRECTIONS.md` D17).

### 2.4 Matmul workload

```yaml
id: gemm_4096
family: matmul
hypothetical: true         # defaults true for this family: a synthetic shape has no source_url
m: 4096                    # rows of operand A; folds batch in
n: 4096                    # columns of operand B
k: 4096                    # contracted dimension
a_dtype: int8              # width of the M x K operand      (default fp16)
b_dtype: int8              # width of the K x N operand      (default fp16)
out_dtype: int32           # width of the M x N result       (default: the wider operand)
```

One `A[M,K] × B[K,N] → C[M,N]`, expanded into a single-operation graph. **Matmul vocabulary, not
transformer vocabulary** — operands A and B and a result C, with no weights/activations
asymmetry, because a bare matmul has none (`docs/CORRECTIONS.md` D18).

| Derived | Rule |
|---|---|
| `operand_dtype` | the **wider** of `a_dtype` and `b_dtype` — both enter the array through one datapath, so a narrow operand saves bytes and buys no throughput |
| `result_dtype` | `out_dtype`, else the wider operand |
| `parameter_count()` | `k · n`; a matmul has no parameters, but the base class asks for a count |

**`deployment.precision` is ignored for this family.** The widths are here, and there is no batch,
context, phase or attention implementation to describe — `M` folds the batch in. `analyze()` still
takes a `DeploymentSpec` because its signature is fixed; the builder discards it.

The result width is an **accumulator** width and changes bytes only, never operations:
`int8 × int8 → int32` performs the same `2·M·N·K` as `int8 × int8 → int8` and writes four bytes
per result instead of one. See `docs/MODEL.md` §5b for the worked table.

---

## 3. Deployment spec

Not a bundled profile; passed on the CLI or loaded from a file with `load_deployment`.

```yaml
mode: inference            # inference | training
phase: both                # prefill | decode | both
batch: 1
input_tokens: 2048
output_tokens: 256
kv_context_tokens: 4096    # optional; defaults to input_tokens + output_tokens
precision:
  weights: int8            # fp32 | tf32 | fp16 | bf16 | fp8 | int8 | int4 | int32
  activations: fp16
  accumulate: fp32         # DECLARED BUT NOT READ -- see below
  kv_cache: fp16
per_layer_precision_overrides:
  "layer.0.attn.qkv": {weights: fp16}
sparsity: {type: structured_2_4, ratio: 0.5, applies_to: [matmul]}
attention_impl: flash2     # vanilla | flash2 | paged | sliding_window
parallelism: {tp: 2, pp: 1, dp: 1, ep: 1, microbatches: 8}
num_chips: 2
optimize_for: latency      # latency | throughput | energy
constraints: {max_latency_s: 0.05, max_power_w: 700, max_memory_bytes: 8.0e+10}
a_strategy: stage           # stage | stream | whole — a lone matmul's A residency (D33/D36)
b_dataflow: write-ahead     # write-ahead | on-demand | persistent — when B's array write lands
a_residency_tiles: null     # power-of-2 divisor of NTILES_PER_KS; null = the whole k-slice
a_prefetch_depth: null      # schedule-only; null = derived from double buffering, as today
iterations: 1                # invocations this report represents; only persistent reads it
```

**Validated here (intra-spec):** `tp x pp x dp x ep == num_chips`; decode requires
`output_tokens > 0`; training has no phase split; `pp > 1` needs at least `pp` microbatches;
`structured_2_4` implies ratio 0.5; `a_residency_tiles`, `a_prefetch_depth` and `iterations` are
each `> 0` when set.

**`a_strategy`, `b_dataflow` and `iterations` are single-matmul-only knobs** (`bwz matmul`,
`scripts/plot_pipeline.py --matmul`): `analysis/schedule.py` reads them only when the graph is one
bare matmul, so they are silently inert on a network, whose activations are already governed by
inter-operation residency (D15). Feasibility here is a **clamp, not a validation error**
(CLAUDE.md #8): `a_strategy: whole` that does not fit the scratchpad falls back to `stage`, and
`b_dataflow: persistent` that does not fit the array's resident tile capacity falls back to
`write-ahead` — both noted in `report.assumptions`, never raised. See `docs/MODEL.md` §6.3a and
`docs/CORRECTIONS.md` D36.

**`precision.accumulate` is not read by anything.** The transformer and CNN builders size output
tensors at the activation dtype, so declaring `accumulate: fp32` against `activations: fp16` does
not add the bytes an fp32 accumulator would move. Only `family: matmul` models a widening
accumulator, through its own `out_dtype` (§2.4). Closing the gap for networks means changing every
activation-traffic number in the repo and has not been done.

**Not validated here.** Whether the chip supports the requested dtype, or whether the weights fit,
is a *feasibility* question. M3 answers it with a `Report` carrying `feasible: false` and the
cheapest fixes — never an exception, never a 500 (CLAUDE.md #8).

---

## 4. Provenance

Three declarations, from `docs/CORRECTIONS.md` D6 and D7:

| field | meaning |
|---|---|
| `source_url` | Required for any profile describing a real product. Datasheet, HF config, or paper. |
| `hypothetical: true` | This is not a product. Permits omitting `source_url`. Flagged in `bwz list` and in every report the profile appears in. |
| `derived_from: <id>` | What a hypothetical profile was varied from. Only valid alongside `hypothetical`. |
| `estimates: {<field>: <why>}` | Per-field: this value is an engineering estimate, not a published figure. Keys must name real fields; dotted paths like `memory.0.bandwidth_bytes_per_s` are allowed. |

Every estimate the analysis *touches* is propagated into `report.assumptions`, so a report states
which of its inputs were guessed and why. This is the mechanism that lets a profile like `chip_a`
ship honestly rather than not at all.

---

## 5. Validation errors

A failure names the file, the field path, the offending value, and the allowed range or set:

```
bwz: cannot load chip profile 'tiny' (/tmp/tiny.yaml)
  memory.0.bandwidth_bytes_per_s: Input should be greater than 0
    got: -5.0 allowed: gt=0
```

Enum failures list the permitted values; missing fields say `this field is required`. If an error
does not tell you what to change, that is a bug worth filing.
