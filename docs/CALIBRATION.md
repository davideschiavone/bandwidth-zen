# Calibration and validation

Predicted-vs-published comparison tables, fitted constants, and error analysis.

**Rules (CLAUDE.md):** every number here comes from a citable published source (MLPerf Inference
results, vendor datasheets/blogs, peer-reviewed papers). Where no reference exists, the entry reads
"no reference point available" — never a plausible guess. Fabricated validation data would make the
whole project worthless, so an empty table is the correct output until it isn't.

---

## Status: no reference points collected

```bash
$ make validate
459 deselected in 0.33s
```

`tests/validation/` is reserved for published reference points and currently holds none. Every
prediction the engine makes therefore rests on **documented defaults, not fitted values**, and
`analysis/_confidence()` returns `LOW` unconditionally as a result — see `docs/CORRECTIONS.md` D7
for why that is enforced in code rather than left to the reader.

The self-consistency suite is a different thing and does run: 459 tests in `tests/unit` and
`tests/integration`, including the `CLAUDE.md` sanity checks and hand-computed goldens. Those prove
the engine agrees with **itself and with its own stated physics**. They say nothing about whether it
agrees with silicon.

## The three unfitted constants

All in `backend/bwz/calibration.py`, each with a source comment. `--ideal` sets all three aside so a
number can be checked against a datasheet by hand (`docs/CLI.md` §2.1):

| constant | default | what it stands for | how it will be fitted |
|---|---|---|---|
| `DEFAULT_DRAM_BANDWIDTH_EFFICIENCY` | 0.85 | achievable fraction of pin bandwidth | regression against measured decode throughput, where DRAM binds |
| `DEFAULT_ACHIEVED_FLOPS_FRACTION` | 0.70 | issue/occupancy losses the shape model does not capture | regression against measured prefill throughput, where compute binds |
| `DEFAULT_KERNEL_LAUNCH_OVERHEAD_S` | 3 µs | per-dispatch fixed cost | small-batch CNN latency, where it dominates |

Splitting the fit by binding term is deliberate: a point where DRAM binds carries no information
about the compute derating, and fitting both against one aggregate latency would let either absorb
the other's error.

**Shape utilisation is not on this list and is not a calibration constant.** It follows from the
array geometry the profile declares — `docs/MODEL.md` §6.1 — and `--ideal` leaves it in place. A
batch-1 GEMM on a 512×512 array runs at `1/512` of that array on ideal silicon too.

## Planned sources

To be verified at collection time; a candidate that turns out not to publish an exact configuration
is dropped rather than approximated.

- MLPerf Inference v3.1/v4.0 — Llama-2-70B on H100 and MI300X (server + offline)
- NVIDIA / AMD vendor benchmark blogs — Llama-3-8B, Mistral-7B on H100 and MI300X
- Community vLLM benchmark reports — secondary points, weighted lower

`chip_a` and `chip_b` have no published benchmarks and will contribute **none**. They inherit the
constants fitted on H100/A100/MI300X and stay explicitly uncalibrated; every report they produce
says so through the `estimates:` propagation (D7) and the flip-margin field.

`metis_aipu` is a partial case worth stating separately: its compute and on-chip capacity come from
an ISSCC 2024 paper and are exact, while its DRAM bandwidth is an inference from a 64-bit LPDDR4x
bus at the JEDEC maximum and is declared in `estimates:`. A prediction of its DRAM-bound decode
therefore rests on an unpublished input, which the flip margin reports on every run.

## Fitted constants

None yet. Each fitted value will be documented here with the dataset it was fitted against, the
residuals, and the outliers it does not explain.

## What would invalidate the model rather than the constants

Worth writing down before there is data, so it cannot be rationalised afterwards. The constants
above are scalars: they can absorb a uniform error, not a structural one. A systematic residual that
**varies with a dimension** points at the physics instead:

- error growing with problem size → the compulsory-traffic assumption (`docs/MODEL.md` §6.2); v1
  charges each operand once and is a lower bound once the working set stops fitting on chip
- error concentrated at small M → the shape-utilisation model (§6.1)
- error that differs between chips of the same class → a profile input, not a constant
- error concentrated on one operator family → that family's cost model in `operators/`
