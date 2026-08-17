# Figures

Every figure here is **computed by calling `analyze()`**, not drawn by hand, and carries the
command that produced it plus the bwz version and commit. If a number in the engine changes, these
change with it — that is the point of generating them rather than illustrating them.

Regenerate all of them:

```bash
make plots
```

Individual invocations and every script flag are in [`../CLI.md`](../CLI.md) §5.

Nothing in `bwz/` imports a plotting library (CLAUDE.md #3). The scripts live in
`backend/scripts/` and import the engine; the dependency never points the other way. matplotlib is
in its own `plots` dependency group, so `make test` does not pull it in.

## What each one is

| File | Script | Shows |
|---|---|---|
| `roofline-<chip>-<dtype>.png` | `plot_roofline.py` | The two ceilings, the ridge point, the M=1 tail line, and a set of workloads placed on them |
| `pipeline-<chip>.png` | `plot_roofline.py` | The three-element machine (D5a) — DRAM, SRAM-as-capacity, array — and which link carries a bandwidth number |
| `pipeline-matmul-<chip>-<dtype>.png` | `plot_pipeline.py` | The tile schedule: which resource is busy when, at total scale and zoomed |
| `pipeline-matmul-<chip>-<dtype>.kanata` | `plot_pipeline.py --kanata` | The same schedule as a Kanata log, for [Konata](https://github.com/shioyadan/Konata) |

Open either `.kanata` file — or both at once, side by side — with:

```bash
make konata TRACE=docs/plots/pipeline-matmul-a100_80gb-fp16.kanata
```

The first run fetches a pinned Konata release into `~/.cache/bandwidth-zen/`; see
[`../CLI.md`](../CLI.md) §6.

## Reading the roofline

Solid roof = datasheet, which is what `--ideal` reports. Dashed roof = the same machine after the
two `calibration.py` constants nobody has fitted yet; the gap between the pair **is** the unfitted
part of any prediction. A profile that declares both efficiencies as 1.0 (chip_a does) gets one
roof and says so.

The x axis is intensity against **DRAM** traffic, not compulsory traffic, so residency moves a
point right and a fully resident workload leaves the chart entirely.

The dotted `M=1` line is **not** a derating — it is `peak/(1+rows)`, geometry from the array's
declared depth, and `--ideal` leaves it exactly where it is.

## Reading the pipeline

Three lanes: DRAM (the one modelled link), on-chip SRAM (capacity — tiles in flight), and the
array. The x axis is normalised to the total time in both registers, which is what makes two chips
comparable when their absolute times differ by orders of magnitude; absolute figures are on the
ticks.

The right-hand figure carries a different unit per lane. DRAM and the array are single serial
resources, so theirs is a **duty cycle** — the fraction of the span they were busy, never above
100%. SRAM is *n* buffers rather than a resource that is busy or idle, so theirs is a **depth**:
`x1.96 of 2` means both halves of the double buffer were occupied almost all the time. Capacity,
not bandwidth, is what SRAM contributes (D5a).

The spans are a **decomposition** of the reported latency, not a second model — DRAM busy sums to
`t_dram`, core busy to `t_compute + t_fixed`, and the tile count is the same one the utilisation
figure divides by. The one thing the picture adds is pipeline fill/drain, which the roofline's
`max(load, compute)` omits; it is stated on the figure rather than folded in. See `docs/MODEL.md`
§6.5 and `docs/CORRECTIONS.md` D19.

## Adding a chip

Both scripts take `--chip` and default to a pair that makes the contrast visible — a datacentre GPU
and an edge NPU. The schedule is a property of the machine, so a new chip means a new figure rather
than a new line on an existing one.
