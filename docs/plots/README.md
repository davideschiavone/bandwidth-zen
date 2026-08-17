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
| `machine-<chip>.png` | `plot_roofline.py` | The three-element machine (D5a) — DRAM, SRAM-as-capacity, array — and which link carries a bandwidth number |
| `timeline-<chip>-<dtype>.png` | `plot_pipeline.py` | Where the time went, per hardware resource, with the bytes and operations on each row |
| `timeline-<chip>-<dtype>.html` | `plot_pipeline.py --html` | The same, **zoomable**: wheel to zoom about the cursor, drag to pan, hover a bar for its own numbers |

The HTML is one self-contained file — no server, no download, no CDN. Open it with `file://`, or:

```bash
xdg-open docs/plots/timeline-a100_80gb-fp16.html
```

It carries more steps than the PNG (256 against 32) because a static figure has to stay legible at
one scale and a zoomable one does not.

Nothing in `bwz/` imports a plotting library (CLAUDE.md #3). The scripts live in
`backend/scripts/` and import the engine; the dependency never points the other way. matplotlib is
in its own `plots` dependency group, so `make test` does not pull it in.

## What each one is

| File | Script | Shows |
|---|---|---|
| `roofline-<chip>-<dtype>.png` | `plot_roofline.py` | The two ceilings, the ridge point, the M=1 tail line, and a set of workloads placed on them |
| `machine-<chip>.png` | `plot_roofline.py` | The three-element machine (D5a) — DRAM, SRAM-as-capacity, array — and which link carries a bandwidth number |
| `timeline-<chip>-<dtype>.png` | `plot_pipeline.py` | Where the time went, per hardware resource, with the bytes and operations on each row |
| `timeline-<chip>-<dtype>.html` | `plot_pipeline.py --html` | The same, **zoomable**: wheel to zoom about the cursor, drag to pan, hover a bar for its own numbers |

The HTML is one self-contained file — no server, no download, no CDN. Open it with `file://`, or:

```bash
xdg-open docs/plots/timeline-a100_80gb-fp16.html
```

It carries more steps than the PNG (256 against 32) because a static figure has to stay legible at
one scale and a zoomable one does not.

Nothing in `bwz/` imports a plotting library (CLAUDE.md #3). The scripts live in
`backend/scripts/` and import the engine; the dependency never points the other way. matplotlib is
in its own `plots` dependency group, so `make test` does not pull it in.

## What each one is

| File | Script | Shows |
|---|---|---|
| `roofline-<chip>-<dtype>.png` | `plot_roofline.py` | The two ceilings, the ridge point, the M=1 tail line, and a set of workloads placed on them |
| `machine-<chip>.png` | `plot_roofline.py` | The three-element machine (D5a) — DRAM, SRAM-as-capacity, array — and which link carries a bandwidth number |
| `timeline-<chip>-<dtype>.png` | `plot_pipeline.py` | Where the time went, per hardware resource, with the bytes and operations on each row |
| `timeline-<chip>-<dtype>.html` | `plot_pipeline.py --html` | The same, **zoomable**: wheel to zoom about the cursor, drag to pan, hover a bar for its own numbers |

The HTML is one self-contained file — no server, no download, no CDN. Open it with `file://`, or:

```bash
xdg-open docs/plots/timeline-a100_80gb-fp16.html
```

It carries more steps than the PNG (256 against 32) because a static figure has to stay legible at
one scale and a zoomable one does not.

Nothing in `bwz/` imports a plotting library (CLAUDE.md #3). The scripts live in
`backend/scripts/` and import the engine; the dependency never points the other way. matplotlib is
in its own `plots` dependency group, so `make test` does not pull it in.

## What each one is

| File | Script | Shows |
|---|---|---|
| `roofline-<chip>-<dtype>.png` | `plot_roofline.py` | The two ceilings, the ridge point, the M=1 tail line, and a set of workloads placed on them |
| `machine-<chip>.png` | `plot_roofline.py` | The three-element machine (D5a) — DRAM, SRAM-as-capacity, array — and which link carries a bandwidth number |
| `timeline-<chip>-<dtype>.png` | `plot_pipeline.py` | Where the time went, per hardware resource, with the bytes and operations on each row |
| `timeline-<chip>-<dtype>.html` | `plot_pipeline.py --html` | The same, **zoomable**: wheel to zoom about the cursor, drag to pan, hover a bar for its own numbers |

The HTML is one self-contained file — no server, no download, no CDN. Open it with `file://`, or:

```bash
xdg-open docs/plots/timeline-a100_80gb-fp16.html
```

It carries more steps than the PNG (256 against 32) because a static figure has to stay legible at
one scale and a zoomable one does not.

## Reading the roofline

Solid roof = datasheet, which is what `--ideal` reports. Dashed roof = the same machine after the
two `calibration.py` constants nobody has fitted yet; the gap between the pair **is** the unfitted
part of any prediction. A profile that declares both efficiencies as 1.0 (chip_a does) gets one
roof and says so.

The x axis is intensity against **DRAM** traffic, not compulsory traffic, so residency moves a
point right and a fully resident workload leaves the chart entirely.

The dotted `M=1` line is **not** a derating — it is `peak/(1+rows)`, geometry from the array's
declared depth, and `--ideal` leaves it exactly where it is.

## Reading the timeline

**Rows are resources, read off the chip profile** — every memory level and every compute unit the
profile declares, not one row per step. That is the axis a comparison needs: "what was the memory
system doing while the array worked" rather than "what happened to this tile".

Three info boxes carry the headline quantities — bytes copied from DRAM and at what rate, bytes
held on chip and in how many buffers, operations computed and at what fraction of peak — and each
row repeats its own share to the right.

**Grey rows are declared by the chip and unused by this model.** A100 has 40 MB of L2 and 6912 CUDA
cores that the v1 roofline never spends; drawing them idle puts the model's boundary on the page
instead of hiding it. Next to `chip_a`, whose 55 MB of SRAM holds all of operand B and whose DRAM
row reads `0 B — nothing crossed`, that contrast *is* the architecture comparison.

The x axis is normalised to the total time in both registers, which is what makes two chips
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
