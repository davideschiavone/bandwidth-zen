# Figures

Every figure here is **computed by calling `analyze()`**, not drawn by hand, and carries the
command that produced it plus the bwz version and commit. If a number in the engine changes, these
change with it — that is the point of generating them rather than illustrating them.

```bash
make plots        # regenerates everything below
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
| `timeline-<chip>-<dtype>.html` | `plot_pipeline.py --html` | The same, **zoomable**, with the roofline for that run below it |
| `timeline-<chip>-<model>-<phase>-<dtype>.*` | `plot_pipeline.py --model` | A network instead of a matmul, one figure per phase |
| `timeline-compare-<a>-vs-<b>-….{png,html}` | `plot_pipeline.py --compare` | Two chips, one workload, **one shared absolute time axis**, plus both rooflines |

**Figures are not the report.** These show where the time went; the numbers, their derivations and
the assumptions drawer come from `bwz matmul` / `bwz run`, and the plot scripts print only
`wrote …`. See [`../CLI.md`](../CLI.md) §2 and §3.

## The zoomable timeline

One self-contained file — no server, no port, no download, no CDN. `file://` is enough:

```bash
xdg-open docs/plots/timeline-a100_80gb-fp16.html
# or: google-chrome docs/plots/timeline-a100_80gb-fp16.html
```

`xdg-open` prints nothing and hands the file to a browser that may already be running, so look for
a **new tab in an existing window** rather than a new window.

- **wheel** zooms about the cursor · **drag** pans · **double-click** resets
- **hover** a bar and the tooltip names the transaction: `LOAD — operands in`,
  `STORE — result written back`, `EXEC — matmul`, `HOLD — on chip`, each with its bytes or
  operations and the rate
- below the timeline, the same run's place on the **roofline**: both ceilings, the ridge point, the
  M=1 tail line, and this workload as a labelled point
- below that, **the arithmetic operation by operation** — operand shapes, the algebra, the flop
  count as an expression (`2·M·N·K = 2·4·8·8 = 512`) and a pseudo-C loop nest with the real extents,
  so the model can be back-tested against code rather than trusted

It carries 256 steps against the PNG's 32 (`--html-steps`), because a static figure has to stay
legible at one scale and a zoomable one does not.

Zoom is x-only: the y axis is a list of resources, not a scale.

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

**Rows are resources, read off the chip profile** — every memory level and every compute unit it
declares, not one row per step. That is the axis a comparison needs: "what was the memory system
doing while the array worked" rather than "what happened to this tile".

Three info boxes carry the headline quantities, and each row repeats its own share to the right:

- **MOVED OVER DRAM** — `LOAD B 67.1 MB · A 36.7 MB` / `STORE C 67.1 MB`, the rate while active, and
  the share of the span. Direction is named because a store happens after the arithmetic that
  produced it and a result nothing consumes must be written (D22); **operand** is named because A
  and B obey different residency fractions and spill at different times — capacity goes to
  activations before weights (D15), so B is always the first to stream (D31). On the row itself:
  operand B **solid**, operand A **hatched**, the result **hollow**.
- **HELD ON CHIP** — bytes, in how many buffers, against capacity.
- **COMPUTED — matmul 98% · attention 2%** — the operator families that did the arithmetic,
  biggest first. "15.3 GOP" does not say whether that was one matmul or a decode step's worth of
  matmul, attention and norms, and for a comparison the mixture is the point.

**Grey rows are declared by the chip and unused by this model.** A100 has 40 MB of L2 and 6912 CUDA
cores that the v1 roofline never spends; drawing them idle puts the model's boundary on the page
instead of hiding it. Next to `chip_a`, whose 55 MB of SRAM holds all of operand B and whose DRAM
row reads `LOAD 0 B · STORE 16.8 MB` — it fetches nothing because 55 MB of SRAM holds all of B, and
still has to write the answer out — that contrast *is* the architecture comparison.

The right-hand figure carries a different unit per lane. DRAM and the array are single serial
resources, so theirs is a **duty cycle** — the fraction of the span they were busy, never above
100%. SRAM is *n* buffers rather than a resource that is busy or idle, so theirs is a **depth**:
`x1.96 of 2` means both halves of the double buffer were occupied almost all the time. Capacity,
not bandwidth, is what SRAM contributes (D5a).

**A bar on a compute row is a wave, not a tile.** A chip with `count` arrays runs that many weight
tiles at once, so the trace draws `ceil(tiles / units)` steps and labels each with how many tiles are
in flight — `8 B tiles 512x512 [3/64]  all in parallel` on Metis's four AI cores. Drawing one bar per
tile said the arrays worked in series and contradicted the utilisation in the same report (D30/D31).
When a bar has to coalesce several waves the qualifier becomes `4 at a time` instead.

The x axis is normalised to the total time in both registers, which is what makes two chips
comparable when their absolute times differ by orders of magnitude; absolute figures are on the
ticks.

The spans are a **decomposition** of the reported latency, not a second model — DRAM busy sums to
`t_dram`, core busy to `t_compute + t_fixed`, and the tile count is the same one the utilisation
figure divides by. The one thing the picture adds is pipeline fill/drain, which the roofline's
`max(load, compute)` omits; it is stated on the figure rather than folded in. See
[`../MODEL.md`](../MODEL.md) §6.5 and [`../CORRECTIONS.md`](../CORRECTIONS.md) D19–D22.

## Comparing two chips

```bash
cd backend
uv run --group plots python scripts/plot_pipeline.py \
  --chip a100_80gb --chip metis_aipu --compare --model gemma3_4b -S 512 --html
```

writes `timeline-compare-a100_80gb-vs-metis_aipu-gemma3_4b-{prefill,decode}-int8.{png,html}`.
Comparison figures are **not committed** — `make plots` builds them, and the numbers quoted below
come from that run.

Three things differ from the per-chip figure, and nothing else does (`../CORRECTIONS.md` D29).

**The time axis is shared and absolute.** A bar three times as long took three times as long. On
Gemma-3-4B prefill at S=512, int8: A100 9.06 ms against Metis 119 ms, and the subtitle states the
13.15x rather than leaving it to be measured off the ticks. The per-chip normalised view is kept,
because it is the better view of one machine's internal balance — at 13x, the faster chip's whole
run is 7% of a shared axis.

**Rows are banded by chip, not aligned across chips.** They cannot be aligned: A100 declares 3
memory levels and 2 compute units, Metis 4 and 2, `chip_a` 2 and 1, and there is no honest
correspondence between `cuda_core` and `dpu`. Each band opens with a header row naming the machine
and carrying its peak, DRAM bandwidth, on-chip capacity, its row counts and its total span with its
verdict. Inside a band everything the single-chip figure does still holds.

**A roofline register is added below.** The timeline shows what happened; this shows why it had to.
Both chips do 3.33 TOP over ~3.8 GB — the same ~870 OP/byte — and land on opposite sides of their
own ridge:

| | peak | DRAM | ridge | this run | verdict |
|---|---|---|---|---|---|
| `a100_80gb` | 624 TOP/s | 2.04 TB/s | 306 OP/byte | 870.4 | **compute bound** |
| `metis_aipu` | 210 TOP/s | 34.1 GB/s | 6145 OP/byte | 868.9 | **DRAM-bw bound** |

Identical arithmetic, identical traffic, different limiter — which is the thing a table of
latencies cannot show. Inside that panel **colour is the chip**, not the resource, which is stated
on the panel; everywhere else in the figure colour keeps its lane meaning.

The HTML comparison carries the same bands and the same shared axis, keeps wheel-zoom, drag-pan,
double-click reset and per-bar hover, and renders **the arithmetic section once** — the operator
list is a property of the workload, and the workload is the same on both machines.

## Adding a chip

Both scripts take `--chip` and default to a pair that makes the contrast visible — a datacentre GPU
and an edge NPU. The schedule is a property of the machine, so a new chip means a new figure rather
than a new line on an existing one; `--compare` is the exception, and it bands rather than merges.
