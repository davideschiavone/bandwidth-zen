# Figures

Every figure here is **computed by calling `analyze()`**, not drawn by hand, and carries the
command that produced it plus the bwz version and commit. If a number in the engine changes, these
change with it — that is the point of generating them rather than illustrating them.

```bash
make plots        # regenerates everything below
```

Individual invocations and every script flag are in [`../CLI.md`](../CLI.md) §6.

Nothing in `bwz/` imports a plotting library (CLAUDE.md #3). `scripts/plot_roofline.py` needs
matplotlib and is kept in its own `plots` dependency group so `make test` does not pull it in. The
HTML pages need nothing beyond the standard library — D37 removed the last thing that imported
matplotlib from them — which is why they live in `bwz/figures/` and are reached from the report
commands themselves rather than from a separate script (D55).

## What each one is

| File | Script | Shows |
|---|---|---|
| `roofline-<chip>-<dtype>.png` | `plot_roofline.py` | The two ceilings, the ridge point, the M=1 line, and a set of workloads placed on them |
| `machine-<chip>.png` | `plot_roofline.py` | The three-element machine (D5a) — DRAM, SRAM-as-capacity, array — and which link carries a bandwidth number |
| `timeline-<chip>-<dtype>.html` | `bwz matmul --timeline` | Where the time went, per hardware resource, **zoomable**, with the roofline for that run below it |
| `timeline-<chip>-<model>-<phase>-<dtype>.html` | `bwz run --timeline` | A network instead of a matmul, one page per phase |
| `timeline-compare-<a>-vs-<b>-….html` | any of them, `--compare-with CHIP` | Two chips, one workload, **one shared absolute time axis**, plus both rooflines |
| `animate-<chip>-<dtype>.html` | `bwz matmul --animate` | A matmul's or the encoder's schedule, **played back** as DRAM -> SRAM -> Accelerator motion. Not on `bwz run`, not with `--compare-with`, opt-in — not part of `make plots` |

**Figures are not the report.** These show where the time went; the numbers, their derivations and
the assumptions drawer come from `bwz matmul` / `bwz run`, and the plot scripts print only
`wrote …`. See [`../CLI.md`](../CLI.md) §2 and §4.

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
- a **banner** naming the decomposition the bars are of: the stationarity, which operand it keeps
  resident, the grid's extents and which of M/N/K each axis is, plus any split-K (D53). Tile
  addresses in the hovers index into exactly that grid, so without it they cannot be read
- below the timeline, the same run's place on the **roofline**: both ceilings, the ridge point, the
  M=1 line, and this workload as a labelled point
- below that, **how it is deployed on the chip — run it yourself**: the loop nest this model
  actually schedules, **per chip**, as a **runnable Python program** rather than a description of
  one (`../CLI.md` §2.6, D54). It walks the grid the timeline draws, stages A on the same events,
  hands tiles to cores the same way, counts what it moves and asserts those counts against this
  page's own numbers — so it cannot narrate a schedule the model did not cost, and a reader can
  run it, edit it and break it. Which loop nest this *is* comes from the chip's stationarity
  (`--stationarity`, `../CLI.md` §2.5.1, D53) — `os` accumulates K inside the tile, `ws` streams M
  past a resident one and needs a shared accumulator the `os` file has no use for — with
  `--a-strategy`/`--b-dataflow` (§2.5.2, D36) choosing how the operands move within it, and a
  second kernel for split-K's reduction when there is one. Rendered once per chip, unlike the
  arithmetic, because the mapping is exactly what differs between two machines; `--compare` is
  where it earns its keep, giving two loop nests, two staging counts and the same `C`. `--emit`
  also saves the file beside the page, ready to run. It validates **counts, not time** — timing it
  against the predicted latency is a category error, and the file says so before anything else
- a **network** has no tile grid to walk and so no program to emit: its operations run in strict
  sequence (D5a). That page keeps a short pseudo-C listing of the sequence instead, checked
  against the schedule the same way (D32)
- below that, **the arithmetic operation by operation** — operand shapes, the algebra, the flop
  count as an expression (`2·M·N·K = 2·4·8·8 = 512`) and a pseudo-C loop nest with the real extents,
  so the model can be back-tested against code rather than trusted

`--steps` (default 256) sets the trace's resolution — the only such knob, since the page zooms
rather than needing a second, coarser register kept legible at a fixed scale (D37).

Zoom is x-only: the y axis is a list of resources, not a scale.

## Playing the flow animation

`--animate` writes a second, self-contained page — same one-file, no-CDN constraint — that plays
the same schedule the timeline draws, as motion between stations instead of a static strip of bars.
Stations are `rows_for`'s own resource list (D43) — the same one the timeline draws as rows, one per
declared memory level and compute unit, grey for what v1 doesn't cost (D20) — not a fixed shape:
Metis gets 6 (LPDDR4x, L2, L1, D-IMC, `d_imc`, `dpu`), A100 gets 5 (HBM2e, L2, L1, `tensor_core`,
`cuda_core`):

```bash
uv run bwz matmul -M 2048 -N 2048 -K 2048 -c metis_aipu -d int8 \
    --b-dataflow on-demand --animate --out /tmp/anim
xdg-open /tmp/anim/animate-metis_aipu-int8.html
```

`bwz matmul` or `bwz encoder-layer`, and opt-in — it never runs as part of `make plots`. Not on
`bwz run`, and refused with `--compare-with`: a full model's per-operation trace can coalesce
hundreds of operations, well past what a resource-station diagram or a debug pane can usefully
show (D42), and a playback follows one chip's schedule.

**Not to scale, deliberately.** Block size is a log-compressed function of each event's own bytes,
so the smallest and largest tiles in one trace both stay visible — reading a size off the page as a
literal byte count would be wrong by design. The **timing is not**: play/pause/scrub/next drive a
virtual clock through the trace's real `start_s`/`end_s` values, and the live panel underneath
states the current time, the reported latency, and any pipeline fill/drain this trace's schedule
shows (D19) — never the other way around. **Next** jumps to the next moment the active-event set
actually changes — a span starting or ending, not merely the next individual event's own start —
stepping through load/hold/exec/store transitions one at a time, debugger-style, for reading a
schedule by hand rather than watching it play.

- solid blocks are operand B, hatched blocks are operand A streaming (`--a-strategy stream`, D31),
  hollow blocks are the result written back
- grey stations (declared, not modelled — same rule as the timeline's grey rows, D20) hover to show
  why, never glow, and are never a motion target — a block travelling DRAM->SRAM through Metis's
  grey L2/L1 just crosses their position without stopping there
- every station with a lane **glows independently** while its engine executes; nothing visibly
  "enters" it, because the byte/flop model has no event distinct from the arithmetic itself for that
  moment — and two engines never glow together, because operations run in strict sequence in this
  model (D5a/D43)
- `--stationarity`/`--split-k` change the grid everything else is drawn against (D53): the banner
  names the effective one, the geometry panel draws the resident operand's grid, and `--split-k`
  adds a second kernel — a DRAM round trip and a vector-unit block — after the tile schedule ends
- `--a-strategy` changes what you see directly — `stream` trickles many small hatched blocks,
  `stage`/`whole` concentrate them at grid-row boundaries — because A's schedule already differs by
  strategy (D33/D31)
- `--b-dataflow on-demand` now shows a real gap between a load starting to move and the previous
  wave's compute station glow ending; `persistent` plays identically to `write-ahead`, and the page
  says so, because that is what the schedule actually does in one pass (D40) — a future extension
  animating `--iterations` repeats back to back is the only way `persistent`'s real advantage (its
  2nd+ pass B load genuinely vanishing) would show up in motion

**A code pane plays alongside the diagram, debugger-style (D41).** For a matmul it is the *runnable
program* the timeline page carries (D54) — the same file `--emit` saves — and the lines that light
up are statements that perform the transfer: `dram.read_b(...)` for the B load, `mma(...)` for the
arithmetic, `dram.write_c(...)` for the store. More than one lights at once exactly when double
buffering means more than one statement is truly concurrent: watch the staging block and the `mma`
line glow together while one block is mid-flight toward a compute station and another slides in
from DRAM. Not every line is highlightable — a statement no event in this model times would be
decoration rather than data.

**`--encoder`'s code pane is honest about a real limit, not a smaller version of the matmul one
(D42/D43).** A network's loop is generic — `for (i = 0; i < OPS; ++i) { load_B(op[i]); ...;
store_C(op[i]); }` — never unrolled per named operation, so the pane can show "a load is happening"
but not "q_proj's load is happening"; the blocks and the hover text *do* carry the real operation
name (`Span.label`), the listing just doesn't. Cross-operation overlap is zero in this model (D5a:
operations run in strict sequence) — the highlight only ever lights up lines together within one
operation's own load/compute, e.g. a norm's `load_A`/`exec` overlapping because that operation's own
schedule was double buffered, never two different operations' lines at once. The two compute
branches (matrix array vs. vector unit, D27) tag *distinct* lines, `exec_core`/`exec_vector` (D43) —
so the matrix and vector stations now glow, and their code lines highlight, independently, matching
which engine a given operation actually ran on rather than lighting both every time.

**A tile-geometry panel shows A/B/C's own shapes for a lone matmul (D48).** Below the flow diagram,
three schematic rectangles — A (`M x K`), B (`K x N`), C (`M x N`) — in the classic GEMM layout, so
the axes A and B share (K) and the axes B and C share (N) line up visually instead of reading as
three unrelated boxes. **Which operand has the 2-D grid follows from the stationarity** (D53), not
from a fixed assumption: the grid has a row axis and a column axis, every tile sweeps the third
dimension in full, and an operand is cut along a dimension exactly when the grid carries it. Under
`ws` that makes B the 2-D one, A a set of k-slice stripes and C a set of column bands; under `os`
(what every matrix core now declares) C is the 2-D one, A is banded along M and B along N. The lit
cell tracks whichever operand the current instant actually touches: A lights on a `load_a` event (A
being staged), the **resident** operand on `exec` (the tile in the array right now), C on `store`
(the result landing) — a coalesced frame that covers many real tiles at once lights the true range
it covers, never one fake single index. The dark info box names the stationarity, each operand's
tile size as rows x cols and the grid's own axes, updating live as playback moves.
Grid lines are capped at roughly 40 per axis; past that a coarser stride draws instead, and the
caption says the real count and stride — never a silent truncation. Not rendered for `--encoder`:
a network's per-operation trace has no single A/B tile grid to draw (D42's own documented limit).

**The same `Operand(row,col)` notation now appears in the ordinary hover too**, on both this page
and the static timeline — not just the geometry panel's own caption. An index is the row number if
the grid carries that dimension on its rows, the column range if on its columns, and `:` if the
tiles sweep it — so the same rule prints `B(row,col)`/`C(:,col)` under `ws` and `C(row,col)`/
`B(:,col)` under `os` without either being special-cased (D53). Hovering a staging or tile bar names
its exact position, and when one step opens several bands at once (routine whenever a step's real
tiles exceed one grid row's width) the label reads "A row-bands 1-6/63" rather than naming only the
first, per the byte total it always correctly charged.

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
  matmul, attention and norms, and for a comparison the mixture is the point. Its first line is the
  **achieved throughput** — `186 TOP/s achieved · 89% of peak` on Metis for an 8192³ INT8 matmul —
  from `PhaseResult.achieved_flops_per_s`, the reported latency, never the drawn span (D19, D35,
  D38). The `array … @ 210 TOP/s while busy` line beneath it is a different, also true number: the
  rate while the array specifically had a tile, which a DRAM-bound run can hold near peak even while
  the chip's delivered rate sits far below it.

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
uv run bwz run --model gemma3_4b -c a100_80gb --compare-with metis_aipu \
  --weights int8 --input-tokens 512 --output-tokens 1 --timeline -q
```

writes `timeline-compare-a100_80gb-vs-metis_aipu-gemma3_4b-{prefill,decode}-int8.html`. Comparison
pages are **not committed** — `make plots` builds them, and the numbers quoted below come from that
run.

Three things differ from the per-chip page, and nothing else does (`../CORRECTIONS.md` D29).

**The time axis is shared and absolute.** A bar three times as long took three times as long. On
Gemma-3-4B prefill at S=512, int8: A100 9.1 ms against Metis 119 ms, and the subtitle states the
13.04x rather than leaving it to be measured off the ticks. The per-chip normalised view is kept,
because it is the better view of one machine's internal balance — at 13x, the faster chip's whole
run is 7% of a shared axis.

**Rows are banded by chip, not aligned across chips.** They cannot be aligned: A100 declares 3
memory levels and 2 compute units, Metis 4 and 2, `chip_a` 2 and 1, and there is no honest
correspondence between `cuda_core` and `dpu`. Each band opens with a header row naming the machine
and carrying its peak, DRAM bandwidth, on-chip capacity, its row counts, its total span with its
verdict, and its achieved throughput — `622 TOP/s achieved · 100% of peak` against `186 TOP/s
achieved · 89% of peak` on an 8192³ INT8 matmul, the same 3.34x the latency ratio is, inverted
(D38). Inside a band everything the single-chip page does still holds.

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

The page keeps wheel-zoom, drag-pan, double-click reset and per-bar hover across both bands, renders
**the arithmetic section once** — the operator list is a property of the workload, and the workload
is the same on both machines — but the **deployment listing once per chip**, because how the work
reaches the silicon is exactly what a comparison is for (D32).

## Adding a chip

Both scripts take `--chip` and default to a pair that makes the contrast visible — a datacentre GPU
and an edge NPU. The schedule is a property of the machine, so a new chip means a new figure rather
than a new line on an existing one; `--compare` is the exception, and it bands rather than merges.
