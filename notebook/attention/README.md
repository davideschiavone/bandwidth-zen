# Attention, blocked

**Website: [https://davideschiavone.github.io/bandwidth-zen/attention/notebook.html](https://davideschiavone.github.io/bandwidth-zen/attention/notebook.html)** · [deck](https://davideschiavone.github.io/bandwidth-zen/attention/deck.html) · [all notebooks](https://davideschiavone.github.io/bandwidth-zen/)

One attention call, `O = softmax(Q Kᵀ / √d) V`, built up from the textbook formula to
**FlashAttention-2**: why plain attention stores the whole score matrix S, the online softmax that
removes the need to, blocks of keys (Bc) and of queries (Br), programs dealt to cores in waves, the
two inner matmuls and their os / ws / is dataflow, heads and batch, and K/V re-reads when a head
straddles two waves.

Everything runs on small shapes with small integers in Q, K and V, so every score can be checked by
hand, and every animation ends with a check that O equals `softmax(QKᵀ/√d)V` computed the plain way.

**The knobs are `bwz attention`'s own flags**, and the rule box above each animation prints the
command with the same values:

| Knob | Flag | What it is |
|---|---|---|
| Queries | `-S` | rows of Q |
| Keys | `--kv-len` | rows of K and V |
| Head width d | `--head-dim` | one head's width — the contraction of Q·Kᵀ |
| Heads, Batch | `--heads`, `--batch` | independent programs, nothing shared between them |
| Br | `--br` | query rows one program owns |
| Bc | `--bc` | keys per block; one rescale of O per block after the first |
| S = QKᵀ flow, O += PV flow | `--stationarity` | the inner matmuls' dataflow (`bwz` pins both at once) |
| Array tile | — (the chip's `systolic_dims`) | how many slices each inner contraction is cut into under ws / is |
| Cores | — (the chip's unit count) | how many programs run at once |

| File | What it is |
|---|---|
| [`deck.html`](deck.html) | 15-slide presentation with step-by-step animations and in-slide knobs |
| [`notebook.html`](notebook.html) | Interactive notebook: 12 chapters, every knob above, plus a printed trace of every step |

## How to open

Download or clone the repository and double-click `deck.html` or `notebook.html`, or use the website
link at the top. Controls, browser requirements and the local-server option are the same as for the
matmul pages: see [`../matmul/README.md`](../matmul/README.md). In the notebook, **Next block** (↓)
jumps to the end of the current key block; in the plain-attention view it jumps a whole query row.

## Limits and conventions

- **Sizes:** queries and keys 1 to 8, head width 1 to 4, up to 4 heads and a batch of 2. Br and Bc
  need not divide the sizes: the last block is ragged, as in `bwz`.
- **One step** is one phase for every busy core at once: `S = Q_i K_jᵀ` on the array, the online
  softmax on the vector unit, `O += P V_j` on the array — and, after the last key block, `O = O / l`.
  Waves are lockstep. Timing is a teaching model, not cycle-accurate; `bwz attention` gives the times.
- **Numbers:** the running example is fixed; "New random" draws integers from −1 to 2. Scores and
  probabilities are shown rounded; the check at the end uses the exact values (tolerance 1e-9).
- **Counts are `bwz`'s:** programs = heads × ceil(S / Br), waves = ceil(programs / cores), K and V
  read once per wave a head's programs span, O rescaled `heads · S · d · (key blocks − 1)` times.
  The inner partial adds are charged as on a unit with no accumulator to sum K in place (a GPU tensor
  core): `(ceil(K / tile) − 1) · M · N` per inner matmul under ws or is, none under os.
- **Inside one block** draws `S = Q_i K_jᵀ` (before the 1/√d scale) with the matmul notebook's own
  view, so its loop orders and counters are the matmul notebook's.

## Rebuilding and testing

```bash
python3 src/build.py      # Python 3.8+, standard library only; rewrites deck.html and notebook.html
cd .. && npm install      # once, in notebook/: installs jsdom for the page checks
npm test                  # in notebook/: both topics
```

`npm test` runs the attention simulator over ~100k configurations of shape, heads, batch, Br, Bc,
cores and inner dataflow, and checks that O equals the textbook softmax and that every counter
equals its closed form. It then loads both pages, runs every animation and every in-slide knob to its
end, and confirms that no matrix cell overlaps another.

```
src/
  engine.js       plain and FlashAttention simulators, their views, the inner-block view, deck widgets
  attention.css   S, Q, K, V, O cell states on the shared colour tokens
  lab.js          notebook chapters and knobs
  deck_slides.html   slide content
  build.py        assembles deck.html and notebook.html (chrome: ../common/, matmul simulator: ../matmul/src/engine2.js)
tests/check.js
```
