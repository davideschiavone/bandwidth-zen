# Matmul, rearranged

One matrix multiplication, `A[M,K] @ B[K,N] -> C[M,N]`, shown as **rearrangements of the textbook
three-loop code**: loop orders, inner / outer / row-wise / column-wise products, output-, weight- and
input-stationary dataflows, tiling, hardware step sizes, multi-core grid and sweep, waves,
accumulation over time vs over space, K-groups, split-K through memory, and weight sets.

Everything runs on one 4×4 example with small integers, so every step can be checked by hand, and
every animation ends with a check that `C == A @ B`.

| File | What it is |
|---|---|
| [`deck.html`](deck.html) | 39-slide presentation with step-by-step animations |
| [`notebook.html`](notebook.html) | Interactive notebook: 19 chapters, knobs for sizes, tiles, loop orders, cores, flows, split-K and weight sets, plus a printed trace of every step |
| [`index.html`](index.html) | Landing page linking both |
| `extras/matmul_rearranged.ipynb` | Earlier static Jupyter draft (not interactive, kept for reference) |

## How to open

**Option 1: in the browser, nothing to install.**
Each `.html` file is self-contained. Download or clone the repository, then double-click
`deck.html` or `notebook.html`. It opens in your default browser.

> Clicking an `.html` file on github.com shows its **source code**, not the page. To view it,
> download it (the "Download raw file" button) and open it locally, or use the GitHub Pages
> link below.

**Option 2: GitHub Pages link.** If Pages is enabled for this repository (Settings → Pages →
"Deploy from a branch", branch `main`, folder `/ (root)`), the pages are served at
`https://<user>.github.io/<repo>/matmul-rearranged/`. If the repository serves Pages from `/docs`
instead, this folder has to live under `docs/` to be published.

**Option 3: a local web server** (optional; only needed if your browser restricts local files):

```bash
cd matmul-rearranged
python3 -m http.server 8000     # then open http://localhost:8000
```

### Requirements

- A current browser: Chrome/Edge 111+, Firefox 113+, Safari 16.2+ (the pages use CSS `color-mix`).
  Older browsers show the animations with wrong or missing colours.
- JavaScript enabled.
- Internet is optional. It only loads the fonts from Google Fonts; offline, system fonts are used
  and everything else works.

## Controls

**Deck**

| Key | Action |
|---|---|
| `→` / Space | next line of the slide, or next animation step |
| `←` | step back |
| `↓` | run a whole inner loop (or the next bigger jump) |
| `PgDn` / `N`, `PgUp` / `B` | next / previous slide |
| `P` | play / pause the animation |
| `E` / `R` | finish / reset the animation |
| `Home` / `End` | first / last slide |

`deck.html#s24` opens slide 24 directly. Slides with knobs (flow, cores, P, slots, load time)
restart their animation when a knob changes.

**Notebook**

Pick a chapter at the top; it sets the knobs and explains what to watch. Then change any knob on
the left. The animation has Step / Play / Finish buttons and a time slider, and the arrow keys work
whenever the cursor is not in a slider. "Printed steps" under the animation shows the same run as
text.

## Limits and conventions

- **Sizes:** M, N, K from 1 to 8. Tile sizes must divide the matrix sizes, and P must divide K.
  The notebook only offers valid values. Above about 6, the matrices get small on a phone.
- **Nothing is saved:** reloading resets the knobs and the animation.
- **Phones:** everything works, but the multi-core timelines scroll sideways.
- **Counting additions:** the first term of every C value is stored, not added, so a full product
  costs M·N·K multiplies and M·N·(K−1) additions (64 and 48 on the example) in every form.
  Forms differ only in order and in where partial sums live. For split-K, "additions in pass 2"
  are not extra: pass 1 does exactly that many fewer.
- **Timing is a teaching model, not cycle-accurate.** One step = one block product per busy core
  (multi-core views), one row of A through one slot (weight sets), and one wave or one pass-2 piece
  (split-K). The combine across cores is drawn as a binary tree of levels. Writing a weight-set
  slot takes a fixed number of steps through a single write port.
- **Hardware figures** appear only in captions. The Axelera Metis figures (4 cores, 4 weight sets
  of 512×512 bytes each, one active at a time) were provided by the author; 4 × 512 × 512 B = 1 MiB
  per core matches Axelera's published 1 MiB of compute memory per core. The tensor-core size is
  the CUDA WMMA m16n16k16 warp-level operation.

## Rebuilding and testing

The two pages are generated from `src/` (the engines, the notebook logic and the slide content):

```bash
python3 src/build.py      # Python 3.8+, standard library only; rewrites deck.html and notebook.html
npm install               # once: installs jsdom for the page checks
npm test                  # ~130k simulator configurations + drives every slide and chapter
```

`npm test` checks, for every configuration of size, tiles, loop order, cores, flow, K-groups,
split-K pieces and weight-set schedule, that the result equals `A @ B` and the multiply and
addition counts are the honest totals. It then loads both pages, runs every animation (and every
in-slide knob) to its end, and confirms that no matrix cell overlaps another.

```
src/
  engine.js     running example, loop orders, early deck widgets
  engine2.js    general simulator: any size, tiling, loop order, cores, flows, waves, K-groups
  engine3.js    split-K through memory, weight sets
  engine.css    shared styles and colour tokens (A blue, B orange, C green, partial pale green)
  lab.js        notebook chapters and knobs
  deck_slides.html   slide content
  build.py      assembles deck.html and notebook.html
tests/check.js
```
