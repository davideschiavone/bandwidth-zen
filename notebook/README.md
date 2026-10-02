# Notebooks

**Website: [https://davideschiavone.github.io/bandwidth-zen/](https://davideschiavone.github.io/bandwidth-zen/)** — click to open; nothing to install.

| Topic | Deck | Notebook | Sources |
|---|---|---|---|
| **Matmul, rearranged** — loop orders, os/ws/is dataflows, tiling, cores and waves, split-K, weight sets | [deck](https://davideschiavone.github.io/bandwidth-zen/matmul/deck.html) | [notebook](https://davideschiavone.github.io/bandwidth-zen/matmul/notebook.html) | [`matmul/`](matmul/README.md) |
| **Attention, blocked** — plain attention, online softmax, Br/Bc blocks, programs on cores, inner dataflow, heads and batch, K/V re-reads | [deck](https://davideschiavone.github.io/bandwidth-zen/attention/deck.html) | [notebook](https://davideschiavone.github.io/bandwidth-zen/attention/notebook.html) | [`attention/`](attention/README.md) |
| **One encoder layer** — the layer op by op, then at real sizes on A100 / H100 / an edge NPU: parameters, scaling with tokens and width, bound per op, batch | [deck](https://davideschiavone.github.io/bandwidth-zen/encoder/deck.html) | [notebook](https://davideschiavone.github.io/bandwidth-zen/encoder/notebook.html) | [`encoder/`](encoder/README.md) |
| **One decoder layer, generating** — the causal mask, prefill then one token per step, the KV cache on and off, GQA; one decode step of a real model on a chip | [deck](https://davideschiavone.github.io/bandwidth-zen/decoder/deck.html) | [notebook](https://davideschiavone.github.io/bandwidth-zen/decoder/notebook.html) | [`decoder/`](decoder/README.md) |

The old addresses `…/deck.html` and `…/notebook.html` redirect to the matmul pages, so existing links
keep working.

## Opening them locally

Every `.html` file is self-contained: clone the repository and double-click
`notebook/index.html`, or any deck or notebook page. (On github.com an `.html` file shows its source;
use the website link above, or download it.) If your browser restricts local files:

```bash
cd notebook
python3 -m http.server 8000     # then open http://localhost:8000
```

## Layout

```
notebook/
  index.html          landing page for both topics
  deck.html, notebook.html   redirects to matmul/ (old links)
  package.json        npm run build / npm test — both topics
  common/
    engine.css        shared colour tokens and styles (A/Q blue, B/K/V orange, C/O green)
    shell.py          shared page shell: reset, fonts, deck and notebook chrome, page builders
    widget.js         shared deck widgets: a view on a slide, with in-slide knobs
  matmul/             src/ (engines, notebook, slides, build.py), tests/, deck.html, notebook.html
  attention/          src/ (engine, notebook, slides, attention.css, build.py), tests/, deck.html, notebook.html
  encoder/            src/ (engine, notebook, slides, encoder.css, build.py), tests/, deck.html, notebook.html
  decoder/            src/ (engine, notebook, slides, decoder.css, build.py), tests/, deck.html, notebook.html
```

The attention pages reuse the matmul simulator (`matmul/src/engine2.js`) for the matrix grid, the
stepping controls and the counters, and its single-core view to show one block of `S = Q·Kᵀ` as the
matmul it is.

## Rebuilding and testing

```bash
cd notebook
npm run build     # Python 3.8+, standard library only: rewrites every topic's deck.html and notebook.html
npm install       # once: installs jsdom for the page checks
npm test          # every topic: simulator configurations, then every slide and chapter driven to its end
```

GitHub Pages redeploys the whole folder whenever it changes on `main`
(`.github/workflows/pages.yml`).
