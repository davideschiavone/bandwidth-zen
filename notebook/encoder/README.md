# One encoder layer

**Website: [https://davideschiavone.github.io/bandwidth-zen/encoder/notebook.html](https://davideschiavone.github.io/bandwidth-zen/encoder/notebook.html)** · [deck](https://davideschiavone.github.io/bandwidth-zen/encoder/deck.html) · [all notebooks](https://davideschiavone.github.io/bandwidth-zen/)

The block BERT and ViT stack: `x → x + Attention(norm(x)) → x + FFN(norm(x))`. Two views:

- **Tiny layer, real numbers** — S ≤ 8 tokens of width d ≤ 8, one operation per step: embedding,
  norm, Q/K/V projections, attention, output projection, residual add, norm, FFN up (and gate),
  activation, FFN down, residual add, final norm. Every run ends with the walk checked against the
  same layer computed token by token, by separate code.
- **Real sizes, on a chip** — the same operations at up to 8192 tokens and width 8192, priced on
  A100 (fp16), H100 (fp16) or an edge NPU (chip_a, int8): operations, parameters, bytes moved,
  operations per byte, compute- or memory-bound, and time, with `t = max(operations / peak,
  bytes / bandwidth)` per operation (datasheet ceilings, one operation after another, nothing fused).

| File | What it is |
|---|---|
| [`deck.html`](deck.html) | 12-slide presentation with step-by-step animations and in-slide knobs |
| [`notebook.html`](notebook.html) | Interactive notebook: 12 chapters, live knobs for tokens, width, heads, FFN width, activation, norm, batch and chip |

## The sizes

**S** is the number of tokens the encoder reads at once (512 for BERT, 197 patches for ViT-B/16),
**d** the model width (768 for BERT-base, 4096 for an 8B model), split into **h** heads of d/h, and
**ffn** the FFN's inner width (usually 4·d, about 8/3·d for gated FFNs).

## Conventions

- **Counts are bwz's.** Projections 2·S·d², attention 4·S²·d + 5·h·S², norms 4 (RMSNorm) or 6
  (LayerNorm) operations per element, activations ReLU 1, GELU 8, SwiGLU 5, GeGLU 9 per element,
  residual adds 1. Parameters include the embedding table and the final norm, as bwz's encoder
  layer does: the defaults (4 tokens, d = 8, 2 heads, ffn = 16) are 664 parameters and 5280
  operations, and the tests pin four variants to bwz's own numbers.
- **Numbers:** embeddings −1 … 2, weights −1 … 1, norms without learned scale; GELU is the tanh form.
- **Times are a teaching roofline**, not bwz's full model: no calibration derates, no residency, no
  fusion, no launch overhead, and the elementwise operations are priced against the matrix peak
  (they are memory-bound regardless).

## Rebuilding and testing

```bash
python3 src/build.py      # Python 3.8+, standard library only; rewrites deck.html and notebook.html
cd .. && npm install      # once, in notebook/: installs jsdom for the page checks
npm test                  # in notebook/: every topic
```

```
src/
  engine.js     the layer walk, its token-by-token reference, the closed forms, the cost model, both views
  encoder.css   the layer strip, the cost table and the time bar
  lab.js        notebook chapters and knobs
  deck_slides.html   slide content
  build.py      assembles deck.html and notebook.html (chrome and widgets: ../common/; grid and controls:
                ../matmul/src/engine2.js; the softmax reference: ../attention/src/engine.js)
tests/check.js
```
