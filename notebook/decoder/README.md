# One decoder layer, generating

**Website: [https://davideschiavone.github.io/bandwidth-zen/decoder/notebook.html](https://davideschiavone.github.io/bandwidth-zen/decoder/notebook.html)** · [deck](https://davideschiavone.github.io/bandwidth-zen/decoder/deck.html) · [all notebooks](https://davideschiavone.github.io/bandwidth-zen/)

The layer GPT and Llama stack: the encoder's norm, attention and FFN, plus the **causal mask** — a token
may attend only to itself and earlier tokens. That mask is what makes the **KV cache** legal: a past
token's K, V and output never change, so they are computed once and read back instead of recomputed.

- **Tiny layer, real numbers** — a prompt of up to 6 tokens, then up to 6 generated ones. Step 1 is
  the prefill (the prompt's lower triangle of P = softmax(S)); every later step adds one row of P, one
  row to the K and V caches and one output row. The KV cache can be switched off: every step then
  recomputes the whole sequence, with identical outputs and far more work. Every token's output is
  checked against a full causal recompute written separately.
- **Real model, on a chip** — one decode step at a given context (or a prefill of a prompt) of GPT-2
  small, Llama-3-8B or Llama-3-70B, priced on A100, H100 or an edge NPU: operations, bytes, OP per
  byte, bound and time per operation, then × layers for time per token, tokens/s and the KV cache size.

| File | What it is |
|---|---|
| [`deck.html`](deck.html) | 12-slide presentation with step-by-step animations and in-slide knobs |
| [`notebook.html`](notebook.html) | Interactive notebook: 12 chapters; knobs for prompt and generated tokens, KV cache, KV heads, model, phase, context, batch and chip |

## The sizes

**T** is the number of tokens so far — the prompt plus everything generated — growing by one per step.
**d** is the model width, split into **h** query heads; **kvh** is the number of key/value heads
(grouped-query attention: kvh < h shrinks the cache, not the arithmetic).

## Conventions

- Counts follow the encoder notebook (and bwz), with causal attention charged on the lower triangle
  only: a row at position p attends to p + 1 keys. With the cache, the operations of all steps add up
  to exactly one causal pass over every token — the tests pin it.
- The generated token ids are given: choosing them needs the LM head on top of the whole stack.
- Times are the encoder notebook's teaching roofline (datasheet ceilings, one operation after
  another, nothing fused). One layer is priced and multiplied by the model's layer count; the
  embedding row and final norm are counted once per layer there, a rounding error.

## Rebuilding and testing

```bash
python3 src/build.py      # Python 3.8+, standard library only; rewrites deck.html and notebook.html
cd .. && npm install      # once, in notebook/: installs jsdom for the page checks
npm test                  # in notebook/: every topic
```

`src/engine.js` builds on the other notebooks: the matmul grid and controls, the attention softmax
reference and S/K/V/O styles, the encoder's arithmetic, chips and cost view, and the shared deck widgets.
