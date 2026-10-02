/* Interactive notebook for one decoder layer, generating: the tiny layer token by token with real
   numbers, then one decode step (or a prefill) at a real model's size on a real chip. */
(function () {
  'use strict';
  const D = window.DEC;
  const MODELS = {
    gpt2: { d: 768, h: 12, kvh: 12, ffn: 3072, act: 'gelu', norm: 'layernorm', vocab: 50257, layers: 12, label: 'GPT-2 small' },
    llama8b: { d: 4096, h: 32, kvh: 8, ffn: 14336, act: 'swiglu', norm: 'rmsnorm', vocab: 128256, layers: 32, label: 'Llama-3-8B' },
    llama70b: { d: 8192, h: 64, kvh: 8, ffn: 28672, act: 'swiglu', norm: 'rmsnorm', vocab: 128256, layers: 80, label: 'Llama-3-70B' }
  };
  // mode 'gen' walks the tiny layer; 'cost' prices one step of a real model's layer.
  const DEF = { mode: 'gen', Tp: 3, G: 3, d: 4, h: 2, kvh: 2, ffn: 8, act: 'swiglu', norm: 'rmsnorm', vocab: 16, seed: 0, cache: true,
    model: 'llama8b', phase: 'decode', L: 4096, S: 2048, batch: 1, chip: 'a100', speed: 1 };
  const S = Object.assign({}, DEF);
  const $ = (id) => document.getElementById(id);
  const el = (t, c, h) => { const e = document.createElement(t); if (c) e.className = c; if (h != null) e.innerHTML = h; return e; };
  const cost = (o) => Object.assign({ mode: 'cost' }, o);

  const CH = [
    { id: 'gen', t: 'Generating, token by token', set: {}, s: '<h2>A decoder generates one token at a time</h2><p>GPT and Llama are stacks of <b>decoder</b> layers. They read the prompt, then produce text one token after another, each new token fed back in. The first step, the <b>prefill</b>, pushes all prompt tokens through at once; every later step, a <b>decode</b> step, pushes <b>one</b>.</p><p class="sizes"><b>The sizes.</b> <b>T</b> is the number of tokens so far — the prompt plus everything generated — and grows by one per step. <b>d</b> is the model width (4096 for Llama-3-8B), split into <b>h</b> query heads; <b>kvh</b> is the number of key/value heads (8 for Llama-3-8B: grouped-query attention). Here everything is tiny so every number fits on screen.</p><p class="try">Try: Step through — the token strip, S, the K/V cache and the output all grow by one row per step.</p>' },
    { id: 'mask', t: 'The causal mask', set: { Tp: 5, G: 1 }, s: '<h2>A decoder cannot look at the future</h2><p>Unlike the encoder, a token may attend only to <b>itself and earlier tokens</b>: when the model is trained to predict token t+1, seeing it would be cheating. So S is a <b>lower triangle</b> — the × cells above the diagonal are masked to −∞ before the softmax, and get probability 0. A causal prefill therefore does about <b>half</b> the attention work of an encoder on the same tokens.</p><p class="try">Try: look at the prefill step — every row stops at the diagonal.</p>' },
    { id: 'cache', t: 'The KV cache', set: { Tp: 3, G: 4 }, s: '<h2>Why past K and V can be kept</h2><p>Because of the mask, a token\'s K, V and output depend only on itself and earlier tokens. A new token arriving later changes <b>none</b> of them. So they are computed once and kept: the <b>KV cache</b>. A decode step computes K and V for the new token only, appends them, and reads the rest back.</p><p class="try">Try: watch the K and V caches gain one row per step, and the magenta "read back" counter grow with the context.</p>' },
    { id: 'nocache', t: 'Without a cache', set: { Tp: 3, G: 4, cache: false }, s: '<h2>Without a cache: recompute everything</h2><p>The same generation, recomputing every row at every step. The outputs come out <b>identical</b> — that is what makes the cache legal — but each step costs a whole pass over the sequence, so the total grows with T² instead of T. In an <b>encoder</b> there is no such choice: without the mask, a new token changes every earlier output, so nothing could be kept.</p><p class="try">Try: KV cache on and off, and compare <b>operations so far</b> at the end.</p>' },
    { id: 'gqa', t: 'Fewer K/V heads (GQA)', set: { d: 8, h: 4, kvh: 1, Tp: 3, G: 3 }, s: '<h2>Grouped-query attention: a smaller cache</h2><p>The cache stores K and V for every token, layer and KV head, so models share each K/V head between several query heads: <b>kvh &lt; h</b>. Llama-3-8B has 32 query heads and 8 KV heads — a cache 4× smaller, with the same attention arithmetic.</p><p class="try">Try: KV heads 1, 2, 4 — the cache columns shrink; the output check still passes.</p>' },
    { id: 'step', t: 'One token\'s cost', set: cost({ phase: 'decode', L: 4096 }), s: '<h2>What one decode step costs</h2><p>A real layer — Llama-3-8B, one new token, context 4096 — priced on A100. Every projection is now a <b>matrix-vector</b> product, [1, d] × [d, d]: each weight is read from memory to be used <b>once</b>. About 2 operations per 2 bytes, against a ridge point of 153: <b>everything is memory-bound</b>, and the time is the time to stream the weights.</p><p class="try">Try: compare with phase <b>prefill</b>: the same weights, used by thousands of tokens.</p>' },
    { id: 'context', t: 'The context grows', set: cost({ phase: 'decode', L: 32768 }), s: '<h2>Longer contexts: the cache takes over</h2><p>Weights cost the same at every step; the cache does not — every new token reads <b>all</b> earlier K and V. At short context, attention is a sliver; at 32k tokens it rivals the weights, and the cache itself runs to gigabytes.</p><p class="try">Try: context 1k, 4k, 32k, 128k and watch the attention row and the KV cache size.</p>' },
    { id: 'batch', t: 'Batch helps weights, not the cache', set: cost({ phase: 'decode', L: 4096, batch: 32 }), s: '<h2>Batching decode</h2><p>Serving many users at once makes every projection [B, d] × [d, d]: the weights are read once and used B times, so their OP per byte rises with B. But each sequence has <b>its own cache</b> — batching multiplies the cache reads instead of sharing them.</p><p class="try">Try: batch 1, 8, 32, 128: tokens/s climbs while the projections approach compute-bound; the attention row stays memory-bound.</p>' },
    { id: 'phases', t: 'Prefill vs decode', set: cost({ phase: 'prefill', S: 2048 }), s: '<h2>Two different machines</h2><p>Prefill is the encoder-like pass over the prompt (causal): thousands of rows per weight, <b>compute-bound</b>, and it sets the time to the first token. Decode is one row per weight, <b>memory-bound</b>, and it sets the time per token after that.</p><p class="try">Try: phase prefill and decode on the same model and chip, and compare the bound column.</p>' },
    { id: 'chips', t: 'Three chips', set: cost({ phase: 'decode', L: 4096, chip: 'h100' }), s: '<h2>The same token on three chips</h2><p>Decode is memory-bound everywhere, so its speed follows <b>memory bandwidth</b>, not compute: H100 beats A100 by its bandwidth ratio (1.6×), not its compute ratio (3.2×), and an edge NPU with 34 GB/s is far behind.</p><p class="try">Try: A100, H100, edge NPU — then a smaller model.</p>' },
    { id: 'play', t: 'Playground: tiny', set: { Tp: 4, G: 6, d: 8, h: 4, kvh: 2, ffn: 16, act: 'geglu', norm: 'layernorm', seed: 5 }, s: '<h2>Playground: the tiny layer</h2><p>Every knob is live; every run ends with every token\'s output checked against a full causal recompute.</p>' },
    { id: 'playcost', t: 'Playground: real sizes', set: cost({ model: 'llama70b', phase: 'decode', L: 8192, batch: 8, chip: 'h100' }), s: '<h2>Playground: real sizes</h2><p>Any model, phase, context, batch and chip. Every run ends with operations and parameters checked against the closed form.</p>' }
  ];

  let view = null;
  const divisors = (n) => Array.from({ length: n }, (_, i) => i + 1).filter((v) => n % v === 0);
  function fix() {
    if (S.mode === 'gen') {
      S.Tp = Math.max(1, Math.min(6, S.Tp)); S.G = Math.max(1, Math.min(6, S.G));
      if (![2, 4, 8].includes(S.d)) S.d = 4;
      if (S.d % S.h) S.h = divisors(S.d).filter((v) => v <= S.h).pop();
      if (S.h % S.kvh) S.kvh = divisors(S.h).filter((v) => v <= S.kvh).pop();
      S.vocab = 16;
    } else {
      const m = MODELS[S.model];
      ['d', 'h', 'ffn', 'act', 'norm', 'vocab', 'layers'].forEach((k) => { S[k] = m[k]; });
      if (!(S.kvh > 0 && S.h % S.kvh === 0 && S.kvh <= S.h) || S._model !== S.model) S.kvh = m.kvh;
      S._model = S.model;
    }
  }
  function seg(label, key, opts, lab) {
    const g = el('div', 'kg'); g.appendChild(el('div', 'kl', label));
    const b = el('div', 'seg');
    opts.forEach((v) => {
      const x = el('button', 'btn', lab ? lab(v) : String(v)); x.type = 'button';
      x.setAttribute('aria-pressed', S[key] === v ? 'true' : 'false');
      x.addEventListener('click', () => { S[key] = v; rebuild(); });
      b.appendChild(x);
    });
    g.appendChild(b); return g;
  }
  function slider(label, key, min, max, step, fmt) {
    const g = el('div', 'kg'); const id = 'k-' + key;
    g.appendChild(el('label', 'kl', label + ' <b id="' + id + '-v">' + (fmt ? fmt(S[key]) : S[key]) + '</b>')).setAttribute('for', id);
    const r = el('input'); r.type = 'range'; r.min = min; r.max = max; r.step = step; r.value = S[key]; r.id = id;
    r.addEventListener('input', () => { S[key] = +r.value; $(id + '-v').textContent = fmt ? fmt(S[key]) : S[key]; if (key === 'speed' && view) { view.cfg.speed = S.speed; if (view.timer) { view.stop(); view.play(); } } });
    r.addEventListener('change', () => { if (key !== 'speed') rebuild(); });
    g.appendChild(r); return g;
  }
  const k1 = (v) => (v >= 1024 ? v / 1024 + 'k' : String(v));
  function knobs() {
    const k = $('knobs'); k.innerHTML = '';
    const sec = (t) => k.appendChild(el('div', 'ksec', t));
    sec('View');
    k.appendChild(seg('Mode', 'mode', ['gen', 'cost'], (v) => (v === 'gen' ? 'Tiny layer, real numbers' : 'Real model, on a chip')));
    if (S.mode === 'gen') {
      sec('Tokens');
      k.appendChild(slider('Prompt tokens', 'Tp', 1, 6, 1));
      k.appendChild(slider('Generated tokens', 'G', 1, 6, 1));
      k.appendChild(seg('KV cache', 'cache', [true, false], (v) => (v ? 'on' : 'off: recompute')));
      sec('Shape');
      k.appendChild(seg('Width d', 'd', [2, 4, 8]));
      k.appendChild(seg('Query heads h', 'h', divisors(S.d)));
      k.appendChild(seg('KV heads kvh', 'kvh', divisors(S.h)));
      k.appendChild(seg('FFN width', 'ffn', [4, 8, 16]));
      k.appendChild(seg('Activation', 'act', ['relu', 'gelu', 'swiglu', 'geglu']));
      k.appendChild(seg('Norm', 'norm', ['rmsnorm', 'layernorm', 'none']));
      const nb = el('div', 'kg'); nb.appendChild(el('div', 'kl', 'Numbers'));
      const r = el('button', 'btn', 'New random'); r.type = 'button'; r.addEventListener('click', () => { S.seed = (S.seed % 97) + 1; rebuild(); });
      const sg = el('div', 'seg'); sg.appendChild(r); nb.appendChild(sg); k.appendChild(nb);
    } else {
      sec('Model');
      k.appendChild(seg('Model', 'model', Object.keys(MODELS), (v) => MODELS[v].label));
      k.appendChild(seg('KV heads kvh', 'kvh', [1, 8, MODELS[S.model].h].filter((v, i, a) => a.indexOf(v) === i)));
      sec('Step');
      k.appendChild(seg('Phase', 'phase', ['prefill', 'decode']));
      if (S.phase === 'decode') k.appendChild(seg('Context (tokens so far)', 'L', [128, 1024, 4096, 32768, 131072], k1));
      else k.appendChild(seg('Prompt tokens', 'S', [128, 512, 2048, 8192], k1));
      k.appendChild(seg('Batch', 'batch', [1, 8, 32, 128]));
      sec('Chip');
      k.appendChild(seg('Chip', 'chip', ['a100', 'h100', 'chip_a'], (v) => ({ a100: 'A100', h100: 'H100', chip_a: 'edge NPU' }[v])));
    }
    sec('Playback');
    k.appendChild(slider('Speed', 'speed', 0.25, 4, 0.25, (v) => v + '×'));
  }
  function rule() {
    const R = $('rule');
    if (S.mode === 'gen') {
      const P = D.problem(S), total = P.steps.reduce((s, x) => s + x.ops, 0), once = D.rowOps(P, 0, P.T);
      R.innerHTML = '<div class="rl"><span class="rk">steps</span>1 prefill of ' + S.Tp + ' token' + (S.Tp > 1 ? 's' : '') + ' + ' + S.G + ' decode step' + (S.G > 1 ? 's' : '') + '<span class="ar">·</span><span class="rk">cache</span>' + S.kvh * (S.d / S.h) + ' K + ' + S.kvh * (S.d / S.h) + ' V numbers per token<span class="ar">·</span><span class="rk">operations</span>' + total + (S.cache ? ' = one causal pass over all ' + P.T + ' tokens' : ' — a single causal pass would be ' + once) + '</div>';
      return;
    }
    const m = MODELS[S.model];
    R.innerHTML = '<div class="rl"><span class="rk">model</span>' + m.label + ': d = ' + m.d + ', ' + m.h + ' query heads, ' + S.kvh + ' KV heads, ffn = ' + m.ffn + ', ' + m.layers + ' layers<span class="ar">·</span><span class="rk">step</span>' + (S.phase === 'decode' ? (S.batch > 1 ? S.batch + ' sequences × ' : '') + 'one new token at context ' + S.L : 'prefill of ' + S.S + ' tokens' + (S.batch > 1 ? ' × ' + S.batch : '')) + '</div>' +
      '<div class="rl small">One layer is priced; the closing line multiplies by ' + m.layers + ' layers. The embedding row and final norm belong to the stack and are counted once per layer here — a rounding error.</div>';
  }
  function trace() {
    const d = $('trace'); if (!view || !$('tracebox').open) return;
    const lines = view.trace();
    d.textContent = lines.length ? lines.join('\n') : '(nothing yet — take a step)';
  }
  function rebuild() {
    fix();
    if (view) view.destroy();
    const host = $('view'); host.innerHTML = '';
    const d = el('div'); host.appendChild(d);
    const cfg = Object.assign({}, S, { onStep: trace });
    if (S.mode === 'gen') cfg.P = D.problem(S);
    view = new (S.mode === 'gen' ? D.GenerateView : D.DecodeCost)(d, cfg);
    knobs(); rule(); trace();
  }
  function chapter(id) {
    const c = CH.find((x) => x.id === id);
    Object.keys(DEF).forEach((k) => { S[k] = DEF[k]; }); delete S._model;
    Object.assign(S, c.set);
    $('story').innerHTML = c.s;
    document.querySelectorAll('#chapters button').forEach((b) => b.setAttribute('aria-pressed', b.dataset.id === id ? 'true' : 'false'));
    rebuild();
  }
  const bar = $('chapters');
  CH.forEach((c, n) => { const b = el('button', 'btn chap', '<span>' + (n + 1) + '</span>' + c.t); b.type = 'button'; b.dataset.id = c.id; b.addEventListener('click', () => chapter(c.id)); bar.appendChild(b); });
  $('tracebox').addEventListener('toggle', trace);
  document.addEventListener('keydown', (e) => {
    if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT')) return;
    if (view && view.keydown(e)) e.preventDefault();
  });
  window.LAB = { S, rebuild, chapter, CH, MODELS, get view() { return view; } };
  chapter('gen');
})();
