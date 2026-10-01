/* Interactive notebook for one encoder layer: the operations in order on a tiny example, then the
   same operations priced at real model sizes on real chips. */
(function () {
  'use strict';
  const E = window.ENC;
  // mode: 'layer' walks the tiny layer with real numbers; 'cost' prices it at any size.
  const DEF = { mode: 'layer', S: 4, d: 8, h: 2, ffn: 16, act: 'relu', norm: 'rmsnorm', vocab: 16, seed: 0, chip: 'a100', batch: 1, ffnx: 4, speed: 1 };
  const S = Object.assign({}, DEF);
  const $ = (id) => document.getElementById(id);
  const el = (t, c, h) => { const e = document.createElement(t); if (c) e.className = c; if (h != null) e.innerHTML = h; return e; };
  const BERT = { mode: 'cost', S: 512, d: 768, h: 12, ffnx: 4, act: 'gelu', norm: 'layernorm', vocab: 30522 };

  const CH = [
    { id: 'layer', t: 'The layer, op by op', set: {}, s: '<h2>One encoder layer, one operation at a time</h2><p>An encoder reads <b>all its tokens at once</b> — a sentence for BERT, image patches for ViT — and every layer turns T token vectors of width d into T new ones. Inside: a <b>norm</b>, three projections to <b>Q, K, V</b>, <b>attention</b>, an <b>output projection</b> and a <b>residual add</b>; then a norm, the <b>FFN</b> (widen to ffn, an activation, narrow back) and another residual add.</p><p class="sizes"><b>The sizes.</b> <b>T</b> is the number of tokens (512 for BERT, 197 patches for ViT-B/16). <b>d</b> is the model width (768 for BERT-base, 4096 for an 8B model), split into <b>h heads</b> of d/h. <b>ffn</b> is the FFN\'s inner width, usually 4·d. Here everything is tiny so every number fits on screen.</p><p class="try">Try: Step through, then <b>Next block</b> to jump a half-layer. The counters end at 5280 operations and 664 parameters — the same as bwz\'s encoder layer at these sizes.</p>' },
    { id: 'attn', t: 'The attention half', set: { S: 4, d: 8, h: 2 }, s: '<h2>The attention half: tokens talk to each other</h2><p>Q, K and V are three ordinary matmuls, <b>[T, d] × [d, d]</b> each — the same weights for every token. Attention is the only place tokens mix: every token\'s output is a weighted average of every token\'s V. The output projection mixes the heads back, and the <b>residual add</b> keeps the original x: the layer only adds a correction.</p><p class="try">Try: Heads 1, 2, 4 — the projections do not change at all; only how attention cuts the columns (the block lines).</p>' },
    { id: 'ffn', t: 'The FFN half', set: { act: 'gelu' }, s: '<h2>The FFN half: each token on its own</h2><p>No mixing between tokens here: every row goes through the same small network — widen d → ffn, apply an <b>activation</b> element by element, narrow ffn → d. Two matmuls, <b>2·T·d·ffn</b> operations each, and usually the biggest share of the layer.</p><p class="try">Try: Activation relu and gelu — gelu costs 8 operations per element against relu\'s 1, but next to the matmuls it barely shows.</p>' },
    { id: 'gated', t: 'Gated FFN', set: { act: 'swiglu', ffn: 8 }, s: '<h2>Gated FFN: SwiGLU, GeGLU</h2><p>Modern models use a <b>third</b> matrix: a gate, widened like the up projection, then <code>silu(gate) · up</code> element by element. One more [d, ffn] matmul and its parameters; models shrink ffn to about 8/3·d to keep the size the same.</p><p class="try">Try: swiglu vs relu at the same ffn — parameters and operations of the FFN go up by half.</p>' },
    { id: 'params', t: 'Where the parameters are', set: Object.assign({}, BERT, { S: 128 }), s: '<h2>Where the parameters are</h2><p>Now the real sizes: a BERT-base layer, d = 768, 12 heads, ffn = 3072. Attention\'s four projections hold <b>4·d²</b> weights, the FFN <b>2·d·ffn = 8·d²</b>: <b>two thirds</b> of the layer is the FFN. Norms are a rounding error. (The embedding table is the stack\'s, not the layer\'s.)</p><p class="try">Try: ffn = 8/3·d with swiglu — the gated FFN with the same parameter count.</p>' },
    { id: 'tokens', t: 'More tokens', set: Object.assign({}, BERT, { S: 4096 }), s: '<h2>More tokens: attention grows as T²</h2><p>Every matmul grows linearly with T — one more row per token. Attention grows with <b>T²</b>: every token against every token. At 512 tokens attention is a small share of BERT\'s layer; at 4096 it dominates.</p><p class="try">Try: tokens 128, 512, 2048, 8192 and watch the attention share.</p>' },
    { id: 'width', t: 'Wider models', set: { mode: 'cost', S: 512, d: 4096, h: 32, ffnx: 4, act: 'gelu', norm: 'rmsnorm', vocab: 32000 }, s: '<h2>Wider models: the matmuls take over</h2><p>Projections and FFN grow with <b>d²</b>, attention only with d. At d = 4096 and 512 tokens nearly all of the time is matmul.</p><p class="try">Try: width 768 → 4096 → 8192 at the same tokens.</p>' },
    { id: 'chips', t: 'Same layer, three chips', set: Object.assign({}, BERT, { S: 512, chip: 'a100' }), s: '<h2>The same layer on three chips</h2><p>An operation is compute-bound when its <b>OP per byte</b> is above the chip\'s ridge point (peak ÷ bandwidth). A BERT projection at 512 tokens does about 220 operations per byte: above A100\'s ridge of 153, so compute-bound; below H100\'s 295, so memory-bound there; far below the edge NPU\'s 6168, where the whole layer waits on memory.</p><p class="try">Try: chip A100, H100, edge NPU — identical arithmetic, the bound flips. Then tokens 128 or 2048.</p>' },
    { id: 'elementwise', t: 'Norms and residuals', set: Object.assign({}, BERT, { S: 512, chip: 'h100' }), s: '<h2>The cheap operations are memory-bound everywhere</h2><p>A norm, an activation or a residual add does a handful of operations per number it reads and writes — about 1 to 4 OP per byte, against ridge points in the hundreds. On every chip they are <b>memory-bound</b>. That is why real kernels <b>fuse</b> them into the matmul next to them: the data is already on chip.</p><p class="try">Try: look at their OP/byte column, then at their share of the time.</p>' },
    { id: 'batch', t: 'Batch: more rows', set: Object.assign({}, BERT, { S: 128, chip: 'h100', batch: 1 }), s: '<h2>Batch: more rows, the same weights</h2><p>A batch of B sentences is just <b>B·T rows</b> for every projection and FFN — the same weights, read once, used B times more: their OP/byte rises with B. Attention is different: each sentence attends only to itself, so a batch adds independent attentions and no reuse.</p><p class="try">Try: batch 1, 4, 16 on H100 at 128 tokens and watch the projections turn compute-bound.</p>' },
    { id: 'play', t: 'Playground: tiny', set: { S: 8, d: 8, h: 4, ffn: 16, act: 'geglu', norm: 'layernorm', seed: 4 }, s: '<h2>Playground: the tiny layer</h2><p>Every knob is live; every run ends with the walk checked against the same layer computed token by token.</p>' },
    { id: 'playcost', t: 'Playground: real sizes', set: { mode: 'cost', S: 2048, d: 4096, h: 32, ffnx: 8 / 3, act: 'swiglu', norm: 'rmsnorm', vocab: 32000, chip: 'h100', batch: 1 }, s: '<h2>Playground: real sizes</h2><p>Defaults are an 8B-class layer read bidirectionally. Every run ends with operations and parameters checked against the closed form.</p>' }
  ];

  let view = null;
  const divisors = (n) => Array.from({ length: n }, (_, i) => i + 1).filter((v) => n % v === 0);
  function fix() {
    if (S.mode === 'layer') {
      S.S = Math.max(1, Math.min(8, S.S)); if (![2, 4, 8].includes(S.d)) S.d = 8;
      if (S.d % S.h) S.h = divisors(S.d).filter((v) => v <= S.h).pop();
      if (![4, 8, 16].includes(S.ffn)) S.ffn = 16;
      S.vocab = 16;
    } else {
      if (S.d % S.h) S.h = divisors(S.d).filter((v) => v <= S.h).pop();
      S.ffn = Math.round(S.d * S.ffnx);
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
  const ACTS = ['relu', 'gelu', 'swiglu', 'geglu'], NORMS = ['rmsnorm', 'layernorm', 'none'];
  function knobs() {
    const k = $('knobs'); k.innerHTML = '';
    const sec = (t) => k.appendChild(el('div', 'ksec', t));
    sec('View');
    k.appendChild(seg('Mode', 'mode', ['layer', 'cost'], (v) => (v === 'layer' ? 'Tiny layer, real numbers' : 'Real sizes, on a chip')));
    sec('Shape');
    if (S.mode === 'layer') {
      k.appendChild(slider('Tokens T', 'S', 1, 8, 1));
      k.appendChild(seg('Width d', 'd', [2, 4, 8]));
      k.appendChild(seg('Heads h', 'h', divisors(S.d)));
      k.appendChild(seg('FFN width ffn', 'ffn', [4, 8, 16]));
    } else {
      k.appendChild(seg('Tokens T', 'S', [1, 16, 128, 512, 2048, 4096, 8192]));
      k.appendChild(seg('Width d', 'd', [256, 768, 1024, 2048, 4096, 8192]));
      k.appendChild(seg('Heads h', 'h', [1, 4, 8, 12, 16, 32, 64].filter((v) => S.d % v === 0)));
      k.appendChild(seg('FFN width', 'ffnx', [2, 8 / 3, 4], (v) => (v === 4 ? '4·d' : v === 2 ? '2·d' : '8/3·d')));
      k.appendChild(seg('Batch', 'batch', [1, 4, 16, 64]));
    }
    sec('Architecture');
    k.appendChild(seg('Activation', 'act', ACTS));
    k.appendChild(seg('Norm', 'norm', NORMS));
    if (S.mode === 'layer') {
      const nums = el('div', 'kg'); nums.appendChild(el('div', 'kl', 'Numbers'));
      const nb = el('div', 'seg');
      const r = el('button', 'btn', 'New random'); r.type = 'button'; r.addEventListener('click', () => { S.seed = (S.seed % 97) + 1; rebuild(); }); nb.appendChild(r);
      nums.appendChild(nb); k.appendChild(nums);
    } else {
      sec('Chip');
      k.appendChild(seg('Chip', 'chip', ['a100', 'h100', 'chip_a'], (v) => ({ a100: 'A100', h100: 'H100', chip_a: 'edge NPU' }[v])));
    }
    sec('Playback');
    k.appendChild(slider('Speed', 'speed', 0.25, 4, 0.25, (v) => v + '×'));
  }
  function rule() {
    const R = $('rule'), cf = E.closedForm(S), gated = S.act === 'swiglu' || S.act === 'geglu';
    const M = (S.mode === 'cost' ? S.batch : 1) * S.S;
    R.innerHTML = '<div class="rl"><span class="rk">rows</span>' + (M !== S.S ? S.batch + ' × ' : '') + S.S + ' token' + (S.S > 1 ? 's' : '') + '<span class="ar">·</span><span class="rk">projections</span>4 × [' + M + ', ' + S.d + '] × [' + S.d + ', ' + S.d + ']<span class="ar">·</span><span class="rk">FFN</span>' + (gated ? 3 : 2) + ' matmuls through ffn = ' + S.ffn + '<span class="ar">·</span><span class="rk">attention</span>' + S.h + ' head' + (S.h > 1 ? 's' : '') + ' × ' + S.S + '×' + S.S + ' scores</div>' +
      '<div class="rl small"><span class="rk">layer</span>' + cf.params.toLocaleString('en') + ' parameters (embedding table included) · ' + cf.ops.toLocaleString('en') + ' operations = 8·M·d² + 4·T²·d + 5·h·T² + ' + (gated ? 6 : 4) + '·M·d·ffn + the norms, activation and residuals</div>';
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
    if (S.mode === 'layer') cfg.P = E.problem(S);
    view = new (S.mode === 'layer' ? E.LayerView : E.CostView)(d, cfg);
    knobs(); rule(); trace();
  }
  function chapter(id) {
    const c = CH.find((x) => x.id === id);
    Object.keys(DEF).forEach((k) => { S[k] = DEF[k]; });
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
  window.LAB = { S, rebuild, chapter, CH, get view() { return view; } };
  chapter('layer');
})();
