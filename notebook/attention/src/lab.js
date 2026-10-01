/* Interactive notebook for attention: knobs that are `bwz attention`'s own flags, and chapters
   that build FlashAttention up from the textbook formula. */
(function () {
  'use strict';
  const A = window.ATT;
  // Knob keys: S = queries Tq (-S), T = keys Tkv (--kv-len), d (--head-dim), heads (--heads), batch (--batch),
  // Br (--br), Bc (--bc), qk/pv (--stationarity, one per inner matmul here), cores, tile.
  // Chapters that teach the rescale pin a seed whose scores climb key after key (seed 16: −1, 0, 1, 2;
  // seed 28 over 8 keys), so the running max really moves; a flat row would hide the whole idea.
  const DEF = { mode: 'flash', S: 4, T: 4, d: 2, heads: 1, batch: 1, seed: 0, Br: 2, Bc: 2, cores: 1, tile: 1, qk: 'os', pv: 'os', hsel: -1, speed: 1 };
  const S = Object.assign({}, DEF);
  const $ = (id) => document.getElementById(id);
  const el = (t, c, h) => { const e = document.createElement(t); if (c) e.className = c; if (h != null) e.innerHTML = h; return e; };

  const CH = [
    { id: 'plain', t: 'Plain attention', set: { mode: 'naive' }, s: '<h2>The formula, done the obvious way</h2><p><code>O = softmax(Q Kᵀ / √d) V</code>. Every query row scores itself against every key (<b>S = QKᵀ/√d</b>), the scores of a row become probabilities (<b>P = softmax(S)</b>), and the output row is the probability-weighted mix of the value rows (<b>O = P V</b>).</p><p class="sizes"><b>The sizes.</b> <b>Tq</b> (queries) is how many tokens are being processed now: the whole prompt in prefill, often 2k–8k, but just 1 per step in decode. <b>Tkv</b> (keys) is how many tokens they can look at — the context so far, so it grows by one with every generated token. <b>d</b> is one head\'s width, usually 64–128; a model has many heads side by side (Llama-3-8B: 32 heads × d = 128 = its width 4096), and a batch repeats all of it per sequence. Here they are tiny so every number fits on screen.</p><p>Done this way, the whole score matrix S exists at once — Tq × Tkv numbers per head, 16.8 million at 4096 tokens — and goes to memory and back, and so does P.</p><p class="try">Try: press <b>Next row</b> to do one query row per click, and watch the violet memory counters.</p>' },
    { id: 'row', t: 'Softmax needs the row', set: { mode: 'naive', S: 2, T: 6, seed: 6 }, s: '<h2>Why S gets stored</h2><p>A softmax divides every exp(score) by the <b>sum</b> of the row, and for safety subtracts the row\'s <b>max</b> first. Both need <b>every</b> score of the row before the first probability can be written. That is the only reason plain attention keeps S: it cannot finish a row until the row is complete.</p><p class="try">Try: more keys (kv-len 8) — the stored row grows with them, and so does the traffic.</p>' },
    { id: 'online', t: 'Online softmax, one row', set: { mode: 'flash', S: 1, T: 4, Br: 1, Bc: 1, seed: 16 }, s: '<h2>Online softmax: one key at a time</h2><p>Keep, for each query row, a <b>running max m</b> and a <b>running sum l</b>. When a new score arrives: the new max is <code>m′ = max(m, s)</code>; everything accumulated so far was weighted with the old max, so multiply it by <code>e^(m − m′)</code>; then add the new term. At the end divide O by l. The result is <b>exactly</b> the softmax — no score is kept.</p><p class="try">Try: step and watch m and l in the table under the code. When m grows, the rescale line lights up.</p>' },
    { id: 'kblocks', t: 'Blocks of keys (Bc)', set: { mode: 'flash', S: 1, T: 8, Br: 1, Bc: 2, seed: 28 }, s: '<h2>Blocks of keys: Bc (<code>--bc</code>)</h2><p>Take the keys <b>Bc at a time</b>. Each block is one small matmul for the scores, one softmax update, one small matmul into O. The rescale of O happens once per block after the first, so bigger blocks mean <b>fewer rescales</b> — at the price of a bigger S block on chip.</p><p class="try">Try: Bc = 1, 2, 4, 8 and compare <b>O rescaled</b>. Bc = 8 is one block: no rescale at all, but all 8 scores on chip.</p>' },
    { id: 'qblocks', t: 'Blocks of queries (Br)', set: { mode: 'flash', S: 4, T: 4, Br: 2, Bc: 2, cores: 1 }, s: '<h2>Blocks of queries: Br (<code>--br</code>)</h2><p>Now several query rows at once: a <b>program</b> owns Br rows of Q and their O, and streams every key block past them. Each block of S is Br × Bc. On one core the programs run one after the other — each is a <b>wave</b> of one.</p><p class="try">Try: Br = 1, 2, 4. Fewer, bigger programs; the same multiplies and exps.</p>' },
    { id: 'cores', t: 'Programs on cores', set: { mode: 'flash', S: 4, T: 4, Br: 2, Bc: 2, cores: 2 }, s: '<h2>Programs run side by side</h2><p>Programs share nothing but K and V, so they run on different cores at the same time, dealt round-robin: core c takes program <code>wave · cores + c</code>. In each step every busy core does the same phase on its own rows.</p><p class="try">Try: Cores = 1, 2, 3, 4 and watch the timeline: waves, and idle cores when programs run out.</p>' },
    { id: 'inner', t: 'Inside one block', set: { mode: 'inner', S: 4, T: 4, d: 4, Br: 4, Bc: 4, tile: 2, qk: 'os' }, s: '<h2>Inside a block: S = Q<sub>i</sub>K<sub>j</sub>ᵀ is a matmul</h2><p>Each score block is an ordinary matmul with <b>M = Br, K = d, N = Bc</b> — drawn here by the matmul notebook\'s own view, block loops and all. Its <b>dataflow</b> (os / ws / is) is the same choice as in the matmul notebook: os keeps a block of S in the accumulator and sweeps d inside it; ws and is cut d over the grid, so the pieces must be added.</p><p class="try">Try: flow ws or is, and watch the additions move from "over time" to partial pieces. <code>bwz attention --stationarity</code> pins this.</p>' },
    { id: 'flows', t: 'Inner dataflow, costed', set: { mode: 'flash', S: 4, T: 4, d: 4, Br: 2, Bc: 2, cores: 2, tile: 2, qk: 'ws', pv: 'ws' }, s: '<h2>What the inner dataflow costs</h2><p>On a unit with no accumulator to sum K in place (a GPU tensor core), ws and is leave <b>(slices − 1) · M · N</b> partial sums for the vector unit per inner matmul: d/tile slices for S = QKᵀ, Bc/tile for O += PV. os leaves none. <code>bwz attention</code> weighs exactly this for every block size and keeps the cheapest.</p><p class="try">Try: switch both flows between os, ws and is and read <b>inner partial adds</b>. Array tile 1 makes every product its own slice.</p>' },
    { id: 'heads', t: 'Heads and batch', set: { mode: 'flash', S: 2, T: 4, d: 2, heads: 2, batch: 2, Br: 2, Bc: 2, cores: 4 }, s: '<h2>Heads and batch: more programs, nothing shared</h2><p>Every (batch, head) pair has its own Q, K and V, so heads and batch simply multiply the number of programs. Unlike a projection — where a batch reuses the same weights — nothing here is reused across them: K and V bytes grow exactly with batch × heads.</p><p class="try">Try: heads 1 → 4, batch 1 → 2, and watch programs, waves and K,V read. Use <b>Show head</b> to look at another head.</p>' },
    { id: 'waves', t: 'Waves re-read K and V', set: { mode: 'flash', S: 6, T: 4, d: 2, heads: 3, Br: 2, Bc: 2, cores: 4 }, s: '<h2>When a head straddles two waves</h2><p>K and V are fetched once per <b>wave</b> and shared by every program of that head in the wave. 3 heads × 3 programs = 9 programs on 4 cores: heads 1 and 2 are split across waves, so their K and V are fetched <b>twice</b>. The magenta counter shows the re-read against the once-only figure — the same effect <code>bwz attention</code> reports on a 4-core Metis.</p><p class="try">Try: Cores = 3 or 9 — every head fits one wave and K,V read drops to the once-only figure.</p>' },
    { id: 'never', t: 'S is never stored', set: { mode: 'flash', S: 4, T: 8, d: 2, Br: 2, Bc: 2, cores: 2 }, s: '<h2>The point: S never leaves the core</h2><p>Plain attention writes and reads back Tq × Tkv scores and as many probabilities. FlashAttention writes <b>only O</b>: blocks of S live on chip for one block and are discarded (dashed). It does <b>the same multiplies and exps</b> plus a few rescales.</p><div class="tw"><table class="cmp"><tr><th></th><th>plain</th><th>FlashAttention</th></tr><tr><td>multiplies</td><td>2·Tq·Tkv·d</td><td>2·Tq·Tkv·d</td></tr><tr><td>exp()</td><td>Tq·Tkv</td><td>Tq·Tkv</td></tr><tr><td>through memory besides Q, K, V, O</td><td class="mem">4·Tq·Tkv (S and P, out and back)</td><td>0</td></tr><tr><td>extra vector work</td><td>—</td><td>Tq·d per key block after the first</td></tr></table></div><p class="try">Try: switch <b>Mode</b> to Plain on the same shape and compare the counters.</p>' },
    { id: 'play', t: 'Playground', set: { mode: 'flash', S: 8, T: 8, d: 4, heads: 2, batch: 2, Br: 3, Bc: 3, cores: 5, tile: 2, seed: 4 }, s: '<h2>Playground</h2><p>Ragged blocks (3 does not divide 8), uneven programs and cores, several heads. Every knob is live and every run still ends with O checked against softmax(QKᵀ/√d)V. The rule box prints the <code>bwz attention</code> command with the same flags.</p>' }
  ];

  let view = null;
  function fix() {
    S.S = Math.max(1, Math.min(8, S.S)); S.T = Math.max(1, Math.min(8, S.T));
    S.Br = Math.max(1, Math.min(S.Br, S.S)); S.Bc = Math.max(1, Math.min(S.Bc, S.T));
    if (S.hsel >= S.heads * S.batch) S.hsel = -1;
    if (S.mode === 'naive') { S.heads = 1; S.batch = 1; }
    if (!(S.S === 4 && S.T === 4 && S.d === 2 && S.heads * S.batch === 1) && S.seed === 0) S.seed = 1;
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
  const upto = (n) => Array.from({ length: n }, (_, i) => i + 1);
  function knobs() {
    const k = $('knobs'); k.innerHTML = '';
    const sec = (t) => k.appendChild(el('div', 'ksec', t));
    sec('Shape');
    k.appendChild(slider('Queries Tq  <code>-S</code>', 'S', 1, 8, 1));
    k.appendChild(slider('Keys Tkv  <code>--kv-len</code>', 'T', 1, 8, 1));
    k.appendChild(seg('Head width d  <code>--head-dim</code>', 'd', [1, 2, 3, 4]));
    if (S.mode !== 'naive') {
      k.appendChild(seg('Heads  <code>--heads</code>', 'heads', [1, 2, 3, 4]));
      k.appendChild(seg('Batch  <code>--batch</code>', 'batch', [1, 2]));
    }
    const nums = el('div', 'kg'); nums.appendChild(el('div', 'kl', 'Numbers'));
    const nb = el('div', 'seg');
    if (S.S === 4 && S.T === 4 && S.d === 2 && S.heads * S.batch === 1) { const e = el('button', 'btn', 'Running example'); e.type = 'button'; e.setAttribute('aria-pressed', S.seed === 0 ? 'true' : 'false'); e.addEventListener('click', () => { S.seed = 0; rebuild(); }); nb.appendChild(e); }
    const r = el('button', 'btn', 'New random −1…2'); r.type = 'button'; r.addEventListener('click', () => { S.seed = (S.seed % 97) + 1; rebuild(); }); nb.appendChild(r);
    nums.appendChild(nb); k.appendChild(nums);
    sec('Algorithm');
    k.appendChild(seg('Mode', 'mode', ['naive', 'flash', 'inner'], (v) => ({ naive: 'Plain', flash: 'FlashAttention', inner: 'Inside a block' }[v])));
    if (S.mode === 'naive') { sec('Playback'); k.appendChild(slider('Speed', 'speed', 0.25, 4, 0.25, (v) => v + '×')); return; }
    sec('Blocks');
    k.appendChild(seg('Br: query rows per program  <code>--br</code>', 'Br', upto(S.S)));
    k.appendChild(seg('Bc: keys per block  <code>--bc</code>', 'Bc', upto(S.T)));
    sec('Inner matmuls  <code>--stationarity</code>');
    k.appendChild(seg('S = QKᵀ flow', 'qk', ['os', 'ws', 'is']));
    if (S.mode === 'flash') k.appendChild(seg('O += PV flow', 'pv', ['os', 'ws', 'is']));
    k.appendChild(seg('Array tile', 'tile', [1, 2]));
    if (S.mode === 'flash') {
      sec('Chip');
      k.appendChild(slider('Cores', 'cores', 1, 8, 1));
      const HT = S.heads * S.batch;
      if (HT > 1) k.appendChild(seg('Show head', 'hsel', [-1].concat(Array.from({ length: HT }, (_, i) => i)), (v) => (v < 0 ? 'the running one' : String(v))));
    }
    sec('Playback');
    k.appendChild(slider('Speed', 'speed', 0.25, 4, 0.25, (v) => v + '×'));
  }
  function command() {
    const f = ['bwz attention -c <chip>', '-S ' + S.S];
    if (S.T !== S.S) f.push('--kv-len ' + S.T);
    f.push('--head-dim ' + S.d);
    if (S.heads > 1) f.push('--heads ' + S.heads);
    if (S.batch > 1) f.push('--batch ' + S.batch);
    f.push('--br ' + S.Br, '--bc ' + S.Bc);
    if (S.qk === S.pv && S.qk !== 'os') f.push('--stationarity ' + S.qk);
    return f.join(' ');
  }
  function rule() {
    const R = $('rule');
    if (S.mode === 'naive') {
      R.innerHTML = '<div class="rl"><span class="rk">plain</span>S = QKᵀ/√d, all ' + S.S + '×' + S.T + ' scores<span class="ar">→</span>P = softmax(S) row by row<span class="ar">→</span>O = P V<span class="ar">·</span><span class="rk">through memory</span><b class="tm">' + 4 * S.S * S.T + '</b> numbers of S and P</div>';
      return;
    }
    const P = A.problem(S), pl = A.flashPlan(P, S), T = A.flashTotals(P, pl, S);
    let h = '<div class="rl"><span class="rk">programs</span>' + P.HT + ' head' + (P.HT > 1 ? 's' : '') + ' × ceil(' + S.S + ' / Br=' + pl.Br + ') = <b>' + pl.programs + '</b><span class="ar">·</span><span class="rk">on</span>' + S.cores + ' core' + (S.cores > 1 ? 's' : '') + ' = <b>' + pl.W + ' wave' + (pl.W > 1 ? 's' : '') + '</b><span class="ar">·</span><span class="rk">key blocks</span>ceil(' + S.T + ' / Bc=' + pl.Bc + ') = <b>' + pl.kb + '</b></div>';
    h += '<div class="rl"><span class="rk">K, V read</span>' + T.kvRead + (T.kvRead > T.kvOnce ? ' <b class="tsp">(' + pl.streams + ' head-streams for ' + P.HT + ' heads)</b>' : ' (each head once)') + '<span class="ar">·</span><span class="rk">O rescaled</span>' + T.rescaled + ' = heads · Tq · d · (blocks − 1)' + (S.qk !== 'os' || S.pv !== 'os' ? '<span class="ar">·</span><span class="rk">inner partial adds</span>' + T.innerAdds : '') + '</div>';
    h += '<div class="rl small"><span class="rk">same flags</span><code>' + command() + '</code></div>';
    R.innerHTML = h;
  }
  function trace() {
    const d = $('trace'); if (!view || !$('tracebox').open) return;
    const lines = view.trace(400);
    d.textContent = lines.length ? lines.join('\n') : '(nothing yet — take a step)';
    d.scrollTop = d.scrollHeight;
  }
  function rebuild() {
    fix();
    if (view) view.destroy();
    const host = $('view'); host.innerHTML = '';
    const d = el('div'); host.appendChild(d);
    const P = A.problem(S);
    const cfg = Object.assign({}, S, { P, onStep: trace });
    view = new ({ naive: A.NaiveView, flash: A.FlashView, inner: A.InnerView }[S.mode])(d, cfg);
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
  chapter('plain');
})();
