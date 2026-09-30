/* Interactive notebook: knobs + chapters that mirror the deck. */
(function () {
  'use strict';
  const X = window.MM2;
  const DEF = { M: 4, N: 4, K: 4, seed: 0, mode: 'single', tiled: false, tm: 2, tn: 2, tk: 2, border: 'ijk', inner: 'ijk', gran: 'op', cores: 4, flow: 'os', g: 1, S: 1, pieces: 2, slots: 2, load: 2, sched: 'overlap', group: 'block', both: true, speed: 1 };
  const S = Object.assign({}, DEF);
  const $ = (id) => document.getElementById(id);
  const el = (t, c, h) => { const e = document.createElement(t); if (c) e.className = c; if (h != null) e.innerHTML = h; return e; };
  const BVn = (o) => o.split('').map((v) => X.BV[v]).join(' ');

  const CH = [
    { id: 'textbook', t: 'Textbook loop', set: {}, s: '<h2>The loop you already know</h2><p>For each row i and column j, walk along k and add up the products. Every C value is started, finished and never touched again. During the whole k loop the running sum stays put, so this is the <b>inner-product form</b>, also called <b>output-stationary (os)</b>.</p><p class="try">Try: press <b>Bigger jump</b> to finish one C value per click.</p>' },
    { id: 'rule', t: 'What stays put', set: { inner: 'jki' }, s: '<h2>The one rule</h2><p>Each operand uses two of the three letters: A[i][k], B[k][j], C[i][j]. The innermost loop changes one letter every step. The operand that does not contain it <b>stays put</b> (it glows).</p><p class="try">Try: change <b>Loop order</b>. Ends in k → C glows (os). Ends in i → B glows (ws). Ends in j → A glows (is).</p>' },
    { id: 'row', t: 'Row-wise', set: { inner: 'ikj', gran: 'inner' }, s: '<h2>Row-wise (Gustavson), ikj</h2><p>One A value is kept while a whole row of B streams past; the results are added into a row of C. One step here is a whole inner loop: <code>C[i,:] += A[i][k] * B[k,:]</code>. A stays put: <b>input-stationary</b>.</p>' },
    { id: 'col', t: 'Column-wise', set: { inner: 'jki', gran: 'inner' }, s: '<h2>Column-wise, jki</h2><p>One B value is kept while a column of A streams past: <code>C[:,j] += A[:,k] * B[k][j]</code>. B stays put: <b>weight-stationary</b>.</p>' },
    { id: 'outer', t: 'Outer product', set: { inner: 'kij', gran: 'inner' }, s: '<h2>Outer product, kij / kji</h2><p>k is outermost. For each k, column k of A times row k of B is added into <b>all</b> of C (a <b>rank-1 update</b>). Every C value stays pale until the last k.</p><p class="try">Try: watch <b>most partial at once</b> — it reaches 16, the whole of C.</p>' },
    { id: 'partial', t: 'Partial sums', set: { inner: 'kij' }, s: '<h2>Accumulating over time</h2><p>Outside the inner-product form, a C value gets one term per visit and waits, pale, in <b>one place</b> while it is added to at <b>many moments</b>. That is <b>accumulating over time</b>. The counter <b>most partial at once</b> is how many such waiting values you must store: 1 for ijk, a row for ikj, a column for jki, all of C for kij.</p><p class="try">Try: switch Loop order between ijk, ikj, kij and compare the counter at the end.</p>' },
    { id: 'tile', t: 'Tiling', set: { tiled: true, gran: 'block' }, s: '<h2>Tiling: the same thing, in chunks</h2><p>Cut every matrix into blocks. The block loops <b>mt, nt, kt</b> run in any of the 6 orders, and each step is a small block product. The rule works one level up: innermost kt → the C block stays put, and so on.</p><p class="try">Try: change <b>Block loop order</b> and the tile sizes, or make the matrices 8×8.</p>' },
    { id: 'open', t: 'One block, opened', set: { tiled: true, gran: 'op' }, s: '<h2>Inside a block product</h2><p>Set the step to <b>1 multiply</b> with tiling on: each block product is just the textbook loop on a small piece. The dashed outlines mark the current block of A, B and C.</p>' },
    { id: 'hw', t: 'Hardware step size', set: { tiled: true, gran: 'inner', inner: 'kij' }, s: '<h2>How big is one step?</h2><p>The loops do not change; the hardware decides how many happen at once. <b>1 multiply</b> is a CPU core. <b>1 inner loop</b> is a vector operation, like a row of A pushed through a block of B stored in an in-memory-compute array (real Metis figure you gave: 512×512). <b>1 block product</b> is a GPU tensor core (real: 16×16 @ 16×16 per warp-level operation).</p>' },
    { id: 'os', t: 'Cores: grid & sweep', set: { mode: 'many', flow: 'os' }, s: '<h2>Several cores: grid and sweep</h2><p>Two block loops are shared out over the cores and run at the same time — the <b>tile grid</b>, one row per core in the timeline. The third loop runs in order inside each core — the <b>sweep</b>. In os each core keeps its own C block, so the sum over K stays on one core.</p><p class="try">Try: hover a bar in the timeline to see which blocks that core keeps and makes.</p>' },
    { id: 'ws', t: 'K on the grid', set: { mode: 'many', flow: 'ws' }, s: '<h2>When K is spread over cores</h2><p>In ws (cores keep B blocks) or is (cores keep A blocks), two cores make pieces of the <b>same</b> C block. Those cells show “2×” until an extra step adds the pieces <b>across cores</b> — accumulating over <b>space</b>.</p><p class="try">Try: hover a bar — the magenta outline shows the other core that makes the same C blocks.</p>' },
    { id: 'waves', t: 'Waves', set: { mode: 'many', flow: 'os', tm: 1, tn: 2, tk: 2, cores: 3 }, s: '<h2>More tiles than cores: waves</h2><p>8 tiles, 3 cores: the cores take tiles in rounds, called <b>waves</b>. The last wave is only partly full, so some cores sit idle (hatched).</p><p class="try">Try: set Cores to 4 (2 full waves), 5, or 8.</p>' },
    { id: 'osidle', t: 'os runs out of tiles', set: { mode: 'many', flow: 'os', cores: 8 }, s: '<h2>os runs out of tiles</h2><p>In os one core owns one output block and sweeps all of K inside it, so the number of tiles is the number of output blocks: 4 here. With 8 cores, 4 sit idle however long K is.</p><p class="try">Try: raise M and N for more output blocks and more busy cores. Or keep them and go on to split-K.</p>' },
    { id: 'split', t: 'Split-K', set: { mode: 'splitk', pieces: 2, cores: 8 }, s: '<h2>Split-K (an os strategy)</h2><p>Cut each block\'s K sweep into <b>P</b> pieces and make <b>(output block, piece of K)</b> the unit of work: 4 blocks × 2 pieces = 8 tiles for 8 cores. Each piece gives a <b>pale</b> partial block that goes out to <b class="tm">memory</b>; a <b class="tm">second pass</b> reads the pieces back and adds them into C.</p><p class="try">Try: P = 1 (plain os, 4 idle cores), P = 4 (16 tiles, 2 waves, 4 pieces through memory). Watch the violet memory counters: 2·P·16 values.</p>' },
    { id: 'splitws', t: 'Split-K vs ws', set: { mode: 'many', flow: 'ws', tk: 1, cores: 8, g: 1 }, s: '<h2>The other way to 8 tiles: ws</h2><p>ws puts <b>every</b> K-slice on the grid: 4 slices × 2 column blocks = 8 tiles, with no trip through memory. Compare with split-K (P = 2, 8 cores):</p><div class="tw"><table class="cmp"><tr><th></th><th>split-K, P = 2</th><th>ws, K on the grid</th></tr><tr><td>pieces per C value</td><td>2 (big, contiguous)</td><td>4 (every slice)</td></tr><tr><td>additions over time</td><td>32</td><td>0</td></tr><tr><td>the rest</td><td class="mem">16 in pass 2, after 32 written + 32 read</td><td class="tsp">48 across cores</td></tr><tr><td>total</td><td>64 multiplies, 48 additions</td><td>64 multiplies, 48 additions</td></tr></table></div><p class="try">Both buy parallel work by cutting K: split-K pays with memory traffic, ws with additions between cores.</p>' },
    { id: 'wsA', t: 'Weight sets: hide loading', set: { mode: 'wsets', group: 'block', sched: 'overlap', slots: 2, load: 2, both: true }, s: '<h2>(a) Weight sets hide the loading</h2><p>One ws core can <b>store</b> 2 blocks of B (2 weight sets, or slots) but computes with one at a time. While it computes with slot 0, the write port fills slot 1 with the next block; then the roles swap. Both timelines are drawn on the same scale.</p><p class="try">Try: load time 6, longer than the 4 compute steps per block, so even overlapped the compute waits. Slots = 1: nothing to overlap with.</p><p class="try">Real (your figures): Axelera Metis has 4 cores, each with 4 weight sets of 512×512 bytes, computing with one at a time. 4 × 512 × 512 B = 1 MiB per core, which matches the compute memory per core Axelera publishes.</p>' },
    { id: 'wsB', t: 'Weight sets: K in one place', set: { mode: 'wsets', group: 'row', sched: 'overlap', slots: 2, load: 2, both: false }, s: '<h2>(b) Weight sets keep K in one place</h2><p>The slots hold successive K-pieces of the <b>same</b> column of B. Each row of A goes through slot 0, then slot 1, adding into <b>one accumulator</b>: that row of C finishes on this core, with no additions between cores.</p><p class="try">Try: Slots = 1. Now every row must pass B[0,0] before B[1,0] is loaded, and <b>most partial at once</b> jumps from 2 to 8.</p>' },
    { id: 'kg', t: 'Slots → K-groups', set: { mode: 'many', flow: 'ws', tm: 2, tn: 2, tk: 1, cores: 4, g: 2 }, s: '<h2>More cores than columns: K-groups</h2><p>B has only 2 column blocks but there are 4 cores. Cut K into 4 slices and give each core a <b>group</b> of g slices in its slots: it sums its group over time, and only one partial per core is added across cores at the end.</p><p class="try">Try: K-tiles per core = 1, 2, 4 and watch the two addition counters trade places.</p>' },
    { id: 'big', t: 'Playground 8×8×8', set: { M: 8, N: 8, K: 8, seed: 3, mode: 'many', flow: 'ws', tm: 2, tn: 4, tk: 2, cores: 6, g: 2 }, s: '<h2>Playground</h2><p>Bigger random numbers, uneven counts of tiles and cores. Every knob is live; every run still ends with C == A @ B, and the additions over time plus across cores always add up to M·N·(K−1).</p>' }
  ];

  let view = null, cur = 'textbook';
  function fix() {
    ['tm', 'tn', 'tk'].forEach((t, x) => {
      const dim = [S.M, S.N, S.K][x];
      if (dim % S[t] !== 0 || S[t] > dim) { const d = X.divisors(dim).filter((v) => v <= S[t]); S[t] = d[d.length - 1]; }
    });
    const nK = S.K / S.tk;
    if (nK % S.g) S.g = 1;
    if (nK % S.S) S.S = 1;
    if (S.K % S.pieces) { const d = X.divisors(S.K).filter((v) => v <= S.pieces); S.pieces = d[d.length - 1]; }
    S.slots = Math.max(1, Math.min(4, S.slots));
    if (S.mode === 'single' && !S.tiled && S.gran === 'block') S.gran = 'inner';
    if (!(S.M === 4 && S.N === 4 && S.K === 4) && S.seed === 0) S.seed = 1;
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
    r.addEventListener('input', () => { S[key] = +r.value; $(id + '-v').textContent = fmt ? fmt(S[key]) : S[key]; if (key === 'speed') { if (view) { view.cfg.speed = S.speed; if (view.timer) { view.stop(); view.play(); } } } });
    r.addEventListener('change', () => { if (key !== 'speed') rebuild(); });
    g.appendChild(r); return g;
  }
  function knobs() {
    const k = $('knobs'); k.innerHTML = '';
    const sec = (t) => k.appendChild(el('div', 'ksec', t));
    sec('Matrices');
    k.appendChild(slider('M (rows of A, C)', 'M', 1, 8, 1));
    k.appendChild(slider('N (cols of B, C)', 'N', 1, 8, 1));
    k.appendChild(slider('K (summed away)', 'K', 1, 8, 1));
    const nums = el('div', 'kg'); nums.appendChild(el('div', 'kl', 'Numbers'));
    const nb = el('div', 'seg');
    if (S.M === 4 && S.N === 4 && S.K === 4) { const e = el('button', 'btn', 'Running example'); e.type = 'button'; e.setAttribute('aria-pressed', S.seed === 0 ? 'true' : 'false'); e.addEventListener('click', () => { S.seed = 0; rebuild(); }); nb.appendChild(e); }
    const r = el('button', 'btn', 'New random 0…3'); r.type = 'button'; r.addEventListener('click', () => { S.seed = (S.seed % 97) + 1; rebuild(); }); nb.appendChild(r);
    nums.appendChild(nb); k.appendChild(nums);
    sec('Where it runs');
    k.appendChild(seg('Mode', 'mode', ['single', 'many', 'splitk', 'wsets'], (v) => ({ single: 'One core', many: 'Many cores', splitk: 'Split-K', wsets: 'Weight sets' }[v])));
    if (S.mode === 'single') {
      k.appendChild(seg('Loop order (inside)', 'inner', X.ORDERS));
      k.appendChild(seg('Tiling', 'tiled', [false, true], (v) => (v ? 'Blocks' : 'None')));
    }
    if (S.mode === 'splitk') {
      sec('Tiles');
      k.appendChild(seg('tm (block rows)', 'tm', X.divisors(S.M)));
      k.appendChild(seg('tn (block cols)', 'tn', X.divisors(S.N)));
      sec('Split-K');
      k.appendChild(seg('P (pieces of K)', 'pieces', X.divisors(S.K)));
      k.appendChild(slider('Cores', 'cores', 1, 8, 1));
    } else if (S.mode === 'wsets') {
      sec('B blocks');
      k.appendChild(seg('tk (block depth)', 'tk', X.divisors(S.K)));
      k.appendChild(seg('tn (block cols)', 'tn', X.divisors(S.N)));
      sec('Weight sets');
      k.appendChild(seg('Slots', 'slots', [1, 2, 3, 4]));
      k.appendChild(slider('Steps to write a block', 'load', 1, 6, 1));
      k.appendChild(seg('Schedule', 'sched', ['seq', 'overlap'], (v) => (v === 'seq' ? 'load, then compute' : 'overlapped')));
      k.appendChild(seg('Order', 'group', ['block', 'row'], (v) => (v === 'block' ? '(a) block by block' : '(b) row through all slots')));
      k.appendChild(seg('Timelines', 'both', [true, false], (v) => (v ? 'both schedules' : 'this one only')));
    } else if (S.mode === 'many' || S.tiled) {
      sec('Tiles');
      k.appendChild(seg('tm (block rows)', 'tm', X.divisors(S.M)));
      k.appendChild(seg('tn (block cols)', 'tn', X.divisors(S.N)));
      k.appendChild(seg('tk (block depth)', 'tk', X.divisors(S.K)));
    }
    if (S.mode === 'single') {
      if (S.tiled) k.appendChild(seg('Block loop order', 'border', X.ORDERS, BVn));
      sec('Step size');
      k.appendChild(seg('One step is', 'gran', S.tiled ? ['op', 'inner', 'block'] : ['op', 'inner'], (v) => ({ op: '1 multiply', inner: '1 inner loop', block: '1 block product' }[v])));
    } else if (S.mode === 'many') {
      sec('Cores');
      k.appendChild(slider('Cores', 'cores', 1, 8, 1));
      k.appendChild(seg('Flow: what each core keeps', 'flow', ['os', 'ws', 'is'], (v) => ({ os: 'os · C', ws: 'ws · B', is: 'is · A' }[v])));
      const nK = S.K / S.tk;
      if (S.flow !== 'os') k.appendChild(seg('K-tiles per core (K-groups)', 'g', X.divisors(nK)));
    }
    sec('Playback');
    k.appendChild(slider('Speed', 'speed', 0.25, 4, 0.25, (v) => v + '×'));
  }
  function rule() {
    const R = $('rule');
    if (S.mode === 'splitk') {
      const P = X.problem(S.M, S.N, S.K, S.seed), pl = X.splitkPlan(P, S), MN = S.M * S.N;
      R.innerHTML = '<div class="rl"><span class="rk">unit of work</span><b>(output block, piece of K)</b><span class="ar">·</span>' + pl.nM * pl.nN + ' blocks × ' + pl.Pp + ' piece' + (pl.Pp > 1 ? 's' : '') + ' = <b>' + pl.tiles.length + ' tiles</b> on ' + S.cores + ' cores = ' + pl.W + ' wave' + (pl.W > 1 ? 's' : '') + '<span class="ar">·</span><span class="rk">kept in each core</span><span class="stay-c">a C block</span><b>os</b></div>' +
        '<div class="rl"><span class="rk">K-sum</span>' + (pl.Pp > 1 ? 'over time inside a piece (' + MN * (S.K - pl.Pp) + ' additions), then <b class="tm">through memory</b>: ' + pl.Pp * MN + ' written, ' + pl.Pp * MN + ' read back, ' + (pl.Pp - 1) * MN + ' additions in pass 2' : '<b>over time only</b>: each block is finished on one core and written once') + '</div>';
      return;
    }
    if (S.mode === 'wsets') {
      const P = X.problem(S.M, S.N, S.K, S.seed), a = X.wsSchedule(P, S, 'seq'), b = X.wsSchedule(P, S, 'overlap');
      R.innerHTML = '<div class="rl"><span class="rk">one core</span>' + S.slots + ' slot' + (S.slots > 1 ? 's' : '') + ' of ' + S.tk + '×' + S.tn + ' B<span class="ar">·</span><span class="rk">stays put</span><span class="stay-b">the B block in the active slot</span><b>ws</b><span class="ar">·</span><span class="rk">one step</span>one row of A through one slot (' + S.tk * S.tn + ' multiplies)</div>' +
        '<div class="rl"><span class="rk">time</span>load, then compute: <b>' + a.T + '</b> steps<span class="ar">·</span>overlapped: <b>' + b.T + '</b> steps<span class="ar">·</span><span class="rk">K-sum</span>over time in one core, <b>0</b> additions across cores</div>';
      return;
    }
    if (S.mode === 'single') {
      const v = S.inner[2], st = X.STAY[v], ch = { A: 'A[i][k]', B: 'B[k][j]', C: 'C[i][j]' };
      const moving = ['A', 'B', 'C'].filter((x) => x !== st).map((x) => '<span class="t' + x.toLowerCase() + '">' + ch[x] + '</span>').join(' and ');
      let h = '<div class="rl"><span class="rk">innermost loop</span><b>' + v + '</b><span class="ar">→</span><span class="rk">changes</span>' + moving + '<span class="ar">→</span><span class="rk">stays put</span><span class="stay-' + st.toLowerCase() + '">' + ch[st] + '</span><span class="ar">→</span><b>' + X.STATN[st] + '</b> · ' + X.FORM[S.inner] + '</div>';
      if (S.tiled) { const bv = S.border[2], bst = X.STAY[bv]; h += '<div class="rl"><span class="rk">innermost block loop</span><b>' + X.BV[bv] + '</b><span class="ar">→</span><span class="rk">stays put</span><span class="stay-' + bst.toLowerCase() + '">a ' + bst + ' block</span><span class="ar">→</span><b>' + X.SHORT[bst] + '</b> at block level</div>'; }
      h += '<div class="rl small">' + { op: 'One step = 1 multiply: a CPU core.', inner: 'One step = a whole inner loop: a vector operation (like one row through an in-memory array).', block: 'One step = a whole block product: a tensor core.' }[S.gran] + '</div>';
      R.innerHTML = h;
    } else {
      const f = S.flow, st = { os: 'C', ws: 'B', is: 'A' }[f];
      const grid = { os: 'mt × nt', ws: 'kgroup × nt', is: 'mt × kgroup' }[f], sweep = { os: 'kt', ws: 'mt (then kt inside the group)', is: 'nt (then kt inside the group)' }[f];
      const pl = X.parallelPlan(X.problem(S.M, S.N, S.K, S.seed), S);
      R.innerHTML = '<div class="rl"><span class="rk">tile grid (same time)</span><b>' + grid + '</b><span class="ar">·</span><span class="rk">sweep (in order)</span><b>' + sweep + '</b><span class="ar">·</span><span class="rk">kept in each core</span><span class="stay-' + st.toLowerCase() + '">a ' + st + ' block</span><b>' + f + '</b></div>' +
        '<div class="rl"><span class="rk">K-sum</span>' + (pl.maxParts > 1 ? 'over time inside each core, then <b class="tsp">across cores</b> (' + pl.maxParts + ' pieces per C block, ' + pl.R + ' combine level' + (pl.R > 1 ? 's' : '') + ')' : '<b>over time only</b>, each C block is made on one core') +
        '<span class="ar">·</span><span class="rk">schedule</span>' + pl.U + ' tiles on ' + S.cores + ' cores = ' + pl.W + ' wave' + (pl.W > 1 ? 's' : '') + '</div>';
    }
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
    const P = X.problem(S.M, S.N, S.K, S.seed);
    const cfg = Object.assign({}, S, { P, onStep: trace });
    view = new ({ single: X.SingleView, many: X.ParallelView, splitk: X.SplitKView, wsets: X.WSView }[S.mode])(d, cfg);
    knobs(); rule(); trace();
  }
  function chapter(id) {
    const c = CH.find((x) => x.id === id); cur = id;
    Object.keys(DEF).forEach((k) => { S[k] = DEF[k]; });
    Object.assign(S, c.set);
    if (S.mode === 'many') { S.tiled = true; }
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
  chapter('textbook');
})();
