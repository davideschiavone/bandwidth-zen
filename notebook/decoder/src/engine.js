/* One transformer decoder layer, generating: the prompt goes through once (prefill), then every new
   token goes through alone (decode), attending to the K and V of every token before it — the KV cache.

   Causal attention is what makes the cache possible: a token may look only at itself and earlier
   tokens, so S is a lower triangle and a past token's K, V and output never change once computed.
   Without a cache (cfg.cache = false) every step recomputes the whole sequence; the outputs are the
   same and the work is not. Every step is checked against a full causal recompute written separately.

   Reuses the encoder notebook's arithmetic, chips and cost view (ENC), the attention notebook's softmax
   reference and S/K/V/O styles (ATT), the matmul notebook's grid and controls (MM2._), and the shared
   deck widgets (NB). Counts use the same conventions as the encoder, with causal attention charged only
   on the lower triangle and GQA (kvh < h key/value heads) shrinking K and V. */
(function (global) {
  'use strict';
  const X = global.MM2, H = X._, ATT = global.ATT, ENC = global.ENC, U = ENC._;
  const { el, addCls, counters, Mat, mix, key, rng } = H;
  const { matmul, normRows, gelu, silu, fmt, fmtN, fmtTime, NORM_OPS, ACT_OPS, GATED, SOFTMAX_OPS, lcg } = U;

  // ---------------- the problem ----------------
  // Tp prompt tokens, then G generated ones. The generated ids are given: choosing them needs the LM
  // head on top of the whole stack, which is not part of one layer.
  function problem(o) {
    const d = o.d, h = o.h, kvh = o.kvh, dh = d / h, dkv = kvh * dh, ffn = o.ffn, vocab = o.vocab || 16;
    const r = lcg(o.seed || 9);
    const mat = (R, C, lo, span) => rng(R).map(() => rng(C).map(() => Math.floor(r() * span) + lo));
    const table = mat(vocab, d, -1, 4), T = o.Tp + o.G, ids = rng(T).map(() => Math.floor(r() * vocab));
    const W = { q: mat(d, d, -1, 3), k: mat(d, dkv, -1, 3), v: mat(d, dkv, -1, 3), o: mat(d, d, -1, 3),
      up: mat(d, ffn, -1, 3), gate: GATED[o.act] ? mat(d, ffn, -1, 3) : null, down: mat(ffn, d, -1, 3) };
    const P = { Tp: o.Tp, G: o.G, T, d, h, kvh, dh, dkv, ffn, vocab, act: o.act, norm: o.norm, table, ids, W, cache: o.cache !== false };
    P.steps = run(P);
    P.REF = reference(P);
    return P;
  }

  // Rows at positions p0 … p0+n−1 through the layer. Their K and V are appended to the cache first;
  // row p then attends to cache rows 0 … p and nothing after — the causal mask.
  function rows(P, p0, n, cache) {
    const xs = rng(n).map((i) => P.table[P.ids[p0 + i]].slice());
    const n1 = normRows(xs, P.norm), q = matmul(n1, P.W.q), k = matmul(n1, P.W.k), v = matmul(n1, P.W.v);
    k.forEach((r) => cache.K.push(r)); v.forEach((r) => cache.V.push(r));
    const probs = [];                                     // head 0's probabilities, for the picture
    const a = q.map((qr, i) => {
      const p = p0 + i, out = new Array(P.d).fill(0);
      for (let hh = 0; hh < P.h; hh++) {
        const g = Math.floor(hh / (P.h / P.kvh)), qs = qr.slice(hh * P.dh, (hh + 1) * P.dh);
        const s = rng(p + 1).map((j) => qs.reduce((acc, x, c) => acc + x * cache.K[j][g * P.dh + c], 0) / Math.sqrt(P.dh));
        const m = Math.max(...s), e = s.map((x) => Math.exp(x - m)), l = e.reduce((x, y) => x + y, 0);
        if (hh === 0) probs.push(e.map((x) => x / l));
        for (let c = 0; c < P.dh; c++) out[hh * P.dh + c] = e.reduce((acc, w, j) => acc + w * cache.V[j][g * P.dh + c], 0) / l;
      }
      return out;
    });
    const x1 = matmul(a, P.W.o).map((r, i) => r.map((x, c) => x + xs[i][c]));
    const n2 = normRows(x1, P.norm), up = matmul(n2, P.W.up), gate = P.W.gate ? matmul(n2, P.W.gate) : null;
    const f = up.map((r, i) => r.map((x, j) => (gate ? (P.act === 'swiglu' ? silu(gate[i][j]) : gelu(gate[i][j])) * x : P.act === 'relu' ? Math.max(0, x) : gelu(x))));
    const out = normRows(matmul(f, P.W.down).map((r, i) => r.map((x, c) => x + x1[i][c])), P.norm);
    return { xs, n1, q, k, v, probs, out };
  }
  // Operations for rows p0 … p0+n−1, by the encoder's conventions; attention over p+1 keys per row.
  function rowOps(P, p0, n) {
    const { d, dkv, ffn, h } = P, nf = NORM_OPS[P.norm], keys = rng(n).reduce((s, i) => s + p0 + i + 1, 0);
    return 2 * n * d * (2 * d + 2 * dkv) + 4 * keys * d + SOFTMAX_OPS * h * keys + (GATED[P.act] ? 6 : 4) * n * d * ffn + n * ffn * ACT_OPS[P.act] + 2 * n * d + 3 * n * d * nf;
  }
  function run(P) {
    const cache = { K: [], V: [] }, steps = [], outs = [];
    for (let s = 0; s <= P.G; s++) {
      const last = s === 0 ? P.Tp - 1 : P.Tp + s - 1;   // the newest position after this step
      let p0, n;
      if (P.cache) { p0 = s === 0 ? 0 : last; n = s === 0 ? P.Tp : 1; }
      else { cache.K = []; cache.V = []; p0 = 0; n = last + 1; }   // no cache: everything again
      const kvRead = P.cache && s > 0 ? 2 * p0 * P.dkv : 0;          // past rows come back from memory
      const r = rows(P, p0, n, cache);
      r.out.forEach((row, i) => { outs[p0 + i] = row; });
      steps.push({ s, p0, n, last, L: last + 1, probs: r.probs, out: r.out, x: r.xs, n1: r.n1, q: r.q, kvRead, kvWrite: 2 * (P.cache ? n : 0) * P.dkv,
        ops: rowOps(P, p0, n), K: cache.K.map((x) => x.slice()), V: cache.V.map((x) => x.slice()) });
    }
    P.outs = outs;
    return steps;
  }
  // A full causal recompute for every position: the whole sequence's Q, K and V at once, then each
  // row on its own with the attention notebook's softmax over keys 0 … p. It shares only the matmul and
  // norm helpers with rows() above — no cache, no step order. The cached walk must land on exactly this.
  function reference(P) {
    const allX = P.ids.map((t) => P.table[t]);
    const n1 = normRows(allX, P.norm), Q = matmul(n1, P.W.q), K = matmul(n1, P.W.k), V = matmul(n1, P.W.v);
    return rng(P.T).map((p) => {
      const a = [].concat(...rng(P.h).map((hh) => {
        const g = Math.floor(hh / (P.h / P.kvh)), cut = (M, c0) => M.slice(0, p + 1).map((r) => r.slice(c0, c0 + P.dh));
        return ATT.refAttention([Q[p].slice(hh * P.dh, (hh + 1) * P.dh)], cut(K, g * P.dh), cut(V, g * P.dh), 1 / Math.sqrt(P.dh))[0];
      }));
      const x1 = matmul([a], P.W.o)[0].map((x, c) => x + allX[p][c]);
      const n2 = normRows([x1], P.norm)[0], up = matmul([n2], P.W.up)[0], gt = P.W.gate ? matmul([n2], P.W.gate)[0] : null;
      const f = up.map((x, j) => (gt ? (P.act === 'swiglu' ? silu(gt[j]) : gelu(gt[j])) * x : P.act === 'relu' ? Math.max(0, x) : gelu(x)));
      return normRows([matmul([f], P.W.down)[0].map((x, c) => x + x1[c])], P.norm)[0];
    });
  }
  const close = (a, b) => a.every((x, i) => Math.abs(x - b[i]) <= 1e-9 * Math.max(1, Math.abs(b[i])));

  // =====================================================================
  // GenerateView: prefill, then one token per step
  // =====================================================================
  function GenerateView(root, cfg) {
    this.cfg = cfg; this.root = root; this.P = cfg.P; root.classList.add('w', 'pv', 'att', 'dec');
    this.baseSpeed = 1300; this.t = 0; this.n = this.P.steps.length;
    const P = this.P;
    if (cfg.title !== false) {
      root.appendChild(el('div', 'w-head', '<span class="chip">prompt ' + P.Tp + ' tokens, then ' + P.G + ' generated</span><span class="chip">d = ' + P.d + ' · ' + P.h + ' heads · ' + P.kvh + ' KV head' + (P.kvh > 1 ? 's' : '') + '</span><span class="chip ' + (P.cache ? 'C' : 'mem') + '">' + (P.cache ? 'KV cache on' : 'no cache: recompute everything') + '</span>'));
    }
    this.strip = el('div', 'dec-tokens'); root.appendChild(this.strip);
    this.tok = P.ids.map((id, p) => { const b = el('span', 'dec-tok ' + (p < P.Tp ? 'prompt' : 'gen'), '<i>' + p + '</i>' + id); this.strip.appendChild(b); return b; });
    // The pseudo-code, lit by step: the prompt block on the prefill, the loop body on every decode step
    // (or, without a cache, the recompute-everything loop), and a last line naming this step's numbers.
    const kw = (w) => '<span class="kw">' + w + '</span>', cm = (w) => '<span class="cm"># ' + w + '</span>';
    this.lines = P.cache ? [
      { v: 'pre', html: cm('prefill: the whole prompt at once') },
      { v: 'pre', html: 'x = embed(prompt)                       ' + cm(P.Tp + ' rows') },
      { v: 'pre', html: '<span class="ta">q</span>, <span class="tb">k</span>, <span class="tb">v</span> = norm(x) · Wq, Wk, Wv' },
      { v: 'pre', html: '<span class="tb">K_cache</span>, <span class="tb">V_cache</span> = <span class="tb">k</span>, <span class="tb">v</span>' },
      { v: 'pre', html: 'P = softmax(<span class="ta">q</span> <span class="tb">K_cache</span>ᵀ/√dh, mask j &gt; i)  ' + cm('lower triangle') },
      { v: 'pre', html: '<span class="tc">out</span> = FFN(x + P <span class="tb">V_cache</span> · Wo)' },
      { v: 'dec', html: kw('for') + ' t ' + kw('in') + ' range(' + P.Tp + ', ' + P.T + '):         ' + cm('decode: one token per step') },
      { v: 'dec', html: '    x = embed(token[t])                 ' + cm('1 row') },
      { v: 'dec', html: '    <span class="ta">q</span>, <span class="tb">k</span>, <span class="tb">v</span> = norm(x) · Wq, Wk, Wv     ' + cm('[1, d] × [d, d]') },
      { v: 'dec', html: '    <span class="tb">K_cache</span>.append(<span class="tb">k</span>); <span class="tb">V_cache</span>.append(<span class="tb">v</span>)' },
      { v: 'dec', html: '    P = softmax(<span class="ta">q</span> <span class="tb">K_cache</span>ᵀ/√dh)       ' + cm('1 row, keys 0 … t') },
      { v: 'dec', html: '    <span class="tc">out[t]</span> = FFN(x + P <span class="tb">V_cache</span> · Wo)' }
    ] : [
      { v: 'all', html: kw('for') + ' t ' + kw('in') + ' range(' + (P.Tp - 1) + ', ' + P.T + '):         ' + cm('no cache: every step from scratch') },
      { v: 'all', html: '    x = embed(tokens[0 … t])            ' + cm('t + 1 rows') },
      { v: 'all', html: '    <span class="ta">q</span>, <span class="tb">k</span>, <span class="tb">v</span> = norm(x) · Wq, Wk, Wv     ' + cm('all rows again') },
      { v: 'all', html: '    P = softmax(<span class="ta">q</span> <span class="tb">k</span>ᵀ/√dh, mask j &gt; i)   ' + cm('the whole triangle again') },
      { v: 'all', html: '    <span class="tc">out</span> = FFN(x + P <span class="tb">v</span> · Wo); keep <span class="tc">out[t]</span>' }
    ];
    const codeRow = el('div', 'dec-code'); root.appendChild(codeRow);
    this.codeEl = el('pre', 'code'); codeRow.appendChild(this.codeEl);
    this.lineEls = this.lines.map((l) => { const d = el('div', 'ln', l.html); this.codeEl.appendChild(d); return d; });
    this.nowEl = el('div', 'dec-now'); codeRow.appendChild(this.nowEl);
    // First the inputs of attention: x, norm(x) and q for the rows this step computes. q is not kept:
    // only K and V go to the cache, because an old query is never needed again.
    const inRow = el('div', 'dec-row'); root.appendChild(inRow);
    this.X = new Mat('A', P.T, P.d, { rv: 't', cv: 'c', title: 'x = embed(token)', dims: P.T + '×' + P.d });
    this.N = new Mat('A', P.T, P.d, { rv: 't', cv: 'c', title: 'norm(x)', dims: P.T + '×' + P.d });
    this.Q = new Mat('Q', P.T, P.d, { rv: 't', cv: 'c', title: 'q = norm(x) · Wq', dims: P.T + '×' + P.d + (P.h > 1 ? ', ' + P.h + ' heads' : '') });
    [this.X, this.N, this.Q].forEach((m, i) => { if (i) inRow.appendChild(el('span', 'arrow', '→')); inRow.appendChild(m.root); });
    inRow.appendChild(el('div', 'dec-note', 'k = norm(x) · Wk and v = norm(x) · Wv are made the same way, ' + P.dkv + ' columns each, and go straight into the caches below. q is used once and dropped.'));
    const top = el('div', 'dec-row'); root.appendChild(top);
    this.S = new Mat('S', P.T, P.T, { rv: 'q', cv: 'k', title: 'P = softmax(S), head 0', dims: 'future masked ×' });
    const kvTitle = (m) => (P.cache ? m + ' cache' : m + ', recomputed every step');
    this.K = new Mat('K', P.T, P.dkv, { rv: 't', cv: 'c', title: kvTitle('K'), dims: P.T + '×' + P.dkv });
    this.V = new Mat('V', P.T, P.dkv, { rv: 't', cv: 'c', title: kvTitle('V'), dims: P.T + '×' + P.dkv });
    this.O = new Mat('O', P.T, P.d, { rv: 't', cv: 'c', title: 'layer out', dims: P.T + '×' + P.d });
    [this.S, this.K, this.V, this.O].forEach((m) => top.appendChild(m.root));
    this.cap = el('div', 'cap'); root.appendChild(this.cap);
    const foot = el('div', 'w-foot'); root.appendChild(foot);
    this.cnt = el('div', 'counters'); foot.appendChild(this.cnt);
    const right = el('div', 'stack right'); right.append(this.controls('To the end'), this.scrubber()); foot.appendChild(right);
    this.keys(); this.go(0);
  }
  GenerateView.prototype.inner = function () { this.go(this.n); return true; };
  GenerateView.prototype.render = function () {
    const P = this.P, done = this.P.steps.slice(0, this.t), cur = done[done.length - 1];
    const L = cur ? cur.L : 0;
    this.tok.forEach((b, p) => { b.classList.toggle('future', p >= L); b.classList.toggle('now', !!cur && p >= cur.p0 && p < cur.p0 + cur.n); });
    // S: every row computed so far shows its probabilities; above the diagonal is the future, masked.
    const cS = {}, sTxt = {};
    const shown = {};
    done.forEach((st) => st.probs.forEach((row, i) => { shown[st.p0 + i] = { row, live: st === cur }; }));
    for (let p = 0; p < P.T; p++) for (let j = 0; j < P.T; j++) {
      const k0 = key(p, j);
      if (j > p) { addCls(cS, k0, 'mask'); continue; }
      if (shown[p]) { addCls(cS, k0, shown[p].live ? 's-soft' : 's-past'); sTxt[k0] = fmt(shown[p].row[j]); }
    }
    this.S.paint(cS, (r, c) => sTxt[key(r, c)] || (c > r ? '×' : ''));
    const cache = cur ? { K: cur.K, V: cur.V } : { K: [], V: [] };
    const paintCache = (m, M) => {
      const cls = {};
      M.forEach((r, p) => r.forEach((_, c) => addCls(cls, key(p, c), cur && p >= cur.p0 && p < cur.p0 + cur.n ? 'read' : 'past')));
      m.paint(cls, (r, c) => (M[r] ? fmt(M[r][c]) : ''));
    };
    paintCache(this.K, cache.K); paintCache(this.V, cache.V);
    const cO = {};
    for (let p = 0; p < L; p++) for (let c = 0; c < P.d; c++) addCls(cO, key(p, c), cur && p >= cur.p0 && p < cur.p0 + cur.n ? 'fin write' : 'fin');
    this.O.paint(cO, (r, c) => (r < L ? fmt(P.outs[r][c]) : '·'));
    // x, norm(x), q: values for every row computed so far; this step's rows lit, earlier ones dimmed,
    // and earlier q rows marked as not kept.
    const rowsNow = (p) => !!cur && p >= cur.p0 && p < cur.p0 + cur.n;
    const val = { x: {}, n1: {}, q: {} };
    done.forEach((st) => ['x', 'n1', 'q'].forEach((f) => st[f].forEach((row, i) => { val[f][st.p0 + i] = row; })));
    const paintIn = (m, f, kind) => {
      const cls = {};
      for (let p = 0; p < L; p++) for (let c = 0; c < P.d; c++) addCls(cls, key(p, c), rowsNow(p) ? 'read' : f === 'q' ? 'qgone' : 'past');
      m.paint(cls, (r, c) => (val[f][r] && (rowsNow(r) || f !== 'q') ? fmt(val[f][r][c]) : r < L && f === 'q' ? '–' : ''));
    };
    paintIn(this.X, 'x'); paintIn(this.N, 'n1'); paintIn(this.Q, 'q');
    const lit = !cur ? null : P.cache ? (cur.s === 0 ? 'pre' : 'dec') : 'all';
    this.lineEls.forEach((d, i) => d.classList.toggle('on', this.lines[i].v === lit));
    this.nowEl.innerHTML = !cur ? 'Nothing has run yet.'
      : cur.s === 0 && P.cache ? '<b>Now:</b> prefill — rows 0 … ' + (P.Tp - 1) + ' together; row i sees keys 0 … i. Cache: ' + P.Tp + ' rows.'
        : P.cache ? '<b>Now:</b> t = ' + cur.last + ' — one row; its q against K_cache rows 0 … ' + cur.last + ' (' + (cur.L - 1) + ' read back, 1 new). Cache: ' + cur.L + ' rows.'
          : '<b>Now:</b> t = ' + cur.last + ' — all ' + cur.n + ' rows recomputed (' + cur.ops + ' operations) to get one new output.';
    const opsSoFar = done.reduce((s, x) => s + x.ops, 0), kvSoFar = done.reduce((s, x) => s + x.kvRead, 0);
    let cap;
    if (!cur) cap = 'Press <b>Step</b>. The first step is the <b>prefill</b>: all ' + P.Tp + ' prompt tokens at once. Then every step adds <b>one</b> token.';
    else if (cur.s === 0) cap = '<b>Prefill</b>: the ' + P.Tp + ' prompt tokens go through together. Each row of S may only look left — at itself and earlier tokens (× is the future, masked). Their K and V fill the first ' + P.Tp + ' rows of the cache.';
    else if (P.cache) cap = '<b>Token ' + cur.last + '</b> (step ' + this.t + '): <b>one</b> row through the layer — its Q against all ' + cur.L + ' cached K rows, its K and V appended. The ' + (cur.L - 1) + ' earlier rows of the cache are read back, not recomputed: their K and V cannot change, because they never saw this token.';
    else cap = '<b>Token ' + cur.last + '</b> (step ' + this.t + ') <b>without a cache</b>: all ' + cur.L + ' rows recomputed from scratch to get one new output. The earlier outputs come out exactly as before — that is why a cache is allowed.';
    if (this.t === this.n) {
      const ok = rng(P.T).every((p) => close(P.outs[p], P.REF[p]));
      cap += ' ' + (ok ? '<span class="ok">Check: every token\'s output equals a full causal recompute ✓</span>' : '<span class="tsp">differs ✗</span>');
    }
    this.cap.innerHTML = cap;
    this.cnt.innerHTML = counters([
      { label: 'step', value: this.t + '/' + this.n }, { label: 'tokens in context', value: L },
      { label: 'rows computed now', value: cur ? cur.n : 0 }, { label: 'operations now', value: cur ? cur.ops : 0 },
      { label: 'operations so far', value: opsSoFar }, { label: 'KV cache rows', value: P.cache ? L : 0 },
      { label: 'K,V numbers read back now', value: cur ? cur.kvRead : 0, cls: 'cm' }, { label: 'K,V read so far', value: kvSoFar, cls: 'cm' }
    ]);
  };
  GenerateView.prototype.trace = function () {
    return this.P.steps.slice(0, this.t).map((s) => (s.s === 0 ? 'prefill   ' : 'token ' + String(s.last).padEnd(4)) + ' rows ' + s.p0 + '…' + (s.p0 + s.n - 1) + ' computed, context ' + s.L + ', ' + s.ops + ' ops, ' + s.kvRead + ' K/V numbers read back');
  };
  mix(GenerateView.prototype);

  // =====================================================================
  // Real sizes: one decode step at context L (or a prefill of T tokens), priced by the encoder's view
  // =====================================================================
  function layerRows(o) {
    const B = o.batch || 1, d = o.d, f = o.ffn, dkv = (d / o.h) * o.kvh, gated = GATED[o.act], nf = NORM_OPS[o.norm];
    const pre = o.phase === 'prefill', M = B * (pre ? o.S : 1);
    const keys = B * (pre ? o.S * (o.S + 1) / 2 : o.L);               // causal: a triangle in prefill
    const kvPast = pre ? 0 : 2 * B * (o.L - 1) * dkv;                 // the cache read back in decode
    const L = [];
    const op = (id, title, group, ops, params, elems, kind) => L.push({ id, title, group, ops, params, elems, kind });
    const nrm = (id, group) => { if (nf) op(id, 'Norm', group, M * d * nf, (o.norm === 'layernorm' ? 2 : 1) * d, d * (o.norm === 'layernorm' ? 2 : 1) + 2 * M * d, 'norm'); };
    op('embed', 'Embedding', 'in', 0, o.vocab * d, M * d, 'gather');
    nrm('attn_norm', 'attn');
    op('q_proj', 'Q projection', 'attn', 2 * M * d * d, d * d, d * d + 2 * M * d, 'matmul');
    op('k_proj', 'K projection → cache', 'attn', 2 * M * d * dkv, d * dkv, d * dkv + M * d + M * dkv, 'matmul');
    op('v_proj', 'V projection → cache', 'attn', 2 * M * d * dkv, d * dkv, d * dkv + M * d + M * dkv, 'matmul');
    op('attn', pre ? 'Attention (causal)' : 'Attention over the cache', 'attn', 4 * keys * d + SOFTMAX_OPS * o.h * keys, 0, M * d * 2 + 2 * M * dkv + kvPast, 'attention');
    op('o_proj', 'Output projection', 'attn', 2 * M * d * d, d * d, d * d + 2 * M * d, 'matmul');
    op('attn_residual', 'Residual add', 'attn', M * d, 0, 3 * M * d, 'add');
    nrm('ffn_norm', 'ffn');
    op('ffn_up', 'FFN up', 'ffn', 2 * M * d * f, d * f, d * f + M * d + M * f, 'matmul');
    if (gated) op('ffn_gate', 'FFN gate', 'ffn', 2 * M * d * f, d * f, d * f + M * d + M * f, 'matmul');
    op('ffn_act', { relu: 'ReLU', gelu: 'GELU', swiglu: 'SwiGLU', geglu: 'GeGLU' }[o.act], 'ffn', M * f * ACT_OPS[o.act], 0, (gated ? 3 : 2) * M * f, 'act');
    op('ffn_down', 'FFN down', 'ffn', 2 * M * f * d, f * d, f * d + M * f + M * d, 'matmul');
    op('ffn_residual', 'Residual add', 'ffn', M * d, 0, 3 * M * d, 'add');
    nrm('final_norm', 'out');
    return L;
  }
  function priced(o, chip) {
    const c = ENC.CHIPS[chip];
    return layerRows(o).map((x) => {
      const bytes = x.elems * c.bytes, tc = x.ops / c.peak, tm = bytes / c.bw;
      return Object.assign({}, x, { bytes, intensity: bytes ? x.ops / bytes : 0, tc, tm, t: Math.max(tc, tm), bound: tc > tm ? 'compute' : 'memory' });
    });
  }
  // The totals by formula alone, for the view's closing check.
  function closed(o) {
    const B = o.batch || 1, d = o.d, f = o.ffn, dkv = (d / o.h) * o.kvh, gated = GATED[o.act], nf = NORM_OPS[o.norm];
    const pre = o.phase === 'prefill', M = B * (pre ? o.S : 1), keys = B * (pre ? o.S * (o.S + 1) / 2 : o.L);
    return {
      ops: 2 * M * d * (2 * d + 2 * dkv) + 4 * keys * d + SOFTMAX_OPS * o.h * keys + (gated ? 6 : 4) * M * d * f + M * f * ACT_OPS[o.act] + 2 * M * d + (nf ? 3 * M * d * nf : 0),
      params: o.vocab * d + 2 * d * d + 2 * d * dkv + (gated ? 3 : 2) * d * f + (nf ? 3 * (o.norm === 'layernorm' ? 2 : 1) * d : 0)
    };
  }
  function DecodeCost(root, cfg) {
    const c = ENC.CHIPS[cfg.chip], dkv = (cfg.d / cfg.h) * cfg.kvh;
    ENC.CostView.call(this, root, Object.assign({}, cfg, {
      rowsFn: priced, closedFn: closed,
      headFn: (o) => (o.phase === 'prefill' ? (o.batch > 1 ? o.batch + ' × ' : '') + 'prefill of ' + o.S + ' tokens' : (o.batch > 1 ? 'batch ' + o.batch + ', ' : '') + 'one new token, context ' + o.L) + ' · ' + o.kvh + ' KV head' + (o.kvh > 1 ? 's' : ''),
      capFn: (rows, total) => {
        const kv = 2 * cfg.L * dkv * c.bytes * cfg.layers * cfg.batch, perTok = total * cfg.layers;
        return cfg.phase === 'prefill'
          ? 'All ' + cfg.layers + ' layers: <b>' + fmtTime(perTok) + '</b> for the prompt — the time to the first token.'
          : 'All ' + cfg.layers + ' layers: <b>' + fmtTime(perTok) + '</b> per token → <b>' + Math.round(cfg.batch / perTok).toLocaleString('en') + ' tokens/s</b>' + (cfg.batch > 1 ? ' over the batch' : '') + '. KV cache at this context: <b>' + fmtN(kv) + 'B</b> (' + cfg.layers + ' layers' + (cfg.batch > 1 ? ' × ' + cfg.batch + ' sequences' : '') + ').';
      }
    }));
  }
  DecodeCost.prototype = ENC.CostView.prototype;

  // ---------------- deck mount ----------------
  const DEFAULTS = { Tp: 3, G: 3, d: 4, h: 2, kvh: 2, ffn: 8, act: 'swiglu', norm: 'rmsnorm', vocab: 16, seed: 0, cache: true,
    S: 512, L: 1024, batch: 1, phase: 'decode', layers: 32, chip: 'a100' };
  const LAB = { cache: 'KV cache', L: 'context', batch: 'batch', kvh: 'KV heads', phase: 'phase', chip: 'chip', G: 'generated' };
  const build = (cfg) => (cfg.type === 'gen' ? problem(cfg) : null);
  const { mount, mountAll } = global.NB.widgets({ gen: GenerateView, cost: DecodeCost }, DEFAULTS, build, LAB);

  global.DEC = { problem, reference, close, rowOps, layerRows, priced, closed, GenerateView, DecodeCost, mount, mountAll, DEFAULTS };
  global.MM = { mountAll };
})(typeof window !== 'undefined' ? window : globalThis);
