/* One transformer encoder layer: the operations it runs, in order, on a tiny example you can check by
   hand — and what each one costs at a real model's size on a real chip.

   Reuses the matmul notebook's grid, controls and counters (MM2._), the attention notebook's softmax
   reference (ATT.refAttention) and the shared deck widgets (NB.widgets). The operation list and every
   count follow bwz's encoder layer: q/k/v/o projections 2·S·d², attention 4·S²·d + 5·h·S², norms 4
   (rmsnorm) or 6 (layernorm) per element, activations relu 1, gelu 8, swiglu 5, geglu 9 per element,
   residual adds 1 — so the default shape (S = 4, d = 8, 2 heads, ffn = 16) is 664 parameters and
   5280 operations, the same as `bwz encoder-layer`'s defaults. */
(function (global) {
  'use strict';
  const X = global.MM2, H = X._, ATT = global.ATT;
  const { el, addCls, counters, Mat, mix, key, rng } = H;
  // Two significant decimals at most, and no "−.00": a value that rounds to zero prints as 0.
  function fmt(x) {
    if (Number.isInteger(x)) return String(x);
    const a = Math.abs(x);
    if (a < 0.005) return '0';
    if (a >= 9.95) return x.toFixed(0);
    if (a >= 0.995) return x.toFixed(1);
    return x.toFixed(2).replace(/^(-?)0\./, '$1.');
  }

  // ---------------- conventions (bwz/calibration.py) ----------------
  const NORM_OPS = { rmsnorm: 4, layernorm: 6, none: 0 };
  const ACT_OPS = { relu: 1, gelu: 8, swiglu: 5, geglu: 9 };
  const GATED = { relu: false, gelu: false, swiglu: true, geglu: true };
  const SOFTMAX_OPS = 5;
  const EPS = 1e-5;

  // ---------------- the problem ----------------
  function lcg(seed) { let s = (seed * 2654435761) >>> 0 || 1; return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296); }
  function problem(o) {
    const S = o.S, d = o.d, h = o.h, ffn = o.ffn, vocab = o.vocab || 16;
    const r = lcg(o.seed || 5);
    const mat = (rows, cols, lo, span) => rng(rows).map(() => rng(cols).map(() => Math.floor(r() * span) + lo));
    const table = mat(vocab, d, -1, 4);                   // embeddings: −1 … 2
    const ids = rng(S).map(() => Math.floor(r() * vocab));
    const W = { q: mat(d, d, -1, 3), k: mat(d, d, -1, 3), v: mat(d, d, -1, 3), o: mat(d, d, -1, 3),
      up: mat(d, ffn, -1, 3), gate: GATED[o.act] ? mat(d, ffn, -1, 3) : null, down: mat(ffn, d, -1, 3) };   // weights: −1 … 1
    const P = { S, d, h, dh: d / h, ffn, vocab, act: o.act, norm: o.norm, table, ids, W, seed: o.seed || 5 };
    P.steps = forward(P);
    P.REF = reference(P);
    return P;
  }

  // ---------------- the forward pass, op by op ----------------
  const matmul = (A, B) => A.map((row) => B[0].map((_, j) => row.reduce((s, x, k) => s + x * B[k][j], 0)));
  const map2 = (A, B, f) => A.map((row, i) => row.map((x, j) => f(x, B[i][j])));
  const gelu = (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
  const silu = (x) => x / (1 + Math.exp(-x));
  function normRows(Xm, kind) {
    if (kind === 'none') return Xm.map((r) => r.slice());
    return Xm.map((row) => {
      const n = row.length;
      if (kind === 'rmsnorm') { const rms = Math.sqrt(row.reduce((s, x) => s + x * x, 0) / n + EPS); return row.map((x) => x / rms); }
      const mean = row.reduce((s, x) => s + x, 0) / n, v = row.reduce((s, x) => s + (x - mean) * (x - mean), 0) / n;
      return row.map((x) => (x - mean) / Math.sqrt(v + EPS));
    });
  }
  function attention(P, Q, K, V) {
    const out = Q.map(() => rng(P.d).map(() => 0));
    for (let hh = 0; hh < P.h; hh++) {
      const cols = rng(P.dh).map((c) => hh * P.dh + c);
      const pick = (M) => M.map((r) => cols.map((c) => r[c]));
      const Oh = ATT.refAttention(pick(Q), pick(K), pick(V), 1 / Math.sqrt(P.dh));
      Oh.forEach((r, i) => r.forEach((x, c) => { out[i][cols[c]] = x; }));
    }
    return out;
  }
  // Each step: what it reads, what it writes, its weights, and its cost by bwz's conventions.
  function forward(P) {
    const { S, d, ffn, h, act, norm } = P, gated = GATED[act];
    const T = {};
    const steps = [];
    const add = (s) => { steps.push(s); return s; };
    T.x0 = P.ids.map((t) => P.table[t].slice());
    add({ id: 'embed', group: 'in', title: 'Embedding', what: 'each token id picks its row of the table', ins: ['ids'], out: 'x0', w: 'table', shape: S + ' rows of ' + P.vocab + '×' + d, ops: 0, params: P.vocab * d, kind: 'gather' });
    const nf = NORM_OPS[norm];
    if (norm !== 'none') { T.n1 = normRows(T.x0, norm); add({ id: 'attn_norm', group: 'attn', title: norm === 'rmsnorm' ? 'RMSNorm' : 'LayerNorm', what: 'each row scaled to unit size', ins: ['x0'], out: 'n1', shape: S + '×' + d, ops: S * d * nf, params: (norm === 'layernorm' ? 2 : 1) * d, kind: 'norm' }); } else T.n1 = T.x0;
    ['q', 'k', 'v'].forEach((m) => { T[m] = matmul(T.n1, P.W[m]); add({ id: m + '_proj', group: 'attn', title: m.toUpperCase() + ' projection', what: m.toUpperCase() + ' = x · W' + m, ins: [norm !== 'none' ? 'n1' : 'x0'], w: m, out: m, shape: '[' + S + ',' + d + ']×[' + d + ',' + d + ']', ops: 2 * S * d * d, params: d * d, kind: 'matmul', mnk: [S, d, d] }); });
    T.a = attention(P, T.q, T.k, T.v);
    add({ id: 'attn', group: 'attn', title: 'Attention', what: h + ' head' + (h > 1 ? 's' : '') + ' of width ' + P.dh + ': softmax(Q Kᵀ/√' + P.dh + ') V each', ins: ['q', 'k', 'v'], out: 'a', shape: h + ' × [' + S + '×' + S + ']', ops: 4 * S * S * d + SOFTMAX_OPS * h * S * S, params: 0, kind: 'attention' });
    T.o = matmul(T.a, P.W.o);
    add({ id: 'o_proj', group: 'attn', title: 'Output projection', what: 'mixes the heads back together', ins: ['a'], w: 'o', out: 'o', shape: '[' + S + ',' + d + ']×[' + d + ',' + d + ']', ops: 2 * S * d * d, params: d * d, kind: 'matmul', mnk: [S, d, d] });
    T.x1 = map2(T.x0, T.o, (a, b) => a + b);
    add({ id: 'attn_residual', group: 'attn', title: 'Residual add', what: 'x + attention: the layer only adds a correction', ins: ['x0', 'o'], out: 'x1', shape: S + '×' + d, ops: S * d, params: 0, kind: 'add' });
    if (norm !== 'none') { T.n2 = normRows(T.x1, norm); add({ id: 'ffn_norm', group: 'ffn', title: norm === 'rmsnorm' ? 'RMSNorm' : 'LayerNorm', what: 'normalise again before the FFN', ins: ['x1'], out: 'n2', shape: S + '×' + d, ops: S * d * nf, params: (norm === 'layernorm' ? 2 : 1) * d, kind: 'norm' }); } else T.n2 = T.x1;
    const nIn = norm !== 'none' ? 'n2' : 'x1';
    T.u = matmul(T.n2, P.W.up);
    add({ id: 'ffn_up', group: 'ffn', title: 'FFN up', what: 'widen every token from d = ' + d + ' to ffn = ' + ffn, ins: [nIn], w: 'up', out: 'u', shape: '[' + S + ',' + d + ']×[' + d + ',' + ffn + ']', ops: 2 * S * d * ffn, params: d * ffn, kind: 'matmul', mnk: [S, ffn, d] });
    if (gated) { T.g = matmul(T.n2, P.W.gate); add({ id: 'ffn_gate', group: 'ffn', title: 'FFN gate', what: 'a second widening, the gate', ins: [nIn], w: 'gate', out: 'g', shape: '[' + S + ',' + d + ']×[' + d + ',' + ffn + ']', ops: 2 * S * d * ffn, params: d * ffn, kind: 'matmul', mnk: [S, ffn, d] }); }
    T.f = gated ? map2(T.g, T.u, (g, u) => (act === 'swiglu' ? silu(g) : gelu(g)) * u) : T.u.map((r) => r.map((x) => (act === 'relu' ? Math.max(0, x) : gelu(x))));
    add({ id: 'ffn_act', group: 'ffn', title: { relu: 'ReLU', gelu: 'GELU', swiglu: 'SwiGLU', geglu: 'GeGLU' }[act], what: { relu: 'max(0, x), element by element', gelu: 'a smooth ReLU, element by element', swiglu: 'silu(gate) · up, element by element', geglu: 'gelu(gate) · up, element by element' }[act], ins: gated ? ['g', 'u'] : ['u'], out: 'f', shape: S + '×' + ffn, ops: S * ffn * ACT_OPS[act], params: 0, kind: 'act' });
    T.y = matmul(T.f, P.W.down);
    add({ id: 'ffn_down', group: 'ffn', title: 'FFN down', what: 'back from ffn = ' + ffn + ' to d = ' + d, ins: ['f'], w: 'down', out: 'y', shape: '[' + S + ',' + ffn + ']×[' + ffn + ',' + d + ']', ops: 2 * S * ffn * d, params: ffn * d, kind: 'matmul', mnk: [S, d, ffn] });
    T.x2 = map2(T.x1, T.y, (a, b) => a + b);
    add({ id: 'ffn_residual', group: 'ffn', title: 'Residual add', what: 'x + FFN', ins: ['x1', 'y'], out: 'x2', shape: S + '×' + d, ops: S * d, params: 0, kind: 'add' });
    if (norm !== 'none') { T.out = normRows(T.x2, norm); add({ id: 'final_norm', group: 'out', title: 'Final norm', what: 'the stack\'s last norm, once after the last layer', ins: ['x2'], out: 'out', shape: S + '×' + d, ops: S * d * nf, params: (norm === 'layernorm' ? 2 : 1) * d, kind: 'norm' }); } else T.out = T.x2;
    P.T = T;
    return steps;
  }

  // The same layer, written the other way round: token by token, as a textbook would. It shares
  // nothing with forward() — its own norm, its own head split, its own projections — except the
  // attention notebook's softmax reference for each head. The walk above must land on exactly this.
  function reference(P) {
    const { S, d, dh, h, ffn, act, norm } = P, gated = GATED[act];
    const nrm = (row) => {
      if (norm === 'none') return row.slice();
      let mean = 0, sq = 0;
      for (const x of row) { mean += x / d; sq += (x * x) / d; }
      return norm === 'rmsnorm' ? row.map((x) => x / Math.sqrt(sq + EPS)) : row.map((x) => (x - mean) / Math.sqrt(sq - mean * mean + EPS));
    };
    const vecmat = (vec, W) => { const out = new Array(W[0].length).fill(0); vec.forEach((x, k) => W[k].forEach((w, j) => { out[j] += x * w; })); return out; };
    const x0 = P.ids.map((t) => P.table[t]);
    const n1 = x0.map(nrm);
    const q = n1.map((r) => vecmat(r, P.W.q)), k = n1.map((r) => vecmat(r, P.W.k)), v = n1.map((r) => vecmat(r, P.W.v));
    const heads = rng(h).map((hh) => ATT.refAttention(q.map((r) => r.slice(hh * dh, hh * dh + dh)), k.map((r) => r.slice(hh * dh, hh * dh + dh)), v.map((r) => r.slice(hh * dh, hh * dh + dh)), 1 / Math.sqrt(dh)));
    return rng(S).map((i) => {
      const a = [].concat(...heads.map((Oh) => Oh[i]));
      const o = vecmat(a, P.W.o), x1 = x0[i].map((x, c) => x + o[c]);
      const n2 = nrm(x1), up = vecmat(n2, P.W.up), gt = gated ? vecmat(n2, P.W.gate) : null;
      const f = rng(ffn).map((j) => (gated ? (act === 'swiglu' ? silu(gt[j]) : gelu(gt[j])) * up[j] : act === 'relu' ? Math.max(0, up[j]) : gelu(up[j])));
      const y = vecmat(f, P.W.down);
      return nrm(x1.map((x, c) => x + y[c]));
    });
  }
  const close = (A, B) => A.every((r, i) => r.every((x, j) => Math.abs(x - B[i][j]) <= 1e-9 * Math.max(1, Math.abs(B[i][j]))));

  // ---------------- closed forms (no walk): what a layer is at any size ----------------
  function layerOps(o) {
    const B = o.batch || 1, M = B * o.S, d = o.d, f = o.ffn, gated = GATED[o.act], nf = NORM_OPS[o.norm];
    const L = [];
    const op = (id, title, group, ops, params, actIn, actOut, kind) => L.push({ id, title, group, ops, params, actIn, actOut, kind });
    op('embed', 'Embedding', 'in', 0, o.vocab * d, 0, M * d, 'gather');
    if (nf) op('attn_norm', 'Norm', 'attn', M * d * nf, (o.norm === 'layernorm' ? 2 : 1) * d, M * d, M * d, 'norm');
    ['q', 'k', 'v'].forEach((m) => op(m + '_proj', m.toUpperCase() + ' projection', 'attn', 2 * M * d * d, d * d, M * d, M * d, 'matmul'));
    op('attn', 'Attention', 'attn', B * (4 * o.S * o.S * d + SOFTMAX_OPS * o.h * o.S * o.S), 0, 3 * M * d, M * d, 'attention');
    op('o_proj', 'Output projection', 'attn', 2 * M * d * d, d * d, M * d, M * d, 'matmul');
    op('attn_residual', 'Residual add', 'attn', M * d, 0, 2 * M * d, M * d, 'add');
    if (nf) op('ffn_norm', 'Norm', 'ffn', M * d * nf, (o.norm === 'layernorm' ? 2 : 1) * d, M * d, M * d, 'norm');
    op('ffn_up', 'FFN up', 'ffn', 2 * M * d * f, d * f, M * d, M * f, 'matmul');
    if (gated) op('ffn_gate', 'FFN gate', 'ffn', 2 * M * d * f, d * f, M * d, M * f, 'matmul');
    op('ffn_act', { relu: 'ReLU', gelu: 'GELU', swiglu: 'SwiGLU', geglu: 'GeGLU' }[o.act], 'ffn', M * f * ACT_OPS[o.act], 0, (gated ? 2 : 1) * M * f, M * f, 'act');
    op('ffn_down', 'FFN down', 'ffn', 2 * M * f * d, f * d, M * f, M * d, 'matmul');
    op('ffn_residual', 'Residual add', 'ffn', M * d, 0, 2 * M * d, M * d, 'add');
    if (nf) op('final_norm', 'Final norm', 'out', M * d * nf, (o.norm === 'layernorm' ? 2 : 1) * d, M * d, M * d, 'norm');
    return L;
  }
  // The totals by formula alone — the check every cost view ends on.
  function closedForm(o) {
    const B = o.batch || 1, M = B * o.S, d = o.d, f = o.ffn, gated = GATED[o.act], nf = NORM_OPS[o.norm], nn = nf ? 3 : 0;
    return {
      ops: 8 * M * d * d + B * (4 * o.S * o.S * d + SOFTMAX_OPS * o.h * o.S * o.S) + (gated ? 6 : 4) * M * d * f + 2 * M * d + M * f * ACT_OPS[o.act] + nn * M * d * nf,
      params: o.vocab * d + 4 * d * d + (gated ? 3 : 2) * d * f + nn * (o.norm === 'layernorm' ? 2 : 1) * d
    };
  }

  // Datasheet ceilings, from the chip profiles (profiles/chips/*.yaml), at each chip's own precision.
  const CHIPS = {
    a100: { name: 'A100 · fp16', peak: 312e12, bw: 2.039e12, bytes: 2 },
    h100: { name: 'H100 · fp16', peak: 989.4e12, bw: 3.35e12, bytes: 2 },
    chip_a: { name: 'edge NPU (chip_a) · int8', peak: 209.7e12, bw: 34e9, bytes: 1 }
  };
  function fmtTime(s) {
    if (s >= 1) return s.toFixed(2) + ' s';
    if (s >= 1e-3) return (s * 1e3).toPrecision(3) + ' ms';
    if (s >= 1e-6) return (s * 1e6).toPrecision(3) + ' µs';
    return (s * 1e9).toPrecision(3) + ' ns';
  }
  function fmtN(n) {
    const u = [[1e12, 'T'], [1e9, 'G'], [1e6, 'M'], [1e3, 'k']];
    for (const [v, s] of u) if (Math.abs(n) >= v) return (n / v).toPrecision(3) + s;
    return String(Math.round(n));
  }

  // =====================================================================
  // LayerView: the tiny layer, one operation per step
  // =====================================================================
  const NAMES = { ids: 'token ids', x0: 'x', n1: 'norm(x)', q: 'Q', k: 'K', v: 'V', a: 'attention out', o: 'O', x1: 'x (after attention)', n2: 'norm(x)', u: 'up', g: 'gate', f: 'act(up)', y: 'FFN out', x2: 'x (after FFN)', out: 'layer out' };
  const WNAMES = { table: 'embedding table', q: 'Wq', k: 'Wk', v: 'Wv', o: 'Wo', up: 'W_up', gate: 'W_gate', down: 'W_down' };
  function LayerView(root, cfg) {
    this.cfg = cfg; this.root = root; this.P = cfg.P; root.classList.add('w', 'pv', 'enc');
    this.baseSpeed = 1100; this.t = 0; this.n = this.P.steps.length;
    const P = this.P;
    this.totalOps = P.steps.reduce((s, x) => s + x.ops, 0); this.totalParams = P.steps.reduce((s, x) => s + x.params, 0);
    if (cfg.title !== false) {
      const head = el('div', 'w-head'); root.appendChild(head);
      head.innerHTML = '<span class="chip">S = ' + P.S + ' tokens × d = ' + P.d + '</span><span class="chip">' + P.h + ' head' + (P.h > 1 ? 's' : '') + ' of ' + P.dh + '</span><span class="chip">ffn = ' + P.ffn + ' · ' + P.act + '</span><span class="chip">' + P.norm + '</span>';
    }
    const top = el('div', 'enc-top'); root.appendChild(top);
    this.strip = el('ol', 'enc-strip'); top.appendChild(this.strip);
    this.items = P.steps.map((s) => { const li = el('li', 'enc-op g-' + s.group, '<b>' + s.title + '</b><span>' + s.shape + '</span><i>' + fmtN(s.ops) + ' ops</i>'); this.strip.appendChild(li); return li; });
    this.stage = el('div', 'enc-stage'); top.appendChild(this.stage);
    this.cap = el('div', 'cap'); root.appendChild(this.cap);
    const foot = el('div', 'w-foot'); root.appendChild(foot);
    this.cnt = el('div', 'counters'); foot.appendChild(this.cnt);
    const right = el('div', 'stack right'); right.append(this.controls('Next block'), this.scrubber()); foot.appendChild(right);
    this.keys(); this.go(0);
  }
  LayerView.prototype.inner = function () {
    // To the end of the current block: after the attention residual, or the end of the layer.
    const ids = this.P.steps.map((s) => s.id);
    const stop = ids.indexOf('attn_residual') + 1;
    this.go(this.t < stop ? stop : this.n); return true;
  };
  function matOf(kind, M, title, opt) {
    const rows = M.length, cols = M[0].length;
    const m = new Mat(kind, rows, cols, Object.assign({ rv: 't', cv: 'c', title, dims: rows + '×' + cols }, opt || {}));
    m.paint({}, (r, c) => fmt(M[r][c]));
    return m;
  }
  LayerView.prototype.render = function () {
    const P = this.P, T = P.T, cur = this.t > 0 ? P.steps[this.t - 1] : null;
    this.items.forEach((li, i) => { li.classList.toggle('done', i < this.t - 1); li.classList.toggle('cur', i === this.t - 1); });
    this.stage.innerHTML = '';
    if (!cur) {
      this.stage.appendChild(el('p', 'muted', 'The layer is the list on the left, top to bottom. Each step runs one operation on all ' + P.S + ' tokens at once: S × d numbers in, S × d (or S × ffn) out.'));
    } else {
      this.stage.appendChild(el('div', 'enc-what', '<b>' + cur.title + '</b> — ' + cur.what));
      const row = el('div', 'enc-row'); this.stage.appendChild(row);
      const heads = (m) => (P.h > 1 && m[0].length === P.d ? { cb: P.dh } : {});
      if (cur.id === 'embed') {
        const ids = new Mat('A', P.S, 1, { rv: 't', cv: ' ', title: 'token ids', dims: P.S + ' tokens' });
        ids.paint({}, (r) => String(P.ids[r])); row.appendChild(ids.root);
        row.appendChild(el('span', 'arrow', '→'));
      } else {
        cur.ins.forEach((name, i) => {
          const m = matOf('A', T[name], NAMES[name], heads(T[name])); row.appendChild(m.root);
          if (i < cur.ins.length - 1) row.appendChild(el('span', 'arrow', cur.kind === 'add' ? '+' : ','));
        });
        if (cur.w) { row.appendChild(el('span', 'arrow', '×')); row.appendChild(matOf('B', P.W[cur.w], WNAMES[cur.w]).root); }
        row.appendChild(el('span', 'arrow', '→'));
      }
      const out = matOf('C', T[cur.out], NAMES[cur.out], heads(T[cur.out]));
      out.paint(Object.fromEntries(T[cur.out].flatMap((r, i) => r.map((_, j) => [key(i, j), 'fin']))), (r, c) => fmt(T[cur.out][r][c]));
      row.appendChild(out.root);
      if (cur.kind === 'attention') this.stage.appendChild(el('p', 'small', 'Each head is the attention notebook\'s computation on its own ' + P.dh + ' columns of Q, K and V (the block lines). Here S is small enough to hold whole; at real sizes it runs as FlashAttention.'));
    }
    const doneOps = P.steps.slice(0, this.t).reduce((s, x) => s + x.ops, 0), doneParams = P.steps.slice(0, this.t).reduce((s, x) => s + x.params, 0);
    let cap = cur ? '<b>Step ' + this.t + '/' + this.n + '</b> · ' + cur.title + ': ' + fmtN(cur.ops) + ' operations' + (cur.params ? ', ' + cur.params + ' parameters' : ', no parameters') + '.' : 'Press <b>Step</b> to run the layer one operation at a time.';
    if (this.t === this.n) {
      const ok = close(T.out, P.REF);
      cap = 'Done: ' + this.n + ' operations. ' + (ok ? '<span class="ok">Check: the walk equals the layer computed token by token ✓</span>' : '<span class="tsp">differs ✗</span>') + ' ' + this.totalParams + ' parameters, ' + this.totalOps + ' operations' + (closedForm(P).ops === this.totalOps && closedForm(P).params === this.totalParams ? ', matching the closed form' : '') + '.';
    }
    this.cap.innerHTML = cap;
    this.cnt.innerHTML = counters([
      { label: 'step', value: this.t + '/' + this.n }, { label: 'operations', value: doneOps + '/' + this.totalOps },
      { label: 'parameters used', value: doneParams + '/' + this.totalParams }
    ]);
  };
  LayerView.prototype.trace = function () {
    return this.P.steps.slice(0, this.t).map((s, i) => 'step ' + String(i + 1).padStart(2) + ': ' + s.id.padEnd(14) + s.shape.padEnd(22) + String(s.ops).padStart(7) + ' ops' + (s.params ? '  ' + s.params + ' params' : ''));
  };
  mix(LayerView.prototype);

  // =====================================================================
  // CostView: the same operations at any size, priced on a chip
  // Each op: t = max(operations / peak, bytes / bandwidth) — the roofline, ideal, one op after another.
  // =====================================================================
  function costOf(o, chip) {
    const c = CHIPS[chip], L = layerOps(o);
    return L.map((x) => {
      const bytes = (x.id === 'embed' ? x.actOut : x.params + x.actIn + x.actOut) * c.bytes;
      const tc = x.ops / c.peak, tm = bytes / c.bw;
      return Object.assign({}, x, { bytes, intensity: bytes ? x.ops / bytes : 0, tc, tm, t: Math.max(tc, tm), bound: tc > tm ? 'compute' : 'memory' });
    });
  }
  function CostView(root, cfg) {
    this.cfg = cfg; this.root = root; root.classList.add('w', 'enc');
    this.baseSpeed = 700; this.t = 0;
    const chip = CHIPS[cfg.chip];
    this.rows = costOf(cfg, cfg.chip); this.n = this.rows.length;
    this.total = this.rows.reduce((s, r) => s + r.t, 0);
    if (cfg.title !== false) {
      const head = el('div', 'w-head'); root.appendChild(head);
      head.innerHTML = '<span class="chip">' + (cfg.batch > 1 ? cfg.batch + ' × ' : '') + 'S = ' + cfg.S + ' tokens · d = ' + cfg.d + ' · ' + cfg.h + ' heads · ffn = ' + cfg.ffn + '</span><span class="chip">' + chip.name + ': ' + fmtN(chip.peak) + 'OP/s, ' + fmtN(chip.bw) + 'B/s → ridge ' + Math.round(chip.peak / chip.bw) + ' OP/byte</span>';
    }
    this.tableWrap = el('div', 'tw'); root.appendChild(this.tableWrap);
    this.bar = el('div', 'enc-bar'); root.appendChild(this.bar);
    this.cap = el('div', 'cap'); root.appendChild(this.cap);
    const foot = el('div', 'w-foot'); root.appendChild(foot);
    this.cnt = el('div', 'counters'); foot.appendChild(this.cnt);
    const right = el('div', 'stack right'); right.append(this.controls('All'), this.scrubber()); foot.appendChild(right);
    this.keys(); this.go(0);
  }
  CostView.prototype.inner = function () { this.go(this.n); return true; };
  CostView.prototype.render = function () {
    const cfg = this.cfg, chip = CHIPS[cfg.chip], ridge = chip.peak / chip.bw;
    const shown = this.rows.slice(0, this.t);
    let h = '<table class="cmp enc-cost"><tr><th>operation</th><th>operations</th><th>parameters</th><th>bytes moved</th><th>OP / byte</th><th>bound on ' + chip.name.split(' ·')[0] + '</th><th>time</th><th>share</th></tr>';
    this.rows.forEach((r, i) => {
      const on = i < this.t;
      h += '<tr class="' + (on ? 'g-' + r.group : 'hid') + (i === this.t - 1 ? ' cur' : '') + '"><td>' + r.title + '</td><td>' + (on ? fmtN(r.ops) : '') + '</td><td>' + (on ? fmtN(r.params) + (r.id === 'embed' ? ' <span class="muted">(the stack\'s)</span>' : '') : '') + '</td><td>' + (on ? fmtN(r.bytes) + 'B' : '') + '</td><td>' + (on ? (r.intensity ? r.intensity.toPrecision(3) : '—') : '') + '</td><td>' + (on ? '<span class="b-' + r.bound + '">' + r.bound + '</span>' : '') + '</td><td>' + (on ? fmtTime(r.t) : '') + '</td><td>' + (on ? '<span class="sh" style="width:' + Math.max(1, Math.round(100 * r.t / this.total)) + 'px"></span>' + Math.round(100 * r.t / this.total) + '%' : '') + '</td></tr>';
    });
    this.tableWrap.innerHTML = h + '</table>';
    // Widths as shares of the layer's time: flex-grow on raw seconds (~1e-6) would sum below 1,
    // and a flex line only hands out that fraction of its width.
    this.bar.innerHTML = shown.map((r) => '<span class="seg-' + r.group + ' b-' + r.bound + '" style="width:' + (100 * r.t / this.total) + '%" title="' + r.title + ': ' + fmtTime(r.t) + '"></span>').join('');
    const sum = (f) => shown.reduce((s, r) => s + r[f], 0);
    const mm = shown.filter((r) => r.kind === 'matmul'), at = shown.filter((r) => r.kind === 'attention'), ew = shown.filter((r) => ['norm', 'add', 'act', 'gather'].includes(r.kind));
    let cap = this.t === 0 ? 'Press <b>Step</b> to price the layer one operation at a time: <b>t = max(operations / peak, bytes / bandwidth)</b> — an operation whose OP/byte is above the ridge point (' + Math.round(ridge) + ') is compute-bound, below it memory-bound.' : '<b>' + this.rows[this.t - 1].title + '</b>: ' + fmtN(this.rows[this.t - 1].ops) + ' operations over ' + fmtN(this.rows[this.t - 1].bytes) + 'B — ' + this.rows[this.t - 1].bound + '-bound.';
    if (this.t === this.n) {
      const cf = closedForm(cfg), ok = sum('ops') === cf.ops && sum('params') === cf.params;
      const share = (L) => Math.round(100 * L.reduce((s, r) => s + r.t, 0) / this.total);
      const attnW = this.rows.filter((r) => r.group === 'attn').reduce((s, r) => s + r.params, 0), ffnW = this.rows.filter((r) => r.group === 'ffn').reduce((s, r) => s + r.params, 0);
      cap = 'Layer: <b>' + fmtTime(this.total) + '</b> — matmuls ' + share(mm) + '%, attention ' + share(at) + '%, norms, activations and residuals ' + share(ew) + '%. Its own weights (the embedding table belongs to the whole stack): attention ' + Math.round(100 * attnW / (attnW + ffnW)) + '%, FFN ' + Math.round(100 * ffnW / (attnW + ffnW)) + '%. ' + (ok ? '<span class="ok">Check: operations and parameters equal the closed form ✓</span>' : '<span class="tsp">✗</span>');
    }
    this.cap.innerHTML = cap;
    this.cnt.innerHTML = counters([
      { label: 'operations', value: fmtN(sum('ops')) }, { label: 'parameters', value: fmtN(sum('params')) },
      { label: 'bytes', value: fmtN(sum('bytes')) + 'B' }, { label: 'time', value: fmtTime(sum('t')) }
    ]);
  };
  CostView.prototype.trace = function () {
    return this.rows.slice(0, this.t).map((r) => r.id.padEnd(14) + fmtN(r.ops).padStart(8) + ' ops ' + (fmtN(r.bytes) + 'B').padStart(9) + '  ' + r.bound.padEnd(8) + fmtTime(r.t).padStart(10));
  };
  mix(CostView.prototype);

  // ---------------- deck mount ----------------
  const DEFAULTS = { S: 4, d: 8, h: 2, ffn: 16, act: 'relu', norm: 'rmsnorm', vocab: 16, batch: 1, seed: 0, chip: 'a100' };
  const LAB = { S: 'tokens S', d: 'width d', act: 'activation', norm: 'norm', chip: 'chip', batch: 'batch', ffn: 'ffn', h: 'heads' };
  // A cost view needs no numbers, only the sizes; a layer view needs the tiny problem.
  const build = (cfg) => (cfg.type === 'cost' ? null : problem(cfg));
  const { mount, mountAll } = global.NB.widgets({ layer: LayerView, cost: CostView }, DEFAULTS, build, LAB);

  global.ENC = { problem, forward, reference, close, layerOps, closedForm, costOf, CHIPS, NORM_OPS, ACT_OPS, LayerView, CostView, mount, mountAll, DEFAULTS };
  global.MM = { mountAll };
})(typeof window !== 'undefined' ? window : globalThis);
