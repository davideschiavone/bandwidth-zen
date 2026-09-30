/* General matmul simulator: any M, N, K (up to 8), any tiling, any loop order, any number of cores.
   Used by the lab page and by the multi-core slides of the deck. Every view replays from scratch,
   so the counters are always the real ones. */
(function (global) {
  'use strict';
  const rng = (n) => Array.from({ length: n }, (_, i) => i);
  const zeros = (r, c) => rng(r).map(() => rng(c).map(() => 0));
  const EXA = [[1, 2, 0, 3], [0, 1, 2, 1], [3, 0, 1, 2], [2, 1, 3, 0]];
  const EXB = [[2, 0, 1, 3], [1, 3, 0, 2], [0, 1, 2, 1], [3, 2, 1, 0]];
  const STAY = { k: 'C', i: 'B', j: 'A' };
  const SHORT = { C: 'os', B: 'ws', A: 'is' };
  const STATN = { C: 'output-stationary', B: 'weight-stationary', A: 'input-stationary' };
  const FORM = { ijk: 'inner product', jik: 'inner product', ikj: 'row-wise (Gustavson)', jki: 'column-wise', kij: 'outer product', kji: 'outer product' };
  const BV = { i: 'mt', j: 'nt', k: 'kt' };
  const DIM = { i: 'M', j: 'N', k: 'K' };
  const IDX = { A: ['i', 'k'], B: ['k', 'j'], C: ['i', 'j'] };
  const ORDERS = ['ijk', 'jik', 'ikj', 'kij', 'jki', 'kji'];
  const key = (i, j) => i + ',' + j;

  function lcg(seed) { let s = (seed * 2654435761) >>> 0 || 1; return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296); }
  function problem(M, N, K, seed) {
    let A, B;
    if (!seed && M === 4 && N === 4 && K === 4) { A = EXA.map((r) => r.slice()); B = EXB.map((r) => r.slice()); }
    else {
      const r = lcg(seed || 7);
      A = rng(M).map(() => rng(K).map(() => Math.floor(r() * 4)));
      B = rng(K).map(() => rng(N).map(() => Math.floor(r() * 4)));
    }
    const REF = rng(M).map((i) => rng(N).map((j) => rng(K).reduce((s, k) => s + A[i][k] * B[k][j], 0)));
    return { M, N, K, A, B, REF, seed: seed || 0 };
  }
  const divisors = (n) => rng(n).map((x) => x + 1).filter((d) => n % d === 0);
  function newAcc(P) { return { C: zeros(P.M, P.N), T: zeros(P.M, P.N), mul: 0, add: 0, part: 0, maxPart: 0 }; }
  function apply(P, s, i, j, k) {
    const a = P.A[i][k], b = P.B[k][j], p = a * b, old = s.C[i][j], first = s.T[i][j] === 0;
    if (!first) s.add++;
    s.C[i][j] = old + p; s.T[i][j]++; s.mul++;
    if (first && P.K > 1) s.part++;
    if (s.T[i][j] === P.K && P.K > 1) s.part--;
    if (s.part > s.maxPart) s.maxPart = s.part;
    return { i, j, k, a, b, p, old, nw: old + p, terms: s.T[i][j], first };
  }
  function eqRef(P, C) { return C.every((r, i) => r.every((x, j) => x === P.REF[i][j])); }

  // ---------------- DOM helpers ----------------
  function el(tag, cls, html) { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
  function btn(label, title, fn, pri) { const b = el('button', 'btn' + (pri ? ' pri' : ''), label); b.type = 'button'; b.title = title; b.setAttribute('aria-label', title); b.addEventListener('click', fn); return b; }
  function addCls(map, k0, c) { map[k0] = map[k0] ? map[k0] + ' ' + c : c; }
  function counters(items) { return items.map((x) => '<span class="' + (x.cls || '') + '">' + x.label + ' <b>' + x.value + '</b></span>').join(''); }

  function Mat(kind, rows, cols, opt) {
    opt = opt || {};
    this.rows = rows; this.cols = cols; this.rb = opt.rb || rows; this.cb = opt.cb || cols;
    this.root = el('div', 'mat m-' + kind + (opt.cls ? ' ' + opt.cls : ''));
    const rv = opt.rv || IDX[kind][0], cv = opt.cv || IDX[kind][1];
    this.root.appendChild(el('div', 'mat-title', '<b>' + (opt.title || kind) + '</b><span>' + (opt.dims || rows + '×' + cols) + '</span>'));
    const g = el('div', 'mat-grid'); g.style.gridTemplateColumns = 'auto repeat(' + cols + ', auto)'; this.grid = g;
    const put = (e, r, c) => { e.style.gridRow = String(r); e.style.gridColumn = String(c); g.appendChild(e); return e; };
    put(el('div', 'mh corner', rv + '\\' + cv), 1, 1);
    this.colH = []; this.rowH = []; this.cells = [];
    const bl = (c) => (this.cb < this.cols && c % this.cb === 0 && c > 0 ? ' bl' : '');
    const bt = (r) => (this.rb < this.rows && r % this.rb === 0 && r > 0 ? ' bt' : '');
    for (let c = 0; c < cols; c++) { const h = el('div', 'mh' + bl(c), String(c)); this.colH.push(h); put(h, 1, c + 2); }
    for (let r = 0; r < rows; r++) {
      const h = el('div', 'mh' + bt(r), String(r)); this.rowH.push(h); put(h, r + 2, 1);
      const row = [];
      for (let c = 0; c < cols; c++) {
        const x = el('div', 'cell'); x.dataset.base = 'cell k-' + kind + bl(c) + bt(r); x.className = x.dataset.base;
        if (opt.vals) x.textContent = opt.vals[r][c];
        row.push(x); put(x, r + 2, c + 2);
      }
      this.cells.push(row);
    }
    this.badges = []; this.root.appendChild(g);
  }
  Mat.prototype.paint = function (cls, text) {
    for (let r = 0; r < this.rows; r++) for (let c = 0; c < this.cols; c++) {
      const x = this.cells[r][c], ex = cls[key(r, c)] || '', want = x.dataset.base + (ex ? ' ' + ex : '');
      if (x.className !== want) x.className = want;
      if (text) { const t = text(r, c); if (x.textContent !== t) x.textContent = t; }
    }
  };
  Mat.prototype.heads = function (rs, cs) {
    this.rowH.forEach((h, r) => h.classList.toggle('on', !!rs && rs.indexOf(r) >= 0));
    this.colH.forEach((h, c) => h.classList.toggle('on', !!cs && cs.indexOf(c) >= 0));
  };
  Mat.prototype.badge = function (r0, c0, rs, cs, text, cls) {
    const b = el('div', 'badge' + (cls ? ' ' + cls : ''), text ? '<span>' + text + '</span>' : '');
    b.style.gridRow = (2 + r0) + ' / span ' + rs; b.style.gridColumn = (2 + c0) + ' / span ' + cs;
    this.grid.appendChild(b); this.badges.push(b); return b;
  };
  Mat.prototype.clearBadges = function () { this.badges.forEach((b) => b.remove()); this.badges = []; };

  function Triple(P, t, note) {
    this.root = el('div', 'triple');
    const big = Math.max(P.M, P.N, P.K);
    if (big > 6) this.root.classList.add('dense');
    this.A = new Mat('A', P.M, P.K, { vals: P.A, rb: t.tm, cb: t.tk, dims: P.M + '×' + P.K + ' (M×K)' });
    this.B = new Mat('B', P.K, P.N, { vals: P.B, rb: t.tk, cb: t.tn, dims: P.K + '×' + P.N + ' (K×N)' });
    this.C = new Mat('C', P.M, P.N, { rb: t.tm, cb: t.tn, dims: P.M + '×' + P.N + ' (M×N)' });
    this.root.append(el('div', 'tx', note || 'C = A @ B'), this.B.root, this.A.root, this.C.root);
  }

  // ---------------- stepping mixin ----------------
  const Step = {
    step() { if (this.t < this.n) { this.go(this.t + 1); return true; } return false; },
    back() { if (this.t > 0) { this.go(this.t - 1); return true; } return false; },
    finish() { this.stop(); if (this.t < this.n) { this.go(this.n); return true; } return false; },
    reset() { this.stop(); this.go(0); },
    atEnd() { return this.t >= this.n; },
    atStart() { return this.t <= 0; },
    play() {
      if (this.timer) { this.stop(); return; }
      if (this.t >= this.n) this.go(0);
      this.timer = setInterval(() => { if (!this.step()) this.stop(); }, this.speed());
      this.syncPlay();
    },
    stop() { if (this.timer) { clearInterval(this.timer); this.timer = null; } this.syncPlay(); },
    syncPlay() { if (this.playBtn) { this.playBtn.textContent = this.timer ? 'Pause' : 'Play'; this.playBtn.setAttribute('aria-pressed', this.timer ? 'true' : 'false'); } },
    speed() { const s = this.cfg.speed || 1; return Math.max(40, Math.round(this.baseSpeed / s)); },
    go(t) { this.t = Math.max(0, Math.min(this.n, t)); this.render(); if (this.scrub) { this.scrub.max = this.n; this.scrub.value = this.t; } if (this.cfg.onStep) this.cfg.onStep(this); },
    controls(innerLabel) {
      const c = el('div', 'controls');
      c.append(btn('Reset', 'Reset (R)', () => this.reset()), btn('◀', 'Step back (←)', () => { this.stop(); this.back(); }), btn('Step ▶', 'One step (→)', () => { this.stop(); this.step(); }, true));
      if (innerLabel) c.append(btn(innerLabel, 'Bigger jump (↓)', () => { this.stop(); this.inner(); }));
      this.playBtn = btn('Play', 'Play / pause (P)', () => this.play());
      c.append(this.playBtn, btn('Finish', 'Jump to the end (E)', () => this.finish()));
      return c;
    },
    scrubber() {
      const w = el('label', 'scrub'); w.innerHTML = '<span>time</span>';
      const r = el('input'); r.type = 'range'; r.min = 0; r.max = this.n; r.value = 0; r.id = 'scrub-' + Math.random().toString(36).slice(2, 8);
      r.addEventListener('input', () => { this.stop(); this.go(+r.value); });
      w.appendChild(r); this.scrub = r; return w;
    },
    keydown(e) {
      const k = e.key;
      if (k === 'ArrowRight') { this.stop(); return this.step(); }
      if (k === 'ArrowLeft') { this.stop(); return this.back(); }
      if (k === 'ArrowDown') { this.stop(); return this.inner(); }
      if (k === 'p' || k === 'P') { this.play(); return true; }
      if (k === 'e' || k === 'E') return this.finish();
      if (k === 'r' || k === 'R') { this.reset(); return true; }
      return false;
    },
    keys() { this.root.tabIndex = 0; this.root.addEventListener('keydown', (e) => { if (e.target !== this.root) return; if (this.keydown(e)) e.preventDefault(); }); },
    destroy() { this.stop(); }
  };
  function mix(P) { Object.keys(Step).forEach((k) => { if (!P[k]) P[k] = Step[k]; }); }

  // =====================================================================
  // Single core
  // cfg: P, tiled, tm, tn, tk, border ('ijk' letters for mt nt kt), inner, gran ('op'|'inner'|'block')
  // =====================================================================
  function singlePlan(P, cfg) {
    const T = cfg.tiled ? { i: cfg.tm, j: cfg.tn, k: cfg.tk } : { i: P.M, j: P.N, k: P.K };
    const nb = { i: P.M / T.i, j: P.N / T.j, k: P.K / T.k };
    const bo = cfg.tiled ? cfg.border : 'ijk', io = cfg.inner;
    const ops = []; let blk = 0, bgrp = 0, igrp = 0;
    for (let a0 = 0; a0 < nb[bo[0]]; a0++) for (let a1 = 0; a1 < nb[bo[1]]; a1++) {
      for (let a2 = 0; a2 < nb[bo[2]]; a2++) {
        const bid = {}; bid[bo[0]] = a0; bid[bo[1]] = a1; bid[bo[2]] = a2;
        for (let s0 = 0; s0 < T[io[0]]; s0++) for (let s1 = 0; s1 < T[io[1]]; s1++) {
          for (let s2 = 0; s2 < T[io[2]]; s2++) {
            const sid = {}; sid[io[0]] = s0; sid[io[1]] = s1; sid[io[2]] = s2;
            ops.push({ i: bid.i * T.i + sid.i, j: bid.j * T.j + sid.j, k: bid.k * T.k + sid.k, bid, sid, blk, bgrp, igrp });
          }
          igrp++;
        }
        blk++;
      }
      bgrp++;
    }
    const field = cfg.gran === 'block' ? 'blk' : cfg.gran === 'inner' ? 'igrp' : null;
    const ends = [];
    ops.forEach((o, x) => { if (!field || x === ops.length - 1 || ops[x + 1][field] !== o[field]) ends.push(x + 1); });
    return { T, nb, bo, io, ops, ends };
  }

  function singleCode(P, cfg, pl) {
    const L = [];
    const ind = (d) => '  '.repeat(d);
    if (cfg.tiled) {
      pl.bo.split('').forEach((v, d) => L.push({ v: 'b' + v, html: ind(d) + '<span class="kw">for</span> ' + BV[v] + ' <span class="kw">in</span> range(' + pl.nb[v] + '):' + (d === 0 ? '<span class="cm">   # block loops</span>' : '') }));
      pl.io.split('').forEach((v, d) => L.push({ v: 's' + v, html: ind(d + 3) + '<span class="kw">for</span> ' + v + ' <span class="kw">in</span> block(' + BV[v] + '):' + (d === 0 ? '<span class="cm">   # inside one block product</span>' : '') }));
      L.push({ v: 'body', html: ind(6) + '<span class="tc">C</span>[i][j] += <span class="ta">A</span>[i][k] * <span class="tb">B</span>[k][j]' });
    } else {
      pl.io.split('').forEach((v, d) => L.push({ v: 's' + v, html: ind(d) + '<span class="kw">for</span> ' + v + ' <span class="kw">in</span> range(' + DIM[v] + '):' }));
      L.push({ v: 'body', html: ind(3) + '<span class="tc">C</span>[i][j] += <span class="ta">A</span>[i][k] * <span class="tb">B</span>[k][j]' });
    }
    return L;
  }

  function SingleView(root, cfg) {
    this.cfg = cfg; this.root = root; this.P = cfg.P; root.classList.add('w');
    this.pl = singlePlan(this.P, cfg); this.n = this.pl.ends.length; this.t = 0;
    this.baseSpeed = cfg.gran === 'block' ? 700 : cfg.gran === 'inner' ? 500 : 260;
    const bodyEl = el('div', 'w-body'); root.appendChild(bodyEl);
    const left = el('div', 'stack'); bodyEl.appendChild(left);
    this.codeEl = el('pre', 'code'); left.appendChild(this.codeEl);
    this.lines = singleCode(this.P, cfg, this.pl);
    this.lineEls = this.lines.map((l) => { const d = el('div', 'ln', l.html); this.codeEl.appendChild(d); return d; });
    this.valEls = this.lines.map((l, n) => { if (l.v === 'body') return null; const s = el('span', 'val'); this.lineEls[n].appendChild(s); return s; });
    const t = cfg.tiled ? cfg : { tm: this.P.M, tn: this.P.N, tk: this.P.K };
    this.tri = new Triple(this.P, t); bodyEl.appendChild(this.tri.root);
    this.cap = el('div', 'cap'); root.appendChild(this.cap);
    const foot = el('div', 'w-foot'); root.appendChild(foot);
    this.cnt = el('div', 'counters'); foot.appendChild(this.cnt);
    const right = el('div', 'stack right'); right.append(this.controls(cfg.gran === 'block' ? null : 'Bigger jump ⏭'), this.scrubber()); foot.appendChild(right);
    this.keys(); this.go(0);
  }
  SingleView.prototype.inner = function () {
    if (this.t >= this.n) return false;
    const ops = this.pl.ops, done = this.t ? this.pl.ends[this.t - 1] : 0;
    const f = this.cfg.gran === 'op' ? 'igrp' : 'bgrp';
    const g = ops[done][f]; let x = done; while (x < ops.length && ops[x][f] === g) x++;
    let s = this.t; while (s < this.n && this.pl.ends[s] < x) s++;
    this.go(Math.min(this.n, s + 1)); return true;
  };
  SingleView.prototype.state = function () {
    const s = newAcc(this.P), done = this.t ? this.pl.ends[this.t - 1] : 0, from = this.t > 1 ? this.pl.ends[this.t - 2] : 0, last = [];
    for (let x = 0; x < done; x++) { const r = apply(this.P, s, this.pl.ops[x].i, this.pl.ops[x].j, this.pl.ops[x].k); if (x >= from) last.push(r); }
    return { s, done, last };
  };
  SingleView.prototype.render = function () {
    const P = this.P, pl = this.pl, cfg = this.cfg;
    const { s, done, last } = this.state();
    const ctx = this.t < this.n;
    const ref = pl.ops[this.t ? done - 1 : 0];
    const blockLevel = cfg.gran === 'block';
    const grpField = blockLevel ? 'bgrp' : 'igrp';
    const stay = blockLevel ? STAY[pl.bo[2]] : STAY[pl.io[2]];
    const cA = {}, cB = {}, cC = {};
    const maps = { A: cA, B: cB, C: cC };
    if (ctx) {
      pl.ops.forEach((o) => { if (o[grpField] !== ref[grpField]) return; addCls(cA, key(o.i, o.k), 'sweep'); addCls(cB, key(o.k, o.j), 'sweep'); addCls(cC, key(o.i, o.j), 'sweep'); });
      const T = pl.T;
      const cellsOf = (op) => {
        if (!blockLevel) return [op === 'A' ? key(ref.i, ref.k) : op === 'B' ? key(ref.k, ref.j) : key(ref.i, ref.j)];
        const [rv, cv] = IDX[op]; const out = [];
        for (let a = 0; a < T[rv]; a++) for (let b = 0; b < T[cv]; b++) out.push(key(ref.bid[rv] * T[rv] + a, ref.bid[cv] * T[cv] + b));
        return out;
      };
      cellsOf(stay).forEach((k0) => addCls(maps[stay], k0, 'stay'));
    }
    for (let r = 0; r < P.M; r++) for (let c = 0; c < P.N; c++) { const t = s.T[r][c]; if (t) addCls(cC, key(r, c), t === P.K ? 'fin' : 'part'); }
    if (this.t > 0) last.forEach((o) => { addCls(cA, key(o.i, o.k), 'read'); addCls(cB, key(o.k, o.j), 'read'); addCls(cC, key(o.i, o.j), 'write'); });
    this.tri.A.paint(cA); this.tri.B.paint(cB);
    this.tri.C.paint(cC, (r, c) => (s.T[r][c] ? String(s.C[r][c]) : '·'));
    ['A', 'B', 'C'].forEach((m) => this.tri[m].clearBadges());
    if (ctx && cfg.tiled && !blockLevel && (pl.nb.i * pl.nb.j * pl.nb.k > 1)) {
      const T = pl.T, b = ref.bid;
      this.tri.A.badge(b.i * T.i, b.k * T.k, T.i, T.k, null, 'soft');
      this.tri.B.badge(b.k * T.k, b.j * T.j, T.k, T.j, null, 'soft');
      this.tri.C.badge(b.i * T.i, b.j * T.j, T.i, T.j, null, 'soft');
    }
    if (ctx) { this.tri.A.heads([ref.i], [ref.k]); this.tri.B.heads([ref.k], [ref.j]); this.tri.C.heads([ref.i], [ref.j]); }
    else ['A', 'B', 'C'].forEach((m) => this.tri[m].heads(null, null));
    // code
    this.lineEls.forEach((d, n) => {
      const l = this.lines[n];
      d.classList.toggle('on', l.v === 'body' && this.t > 0 && ctx);
      d.classList.toggle('inner', ctx && ((blockLevel && l.v === 'b' + pl.bo[2]) || (!blockLevel && l.v === 's' + pl.io[2])));
      const sp = this.valEls[n];
      if (sp) {
        let txt = '';
        if (this.t > 0 && ctx) { const v = l.v[1]; txt = l.v[0] === 'b' ? BV[v] + ' = ' + ref.bid[v] : v + ' = ' + ref[v]; }
        sp.textContent = txt;
      }
    });
    // caption
    let cap;
    if (this.t === 0) cap = 'Press <b>Step</b> (or →). Tinted cells are the ones the current ' + (blockLevel ? 'block loop' : 'inner loop') + ' will visit; the glowing one stays put.';
    else if (this.t === this.n) cap = 'Done: ' + s.mul + ' multiplies, ' + s.add + ' additions. ' + (eqRef(P, s.C) ? '<span class="ok">Check: C == A @ B ✓</span>' : '<span class="tsp">C differs ✗</span>');
    else if (last.length === 1) {
      const o = last[0], nm = 'C[' + o.i + '][' + o.j + ']';
      cap = '<code><span class="tc">' + nm + '</span> += <span class="ta">A[' + o.i + '][' + o.k + ']</span>·<span class="tb">B[' + o.k + '][' + o.j + ']</span></code> = ' + o.a + '·' + o.b + ' = ' + o.p + ' → ' +
        (o.first ? nm + ' = ' + o.nw + ' <small>(first term, just stored)</small>' : nm + ' = ' + o.old + ' + ' + o.p + ' = <b>' + o.nw + '</b>') + ' · term ' + o.terms + ' of ' + P.K + (o.terms === P.K ? ' <b class="tc">finished</b>' : ' <span class="tc">partial</span>');
    } else {
      const b = ref.bid;
      cap = blockLevel
        ? '<code><span class="tc">C[' + b.i + ',' + b.j + ']</span> += <span class="ta">A[' + b.i + ',' + b.k + ']</span> @ <span class="tb">B[' + b.k + ',' + b.j + ']</span></code> — one block product, ' + last.length + ' multiplies at once.'
        : 'One whole inner loop over <b>' + pl.io[2] + '</b>: ' + last.length + ' multiplies. ' + stay + ' stayed put the whole time.';
    }
    this.cap.innerHTML = cap;
    const MNK = P.M * P.N * P.K;
    this.cnt.innerHTML = counters([
      { label: 'step', value: this.t + '/' + this.n }, { label: 'multiplies', value: s.mul + '/' + MNK },
      { label: 'additions', value: s.add + '/' + P.M * P.N * (P.K - 1) },
      { label: 'partial now', value: s.part }, { label: 'most partial at once', value: s.maxPart }
    ]);
  };
  SingleView.prototype.trace = function (limit) {
    const { done } = this.state(); const s = newAcc(this.P); const out = [];
    for (let x = 0; x < done; x++) {
      const o = this.pl.ops[x], r = apply(this.P, s, o.i, o.j, o.k);
      if (x >= done - (limit || 400)) {
        out.push('step ' + String(x + 1).padStart(3) + ': C[' + o.i + '][' + o.j + '] += A[' + o.i + '][' + o.k + ']*B[' + o.k + '][' + o.j + '] = ' + r.a + '*' + r.b + ' = ' + r.p + '  ->  ' +
          (r.first ? r.nw + ' (first term)' : r.old + ' + ' + r.p + ' = ' + r.nw) + '   term ' + r.terms + '/' + this.P.K + (r.terms === this.P.K ? ' FINISHED' : ''));
      }
    }
    return out;
  };
  mix(SingleView.prototype);

  // =====================================================================
  // Many cores
  // cfg: P, tm, tn, tk, cores, flow ('os'|'ws'|'is'), g (K-tiles per core, ws/is), S (split-K pieces, os)
  // =====================================================================
  function parallelPlan(P, cfg) {
    const nM = P.M / cfg.tm, nN = P.N / cfg.tn, nK = P.K / cfg.tk;
    const units = [];
    const flow = cfg.flow;
    if (flow === 'os') {
      const S = cfg.S || 1, ps = nK / S;
      for (let mt = 0; mt < nM; mt++) for (let nt = 0; nt < nN; nt++) for (let s = 0; s < S; s++) {
        const prods = rng(ps).map((x) => ({ mt, nt, kt: s * ps + x }));
        units.push({ prods, stay: 'C', stayBlocks: [[mt, nt]], cblocks: [[mt, nt]], label: 'C' + mt + nt + (S > 1 ? ' · piece ' + s : ''), short: 'C[' + mt + ',' + nt + ']' + (S > 1 ? ' p' + s : '') });
      }
    } else if (flow === 'ws') {
      const g = cfg.g || 1, nG = nK / g;
      for (let kg = 0; kg < nG; kg++) for (let nt = 0; nt < nN; nt++) {
        const kts = rng(g).map((x) => kg * g + x);
        const prods = []; rng(nM).forEach((mt) => kts.forEach((kt) => prods.push({ mt, nt, kt })));
        units.push({ prods, stay: 'B', stayBlocks: kts.map((kt) => [kt, nt]), cblocks: rng(nM).map((mt) => [mt, nt]), label: 'B[' + kts.join('') + ',' + nt + ']', short: 'B[' + (g > 1 ? kts[0] + '–' + kts[g - 1] : kts[0]) + ',' + nt + ']' });
      }
    } else {
      const g = cfg.g || 1, nG = nK / g;
      for (let mt = 0; mt < nM; mt++) for (let kg = 0; kg < nG; kg++) {
        const kts = rng(g).map((x) => kg * g + x);
        const prods = []; rng(nN).forEach((nt) => kts.forEach((kt) => prods.push({ mt, nt, kt })));
        units.push({ prods, stay: 'A', stayBlocks: kts.map((kt) => [mt, kt]), cblocks: rng(nN).map((nt) => [mt, nt]), label: 'A[' + mt + ',' + kts.join('') + ']', short: 'A[' + mt + ',' + (g > 1 ? kts[0] + '–' + kts[g - 1] : kts[0]) + ']' });
      }
    }
    const C = cfg.cores, U = units.length, L = units[0].prods.length, W = Math.ceil(U / C);
    units.forEach((u, x) => { u.id = x; u.wave = Math.floor(x / C); u.core = x % C; u.start = u.wave * L; });
    const parts = {};
    units.forEach((u) => u.cblocks.forEach(([a, b]) => { const k0 = key(a, b); (parts[k0] = parts[k0] || []).push(u.id); }));
    const maxParts = Math.max.apply(null, Object.values(parts).map((x) => x.length));
    const R = maxParts > 1 ? Math.ceil(Math.log2(maxParts)) : 0;
    return { nM, nN, nK, units, C, U, L, W, R, parts, maxParts, compute: W * L, n: W * L + R };
  }

  function parallelState(P, cfg, pl, t) {
    const tm = cfg.tm, tn = cfg.tn, tk = cfg.tk;
    const comp = Math.min(t, pl.compute);
    const bufs = pl.units.map(() => ({}));
    let mul = 0, tadd = 0, sadd = 0;
    pl.units.forEach((u) => {
      const done = Math.max(0, Math.min(pl.L, comp - u.start));
      for (let x = 0; x < done; x++) {
        const p = u.prods[x];
        for (let a = 0; a < tm; a++) for (let b = 0; b < tn; b++) for (let c = 0; c < tk; c++) {
          const i = p.mt * tm + a, j = p.nt * tn + b, k = p.kt * tk + c;
          const k0 = key(i, j), bf = bufs[u.id];
          if (!bf[k0]) bf[k0] = { v: 0, T: 0 };
          if (bf[k0].T > 0) tadd++;
          bf[k0].v += P.A[i][k] * P.B[k][j]; bf[k0].T++; mul++;
        }
      }
    });
    const lv = Math.max(0, t - pl.compute);
    const C = zeros(P.M, P.N), T = zeros(P.M, P.N), pieces = zeros(P.M, P.N);
    for (let i = 0; i < P.M; i++) for (let j = 0; j < P.N; j++) {
      const bk = key(Math.floor(i / tm), Math.floor(j / tn));
      let list = pl.parts[bk].map((u) => bufs[u][key(i, j)]).filter(Boolean).map((x) => ({ v: x.v, T: x.T }));
      if (pl.parts[bk].length === 1 || lv > 0) {
        for (let L = 0; L < lv && list.length > 1; L++) {
          const nx = [];
          for (let q = 0; q < list.length; q += 2) {
            if (q + 1 < list.length) { nx.push({ v: list[q].v + list[q + 1].v, T: list[q].T + list[q + 1].T }); sadd++; } else nx.push(list[q]);
          }
          list = nx;
        }
      }
      pieces[i][j] = list.length;
      if (list.length === 1) { C[i][j] = list[0].v; T[i][j] = list[0].T; }
    }
    return { bufs, mul, tadd, sadd, C, T, pieces, lv };
  }

  function parallelCode(cfg, pl) {
    const g = cfg.g || 1, S = cfg.S || 1;
    const L = [];
    const kw = (x) => '<span class="kw">' + x + '</span>', cm = (x) => '<span class="cm">   # ' + x + '</span>';
    if (cfg.flow === 'os') {
      L.push({ v: 'grid', html: kw('parallel for') + ' (mt, nt' + (S > 1 ? ', piece' : '') + ') ' + kw('in') + ' grid:' + cm(pl.U + ' tiles, ' + cfg.cores + ' cores') });
      L.push({ v: 'sweep', html: '    ' + kw('for') + ' kt ' + kw('in') + ' ' + (S > 1 ? 'piece_range(piece)' : 'range(' + pl.nK + ')') + ':' + cm('the sweep, in order') });
      L.push({ v: 'body', html: '        ' + (S > 1 ? '<span class="tc">P</span>[piece]' : '<span class="tc">C</span>') + '[mt,nt] += <span class="ta">A</span>[mt,kt] @ <span class="tb">B</span>[kt,nt]' });
      if (S > 1) L.push({ v: 'red', html: '<span class="tc">C</span>[mt,nt] = ' + kw('sum') + ' of <span class="tc">P</span>[piece][mt,nt]' + cm('across cores') });
    } else {
      const ws = cfg.flow === 'ws';
      L.push({ v: 'grid', html: kw('parallel for') + ' (' + (ws ? 'kgroup, nt' : 'mt, kgroup') + ') ' + kw('in') + ' grid:' + cm(pl.U + ' tiles, ' + cfg.cores + ' cores') });
      L.push({ v: 'sweep', html: '    ' + kw('for') + ' ' + (ws ? 'mt' : 'nt') + ' ' + kw('in') + ' range(' + (ws ? pl.nM : pl.nN) + '):' + cm('the sweep, in order') });
      L.push({ v: 'kg', html: '        ' + kw('for') + ' kt ' + kw('in') + ' kgroup:' + cm(g + ' K-tile' + (g > 1 ? 's' : '') + ' per core, summed over time') });
      L.push({ v: 'body', html: '            <span class="tc">P</span>[kgroup][mt,nt] += <span class="ta">A</span>[mt,kt] @ <span class="tb">B</span>[kt,nt]' });
      if (pl.maxParts > 1) L.push({ v: 'red', html: '<span class="tc">C</span>[mt,nt] = ' + kw('sum') + ' of <span class="tc">P</span>[kgroup][mt,nt]' + cm('across cores') });
      else L.push({ v: 'red', html: '<span class="tc">C</span>[mt,nt] = <span class="tc">P</span>[0][mt,nt]' + cm('one group holds all of K: nothing to combine') });
    }
    return L;
  }

  function ParallelView(root, cfg) {
    this.cfg = cfg; this.root = root; this.P = cfg.P; root.classList.add('w', 'pv');
    this.baseSpeed = 1000; this.t = 0;
    this.pl = parallelPlan(this.P, cfg); this.n = this.pl.n;
    if (cfg.title !== false) {
      const head = el('div', 'w-head'); root.appendChild(head);
      const f = cfg.flow, st = { os: 'C', ws: 'B', is: 'A' }[f];
      head.innerHTML = '<span class="chip ' + st + '">each core keeps a ' + st + ' block · ' + f + '</span>' +
        '<span class="chip">' + this.pl.U + ' tile' + (this.pl.U > 1 ? 's' : '') + ' → ' + cfg.cores + ' core' + (cfg.cores > 1 ? 's' : '') + ' → ' + this.pl.W + ' wave' + (this.pl.W > 1 ? 's' : '') + '</span>' +
        '<span class="chip">' + this.pl.L + ' block product' + (this.pl.L > 1 ? 's' : '') + ' per tile, in order</span>' +
        (this.pl.maxParts > 1 ? '<span class="chip sp">each C block in ' + this.pl.maxParts + ' pieces → combine</span>' : '<span class="chip">each C block made in one place</span>');
    }
    const top = el('div', 'pv-top'); root.appendChild(top);
    this.tri = new Triple(this.P, cfg, 'tiles: ' + cfg.tm + '×' + cfg.tn + '×' + cfg.tk); top.appendChild(this.tri.root);
    const codeWrap = el('div', 'stack'); top.appendChild(codeWrap);
    this.codeEl = el('pre', 'code'); codeWrap.appendChild(this.codeEl);
    this.lines = parallelCode(cfg, this.pl);
    this.lineEls = this.lines.map((l) => { const d = el('div', 'ln', l.html); this.codeEl.appendChild(d); return d; });
    this.legend = el('div', 'legend', '<span><i class="lg read"></i>block being read now (tag = which core)</span><span><i class="lg stay"></i>kept inside a core</span><span><i class="lg part"></i>partial</span><span><i class="lg scat"></i>exists as pieces on several cores</span><span><i class="lg fin"></i>finished</span>');
    codeWrap.appendChild(this.legend);
    root.appendChild(el('div', 'pv-lab', 'The chip: what every core is doing right now'));
    this.floor = el('div', 'floor'); root.appendChild(this.floor);
    this.coreEls = rng(cfg.cores).map((c) => { const b = el('div', 'corebox'); this.floor.appendChild(b); return b; });
    root.appendChild(el('div', 'pv-lab', 'Timeline: one row per core, one column per step'));
    this.ganttWrap = el('div', 'gantt-wrap'); root.appendChild(this.ganttWrap);
    this.buildGantt();
    this.cap = el('div', 'cap'); root.appendChild(this.cap);
    const foot = el('div', 'w-foot'); root.appendChild(foot);
    this.cnt = el('div', 'counters'); foot.appendChild(this.cnt);
    const right = el('div', 'stack right'); right.append(this.controls(null), this.scrubber()); foot.appendChild(right);
    this.keys(); this.go(0);
  }
  ParallelView.prototype.inner = function () { return this.step(); };
  ParallelView.prototype.buildGantt = function () {
    const pl = this.pl, cfg = this.cfg;
    const g = el('div', 'gantt'); this.gantt = g;
    const cols = pl.n;
    g.style.gridTemplateColumns = '4.2rem repeat(' + cols + ', minmax(' + (cols > 24 ? '1.6rem' : '2.6rem') + ', 1fr))';
    const rows = cfg.cores + 2 + (pl.R ? 1 : 0);
    // header: waves
    g.appendChild(Object.assign(el('div', 'g-h'), { style: 'grid-row:1;grid-column:1' }));
    for (let w = 0; w < pl.W; w++) {
      const h = el('div', 'g-wave', 'wave ' + (w + 1));
      h.style.gridRow = '1'; h.style.gridColumn = (2 + w * pl.L) + ' / span ' + pl.L; g.appendChild(h);
    }
    if (pl.R) { const h = el('div', 'g-wave sp', 'combine'); h.style.gridRow = '1'; h.style.gridColumn = (2 + pl.compute) + ' / span ' + pl.R; g.appendChild(h); }
    // step numbers
    const sl = el('div', 'g-lab', 'step'); sl.style.gridRow = '2'; sl.style.gridColumn = '1'; g.appendChild(sl);
    for (let c = 0; c < cols; c++) { const x = el('div', 'g-num', String(c + 1)); x.style.gridRow = '2'; x.style.gridColumn = String(2 + c); g.appendChild(x); }
    // wave bands
    for (let w = 0; w < pl.W; w++) {
      const band = el('div', 'g-band' + (w % 2 ? ' alt' : ''));
      band.style.gridRow = '3 / span ' + cfg.cores; band.style.gridColumn = (2 + w * pl.L) + ' / span ' + pl.L; g.appendChild(band);
    }
    // core rows
    this.bars = [];
    for (let c = 0; c < cfg.cores; c++) {
      const lab = el('div', 'g-lab', 'core ' + c); lab.style.gridRow = String(3 + c); lab.style.gridColumn = '1'; g.appendChild(lab);
      for (let w = 0; w < pl.W; w++) {
        const u = pl.units.find((x) => x.wave === w && x.core === c);
        const b = el('div', u ? 'bar f-' + cfg.flow : 'bar idle', u ? '<span class="fill"></span><span class="bt">' + u.short + '</span>' : '<span class="bt">idle</span>');
        b.style.gridRow = String(3 + c); b.style.gridColumn = (2 + w * pl.L) + ' / span ' + pl.L;
        if (u) {
          b.title = 'tile ' + u.id + ' on core ' + c + ', wave ' + (w + 1) + ': keeps ' + u.label + ', makes ' + u.cblocks.map(([a, bb]) => 'C[' + a + ',' + bb + ']').join(' ');
          b.dataset.u = u.id;
          b.addEventListener('mouseenter', () => this.hover(u)); b.addEventListener('mouseleave', () => this.hover(null));
          b.addEventListener('focus', () => this.hover(u)); b.addEventListener('blur', () => this.hover(null));
          b.tabIndex = 0;
          this.bars.push({ el: b, u });
        }
        g.appendChild(b);
      }
    }
    if (pl.R) {
      const lab = el('div', 'g-lab sp', 'across cores'); lab.style.gridRow = String(3 + cfg.cores); lab.style.gridColumn = '1'; g.appendChild(lab);
      this.redBars = rng(pl.R).map((r) => {
        const b = el('div', 'bar red', '<span class="fill"></span><span class="bt">+ level ' + (r + 1) + '</span>');
        b.style.gridRow = String(3 + cfg.cores); b.style.gridColumn = String(2 + pl.compute + r); g.appendChild(b); return b;
      });
    }
    this.cursor = el('div', 'g-cursor'); this.cursor.style.gridRow = '2 / span ' + (rows - 1); g.appendChild(this.cursor);
    this.ganttWrap.appendChild(g);
  };
  ParallelView.prototype.hover = function (u) { this.hov = u; this.render(); };
  ParallelView.prototype.render = function () {
    const P = this.P, cfg = this.cfg, pl = this.pl, tm = cfg.tm, tn = cfg.tn, tk = cfg.tk;
    const st = parallelState(P, cfg, pl, this.t);
    const slot = this.t - 1; const inCompute = slot >= 0 && slot < pl.compute; const inRed = slot >= pl.compute;
    const wave = inCompute ? Math.floor(slot / pl.L) : -1, pos = inCompute ? slot % pl.L : -1;
    const active = inCompute ? pl.units.filter((u) => u.wave === wave) : [];
    const cA = {}, cB = {}, cC = {}, maps = { A: cA, B: cB, C: cC };
    const bdim = { A: [tm, tk], B: [tk, tn], C: [tm, tn] };
    const blockCells = (op, r0, c0) => { const [R, Cc] = bdim[op]; const out = []; for (let a = 0; a < R; a++) for (let b = 0; b < Cc; b++) out.push(key(r0 * R + a, c0 * Cc + b)); return out; };
    // blocks kept by active cores glow; blocks read now are lit
    const tags = { A: {}, B: {}, C: {} };
    active.forEach((u) => {
      const p = u.prods[pos];
      u.stayBlocks.forEach(([a, b]) => blockCells(u.stay, a, b).forEach((k0) => addCls(maps[u.stay], k0, 'stay')));
      blockCells('A', p.mt, p.kt).forEach((k0) => addCls(cA, k0, 'read'));
      blockCells('B', p.kt, p.nt).forEach((k0) => addCls(cB, k0, 'read'));
      blockCells('C', p.mt, p.nt).forEach((k0) => addCls(cC, k0, 'write'));
      [['A', p.mt, p.kt], ['B', p.kt, p.nt], ['C', p.mt, p.nt]].forEach(([op, a, b]) => { const k0 = key(a, b); (tags[op][k0] = tags[op][k0] || []).push(u.core); });
    });
    for (let i = 0; i < P.M; i++) for (let j = 0; j < P.N; j++) {
      const k0 = key(i, j);
      if (st.pieces[i][j] > 1) addCls(cC, k0, 'scat');
      else if (st.T[i][j]) addCls(cC, k0, st.T[i][j] === P.K ? 'fin' : 'part');
      if (inRed && pl.parts[key(Math.floor(i / tm), Math.floor(j / tn))].length > 1) addCls(cC, k0, 'spadd');
    }
    if (this.hov) {
      const u = this.hov;
      u.stayBlocks.forEach(([a, b]) => blockCells(u.stay, a, b).forEach((k0) => addCls(maps[u.stay], k0, 'hov')));
      u.cblocks.forEach(([a, b]) => blockCells('C', a, b).forEach((k0) => addCls(cC, k0, 'hov')));
    }
    this.tri.A.paint(cA); this.tri.B.paint(cB);
    this.tri.C.paint(cC, (i, j) => (st.pieces[i][j] > 1 ? st.pieces[i][j] + '×' : st.T[i][j] ? String(st.C[i][j]) : '·'));
    ['A', 'B', 'C'].forEach((m) => {
      this.tri[m].clearBadges();
      const [R, Cc] = m === 'A' ? [tm, tk] : m === 'B' ? [tk, tn] : [tm, tn];
      Object.keys(tags[m]).forEach((k0) => { const [a, b] = k0.split(',').map(Number); this.tri[m].badge(a * R, b * Cc, R, Cc, tags[m][k0].map((c) => 'c' + c).join(' '), 'tag'); });
    });
    // cores
    this.coreEls.forEach((box, c) => {
      const u = active.find((x) => x.core === c);
      box.className = 'corebox' + (u ? '' : ' idle');
      if (!u) {
        const lastU = pl.units.filter((x) => x.core === c && x.start < Math.max(0, this.t)).pop();
        let msg = this.t === 0 ? 'waiting to start' : inRed ? 'done — its partial sums are being combined across cores' : slot >= pl.compute ? 'done' : 'idle: no tile left for this core in wave ' + (wave + 1);
        if (!inRed && this.t >= pl.n) msg = 'done';
        box.innerHTML = '<h5><span>core ' + c + '</span><span class="muted">' + (lastU ? 'last: ' + lastU.short : '') + '</span></h5><div class="op">' + msg + '</div>';
        return;
      }
      const p = u.prods[pos];
      const blk = (op, r0, c0, R, Cc, vals, cls, label) => {
        const w = el('div', 'cb-blk'); w.appendChild(el('div', 'cb-l', label));
        const m = el('div', 'mini'); m.style.gridTemplateColumns = 'repeat(' + Cc + ', var(--cell))';
        for (let a = 0; a < R; a++) for (let b = 0; b < Cc; b++) {
          const i = r0 * R + a, j = c0 * Cc + b;
          let txt = '', extra = cls;
          if (op === 'C') { const bf = st.bufs[u.id][key(i, j)]; txt = bf ? String(bf.v) : '·'; extra += bf ? (bf.T === P.K ? ' fin' : ' part') : ''; }
          else txt = String(vals[i][j]);
          m.appendChild(el('div', 'cell k-' + op + ' ' + extra, txt));
        }
        w.appendChild(m); return w;
      };
      box.innerHTML = '';
      box.appendChild(el('h5', null, '<span>core ' + c + '</span><span class="muted">tile ' + u.id + ' · ' + (pos + 1) + '/' + pl.L + '</span>'));
      if (u.stayBlocks.length > 1) box.appendChild(el('div', 'op', 'in its slots: ' + u.stayBlocks.map(([a, b]) => { const nm = u.stay + '[' + a + ',' + b + ']'; const isCur = (u.stay === 'B' ? a === p.kt && b === p.nt : a === p.mt && b === p.kt); return isCur ? '<b>' + nm + '</b>' : nm; }).join(' · ')));
      const row = el('div', 'cb-row');
      row.append(
        blk('A', p.mt, p.kt, tm, tk, P.A, u.stay === 'A' ? 'stay' : 'read', (u.stay === 'A' ? 'keeps' : 'gets') + ' A[' + p.mt + ',' + p.kt + ']'),
        el('span', 'arrow', '@'),
        blk('B', p.kt, p.nt, tk, tn, P.B, u.stay === 'B' ? 'stay' : 'read', (u.stay === 'B' ? 'keeps' : 'gets') + ' B[' + p.kt + ',' + p.nt + ']'),
        el('span', 'arrow', '→'),
        blk('C', p.mt, p.nt, tm, tn, null, u.stay === 'C' ? 'stay' : '', (u.stay === 'C' ? 'keeps ' : 'adds into ') + (pl.parts[key(p.mt, p.nt)].length > 1 ? 'its piece of ' : '') + 'C[' + p.mt + ',' + p.nt + ']'));
      box.appendChild(row);
    });
    // gantt progress
    this.bars.forEach(({ el: b, u }) => {
      const done = Math.max(0, Math.min(pl.L, Math.min(this.t, pl.compute) - u.start));
      b.querySelector('.fill').style.width = (100 * done / pl.L) + '%';
      b.classList.toggle('now', u.wave === wave);
      const sib = this.hov && this.hov !== u && u.cblocks.some(([a, c]) => this.hov.cblocks.some(([x, y]) => x === a && y === c));
      b.classList.toggle('hovme', this.hov === u); b.classList.toggle('sib', !!sib);
    });
    if (this.redBars) this.redBars.forEach((b, r) => { b.querySelector('.fill').style.width = (st.lv > r ? 100 : 0) + '%'; b.classList.toggle('now', inRed && slot - pl.compute === r); });
    this.cursor.style.gridColumn = String(2 + Math.max(0, slot));
    this.cursor.hidden = this.t === 0 || this.t > pl.n;
    // code
    this.lineEls.forEach((d, n) => { const v = this.lines[n].v; d.classList.toggle('on', (inCompute && v === 'body') || (inRed && v === 'red')); });
    // caption
    const busy = active.length, idle = cfg.cores - busy;
    let cap;
    if (this.t === 0) cap = 'Press <b>Step</b>. In every step each busy core does <b>one block product</b>, and all busy cores do it <b>at the same time</b>.';
    else if (inCompute) cap = '<b>Step ' + this.t + '</b> · wave ' + (wave + 1) + ' of ' + pl.W + ', sweep position ' + (pos + 1) + ' of ' + pl.L + ': ' + busy + ' core' + (busy > 1 ? 's' : '') + ' busy' + (idle ? ', <b>' + idle + ' idle</b> (no tile left for them)' : '') + '. ' +
      (pl.maxParts > 1 ? 'Each core only makes a <b>piece</b> of its C blocks (shown as “2×” = still in 2 pieces).' : 'Each core adds straight into its own C block, over time.');
    else if (inRed) cap = '<b class="tsp">Combine level ' + (slot - pl.compute + 1) + ':</b> pieces of the same C block, made on different cores, are added together. These additions happen <b>across cores</b> (in space).';
    if (this.t === pl.n) cap = 'Done in ' + pl.n + ' steps. ' + (eqRef(P, st.C) ? '<span class="ok">Check: C == A @ B ✓</span>' : '<span class="tsp">C differs ✗</span>');
    this.cap.innerHTML = cap;
    const util = Math.round(100 * pl.U * pl.L / (pl.W * pl.L * cfg.cores));
    this.cnt.innerHTML = counters([
      { label: 'step', value: this.t + '/' + pl.n }, { label: 'multiplies', value: st.mul + '/' + P.M * P.N * P.K },
      { label: 'additions over time', value: st.tadd }, { label: 'additions across cores', value: st.sadd, cls: 'csp' },
      { label: 'total', value: (st.tadd + st.sadd) + '/' + P.M * P.N * (P.K - 1) }, { label: 'core use', value: util + '%' }
    ]);
  };
  ParallelView.prototype.trace = function () {
    const pl = this.pl, out = [];
    for (let s = 0; s < Math.min(this.t, pl.n); s++) {
      if (s < pl.compute) {
        const w = Math.floor(s / pl.L), pos = s % pl.L;
        const act = pl.units.filter((u) => u.wave === w);
        out.push('step ' + String(s + 1).padStart(2) + ' (wave ' + (w + 1) + '): ' + act.map((u) => { const p = u.prods[pos]; return 'core' + u.core + ' A[' + p.mt + ',' + p.kt + ']@B[' + p.kt + ',' + p.nt + ']→C[' + p.mt + ',' + p.nt + ']'; }).join(' | ') + (act.length < this.cfg.cores ? ' | ' + (this.cfg.cores - act.length) + ' idle' : ''));
      } else out.push('step ' + String(s + 1).padStart(2) + ' (combine level ' + (s - pl.compute + 1) + '): pieces added across cores');
    }
    return out;
  };
  mix(ParallelView.prototype);

  // ---------------- deck mount ----------------
  function PV(root, cfg) {
    // deck widget: fixed problem, optional in-slide knobs
    const P = problem(cfg.M || 4, cfg.N || 4, cfg.K || 4, 0);
    const st = Object.assign({}, cfg);
    root.classList.add('stack');
    const knobs = el('div', 'pv-knobs'); root.appendChild(knobs);
    const host = el('div'); root.appendChild(host);
    const make = () => {
      if (this.view) this.view.destroy();
      host.innerHTML = ''; const d = el('div'); host.appendChild(d);
      this.view = new ParallelView(d, Object.assign({}, st, { P }));
      knobs.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(st[b.dataset.k]) === b.dataset.v ? 'true' : 'false'));
    };
    const LAB = { flow: { os: 'os · cores own C blocks', ws: 'ws · cores keep B blocks', is: 'is · cores keep A blocks' } };
    (cfg.knobs || []).forEach((kn) => {
      const grp = el('div', 'seg'); grp.appendChild(el('span', 'seg-l', { flow: 'flow', cores: 'cores', g: 'K-tiles per core', S: 'split-K pieces' }[kn.k]));
      kn.v.forEach((v) => { const b = btn(LAB[kn.k] ? LAB[kn.k][v] : String(v), kn.k + ' ' + v, () => { st[kn.k] = v; make(); }); b.dataset.k = kn.k; b.dataset.v = String(v); grp.appendChild(b); });
      knobs.appendChild(grp);
    });
    if (!(cfg.knobs || []).length) knobs.hidden = true;
    make();
    this.root = root;
  }
  ['step', 'back', 'inner', 'finish', 'reset', 'atEnd', 'atStart', 'play', 'stop'].forEach((m) => { PV.prototype[m] = function () { return this.view[m].apply(this.view, arguments); }; });

  global.MM2 = { problem, divisors, singlePlan, parallelPlan, parallelState, SingleView, ParallelView, newAcc, apply, eqRef, STAY, SHORT, STATN, FORM, BV, ORDERS, TYPES: { pv: PV }, _: { el, btn, addCls, counters, Mat, Triple, mix, key, rng, zeros } };
})(typeof window !== 'undefined' ? window : globalThis);
