/* Matmul animation engine — shared by the slide deck and the notebook page.
   Everything is replayed from the running example, so every picture is checkable by hand. */
(function (global) {
  'use strict';

  // ---------------- the running example ----------------
  const A = [[1, 2, 0, 3], [0, 1, 2, 1], [3, 0, 1, 2], [2, 1, 3, 0]];
  const B = [[2, 0, 1, 3], [1, 3, 0, 2], [0, 1, 2, 1], [3, 2, 1, 0]];
  const M = 4, N = 4, K = 4;
  const rng = (n) => Array.from({ length: n }, (_, i) => i);
  const zeros = (r = 4, c = 4) => rng(r).map(() => rng(c).map(() => 0));
  const REF = rng(M).map((i) => rng(N).map((j) => rng(K).reduce((s, k) => s + A[i][k] * B[k][j], 0)));

  // ---------------- the rule ----------------
  const IDX = { A: ['i', 'k'], B: ['k', 'j'], C: ['i', 'j'] };
  const STAY = { k: 'C', i: 'B', j: 'A' };
  const STAT = { C: 'output-stationary', B: 'weight-stationary', A: 'input-stationary' };
  const SHORT = { C: 'os', B: 'ws', A: 'is' };
  const FORM = { ijk: 'inner product', jik: 'inner product', ikj: 'row-wise (Gustavson)', jki: 'column-wise', kij: 'outer product', kji: 'outer product' };
  const BVAR = { i: 'mt', j: 'nt', k: 'kt' };
  const ORDERS = ['ijk', 'jik', 'ikj', 'kij', 'jki', 'kji'];

  function makeFrames(order, bs, ranges) {
    bs = bs || 1;
    const nb = 4 / bs;
    const R = Object.assign({ i: rng(nb), j: rng(nb), k: rng(nb) }, ranges || {});
    const v = order.split('');
    const fr = [];
    for (const a of R[v[0]]) for (const b of R[v[1]]) for (const c of R[v[2]]) {
      const idx = {}; idx[v[0]] = a; idx[v[1]] = b; idx[v[2]] = c;
      const ops = [];
      for (let ii = 0; ii < bs; ii++) for (let jj = 0; jj < bs; jj++) for (let kk = 0; kk < bs; kk++)
        ops.push([idx.i * bs + ii, idx.j * bs + jj, idx.k * bs + kk]);
      fr.push({ idx, ops });
    }
    return { frames: fr, R, v };
  }
  function newState() { return { C: zeros(), T: zeros(), mul: 0, add: 0 }; }
  // one elementary step. The first term of a C value is just stored, so it costs no addition.
  function apply(s, op) {
    const [i, j, k] = op;
    const a = A[i][k], b = B[k][j], p = a * b, old = s.C[i][j], first = s.T[i][j] === 0;
    if (!first) s.add++;
    s.C[i][j] = old + p; s.T[i][j]++; s.mul++;
    return { i, j, k, a, b, p, old, nw: old + p, terms: s.T[i][j], first };
  }
  function census(T) { let p = 0, f = 0; T.forEach((r) => r.forEach((t) => { if (t === K) f++; else if (t > 0) p++; })); return { p, f }; }
  function equalsRef(C) { return C.every((r, i) => r.every((x, j) => x === REF[i][j])); }

  // ---------------- DOM helpers ----------------
  function el(tag, cls, html) { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
  function btn(label, title, fn, pri) { const b = el('button', 'btn' + (pri ? ' pri' : ''), label); b.type = 'button'; b.title = title; b.setAttribute('aria-label', title); b.addEventListener('click', fn); return b; }
  function key(i, j) { return i + ',' + j; }

  // A matrix picture. kind: 'A' | 'B' | 'C'. opt: vals, rows, cols, rowOff, colOff, bs, rv, cv, title, dims
  function Mat(kind, opt) {
    opt = opt || {};
    this.kind = kind;
    this.rows = opt.rows || 4; this.cols = opt.cols || 4;
    this.rowOff = opt.rowOff || 0; this.colOff = opt.colOff || 0;
    this.bs = opt.bs || 1;
    this.root = el('div', 'mat m-' + kind + (opt.cls ? ' ' + opt.cls : ''));
    const rv = opt.rv || IDX[kind][0], cv = opt.cv || IDX[kind][1];
    const dims = opt.dims || { A: 'M×K', B: 'K×N', C: 'M×N' }[kind];
    this.root.appendChild(el('div', 'mat-title', '<b>' + (opt.title || kind) + '</b><span>' + dims + '</span>'));
    const g = el('div', 'mat-grid');
    g.style.gridTemplateColumns = 'auto repeat(' + this.cols + ', auto)';
    this.grid = g;
    const put = (e, r, c) => { e.style.gridRow = String(r); e.style.gridColumn = String(c); g.appendChild(e); return e; };
    put(el('div', 'mh corner', rv + '\\' + cv), 1, 1);
    this.colH = []; this.rowH = []; this.cells = [];
    for (let c = 0; c < this.cols; c++) {
      const cc = c + this.colOff;
      const h = el('div', 'mh' + (this.bs > 1 && cc % this.bs === 0 && c > 0 ? ' bl' : ''), String(cc));
      this.colH.push(h); put(h, 1, c + 2);
    }
    for (let r = 0; r < this.rows; r++) {
      const rr = r + this.rowOff;
      const h = el('div', 'mh' + (this.bs > 1 && rr % this.bs === 0 && r > 0 ? ' bt' : ''), String(rr));
      this.rowH.push(h); put(h, r + 2, 1);
      const row = [];
      for (let c = 0; c < this.cols; c++) {
        const cc = c + this.colOff;
        const x = el('div', 'cell');
        x.dataset.base = 'cell k-' + kind +
          (this.bs > 1 && cc % this.bs === 0 && c > 0 ? ' bl' : '') +
          (this.bs > 1 && rr % this.bs === 0 && r > 0 ? ' bt' : '');
        x.className = x.dataset.base;
        if (opt.vals) x.textContent = opt.vals[rr][cc];
        row.push(x); put(x, r + 2, c + 2);
      }
      this.cells.push(row);
    }
    this.badges = [];
    this.root.appendChild(g);
  }
  // cls: map "r,c" (absolute indices) -> extra class string; text: optional function(r,c) -> string
  Mat.prototype.paint = function (cls, text) {
    for (let r = 0; r < this.rows; r++) for (let c = 0; c < this.cols; c++) {
      const rr = r + this.rowOff, cc = c + this.colOff, x = this.cells[r][c];
      const extra = cls[key(rr, cc)] || '';
      const want = x.dataset.base + (extra ? ' ' + extra : '');
      if (x.className !== want) x.className = want;
      if (text) { const t = text(rr, cc); if (x.textContent !== t) x.textContent = t; }
    }
  };
  Mat.prototype.heads = function (rowOn, colOn) {
    this.rowH.forEach((h, r) => h.classList.toggle('on', rowOn != null && rowOn.indexOf(r + this.rowOff) >= 0));
    this.colH.forEach((h, c) => h.classList.toggle('on', colOn != null && colOn.indexOf(c + this.colOff) >= 0));
  };
  // outline a rectangle of cells (absolute indices) with a label
  Mat.prototype.badge = function (r0, c0, rs, cs, text, cls) {
    const b = el('div', 'badge' + (cls ? ' ' + cls : ''), text ? '<span>' + text + '</span>' : '');
    b.style.gridRow = (2 + r0 - this.rowOff) + ' / span ' + rs;
    b.style.gridColumn = (2 + c0 - this.colOff) + ' / span ' + cs;
    this.grid.appendChild(b); this.badges.push(b); return b;
  };
  Mat.prototype.clearBadges = function () { this.badges.forEach((b) => b.remove()); this.badges = []; };

  function Triple(opt) {
    opt = opt || {};
    this.root = el('div', 'triple');
    const bs = opt.bs || 1;
    this.A = new Mat('A', { vals: A, bs });
    this.B = new Mat('B', { vals: B, bs });
    this.C = new Mat('C', { bs });
    const tx = el('div', 'tx', opt.note || 'C = A @ B<br>C[i][j] sits where<br>row i meets col j');
    this.root.append(tx, this.B.root, this.A.root, this.C.root);
  }
  function cText(s) { return (r, c) => (s.T[r][c] > 0 ? String(s.C[r][c]) : '·'); }
  function cClass(s, cls) {
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
      const t = s.T[r][c];
      const k0 = key(r, c);
      const add = t === K ? 'fin' : t > 0 ? 'part' : '';
      if (add) cls[k0] = cls[k0] ? cls[k0] + ' ' + add : add;
    }
  }
  function addCls(map, k0, c) { map[k0] = map[k0] ? map[k0] + ' ' + c : c; }

  // ---------------- controls interface shared by all widgets ----------------
  // Each widget implements: n (number of steps), t, go(t), innerSize(), render()
  const Steppable = {
    step() { if (this.t < this.n) { this.go(this.t + 1); return true; } return false; },
    back() { if (this.t > 0) { this.go(this.t - 1); return true; } return false; },
    inner() {
      if (this.t >= this.n) return false;
      const s = this.innerSize ? this.innerSize() : 1;
      let t = this.t + 1; while (t < this.n && t % s !== 0) t++;
      this.go(t); return true;
    },
    finish() { this.stop(); if (this.t < this.n) { this.go(this.n); return true; } return false; },
    reset() { this.stop(); this.go(0); },
    atEnd() { return this.t >= this.n; },
    atStart() { return this.t <= 0; },
    play() {
      if (this.timer) { this.stop(); return; }
      if (this.t >= this.n) this.go(0);
      const self = this;
      this.timer = setInterval(() => { if (!self.step()) self.stop(); }, this.speed || 320);
      this.syncPlay();
    },
    stop() { if (this.timer) { clearInterval(this.timer); this.timer = null; } this.syncPlay(); },
    syncPlay() { if (this.playBtn) { this.playBtn.textContent = this.timer ? 'Pause' : 'Play'; this.playBtn.setAttribute('aria-pressed', this.timer ? 'true' : 'false'); } },
    makeControls(opts) {
      opts = opts || {};
      const c = el('div', 'controls');
      c.append(
        btn('Reset', 'Reset to the start (R)', () => this.reset()),
        btn('◀', 'Step back (←)', () => { this.stop(); this.back(); }),
        btn('Step ▶', 'One step (→)', () => { this.stop(); this.step(); }, true)
      );
      if (!opts.noInner) c.append(btn(opts.innerLabel || 'Whole inner loop ⏭', 'Jump to the end of the current inner loop (↓)', () => { this.stop(); this.inner(); }));
      this.playBtn = btn('Play', 'Play / pause (P)', () => this.play());
      c.append(this.playBtn, btn('Finish', 'Jump to the end (E)', () => this.finish()));
      return c;
    },
    keydown(e) {
      const k = e.key;
      if (k === 'ArrowRight') { this.stop(); return this.step(); }
      if (k === 'ArrowLeft') { this.stop(); return this.back(); }
      if (k === 'ArrowDown') { this.stop(); return this.inner(); }
      if (k === 'p' || k === 'P') { this.play(); return true; }
      if (k === 'e' || k === 'E') { return this.finish(); }
      if (k === 'r' || k === 'R') { this.reset(); return true; }
      return false;
    },
    enableKeys() {
      this.root.tabIndex = 0;
      this.root.addEventListener('keydown', (e) => { if (e.target !== this.root) return; if (this.keydown(e)) e.preventDefault(); });
    }
  };
  function mix(P) { Object.keys(Steppable).forEach((k) => { if (!P[k]) P[k] = Steppable[k]; }); }

  function countersHTML(items) {
    return items.map((x) => '<span class="' + (x.cls || '') + '">' + x.label + ' <b>' + x.value + '</b></span>').join('');
  }
  function checkLine(C, extra) {
    return equalsRef(C) ? '<span class="ok">Check: C == A @ B ✓</span>' + (extra || '') : '<span class="tsp">C differs from A @ B ✗</span>';
  }

  // ---------------- code panel ----------------
  function codeLines(order, bs, ranges, fullR) {
    const vn = (v) => (bs > 1 ? BVAR[v] : v);
    const size = bs > 1 ? { i: '2', j: '2', k: '2' } : { i: 'M', j: 'N', k: 'K' };
    const lines = [];
    order.split('').forEach((v, d) => {
      const full = !ranges || !ranges[v] || ranges[v].length === fullR;
      const r = full ? 'range(' + size[v] + ')' : '(' + ranges[v].join(', ') + ')';
      lines.push({ v, html: '  '.repeat(d) + '<span class="kw">for</span> ' + vn(v) + ' <span class="kw">in</span> ' + r + ':' });
    });
    lines.push({ v: 'body', html: '      ' + (bs > 1
      ? '<span class="tc">C</span>[mt,nt] += <span class="ta">A</span>[mt,kt] @ <span class="tb">B</span>[kt,nt]'
      : '<span class="tc">C</span>[i][j] += <span class="ta">A</span>[i][k] * <span class="tb">B</span>[k][j]') });
    return lines;
  }

  // ---------------- Player: one loop order, scalar (bs=1) or tiled (bs=2) ----------------
  // cfg: order, bs, ranges, pre (ops applied before step 0), register, labels, selector, compact, controls, title, note
  function Player(root, cfg) {
    this.cfg = Object.assign({ order: 'ijk', bs: 1, labels: true, controls: true }, cfg || {});
    this.root = root; root.classList.add('w'); if (this.cfg.compact) root.classList.add('compact');
    this.speed = this.cfg.speed || (this.cfg.bs > 1 ? 700 : 300);
    this.t = 0;
    this.head = el('div', 'w-head'); root.appendChild(this.head);
    if (this.cfg.selector) {
      const seg = el('div', 'seg'); this.segBtns = {};
      (this.cfg.orders || ORDERS).forEach((o) => {
        const b = btn(this.cfg.bs > 1 ? o.split('').map((v) => BVAR[v]).join(' ') : o, 'Loop order ' + o, () => { this.stop(); this.setOrder(o); });
        this.segBtns[o] = b; seg.appendChild(b);
      });
      root.appendChild(seg);
    }
    const body = el('div', 'w-body'); root.appendChild(body);
    this.left = el('div', 'stack'); body.appendChild(this.left);
    this.codeEl = el('pre', 'code'); this.left.appendChild(this.codeEl);
    if (this.cfg.register) {
      this.regEl = el('div', 'reg', '<span>running sum<br><small>(one register)</small></span><span class="box">·</span>');
      this.left.appendChild(this.regEl);
    }
    this.tri = new Triple({ bs: this.cfg.bs, note: this.cfg.note });
    body.appendChild(this.tri.root);
    this.cap = el('div', 'cap'); root.appendChild(this.cap);
    const foot = el('div', 'w-foot'); root.appendChild(foot);
    this.cnt = el('div', 'counters'); foot.appendChild(this.cnt);
    if (this.cfg.controls) { foot.appendChild(this.makeControls()); this.enableKeys(); }
    this.setOrder(this.cfg.order);
  }
  Player.prototype.setOrder = function (order) {
    this.order = order;
    const mf = makeFrames(order, this.cfg.bs, this.cfg.ranges);
    this.frames = mf.frames; this.R = mf.R; this.v = mf.v; this.n = this.frames.length;
    this.lines = codeLines(order, this.cfg.bs, this.cfg.ranges, 4 / this.cfg.bs);
    this.codeEl.innerHTML = '';
    this.lineEls = this.lines.map((l) => { const d = el('div', 'ln', l.html); this.codeEl.appendChild(d); return d; });
    this.valEls = this.lines.slice(0, 3).map((l, d) => { const s = el('span', 'val'); this.lineEls[d].appendChild(s); return s; });
    if (this.segBtns) Object.keys(this.segBtns).forEach((o) => this.segBtns[o].setAttribute('aria-pressed', o === order ? 'true' : 'false'));
    this.renderHead();
    this.go(0);
  };
  Player.prototype.innerSize = function () { return this.R[this.v[2]].length; };
  Player.prototype.renderHead = function () {
    this.head.innerHTML = '';
    if (this.cfg.title) this.head.appendChild(el('span', 'w-title', this.cfg.title));
    const inner = this.v[2], stay = STAY[inner], bsv = this.cfg.bs > 1;
    const vn = (x) => (bsv ? BVAR[x] : x);
    this.head.appendChild(el('span', 'chip', 'order ' + this.v.map(vn).join(' ')));
    if (this.cfg.labels) {
      const sIdx = IDX[stay].map(vn);
      this.head.appendChild(el('span', 'chip', 'innermost ' + vn(inner)));
      this.head.appendChild(el('span', 'chip ' + stay, 'stays put: ' + stay + '[' + sIdx.join(bsv ? ',' : '][') + '] · ' + SHORT[stay]));
      this.head.appendChild(el('span', 'chip', FORM[this.order]));
    }
  };
  Player.prototype.go = function (t) {
    this.t = Math.max(0, Math.min(this.n, t));
    this.render();
  };
  Player.prototype.render = function () {
    const s = newState(); const bs = this.cfg.bs;
    (this.cfg.pre || []).forEach((op) => apply(s, op));
    let last = [];
    for (let f = 0; f < this.t; f++) { last = this.frames[f].ops.map((op) => apply(s, op)); }
    const fi = this.t > 0 ? this.t - 1 : 0;
    const fr = this.frames[fi];
    const live = this.t > 0 && this.t <= this.n;
    const clsA = {}, clsB = {}, clsC = {};
    // region swept by the inner loop (outer two indices fixed)
    const inner = this.v[2], stay = STAY[inner];
    const region = (op) => {
      const out = [];
      const vals = IDX[op].indexOf(inner) >= 0 ? this.R[inner] : [null];
      vals.forEach((x) => {
        const id = Object.assign({}, fr.idx); if (x !== null) id[inner] = x;
        const r0 = id[IDX[op][0]] * bs, c0 = id[IDX[op][1]] * bs;
        for (let a = 0; a < bs; a++) for (let b = 0; b < bs; b++) out.push(key(r0 + a, c0 + b));
      });
      return out;
    };
    const showCtx = this.t < this.n;
    if (showCtx) {
      region('A').forEach((k0) => addCls(clsA, k0, 'sweep'));
      region('B').forEach((k0) => addCls(clsB, k0, 'sweep'));
      region('C').forEach((k0) => addCls(clsC, k0, 'sweep'));
      const stayMap = { A: clsA, B: clsB, C: clsC }[stay];
      region(stay).forEach((k0) => addCls(stayMap, k0, 'stay'));
    }
    cClass(s, clsC);
    if (live) last.forEach((o) => { addCls(clsA, key(o.i, o.k), 'read'); addCls(clsB, key(o.k, o.j), 'read'); addCls(clsC, key(o.i, o.j), 'write'); });
    this.tri.A.paint(clsA); this.tri.B.paint(clsB); this.tri.C.paint(clsC, cText(s));
    // headers: current indices
    if (showCtx) {
      const rowsOf = (v) => rng(bs).map((a) => fr.idx[v] * bs + a);
      this.tri.A.heads(rowsOf('i'), rowsOf('k')); this.tri.B.heads(rowsOf('k'), rowsOf('j')); this.tri.C.heads(rowsOf('i'), rowsOf('j'));
    } else { ['A', 'B', 'C'].forEach((m) => this.tri[m].heads(null, null)); }
    // code
    this.lineEls.forEach((d, n) => { d.classList.toggle('on', n === 3 && live && showCtx); d.classList.toggle('inner', n === 2 && showCtx); });
    const vn = (x) => (bs > 1 ? BVAR[x] : x);
    this.valEls.forEach((sp, d) => { sp.textContent = showCtx && this.t > 0 ? vn(this.v[d]) + ' = ' + fr.idx[this.v[d]] : ''; });
    // register
    if (this.regEl) {
      const box = this.regEl.querySelector('.box');
      box.textContent = live && last.length ? String(last[last.length - 1].nw) : '·';
    }
    // caption
    const cs = census(s.T);
    let cap = '';
    if (this.t === 0) cap = this.cfg.startCap || 'Press <b>Step</b> (or →) to take the first step. Row, column and cell that the inner loop will visit are tinted.';
    else if (bs === 1) {
      const o = last[0];
      const name = 'C[' + o.i + '][' + o.j + ']';
      cap = '<code><span class="tc">' + name + '</span> += <span class="ta">A[' + o.i + '][' + o.k + ']</span>·<span class="tb">B[' + o.k + '][' + o.j + ']</span></code> = ' + o.a + '·' + o.b + ' = ' + o.p + ' → ' +
        (o.first ? name + ' = ' + o.nw + ' <small>(first term: just stored, no addition)</small>' : name + ' = ' + o.old + ' + ' + o.p + ' = <b>' + o.nw + '</b>') +
        ' · term ' + o.terms + ' of 4 ' + (o.terms === 4 ? '<b class="tc">— finished</b>' : '<span class="tc">— partial</span>');
    } else {
      const id = fr.idx;
      const done = rng(2).every((a) => rng(2).every((b) => s.T[id.i * 2 + a][id.j * 2 + b] === K));
      cap = '<code><span class="tc">C[' + id.i + ',' + id.j + ']</span> += <span class="ta">A[' + id.i + ',' + id.k + ']</span> @ <span class="tb">B[' + id.k + ',' + id.j + ']</span></code> — one 2×2 block product = 8 multiplies. ' +
        (done ? '<b class="tc">Block C[' + id.i + ',' + id.j + '] finished.</b>' : '<span class="tc">Block C[' + id.i + ',' + id.j + '] still partial.</span>');
    }
    if (this.t === this.n) {
      const cb = this.cfg.checkBlock;
      if (cb) {
        const ok = rng(cb[2]).every((a) => rng(cb[2]).every((b) => s.C[cb[0] + a][cb[1] + b] === REF[cb[0] + a][cb[1] + b] && s.T[cb[0] + a][cb[1] + b] === K));
        const vals = rng(cb[2]).map((a) => rng(cb[2]).map((b) => s.C[cb[0] + a][cb[1] + b]).join(' ')).join(' / ');
        cap = 'Done: block C[0,0] = [' + vals + '] ' + (ok ? '<span class="ok">= the top-left block of A @ B ✓</span>' : '<span class="tsp">✗</span>') +
          ' The other 7 block products are the same 8-multiply pattern; together they make the full 64 multiplies and the same C.';
      } else cap = 'Done: ' + s.mul + ' multiplies, ' + s.add + ' additions. ' + checkLine(s.C);
    }
    this.cap.innerHTML = cap;
    this.cnt.innerHTML = countersHTML([
      { label: 'step', value: this.t + '/' + this.n },
      { label: 'multiplies', value: s.mul + '/64' },
      { label: 'additions', value: s.add + '/48' },
      { label: 'partial', value: cs.p },
      { label: 'finished', value: cs.f + '/16' }
    ]);
  };
  mix(Player.prototype);

  // ---------------- Group: several players stepped together ----------------
  function Group(root, cfg) {
    this.root = root;
    root.classList.add('stack');
    const grid = el('div', 'pgroup'); root.appendChild(grid);
    this.players = cfg.players.map((pc) => { const d = el('div'); grid.appendChild(d); return new Player(d, Object.assign({ compact: true, controls: false }, pc)); });
    this.n = Math.max.apply(null, this.players.map((p) => p.n));
    this.t = 0; this.speed = cfg.speed || 300;
    const bar = el('div', 'w-foot'); bar.appendChild(el('div', 'counters', cfg.hint || 'All panels take the same step at the same time.'));
    bar.appendChild(this.makeControls()); root.appendChild(bar);
    this.enableKeys();
  }
  Group.prototype.go = function (t) { this.t = Math.max(0, Math.min(this.n, t)); this.players.forEach((p) => p.go(this.t)); };
  Group.prototype.innerSize = function () { return this.players[0].innerSize(); };
  mix(Group.prototype);

  // ---------------- Hardware 1: scalar × scalar (CPU core) ----------------
  function HWScalar(root, cfg) {
    this.root = root; root.classList.add('w'); this.speed = 260;
    root.appendChild(el('div', 'w-head', '<span class="w-title">CPU core</span><span class="chip">one multiply per step</span>'));
    const body = el('div', 'w-body'); root.appendChild(body);
    this.pic = el('div', 'hw'); body.appendChild(this.pic);
    this.mat = new Mat('C', { cls: 'compact' }); body.appendChild(this.mat.root);
    this.cap = el('div', 'cap'); root.appendChild(this.cap);
    const foot = el('div', 'w-foot'); root.appendChild(foot);
    this.cnt = el('div', 'counters'); foot.appendChild(this.cnt);
    foot.appendChild(this.makeControls({ innerLabel: 'Next C value ⏭' })); this.enableKeys();
    this.frames = makeFrames('ijk').frames; this.n = 64; this.go(0);
  }
  HWScalar.prototype.innerSize = function () { return 4; };
  HWScalar.prototype.go = function (t) {
    this.t = Math.max(0, Math.min(this.n, t));
    const s = newState(); let o = null;
    for (let f = 0; f < this.t; f++) o = apply(s, this.frames[f].ops[0]);
    const a = o ? o.a : '·', b = o ? o.b : '·', p = o ? o.p : '·';
    this.pic.innerHTML =
      '<span class="pill A">A ' + (o ? '[' + o.i + '][' + o.k + ']=' + a : '') + '</span><span class="node">×</span><span class="pill B">B ' + (o ? '[' + o.k + '][' + o.j + ']=' + b : '') + '</span>' +
      '<span class="arrow">→</span><span class="pill Cp">' + p + '</span><span class="node">+</span><span class="arrow">→</span>' +
      '<span class="pill ' + (o && o.terms === 4 ? 'C' : 'Cp') + '">C' + (o ? '[' + o.i + '][' + o.j + '] = ' + o.nw : '') + '</span>';
    const cls = {}; cClass(s, cls); if (o) addCls(cls, key(o.i, o.j), 'write');
    this.mat.paint(cls, cText(s));
    this.cap.innerHTML = this.t === this.n ? 'After 64 steps: ' + checkLine(s.C) : 'One multiply and one add per step. Real size: <b>1×1</b> — so the whole 4×4 product takes 64 steps (a 512×512×512 one takes 134 million).';
    this.cnt.innerHTML = countersHTML([{ label: 'steps', value: this.t + '/64' }, { label: 'multiplies', value: s.mul + '/64' }, { label: 'additions', value: s.add + '/48' }]);
  };
  mix(HWScalar.prototype);

  // ---------------- Hardware 2: vector × matrix (in-memory compute array) ----------------
  function HWIMC(root, cfg) {
    this.root = root; root.classList.add('w'); this.speed = 900;
    root.appendChild(el('div', 'w-head', '<span class="w-title">In-memory-compute array</span><span class="chip">one row of A in → one row of C out</span><span class="chip B">B stays in the memory cells · ws</span>'));
    const body = el('div', 'w-body'); root.appendChild(body);
    this.arr = el('div', 'imc'); body.appendChild(this.arr);
    this.arr.style.gridTemplateColumns = 'auto repeat(4, var(--cell))';
    this.arr.appendChild(el('div', 'lab', 'A[i][k] in'));
    for (let j = 0; j < 4; j++) this.arr.appendChild(el('div', 'lab', 'col j=' + j));
    this.drv = []; this.cells = [];
    for (let k = 0; k < 4; k++) {
      const d = el('div', 'pill A drv', '·'); this.drv.push(d); this.arr.appendChild(d);
      const row = [];
      for (let j = 0; j < 4; j++) { const c = el('div', 'cell k-B stay', String(B[k][j])); row.push(c); this.arr.appendChild(c); }
      this.cells.push(row);
    }
    this.arr.appendChild(el('div', 'lab', 'C[i][j] out'));
    this.out = [];
    for (let j = 0; j < 4; j++) { const c = el('div', 'cell k-C', '·'); this.out.push(c); this.arr.appendChild(c); }
    this.mat = new Mat('C', { cls: 'compact' }); body.appendChild(this.mat.root);
    this.cap = el('div', 'cap'); root.appendChild(this.cap);
    const foot = el('div', 'w-foot'); root.appendChild(foot);
    this.cnt = el('div', 'counters'); foot.appendChild(this.cnt);
    foot.appendChild(this.makeControls({ noInner: true })); this.enableKeys();
    this.n = 4; this.go(0);
  }
  HWIMC.prototype.go = function (t) {
    this.t = Math.max(0, Math.min(this.n, t));
    const s = newState();
    for (let i = 0; i < this.t; i++) for (let k = 0; k < 4; k++) for (let j = 0; j < 4; j++) apply(s, [i, j, k]);
    const i = this.t - 1;
    this.drv.forEach((d, k) => { d.textContent = i >= 0 && this.t <= this.n ? String(A[i][k]) : '·'; });
    this.cells.forEach((row) => row.forEach((c) => { c.className = 'cell k-B stay' + (i >= 0 ? ' read' : ''); }));
    this.out.forEach((c, j) => { c.className = 'cell k-C' + (i >= 0 ? ' fin' : ''); c.textContent = i >= 0 ? String(s.C[i][j]) : '·'; });
    const cls = {}; cClass(s, cls); if (i >= 0) for (let j = 0; j < 4; j++) addCls(cls, key(i, j), 'write');
    this.mat.paint(cls, cText(s));
    this.cap.innerHTML = (this.t === 0 ? 'The 16 values of B are written <b>into</b> the memory cells once and stay there. Step to push rows of A through.' :
      'Step ' + this.t + ': row ' + i + ' of A drives the 4 rows of cells (one per k); every cell multiplies; every column adds its 4 products → row ' + i + ' of C, finished because all of K fits in the array. ') +
      (this.t === this.n ? checkLine(s.C) : '') +
      '<br><small>Tiny version: 4×4 cells, 16 multiplies per step. Real (the Metis figure you gave): a 512×512 block of B, 262,144 multiplies per step. When K is larger than the array, each output row is only partial and must be added up over several steps.</small>';
    this.cnt.innerHTML = countersHTML([{ label: 'steps', value: this.t + '/4' }, { label: 'multiplies', value: s.mul + '/64' }, { label: 'additions', value: s.add + '/48' }]);
  };
  mix(HWIMC.prototype);

  // ---------------- Hardware 3: matrix × matrix (tensor core) ----------------
  function HWTensor(root, cfg) {
    this.root = root; root.classList.add('w'); this.speed = 800;
    root.appendChild(el('div', 'w-head', '<span class="w-title">Tensor core</span><span class="chip">one block of A × one block of B per step</span>'));
    const body = el('div', 'w-body'); root.appendChild(body);
    this.unit = el('div', 'unit'); body.appendChild(this.unit);
    this.mat = new Mat('C', { cls: 'compact', bs: 2 }); body.appendChild(this.mat.root);
    this.cap = el('div', 'cap'); root.appendChild(this.cap);
    const foot = el('div', 'w-foot'); root.appendChild(foot);
    this.cnt = el('div', 'counters'); foot.appendChild(this.cnt);
    foot.appendChild(this.makeControls({ noInner: true })); this.enableKeys();
    this.frames = makeFrames('ijk', 2).frames; this.n = 8; this.go(0);
  }
  HWTensor.prototype.go = function (t) {
    this.t = Math.max(0, Math.min(this.n, t));
    const s = newState();
    for (let f = 0; f < this.t; f++) this.frames[f].ops.forEach((op) => apply(s, op));
    const fr = this.frames[Math.max(0, this.t - 1)].idx;
    const on = this.t > 0;
    const blk = (kind, r0, c0, vals) => {
      const m = el('div', 'mini'); m.style.gridTemplateColumns = 'repeat(2, var(--cell))';
      for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) {
        const c = el('div', 'cell k-' + kind + (on && kind !== 'C' ? ' read' : ''));
        if (kind === 'C') { const tt = s.T[r0 + a][c0 + b]; c.classList.add(tt === K ? 'fin' : tt > 0 ? 'part' : 'x'); c.textContent = tt ? s.C[r0 + a][c0 + b] : '·'; }
        else c.textContent = on ? vals[r0 + a][c0 + b] : '·';
        m.appendChild(c);
      }
      return m;
    };
    this.unit.innerHTML = '';
    this.unit.append(el('span', 'unit-lab', 'tensor<br>unit'), blk('A', fr.i * 2, fr.k * 2, A), el('span', 'arrow', '@'), blk('B', fr.k * 2, fr.j * 2, B), el('span', 'arrow', '→ +='), blk('C', fr.i * 2, fr.j * 2));
    const cls = {}; cClass(s, cls);
    if (on) for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) addCls(cls, key(fr.i * 2 + a, fr.j * 2 + b), 'write');
    this.mat.paint(cls, cText(s));
    this.cap.innerHTML = (on ? 'Step ' + this.t + ': <code>C[' + fr.i + ',' + fr.j + '] += A[' + fr.i + ',' + fr.k + '] @ B[' + fr.k + ',' + fr.j + ']</code> — 8 multiplies at once. ' : 'Each step multiplies a whole block of A by a whole block of B. ') +
      (this.t === this.n ? checkLine(s.C) : '') +
      '<br><small>Tiny version: 2×2 @ 2×2 = 8 multiplies per step, 8 steps. Real: the 16×16 @ 16×16 operation a GPU program issues (e.g. CUDA WMMA m16n16k16) = 4,096 multiplies; the hardware finishes it over a few clock cycles.</small>';
    this.cnt.innerHTML = countersHTML([{ label: 'steps', value: this.t + '/8' }, { label: 'multiplies', value: s.mul + '/64' }, { label: 'additions', value: s.add + '/48' }]);
  };
  mix(HWTensor.prototype);

  // ---------------- Grid vs sweep on 4 cores (block level) ----------------
  const GMODES = {
    os: { grid: ['i', 'j'], sweep: 'k', owner: 'C', label: 'os · cores own C blocks' },
    ws: { grid: ['k', 'j'], sweep: 'i', owner: 'B', label: 'ws · cores keep B blocks' },
    is: { grid: ['i', 'k'], sweep: 'j', owner: 'A', label: 'is · cores keep A blocks' }
  };
  function gridSim(mode, t) {
    const G = GMODES[mode];
    const cores = [];
    for (let g0 = 0; g0 < 2; g0++) for (let g1 = 0; g1 < 2; g1++) { const g = {}; g[G.grid[0]] = g0; g[G.grid[1]] = g1; cores.push({ id: cores.length, g }); }
    const n = mode === 'os' ? 2 : 3;
    const s = newState(); s.tadd = 0; s.sadd = 0;
    const buf = cores.map(() => ({}));
    const jobsAt = (st) => cores.map((c) => { const id = Object.assign({}, c.g); id[G.sweep] = st; return id; });
    let reduced = false;
    for (let f = 0; f < t; f++) {
      if (f === 2) { // reduce across cores (space)
        for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
          const parts = buf.map((b) => b[key(i, j)]).filter(Boolean);
          s.C[i][j] = parts.reduce((x, p) => x + p.v, 0); s.T[i][j] = K; s.sadd += parts.length - 1;
        }
        reduced = true; continue;
      }
      jobsAt(f).forEach((id, ci) => {
        for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) for (let c = 0; c < 2; c++) {
          const i = id.i * 2 + a, j = id.j * 2 + b, k = id.k * 2 + c, p = A[i][k] * B[k][j];
          s.mul++;
          const bk = key(i, j);
          if (!buf[ci][bk]) buf[ci][bk] = { v: 0, T: 0 };
          if (buf[ci][bk].T > 0) s.tadd++;
          buf[ci][bk].v += p; buf[ci][bk].T++;
          if (mode === 'os') { s.C[i][j] = buf[ci][bk].v; s.T[i][j] = buf[ci][bk].T; }
        }
      });
    }
    s.add = s.tadd + s.sadd;
    return { G, cores, n, s, buf, jobsAt, reduced };
  }
  function GridSweep(root, cfg) {
    this.root = root; root.classList.add('w'); this.speed = 1100;
    this.mode = (cfg && cfg.mode) || 'os';
    this.head = el('div', 'w-head'); root.appendChild(this.head);
    if (!cfg || cfg.tabs !== false) {
      const seg = el('div', 'seg'); this.tabs = {};
      ((cfg && cfg.modes) || ['os', 'ws', 'is']).forEach((m) => { const b = btn(GMODES[m].label, m, () => { this.stop(); this.setMode(m); }); this.tabs[m] = b; seg.appendChild(b); });
      root.appendChild(seg);
    }
    const body = el('div', 'w-body'); root.appendChild(body);
    this.tri = new Triple({ bs: 2, note: 'block view<br>4 cores' });
    body.appendChild(this.tri.root);
    const right = el('div', 'stack'); right.style.flex = '1 1 18rem'; body.appendChild(right);
    this.coresEl = el('div', 'cores'); right.appendChild(this.coresEl);
    this.tlEl = el('div', 'tl-wrap'); right.appendChild(this.tlEl);
    this.cap = el('div', 'cap'); root.appendChild(this.cap);
    const foot = el('div', 'w-foot'); root.appendChild(foot);
    this.cnt = el('div', 'counters'); foot.appendChild(this.cnt);
    foot.appendChild(this.makeControls({ noInner: true })); this.enableKeys();
    this.setMode(this.mode);
  }
  GridSweep.prototype.setMode = function (m) {
    this.mode = m;
    if (this.tabs) Object.keys(this.tabs).forEach((k) => this.tabs[k].setAttribute('aria-pressed', k === m ? 'true' : 'false'));
    const G = GMODES[m];
    this.head.innerHTML = '<span class="w-title">4 cores</span><span class="chip ' + G.owner + '">stays put in each core: ' + G.owner + ' block · ' + m + '</span><span class="chip">shared out over cores: ' + G.grid.map((v) => BVAR[v]).join(' × ') + '</span><span class="chip">in order on each core: ' + BVAR[G.sweep] + '</span>';
    ['A', 'B', 'C'].forEach((x) => this.tri[x].clearBadges());
    const sim = gridSim(m, 0);
    this.n = sim.n;
    sim.cores.forEach((c) => {
      const r0 = c.g[IDX[G.owner][0]] * 2, c0 = c.g[IDX[G.owner][1]] * 2;
      this.tri[G.owner].badge(r0, c0, 2, 2, 'core ' + c.id);
    });
    this.go(0);
  };
  GridSweep.prototype.go = function (t) {
    this.t = Math.max(0, Math.min(this.n, t));
    const sim = gridSim(this.mode, this.t); const G = sim.G, s = sim.s;
    const clsA = {}, clsB = {}, clsC = {};
    const stayMap = { A: clsA, B: clsB, C: clsC }[G.owner];
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) addCls(stayMap, key(r, c), 'stay');
    const cur = this.t - 1;
    const isRed = this.mode !== 'os' && cur === 2;
    if (cur >= 0 && !isRed) {
      sim.jobsAt(cur).forEach((id) => {
        for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) {
          addCls(clsA, key(id.i * 2 + a, id.k * 2 + b), 'read'); addCls(clsB, key(id.k * 2 + a, id.j * 2 + b), 'read');
          if (this.mode === 'os') addCls(clsC, key(id.i * 2 + a, id.j * 2 + b), 'write');
        }
      });
    }
    cClass(s, clsC);
    if (isRed) for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) addCls(clsC, key(r, c), 'spadd');
    this.tri.A.paint(clsA); this.tri.B.paint(clsB); this.tri.C.paint(clsC, cText(s));
    // core panels
    this.coresEl.innerHTML = '';
    sim.cores.forEach((c, ci) => {
      const box = el('div', 'core' + (sim.reduced ? ' spent' : ''));
      const own = G.owner + '[' + c.g[IDX[G.owner][0]] + ',' + c.g[IDX[G.owner][1]] + ']';
      box.appendChild(el('h5', null, '<span>core ' + c.id + '</span><span class="t' + G.owner.toLowerCase() + '">' + own + '</span>'));
      let op = '';
      if (cur >= 0 && !isRed) { const id = sim.jobsAt(cur)[ci]; op = 'A[' + id.i + ',' + id.k + ']@B[' + id.k + ',' + id.j + '] → C[' + id.i + ',' + id.j + ']'; }
      if (isRed) op = 'sends its partials →';
      box.appendChild(el('div', 'op', op || 'waiting'));
      // buffer extent
      let rows, cols;
      if (this.mode === 'os') { rows = [c.g.i * 2, c.g.i * 2 + 1]; cols = [c.g.j * 2, c.g.j * 2 + 1]; }
      else if (this.mode === 'ws') { rows = [0, 1, 2, 3]; cols = [c.g.j * 2, c.g.j * 2 + 1]; }
      else { rows = [c.g.i * 2, c.g.i * 2 + 1]; cols = [0, 1, 2, 3]; }
      const mini = el('div', 'mini'); mini.style.gridTemplateColumns = 'repeat(' + cols.length + ', var(--cell))';
      rows.forEach((r) => cols.forEach((cc) => {
        const b = sim.buf[ci][key(r, cc)];
        const x = el('div', 'cell k-C' + (b ? (this.mode === 'os' && b.T === K ? ' fin' : ' part') : ''), b ? String(b.v) : '·');
        mini.appendChild(x);
      }));
      box.appendChild(el('div', 'op', this.mode === 'os' ? 'its C block (final home):' : 'its partial sums of C:'));
      box.appendChild(mini);
      this.coresEl.appendChild(box);
    });
    // timeline
    const cols = this.mode === 'os' ? [0, 1] : [0, 1, 'red'];
    let h = '<table class="tl"><tr><th></th>' + cols.map((c) => '<th>' + (c === 'red' ? 'add across cores' : 'time ' + c + ' (' + BVAR[G.sweep] + '=' + c + ')') + '</th>').join('') + '</tr>';
    sim.cores.forEach((c, ci) => {
      h += '<tr><th>core ' + c.id + '</th>' + cols.map((col, n) => {
        const st = n === cur ? ' now' : n < cur ? ' done' : '';
        if (col === 'red') return '<td class="red' + st + '">' + (this.mode === 'ws' ? (c.g.k === 0 ? '+ core ' + (ci + 2) : '↑') : (c.g.k === 0 ? '+ core ' + (ci + 1) : '←')) + '</td>';
        const id = sim.jobsAt(col)[ci];
        return '<td class="' + st.trim() + '">A' + id.i + id.k + '·B' + id.k + id.j + '→C' + id.i + id.j + '</td>';
      }).join('') + '</tr>';
    });
    this.tlEl.innerHTML = h + '</table>';
    // caption
    const texts = {
      os: ['Each core owns one block of C. At every time step all 4 cores work at the same time (the <b>grid</b>); each walks through kt in order (the <b>sweep</b>).', 'Time 0: every core adds its first block product into its own C block — partial.', 'Time 1: the second block product lands in the same place — every C block finished, no core ever talked to another. '],
      ws: ['Each core keeps one block of B. All 4 cores run at once; each sweeps down the block rows of A (mt = 0, 1).', 'Time 0: core (kt, nt) makes the kt-th piece of C[0, nt]. The other piece of the same block is being made on a different core.', 'Time 1: same for C[1, nt]. Every C block now exists as two pieces on two cores.', '<b class="tsp">Extra step:</b> the two pieces of every C block are added across cores. These 16 additions happen in <b>space</b>. '],
      is: ['Each core keeps one block of A. All 4 cores run at once; each sweeps along the block columns of B (nt = 0, 1).', 'Time 0: core (mt, kt) makes the kt-th piece of C[mt, 0].', 'Time 1: same for C[mt, 1]. Every C block exists as two pieces on two cores.', '<b class="tsp">Extra step:</b> the pieces are added across cores — 16 additions in <b>space</b>. ']
    };
    this.cap.innerHTML = texts[this.mode][this.t] + (this.t === this.n ? checkLine(s.C) : '');
    this.cnt.innerHTML = countersHTML([
      { label: 'time step', value: this.t + '/' + this.n },
      { label: 'multiplies', value: s.mul + '/64' },
      { label: 'additions over time', value: s.tadd },
      { label: 'additions across cores', value: s.sadd, cls: 'csp' },
      { label: 'total additions', value: s.add + '/48' }
    ]);
  };
  mix(GridSweep.prototype);

  // ---------------- Waves: more tiles than cores ----------------
  function Waves(root, cfg) {
    this.root = root; root.classList.add('w'); this.speed = 1000;
    this.cores = (cfg && cfg.cores) || 4;
    this.head = el('div', 'w-head'); root.appendChild(this.head);
    const seg = el('div', 'seg'); this.cb = {};
    [2, 3, 4, 5, 8].forEach((n) => { const b = btn(n + ' cores', n + ' cores', () => { this.stop(); this.setCores(n); }); this.cb[n] = b; seg.appendChild(b); });
    root.appendChild(seg);
    const body = el('div', 'w-body'); root.appendChild(body);
    this.mat = new Mat('C', {}); body.appendChild(this.mat.root);
    for (let i = 0; i < 4; i++) for (let jb = 0; jb < 2; jb++) this.mat.badge(i, jb * 2, 1, 2, 'T' + (i * 2 + jb), 'soft');
    this.tlEl = el('div', 'tl-wrap'); body.appendChild(this.tlEl);
    this.cap = el('div', 'cap'); root.appendChild(this.cap);
    const foot = el('div', 'w-foot'); root.appendChild(foot);
    this.cnt = el('div', 'counters'); foot.appendChild(this.cnt);
    foot.appendChild(this.makeControls({ noInner: true })); this.enableKeys();
    this.setCores(this.cores);
  }
  Waves.prototype.setCores = function (n) {
    this.cores = n; this.n = Math.ceil(8 / n);
    Object.keys(this.cb).forEach((k) => this.cb[k].setAttribute('aria-pressed', +k === n ? 'true' : 'false'));
    this.head.innerHTML = '<span class="w-title">Waves</span><span class="chip">8 output tiles (1×2 each, os)</span><span class="chip">' + n + ' cores → ' + this.n + ' wave' + (this.n > 1 ? 's' : '') + '</span>';
    this.go(0);
  };
  Waves.prototype.go = function (t) {
    this.t = Math.max(0, Math.min(this.n, t));
    const s = newState(); const n = this.cores;
    const doneTiles = Math.min(8, this.t * n);
    for (let tile = 0; tile < doneTiles; tile++) {
      const i = tile >> 1, jb = tile & 1;
      for (let j = jb * 2; j < jb * 2 + 2; j++) for (let k = 0; k < 4; k++) apply(s, [i, j, k]);
    }
    const cls = {}; cClass(s, cls);
    const curStart = (this.t - 1) * n;
    if (this.t > 0) for (let tile = curStart; tile < Math.min(8, curStart + n); tile++) { const i = tile >> 1, jb = tile & 1; addCls(cls, key(i, jb * 2), 'write'); addCls(cls, key(i, jb * 2 + 1), 'write'); }
    this.mat.paint(cls, cText(s));
    let h = '<table class="tl"><tr><th></th>' + rng(n).map((c) => '<th>core ' + c + '</th>').join('') + '</tr>';
    for (let w = 0; w < this.n; w++) {
      h += '<tr><th>wave ' + (w + 1) + '</th>' + rng(n).map((c) => {
        const tile = w * n + c; const st = w === this.t - 1 ? 'now' : w < this.t - 1 ? 'done' : '';
        return tile < 8 ? '<td class="' + st + '">T' + tile + '</td>' : '<td class="idle">idle</td>';
      }).join('') + '</tr>';
    }
    this.tlEl.innerHTML = h + '</table>';
    const idle = this.n * n - 8, util = Math.round(800 / (this.n * n));
    this.cap.innerHTML = 'Each tile is a whole output tile swept over K on one core (8 multiplies). The cores take tiles in <b>waves</b>. ' +
      (idle ? 'The last wave is only partly full: <b>' + idle + ' core' + (idle > 1 ? 's' : '') + ' idle</b>. ' : 'Every wave is full, no core waits. ') +
      (this.t === this.n ? checkLine(s.C) : '');
    this.cnt.innerHTML = countersHTML([{ label: 'wave', value: this.t + '/' + this.n }, { label: 'multiplies', value: s.mul + '/64' }, { label: 'additions', value: s.add + '/48' }, { label: 'core utilisation', value: util + '%' }]);
  };
  mix(Waves.prototype);

  // ---------------- K-groups: where the K-sum happens ----------------
  function kgPlan(g) {
    const per = K / g; const levels = Math.log2(g);
    return { per, levels, n: per + levels };
  }
  function KGroups(root, cfg) {
    this.root = root; root.classList.add('w'); this.speed = 900;
    this.g = (cfg && cfg.g) || 2;
    this.head = el('div', 'w-head'); root.appendChild(this.head);
    const seg = el('div', 'seg'); this.gb = {};
    [[1, 'all K on one core'], [2, 'K-groups: g = 2'], [4, 'every k on its own core']].forEach(([g, lab]) => { const b = btn(lab, lab, () => { this.stop(); this.setG(g); }); this.gb[g] = b; seg.appendChild(b); });
    root.appendChild(seg);
    this.coresEl = el('div', 'cores'); root.appendChild(this.coresEl);
    this.redEl = el('div', 'red-row'); root.appendChild(this.redEl);
    this.cap = el('div', 'cap'); root.appendChild(this.cap);
    this.tab = el('div', 'tl-wrap'); root.appendChild(this.tab);
    const foot = el('div', 'w-foot'); root.appendChild(foot);
    this.cnt = el('div', 'counters'); foot.appendChild(this.cnt);
    foot.appendChild(this.makeControls({ noInner: true })); this.enableKeys();
    this.setG(this.g);
  }
  KGroups.prototype.setG = function (g) {
    this.g = g; const P = kgPlan(g); this.n = P.n;
    Object.keys(this.gb).forEach((k) => this.gb[k].setAttribute('aria-pressed', +k === g ? 'true' : 'false'));
    this.head.innerHTML = '<span class="w-title">One output, C[0][0] = 13</span><span class="chip">K = 4 terms</span><span class="chip">' + g + ' core' + (g > 1 ? 's' : '') + ', ' + P.per + ' term' + (P.per > 1 ? 's' : '') + ' each</span>';
    this.go(0);
  };
  KGroups.prototype.go = function (t) {
    this.t = Math.max(0, Math.min(this.n, t));
    const g = this.g, P = kgPlan(g);
    const terms = rng(K).map((k) => ({ k, a: A[0][k], b: B[k][0], p: A[0][k] * B[k][0] }));
    const local = Math.min(this.t, P.per);
    let mul = 0, tadd = 0, sadd = 0;
    this.coresEl.innerHTML = '';
    const sums = [];
    for (let c = 0; c < g; c++) {
      const box = el('div', 'core');
      const mine = terms.slice(c * P.per, (c + 1) * P.per);
      box.appendChild(el('h5', null, '<span>core ' + c + '</span><span class="mono">k = ' + mine.map((x) => x.k).join(',') + '</span>'));
      const tw = el('div', 'terms');
      let sum = 0;
      mine.forEach((x, n) => {
        const done = n < local; if (done) { mul++; if (n > 0) tadd++; sum += x.p; }
        tw.appendChild(el('div', 'term' + (done ? ' done' : '') + (n === local - 1 && this.t <= P.per ? ' now' : ''), '<span class="ta">' + x.a + '</span>·<span class="tb">' + x.b + '</span> = ' + x.p));
      });
      box.appendChild(tw);
      box.appendChild(el('div', 'op', 'local sum (over time): <b class="tc">' + (local ? sum : '·') + '</b>'));
      sums.push(sum);
      this.coresEl.appendChild(box);
    }
    // reduction tree
    const lvDone = Math.max(0, this.t - P.per);
    let row = sums.slice(); const parts = [];
    for (let L = 0; L < P.levels; L++) {
      const nxt = [];
      for (let q = 0; q < row.length; q += 2) nxt.push(row[q] + row[q + 1]);
      const done = L < lvDone;
      if (done) sadd += nxt.length;
      parts.push('<span class="red-node' + (done ? '' : ' todo') + '">level ' + (L + 1) + ': ' + nxt.map((x, q) => (done ? row[2 * q] + ' + ' + row[2 * q + 1] + ' = ' + x : '+')).join(' · ') + '</span>');
      row = nxt;
    }
    const finished = this.t === this.n;
    this.redEl.innerHTML = (P.levels ? '<span class="tsp">add across cores:</span> ' + parts.join('<span class="arrow">→</span>') + '<span class="arrow">→</span>' : '<span class="muted">no additions across cores needed</span><span class="arrow">→</span>') +
      '<span class="pill ' + (finished ? 'C' : 'Cp') + '">C[0][0] = ' + (finished ? row[0] : '…') + '</span>';
    const texts = g === 1 ? 'One core walks through all 4 terms, adding into its own register: 3 additions <b>over time</b>, nothing to send anywhere. This is one array column that holds all of K.'
      : g === 4 ? 'Four cores each make one term at the same time. Nothing to add locally, so all 3 additions happen <b class="tsp">across cores</b>.'
        : 'Two cores share this output. Each adds its own 2 terms over time (1 addition each), then only the 2 partial sums meet: 1 addition <b class="tsp">across cores</b>.';
    this.cap.innerHTML = texts + (finished ? ' <span class="ok">13 = A[0,:]·B[:,0] ✓</span>' : '');
    let tb = '<table class="mini-tab"><tr><th>whole 4×4 product</th><th>multiplies</th><th>additions over time</th><th>additions across cores</th><th>time steps</th></tr>';
    [1, 2, 4].forEach((gg) => { const q = kgPlan(gg); tb += '<tr class="' + (gg === g ? 'cur' : '') + '"><td>' + gg + ' core' + (gg > 1 ? 's' : '') + ' per output</td><td>64</td><td>' + 16 * gg * (q.per - 1) + '</td><td class="tsp">' + 16 * (gg - 1) + '</td><td>' + q.per + ' + ' + q.levels + '</td></tr>'; });
    this.tab.innerHTML = tb + '</table>';
    this.cnt.innerHTML = countersHTML([{ label: 'step', value: this.t + '/' + this.n }, { label: 'multiplies', value: mul + '/4' }, { label: 'additions over time', value: tadd }, { label: 'additions across cores', value: sadd, cls: 'csp' }, { label: 'total', value: (tadd + sadd) + '/3' }]);
  };
  mix(KGroups.prototype);

  // ---------------- Split-K ----------------
  function SplitK(root, cfg) {
    this.root = root; root.classList.add('w'); this.speed = 1000;
    root.appendChild(el('div', 'w-head', '<span class="w-title">Split-K</span><span class="chip">K cut in 2 pieces: k = 0,1 and k = 2,3</span><span class="chip">each piece = a whole C, on its own core</span>'));
    const top = el('div', 'w-body'); root.appendChild(top);
    this.mA = new Mat('A', { vals: A, cls: 'compact' }); this.mB = new Mat('B', { vals: B, cls: 'compact' });
    this.mA.badge(0, 0, 4, 2, 'piece 0', 'soft'); this.mA.badge(0, 2, 4, 2, 'piece 1', 'soft');
    this.mB.badge(0, 0, 2, 4, 'piece 0', 'soft'); this.mB.badge(2, 0, 2, 4, 'piece 1', 'soft');
    top.append(this.mA.root, this.mB.root);
    const bot = el('div', 'w-body'); bot.style.alignItems = 'center'; root.appendChild(bot);
    this.m0 = new Mat('C', { title: 'C⁽⁰⁾', dims: 'core 0', cls: 'compact' });
    this.m1 = new Mat('C', { title: 'C⁽¹⁾', dims: 'core 1', cls: 'compact' });
    this.mC = new Mat('C', { title: 'C', dims: 'final', cls: 'compact' });
    bot.append(this.m0.root, el('span', 'node', '+'), this.m1.root, el('span', 'arrow', '='), this.mC.root);
    this.cap = el('div', 'cap'); root.appendChild(this.cap);
    const foot = el('div', 'w-foot'); root.appendChild(foot);
    this.cnt = el('div', 'counters'); foot.appendChild(this.cnt);
    foot.appendChild(this.makeControls({ noInner: true })); this.enableKeys();
    this.n = 3; this.go(0);
  }
  SplitK.prototype.go = function (t) {
    this.t = Math.max(0, Math.min(this.n, t));
    const P = [newState(), newState()], fin = newState();
    let mul = 0, tadd = 0, sadd = 0;
    const steps = Math.min(this.t, 2);
    for (let st = 0; st < steps; st++) for (let pc = 0; pc < 2; pc++) {
      const k = pc * 2 + st;
      for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) apply(P[pc], [i, j, k]);
    }
    mul = P[0].mul + P[1].mul; tadd = P[0].add + P[1].add;
    const reduced = this.t === 3;
    if (reduced) for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { fin.C[i][j] = P[0].C[i][j] + P[1].C[i][j]; fin.T[i][j] = K; sadd++; }
    const pieceCls = (S) => { const c = {}; for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) if (S.T[i][j]) c[key(i, j)] = 'part'; return c; };
    this.m0.paint(pieceCls(P[0]), cText(P[0])); this.m1.paint(pieceCls(P[1]), cText(P[1]));
    const cf = {}; cClass(fin, cf); if (reduced) for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) addCls(cf, key(i, j), 'spadd');
    this.mC.paint(cf, cText(fin));
    const ca = {}, cb = {};
    if (this.t > 0 && this.t < 3) [0, 1].forEach((pc) => { const k = pc * 2 + this.t - 1; for (let r = 0; r < 4; r++) { addCls(ca, key(r, k), 'read'); addCls(cb, key(k, r), 'read'); } });
    this.mA.paint(ca); this.mB.paint(cb);
    const texts = ['Two cores each get half of K. Each builds a <b>whole</b> 4×4 output from its half — both at the same time.',
      'Time 0: core 0 adds column 0 of A times row 0 of B; core 1 does column 2 × row 2. Both results are partial.',
      'Time 1: columns 1 and 3. Each piece is now complete — but each is still only <b>half</b> of the sum for every C value.',
      '<b class="tsp">Extra step:</b> the two pieces are added across cores, 16 additions in space. '];
    this.cap.innerHTML = texts[this.t] + (reduced ? checkLine(fin.C) : '');
    this.cnt.innerHTML = countersHTML([{ label: 'step', value: this.t + '/3' }, { label: 'multiplies', value: mul + '/64' }, { label: 'additions over time', value: tadd }, { label: 'additions across cores', value: sadd, cls: 'csp' }, { label: 'total', value: (tadd + sadd) + '/48' }]);
  };
  mix(SplitK.prototype);

  // ---------------- static pictures ----------------
  function StaticTriple(root, cfg) {
    cfg = cfg || {};
    root.classList.add('w');
    const tri = new Triple({ bs: cfg.bs || 1, note: cfg.note });
    root.appendChild(tri.root);
    const s = newState();
    if (cfg.filled) makeFrames('ijk').frames.forEach((f) => apply(s, f.ops[0]));
    const cls = {}; cClass(s, cls);
    if (cfg.hl) { // highlight one elementary step
      const [i, j, k] = cfg.hl;
      tri.A.paint({ [key(i, k)]: 'read' }); tri.B.paint({ [key(k, j)]: 'read' }); addCls(cls, key(i, j), 'write');
      tri.A.heads([i], [k]); tri.B.heads([k], [j]); tri.C.heads([i], [j]);
    }
    if (cfg.blocks) {
      for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) {
        tri.A.badge(a * 2, b * 2, 2, 2, 'A[' + a + ',' + b + ']', 'soft');
        tri.B.badge(a * 2, b * 2, 2, 2, 'B[' + a + ',' + b + ']', 'soft');
        tri.C.badge(a * 2, b * 2, 2, 2, 'C[' + a + ',' + b + ']', 'soft');
      }
    }
    tri.C.paint(cls, cText(s));
    if (cfg.caption) root.appendChild(el('div', 'cap', cfg.caption));
    this.root = root;
  }

  const TYPES = { player: Player, group: Group, cpu: HWScalar, imc: HWIMC, tensor: HWTensor, grid: GridSweep, waves: Waves, kgroups: KGroups, splitk: SplitK, static: StaticTriple };
  function mount(node) {
    const cfg = JSON.parse(node.getAttribute('data-widget'));
    const T = TYPES[cfg.type] || (global.MM2 && global.MM2.TYPES[cfg.type]);
    if (!T) throw new Error('unknown widget ' + cfg.type);
    const w = new T(node, cfg);
    node._widget = w;
    return w;
  }
  function mountAll(scope) { return Array.from((scope || document).querySelectorAll('[data-widget]')).map(mount); }

  // ---------------- pure self-test (runs in node too) ----------------
  function selfTest() {
    const out = [];
    ORDERS.forEach((o) => [1, 2].forEach((bs) => {
      const s = newState(); makeFrames(o, bs).frames.forEach((f) => f.ops.forEach((op) => apply(s, op)));
      out.push({ what: o + (bs > 1 ? ' (tiled)' : ''), mul: s.mul, add: s.add, ok: equalsRef(s.C) });
    }));
    ['os', 'ws', 'is'].forEach((m) => { const r = gridSim(m, m === 'os' ? 2 : 3); out.push({ what: 'grid ' + m, mul: r.s.mul, add: r.s.tadd + '+' + r.s.sadd, ok: equalsRef(r.s.C) && r.s.tadd + r.s.sadd === 48 }); });
    return out;
  }

  global.MM = { A, B, REF, M, N, K, IDX, STAY, STAT, SHORT, FORM, ORDERS, makeFrames, newState, apply, census, equalsRef, gridSim, mount, mountAll, selfTest };
})(typeof window !== 'undefined' ? window : globalThis);
