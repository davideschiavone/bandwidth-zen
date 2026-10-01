/* Attention simulator: plain attention (S through memory) and FlashAttention-2 (S never stored),
   on any small shape, any block sizes, any number of cores. Every view replays from scratch, so the
   counters are always the real ones, and every run ends by checking O against softmax(QKᵀ/√d)V.

   Reuses the matmul notebook's helpers (../../matmul/src/engine2.js, MM2._): the matrix grid, the
   stepping controls and the counters look and behave the same in both notebooks. The counts follow
   the same formulas as `bwz attention` (docs/MODEL.md §6.9): programs = heads × ceil(S/Br),
   waves = ceil(programs / cores), K and V read once per wave a head's programs span, O rescaled
   d numbers per query row on every key block after the first. */
(function (global) {
  'use strict';
  const X = global.MM2, H = X._;
  const { el, btn, addCls, counters, Mat, mix, key, rng, zeros } = H;

  // ---------------- the problem ----------------
  // The running example: 4 queries, 4 keys, d = 2, one head. Small integers so every score can be
  // checked by hand; the softmax makes the rest decimals.
  const EX = {
    Q: [[1, 0], [0, 1], [1, 1], [2, 0]],
    K: [[1, 0], [0, 1], [1, 1], [0, 2]],
    V: [[1, 2], [3, 0], [0, 1], [2, 2]]
  };
  function lcg(seed) { let s = (seed * 2654435761) >>> 0 || 1; return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296); }
  function problem(o) {
    const Tq = o.S, Tkv = o.T, d = o.d, heads = o.heads || 1, batch = o.batch || 1, HT = heads * batch;
    const ex = !o.seed && Tq === 4 && Tkv === 4 && d === 2 && HT === 1;
    const r = lcg(o.seed || 11);
    const draw = (rows) => rng(rows).map(() => rng(d).map(() => Math.floor(r() * 4) - 1));
    const Q = [], K = [], V = [];
    for (let h = 0; h < HT; h++) {
      Q.push(ex ? EX.Q.map((x) => x.slice()) : draw(Tq));
      K.push(ex ? EX.K.map((x) => x.slice()) : draw(Tkv));
      V.push(ex ? EX.V.map((x) => x.slice()) : draw(Tkv));
    }
    const scale = 1 / Math.sqrt(d);
    const REF = Q.map((q, h) => refAttention(q, K[h], V[h], scale));
    return { Tq, Tkv, d, heads, batch, HT, Q, K, V, REF, scale, seed: ex ? 0 : (o.seed || 11) };
  }
  // softmax(Q Kᵀ / √d) V the textbook way: every score of a row first, then the softmax, then P V.
  function refAttention(Q, K, V, scale) {
    return Q.map((q) => {
      const s = K.map((k) => q.reduce((a, x, c) => a + x * k[c], 0) * scale);
      const m = Math.max(...s), e = s.map((x) => Math.exp(x - m)), l = e.reduce((a, x) => a + x, 0);
      return rng(V[0].length).map((c) => e.reduce((a, p, j) => a + p * V[j][c], 0) / l);
    });
  }
  const TOL = 1e-9;
  function closeTo(O, REF) { return O.every((row, i) => row.every((x, c) => Math.abs(x - REF[i][c]) <= TOL)); }
  function fmt(x) {
    if (x === -Infinity) return '−∞';
    if (Number.isInteger(x)) return String(x);
    const a = Math.abs(x);
    return a >= 10 ? x.toFixed(0) : a >= 1 ? x.toFixed(1) : x.toFixed(2).replace(/^(-?)0\./, '$1.');
  }
  const ceil = Math.ceil, floor = Math.floor;

  // An inner matmul's partial sums, as `bwz` charges them on a unit with no periphery accumulator
  // (A100's tensor cores): os sums K inside one tile and owes nothing; ws and is cut K into
  // ceil(K / tile) slices, and every slice beyond the first is an addition on the vector unit.
  function innerAdds(flow, M, N, Kd, tile) { return flow === 'os' ? 0 : (ceil(Kd / Math.max(1, tile)) - 1) * M * N; }

  // =====================================================================
  // FlashAttention-2
  // cfg: Br, Bc, cores, tile, qk, pv (inner dataflows)
  // One PROGRAM = one (head, Br-row block of Q), on one core. Programs are dealt round-robin, one per
  // core per lockstep wave. Inside a wave every core runs its program's key blocks in step:
  //   qk  S = Q_i K_jᵀ / √d          (array)
  //   sm  online softmax: m', P, l, and the rescale of O from the second block on   (vector unit)
  //   pv  O += P V_j                  (array)
  // and after the last block, norm: O = O / l, O written back.
  // =====================================================================
  function flashPlan(P, cfg) {
    const Br = Math.max(1, Math.min(cfg.Br, P.Tq)), Bc = Math.max(1, Math.min(cfg.Bc, P.Tkv));
    const qb = ceil(P.Tq / Br), kb = ceil(P.Tkv / Bc);
    const programs = P.HT * qb, used = Math.max(1, Math.min(cfg.cores, programs)), W = ceil(programs / used);
    const L = 3 * kb + 1, n = W * L;
    const progs = rng(programs).map((p) => {
      const head = floor(p / qb), blk = p % qb;
      return { id: p, head, blk, r0: blk * Br, r1: Math.min((blk + 1) * Br, P.Tq), wave: floor(p / used), core: p % used };
    });
    const kbl = rng(kb).map((j) => [j * Bc, Math.min((j + 1) * Bc, P.Tkv)]);
    // K and V cross memory once per wave a head's programs span (D33's staging, as in bwz).
    let streams = 0;
    for (let h = 0; h < P.HT; h++) streams += floor(((h + 1) * qb - 1) / used) - floor((h * qb) / used) + 1;
    return { Br, Bc, qb, kb, programs, used, W, L, n, progs, kbl, streams };
  }
  function phaseAt(pl, s) {
    const wave = floor(s / pl.L), pos = s % pl.L;
    if (pos < 3 * pl.kb) return { wave, j: floor(pos / 3), ph: ['qk', 'sm', 'pv'][pos % 3] };
    return { wave, j: pl.kb - 1, ph: 'norm' };
  }
  function flashState(P, cfg, pl, t) {
    const d = P.d, tile = cfg.tile || 1;
    const st = {
      prog: {}, mul: 0, exps: 0, rescaled: 0, normalised: 0, qRead: 0, kvRead: 0, oWritten: 0,
      staged: new Set(), heldMax: 0, innerAdds: 0,
      O: P.Q.map(() => zeros(P.Tq, d)), Odone: P.Q.map(() => rng(P.Tq).map(() => false))
    };
    for (let s = 0; s < Math.min(t, pl.n); s++) {
      const { wave, j, ph } = phaseAt(pl, s);
      const act = pl.progs.filter((p) => p.wave === wave);
      let held = 0;
      act.forEach((p) => {
        const rows = p.r1 - p.r0, [c0, c1] = pl.kbl[j], w = c1 - c0, h = p.head;
        const ps = st.prog[p.id] || (st.prog[p.id] = {
          m: rng(rows).map(() => -Infinity), l: rng(rows).map(() => 0), O: zeros(rows, d),
          S: null, Pm: null, mnew: null, alpha: null, j: -1, ph: null, doneBlocks: 0, done: false
        });
        ps.j = j; ps.ph = ph;
        if (ph === 'qk') {
          if (j === 0) st.qRead += rows * d;
          const k0 = wave + ',' + h + ',' + j;
          if (!st.staged.has(k0)) { st.staged.add(k0); st.kvRead += 2 * w * d; }
          ps.S = rng(rows).map((a) => rng(w).map((b) => P.Q[h][p.r0 + a].reduce((acc, x, c) => acc + x * P.K[h][c0 + b][c], 0) * P.scale));
          ps.Pm = null;
          st.mul += rows * w * d;
          st.innerAdds += innerAdds(cfg.qk, rows, w, d, tile);
          held += rows * w;
        } else if (ph === 'sm') {
          ps.mnew = ps.m.map((m, a) => Math.max(m, ...ps.S[a]));
          ps.Pm = ps.S.map((row, a) => row.map((x) => Math.exp(x - ps.mnew[a])));
          ps.alpha = ps.m.map((m, a) => Math.exp(m - ps.mnew[a]));
          ps.l = ps.l.map((l, a) => ps.alpha[a] * l + ps.Pm[a].reduce((x, y) => x + y, 0));
          if (j > 0) { ps.O = ps.O.map((row, a) => row.map((x) => x * ps.alpha[a])); st.rescaled += rows * d; }
          st.exps += rows * w;
          held += rows * w;
        } else if (ph === 'pv') {
          ps.O = ps.O.map((row, a) => row.map((x, c) => x + ps.Pm[a].reduce((acc, pv, b) => acc + pv * P.V[h][c0 + b][c], 0)));
          ps.m = ps.mnew;
          ps.doneBlocks = j + 1;
          st.mul += rows * d * w;
          st.innerAdds += innerAdds(cfg.pv, rows, d, w, tile);
        } else {
          ps.O = ps.O.map((row, a) => row.map((x) => x / ps.l[a]));
          for (let a = 0; a < rows; a++) { st.O[h][p.r0 + a] = ps.O[a].slice(); st.Odone[h][p.r0 + a] = true; }
          st.normalised += rows * d; st.oWritten += rows * d; ps.done = true;
        }
      });
      st.heldMax = Math.max(st.heldMax, held);
    }
    return st;
  }
  function flashTotals(P, pl, cfg) {
    const tile = cfg.tile || 1;
    let adds = 0;
    pl.progs.forEach((p) => pl.kbl.forEach(([c0, c1]) => {
      const rows = p.r1 - p.r0, w = c1 - c0;
      adds += innerAdds(cfg.qk, rows, w, P.d, tile) + innerAdds(cfg.pv, rows, P.d, w, tile);
    }));
    return {
      mul: 2 * P.HT * P.Tq * P.Tkv * P.d, exps: P.HT * P.Tq * P.Tkv,
      rescaled: P.HT * P.Tq * P.d * (pl.kb - 1), normalised: P.HT * P.Tq * P.d,
      qRead: P.HT * P.Tq * P.d, oWritten: P.HT * P.Tq * P.d,
      kvRead: 2 * pl.streams * P.Tkv * P.d, kvOnce: 2 * P.HT * P.Tkv * P.d, innerAdds: adds
    };
  }
  function flashCode(pl) {
    return [
      { html: '<span class="kw">for</span> wave <span class="kw">in</span> range(' + pl.W + '):   <span class="cm"># lockstep: every core, one program</span>', v: 'wave' },
      { html: '  program = wave·' + pl.used + ' + core  <span class="cm"># → (head, Q block)</span>', v: 'wave' },
      { html: '  <span class="kw">for</span> j <span class="kw">in</span> range(' + pl.kb + '):  <span class="cm"># key blocks, Bc = ' + pl.Bc + '</span>', v: 'loop' },
      { html: '    S = <span class="ta">Q_i</span> <span class="tb">K_j</span>ᵀ / √d          <span class="cm"># array</span>', v: 'qk' },
      { html: "    m' = max(m, rowmax S)", v: 'sm' },
      { html: "    P  = exp(S − m')", v: 'sm' },
      { html: "    l  = e^(m−m')·l + rowsum P", v: 'sm' },
      { html: "    <span class=\"tc\">O</span> = e^(m−m')·<span class=\"tc\">O</span>      <span class=\"cm\"># rescale, j ≥ 1</span>", v: 'sm' },
      { html: '    <span class="tc">O</span> += P <span class="tb">V_j</span>; m = m\'   <span class="cm"># array</span>', v: 'pv' },
      { html: '  <span class="tc">O</span> = <span class="tc">O</span> / l;  write <span class="tc">O_i</span>', v: 'norm' }
    ];
  }

  // =====================================================================
  // Plain attention, one head, one query row per step and phase:
  //   S row = q Kᵀ / √d   → written to memory
  //   P row = softmax(S row)  (S read back, P written)
  //   O row = P row V     (P read back)
  // =====================================================================
  function naivePlan(P) { return { n: 3 * P.Tq }; }
  function naiveState(P, t) {
    const st = { S: zeros(P.Tq, P.Tkv), Pm: zeros(P.Tq, P.Tkv), O: zeros(P.Tq, P.d), sOn: rng(P.Tq).map(() => 0), mul: 0, exps: 0, sW: 0, sR: 0, pW: 0, pR: 0 };
    const q = P.Q[0], K = P.K[0], V = P.V[0];
    for (let s = 0; s < t; s++) {
      const i = floor(s / 3), ph = s % 3;
      if (ph === 0) { st.S[i] = K.map((k) => q[i].reduce((a, x, c) => a + x * k[c], 0) * P.scale); st.mul += P.Tkv * P.d; st.sW += P.Tkv; st.sOn[i] = 1; }
      else if (ph === 1) { const m = Math.max(...st.S[i]), e = st.S[i].map((x) => Math.exp(x - m)), l = e.reduce((a, x) => a + x, 0); st.Pm[i] = e.map((x) => x / l); st.exps += P.Tkv; st.sR += P.Tkv; st.pW += P.Tkv; st.sOn[i] = 2; }
      else { st.O[i] = rng(P.d).map((c) => st.Pm[i].reduce((a, p, j) => a + p * V[j][c], 0)); st.mul += P.Tkv * P.d; st.pR += P.Tkv; st.sOn[i] = 3; }
    }
    return st;
  }

  // ---------------- shared drawing ----------------
  // Q on the left of S, Kᵀ on top of it, Vᵀ under it and O on its right: a row of S lines up with a
  // row of Q and of O, a column of S with a column of Kᵀ and of Vᵀ — the shared axes of the formula.
  function Quint(P, rb, cb, sTitle) {
    this.root = el('div', 'att-grid');
    if (Math.max(P.Tq, P.Tkv) > 6) this.root.classList.add('dense');
    const o = (kind, r, c, extra) => new Mat(kind, r, c, Object.assign({ rv: extra.rv, cv: extra.cv }, extra));
    this.Q = o('Q', P.Tq, P.d, { rv: 'q', cv: 'd', rb, title: 'Q', dims: P.Tq + '×' + P.d });
    this.K = o('K', P.d, P.Tkv, { rv: 'd', cv: 'k', cb, title: 'Kᵀ', dims: P.d + '×' + P.Tkv });
    this.S = o('S', P.Tq, P.Tkv, { rv: 'q', cv: 'k', rb, cb, title: sTitle, dims: P.Tq + '×' + P.Tkv });
    this.V = o('V', P.d, P.Tkv, { rv: 'd', cv: 'k', cb, title: 'Vᵀ', dims: 'V is ' + P.Tkv + '×' + P.d + ', drawn on its side' });
    this.O = o('O', P.Tq, P.d, { rv: 'q', cv: 'd', rb, title: 'O', dims: P.Tq + '×' + P.d });
    const slot = (m, cls) => { m.root.classList.add(cls); this.root.appendChild(m.root); };
    slot(this.K, 'g-k'); slot(this.Q, 'g-q'); slot(this.S, 'g-s'); slot(this.O, 'g-o'); slot(this.V, 'g-v');
  }
  Quint.prototype.values = function (P, h) {
    this.Q.paint({}, (r, c) => String(P.Q[h][r][c]));
    this.K.paint({}, (r, c) => String(P.K[h][c][r]));
    this.V.paint({}, (r, c) => String(P.V[h][c][r]));
  };

  // =====================================================================
  // Views
  // =====================================================================
  function FlashView(root, cfg) {
    this.cfg = cfg; this.root = root; this.P = cfg.P; root.classList.add('w', 'pv', 'att');
    this.baseSpeed = 800; this.t = 0;
    this.pl = flashPlan(this.P, cfg); this.n = this.pl.n; this.tot = flashTotals(this.P, this.pl, cfg);
    const pl = this.pl;
    if (cfg.title !== false) {
      const head = el('div', 'w-head'); root.appendChild(head);
      head.innerHTML = '<span class="chip">Br = ' + pl.Br + ' query rows × Bc = ' + pl.Bc + ' keys per block</span>' +
        '<span class="chip">' + pl.programs + ' program' + (pl.programs > 1 ? 's' : '') + ' → ' + pl.used + ' core' + (pl.used > 1 ? 's' : '') + ' → ' + pl.W + ' wave' + (pl.W > 1 ? 's' : '') + '</span>' +
        '<span class="chip">' + pl.kb + ' key block' + (pl.kb > 1 ? 's' : '') + ' per program</span>' +
        '<span class="chip C">S never stored</span>';
    }
    const top = el('div', 'pv-top'); root.appendChild(top);
    this.q = new Quint(this.P, pl.Br, pl.Bc, 'S = QKᵀ/√d → P');
    const left = el('div', 'stack'); left.appendChild(this.q.root); top.appendChild(left);
    this.headLab = el('div', 'att-head'); left.insertBefore(this.headLab, this.q.root);
    const codeWrap = el('div', 'stack'); top.appendChild(codeWrap);
    this.codeEl = el('pre', 'code'); codeWrap.appendChild(this.codeEl);
    this.lines = flashCode(pl);
    this.lineEls = this.lines.map((l) => { const d = el('div', 'ln', l.html); this.codeEl.appendChild(d); return d; });
    this.run = el('div', 'att-run'); codeWrap.appendChild(this.run);
    codeWrap.appendChild(el('div', 'legend', '<span><i class="lg read"></i>read now</span><span><i class="lg s-live"></i>S block on the array</span><span><i class="lg s-soft"></i>softmaxed on the vector unit</span><span><i class="lg s-gone"></i>used and discarded</span><span><i class="lg part"></i>O, unnormalised</span><span><i class="lg fin"></i>O, final</span>'));
    root.appendChild(el('div', 'pv-lab', 'The chip: what every core is doing right now'));
    this.floor = el('div', 'floor'); root.appendChild(this.floor);
    this.coreEls = rng(cfg.cores).map(() => { const b = el('div', 'corebox'); this.floor.appendChild(b); return b; });
    root.appendChild(el('div', 'pv-lab', 'Timeline: one row per core, one column per step'));
    this.ganttWrap = el('div', 'gantt-wrap'); root.appendChild(this.ganttWrap);
    this.buildGantt();
    this.cap = el('div', 'cap'); root.appendChild(this.cap);
    const foot = el('div', 'w-foot'); root.appendChild(foot);
    this.cnt = el('div', 'counters'); foot.appendChild(this.cnt);
    const right = el('div', 'stack right'); right.append(this.controls('Next block'), this.scrubber()); foot.appendChild(right);
    this.keys(); this.go(0);
  }
  FlashView.prototype.inner = function () {
    // Jump to the end of the current key block (or the next one).
    // A block ends after its third phase; a wave ends after the normalise. Both land on a
    // multiple of 3 within the wave, so that is where "next block" stops.
    const L = this.pl.L;
    let t = this.t + 1;
    while (t < this.n && (t % L) % 3 !== 0) t++;
    this.go(Math.min(this.n, t)); return true;
  };
  FlashView.prototype.buildGantt = function () {
    const pl = this.pl, cfg = this.cfg, g = el('div', 'gantt');
    g.style.gridTemplateColumns = '4.2rem repeat(' + pl.n + ', minmax(' + (pl.n > 24 ? '1.3rem' : '2.1rem') + ', 1fr))';
    g.appendChild(Object.assign(el('div', 'g-h'), { style: 'grid-row:1;grid-column:1' }));
    for (let w = 0; w < pl.W; w++) {
      const h = el('div', 'g-wave', 'wave ' + (w + 1)); h.style.gridRow = '1'; h.style.gridColumn = (2 + w * pl.L) + ' / span ' + pl.L; g.appendChild(h);
    }
    const sl = el('div', 'g-lab', 'step'); sl.style.gridRow = '2'; sl.style.gridColumn = '1'; g.appendChild(sl);
    for (let c = 0; c < pl.n; c++) {
      const ph = phaseAt(pl, c).ph;
      const x = el('div', 'g-num ph-' + ph, { qk: 'S', sm: 'σ', pv: 'O', norm: '/l' }[ph]);
      x.title = { qk: 'S = QKᵀ on the array', sm: 'online softmax on the vector unit', pv: 'O += PV on the array', norm: 'O = O / l, written back' }[ph];
      x.style.gridRow = '2'; x.style.gridColumn = String(2 + c); g.appendChild(x);
    }
    for (let w = 0; w < pl.W; w++) {
      const band = el('div', 'g-band' + (w % 2 ? ' alt' : '')); band.style.gridRow = '3 / span ' + cfg.cores; band.style.gridColumn = (2 + w * pl.L) + ' / span ' + pl.L; g.appendChild(band);
    }
    this.bars = [];
    for (let c = 0; c < cfg.cores; c++) {
      const lab = el('div', 'g-lab', 'core ' + c); lab.style.gridRow = String(3 + c); lab.style.gridColumn = '1'; g.appendChild(lab);
      for (let w = 0; w < pl.W; w++) {
        const u = pl.progs.find((x) => x.wave === w && x.core === c);
        const b = el('div', u ? 'bar f-os' : 'bar idle', u ? '<span class="fill"></span><span class="bt">h' + u.head + ' q' + u.r0 + (u.r1 - u.r0 > 1 ? '–' + (u.r1 - 1) : '') + '</span>' : '<span class="bt">idle</span>');
        b.style.gridRow = String(3 + c); b.style.gridColumn = (2 + w * pl.L) + ' / span ' + pl.L;
        if (u) { b.title = 'program ' + u.id + ': head ' + u.head + ', query rows ' + u.r0 + '…' + (u.r1 - 1) + ', on core ' + c + ', wave ' + (w + 1); this.bars.push({ el: b, u }); }
        g.appendChild(b);
      }
    }
    this.cursor = el('div', 'g-cursor'); this.cursor.style.gridRow = '2 / span ' + (cfg.cores + 1); g.appendChild(this.cursor);
    this.ganttWrap.appendChild(g);
  };
  FlashView.prototype.render = function () {
    const P = this.P, pl = this.pl, cfg = this.cfg;
    const st = flashState(P, cfg, pl, this.t);
    const slot = this.t - 1, live = slot >= 0 && slot < pl.n ? phaseAt(pl, slot) : null;
    const active = live ? pl.progs.filter((p) => p.wave === live.wave) : [];
    const shown = cfg.hsel != null && cfg.hsel >= 0 ? Math.min(cfg.hsel, P.HT - 1) : active.length ? active[0].head : (this.t >= pl.n ? P.HT - 1 : 0);
    const nHeads = new Set(active.map((p) => p.head)).size;
    this.headLab.innerHTML = 'Showing <b>head ' + shown + '</b> of ' + P.HT + (P.batch > 1 ? ' (batch ' + floor(shown / P.heads) + ', head ' + (shown % P.heads) + ')' : '') + (active.length && nHeads > 1 ? ' · this wave also runs head' + (nHeads > 2 ? 's ' : ' ') + [...new Set(active.map((p) => p.head))].filter((h) => h !== shown).join(', ') : '');
    this.q.values(P, shown);
    const cQ = {}, cK = {}, cV = {}, cS = {}, cO = {};
    const sTxt = {}, oTxt = {};
    pl.progs.filter((p) => p.head === shown).forEach((p) => {
      const ps = st.prog[p.id]; if (!ps) return;
      const isLive = live && p.wave === live.wave;
      for (let a = p.r0; a < p.r1; a++) {
        // S blocks this program already used: discarded.
        for (let jj = 0; jj < ps.doneBlocks; jj++) { const [c0, c1] = pl.kbl[jj]; for (let b = c0; b < c1; b++) addCls(cS, key(a, b), 's-gone'); }
        if (isLive && live.ph !== 'norm') {
          const [c0, c1] = pl.kbl[live.j];
          for (let b = c0; b < c1; b++) {
            const k0 = key(a, b);
            if (live.ph === 'qk') { addCls(cS, k0, 's-live'); sTxt[k0] = fmt(ps.S[a - p.r0][b - c0]); }
            else if (live.ph === 'sm') { addCls(cS, k0, 's-soft'); sTxt[k0] = fmt(ps.Pm[a - p.r0][b - c0]); }
            else { addCls(cS, k0, 's-soft'); sTxt[k0] = fmt(ps.Pm[a - p.r0][b - c0]); }
          }
        }
        for (let c = 0; c < P.d; c++) {
          const k0 = key(a, c);
          if (ps.done) { addCls(cO, k0, 'fin'); oTxt[k0] = fmt(st.O[shown][a][c]); }
          else if (ps.doneBlocks > 0 || (isLive && live.ph === 'pv')) { addCls(cO, k0, 'part'); oTxt[k0] = fmt(ps.O[a - p.r0][c]); }
          if (isLive) addCls(cQ, k0, live.ph === 'qk' ? 'read' : 'stay');
          if (isLive && (live.ph === 'pv' || (live.ph === 'sm' && live.j > 0) || live.ph === 'norm')) addCls(cO, k0, 'write');
        }
      }
      if (isLive && live.ph !== 'norm') {
        const [c0, c1] = pl.kbl[live.j];
        for (let b = c0; b < c1; b++) for (let c = 0; c < P.d; c++) {
          if (live.ph === 'qk') addCls(cK, key(c, b), 'read');
          if (live.ph === 'pv') addCls(cV, key(c, b), 'read');
        }
      }
    });
    this.q.Q.paint(cQ, (r, c) => String(P.Q[shown][r][c]));
    this.q.K.paint(cK, (r, c) => String(P.K[shown][c][r]));
    this.q.V.paint(cV, (r, c) => String(P.V[shown][c][r]));
    this.q.S.paint(cS, (r, c) => sTxt[key(r, c)] || '');
    this.q.O.paint(cO, (r, c) => oTxt[key(r, c)] || '·');
    // the running max and sum of the live program(s) of the shown head
    const liveHere = active.filter((p) => p.head === shown);
    if (liveHere.length) {
      let h = '<table class="runtab"><tr><th>row</th><th>m (running max)</th><th>l (running sum)</th>' + (live.ph === 'sm' && live.j > 0 ? '<th>e^(m−m′)</th>' : '') + '</tr>';
      liveHere.forEach((p) => {
        const ps = st.prog[p.id];
        for (let a = 0; a < p.r1 - p.r0; a++) {
          const m = live.ph === 'sm' ? ps.mnew[a] : ps.m[a];
          h += '<tr><td>q' + (p.r0 + a) + '</td><td>' + fmt(m) + '</td><td>' + fmt(ps.l[a]) + '</td>' + (live.ph === 'sm' && live.j > 0 ? '<td>' + fmt(ps.alpha[a]) + '</td>' : '') + '</tr>';
        }
      });
      this.run.innerHTML = h + '</table>';
    } else this.run.innerHTML = '<span class="muted">' + (this.t === 0 ? 'Running max m and sum l appear here.' : 'No program of this head is running now.') + '</span>';
    // cores
    this.coreEls.forEach((box, c) => {
      const u = active.find((x) => x.core === c);
      box.className = 'corebox' + (u ? '' : ' idle');
      if (!u) {
        box.innerHTML = '<h5><span>core ' + c + '</span></h5><div class="op">' + (this.t === 0 ? 'waiting to start' : this.t >= pl.n ? 'done' : 'idle: no program left for this core in wave ' + (live.wave + 1)) + '</div>';
        return;
      }
      const [c0, c1] = pl.kbl[live.j];
      const doing = {
        qk: 'S = Q[' + u.r0 + '…' + (u.r1 - 1) + '] · K[' + c0 + '…' + (c1 - 1) + ']ᵀ on the array',
        sm: 'online softmax' + (live.j > 0 ? ' + rescale O' : '') + ' on the vector unit',
        pv: 'O += P · V[' + c0 + '…' + (c1 - 1) + '] on the array',
        norm: 'O = O / l, write O[' + u.r0 + '…' + (u.r1 - 1) + ']'
      }[live.ph];
      box.innerHTML = '<h5><span>core ' + c + '</span><span class="muted">program ' + u.id + ' · head ' + u.head + ' · key block ' + (live.j + 1) + '/' + pl.kb + '</span></h5><div class="op"><b>' + doing + '</b></div>';
    });
    this.bars.forEach(({ el: b, u }) => {
      const start = u.wave * pl.L, done = Math.max(0, Math.min(pl.L, this.t - start));
      b.querySelector('.fill').style.width = (100 * done / pl.L) + '%';
      b.classList.toggle('now', !!live && u.wave === live.wave);
    });
    this.cursor.style.gridColumn = String(2 + Math.max(0, slot)); this.cursor.hidden = this.t === 0 || this.t > pl.n;
    // The rescale line (index 7) is part of the softmax phase, but on the first block there is
    // nothing in O to rescale, so it stays dark there.
    this.lineEls.forEach((d, n) => { const v = this.lines[n].v; d.classList.toggle('on', !!live && v === live.ph && !(n === 7 && live.j === 0)); });
    let cap;
    if (this.t === 0) cap = 'Press <b>Step</b>. Each step, every busy core does the same thing on its own program: <b>S = QKᵀ</b> on the array, then the <b>online softmax</b> on the vector unit, then <b>O += PV</b> on the array — one key block at a time.';
    else if (live.ph === 'qk') cap = '<b>Step ' + this.t + '</b> · wave ' + (live.wave + 1) + '/' + pl.W + ', key block ' + (live.j + 1) + '/' + pl.kb + ': scores of ' + (live.j === 0 ? 'the first' : 'the next') + ' ' + (pl.kbl[live.j][1] - pl.kbl[live.j][0]) + ' key' + (pl.kbl[live.j][1] - pl.kbl[live.j][0] > 1 ? 's' : '') + '. ' + (live.j === 0 ? 'K and V for this block are fetched <b>once</b> and shared by every program of the same head in this wave.' : 'The earlier blocks of S are already gone.');
    else if (live.ph === 'sm') cap = '<b>Step ' + this.t + '</b> · online softmax: the new running max m′, P = exp(S − m′), and the running sum l.' + (live.j > 0 ? ' The max may have grown, so everything O holds is <b>rescaled</b> by e^(m−m′) — the one extra piece of work FlashAttention adds.' : ' O is still empty, so there is nothing to rescale.');
    else if (live.ph === 'pv') cap = '<b>Step ' + this.t + '</b> · O += P·V for this block, on the array. After this, the S block is <b>thrown away</b>.';
    else cap = '<b>Step ' + this.t + '</b> · all key blocks seen: O = O / l, and O is written back — the only thing this program ever writes.';
    if (this.t === pl.n) {
      const ok = P.REF.every((R, h) => closeTo(st.O[h], R));
      cap = 'Done in ' + pl.n + ' steps. ' + (ok ? '<span class="ok">Check: O == softmax(QKᵀ/√d)·V ✓</span>' : '<span class="tsp">O differs ✗</span>') + ' S was never stored: at most <b>' + st.heldMax + '</b> scores were on chip at once, against ' + P.HT * P.Tq * P.Tkv + ' in all.';
    }
    this.cap.innerHTML = cap;
    const util = Math.round(100 * pl.programs / (pl.W * cfg.cores));
    const T = this.tot;
    const items = [
      { label: 'step', value: this.t + '/' + pl.n }, { label: 'multiplies', value: st.mul + '/' + T.mul },
      { label: 'exp()', value: st.exps + '/' + T.exps }, { label: 'O rescaled', value: st.rescaled + '/' + T.rescaled, cls: 'csp' },
      { label: 'Q read', value: st.qRead }, { label: 'K,V read', value: st.kvRead + (T.kvRead > T.kvOnce ? ' (once: ' + T.kvOnce + ')' : ''), cls: T.kvRead > T.kvOnce ? 'csp' : '' },
      { label: 'O written', value: st.oWritten }, { label: 'S in memory', value: 0 },
      { label: 'core use', value: util + '%' }
    ];
    if (cfg.qk !== 'os' || cfg.pv !== 'os') items.push({ label: 'inner partial adds', value: st.innerAdds + '/' + T.innerAdds, cls: 'csp' });
    this.cnt.innerHTML = counters(items);
  };
  FlashView.prototype.trace = function (limit) {
    const pl = this.pl, out = [];
    for (let s = 0; s < Math.min(this.t, pl.n); s++) {
      const { wave, j, ph } = phaseAt(pl, s);
      const act = pl.progs.filter((p) => p.wave === wave);
      const [c0, c1] = pl.kbl[j];
      const what = {
        qk: (p) => 'core' + p.core + ' h' + p.head + ' S=Q[' + p.r0 + ':' + p.r1 + ']K[' + c0 + ':' + c1 + ']ᵀ',
        sm: (p) => 'core' + p.core + ' h' + p.head + ' softmax' + (j ? '+rescale' : ''),
        pv: (p) => 'core' + p.core + ' h' + p.head + ' O+=PV[' + c0 + ':' + c1 + ']',
        norm: (p) => 'core' + p.core + ' h' + p.head + ' O/=l, write O[' + p.r0 + ':' + p.r1 + ']'
      }[ph];
      out.push('step ' + String(s + 1).padStart(2) + ' (wave ' + (wave + 1) + ', key block ' + (j + 1) + '): ' + act.map(what).join(' | ') + (act.length < this.cfg.cores ? ' | ' + (this.cfg.cores - act.length) + ' idle' : ''));
    }
    return limit ? out.slice(-limit) : out;
  };
  mix(FlashView.prototype);

  function NaiveView(root, cfg) {
    this.cfg = cfg; this.root = root; this.P = cfg.P; root.classList.add('w', 'pv', 'att');
    this.baseSpeed = 900; this.t = 0; this.n = naivePlan(this.P).n;
    const P = this.P;
    if (cfg.title !== false) {
      const head = el('div', 'w-head'); root.appendChild(head);
      head.innerHTML = '<span class="chip">one query row per step, three phases</span><span class="chip">S = ' + P.Tq + '×' + P.Tkv + ' scores, all kept</span><span class="chip mem">S and P go through memory</span>';
    }
    const top = el('div', 'pv-top'); root.appendChild(top);
    this.q = new Quint(P, P.Tq, P.Tkv, 'S = QKᵀ/√d, then P');
    top.appendChild(this.q.root);
    const codeWrap = el('div', 'stack'); top.appendChild(codeWrap);
    this.codeEl = el('pre', 'code'); codeWrap.appendChild(this.codeEl);
    this.lines = [
      { html: '<span class="kw">for</span> i <span class="kw">in</span> range(' + P.Tq + '):  S[i] = <span class="ta">Q[i]</span> <span class="tb">K</span>ᵀ / √d   <span class="cm"># write S</span>', v: 0 },
      { html: '<span class="kw">for</span> i <span class="kw">in</span> range(' + P.Tq + '):  P[i] = softmax(S[i]) <span class="cm"># read S, write P</span>', v: 1 },
      { html: '<span class="kw">for</span> i <span class="kw">in</span> range(' + P.Tq + '):  <span class="tc">O[i]</span> = P[i] <span class="tb">V</span>        <span class="cm"># read P</span>', v: 2 }
    ];
    this.lineEls = this.lines.map((l) => { const d = el('div', 'ln', l.html); this.codeEl.appendChild(d); return d; });
    codeWrap.appendChild(el('div', 'legend', '<span><i class="lg read"></i>read now</span><span><i class="lg s-mem"></i>S (or P) stored</span><span><i class="lg fin"></i>O row done</span>'));
    codeWrap.appendChild(el('p', 'small', 'Drawn one row at a time for clarity: phase by phase for each row. A real kernel computes the whole of S first, then the softmax, then O — the memory traffic is the same.'));
    this.cap = el('div', 'cap'); root.appendChild(this.cap);
    const foot = el('div', 'w-foot'); root.appendChild(foot);
    this.cnt = el('div', 'counters'); foot.appendChild(this.cnt);
    const right = el('div', 'stack right'); right.append(this.controls('Next row'), this.scrubber()); foot.appendChild(right);
    this.keys(); this.go(0);
  }
  NaiveView.prototype.inner = function () { const t = Math.min(this.n, (floor(this.t / 3) + 1) * 3); this.go(t === this.t ? Math.min(this.n, t + 3) : t); return true; };
  NaiveView.prototype.render = function () {
    const P = this.P, st = naiveState(P, this.t), slot = this.t - 1, i = floor(slot / 3), ph = slot % 3;
    this.q.values(P, 0);
    const cQ = {}, cK = {}, cV = {}, cS = {}, cO = {};
    for (let r = 0; r < P.Tq; r++) {
      for (let c = 0; c < P.Tkv; c++) if (st.sOn[r]) addCls(cS, key(r, c), 's-mem');
      if (st.sOn[r] === 3) for (let c = 0; c < P.d; c++) addCls(cO, key(r, c), 'fin');
    }
    if (slot >= 0) {
      if (ph === 0) { for (let c = 0; c < P.d; c++) addCls(cQ, key(i, c), 'read'); for (let c = 0; c < P.d; c++) for (let b = 0; b < P.Tkv; b++) addCls(cK, key(c, b), 'read'); for (let b = 0; b < P.Tkv; b++) addCls(cS, key(i, b), 'write'); }
      if (ph === 1) for (let b = 0; b < P.Tkv; b++) addCls(cS, key(i, b), 'read-s');
      if (ph === 2) { for (let b = 0; b < P.Tkv; b++) addCls(cS, key(i, b), 'read-s'); for (let c = 0; c < P.d; c++) for (let b = 0; b < P.Tkv; b++) addCls(cV, key(c, b), 'read'); for (let c = 0; c < P.d; c++) addCls(cO, key(i, c), 'write'); }
    }
    this.q.Q.paint(cQ, (r, c) => String(P.Q[0][r][c]));
    this.q.K.paint(cK, (r, c) => String(P.K[0][c][r]));
    this.q.V.paint(cV, (r, c) => String(P.V[0][c][r]));
    this.q.S.paint(cS, (r, c) => (st.sOn[r] >= 2 ? fmt(st.Pm[r][c]) : st.sOn[r] ? fmt(st.S[r][c]) : ''));
    this.q.O.paint(cO, (r, c) => (st.sOn[r] === 3 ? fmt(st.O[r][c]) : '·'));
    this.lineEls.forEach((d, n) => d.classList.toggle('on', slot >= 0 && n === ph));
    let cap = this.t === 0 ? 'Press <b>Step</b>. Plain attention computes <b>every</b> score, keeps them, and only then turns them into probabilities.' :
      ['<b>Row ' + i + '</b>: its scores against every key, written to memory.', '<b>Row ' + i + '</b>: softmax needs the <b>whole row</b> — its max and its sum — so the stored scores are read back and P is written.', '<b>Row ' + i + '</b>: O = P·V, reading P back.'][ph];
    if (this.t === this.n) {
      const ok = closeTo(st.O, P.REF[0]);
      cap = 'Done. ' + (ok ? '<span class="ok">Check: O == softmax(QKᵀ/√d)·V ✓</span>' : '<span class="tsp">O differs ✗</span>') + ' ' + P.Tq * P.Tkv + ' scores were stored and read back, then the same number of probabilities: <b>' + (st.sW + st.sR + st.pW + st.pR) + '</b> numbers through memory that FlashAttention never moves.';
    }
    this.cap.innerHTML = cap;
    this.cnt.innerHTML = counters([
      { label: 'step', value: this.t + '/' + this.n }, { label: 'multiplies', value: st.mul + '/' + 2 * P.Tq * P.Tkv * P.d },
      { label: 'exp()', value: st.exps }, { label: 'S written', value: st.sW, cls: 'cm' }, { label: 'S read', value: st.sR, cls: 'cm' },
      { label: 'P written', value: st.pW, cls: 'cm' }, { label: 'P read', value: st.pR, cls: 'cm' }, { label: 'S held at once', value: st.sOn.filter((x) => x).length * P.Tkv }
    ]);
  };
  NaiveView.prototype.trace = function () {
    const out = [];
    for (let s = 0; s < this.t; s++) { const i = floor(s / 3); out.push('step ' + String(s + 1).padStart(2) + ': ' + ['S[' + i + '] = Q[' + i + ']·Kᵀ/√d → memory', 'P[' + i + '] = softmax(S[' + i + ']) (S read, P written)', 'O[' + i + '] = P[' + i + ']·V (P read)'][s % 3]); }
    return out;
  };
  mix(NaiveView.prototype);

  // Inside one block: S = Q_i K_jᵀ is an ordinary matmul, M = Br, K = d, N = Bc — shown by the
  // matmul notebook's own SingleView, with the array tile and the stationarity as its block loops.
  const BORDER = { os: 'ijk', ws: 'jki', is: 'ikj' };
  function innerProblem(P, cfg) {
    const pl = flashPlan(P, cfg), [c0, c1] = pl.kbl[0], rows = pl.progs[0].r1 - pl.progs[0].r0;
    const A = rng(rows).map((a) => P.Q[0][a].slice());
    const B = rng(P.d).map((c) => rng(c1 - c0).map((b) => P.K[0][c0 + b][c]));
    const M = rows, N = c1 - c0, K = P.d;
    const REF = rng(M).map((i) => rng(N).map((j) => rng(K).reduce((s, k) => s + A[i][k] * B[k][j], 0)));
    return { M, N, K, A, B, REF, seed: 1 };
  }
  function largestDivisorUpTo(n, cap) { return X.divisors(n).filter((v) => v <= cap).pop(); }
  function InnerView(root, cfg) {
    const Pm = innerProblem(cfg.P, cfg), tile = cfg.tile || 1;
    const sub = Object.assign({}, cfg, {
      P: Pm, tiled: true, tm: largestDivisorUpTo(Pm.M, tile), tn: largestDivisorUpTo(Pm.N, tile), tk: largestDivisorUpTo(Pm.K, tile),
      border: BORDER[cfg.qk], inner: 'ijk', gran: 'block'
    });
    root.classList.add('stack');
    root.appendChild(el('div', 'att-head', 'Program 0, key block 0: <b>S = Q[0…' + (Pm.M - 1) + '] · K[0…' + (Pm.N - 1) + ']ᵀ</b> — M = Br = ' + Pm.M + ', K = d = ' + Pm.K + ', N = Bc = ' + Pm.N + ', before the 1/√d scale. Array tile ' + sub.tm + '×' + sub.tn + '×' + sub.tk + ', ' + cfg.qk + '.'));
    const host = el('div'); root.appendChild(host);
    this.view = new X.SingleView(host, sub);
    this.cfg = this.view.cfg; this.root = root;
  }
  Object.defineProperty(InnerView.prototype, 'timer', { get() { return this.view.timer; } });
  ['step', 'back', 'go', 'inner', 'finish', 'reset', 'atEnd', 'atStart', 'play', 'stop', 'keydown', 'destroy', 'trace'].forEach((m) => { InnerView.prototype[m] = function () { return this.view[m].apply(this.view, arguments); }; });

  // ---------------- deck mount (shared factory: ../../common/widget.js) ----------------
  const DEFAULTS = { S: 4, T: 4, d: 2, heads: 1, batch: 1, Br: 2, Bc: 2, cores: 1, tile: 1, qk: 'os', pv: 'os', seed: 0 };
  const KNOB_LAB = { Br: 'Br', Bc: 'Bc', cores: 'cores', qk: 'S = QKᵀ flow', heads: 'heads', batch: 'batch', S: 'queries', tile: 'array tile' };
  const { mount, mountAll } = global.NB.widgets({ flash: FlashView, naive: NaiveView, inner: InnerView }, DEFAULTS, problem, KNOB_LAB);

  global.ATT = { problem, refAttention, closeTo, flashPlan, flashState, flashTotals, phaseAt, naivePlan, naiveState, innerAdds, innerProblem, FlashView, NaiveView, InnerView, mount, mountAll, fmt, DEFAULTS, TOL };
  // The deck's shared script calls MM.mountAll(); on the attention deck it mounts attention widgets.
  if (!global.MM) global.MM = { mountAll };
})(typeof window !== 'undefined' ? window : globalThis);
