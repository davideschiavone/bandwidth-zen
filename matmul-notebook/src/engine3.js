/* Point 9 (split-K through memory) and point 10 (weight sets). Built on MM2's helpers. */
(function (global) {
  'use strict';
  const X = global.MM2, H = X._;
  const { el, btn, addCls, counters, Mat, Triple, mix, key, rng, zeros } = H;
  const bname = (kt, nt) => 'B[' + kt + ',' + nt + ']';
  const pct = (a, b) => Math.round(100 * a / b) + '%';

  // =====================================================================
  // 9 · Split-K through memory
  // cfg: P, tm, tn, pieces, cores
  // =====================================================================
  function splitkPlan(P, cfg) {
    const nM = P.M / cfg.tm, nN = P.N / cfg.tn, Pp = cfg.pieces, kp = P.K / Pp;
    const tiles = [];
    for (let p = 0; p < Pp; p++) for (let mt = 0; mt < nM; mt++) for (let nt = 0; nt < nN; nt++) tiles.push({ mt, nt, p, k0: p * kp, k1: (p + 1) * kp });
    const W = Math.ceil(tiles.length / cfg.cores);
    tiles.forEach((t, x) => { t.id = x; t.core = x % cfg.cores; t.wave = Math.floor(x / cfg.cores); });
    const pass2 = Pp > 1 ? Pp : 0;
    return { nM, nN, Pp, kp, tiles, W, pass2, n: 2 * W + pass2 };
  }
  function splitkState(P, cfg, pl, t) {
    const tm = cfg.tm, tn = cfg.tn;
    const q = Math.max(0, t - 2 * pl.W);
    const mem = rng(pl.Pp).map(() => ({ v: zeros(P.M, P.N), w: zeros(P.M, P.N) }));
    const C = zeros(P.M, P.N), T = zeros(P.M, P.N);
    let mul = 0, add1 = 0, written = 0, read = 0, add2 = 0;
    const local = {};
    pl.tiles.forEach((tile) => {
      if (2 * tile.wave + 1 > t) return;
      const vals = rng(tm).map((a) => rng(tn).map((b) => { let s = 0; for (let k = tile.k0; k < tile.k1; k++) s += P.A[tile.mt * tm + a][k] * P.B[k][tile.nt * tn + b]; return s; }));
      mul += tm * tn * pl.kp; add1 += tm * tn * (pl.kp - 1);
      const wr = 2 * tile.wave + 2 <= t;
      local[tile.id] = { vals, wr };
      if (!wr) return;
      for (let a = 0; a < tm; a++) for (let b = 0; b < tn; b++) {
        const i = tile.mt * tm + a, j = tile.nt * tn + b;
        if (pl.Pp === 1) { C[i][j] = vals[a][b]; T[i][j] = 1; }
        else { mem[tile.p].v[i][j] = vals[a][b]; mem[tile.p].w[i][j] = 1; written++; }
      }
    });
    if (pl.Pp > 1) for (let p = 0; p < q; p++) for (let i = 0; i < P.M; i++) for (let j = 0; j < P.N; j++) {
      C[i][j] += mem[p].v[i][j]; T[i][j]++; read++; if (p > 0) add2++;
    }
    return { mem, C, T, mul, add1, add2, written, read, local, q, done: pl.Pp > 1 ? pl.Pp : 1 };
  }

  function gantt(cols, minw) {
    const g = el('div', 'gantt');
    g.style.gridTemplateColumns = '4.6rem repeat(' + cols + ', minmax(' + (minw || '2.4rem') + ', 1fr))';
    return g;
  }
  function put(g, node, r, c, span) { node.style.gridRow = String(r); node.style.gridColumn = c + (span && span > 1 ? ' / span ' + span : ''); g.appendChild(node); return node; }

  function SplitKView(root, cfg) {
    this.cfg = cfg; this.root = root; this.P = cfg.P; root.classList.add('w', 'pv');
    this.baseSpeed = 1100; this.t = 0;
    const P = this.P, pl = this.pl = splitkPlan(P, cfg); this.n = pl.n;
    const blocks = pl.nM * pl.nN;
    const head = el('div', 'w-head'); root.appendChild(head);
    head.innerHTML = '<span class="chip C">each core keeps one C block · os</span>' +
      '<span class="chip">' + pl.tiles.length + ' tiles = ' + blocks + ' output block' + (blocks > 1 ? 's' : '') + ' × ' + pl.Pp + ' piece' + (pl.Pp > 1 ? 's' : '') + ' of K</span>' +
      '<span class="chip">' + cfg.cores + ' core' + (cfg.cores > 1 ? 's' : '') + ' → ' + pl.W + ' wave' + (pl.W > 1 ? 's' : '') + '</span>' +
      (pl.Pp > 1 ? '<span class="chip mem">partials go through memory</span>' : '<span class="chip">no partials: blocks go straight to C</span>');
    const top = el('div', 'pv-top'); root.appendChild(top);
    this.tri = new Triple(P, { tm: cfg.tm, tn: cfg.tn, tk: pl.kp }, pl.Pp > 1 ? 'K cut into ' + pl.Pp + ' pieces<br>of ' + pl.kp : 'K in one piece');
    top.appendChild(this.tri.root);
    const side = el('div', 'stack'); top.appendChild(side);
    this.codeEl = el('pre', 'code'); side.appendChild(this.codeEl);
    const kw = (x) => '<span class="kw">' + x + '</span>', cm = (x) => '<span class="cm">   # ' + x + '</span>';
    this.lines = pl.Pp > 1 ? [
      { v: 'grid', html: kw('parallel for') + ' (mt, nt, piece) ' + kw('in') + ' tiles:' + cm('pass 1, ' + pl.tiles.length + ' tiles') },
      { v: 'sweep', html: '    ' + kw('for') + ' k ' + kw('in') + ' piece:' + cm(pl.kp + ' values of k, in order') },
      { v: 'body', html: '        acc += <span class="ta">A</span>[mt,k] ⊗ <span class="tb">B</span>[k,nt]' },
      { v: 'write', html: '    <span class="tm">memory</span>[piece][mt,nt] = acc' + cm('pale partial block out') },
      { v: 'p2', html: kw('for') + ' piece ' + kw('in') + ' range(' + pl.Pp + '):' + cm('pass 2') },
      { v: 'p2b', html: '    <span class="tc">C</span> += <span class="tm">memory</span>[piece]' + cm('read back, add') }
    ] : [
      { v: 'grid', html: kw('parallel for') + ' (mt, nt) ' + kw('in') + ' tiles:' + cm(pl.tiles.length + ' tiles only') },
      { v: 'sweep', html: '    ' + kw('for') + ' k ' + kw('in') + ' range(' + P.K + '):' + cm('all of K on one core') },
      { v: 'body', html: '        acc += <span class="ta">A</span>[mt,k] ⊗ <span class="tb">B</span>[k,nt]' },
      { v: 'write', html: '    <span class="tc">C</span>[mt,nt] = acc' + cm('finished: straight to C') }
    ];
    this.lineEls = this.lines.map((l) => { const d = el('div', 'ln', l.html); this.codeEl.appendChild(d); return d; });
    side.appendChild(el('div', 'legend', '<span><i class="lg read"></i>read now</span><span><i class="lg part"></i>partial</span><span><i class="lg fin"></i>finished</span><span><i class="lg memw"></i>written to memory</span><span><i class="lg memr"></i>read back (pass 2)</span>'));
    if (pl.Pp > 1) for (let p = 0; p < pl.Pp; p++) { this.tri.A.badge(0, p * pl.kp, P.M, pl.kp, 'piece ' + p, 'soft'); this.tri.B.badge(p * pl.kp, 0, pl.kp, P.N, 'piece ' + p, 'soft'); }
    root.appendChild(el('div', 'pv-lab', 'The chip: what every core is doing right now'));
    this.floor = el('div', 'floor'); root.appendChild(this.floor);
    this.coreEls = rng(cfg.cores).map(() => { const b = el('div', 'corebox'); this.floor.appendChild(b); return b; });
    root.appendChild(el('div', 'pv-lab', 'Memory'));
    this.memEl = el('div', 'membox'); root.appendChild(this.memEl);
    this.memMats = [];
    if (pl.Pp > 1) for (let p = 0; p < pl.Pp; p++) { const m = new Mat('C', P.M, P.N, { title: 'piece ' + p, dims: 'k ' + (p * pl.kp) + '–' + ((p + 1) * pl.kp - 1), rb: cfg.tm, cb: cfg.tn, cls: 'compact' }); this.memMats.push(m); this.memEl.appendChild(m.root); }
    else this.memEl.appendChild(el('div', 'op', 'With one piece nothing partial ever leaves a core: each finished block is written once, into C.'));
    root.appendChild(el('div', 'pv-lab', 'Timeline'));
    this.gw = el('div', 'gantt-wrap'); root.appendChild(this.gw);
    this.buildGantt();
    this.cap = el('div', 'cap'); root.appendChild(this.cap);
    const foot = el('div', 'w-foot'); root.appendChild(foot);
    this.cnt = el('div', 'counters'); foot.appendChild(this.cnt);
    const right = el('div', 'stack right'); right.append(this.controls(null), this.scrubber()); foot.appendChild(right);
    this.keys(); this.go(0);
  }
  SplitKView.prototype.inner = function () { return this.step(); };
  SplitKView.prototype.buildGantt = function () {
    const pl = this.pl, cfg = this.cfg, g = gantt(pl.n);
    put(g, el('div', 'g-h'), 1, 1);
    for (let w = 0; w < pl.W; w++) put(g, el('div', 'g-wave', 'pass 1 · wave ' + (w + 1)), 1, 2 + 2 * w, 2);
    if (pl.pass2) put(g, el('div', 'g-wave mem', 'pass 2'), 1, 2 + 2 * pl.W, pl.pass2);
    put(g, el('div', 'g-lab', 'step'), 2, 1);
    for (let c = 0; c < pl.n; c++) put(g, el('div', 'g-num', String(c + 1)), 2, 2 + c);
    this.bars = [];
    const bar = (cls, txt, r, c, span, s0, s1) => { const b = put(g, el('div', 'bar ' + cls, '<span class="fill"></span><span class="bt">' + txt + '</span>'), r, c, span); this.bars.push({ b, s0, s1 }); return b; };
    for (let c = 0; c < cfg.cores; c++) {
      put(g, el('div', 'g-lab', 'core ' + c), 3 + c, 1);
      for (let w = 0; w < pl.W; w++) {
        const t = pl.tiles.find((x) => x.wave === w && x.core === c);
        if (!t) { put(g, el('div', 'bar idle', '<span class="bt">idle</span>'), 3 + c, 2 + 2 * w, 2); continue; }
        bar('f-os', 'C' + t.mt + t.nt + (pl.Pp > 1 ? ' k' + t.k0 + '–' + (t.k1 - 1) : ''), 3 + c, 2 + 2 * w, 1, 2 * w, 2 * w + 1);
        bar(pl.Pp > 1 ? 'mem' : 'f-os', pl.Pp > 1 ? '→ mem' : '→ C', 3 + c, 3 + 2 * w, 1, 2 * w + 1, 2 * w + 2);
      }
    }
    if (pl.pass2) {
      put(g, el('div', 'g-lab mem', 'pass 2'), 3 + cfg.cores, 1);
      for (let p = 0; p < pl.Pp; p++) bar('mem2', '+ piece ' + p, 3 + cfg.cores, 2 + 2 * pl.W + p, 1, 2 * pl.W + p, 2 * pl.W + p + 1);
    }
    this.cursor = el('div', 'g-cursor'); this.cursor.style.gridRow = '2 / span ' + (cfg.cores + (pl.pass2 ? 2 : 1)); g.appendChild(this.cursor);
    this.gw.appendChild(g);
  };
  SplitKView.prototype.render = function () {
    const P = this.P, cfg = this.cfg, pl = this.pl, tm = cfg.tm, tn = cfg.tn;
    const st = splitkState(P, cfg, pl, this.t);
    const s = this.t - 1, inP1 = s >= 0 && s < 2 * pl.W, w = inP1 ? Math.floor(s / 2) : -1, ph = inP1 ? s % 2 : -1;
    const p2 = s >= 2 * pl.W ? s - 2 * pl.W : -1;
    const active = inP1 ? pl.tiles.filter((x) => x.wave === w) : [];
    const cA = {}, cB = {}, cC = {};
    if (ph === 0) active.forEach((t) => {
      for (let a = 0; a < tm; a++) for (let k = t.k0; k < t.k1; k++) addCls(cA, key(t.mt * tm + a, k), 'read');
      for (let k = t.k0; k < t.k1; k++) for (let b = 0; b < tn; b++) addCls(cB, key(k, t.nt * tn + b), 'read');
    });
    for (let i = 0; i < P.M; i++) for (let j = 0; j < P.N; j++) {
      if (st.T[i][j]) addCls(cC, key(i, j), st.T[i][j] === st.done ? 'fin' : 'part');
      if (p2 >= 0) addCls(cC, key(i, j), 'memr');
      if (ph === 1 && pl.Pp === 1 && active.some((t) => Math.floor(i / tm) === t.mt && Math.floor(j / tn) === t.nt)) addCls(cC, key(i, j), 'write');
    }
    this.tri.A.paint(cA); this.tri.B.paint(cB);
    this.tri.C.paint(cC, (i, j) => (st.T[i][j] ? String(st.C[i][j]) : '·'));
    // memory
    this.memMats.forEach((m, p) => {
      const cl = {};
      for (let i = 0; i < P.M; i++) for (let j = 0; j < P.N; j++) {
        if (!st.mem[p].w[i][j]) continue;
        let c = 'part';
        if (ph === 1 && active.some((t) => t.p === p && Math.floor(i / tm) === t.mt && Math.floor(j / tn) === t.nt)) c += ' memw';
        if (p === p2) c += ' memr';
        else if (p < st.q) c += ' dim';
        addCls(cl, key(i, j), c);
      }
      m.paint(cl, (i, j) => (st.mem[p].w[i][j] ? String(st.mem[p].v[i][j]) : '·'));
    });
    // cores
    this.coreEls.forEach((box, c) => {
      const t = active.find((x) => x.core === c);
      if (!t) {
        box.className = 'corebox idle';
        const msg = this.t === 0 ? 'waiting to start' : p2 >= 0 || this.t >= pl.n ? 'done with pass 1' : 'idle: no tile left for this core in wave ' + (w + 1);
        box.innerHTML = '<h5><span>core ' + c + '</span></h5><div class="op">' + msg + '</div>';
        return;
      }
      box.className = 'corebox';
      const loc = st.local[t.id];
      box.innerHTML = '';
      box.appendChild(el('h5', null, '<span>core ' + c + '</span><span class="muted">tile ' + t.id + '</span>'));
      box.appendChild(el('div', 'op', 'C[' + t.mt + ',' + t.nt + '] from k ' + t.k0 + '–' + (t.k1 - 1) + (pl.Pp > 1 ? ' (piece ' + t.p + ')' : ' (all of K)')));
      const row = el('div', 'cb-row');
      const m = el('div', 'mini'); m.style.gridTemplateColumns = 'repeat(' + tn + ', var(--cell))';
      for (let a = 0; a < tm; a++) for (let b = 0; b < tn; b++) m.appendChild(el('div', 'cell k-C ' + (pl.Pp > 1 ? 'part' : 'fin') + (ph === 1 ? (pl.Pp > 1 ? ' memw' : ' write') : ''), String(loc.vals[a][b])));
      row.append(el('span', 'cb-l', ph === 0 ? 'sums ' + pl.kp + ' term' + (pl.kp > 1 ? 's' : '') + ':' : pl.Pp > 1 ? 'writes →' : 'writes to C →'), m, el('span', 'op', ph === 1 ? (pl.Pp > 1 ? '→ memory, piece ' + t.p : '→ C, finished') : pl.Pp > 1 ? 'partial (pale)' : 'finished'));
      box.appendChild(row);
    });
    // gantt
    this.bars.forEach(({ b, s0, s1 }) => {
      const f = Math.max(0, Math.min(1, (this.t - s0) / (s1 - s0)));
      b.querySelector('.fill').style.width = (100 * f) + '%';
      b.classList.toggle('now', s >= s0 && s < s1);
    });
    this.cursor.style.gridColumn = String(2 + Math.max(0, s)); this.cursor.hidden = this.t === 0;
    this.lineEls.forEach((d, n) => { const v = this.lines[n].v; d.classList.toggle('on', (ph === 0 && v === 'body') || (ph === 1 && v === 'write') || (p2 >= 0 && v === 'p2b')); });
    // caption
    const busy = active.length, idle = cfg.cores - busy, MN = P.M * P.N;
    let cap;
    if (this.t === 0) cap = pl.Pp === 1
      ? 'One piece = plain os. The unit of work is one output block, so there are only <b>' + pl.tiles.length + ' tiles</b> for ' + cfg.cores + ' cores' + (pl.tiles.length < cfg.cores ? ': <b>' + (cfg.cores - pl.tiles.length) + ' cores will sit idle</b> however long the sweep is.' : '.')
      : 'The unit of work is now <b>(output block, piece of K)</b>: ' + pl.nM * pl.nN + ' blocks × ' + pl.Pp + ' pieces = <b>' + pl.tiles.length + ' tiles</b> for ' + cfg.cores + ' cores. Step to run pass 1.';
    else if (ph === 0) cap = '<b>Pass 1, wave ' + (w + 1) + ':</b> ' + busy + ' core' + (busy > 1 ? 's' : '') + ' at once' + (idle ? ', <b>' + idle + ' idle</b>' : '') + '. Each sweeps only its piece of K for its output block' + (pl.Pp > 1 ? ', so its block is <b>partial</b> (pale): ' + pl.kp + ' of the ' + P.K + ' terms.' : ': all ' + P.K + ' terms, finished.');
    else if (ph === 1) cap = pl.Pp > 1 ? '<b class="tm">Write:</b> every core sends its pale partial block out to memory (' + busy * cfg.tm * cfg.tn + ' values).' : 'Every core writes its finished block into C.';
    else cap = '<b class="tm">Pass 2, piece ' + p2 + ':</b> read ' + MN + ' values back from memory and ' + (p2 === 0 ? 'copy them into C (first piece: nothing to add yet)' : 'add them into C (' + MN + ' additions)') + '. ' + (p2 === pl.Pp - 1 ? 'C is finished.' : 'C is still partial.');
    if (this.t === pl.n) cap = 'Done in ' + pl.n + ' steps. ' + (X.eqRef(P, st.C) ? '<span class="ok">Check: C == A @ B ✓</span>' : '<span class="tsp">C differs ✗</span>');
    this.cap.innerHTML = cap;
    const use = pct(pl.tiles.length, pl.W * cfg.cores);
    this.cnt.innerHTML = counters([
      { label: 'step', value: this.t + '/' + pl.n }, { label: 'multiplies', value: st.mul + '/' + P.M * P.N * P.K },
      { label: 'additions pass 1', value: st.add1 }, { label: 'additions pass 2', value: st.add2, cls: 'cmem' },
      { label: 'total', value: (st.add1 + st.add2) + '/' + MN * (P.K - 1) },
      { label: 'to memory', value: pl.Pp > 1 ? st.written + '/' + pl.Pp * MN : '–', cls: 'cmem' }, { label: 'back from memory', value: pl.Pp > 1 ? st.read + '/' + pl.Pp * MN : '–', cls: 'cmem' },
      { label: 'core use (pass 1)', value: use }
    ]);
  };
  SplitKView.prototype.trace = function () {
    const pl = this.pl, out = [];
    for (let s = 0; s < Math.min(this.t, pl.n); s++) {
      if (s < 2 * pl.W) {
        const w = Math.floor(s / 2), act = pl.tiles.filter((x) => x.wave === w);
        out.push('step ' + String(s + 1).padStart(2) + (s % 2 === 0 ? ' pass 1 compute: ' + act.map((t) => 'core' + t.core + ' C[' + t.mt + ',' + t.nt + '] k' + t.k0 + '-' + (t.k1 - 1)).join(' | ') : ' write: ' + act.length + ' blocks → ' + (pl.Pp > 1 ? 'memory' : 'C')) + (act.length < this.cfg.cores ? ' | ' + (this.cfg.cores - act.length) + ' idle' : ''));
      } else { const p = s - 2 * pl.W; out.push('step ' + String(s + 1).padStart(2) + ' pass 2: C ' + (p ? '+=' : '=') + ' memory[piece ' + p + ']'); }
    }
    return out;
  };
  mix(SplitKView.prototype);

  // =====================================================================
  // 10 · Weight sets: one weight-stationary core with S slots
  // cfg: P, tn, tk, slots, load (steps to write one block), sched ('seq'|'overlap'), group ('block'|'row'), both
  // =====================================================================
  function wsTasks(P, cfg) {
    const nN = P.N / cfg.tn, nK = P.K / cfg.tk;
    const g = cfg.group === 'row' ? Math.min(cfg.slots, nK) : 1;
    const tasks = [];
    for (let nt = 0; nt < nN; nt++) for (let k0 = 0; k0 < nK; k0 += g) {
      const kts = rng(Math.min(g, nK - k0)).map((x) => k0 + x);
      for (let i = 0; i < P.M; i++) kts.forEach((kt) => tasks.push({ i, kt, nt, b: kt + ',' + nt, lastK: kt === nK - 1 }));
    }
    return { tasks, nN, nK, g };
  }
  function wsSchedule(P, cfg, sched) {
    const { tasks, g } = wsTasks(P, cfg);
    const blocks = [], last = {};
    tasks.forEach((t, x) => { if (!(t.b in last)) blocks.push(t.b); last[t.b] = x; });
    const S = cfg.slots, L = cfg.load;
    const slot = rng(S).map(() => null);
    const loads = [], comps = [];
    let lIdx = 0, lBusy = 0, ti = 0, t = 0;
    while (ti < tasks.length && t < 5000) {
      const need = tasks[ti].b;
      const resident = slot.findIndex((q) => q && q.b === need);
      if (t >= lBusy && lIdx < blocks.length) {
        const b = blocks[lIdx];
        const free = slot.findIndex((q) => !q || last[q.b] < ti);
        const ok = sched === 'overlap' || (resident < 0 && b === need);
        if (free >= 0 && ok) { slot[free] = { b, ready: t + L }; loads.push({ b, s: free, start: t, end: t + L }); lBusy = t + L; lIdx++; }
      }
      const s = slot.findIndex((q) => q && q.b === need && q.ready <= t);
      if (s >= 0) { comps.push({ task: tasks[ti], t, s }); ti++; }
      t++;
    }
    return { tasks, blocks, loads, comps, T: t, g, last };
  }
  function applyTask(P, s, cfg, task) {
    for (let k = task.kt * cfg.tk; k < (task.kt + 1) * cfg.tk; k++) for (let j = task.nt * cfg.tn; j < (task.nt + 1) * cfg.tn; j++) X.apply(P, s, task.i, j, k);
  }

  function WSView(root, cfg) {
    this.cfg = cfg; this.root = root; this.P = cfg.P; root.classList.add('w', 'pv', 'wsv');
    this.baseSpeed = 650; this.t = 0;
    const P = this.P;
    this.sch = wsSchedule(P, cfg, cfg.sched); this.n = this.sch.T;
    this.other = cfg.both ? wsSchedule(P, cfg, cfg.sched === 'seq' ? 'overlap' : 'seq') : null;
    const nK = P.K / cfg.tk;
    const head = el('div', 'w-head'); root.appendChild(head);
    head.innerHTML = '<span class="chip B">1 core · ' + cfg.slots + ' weight set' + (cfg.slots > 1 ? 's' : '') + ' (slots) of ' + cfg.tk + '×' + cfg.tn + ' B · ws</span>' +
      '<span class="chip">computes with one slot at a time</span><span class="chip">writing one block into a slot: ' + cfg.load + ' step' + (cfg.load > 1 ? 's' : '') + '</span>' +
      '<span class="chip">' + (cfg.sched === 'seq' ? 'load, then compute' : 'loading overlapped with compute') + '</span>' +
      (cfg.group === 'row' ? '<span class="chip">each row of A goes through ' + this.sch.g + ' slot' + (this.sch.g > 1 ? 's' : '') + ' in turn</span>' : '<span class="chip">all rows of A through one block, then the next</span>');
    const top = el('div', 'pv-top'); root.appendChild(top);
    this.tri = new Triple(P, { tm: P.M, tn: cfg.tn, tk: cfg.tk }, 'B blocks<br>' + cfg.tk + '×' + cfg.tn);
    top.appendChild(this.tri.root);
    const side = el('div', 'stack'); top.appendChild(side);
    this.coreEl = el('div', 'wscore'); side.appendChild(this.coreEl);
    this.portEl = el('div', 'ws-port'); side.appendChild(this.portEl);
    side.appendChild(el('div', 'legend', '<span><i class="lg read"></i>read now</span><span><i class="lg stay"></i>slot computing</span><span><i class="lg load"></i>slot being written</span><span><i class="lg part"></i>partial</span><span><i class="lg fin"></i>finished</span>'));
    root.appendChild(el('div', 'pv-lab', cfg.both ? 'Timeline, both ways (same scale)' : 'Timeline'));
    this.tlEl = el('div', 'stack'); root.appendChild(this.tlEl);
    const draw = (sch, anim) => {
      const cols = Math.max(this.sch.T, this.other ? this.other.T : 0);
      const lab = sch === this.sch ? (cfg.sched === 'seq' ? 'Load, then compute' : 'Overlapped') : (cfg.sched === 'seq' ? 'Overlapped' : 'Load, then compute');
      const box = el('div', 'tl-box' + (anim ? ' anim' : ''));
      box.appendChild(el('div', 'tl-title', '<b>' + lab + '</b> — done after <b>' + sch.T + '</b> steps' + (anim ? ' <span class="muted">(animated)</span>' : '')));
      const wrap = el('div', 'gantt-wrap'); box.appendChild(wrap);
      const g = gantt(cols, cols > 30 ? '1.4rem' : '1.9rem'); wrap.appendChild(g);
      put(g, el('div', 'g-lab', 'step'), 1, 1);
      for (let c = 0; c < cols; c++) put(g, el('div', 'g-num', String(c + 1)), 1, 2 + c);
      put(g, el('div', 'g-lab', 'write port'), 2, 1); put(g, el('div', 'g-lab', 'compute'), 3, 1);
      const bars = [];
      sch.loads.forEach((l) => { const [kt, nt] = l.b.split(','); bars.push({ b: put(g, el('div', 'bar wld', '<span class="fill"></span><span class="bt">B' + kt + nt + '→s' + l.s + '</span>'), 2, 2 + l.start, l.end - l.start), s0: l.start, s1: l.end }); });
      let x = 0;
      while (x < sch.T) {
        const c = sch.comps.find((q) => q.t === x);
        if (!c) { let y = x; while (y < sch.T && !sch.comps.some((q) => q.t === y)) y++; bars.push({ b: put(g, el('div', 'bar wait', '<span class="fill"></span><span class="bt">wait</span>'), 3, 2 + x, y - x), s0: x, s1: y }); x = y; continue; }
        let y = x, i1 = c.task.i;
        while (y + 1 < sch.T) { const d = sch.comps.find((q) => q.t === y + 1); if (!d || d.task.b !== c.task.b) break; y++; i1 = d.task.i; }
        const [kt, nt] = c.task.b.split(',');
        bars.push({ b: put(g, el('div', 'bar wcp', '<span class="fill"></span><span class="bt">B' + kt + nt + ' r' + c.task.i + (i1 !== c.task.i ? '–' + i1 : '') + '</span>'), 3, 2 + x, y - x + 1), s0: x, s1: y + 1 });
        x = y + 1;
      }
      if (sch.T < cols) put(g, el('div', 'g-end', '✓ ' + sch.T), 3, 2 + sch.T, cols - sch.T);
      if (anim) { this.bars = bars; this.cursor = el('div', 'g-cursor'); this.cursor.style.gridRow = '1 / span 3'; g.appendChild(this.cursor); }
      this.tlEl.appendChild(box);
    };
    const order = this.other ? (cfg.sched === 'seq' ? [[this.sch, true], [this.other, false]] : [[this.other, false], [this.sch, true]]) : [[this.sch, true]];
    order.forEach(([s, a]) => draw(s, a));
    this.cap = el('div', 'cap'); root.appendChild(this.cap);
    const foot = el('div', 'w-foot'); root.appendChild(foot);
    this.cnt = el('div', 'counters'); foot.appendChild(this.cnt);
    const right = el('div', 'stack right'); right.append(this.controls(null), this.scrubber()); foot.appendChild(right);
    this.keys(); this.go(0);
  }
  WSView.prototype.inner = function () { return this.step(); };
  WSView.prototype.render = function () {
    const P = this.P, cfg = this.cfg, sch = this.sch, now = this.t - 1;
    const s = X.newAcc(P); let cur = null, done = 0;
    sch.comps.forEach((c) => { if (c.t <= now) { applyTask(P, s, cfg, c.task); done++; if (c.t === now) cur = c; } });
    const loadNow = sch.loads.find((l) => l.start <= now && now < l.end);
    const slotSt = rng(cfg.slots).map((q) => {
      const ls = sch.loads.filter((l) => l.s === q && l.start <= now); const ld = ls[ls.length - 1];
      if (!ld) return { st: 'empty' };
      if (now < ld.end) return { st: 'loading', ld, prog: (now - ld.start + 1) / (ld.end - ld.start) };
      if (cur && cur.s === q) return { st: 'on', ld };
      return { st: sch.last[ld.b] < done ? 'used' : 'ready', ld };
    });
    // matrices
    const cA = {}, cB = {}, cC = {};
    this.tri.B.clearBadges();
    slotSt.forEach((q, n) => {
      if (!q.ld) return;
      const [kt, nt] = q.ld.b.split(',').map(Number);
      if (q.st === 'used') return;
      for (let a = 0; a < cfg.tk; a++) for (let b = 0; b < cfg.tn; b++) addCls(cB, key(kt * cfg.tk + a, nt * cfg.tn + b), q.st === 'on' ? 'read stay' : q.st === 'loading' ? 'sweep' : 'stayq');
      this.tri.B.badge(kt * cfg.tk, nt * cfg.tn, cfg.tk, cfg.tn, 'slot ' + n + (q.st === 'loading' ? ' ←' : ''), 'tag' + (q.st === 'loading' ? ' ld' : ''));
    });
    if (cur) for (let k = cur.task.kt * cfg.tk; k < (cur.task.kt + 1) * cfg.tk; k++) addCls(cA, key(cur.task.i, k), 'read');
    for (let i = 0; i < P.M; i++) for (let j = 0; j < P.N; j++) if (s.T[i][j]) addCls(cC, key(i, j), s.T[i][j] === P.K ? 'fin' : 'part');
    if (cur) for (let j = cur.task.nt * cfg.tn; j < (cur.task.nt + 1) * cfg.tn; j++) addCls(cC, key(cur.task.i, j), 'write');
    this.tri.A.paint(cA); this.tri.B.paint(cB); this.tri.C.paint(cC, (i, j) => (s.T[i][j] ? String(s.C[i][j]) : '·'));
    this.tri.A.heads(cur ? [cur.task.i] : null, null);
    // core diagram
    const mini = (cells, cols) => { const m = el('div', 'mini'); m.style.gridTemplateColumns = 'repeat(' + cols + ', var(--cell))'; cells.forEach((c) => m.appendChild(c)); return m; };
    const ce = this.coreEl; ce.innerHTML = '';
    const inBox = el('div', 'ws-io'); inBox.appendChild(el('div', 'cb-l', cur ? 'row ' + cur.task.i + ' of A, k ' + cur.task.kt * cfg.tk + '–' + ((cur.task.kt + 1) * cfg.tk - 1) : 'row of A in'));
    inBox.appendChild(mini(rng(cfg.tk).map((a) => el('div', 'cell k-A' + (cur ? ' read' : ''), cur ? String(P.A[cur.task.i][cur.task.kt * cfg.tk + a]) : '·')), cfg.tk));
    const slotsBox = el('div', 'ws-slots');
    slotSt.forEach((q, n) => {
      const box = el('div', 'slot ' + q.st);
      const [kt, nt] = q.ld ? q.ld.b.split(',').map(Number) : [0, 0];
      const label = { empty: 'empty', loading: 'being written ' + Math.round(q.prog * 100) + '%', on: 'computing', ready: 'loaded, waiting', used: 'used up' }[q.st];
      box.appendChild(el('div', 'slot-h', '<b>slot ' + n + '</b><span>' + (q.ld ? bname(kt, nt) : '') + '</span>'));
      const rows = q.st === 'loading' ? Math.ceil(q.prog * cfg.tk) : cfg.tk;
      box.appendChild(mini(rng(cfg.tk * cfg.tn).map((x) => { const a = Math.floor(x / cfg.tn), b = x % cfg.tn; const show = q.ld && q.st !== 'empty' && a < rows; return el('div', 'cell k-B' + (q.st === 'on' ? ' stay' : '') + (show ? '' : ' ghost'), show ? String(P.B[kt * cfg.tk + a][nt * cfg.tn + b]) : '·'); }), cfg.tn));
      box.appendChild(el('div', 'slot-s', label));
      slotsBox.appendChild(box);
    });
    const accBox = el('div', 'ws-io');
    const accRow = cur ? cur.task.i : null, accCols = cur ? rng(cfg.tn).map((b) => cur.task.nt * cfg.tn + b) : [];
    accBox.appendChild(el('div', 'cb-l', cur ? 'accumulator → C[' + accRow + ', ' + accCols[0] + '–' + accCols[accCols.length - 1] + ']' : 'accumulator'));
    accBox.appendChild(mini(rng(cfg.tn).map((b) => { if (!cur) return el('div', 'cell k-C', '·'); const j = accCols[b]; return el('div', 'cell k-C ' + (s.T[accRow][j] === P.K ? 'fin' : 'part'), String(s.C[accRow][j])); }), cfg.tn));
    ce.append(inBox, el('span', 'arrow', '→'), slotsBox, el('span', 'arrow', '→'), accBox);
    this.portEl.innerHTML = loadNow ? '<span class="tb">write port:</span> memory → <b>' + bname.apply(null, loadNow.b.split(',')) + '</b> → slot ' + loadNow.s + ' <span class="muted">(' + (now - loadNow.start + 1) + '/' + (loadNow.end - loadNow.start) + ')</span>' : '<span class="muted">write port idle</span>';
    // timeline
    this.bars.forEach(({ b, s0, s1 }) => { const f = Math.max(0, Math.min(1, (this.t - s0) / (s1 - s0))); b.querySelector('.fill').style.width = (100 * f) + '%'; b.classList.toggle('now', now >= s0 && now < s1); });
    this.cursor.style.gridColumn = String(2 + Math.max(0, now)); this.cursor.hidden = this.t === 0;
    // caption
    let cap;
    if (this.t === 0) cap = cfg.group === 'row'
      ? 'The slots will hold successive K-pieces of the <b>same</b> column of B. Each row of A goes through slot 0, then slot 1, adding into <b>one accumulator</b>: that row of C finishes right here.'
      : 'The core can <b>store</b> ' + cfg.slots + ' block' + (cfg.slots > 1 ? 's' : '') + ' of B but computes with only one. Step to watch the write port (top timeline row) and the compute (bottom row).';
    else if (cur) {
      const b = bname(cur.task.kt, cur.task.nt);
      cap = 'Row ' + cur.task.i + ' of A goes through <b>slot ' + cur.s + '</b> (' + b + '): ' + cfg.tk * cfg.tn + ' multiplies, ' + cfg.tn + ' results added into the accumulator.';
      if (loadNow) cap += ' <b>At the same time</b> the write port fills slot ' + loadNow.s + ' with ' + bname.apply(null, loadNow.b.split(',')) + ': the loading is hidden.';
      if (cfg.group === 'row' && cur.task.lastK) cap += ' That was the last K-piece: <b class="tc">C[' + cur.task.i + ', ' + accCols[0] + '–' + accCols[accCols.length - 1] + '] is finished</b>, with no additions between cores.';
    } else if (loadNow) cap = (cfg.sched === 'seq' ? 'Load, then compute: ' : '') + '<b>Compute waits</b> — the block it needs is still being written into slot ' + loadNow.s + '.';
    if (this.t === this.n) cap = 'Done after ' + this.n + ' steps' + (this.other ? ' (the other schedule takes ' + this.other.T + ')' : '') + '. ' + (X.eqRef(P, s.C) ? '<span class="ok">Check: C == A @ B ✓</span>' : '<span class="tsp">C differs ✗</span>');
    this.cap.innerHTML = cap;
    const loaded = sch.loads.filter((l) => l.end <= this.t).length;
    this.cnt.innerHTML = counters([
      { label: 'time step', value: this.t + '/' + this.n }, { label: 'multiplies', value: s.mul + '/' + P.M * P.N * P.K },
      { label: 'additions over time', value: s.add + '/' + P.M * P.N * (P.K - 1) }, { label: 'additions across cores', value: 0, cls: 'csp' },
      { label: 'partial now', value: s.part }, { label: 'most partial at once', value: s.maxPart },
      { label: 'blocks written into slots', value: loaded + '/' + sch.blocks.length }, { label: 'compute busy', value: pct(sch.comps.filter((c) => c.t <= now).length, Math.max(1, this.t)) }
    ]);
  };
  WSView.prototype.trace = function () {
    const out = [], sch = this.sch;
    for (let t = 0; t < Math.min(this.t, sch.T); t++) {
      const c = sch.comps.find((q) => q.t === t), l = sch.loads.find((q) => q.start <= t && t < q.end);
      out.push('t=' + String(t + 1).padStart(2) + ': ' + (c ? 'compute row ' + c.task.i + ' through slot ' + c.s + ' (' + bname.apply(null, c.task.b.split(',')) + ')' : 'compute WAITS') + (l ? ' | write port: ' + bname.apply(null, l.b.split(',')) + ' -> slot ' + l.s + ' (' + (t - l.start + 1) + '/' + (l.end - l.start) + ')' : ''));
    }
    return out;
  };
  mix(WSView.prototype);

  // ---------------- deck wrapper with in-slide knobs ----------------
  const KLAB = {
    pieces: ['P (pieces of K)', null], cores: ['cores', null], load: ['steps to write a block', null], slots: ['weight sets (slots)', null],
    sched: ['schedule', { seq: 'load, then compute', overlap: 'overlapped' }], group: ['order', { block: 'block by block', row: 'row through all slots' }]
  };
  function knobbed(View) {
    function W(root, cfg) {
      const P = X.problem(cfg.M || 4, cfg.N || 4, cfg.K || 4, 0);
      const st = Object.assign({}, cfg);
      root.classList.add('stack');
      const knobs = el('div', 'pv-knobs'); root.appendChild(knobs);
      const host = el('div'); root.appendChild(host);
      const make = () => {
        if (this.view) this.view.destroy();
        host.innerHTML = ''; const d = el('div'); host.appendChild(d);
        this.view = new View(d, Object.assign({}, st, { P }));
        knobs.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(st[b.dataset.k]) === b.dataset.v ? 'true' : 'false'));
      };
      (cfg.knobs || []).forEach((kn) => {
        const grp = el('div', 'seg'); grp.appendChild(el('span', 'seg-l', KLAB[kn.k][0]));
        kn.v.forEach((v) => { const b = btn(KLAB[kn.k][1] ? KLAB[kn.k][1][v] : String(v), kn.k + ' ' + v, () => { st[kn.k] = v; make(); }); b.dataset.k = kn.k; b.dataset.v = String(v); grp.appendChild(b); });
        knobs.appendChild(grp);
      });
      if (!(cfg.knobs || []).length) knobs.hidden = true;
      make(); this.root = root;
    }
    ['step', 'back', 'inner', 'finish', 'reset', 'atEnd', 'atStart', 'play', 'stop'].forEach((m) => { W.prototype[m] = function () { return this.view[m].apply(this.view, arguments); }; });
    return W;
  }
  X.TYPES.skm = knobbed(SplitKView);
  X.TYPES.wsets = knobbed(WSView);
  Object.assign(X, { splitkPlan, splitkState, SplitKView, wsSchedule, wsTasks, applyTask, WSView });
})(typeof window !== 'undefined' ? window : globalThis);
