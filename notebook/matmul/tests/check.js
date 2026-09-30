// Checks for the matmul explainer.
//   npm install      (once, installs jsdom for the page checks)
//   npm test
// Part 1 needs nothing but Node: it runs the simulators over many configurations.
// Part 2 loads the built deck.html and notebook.html in jsdom and drives every animation to its end.
'use strict';
const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..');
let failures = 0;
const fail = (msg) => { failures++; console.log('  FAIL ' + msg); };

// ---------------- part 1: simulators ----------------
require(path.join(ROOT, 'src/engine.js'));
require(path.join(ROOT, 'src/engine2.js'));
require(path.join(ROOT, 'src/engine3.js'));
const X = globalThis.MM2;
let n = 0;
for (const r of globalThis.MM.selfTest()) { n++; if (!r.ok) fail('running example ' + r.what); }
for (const M of [1, 2, 3, 4, 6, 8]) for (const N of [1, 2, 4, 5, 8]) for (const K of [1, 2, 3, 4, 8]) {
  const P = X.problem(M, N, K, (M * 31 + N * 7 + K) % 5);
  const MNK = M * N * K, ADD = M * N * (K - 1);
  for (const tm of X.divisors(M)) for (const tn of X.divisors(N)) for (const tk of X.divisors(K)) {
    for (const inner of X.ORDERS) for (const gran of ['op', 'inner', 'block']) {
      const pl = X.singlePlan(P, { tiled: true, tm, tn, tk, border: X.ORDERS[(tm + tn + tk) % 6], inner, gran });
      const s = X.newAcc(P); pl.ops.forEach((o) => X.apply(P, s, o.i, o.j, o.k)); n++;
      if (!X.eqRef(P, s.C) || s.mul !== MNK || s.add !== ADD) fail(['single', M, N, K, tm, tn, tk, inner, gran].join());
    }
    const nK = K / tk;
    for (const cores of [1, 3, 4, 8]) for (const flow of ['os', 'ws', 'is']) for (const g of X.divisors(nK)) {
      const cfg = { tm, tn, tk, cores, flow, g }; const pl = X.parallelPlan(P, cfg); const st = X.parallelState(P, cfg, pl, pl.n); n++;
      if (!X.eqRef(P, st.C) || st.mul !== MNK || st.tadd + st.sadd !== ADD) fail(['parallel', M, N, K, tm, tn, tk, cores, flow, g].join());
    }
  }
  for (const tm of X.divisors(M)) for (const tn of X.divisors(N)) for (const pieces of X.divisors(K)) for (const cores of [1, 3, 8]) {
    const cfg = { tm, tn, pieces, cores }; const pl = X.splitkPlan(P, cfg); const st = X.splitkState(P, cfg, pl, pl.n); n++;
    const mem = pieces > 1 ? pieces * M * N : 0;
    if (!X.eqRef(P, st.C) || st.mul !== MNK || st.add1 + st.add2 !== ADD || st.written !== mem || st.read !== mem) fail(['splitk', M, N, K, tm, tn, pieces, cores].join());
  }
  for (const tn of X.divisors(N)) for (const tk of X.divisors(K)) for (const slots of [1, 2, 3]) for (const load of [1, 2, 6]) for (const sched of ['seq', 'overlap']) for (const group of ['block', 'row']) {
    const cfg = { tn, tk, slots, load, sched, group }; const sc = X.wsSchedule(P, cfg, sched); n++;
    const s = X.newAcc(P); sc.comps.forEach((c) => X.applyTask(P, s, cfg, c.task));
    const oneComputePerStep = new Set(sc.comps.map((c) => c.t)).size === sc.comps.length;
    const oneLoadAtATime = sc.loads.every((l, i) => i === 0 || l.start >= sc.loads[i - 1].end);
    if (!X.eqRef(P, s.C) || s.mul !== MNK || s.add !== ADD || !oneComputePerStep || !oneLoadAtATime) fail(['weightsets', M, N, K, tn, tk, slots, load, sched, group].join());
  }
}
console.log('simulators: ' + n + ' configurations checked');

// ---------------- part 2: pages ----------------
let JSDOM;
try { ({ JSDOM } = require('jsdom')); } catch (e) { console.log('pages: skipped (run "npm install" to get jsdom)'); JSDOM = null; }
function load(file) {
  const errors = [];
  const dom = new JSDOM(fs.readFileSync(path.join(ROOT, file), 'utf8'), {
    runScripts: 'dangerously', pretendToBeVisual: true,
    beforeParse(w) { w.addEventListener('error', (e) => errors.push(e.message)); w.history.replaceState = () => {}; }
  });
  return { w: dom.window, errors };
}
function gridsOk(w, tag) {
  w.document.querySelectorAll('.mat-grid').forEach((g) => {
    const seen = new Set();
    [...g.children].filter((c) => !c.classList.contains('badge')).forEach((c) => {
      const k = c.style.gridRow + '/' + c.style.gridColumn;
      if (!c.style.gridRow || seen.has(k)) fail(tag + ': matrix cell not pinned or overlapping');
      seen.add(k);
    });
  });
}
if (JSDOM) {
  { // deck
    const { w, errors } = load('deck.html');
    let widgets = 0;
    w.document.querySelectorAll('[data-widget]').forEach((node, i) => {
      const wd = node._widget; if (!wd || !wd.step) return;
      const knobs = [...node.querySelectorAll('.pv-knobs button')];
      (knobs.length ? knobs : [null]).forEach((b) => {
        if (b) b.click();
        let guard = 0; while (wd.step() && guard++ < 1000) {}
        widgets++;
        const cap = node.textContent;
        if (/✗/.test(cap) || (!/✓/.test(cap))) fail('deck widget ' + i + (b ? ' ' + b.dataset.k + '=' + b.dataset.v : '') + ' did not end with a passing check');
      });
    });
    gridsOk(w, 'deck');
    const key = (k) => w.document.dispatchEvent(new w.KeyboardEvent('keydown', { key: k, bubbles: true }));
    for (let i = 0; i < 4000; i++) key('ArrowRight');
    const cnt = w.document.getElementById('cnt').textContent.split('/').map((x) => +x.trim());
    if (cnt[0] !== cnt[1]) fail('deck: arrow keys did not reach the last slide');
    errors.forEach((e) => fail('deck script error: ' + e));
    console.log('deck: ' + widgets + ' animation runs, ' + cnt[1] + ' slides');
  }
  { // notebook
    const { w, errors } = load('notebook.html');
    const L = w.LAB; let runs = 0;
    const finish = (tag) => { L.view.finish(); runs++; if (!/✓/.test(w.document.getElementById('view').textContent)) fail('notebook ' + tag); };
    L.CH.forEach((c) => { L.chapter(c.id); finish('chapter ' + c.id); gridsOk(w, 'notebook ' + c.id); });
    for (const [M, N, K] of [[4, 4, 4], [3, 6, 4], [8, 8, 8], [2, 5, 6]]) {
      for (const mode of ['single', 'many', 'splitk', 'wsets']) {
        Object.assign(L.S, { M, N, K, mode, seed: 2, tm: 2, tn: 2, tk: 2, cores: 3, flow: 'ws', g: 1, pieces: 2, slots: 2, load: 2, sched: 'overlap', group: 'row', inner: 'kji', gran: 'inner', tiled: true });
        L.rebuild(); finish([mode, M, N, K].join());
      }
    }
    L.chapter('wsA'); [...w.document.querySelectorAll('#knobs button')].forEach((b) => b.click());
    errors.forEach((e) => fail('notebook script error: ' + e));
    console.log('notebook: ' + L.CH.length + ' chapters, ' + runs + ' runs');
  }
}
console.log(failures ? failures + ' FAILURE(S)' : 'all checks passed');
process.exit(failures ? 1 : 0);
