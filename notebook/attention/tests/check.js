// Checks for the attention notebook.
//   npm install      (once, in notebook/: installs jsdom for the page checks)
//   npm test
// Part 1 needs nothing but Node: it runs the simulators over many configurations and checks O against
// softmax(QKᵀ/√d)V and every counter against its closed form — the same formulas `bwz attention` uses.
// Part 2 loads the built deck.html and notebook.html in jsdom and drives every animation to its end.
'use strict';
const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..');
let failures = 0;
const fail = (msg) => { failures++; console.log('  FAIL ' + msg); };

// ---------------- part 1: simulators ----------------
require(path.join(ROOT, '..', 'matmul', 'src', 'engine2.js'));
require(path.join(ROOT, 'src', 'engine.js'));
const A = globalThis.ATT, X = globalThis.MM2;
let n = 0;
for (const S of [1, 2, 3, 4, 5, 7]) for (const T of [1, 2, 3, 4, 6, 8]) for (const d of [1, 2, 3, 4]) for (const [heads, batch] of [[1, 1], [2, 1], [3, 1], [2, 2]]) {
  const P = A.problem({ S, T, d, heads, batch, seed: (S * 7 + T * 3 + d + heads) % 6 });
  for (let Br = 1; Br <= S; Br++) for (let Bc = 1; Bc <= T; Bc++) for (const cores of [1, 2, 4, 5]) for (const [qk, pv] of [['os', 'os'], ['ws', 'is'], ['is', 'ws']]) {
    const cfg = { Br, Bc, cores, tile: 1 + ((Br + Bc) % 2), qk, pv };
    const pl = A.flashPlan(P, cfg), st = A.flashState(P, cfg, pl, pl.n), T0 = A.flashTotals(P, pl, cfg); n++;
    const qb = Math.ceil(S / Br), kb = Math.ceil(T / Bc), HT = heads * batch;
    // The closed forms, written out here rather than read from the engine.
    let streams = 0; const used = Math.min(cores, HT * qb);
    for (let h = 0; h < HT; h++) streams += Math.floor(((h + 1) * qb - 1) / used) - Math.floor(h * qb / used) + 1;
    const want = { mul: 2 * HT * S * T * d, exps: HT * S * T, rescaled: HT * S * d * (kb - 1), normalised: HT * S * d, qRead: HT * S * d, oWritten: HT * S * d, kvRead: 2 * streams * T * d, innerAdds: T0.innerAdds };
    const bad = Object.keys(want).filter((k) => st[k] !== want[k]);
    if (!P.REF.every((R, h) => A.closeTo(st.O[h], R))) bad.push('O');
    if (pl.programs !== HT * qb || pl.W !== Math.ceil(HT * qb / used) || pl.n !== pl.W * (3 * kb + 1)) bad.push('plan');
    if (bad.length) fail(['flash', S, T, d, heads, batch, Br, Bc, cores, qk, pv, bad.join('+')].join());
  }
  if (heads * batch === 1) {
    const st = A.naiveState(P, A.naivePlan(P).n); n++;
    if (!A.closeTo(st.O, P.REF[0]) || st.sW !== S * T || st.sR !== S * T || st.pW !== S * T || st.pR !== S * T || st.mul !== 2 * S * T * d) fail(['naive', S, T, d].join());
  }
}
// Inside a block: the matmul notebook's own plan must compute the block of scores exactly.
for (const S of [1, 2, 4]) for (const T of [1, 3, 4]) for (const d of [1, 2, 4]) for (const flow of ['os', 'ws', 'is']) {
  const P = A.problem({ S, T, d, seed: 3 }), Pm = A.innerProblem(P, { Br: S, Bc: T, cores: 1 });
  const pl = X.singlePlan(Pm, { tiled: true, tm: 1, tn: 1, tk: 1, border: { os: 'ijk', ws: 'jki', is: 'ikj' }[flow], inner: 'ijk', gran: 'block' });
  const s = X.newAcc(Pm); pl.ops.forEach((o) => X.apply(Pm, s, o.i, o.j, o.k)); n++;
  if (!X.eqRef(Pm, s.C)) fail(['inner', S, T, d, flow].join());
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
        let guard = 0; while (wd.step() && guard++ < 5000) {}
        widgets++;
        if (/✗/.test(node.textContent) || !/✓/.test(node.textContent)) fail('deck widget ' + i + (b ? ' ' + b.dataset.k + '=' + b.dataset.v : '') + ' did not end with a passing check');
      });
    });
    gridsOk(w, 'deck');
    const key = (k) => w.document.dispatchEvent(new w.KeyboardEvent('keydown', { key: k, bubbles: true }));
    for (let i = 0; i < 6000; i++) key('ArrowRight');
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
    for (const [S, T, d] of [[4, 4, 2], [3, 7, 3], [8, 8, 4], [1, 5, 1]]) {
      for (const mode of ['naive', 'flash', 'inner']) {
        Object.assign(L.S, { S, T, d, mode, seed: 2, Br: 3, Bc: 2, cores: 3, heads: 2, batch: 2, tile: 2, qk: 'ws', pv: 'is', hsel: -1 });
        L.rebuild(); finish([mode, S, T, d].join());
      }
    }
    L.chapter('heads'); [...w.document.querySelectorAll('#knobs button')].forEach((b) => b.click());
    L.chapter('play'); L.view.step(); L.view.inner(); L.view.back();
    errors.forEach((e) => fail('notebook script error: ' + e));
    console.log('notebook: ' + L.CH.length + ' chapters, ' + runs + ' runs');
  }
}
console.log(failures ? failures + ' FAILURE(S)' : 'all checks passed');
process.exit(failures ? 1 : 0);
