// Checks for the decoder notebook.
//   npm install      (once, in notebook/: installs jsdom for the page checks)
//   npm test
// Part 1: with the cache and without, every token's output equals a full causal recompute; with the
// cache the work is exactly one causal pass; every priced step adds up to its closed form.
// Part 2: loads the built deck.html and notebook.html in jsdom and drives every animation to its end.
'use strict';
const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..');
let failures = 0;
const fail = (msg) => { failures++; console.log('  FAIL ' + msg); };

for (const f of [['matmul', 'src', 'engine2.js'], ['common', 'widget.js'], ['attention', 'src', 'engine.js'], ['encoder', 'src', 'engine.js']]) require(path.join(ROOT, '..', ...f));
require(path.join(ROOT, 'src', 'engine.js'));
const D = globalThis.DEC;
let n = 0;
for (const Tp of [1, 2, 3, 6]) for (const G of [1, 2, 4, 6]) for (const [d, h, kvh] of [[2, 1, 1], [2, 2, 1], [4, 2, 2], [4, 2, 1], [4, 4, 2], [8, 4, 1], [8, 8, 2]]) for (const act of ['relu', 'gelu', 'swiglu', 'geglu']) for (const norm of ['rmsnorm', 'layernorm', 'none']) for (const cache of [true, false]) {
  const P = D.problem({ Tp, G, d, h, kvh, ffn: 8, act, norm, seed: (Tp * 5 + G + d) % 7, cache }); n++;
  const total = P.steps.reduce((s, x) => s + x.ops, 0), once = D.rowOps(P, 0, P.T);
  const kvRead = P.steps.reduce((s, x) => s + x.kvRead, 0), dkv = kvh * d / h;
  const wantRead = cache ? 2 * dkv * (rng(G).reduce((s, j) => s + Tp + j, 0)) : 0;
  if (!P.outs.every((r, p) => D.close(r, P.REF[p])) || (cache ? total !== once : total <= once && G > 0) || kvRead !== wantRead) fail(['gen', Tp, G, d, h, kvh, act, norm, cache].join());
}
function rng(k) { return Array.from({ length: k }, (_, i) => i); }
for (const phase of ['decode', 'prefill']) for (const [d, h, kvh, ffn, act] of [[768, 12, 12, 3072, 'gelu'], [4096, 32, 8, 14336, 'swiglu'], [8192, 64, 8, 28672, 'swiglu']]) for (const L of [1, 4096, 131072]) for (const batch of [1, 32]) for (const chip of ['a100', 'h100', 'chip_a']) {
  const o = { phase, d, h, kvh, ffn, act, norm: 'rmsnorm', vocab: 32000, layers: 32, L, S: L, batch };
  const rows = D.priced(o, chip), cf = D.closed(o); n++;
  if (rows.reduce((s, r) => s + r.ops, 0) !== cf.ops || rows.reduce((s, r) => s + r.params, 0) !== cf.params) fail(['cost', phase, d, L, batch, chip].join());
}
// Decode is memory-bound at batch 1 on every chip: no projection may come out compute-bound.
for (const chip of ['a100', 'h100', 'chip_a']) { n++; if (D.priced({ phase: 'decode', d: 4096, h: 32, kvh: 8, ffn: 14336, act: 'swiglu', norm: 'rmsnorm', vocab: 128256, L: 4096, batch: 1 }, chip).some((r) => r.bound === 'compute')) fail('batch-1 decode should be memory-bound on ' + chip); }
console.log('simulators: ' + n + ' configurations checked');

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
if (JSDOM) {
  {
    const { w, errors } = load('deck.html');
    let widgets = 0;
    w.document.querySelectorAll('[data-widget]').forEach((node, i) => {
      const wd = node._widget; if (!wd || !wd.step) return;
      const knobs = [...node.querySelectorAll('.pv-knobs button')];
      (knobs.length ? knobs : [null]).forEach((b) => {
        if (b) b.click();
        let guard = 0; while (wd.step() && guard++ < 1000) {}
        widgets++;
        if (/✗/.test(node.textContent) || !/✓/.test(node.textContent)) fail('deck widget ' + i + (b ? ' ' + b.dataset.k + '=' + b.dataset.v : '') + ' did not end with a passing check');
      });
    });
    const key = (k) => w.document.dispatchEvent(new w.KeyboardEvent('keydown', { key: k, bubbles: true }));
    for (let i = 0; i < 3000; i++) key('ArrowRight');
    const cnt = w.document.getElementById('cnt').textContent.split('/').map((x) => +x.trim());
    if (cnt[0] !== cnt[1]) fail('deck: arrow keys did not reach the last slide');
    errors.forEach((e) => fail('deck script error: ' + e));
    console.log('deck: ' + widgets + ' animation runs, ' + cnt[1] + ' slides');
  }
  {
    const { w, errors } = load('notebook.html');
    const L = w.LAB; let runs = 0;
    const finish = (tag) => { L.view.finish(); runs++; if (!/✓/.test(w.document.getElementById('view').textContent)) fail('notebook ' + tag); };
    L.CH.forEach((c) => { L.chapter(c.id); finish('chapter ' + c.id); });
    for (const cache of [true, false]) for (const [d, h, kvh] of [[8, 4, 1], [4, 2, 2]]) { Object.assign(L.S, { mode: 'gen', Tp: 5, G: 5, d, h, kvh, ffn: 16, act: 'gelu', norm: 'layernorm', cache, seed: 3 }); L.rebuild(); finish(['gen', cache, d, h, kvh].join()); }
    for (const model of Object.keys(L.MODELS)) for (const phase of ['decode', 'prefill']) { Object.assign(L.S, { mode: 'cost', model, phase, L: 32768, S: 2048, batch: 8, chip: 'chip_a' }); L.rebuild(); finish(['cost', model, phase].join()); }
    L.chapter('gen'); [...w.document.querySelectorAll('#knobs button')].forEach((b) => b.click());
    L.chapter('step'); [...w.document.querySelectorAll('#knobs button')].forEach((b) => b.click());
    errors.forEach((e) => fail('notebook script error: ' + e));
    console.log('notebook: ' + L.CH.length + ' chapters, ' + runs + ' runs');
  }
}
console.log(failures ? failures + ' FAILURE(S)' : 'all checks passed');
process.exit(failures ? 1 : 0);
