// Checks for the encoder notebook.
//   npm install      (once, in notebook/: installs jsdom for the page checks)
//   npm test
// Part 1: the layer walk equals the same layer computed token by token, and every count equals its
// closed form — which reproduces bwz's encoder layer (5280 operations, 664 parameters at the defaults).
// Part 2: loads the built deck.html and notebook.html in jsdom and drives every animation to its end.
'use strict';
const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..');
let failures = 0;
const fail = (msg) => { failures++; console.log('  FAIL ' + msg); };

// ---------------- part 1: simulators ----------------
for (const f of [['matmul', 'src', 'engine2.js'], ['common', 'widget.js'], ['attention', 'src', 'engine.js']]) require(path.join(ROOT, '..', ...f));
require(path.join(ROOT, 'src', 'engine.js'));
const E = globalThis.ENC;
let n = 0;
for (const S of [1, 2, 3, 4, 6, 8]) for (const [d, h] of [[2, 1], [2, 2], [4, 1], [4, 2], [8, 2], [8, 4], [8, 8]]) for (const ffn of [4, 8, 16]) for (const act of ['relu', 'gelu', 'swiglu', 'geglu']) for (const norm of ['rmsnorm', 'layernorm', 'none']) for (const seed of [0, 3, 7]) {
  const o = { S, d, h, ffn, act, norm, seed, vocab: 16 }, P = E.problem(o), cf = E.closedForm(o); n++;
  const ops = P.steps.reduce((s, x) => s + x.ops, 0), params = P.steps.reduce((s, x) => s + x.params, 0);
  const L = E.layerOps(o);
  if (!E.close(P.T.out, P.REF) || ops !== cf.ops || params !== cf.params || L.length !== P.steps.length || L.some((x, i) => x.ops !== P.steps[i].ops || x.params !== P.steps[i].params)) fail(['layer', S, d, h, ffn, act, norm, seed].join());
}
// bwz's own counts for its encoder layer (bwz encoder-layer, d=8, 2 heads, ffn=16, vocab=16, 4 tokens).
const BWZ = [['relu', 'rmsnorm', 5280, 664], ['swiglu', 'layernorm', 6752, 816], ['gelu', 'none', 5344, 640], ['geglu', 'rmsnorm', 6816, 792]];
for (const [act, norm, ops, params] of BWZ) { const cf = E.closedForm({ S: 4, d: 8, h: 2, ffn: 16, act, norm, vocab: 16 }); n++; if (cf.ops !== ops || cf.params !== params) fail('bwz golden ' + act + ' ' + norm + ': ' + cf.ops + '/' + cf.params); }
for (const S of [1, 128, 4096]) for (const d of [256, 4096]) for (const batch of [1, 16]) for (const chip of Object.keys(E.CHIPS)) for (const act of ['gelu', 'swiglu']) {
  const o = { S, d, h: 8, ffn: 4 * d, act, norm: 'layernorm', vocab: 1000, batch }, rows = E.costOf(o, chip), cf = E.closedForm(o); n++;
  const ops = rows.reduce((s, r) => s + r.ops, 0), params = rows.reduce((s, r) => s + r.params, 0);
  if (ops !== cf.ops || params !== cf.params || rows.some((r) => !(r.t >= r.tc && r.t >= r.tm && r.t > 0 || r.ops === 0))) fail(['cost', S, d, batch, chip, act].join());
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
  { // notebook
    const { w, errors } = load('notebook.html');
    const L = w.LAB; let runs = 0;
    const finish = (tag) => { L.view.finish(); runs++; if (!/✓/.test(w.document.getElementById('view').textContent)) fail('notebook ' + tag); };
    L.CH.forEach((c) => { L.chapter(c.id); finish('chapter ' + c.id); });
    for (const act of ['relu', 'gelu', 'swiglu', 'geglu']) for (const norm of ['rmsnorm', 'layernorm', 'none']) {
      Object.assign(L.S, { mode: 'layer', S: 5, d: 4, h: 2, ffn: 8, act, norm, seed: 2 }); L.rebuild(); finish(['layer', act, norm].join());
      Object.assign(L.S, { mode: 'cost', S: 2048, d: 1024, h: 16, ffnx: 4, act, norm, chip: 'chip_a', batch: 4 }); L.rebuild(); finish(['cost', act, norm].join());
    }
    L.chapter('chips'); [...w.document.querySelectorAll('#knobs button')].forEach((b) => b.click());
    L.chapter('layer'); [...w.document.querySelectorAll('#knobs button')].forEach((b) => b.click());
    errors.forEach((e) => fail('notebook script error: ' + e));
    console.log('notebook: ' + L.CH.length + ' chapters, ' + runs + ' runs');
  }
}
console.log(failures ? failures + ' FAILURE(S)' : 'all checks passed');
process.exit(failures ? 1 : 0);
