"""Shared page shell for every notebook under notebook/: the reset, the fonts, the deck and
notebook chrome (CSS + JS), the document wrapper, and the two page builders.

Each notebook's src/build.py passes its own title, header and scripts; the look, the keys and the
controls are the same everywhere because they come from here.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSS = open(os.path.join(HERE, 'engine.css')).read()

RESET = ('body{margin:0;font:14px system-ui,sans-serif;'
         '-webkit-text-size-adjust:100%}img{max-width:100%}[hidden]{display:none!important}'
         ':root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}')

def document(page):
    """Turn '<title>..<style>..</style>body..' into a complete HTML document that opens from disk."""
    cut = page.index('</style>') + len('</style>')
    head, body = page[:cut], page[cut:]
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '<style>' + RESET + '</style>\n' + head + '\n</head>\n<body>\n' + body + '\n</body>\n</html>\n')

FONTS = '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">'

# ---------------- deck ----------------
DECK_CSS = r'''
html, body { height: 100%; }
.deck { height: 100%; display: grid; grid-template-rows: auto minmax(0, 1fr) auto; }
.top, .bot { display: flex; align-items: center; justify-content: space-between; gap: 1rem; padding: 0.55rem clamp(16px, 3vw, 2rem); border-bottom: 1px solid var(--line); background: var(--surface); }
.bot { border-bottom: none; border-top: 1px solid var(--line); flex-wrap: wrap; }
.top .name { font: 700 0.95rem var(--font-display); }
.top .sec { font: 500 0.8rem var(--font-mono); color: var(--muted); }
.stage { overflow: auto; }
.slide { display: none; flex-direction: column; gap: 1rem; max-width: 76rem; margin: 0 auto; padding-inline: clamp(16px, 4vw, 3rem); padding-block: 1.4rem 2.2rem; }
.slide.on { display: flex; animation: sin .28s ease-out; }
@keyframes sin { from { opacity: 0; transform: translateX(10px); } to { opacity: 1; transform: none; } }
.slide h2 { font: 700 clamp(1.5rem, 3.2vw, 2.3rem)/1.1 var(--font-display); margin: 0; letter-spacing: -0.015em; text-wrap: balance; }
.slide p { margin: 0; line-height: 1.5; font-size: clamp(1rem, 1.6vw, 1.15rem); max-width: 60ch; }
.slide p.small { font-size: 0.9rem; color: var(--muted); }
.slide p.big { font-size: clamp(1.15rem, 2vw, 1.4rem); }
.b { opacity: 0; transform: translateY(8px); transition: opacity .3s, transform .3s; }
.b.shown { opacity: 1; transform: none; }
tr.b { transform: none; }
.split { display: flex; flex-wrap: wrap; gap: 1.5rem 2.5rem; align-items: flex-start; }
.split > * { min-width: 0; flex: 1 1 20rem; }
.split > [data-widget] { flex: 0 1 auto; }
.points { display: flex; flex-direction: column; gap: 0.8rem; }
.points.wide p { max-width: 62ch; }
.title-slide { justify-content: center; min-height: 100%; gap: 1.2rem; }
.title-slide h1 { font: 800 clamp(3rem, 10vw, 6.5rem)/0.95 var(--font-display); letter-spacing: -0.04em; margin: 0; }
.eyebrow { font: 600 0.85rem var(--font-mono) !important; color: var(--muted); }
.formula { font: 600 clamp(1.2rem, 3vw, 2rem) var(--font-mono) !important; }
.lede { font-size: clamp(1.05rem, 2vw, 1.3rem) !important; max-width: 52ch !important; }
.keys { display: flex; flex-wrap: wrap; gap: 0.5rem 1.2rem; font-size: 0.85rem; color: var(--muted); }
kbd { font: 600 0.75rem var(--font-mono); border: 1px solid var(--line); border-bottom-width: 2px; border-radius: 4px; padding: 0 0.35rem; background: var(--surface); color: var(--fg); }
.tw { overflow-x: auto; }
table.rule { border-collapse: collapse; font-size: clamp(0.85rem, 1.4vw, 1rem); }
table.rule th { text-align: left; font: 600 0.72rem var(--font-body); text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); border-bottom: 2px solid var(--line); padding: 0.5rem 0.8rem; }
table.rule td { border-bottom: 1px solid var(--line); padding: 0.55rem 0.8rem; vertical-align: top; }
table.rule code { background: var(--surface-2); padding: 0.05rem 0.3rem; border-radius: 4px; }
table.cheat { font-size: 0.82rem; }
table.cheat td, table.cheat th { padding: 0.4rem 0.6rem; }
.stay-a, .stay-b, .stay-c { font-family: var(--font-mono); font-weight: 600; padding: 0 0.3rem; border-radius: 4px; white-space: nowrap; }
.stay-a { box-shadow: 0 0 0 2px var(--a); color: var(--a); } .stay-b { box-shadow: 0 0 0 2px var(--b); color: var(--b); } .stay-c { box-shadow: 0 0 0 2px var(--c); color: var(--c); }
.navb { display: flex; gap: 0.4rem; align-items: center; }
.count { font: 500 0.8rem var(--font-mono); color: var(--muted); min-width: 4.5rem; text-align: center; }
.prog { flex: 1 1 12rem; display: flex; gap: 3px; min-width: 0; }
.prog i { flex: 1; height: 5px; border-radius: 3px; background: var(--line); cursor: pointer; }
.prog i.done { background: color-mix(in oklab, var(--c) 50%, var(--line)); }
.prog i.cur { background: var(--c); }
.hint { font-size: 0.75rem; color: var(--muted); }
@media (prefers-reduced-motion: reduce) { .slide.on { animation: none; } .b { transition: none; } }
'''

DECK_JS = r'''
(function () {
  const slides = Array.from(document.querySelectorAll('.slide'));
  MM.mountAll();
  const secEl = document.getElementById('sec'), cntEl = document.getElementById('cnt'), prog = document.getElementById('prog');
  slides.forEach((s, i) => { const d = document.createElement('i'); d.title = (i + 1) + ' · ' + s.dataset.sec; d.addEventListener('click', () => show(i)); prog.appendChild(d); });
  let cur = 0;
  const builds = (s) => Array.from(s.querySelectorAll('.b'));
  const widget = (s) => { const n = s.querySelector('[data-widget]'); return n && n._widget && n._widget.step ? n._widget : null; };
  function show(n, revealAll) {
    n = Math.max(0, Math.min(slides.length - 1, n));
    const w0 = widget(slides[cur]); if (w0) w0.stop();
    cur = n;
    slides.forEach((s, i) => s.classList.toggle('on', i === cur));
    if (revealAll) builds(slides[cur]).forEach((b) => b.classList.add('shown'));
    secEl.textContent = slides[cur].dataset.sec;
    cntEl.textContent = (cur + 1) + ' / ' + slides.length;
    Array.from(prog.children).forEach((d, i) => { d.className = i === cur ? 'cur' : i < cur ? 'done' : ''; });
    document.querySelector('.stage').scrollTop = 0;
    try { history.replaceState(null, '', '#s' + (cur + 1)); } catch (e) {}
  }
  function next() {
    const s = slides[cur];
    const hid = builds(s).find((b) => !b.classList.contains('shown'));
    if (hid) { hid.classList.add('shown'); return; }
    const w = widget(s);
    if (w && !w.atEnd()) { w.stop(); w.step(); return; }
    show(cur + 1);
  }
  function prev() {
    const s = slides[cur];
    const w = widget(s);
    if (w && !w.atStart()) { w.stop(); w.back(); return; }
    const shown = builds(s).filter((b) => b.classList.contains('shown'));
    if (shown.length) { shown[shown.length - 1].classList.remove('shown'); return; }
    show(cur - 1, true);
  }
  function inner() { const w = widget(slides[cur]); if (w && !w.atEnd()) { w.stop(); builds(slides[cur]).forEach((b) => b.classList.add('shown')); w.inner(); } else next(); }
  document.getElementById('bprev').addEventListener('click', prev);
  document.getElementById('bnext').addEventListener('click', next);
  document.getElementById('bskip').addEventListener('click', () => show(cur + 1));
  document.addEventListener('keydown', (e) => {
    if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key, w = widget(slides[cur]);
    let done = true;
    if (k === 'ArrowRight' || k === ' ') next();
    else if (k === 'ArrowLeft') prev();
    else if (k === 'ArrowDown') inner();
    else if (k === 'ArrowUp') { if (w) { w.stop(); w.reset(); } }
    else if (k === 'PageDown' || k === 'n' || k === 'N') show(cur + 1);
    else if (k === 'PageUp' || k === 'b' || k === 'B') show(cur - 1, true);
    else if (k === 'Home') show(0);
    else if (k === 'End') show(slides.length - 1, true);
    else if ((k === 'p' || k === 'P') && w) w.play();
    else if ((k === 'e' || k === 'E') && w) w.finish();
    else if ((k === 'r' || k === 'R') && w) w.reset();
    else done = false;
    if (done) e.preventDefault();
  });
  const m = /^#s(\d+)$/.exec(location.hash);
  show(m ? +m[1] - 1 : 0, !!m);
})();
'''


LAB_CSS = r"""
html, body { height: auto; }
.lab-head { max-width: 100rem; margin: 0 auto; padding-inline: clamp(16px, 3vw, 2rem); padding-block: 1.4rem 0.4rem; display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.4rem 1.2rem; }
.lab-head h1 { font: 800 clamp(1.8rem, 4.5vw, 2.8rem)/1 var(--font-display); letter-spacing: -0.03em; margin: 0; }
.lab-head .formula { font: 600 1.05rem var(--font-mono); }
.lab-head .eyebrow { font: 600 0.78rem var(--font-mono); color: var(--muted); width: 100%; margin: 0; }
.chapters { max-width: 100rem; margin: 0 auto; padding-inline: clamp(16px, 3vw, 2rem); padding-block: 0.7rem; display: flex; flex-wrap: wrap; gap: 0.35rem; }
.btn.chap { display: inline-flex; gap: 0.4rem; align-items: center; font-size: 0.78rem; }
.btn.chap span { font: 600 0.66rem var(--font-mono); color: var(--muted); }
.btn.chap[aria-pressed="true"] span { color: var(--bg); }
.lab { max-width: 100rem; margin: 0 auto; padding-inline: clamp(16px, 3vw, 2rem); padding-block: 0.4rem 3rem; display: grid; grid-template-columns: minmax(0, 1fr); gap: 1.2rem; }
@media (min-width: 980px) { .lab { grid-template-columns: 16.5rem minmax(0, 1fr); } .knobs { position: sticky; top: calc(env(safe-area-inset-top, 0px) + 0.8rem); max-height: calc(100vh - 1.6rem); overflow-y: auto; } }
.knobs { align-self: start; background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 0.9rem; display: flex; flex-direction: column; gap: 0.7rem; }
.ksec { font: 700 0.7rem var(--font-display); text-transform: uppercase; letter-spacing: 0.09em; color: var(--c); border-top: 1px solid var(--line); padding-top: 0.6rem; }
.ksec:first-child { border-top: none; padding-top: 0; }
.kg { display: flex; flex-direction: column; gap: 0.3rem; }
.kl { font-size: 0.78rem; color: var(--muted); display: flex; justify-content: space-between; }
.kl b { color: var(--fg); font-family: var(--font-mono); }
.kg input[type=range] { width: 100%; accent-color: var(--c); }
.kg .btn { font-size: 0.72rem; padding: 0.22rem 0.45rem; font-family: var(--font-mono); }
.labmain { display: flex; flex-direction: column; gap: 0.9rem; min-width: 0; }
.story { max-width: 62rem; }
.story h2 { font: 700 1.45rem/1.15 var(--font-display); margin: 0 0 0.35rem; text-wrap: balance; }
.story p { margin: 0.3rem 0; line-height: 1.55; max-width: 70ch; }
.story .try { color: var(--muted); font-size: 0.9rem; }
.story code { background: var(--surface-2); padding: 0 0.3rem; border-radius: 4px; }
.rule { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 0.6rem 0.8rem; display: flex; flex-direction: column; gap: 0.35rem; }
.rl { display: flex; flex-wrap: wrap; align-items: center; gap: 0.3rem 0.5rem; font-size: 0.88rem; }
.rl.small { font-size: 0.8rem; color: var(--muted); }
.rk { font: 600 0.66rem var(--font-body); text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
.ar { color: var(--muted); }
.stay-a, .stay-b, .stay-c { font-family: var(--font-mono); font-weight: 600; padding: 0 0.3rem; border-radius: 4px; white-space: nowrap; }
.stay-a { box-shadow: 0 0 0 2px var(--a); color: var(--a); } .stay-b { box-shadow: 0 0 0 2px var(--b); color: var(--b); } .stay-c { box-shadow: 0 0 0 2px var(--c); color: var(--c); }
.story .tw { overflow-x: auto; margin: 0.4rem 0; }
table.cmp { border-collapse: collapse; font-size: 0.86rem; }
table.cmp th { text-align: left; font: 600 0.7rem var(--font-body); text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); border-bottom: 2px solid var(--line); padding: 0.35rem 0.6rem; }
table.cmp td { border-bottom: 1px solid var(--line); padding: 0.35rem 0.6rem; }
table.cmp td.mem { color: var(--mem); font-weight: 600; }
details.console { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; }
details.console summary { cursor: pointer; padding: 0.5rem 0.8rem; font: 600 0.82rem var(--font-body); }
details.console pre { margin: 0; padding: 0.6rem 0.8rem; max-height: 16rem; overflow: auto; font: 0.74rem/1.5 var(--font-mono); border-top: 1px solid var(--line); background: var(--surface-2); white-space: pre; }
"""


def lab_page(title, eyebrow, heading, formula, scripts, extra_css=''):
    """The interactive notebook: chapter bar, knob panel, story, rule, view and printed steps."""
    return document(
        f'<title>{title}</title>{FONTS}<style>{CSS}{LAB_CSS}{extra_css}</style>'
        f'<header class="lab-head"><p class="eyebrow">{eyebrow}</p><h1>{heading}</h1>'
        f'<span class="formula">{formula}</span></header>'
        '<nav class="chapters" id="chapters" aria-label="Chapters"></nav>'
        '<div class="lab"><aside class="knobs" id="knobs" aria-label="Knobs"></aside>'
        '<main class="labmain"><section class="story" id="story"></section><div class="rule" id="rule"></div><div id="view"></div>'
        '<details class="console" id="tracebox" open><summary>Printed steps — the same run as text</summary><pre id="trace"></pre></details></main></div>'
        + ''.join(f'<script>{s}</script>' for s in scripts))


def deck_page(title, name, slides, scripts, extra_css=''):
    """The slide deck: top bar, stage, footer with the navigation and the progress strip."""
    return document(
        f'<title>{title}</title>{FONTS}<style>{CSS}{DECK_CSS}{extra_css}</style>'
        f'<div class="deck"><header class="top"><span class="name">{name}</span><span class="sec" id="sec"></span></header>'
        f'<main class="stage">{slides}</main>'
        '<footer class="bot"><div class="navb"><button class="btn" id="bprev" type="button" aria-label="Back (←)">◀ Back</button>'
        '<span class="count" id="cnt"></span><button class="btn pri" id="bnext" type="button" aria-label="Next build or step (→)">Next ▶</button>'
        '<button class="btn" id="bskip" type="button" aria-label="Next slide (PgDn)">Next slide ⇥</button></div>'
        '<div class="prog" id="prog"></div><span class="hint">→ step · ↓ inner loop · PgDn slide · P play</span></footer></div>'
        + ''.join(f'<script>{s}</script>' for s in scripts) + f'<script>{DECK_JS}</script>')
