/* Deck widgets shared by every notebook: a view mounted on a slide, with optional in-slide knobs.

   NB.widgets(VIEWS, DEFAULTS, problem, LABELS) returns { mount, mountAll }. A slide declares
     <div data-widget='{"type":"flash","S":4,"knobs":[{"k":"Bc","v":[1,2,4]}]}'></div>
   and the widget builds VIEWS[type] on problem(cfg), rebuilding it when a knob button is pressed.
   The deck's own script drives it through step/back/inner/finish/reset/atEnd/atStart/play/stop,
   and the page checks press every knob button and run every animation to its end. */
(function (global) {
  'use strict';
  function el(tag, cls, html) { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
  function btn(label, title, fn) { const b = el('button', 'btn', label); b.type = 'button'; b.title = title; b.setAttribute('aria-label', title); b.addEventListener('click', fn); return b; }

  function widgets(VIEWS, DEFAULTS, problem, LABELS) {
    function Widget(root, cfg) {
      const st = Object.assign({}, DEFAULTS, cfg);
      root.classList.add('stack');
      const knobs = el('div', 'pv-knobs'); root.appendChild(knobs);
      const host = el('div'); root.appendChild(host);
      const make = () => {
        if (this.view) this.view.destroy();
        host.innerHTML = ''; const d = el('div'); host.appendChild(d);
        this.view = new VIEWS[st.type](d, Object.assign({}, st, { P: problem(st) }));
        knobs.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(st[b.dataset.k]) === b.dataset.v ? 'true' : 'false'));
      };
      (cfg.knobs || []).forEach((kn) => {
        const grp = el('div', 'seg'); grp.appendChild(el('span', 'seg-l', (LABELS && LABELS[kn.k]) || kn.k));
        kn.v.forEach((v) => {
          const b = btn(kn.labels ? kn.labels[kn.v.indexOf(v)] : String(v), kn.k + ' ' + v, () => { st[kn.k] = v; make(); });
          b.dataset.k = kn.k; b.dataset.v = String(v); grp.appendChild(b);
        });
        knobs.appendChild(grp);
      });
      if (!(cfg.knobs || []).length) knobs.hidden = true;
      make();
      this.root = root;
    }
    ['step', 'back', 'inner', 'finish', 'reset', 'atEnd', 'atStart', 'play', 'stop'].forEach((m) => { Widget.prototype[m] = function () { return this.view[m].apply(this.view, arguments); }; });
    function mount(node) { node._widget = new Widget(node, JSON.parse(node.dataset.widget)); return node._widget; }
    function mountAll(scope) { return Array.from((scope || document).querySelectorAll('[data-widget]')).map(mount); }
    return { mount, mountAll };
  }
  global.NB = { widgets };
})(typeof window !== 'undefined' ? window : globalThis);
