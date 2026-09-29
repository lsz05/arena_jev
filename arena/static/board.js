// Shared helpers for the arena pages: theme toggle, tooltip, formatting, and a sortable table.
const $ = (s, el = document) => el.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
const dash = `<span class="muted">–</span>`;
const fmt = (v, fn) => v == null ? dash : fn(v);
const sizeCell = r => r.params == null ? dash
  : `<span title="${r.params.toLocaleString("en-US")} parameters${r.params_source ? " · " + esc(r.params_source) : ""}${r.params_measured ? " · counted at load time" : " · from the weight files"}">${(r.params / 1e6).toFixed(2)}M</span>`;
const modelCell = r => `<div class="mname">${esc(r.display)}${r.caveat ? ` <span class="caveat" title="${esc(r.caveat)}">⚠</span>` : ""}</div>` +
  (r.repo ? `<div class="mrepo"><a href="https://huggingface.co/${esc(r.repo)}" target="_blank" rel="noopener">${esc(r.repo)}</a></div>` : "");

function initTheme() {
  try { const t = localStorage.getItem("uno-theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) {}
  const b = $("#theme");
  if (b) b.onclick = () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("uno-theme", root.dataset.theme); } catch (e) {}
  };
}

const Tip = {
  el: null,
  show(e, html) { this.el = this.el || $("#tip"); this.el.innerHTML = html; this.el.style.display = "block"; this.move(e); },
  move(e) {
    const t = this.el, pad = 14; let x = e.clientX + pad, y = e.clientY + pad;
    if (x + t.offsetWidth > innerWidth - 8) x = e.clientX - t.offsetWidth - pad;
    if (y + t.offsetHeight > innerHeight - 8) y = e.clientY - t.offsetHeight - pad;
    t.style.left = x + "px"; t.style.top = y + "px";
  },
  hide() { if (this.el) this.el.style.display = "none"; },
  bind(el, html) { el.addEventListener("mouseenter", e => Tip.show(e, html)); el.addEventListener("mousemove", e => Tip.move(e)); el.addEventListener("mouseleave", () => Tip.hide()); },
};

// A sortable, rankable table. cols: [{key, label, dir (-1 high first), cell(row) -> html, num, rank (bool: can rank by)}]
function SortTable(opts) {
  const state = {key: opts.sortKey, dir: opts.sortDir ?? -1};
  function render(rows) {
    const col = opts.cols.find(c => c.key === state.key);
    const val = r => col && col.value ? col.value(r) : r[state.key];
    const sorted = rows.slice().sort((a, b) => {
      const va = val(a), vb = val(b);
      if (va == null || vb == null) return (va == null) - (vb == null) || String(a.display).localeCompare(b.display);
      return (va - vb) * state.dir || String(a.display).localeCompare(b.display);
    });
    if (opts.chips) {
      $(opts.chips).innerHTML = `<span class="muted">Rank by</span>` + opts.cols.filter(c => c.rank).map(c =>
        `<button class="chip ${c.key === state.key ? "on" : ""}" data-k="${c.key}" title="${esc(c.title || "")}">${c.label}${c.key === state.key ? (state.dir < 0 ? " ↓" : " ↑") : ""}</button>`).join("");
      $(opts.chips).querySelectorAll("button").forEach(b => b.onclick = () => pick(b.dataset.k));
    }
    const on = c => c.key === state.key ? "on" : "";
    let h = "<thead><tr><th>#</th>" + opts.cols.map(c => `<th class="${c.num ? "num" : ""} ${on(c)}" ${c.rank ? `data-k="${c.key}"` : ""} title="${esc(c.title || "")}">${c.label}${c.key === state.key ? (state.dir < 0 ? " ↓" : " ↑") : ""}</th>`).join("") + "</tr></thead><tbody>";
    let rank = 0;
    for (const r of sorted) {
      const has = val(r) != null;
      if (has) rank += 1;
      h += `<tr class="${has && rank <= 3 ? "top" : ""}"><td>${has ? `<span class="rk rk${rank <= 3 ? rank : ""}">${rank}</span>` : dash}</td>` +
        opts.cols.map(c => `<td class="${c.num ? "num" : ""} ${on(c)}">${c.cell(r)}</td>`).join("") + "</tr>";
    }
    $(opts.table).innerHTML = h + "</tbody>";
    $(opts.table).querySelectorAll("th[data-k]").forEach(th => th.onclick = () => pick(th.dataset.k));
    function pick(k) {
      const c = opts.cols.find(x => x.key === k);
      if (state.key === k) state.dir = -state.dir; else { state.key = k; state.dir = c.dir ?? -1; }
      render(rows);
    }
  }
  return {render, state};
}
