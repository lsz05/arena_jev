// Shared helpers for the arena pages: theme toggle, tooltip, formatting, and a sortable table.
const $ = (s, el = document) => el.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
const dash = `<span class="muted">–</span>`;
const fmt = (v, fn) => v == null ? dash : fn(v);
const sizeCell = r => r.params == null ? dash
  : `<span class="has-tip" data-tip="${r.params.toLocaleString("en-US")} parameters${r.params_source ? " · " + esc(r.params_source) : ""}${r.params_measured ? " · counted at load time" : " · from the weight files"}">${(r.params / 1e6).toFixed(2)}M</span>`;
const caveatMark = r => r.caveat ? ` <span class="caveat" tabindex="0" role="img" aria-label="Caveat: ${esc(r.caveat)}" data-tip="${esc(r.caveat)}">⚠</span>` : "";
const modelCell = r => `<div class="mname">${esc(r.display)}${caveatMark(r)}</div>` +
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

// Any element with data-tip shows it in the tooltip: on hover, on keyboard focus, and on tap (touch screens have no hover).
function tipAt(el) { const b = el.getBoundingClientRect(); return {clientX: b.left, clientY: b.bottom}; }
document.addEventListener("mouseover", e => { const el = e.target.closest && e.target.closest("[data-tip]"); if (el) Tip.show(e, esc(el.dataset.tip)); });
document.addEventListener("mousemove", e => { if (e.target.closest && e.target.closest("[data-tip]")) Tip.move(e); });
document.addEventListener("mouseout", e => { const el = e.target.closest && e.target.closest("[data-tip]"); if (el && !el.contains(e.relatedTarget)) Tip.hide(); });
document.addEventListener("focusin", e => { const el = e.target.closest && e.target.closest("[data-tip]"); if (el) Tip.show(tipAt(el), esc(el.dataset.tip)); });
document.addEventListener("focusout", () => Tip.hide());
document.addEventListener("click", e => { const el = e.target.closest && e.target.closest("[data-tip]"); if (el) Tip.show(tipAt(el), esc(el.dataset.tip)); else Tip.hide(); });

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

// The definition of a column (tip, or the older title) as a tooltip attribute.
const tipAttr = c => (c.tip || c.title) ? `data-tip="${esc(c.tip || c.title)}"` : "";

// The definitions of the columns that have one, one per line: the note under a table.
function defsList(cols, extra = []) {
  const items = cols.filter(c => c.tip).map(c => [c.group ? `${c.group} · ${c.label}` : c.label, c.tip]).concat(extra);
  return `<ul class="defs">` + items.map(([k, v]) => `<li><b>${esc(k)}</b>: ${esc(v)}</li>`).join("") + "</ul>";
}

// A sortable, rankable table. cols: [{key, label, dir (-1 high first), cell(row) -> html, num, rank (bool: can rank by),
// group (optional: consecutive columns with the same group share a heading row above their labels),
// tip (optional: the metric's definition, shown on hover over its header and chip, and listed by defsList),
// hl (false: no top-3 highlighting, for columns that are not a measure of quality such as size or game count)}]
// opts.groupTips: {group name: definition} for the group headings.
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
    const arrow = c => c.key === state.key ? (state.dir < 0 ? " ↓" : " ↑") : "";
    if (opts.chips) {
      const chip = c => `<button class="chip ${c.key === state.key ? "on" : ""}" data-k="${c.key}" ${tipAttr(c)}>${c.label}${arrow(c)}</button>`;
      const ranked = opts.cols.filter(c => c.rank);
      const box = $(opts.chips);
      if (!ranked.some(c => c.group)) {
        box.classList.remove("two-row");
        box.innerHTML = `<span class="muted">Rank by</span>` + ranked.map(chip).join("");
      } else {  // two rows like the table header: "Rank by" and the group names above their chips
        const segs = [];
        ranked.forEach(c => { const g = c.group || ""; if (!segs.length || segs[segs.length - 1].g !== g) segs.push({g, cols: []}); segs[segs.length - 1].cols.push(c); });
        box.classList.add("two-row");
        const gtip = g => (opts.groupTips || {})[g] ? ` data-tip="${esc(opts.groupTips[g])}"` : "";
        box.innerHTML = segs.map((sg, i) => `<div class="chipseg ${sg.g ? "grouped" : ""}"><div class="chip-head"${gtip(sg.g)}>${i === 0 && !sg.g ? "Rank by" : esc(sg.g)}</div>` +
          `<div class="chip-row">${sg.cols.map(chip).join("")}</div></div>`).join("");
      }
      box.querySelectorAll("button").forEach(b => b.onclick = () => pick(b.dataset.k));
    }
    const on = c => c.key === state.key ? "on" : "";
    const grouped = opts.cols.some(c => c.group);
    const starts = new Set(opts.cols.filter((c, i) => c.group && c.group !== opts.cols[i - 1]?.group).map(c => c.key));
    const cls = c => `${c.num ? "num" : ""} ${on(c)} ${starts.has(c.key) ? "gstart" : ""}`;
    const label = (c, span = "") => `<th class="${cls(c)}" ${span} ${c.rank ? `data-k="${c.key}"` : ""} ${tipAttr(c)}>${c.label}${arrow(c)}</th>`;
    let h = "<thead>";
    if (!grouped) {
      h += "<tr><th>#</th>" + opts.cols.map(c => label(c)).join("") + "</tr>";
    } else {  // two rows: group names over their columns; ungrouped columns span both rows
      let top = `<tr><th rowspan="2">#</th>`, bottom = "<tr>";
      opts.cols.forEach((c, i) => {
        if (!c.group) { top += label(c, 'rowspan="2"'); return; }
        if (starts.has(c.key)) {
          let n = 0;
          while (opts.cols[i + n]?.group === c.group) n++;
          const gt = (opts.groupTips || {})[c.group];
          top += `<th class="grp gstart" colspan="${n}"${gt ? ` data-tip="${esc(gt)}"` : ""}>${esc(c.group)}</th>`;
        }
        bottom += label(c);
      });
      h += top + "</tr>" + bottom + "</tr>";
    }
    h += "</thead><tbody>";
    // per metric: its best value in red bold on a pink ground, the rest of its top 3 on pink (ties share a place;
    // "best" follows the metric's own direction, not the current sort). hl: false leaves a column plain, and so does
    // a column whose values are all the same or in which most rows would be marked (they tie at the top).
    const place = {};
    for (const c of opts.cols.filter(c => c.rank && c.hl !== false)) {
      const get = r => c.value ? c.value(r) : r[c.key];
      const vals = rows.map(get).filter(v => v != null && !Number.isNaN(v));
      if (new Set(vals).size < 2) continue;  // every value the same (e.g. no fallbacks at all): nothing to point out
      const better = (c.dir ?? -1) < 0 ? (a, b) => a > b : (a, b) => a < b;
      const rankOf = v => 1 + vals.filter(x => better(x, v)).length;
      if (vals.filter(v => rankOf(v) <= 3).length > vals.length / 2) continue;  // most rows tie at the top: no signal
      place[c.key] = r => { const v = get(r); return v == null ? 0 : rankOf(v); };
    }
    const hl = (c, r) => { const k = place[c.key] ? place[c.key](r) : 0; return k === 1 ? "hl1" : k > 1 && k <= 3 ? "hl3" : ""; };
    let rank = 0;
    for (const r of sorted) {
      const has = val(r) != null;
      if (has) rank += 1;
      h += `<tr><td>${has ? `<span class="rk rk${rank <= 3 ? rank : ""}">${rank}</span>` : dash}</td>` +
        opts.cols.map(c => `<td class="${cls(c)} ${hl(c, r)}">${c.cell(r)}</td>`).join("") + "</tr>";
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
