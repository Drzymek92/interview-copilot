"use strict";
// Shared helpers for the hub views (D35). Everything call-/CV-derived is painted with
// textContent, never innerHTML — the pages render interview content.

const el = (id) => document.getElementById(id);

function h(tag, props, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (k === "dataset") Object.assign(node.dataset, v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return node;
}

function fmt(n) {
  return new Intl.NumberFormat(navigator.language || "en", { maximumFractionDigits: 1 }).format(n);
}

async function api(path, opts) {
  const init = { headers: { "Content-Type": "application/json" }, ...(opts || {}) };
  if (init.body && typeof init.body !== "string") init.body = JSON.stringify(init.body);
  try {
    const r = await fetch(path, init);
    let data = {};
    try { data = await r.json(); } catch (e) { /* non-JSON */ }
    return { ok: r.ok && !data.error, status: r.status, data };
  } catch (e) {
    return { ok: false, status: 0, data: { error: "could not reach the copilot — is it still running?" } };
  }
}

function setMsg(node, text, cls) {
  if (!node) return;
  node.textContent = text || "";
  node.className = "msg" + (cls ? " " + cls : "");
}

function params() { return new URLSearchParams(location.search); }

/* "20260914_105944" → "14 Sep 2026, 10:59" (local display only) */
function stampLabel(stamp) {
  const m = /^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})/.exec(stamp || "");
  if (!m) return stamp || "";
  const d = new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]);
  return d.toLocaleString(navigator.language || "en", { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
}

/* ---------------- context bundle picker ---------------- */
// D37: the first option is "no bundle selected" (value "") — the server then runs on the default, a
// generic AI tech job bundle. Its own row (s.default) is folded into that option, not listed twice.
let SESSIONS = [];
const DEFAULT_LABEL = "Default — generic AI tech job (no bundle selected)";
const DEFAULT_NOTE = "Nothing selected? The mode runs on the default bundle: a generic AI tech job "
  + "(typical AI/ML-engineer job description, interview plan and practice questions). It has no CV, so "
  + "answers and grading are not tailored to you — generate a bundle with your CV for that.";

async function loadSessions(select, preferred) {
  const r = await api("/api/sessions");
  SESSIONS = (r.data && r.data.sessions) || [];
  select.textContent = "";
  select.appendChild(h("option", { value: "", text: DEFAULT_LABEL }));
  for (const s of SESSIONS) {
    if (s.default) continue;
    let label = s.id;
    if (s.id === "blank") label = "blank — empty, no context at all";
    else if (s.ok && (s.role || s.company)) label = s.id + " — " + [s.role, s.company].filter(Boolean).join(" · ");
    if (!s.ok) label += "  (broken)";
    select.appendChild(h("option", { value: s.id, disabled: !s.ok, text: label.length > 110 ? label.slice(0, 108) + "…" : label }));
  }
  select.value = preferred && SESSIONS.some((s) => s.id === preferred && s.ok && !s.default) ? preferred : "";
  return SESSIONS;
}

function sessionById(id) { return id ? SESSIONS.find((s) => s.id === id) : SESSIONS.find((s) => s.default); }

function renderBundleSummary(box, id, opts) {
  box.textContent = "";
  const s = sessionById(id);
  if (!id) {
    box.appendChild(h("div", { class: "role", text: "Generic AI tech job" }));
    box.appendChild(h("div", { text: s ? s.plan_steps + " plan steps · " + s.questions + " practice questions · no CV" : "built-in default" }));
    return;
  }
  if (!s) return;
  if (!s.ok) { box.appendChild(h("div", { class: "bad-text", text: s.error })); return; }
  if (s.id === "blank") {
    box.appendChild(h("div", { text: "No job description, CV, plan or answer bank. Suggestions will be generic, and training has no questions to start from." }));
    return;
  }
  box.appendChild(h("div", { class: "role", text: s.role || "(no role set)" }));
  if (s.company) box.appendChild(h("div", { text: s.company }));
  const bits = [s.plan_steps + " plan step" + (s.plan_steps === 1 ? "" : "s"),
                s.answer_bank + " STAR stor" + (s.answer_bank === 1 ? "y" : "ies"),
                s.questions + " practice question" + (s.questions === 1 ? "" : "s"),
                s.has_resume ? "CV loaded" : "no CV"];
  box.appendChild(h("div", { text: bits.join(" · ") }));
  if (s.placeholders && s.placeholders.length) {
    box.appendChild(h("div", { class: "warn-text", text: "⚠ Still placeholders: " + s.placeholders.join(", ") + " — fill them in scripts/inputs/sessions/" + s.id + "/ before a real interview." }));
  }
  if (opts && opts.extra) box.appendChild(opts.extra(s));
}

/* ---------------- file → text (D37: PDF / .txt / .md, parsed locally by the hub) ---------------- */
function readAsBase64(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(String(fr.result).replace(/^data:[^,]*,/, ""));
    fr.onerror = () => reject(fr.error);
    fr.readAsDataURL(file);
  });
}
async function fileToText(file) {
  if (file.size > 10 * 1024 * 1024) return { ok: false, data: { error: "the file is larger than 10 MB" } };
  let b64;
  try { b64 = await readAsBase64(file); } catch (e) { return { ok: false, data: { error: "could not read the file" } }; }
  return api("/api/extract", { method: "POST", body: { filename: file.name, data_b64: b64 } });
}

/* ---------------- settings form (Settings page + ⚙ drawers, D36) ---------------- */
const SOURCE_TEXT = { default: "default", saved: "saved", env: "🔒 env" };
const SOURCE_TITLE = {
  default: "Built-in default.",
  saved: "Saved by you in config/user_settings.json.",
  env: "Set by an environment variable — it always wins, so it can't be changed here. Unset it to edit.",
};

// filled by loadDevices(): COPILOT_MIC → microphones, COPILOT_SOURCE → output monitors
const DEVICE_LISTS = {};
async function loadDevices() {
  const r = await api("/api/devices");
  if (r.ok && r.data.available) {
    DEVICE_LISTS.COPILOT_MIC = r.data.mics;
    DEVICE_LISTS.COPILOT_SOURCE = r.data.monitors;
  }
  return r.data || {};
}

function knobControl(k) {
  const locked = k.source === "env";
  let input;
  if (k.kind === "bool") {
    input = h("input", { type: "checkbox", id: "k-" + k.env, "aria-label": k.label });
    input.checked = k.value === "1";
    const wrap = h("label", { class: "switch" + (locked ? " disabled" : "") }, input, h("span", { class: "track" }));
    input.disabled = locked;
    return { node: wrap, read: () => (input.checked ? "1" : "0"), input };
  }
  if (k.kind === "choice") {
    input = h("select", { id: "k-" + k.env, "aria-label": k.label });
    for (const c of k.choices) input.appendChild(h("option", { value: c, text: c }));
    input.value = k.value;
  } else if (DEVICE_LISTS[k.env] && DEVICE_LISTS[k.env].length) {
    // a device knob: pick from the sources pactl reports (still free text — a device may be unplugged now)
    const listId = "dl-" + k.env;
    input = h("input", { id: "k-" + k.env, "aria-label": k.label, type: "text", list: listId, spellcheck: "false" });
    input.value = k.value;
    const dl = h("datalist", { id: listId }, h("option", { value: "auto" }),
      ...DEVICE_LISTS[k.env].map((n) => h("option", { value: n })));
    input.disabled = locked;
    return { node: h("span", { style: "display:contents" }, input, dl), read: () => input.value.trim(), input };
  } else {
    input = h("input", { id: "k-" + k.env, "aria-label": k.label,
      type: k.kind === "int" || k.kind === "float" ? "number" : "text",
      step: k.kind === "float" ? "any" : (k.kind === "int" ? "1" : null),
      min: k.min, max: k.max, placeholder: k.kind === "str" && !k.default ? "(backend default)" : null });
    input.value = k.value;
  }
  input.disabled = locked;
  return { node: input, read: () => input.value.trim(), input };
}

function sameValue(k, v) {
  if (k.kind === "int" || k.kind === "float") return Number(v) === Number(k.value) && v !== "";
  return String(v) === String(k.value);
}

function renderSettingsForm(container, knobs, opts) {
  opts = opts || {};
  container.textContent = "";
  const controls = [];
  const msg = h("span", { class: "msg", role: "status" });
  const saveBtn = h("button", { type: "button", class: "btn primary", disabled: true, text: "Save" });

  const refreshDirty = () => {
    let dirty = 0;
    for (const c of controls) {
      const changed = !sameValue(c.k, c.read());
      c.row.classList.toggle("changed", changed);
      dirty += changed;
    }
    saveBtn.disabled = dirty === 0;
    return dirty;
  };

  for (const k of knobs) {
    const ctl = knobControl(k);
    const badge = h("span", { class: "badge src-" + k.source, title: SOURCE_TITLE[k.source], text: SOURCE_TEXT[k.source] });
    const reset = k.source === "saved"
      ? h("button", { type: "button", class: "linkish", text: "reset to default (" + (k.default || "empty") + ")",
          onclick: async () => {
            const r = await api("/api/config/reset", { method: "POST", body: { names: [k.env] } });
            if (r.ok) { opts.onSaved && opts.onSaved(r.data.knobs); }
            else setMsg(msg, r.data.error, "error");
          } })
      : null;
    const row = h("div", { class: "knob" },
      h("div", {},
        h("div", { class: "knob-label" }, h("label", { for: "k-" + k.env, text: k.label }), badge),
        k.help ? h("div", { class: "knob-help", text: k.help }) : null,
        h("div", { class: "knob-help" }, h("span", { class: "knob-env", text: k.env }), reset ? " · " : "", reset)),
      h("div", { class: "knob-control" }, ctl.node));
    ctl.input.addEventListener(k.kind === "bool" || k.kind === "choice" ? "change" : "input", refreshDirty);
    controls.push({ k, row, read: ctl.read });
    container.appendChild(row);
  }

  saveBtn.addEventListener("click", async () => {
    const updates = {};
    for (const c of controls) if (!sameValue(c.k, c.read())) updates[c.k.env] = c.read();
    if (!Object.keys(updates).length) return;
    saveBtn.disabled = true;
    setMsg(msg, "saving…");
    const r = await api("/api/config", { method: "POST", body: { updates } });
    if (!r.ok) { setMsg(msg, r.data.error || "could not save", "error"); refreshDirty(); return; }
    const running = r.data.active;
    opts.onSaved && opts.onSaved(r.data.knobs,
      running ? "Saved. The running " + running.mode + " keeps its settings until you end it; they apply from the next start."
              : "Saved — applies from the next start.");
  });
  container.appendChild(h("div", { class: "form-foot" }, saveBtn, msg));
  if (opts.message) setMsg(msg, opts.message, "ok");
}

/* ⚙ drawer: one mode's settings, beside the view */
function initSettingsDrawer(button, group, title) {
  const scrim = h("div", { class: "drawer-scrim", hidden: true });
  const body = h("div", { class: "body" });
  const drawer = h("aside", { class: "drawer", hidden: true, role: "dialog", "aria-modal": "true", "aria-label": title },
    h("div", { class: "head" }, h("h3", { style: "margin:0", text: title }),
      h("button", { type: "button", class: "x", "aria-label": "Close settings", text: "×", onclick: close })),
    body);
  document.body.append(scrim, drawer);
  scrim.addEventListener("click", close);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !drawer.hidden) close(); });

  function paint(knobs, message) {
    body.textContent = "";
    body.appendChild(h("p", { class: "note" }, "Only this mode's settings. Devices, models and languages are in ",
      h("a", { href: "/settings", text: "general settings" }), "."));
    const form = h("div");
    body.appendChild(form);
    renderSettingsForm(form, knobs.filter((k) => k.groups.includes(group)), { onSaved: paint, message });
  }
  async function open() {
    const r = await api("/api/config?group=" + encodeURIComponent(group));
    if (!r.ok) return;
    paint(r.data.knobs);
    scrim.hidden = false; drawer.hidden = false;
    const first = drawer.querySelector("input,select,button.x");
    if (first) first.focus();
  }
  function close() { scrim.hidden = true; drawer.hidden = true; button.focus(); }
  button.addEventListener("click", open);
}

/* ---------------- background jobs ---------------- */
function pollJob(id, onUpdate, intervalMs) {
  let stopped = false;
  const tick = async () => {
    if (stopped) return;
    const r = await api("/api/jobs/" + encodeURIComponent(id));
    if (!r.ok) { onUpdate({ status: "error", error: r.data.error || "lost the job" }); return; }
    onUpdate(r.data.job);
    if (r.data.job.status === "running") setTimeout(tick, intervalMs || 1000);
  };
  tick();
  return () => { stopped = true; };
}

function jobBox(text) {
  const t = h("span", { text });
  const secs = h("span", { class: "faint", text: "" });
  const box = h("div", { class: "job", role: "status" }, h("div", { class: "spinner", "aria-hidden": "true" }), h("div", {}, t, " ", secs));
  return { box, set: (txt, elapsed) => { t.textContent = txt; secs.textContent = elapsed != null ? "· " + Math.round(elapsed) + " s" : ""; } };
}

/* ---------------- scorecard (training + review) ---------------- */
function renderScorecard(box, sc, opts) {
  box.textContent = "";
  opts = opts || {};
  if (!sc) return;
  const o = sc.overall || {};
  box.appendChild(h("div", { class: "score-hero" },
    h("span", { class: "score-pct", text: (o.pct != null ? o.pct : 0) + "%" }),
    h("span", { class: "score-sub", text: fmt(o.total || 0) + " / " + fmt(o.max || 0) + " points · " + (o.answered || 0) + " answered" }),
    (sc.report_path || sc.report) ? h("span", { class: "faint", style: "font-size:12px", text: "saved: " + (sc.report || sc.report_path) }) : null));
  const qs = sc.questions || [];
  if (!qs.length) { box.appendChild(h("p", { class: "empty", text: "No answers were captured, so there is nothing to score." })); return; }
  for (const q of qs) {
    const card = h("div", { class: "sc-q" },
      h("div", { class: "sc-q-head" },
        h("span", { class: "sc-q-title", text: "Q" + q.index + (q.competency ? " · " + q.competency : "") }),
        h("span", { class: "sc-q-score", text: fmt(q.total) + " / " + fmt(q.max) + " (" + q.pct + "%)" })),
      h("div", { class: "sc-bar", "aria-hidden": "true" }, h("div", { style: "width:" + Math.max(0, Math.min(100, q.pct)) + "%" })),
      h("div", { class: "sc-question", text: q.question }),
      h("div", { class: "sc-answer", text: q.answer || "(no answer given)" }));
    for (const c of (q.criteria || [])) {
      card.appendChild(h("div", { class: "sc-crit" },
        h("span", { class: "level level-" + c.level, text: c.level }),
        h("span", { text: (c.id === "overall" ? "" : c.label) + (c.note ? (c.id === "overall" ? "" : " — ") + c.note : "") })));
    }
    if (q.concepts_to_refresh && q.concepts_to_refresh.length) {
      card.appendChild(h("div", { class: "sc-meta" }, h("span", { class: "lbl", text: "Refresh: " }), h("span", { class: "refresh", text: q.concepts_to_refresh.join(", ") })));
    }
    if (q.strengths && q.strengths.length) {
      card.appendChild(h("div", { class: "sc-meta" }, h("span", { class: "lbl", text: "Strengths: " }), h("span", { class: "strength", text: q.strengths.join(", ") })));
    }
    box.appendChild(card);
  }
}

/* ---------------- tiny markdown → DOM (reports; textContent only) ---------------- */
function renderMarkdown(box, md) {
  box.textContent = "";
  box.classList.add("md");
  const lines = (md || "").split("\n");
  let list = null, pre = null, table = null;
  const inline = (parent, text) => {
    // **bold** and `code` only; everything else is plain text
    const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
    let last = 0, m;
    while ((m = re.exec(text))) {
      if (m.index > last) parent.appendChild(document.createTextNode(text.slice(last, m.index)));
      const tok = m[0];
      parent.appendChild(tok.startsWith("**") ? h("strong", { text: tok.slice(2, -2) }) : h("code", { text: tok.slice(1, -1) }));
      last = m.index + tok.length;
    }
    if (last < text.length) parent.appendChild(document.createTextNode(text.slice(last)));
    return parent;
  };
  for (const line of lines) {
    if (line.startsWith("```")) {
      if (pre) { pre = null; } else { pre = h("pre"); box.appendChild(pre); }
      continue;
    }
    if (pre) { pre.textContent += line + "\n"; continue; }
    if (/^\s*\|.*\|\s*$/.test(line)) {
      if (/^\s*\|[\s:|-]+\|\s*$/.test(line)) continue;   // the |---|:--| divider row
      if (!table) { table = h("table"); box.appendChild(table); }
      const tr = h("tr");
      for (const cell of line.trim().slice(1, -1).split("|")) tr.appendChild(inline(h("td"), cell.trim()));
      table.appendChild(tr);
      continue;
    }
    table = null;
    const hd = /^(#{1,3})\s+(.*)$/.exec(line);
    const li = /^\s*[-*]\s+(.*)$/.exec(line);
    if (hd) { list = null; box.appendChild(inline(h("h" + hd[1].length), hd[2])); }
    else if (li) { if (!list) { list = h("ul"); box.appendChild(list); } list.appendChild(inline(h("li"), li[1])); }
    else if (/^\s*---+\s*$/.test(line)) { list = null; box.appendChild(h("hr")); }
    else if (line.trim()) { list = null; box.appendChild(inline(h("p"), line)); }
    else list = null;
  }
}
