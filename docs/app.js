// Page logic: controls, the Pyodide worker, charts and tables.
// The analysis itself is Python (py/analysis.py), run by worker.js.

"use strict";

const APP_VERSION = "2026-09-21";

const CRIMES = [
  ["violent", "Violent crime"], ["murder", "Murder"], ["rape", "Rape"], ["robbery", "Robbery"],
  ["assault", "Aggravated assault"], ["property", "Property crime"], ["burglary", "Burglary"],
  ["theft", "Theft"], ["motor", "Motor vehicle theft"],
];

// Agencies with 2025 National Guard deployments
const GUARD_2025 = ["TNMPD0000", "DCMPD0000", "CA0194200", "LANPD0000", "ILCPD0000", "OR0260200"];

// Same as DEFAULTS in py/analysis.py: the Memphis Safe Task Force example
const DEFAULTS = {
  city: "TNMPD0000", crime: "violent", start: "2025-09-29", partial: "pre", first: null, last: null,
  estimator: "lasso", penalty: "cv", alpha: 1, interval: "rolling", min_train: 36, level: 0.95,
  min_pop: 0, exclude: GUARD_2025,
};

// Short names for the URL hash
const HASH_KEYS = {
  city: "city", crime: "crime", start: "start", partial: "partial", first: "from", last: "to",
  estimator: "est", penalty: "pen", alpha: "alpha", interval: "int", min_train: "train", level: "level",
  min_pop: "minpop", exclude: "ex",
};

const $ = (id) => document.getElementById(id);
const state = {
  data: null, byId: new Map(), byLabel: new Map(), cfg: null, worker: null, ready: false,
  runId: 0, running: false, result: null,
};

// ---------------------------------------------------------------------------
// Formatting

const MINUS = "−";
function num(x, d = 1, signed = false) {
  if (x === null || x === undefined || !Number.isFinite(x)) return "—";
  const s = Math.abs(x).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  if (x < 0 && s.replace(/[0.,]/g, "") !== "") return MINUS + s;
  return (signed && x > 0 ? "+" : "") + s;
}
function monthLabel(ym) {
  const [y, m] = ym.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, 1)).toLocaleString("en-US", { month: "short", year: "numeric", timeZone: "UTC" });
}
function addMonth(ym, k) {
  const [y, m] = ym.split("-").map(Number);
  const t = y * 12 + (m - 1) + k;
  return `${String(Math.floor(t / 12)).padStart(4, "0")}-${String((t % 12) + 1).padStart(2, "0")}`;
}
const pctLevel = (lv) => `${Math.round(lv * 100)}%`;
const toDate = (ym) => `${ym}-01`;
function interval(lo, hi, d, suffix = "") {
  if (lo === null || hi === null) return "interval not available";
  return `${num(lo, d)}${suffix} to ${num(hi, d)}${suffix}`;
}

// ---------------------------------------------------------------------------
// Configuration: form <-> config <-> URL hash

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  const cfg = structuredClone(DEFAULTS);
  for (const [k, h] of Object.entries(HASH_KEYS)) {
    if (!p.has(h)) continue;
    const v = p.get(h);
    if (k === "exclude") cfg.exclude = v ? v.split(",") : [];
    else if (["alpha", "min_train", "level", "min_pop"].includes(k)) cfg[k] = Number(v);
    else cfg[k] = v || null;
  }
  return cfg;
}

function writeHash(cfg) {
  const p = new URLSearchParams();
  for (const [k, h] of Object.entries(HASH_KEYS)) {
    const v = cfg[k];
    if (k === "exclude") p.set(h, v.join(","));
    else if (v !== null && v !== undefined && v !== "") p.set(h, String(v));
  }
  history.replaceState(null, "", `#${p.toString()}`);
}

function radio(name) {
  const el = document.querySelector(`input[name="${name}"]:checked`);
  return el ? el.value : null;
}
function setRadio(name, value) {
  const el = document.querySelector(`input[name="${name}"][value="${value}"]`);
  if (el) el.checked = true;
}

function writeForm(cfg) {
  const city = state.byId.get(cfg.city);
  $("city").value = city ? city.label : "";
  $("crime").value = cfg.crime;
  $("start").value = cfg.start || "";
  $("partial").value = cfg.partial;
  $("first").value = cfg.first || state.data.dates[0];
  $("last").value = cfg.last || state.data.dates.at(-1);
  setRadio("estimator", cfg.estimator);
  setRadio("penalty", cfg.penalty);
  $("alpha").value = cfg.alpha;
  setRadio("interval", cfg.interval);
  $("min_train").value = cfg.min_train;
  $("level").value = String(cfg.level);
  $("min_pop").value = cfg.min_pop;
  state.exclude = [...cfg.exclude];
  renderChips();
  syncVisibility();
  updatePeriods();
}

function readForm() {
  const dates = state.data.dates;
  const first = $("first").value || dates[0];
  const last = $("last").value || dates.at(-1);
  const city = state.byLabel.get($("city").value.trim());
  return {
    city: city ? city.id : null,
    crime: $("crime").value,
    start: $("start").value,
    partial: $("partial").value,
    first: first === dates[0] ? null : first,
    last: last === dates.at(-1) ? null : last,
    estimator: radio("estimator"),
    penalty: radio("penalty"),
    alpha: Number($("alpha").value),
    interval: radio("interval"),
    min_train: Number($("min_train").value),
    level: Number($("level").value),
    min_pop: Number($("min_pop").value) || 0,
    exclude: [...state.exclude],
  };
}

function syncVisibility() {
  const synth = radio("estimator") === "synth";
  $("penalty-box").hidden = synth;
  $("alpha").hidden = radio("penalty") !== "fixed";
  $("min-train-box").style.visibility = radio("interval") === "rolling" ? "visible" : "hidden";
}

function periods(cfg) {
  const dates = state.data.dates;
  if (!/^\d{4}-\d{2}-\d{2}$/.test(cfg.start || "")) return null;
  const first = cfg.first || dates[0];
  const last = cfg.last || dates.at(-1);
  const sm = cfg.start.slice(0, 7);
  const partial = Number(cfg.start.slice(8, 10)) > 1 ? sm : null;
  const firstPost = partial && cfg.partial !== "post" ? addMonth(sm, 1) : sm;
  let win = dates.filter((d) => d >= first && d <= last);
  if (partial && cfg.partial === "drop") win = win.filter((d) => d !== partial);
  return { pre: win.filter((d) => d < firstPost), post: win.filter((d) => d >= firstPost), partial, firstPost };
}

function span(months) {
  if (!months.length) return "none";
  const n = months.length;
  return `${monthLabel(months[0])} – ${monthLabel(months[n - 1])} (${n} month${n === 1 ? "" : "s"})`;
}

function updatePeriods() {
  const cfg = readForm();
  const p = periods(cfg);
  const el = $("periods");
  if (!p) { el.textContent = "Enter a start date."; return; }
  let txt = `Pre-period ${span(p.pre)}. Post-period ${span(p.post)}.`;
  if (p.partial) {
    const what = { pre: "counted as pre-period", post: "counted as post-period", drop: "left out" }[cfg.partial];
    txt += ` ${monthLabel(p.partial)} is partly treated and is ${what}.`;
  }
  el.textContent = txt;
}

// ---------------------------------------------------------------------------
// Excluded donor cities

function renderChips() {
  const ul = $("exclude-list");
  ul.replaceChildren();
  if (!state.exclude.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "None left out";
    ul.append(li);
    return;
  }
  for (const id of state.exclude) {
    const c = state.byId.get(id);
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.textContent = c ? c.label : id;
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = "×";
    b.setAttribute("aria-label", `Put ${c ? c.label : id} back in the donor pool`);
    b.addEventListener("click", () => {
      state.exclude = state.exclude.filter((x) => x !== id);
      renderChips();
    });
    li.append(name, b);
    ul.append(li);
  }
}

function addExclusion() {
  const inp = $("exclude-add");
  const c = state.byLabel.get(inp.value.trim());
  if (!c) {
    inp.setCustomValidity("Pick a city from the list.");
    inp.reportValidity();
    return;
  }
  inp.setCustomValidity("");
  if (!state.exclude.includes(c.id)) state.exclude.push(c.id);
  inp.value = "";
  renderChips();
}

// ---------------------------------------------------------------------------
// Worker

function setStatus(msg, busy = true) {
  $("status-text").textContent = msg;
  $("status").classList.toggle("idle", !busy);
}
function showError(msg) {
  const el = $("error");
  el.textContent = msg || "";
  el.hidden = !msg;
}

function startWorker(text) {
  const w = new Worker(`worker.js?v=${APP_VERSION}`, { type: "module" });
  state.worker = w;
  w.onmessage = (e) => {
    const m = e.data;
    if (m.type === "status") setStatus(m.msg);
    else if (m.type === "ready") {
      state.ready = true;
      $("py-version").textContent = `${m.runtime} (Pyodide)`;
      $("run-btn").disabled = false;
      runAnalysis();
    } else if (m.type === "fatal") {
      setStatus("Python could not start", false);
      showError(`Could not start the Python runtime: ${m.msg}`);
    } else if (m.id !== state.runId) {
      // stale message from an earlier run
    } else if (m.type === "progress") setStatus(m.msg);
    else if (m.type === "result") finishRun(JSON.parse(m.json));
    else if (m.type === "error") finishRun({ error: `Unexpected error in the analysis: ${m.msg}` });
  };
  w.onerror = (e) => {
    setStatus("Python could not start", false);
    showError(`The analysis worker failed: ${e.message || "unknown error"}`);
  };
  w.postMessage({ type: "init", data: text, version: APP_VERSION });
}

function runAnalysis() {
  if (!state.ready || state.running) return;
  const cfg = readForm();
  const cityInput = $("city");
  if (!cfg.city) {
    cityInput.setCustomValidity("Pick a city from the list.");
    cityInput.reportValidity();
    return;
  }
  cityInput.setCustomValidity("");
  showError("");
  state.cfg = cfg;
  state.running = true;
  state.runId += 1;
  $("run-btn").disabled = true;
  $("results").classList.add("busy");
  setStatus("Running");
  state.worker.postMessage({ type: "run", id: state.runId, config: cfg });
}

function finishRun(res) {
  state.running = false;
  $("run-btn").disabled = false;
  $("results").classList.remove("busy");
  if (res.error) {
    setStatus("The analysis could not run with these settings", false);
    showError(`${res.error}${state.result ? " The results below are from the last successful run." : ""}`);
    $("results").classList.add("stale");
    return;
  }
  $("results").classList.remove("stale");
  state.result = res;
  writeHash(state.cfg);
  const t = res.timing.total;
  setStatus(`Done in ${t < 1 ? t.toFixed(2) : t.toFixed(1)} seconds.`, false);
  renderResults();
}

// ---------------------------------------------------------------------------
// Charts

function theme() {
  const cs = getComputedStyle(document.documentElement);
  const v = (n) => cs.getPropertyValue(n).trim();
  return {
    surface: v("--surface"), ink: v("--ink"), ink2: v("--ink-2"), muted: v("--muted"), grid: v("--grid"),
    axis: v("--axis"), s1: v("--series-1"), s2: v("--series-2"), wash: v("--accent-wash"),
    font: v("--font"), dark: getComputedStyle(document.body).colorScheme === "dark",
  };
}

function layout(t, extra = {}) {
  const axis = {
    gridcolor: t.grid, linecolor: t.axis, showline: true, zeroline: false, ticks: "",
    tickfont: { color: t.muted }, automargin: true,
  };
  return {
    paper_bgcolor: t.surface, plot_bgcolor: t.surface,
    font: { family: t.font, color: t.ink2, size: 12 },
    margin: { l: 8, r: 12, t: 30, b: 8 },
    xaxis: { ...axis, type: "date", hoverformat: "%b %Y" , ...(extra.xaxis || {}) },
    yaxis: { ...axis, separatethousands: true, ...(extra.yaxis || {}) },
    legend: { orientation: "h", x: 0, xanchor: "left", y: 1.0, yanchor: "bottom", bgcolor: "rgba(0,0,0,0)",
      font: { color: t.ink2 } },
    hovermode: "x unified",
    hoverlabel: { bgcolor: t.surface, bordercolor: t.axis, font: { color: t.ink, family: t.font } },
    showlegend: true,
    ...Object.fromEntries(Object.entries(extra).filter(([k]) => !["xaxis", "yaxis"].includes(k))),
  };
}

const PLOT_CONFIG = {
  responsive: true, displaylogo: false,
  modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
  toImageButtonOptions: { format: "png", scale: 2 },
};

// Vertical line between the last pre-period month and the first post-period month
function interventionShape(res, t) {
  const [y, m] = res.first_post.split("-").map(Number);
  const x = new Date(Date.UTC(y, m - 1, 1) - 15.2 * 864e5).toISOString().slice(0, 10);
  return {
    shape: { type: "line", xref: "x", yref: "paper", x0: x, x1: x, y0: 0, y1: 1, line: { color: t.muted, width: 1.5 } },
    annotation: { x, xref: "x", yref: "paper", y: 1, yanchor: "top", xanchor: "right", text: "Intervention ",
      showarrow: false, font: { color: t.ink2, size: 11 } },
  };
}

function zeroLine(t) {
  return { type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: t.ink2, width: 1 } };
}

function bandTraces(x, lo, hi, t, name, hover = "skip") {
  return [
    { x, y: lo, type: "scatter", mode: "lines", line: { width: 0 }, hoverinfo: hover, showlegend: false, name: `${name} low` },
    { x, y: hi, type: "scatter", mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: t.wash,
      hoverinfo: hover, name },
  ];
}

function renderFit(res, t, d) {
  const x = res.dates.map(toDate);
  const post = x.slice(res.T0);
  const lv = pctLevel(res.config.level);
  const text = res.pred.map((_, i) => i < res.T0 ? "" :
    `  (${lv}: ${interval(res.pred_lo[i - res.T0], res.pred_hi[i - res.T0], d)})`);
  const iv = interventionShape(res, t);
  Plotly.react("chart-fit", [
    ...bandTraces(post, res.pred_lo, res.pred_hi, t, `${lv} interval`),
    { x, y: res.obs, type: "scatter", mode: "lines", name: res.city.label, line: { color: t.ink, width: 2 },
      hovertemplate: `%{y:,.${d}f}<extra>${res.city.label}</extra>` },
    { x, y: res.pred, text, type: "scatter", mode: "lines", name: "Synthetic", line: { color: t.s1, width: 2 },
      hovertemplate: `%{y:,.${d}f}%{text}<extra>Synthetic</extra>` },
  ], layout(t, { shapes: [iv.shape], annotations: [iv.annotation] }), PLOT_CONFIG);
}

function renderMonthly(res, t, d) {
  const x = res.dates.slice(res.T0).map(toDate);
  const lv = pctLevel(res.config.level);
  const view = radio("monthly-view");
  let traces, shapes = [];
  const few = x.length <= 36;
  if (view === "levels") {
    const obs = res.obs.slice(res.T0);
    const pred = res.pred.slice(res.T0);
    traces = [
      ...bandTraces(x, res.pred_lo, res.pred_hi, t, `${lv} interval`),
      { x, y: obs, type: "scatter", mode: few ? "lines+markers" : "lines", name: res.city.label,
        line: { color: t.ink, width: 2 }, marker: { size: 8, color: t.ink, line: { color: t.surface, width: 2 } },
        hovertemplate: `%{y:,.${d}f}<extra>${res.city.label}</extra>` },
      { x, y: pred, type: "scatter", mode: few ? "lines+markers" : "lines", name: "Synthetic",
        line: { color: t.s1, width: 2 }, marker: { size: 8, color: t.s1, line: { color: t.surface, width: 2 } },
        hovertemplate: `%{y:,.${d}f}<extra>Synthetic</extra>` },
    ];
  } else {
    const text = res.dif.map((_, i) => `  (${lv}: ${interval(res.dif_lo[i], res.dif_hi[i], d)})`);
    const hasBand = res.dif_lo.some((v) => v !== null);
    traces = [
      { x, y: res.dif, text, type: "scatter", mode: few ? "lines+markers" : "lines", name: "Observed − synthetic",
        line: { color: t.s1, width: 2 }, marker: { size: 8, color: t.s1, line: { color: t.surface, width: 2 } },
        error_y: hasBand ? {
          type: "data", symmetric: false, color: t.s1, thickness: 1.5, width: 0,
          array: res.dif_hi.map((h, i) => (h === null ? null : h - res.dif[i])),
          arrayminus: res.dif_lo.map((l, i) => (l === null ? null : res.dif[i] - l)),
        } : undefined,
        hovertemplate: `%{y:,.${d}f}%{text}<extra></extra>` },
    ];
    shapes = [zeroLine(t)];
  }
  Plotly.react("chart-monthly", traces, layout(t, { shapes, showlegend: view === "levels" }), PLOT_CONFIG);
}

function renderCum(res, t, d) {
  const x = res.dates.slice(res.T0).map(toDate);
  const lv = pctLevel(res.config.level);
  const text = res.cum.map((_, i) => `  (${lv}: ${interval(res.cum_lo[i], res.cum_hi[i], d)})`);
  Plotly.react("chart-cum", [
    ...bandTraces(x, res.cum_lo, res.cum_hi, t, `${lv} band`),
    { x, y: res.cum, text, type: "scatter", mode: x.length <= 36 ? "lines+markers" : "lines", name: "Cumulative difference",
      line: { color: t.s1, width: 2 }, marker: { size: 8, color: t.s1, line: { color: t.surface, width: 2 } },
      hovertemplate: `%{y:,.${d}f}%{text}<extra></extra>` },
  ], layout(t, { shapes: [zeroLine(t)] }), PLOT_CONFIG);
}

function quantile(values, q) {
  const a = values.filter(Number.isFinite).sort((p, r) => p - r);
  if (!a.length) return NaN;
  return a[Math.min(a.length - 1, Math.floor(q * a.length))];
}

function renderDonors(res, t, d) {
  const x = res.dates.map(toDate);
  const weighted = new Set(res.weights.map((w) => w.id));
  const view = radio("donor-view");
  const log = radio("donor-scale") === "log";
  // One trace per group, lines separated by gaps, so hundreds of cities draw fast
  const group = (keep) => {
    const gx = [], gy = [], gt = [];
    res.donors.ids.forEach((id, j) => {
      if (!keep(id)) return;
      const lab = res.donors.labels[j];
      for (let i = 0; i < x.length; i++) { gx.push(x[i]); gy.push(res.donors.rates[j][i]); gt.push(lab); }
      gx.push(null); gy.push(null); gt.push("");
    });
    return { x: gx, y: gy, text: gt };
  };
  const other = group((id) => !weighted.has(id));
  const wt = group((id) => weighted.has(id));
  const hover = `%{text}<br>%{x|%b %Y}: %{y:,.${d}f}<extra></extra>`;
  const traces = [
    { ...other, type: "scattergl", mode: "lines", name: "Other donors", hovertemplate: hover,
      line: { color: t.muted, width: 1 }, opacity: t.dark ? 0.35 : 0.3, visible: view === "all" ? true : "legendonly" },
    { ...wt, type: "scatter", mode: "lines", name: "Donors with weight", hovertemplate: hover,
      line: { color: t.s2, width: 1.5 } },
    { x, y: res.pred, type: "scatter", mode: "lines", name: "Synthetic", line: { color: t.s1, width: 2 },
      hovertemplate: `Synthetic<br>%{x|%b %Y}: %{y:,.${d}f}<extra></extra>` },
    { x, y: res.obs, type: "scatter", mode: "lines", name: res.city.label, line: { color: t.ink, width: 2.5 },
      hovertemplate: `${res.city.label}<br>%{x|%b %Y}: %{y:,.${d}f}<extra></extra>` },
  ];
  // Keep the treated city readable: cap the default range near the top 1% of donor values
  let yaxis = { type: log ? "log" : "linear" };
  if (log) {
    // Ticks at 1, 2, 5 times powers of ten, instead of Plotly's minor digits
    const vals = [res.obs, res.pred, wt.y, view === "all" ? other.y : []].flat().filter((v) => v > 0);
    const lo = Math.min(...vals), hi = Math.max(...vals);
    const ticks = [];
    for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
      for (const k of [1, 2, 5]) ticks.push(k * 10 ** e);
    }
    yaxis.tickvals = ticks;
    yaxis.ticktext = ticks.map((v) => v.toLocaleString("en-US", { maximumFractionDigits: 3 }));
  } else {
    const shown = view === "all" ? other.y.concat(wt.y) : wt.y;
    const top = Math.max(quantile(shown, 0.99), ...res.obs, ...res.pred, ...wt.y.filter(Number.isFinite));
    if (Number.isFinite(top)) yaxis.range = [0, top * 1.05];
  }
  const iv = interventionShape(res, t);
  Plotly.react("chart-donors", traces, layout(t, {
    yaxis, hovermode: "closest", shapes: [iv.shape], annotations: [iv.annotation],
  }), PLOT_CONFIG);
}

// ---------------------------------------------------------------------------
// Summary and tables

function tile(label, value, sub) {
  const div = document.createElement("div");
  div.className = "tile";
  const a = document.createElement("div"); a.className = "t-label"; a.textContent = label;
  const b = document.createElement("div"); b.className = "t-value"; b.textContent = value;
  const c = document.createElement("div"); c.className = "t-sub"; c.textContent = sub;
  div.append(a, b, c);
  return div;
}

function renderSummary(res, d) {
  const s = res.summary, f = res.fit, lv = pctLevel(res.config.level);
  const post = res.dates.slice(res.T0);
  const postSpan = `${monthLabel(post[0])} – ${monthLabel(post.at(-1))}`;
  $("result-title").textContent = `${res.city.label}: ${res.crime_label.toLowerCase()}`;
  $("result-sub").textContent = `${res.estimator_label}, ${res.interval_label.toLowerCase()} (${lv}). ` +
    `Pre-period ${monthLabel(res.dates[0])} – ${monthLabel(res.dates[res.T0 - 1])} (${res.T0} months), ` +
    `post-period ${postSpan} (${res.H} month${res.H === 1 ? "" : "s"}).`;
  $("tiles").replaceChildren(
    tile(`Cumulative change per 100,000`, num(s.cum, d, true), `${lv}: ${interval(s.cum_lo, s.cum_hi, d)}`),
    tile("Change vs synthetic", `${num(s.pct, 1, true)}%`, `${lv}: ${interval(s.pct_lo, s.pct_hi, 1, "%")}`),
    tile("Implied crimes", num(s.count, 0, true), `${lv}: ${interval(s.count_lo, s.count_hi, 0)}`),
    tile("Pre-period fit", `R² ${num(f.r2, 2)}`,
      `RMSE ${num(f.rmse, d)} · ${f.n_nonzero} of ${f.n_donors.toLocaleString("en-US")} donors weighted`),
  );
  let note = `Over ${postSpan}, ${res.city.label} had ${num(s.obs_total, d)} ${res.crime_label.toLowerCase()} ` +
    `offenses per 100,000 residents, against ${num(s.pred_total, d)} for the synthetic control. `;
  if (s.cum_lo === null || s.cum_hi === null) {
    note += `The ${lv} cumulative band is not available: too few pre-period forecasts reach ${res.H} months ahead. ` +
      "Use a shorter first training window, an earlier start to the data, or a lower level.";
  } else {
    note += `The ${lv} band for the cumulative change ${s.excludes_zero ? "excludes" : "includes"} zero.`;
  }
  const dr = res.dropped;
  const parts = [];
  if (dr.excluded.length) parts.push(`${dr.excluded.length} left out`);
  if (dr.missing.length) parts.push(`${dr.missing.length} missing months in the window`);
  if (dr.small) parts.push(`${dr.small} below the population cutoff`);
  if (parts.length) note += ` Donor pool: ${f.n_donors.toLocaleString("en-US")} cities (not used: ${parts.join(", ")}).`;
  $("result-note").textContent = note;
}

function table(el, head, rows, leftCols = [0]) {
  el.replaceChildren();
  const thead = el.createTHead().insertRow();
  head.forEach((h, i) => {
    const th = document.createElement("th");
    th.textContent = h;
    if (leftCols.includes(i)) th.className = "l";
    thead.append(th);
  });
  const tb = el.createTBody();
  for (const r of rows) {
    const tr = tb.insertRow();
    if (r.cls) tr.className = r.cls;
    r.cells.forEach((v, i) => {
      const td = tr.insertCell();
      td.textContent = v;
      if (leftCols.includes(i)) td.className = "l";
    });
  }
}

function renderTables(res, d) {
  const f = res.fit;
  const pen = res.config.estimator === "synth" ? "Weights are non-negative and sum to one." :
    `Penalty (alpha) ${num(f.alpha, 3)}, ${res.config.penalty === "cv" ? "chosen by 5-fold cross-validation" : "fixed"}.`;
  $("coef-sub").textContent = `${f.n_nonzero} of ${f.n_donors.toLocaleString("en-US")} donor cities have non-zero ` +
    `weight. ${pen} Contribution is the coefficient times the donor's pre-period mean rate; with the intercept they ` +
    "add up to the synthetic control's pre-period mean.";
  const rows = [];
  if (res.config.estimator !== "lasso_noint" && res.config.estimator !== "synth") {
    rows.push({ cls: "intercept", cells: ["", "Intercept", "", num(f.intercept, 3), "", num(f.intercept, d)] });
  }
  res.weights.forEach((w, i) => rows.push({ cells: [
    String(i + 1), w.label, w.pop.toLocaleString("en-US"), num(w.coef, 3), num(w.pre_mean, d), num(w.contrib, d),
  ] }));
  const sumW = res.weights.reduce((a, w) => a + w.coef, 0);
  rows.push({ cls: "intercept", cells: ["", "Sum of coefficients", "", num(sumW, 3), "", ""] });
  table($("coef-table"), ["#", "City", "Population", "Coefficient", "Pre-period mean", "Contribution"], rows, [0, 1]);

  const post = res.dates.slice(res.T0);
  const refLabel = res.config.interval === "rolling" ? "Forecasts used" : "Jackknife errors";
  table($("month-table"), ["Month", "Observed", "Synthetic", "Difference", "Low", "High",
    "Cumulative", "Cum. low", "Cum. high", refLabel],
  post.map((m, i) => ({ cells: [
    monthLabel(m), num(res.obs[res.T0 + i], d), num(res.pred[res.T0 + i], d), num(res.dif[i], d),
    num(res.dif_lo[i], d), num(res.dif_hi[i], d), num(res.cum[i], d), num(res.cum_lo[i], d), num(res.cum_hi[i], d),
    String(res.n_ref[i]),
  ] })));
}

function renderResults() {
  const res = state.result;
  if (!res) return;
  $("results").hidden = false;
  const t = theme();
  const maxRate = Math.max(...res.obs.filter(Number.isFinite));
  const d = maxRate < 10 ? 2 : 1;
  renderSummary(res, d);
  renderFit(res, t, d);
  renderMonthly(res, t, d);
  renderCum(res, t, d);
  renderDonors(res, t, d);
  renderTables(res, d);
}

// ---------------------------------------------------------------------------
// Downloads

function csvCell(v) {
  if (v === null || v === undefined) return "";
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}
function download(name, header, rows) {
  const text = [header, ...rows].map((r) => r.map(csvCell).join(",")).join("\n") + "\n";
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function fileStem(res) {
  return `synth_${res.city.id}_${res.config.crime}_${res.first_post}`;
}

function downloadMonthly() {
  const res = state.result;
  if (!res) return;
  const rows = res.dates.map((m, i) => {
    const j = i - res.T0;
    const p = j >= 0;
    return [m, p ? "post" : "pre", res.obs[i], res.pred[i], res.gap[i],
      p ? res.pred_lo[j] : null, p ? res.pred_hi[j] : null, p ? res.dif_lo[j] : null, p ? res.dif_hi[j] : null,
      p ? res.cum[j] : null, p ? res.cum_lo[j] : null, p ? res.cum_hi[j] : null, p ? res.n_ref[j] : null];
  });
  download(`${fileStem(res)}_monthly.csv`, ["month", "period", "observed", "synthetic", "difference",
    "synthetic_low", "synthetic_high", "difference_low", "difference_high", "cumulative", "cumulative_low",
    "cumulative_high", "n_reference_errors"], rows);
}

function downloadWeights() {
  const res = state.result;
  if (!res) return;
  const rows = [["", "Intercept", "", res.fit.intercept, "", ""]].concat(
    res.weights.map((w) => [w.id, w.label, w.pop, w.coef, w.pre_mean, w.contrib]));
  download(`${fileStem(res)}_coefficients.csv`, ["id", "city", "population", "coefficient", "pre_mean_rate",
    "contribution"], rows);
}

// ---------------------------------------------------------------------------
// Start up

function populate() {
  const dl = $("city-list");
  for (const c of state.data.cities) {
    const o = document.createElement("option");
    o.value = c.label;
    dl.append(o);
  }
  const sel = $("crime");
  for (const [v, lab] of CRIMES) {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = lab;
    sel.append(o);
  }
  const d0 = state.data.dates[0], d1 = state.data.dates.at(-1);
  for (const id of ["first", "last"]) { $(id).min = d0; $(id).max = d1; }
  const s = state.data.source;
  $("data-version").textContent = `${s.file || "snapshot"}, downloaded ${(s.downloaded_at_utc || "").slice(0, 10)}`;
}

function wire() {
  $("controls").addEventListener("submit", (e) => { e.preventDefault(); runAnalysis(); });
  $("controls").addEventListener("change", () => { syncVisibility(); updatePeriods(); });
  $("city").addEventListener("input", () => $("city").setCustomValidity(""));
  $("exclude-add").addEventListener("input", () => $("exclude-add").setCustomValidity(""));
  $("exclude-add-btn").addEventListener("click", addExclusion);
  $("exclude-add").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); addExclusion(); } });
  $("guard-btn").addEventListener("click", () => {
    for (const id of GUARD_2025) if (!state.exclude.includes(id)) state.exclude.push(id);
    renderChips();
  });
  $("clear-btn").addEventListener("click", () => { state.exclude = []; renderChips(); });
  $("reset-btn").addEventListener("click", () => { writeForm(structuredClone(DEFAULTS)); runAnalysis(); });
  $("link-btn").addEventListener("click", async () => {
    writeHash(readForm());
    try {
      await navigator.clipboard.writeText(location.href);
      $("link-btn").textContent = "Link copied";
    } catch {
      $("link-btn").textContent = "Link is in the address bar";
    }
    setTimeout(() => { $("link-btn").textContent = "Copy link"; }, 2000);
  });
  for (const name of ["monthly-view", "donor-view", "donor-scale"]) {
    document.querySelectorAll(`input[name="${name}"]`).forEach((el) => el.addEventListener("change", () => {
      const res = state.result;
      if (!res) return;
      const t = theme();
      const d = Math.max(...res.obs.filter(Number.isFinite)) < 10 ? 2 : 1;
      if (name === "monthly-view") renderMonthly(res, t, d); else renderDonors(res, t, d);
    }));
  }
  $("dl-monthly").addEventListener("click", downloadMonthly);
  $("dl-weights").addEventListener("click", downloadWeights);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderResults);
}

async function main() {
  wire();
  if (typeof Worker === "undefined" || typeof WebAssembly === "undefined") {
    setStatus("This browser cannot run the analysis", false);
    showError("The analysis needs Web Workers and WebAssembly, which this browser does not support.");
    return;
  }
  let text;
  try {
    const r = await fetch(`data/rtci.json?v=${APP_VERSION}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    text = await r.text();
    state.data = JSON.parse(text);
  } catch (err) {
    setStatus("Could not load the data", false);
    showError(`Could not load data/rtci.json (${err.message}). Serve the site over HTTP, not from a file.`);
    return;
  }
  for (const c of state.data.cities) { state.byId.set(c.id, c); state.byLabel.set(c.label, c); }
  populate();
  writeForm(readHash());
  if (typeof Plotly === "undefined") {
    showError("The charting library (Plotly) did not load; check the network connection.");
  }
  startWorker(text);
}

main();
