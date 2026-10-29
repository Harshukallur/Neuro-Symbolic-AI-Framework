"use strict";

const STRATEGY_LABEL = { fedavg: "FedAvg", fedprox: "FedProx", fedadagrad: "FedAdagrad" };
const STRATEGY_COLOR = { fedavg: "#0c5c56", fedprox: "#9a3412", fedadagrad: "#1e3a5f" };
const CLASS_COLORS = ["#0c5c56", "#b86e24", "#8f2942"];

const state = {
  overview: null,
  domain: "healthcare",
  strategy: null,
  role: "coordinator",
  clientId: 0,
  auth: null,
  view: "results",
  cases: [],
  casesKey: "",
  caseToken: 0,
  selectedIndex: null,
  filter: "all",
  query: "",
  formDomain: null,
  clientOverview: null,
  clientOverviewKey: "",
  clientCases: [],
  clientCasesKey: "",
  clientCaseToken: 0,
  clientSelectedIndex: null
};

function defaultViewForRole(role) {
  if (role === "client") return state.auth ? "client" : "login";
  return "results";
}

function loadAuth() {
  try {
    const raw = sessionStorage.getItem("nsf-auth");
    if (raw) {
      const data = JSON.parse(raw);
      if (data && data.token && Number.isInteger(data.clientId)) {
        state.auth = data;
        state.clientId = data.clientId;
      }
    }
  } catch (error) {
    state.auth = null;
  }
}

function saveAuth() {
  try {
    if (state.auth) sessionStorage.setItem("nsf-auth", JSON.stringify(state.auth));
    else sessionStorage.removeItem("nsf-auth");
  } catch (error) {
    /* ignore storage errors */
  }
}

document.addEventListener("DOMContentLoaded", init);

async function init() {
  bind();
  setStatus("Loading saved results…");
  try {
    loadAuth();
    state.overview = await api("/api/overview");
    state.strategy = currentDomain().best_strategy;
    fillStrategySelect();
    await render();
  } catch (error) {
    setStatus(error.message, true);
  }
}

function bind() {
  document.querySelectorAll("[data-domain]").forEach((button) => {
    button.addEventListener("click", () => {
      if (!state.overview || button.dataset.domain === state.domain) return;
      state.domain = button.dataset.domain;
      state.strategy = currentDomain().best_strategy;
      state.query = "";
      state.filter = "all";
      document.getElementById("case-search").value = "";
      document.getElementById("case-filter").value = "all";
      resetDetail();
      resetClientDetail();
      state.clientOverviewKey = "";
      state.clientCasesKey = "";
      fillStrategySelect();
      render();
    });
  });

  document.querySelectorAll("[data-role]").forEach((button) => {
    button.addEventListener("click", () => {
      if (!state.overview || button.dataset.role === state.role) return;
      state.role = button.dataset.role;
      state.view = defaultViewForRole(state.role);
      render();
    });
  });

  document.getElementById("login-form").addEventListener("submit", (event) => {
    event.preventDefault();
    submitLogin();
  });

  document.getElementById("logout-btn").addEventListener("click", () => {
    doLogout();
  });

  document.getElementById("strategy").addEventListener("change", (event) => {
    state.strategy = event.target.value;
    resetDetail();
    resetClientDetail();
    state.clientCasesKey = "";
    render();
  });

  document.querySelectorAll("[data-view]").forEach((button) => {
    button.addEventListener("click", () => {
      state.view = button.dataset.view;
      render();
    });
  });

  document.getElementById("case-search").addEventListener("input", (event) => {
    state.query = event.target.value;
    if (state.casesKey === `${state.domain}:${state.strategy}`) renderCaseList();
  });

  document.getElementById("case-filter").addEventListener("change", (event) => {
    state.filter = event.target.value;
    if (state.casesKey === `${state.domain}:${state.strategy}`) renderCaseList();
  });

  document.getElementById("predict-form").addEventListener("submit", (event) => {
    event.preventDefault();
    submitPredict();
  });

  document.getElementById("reset-example").addEventListener("click", () => {
    if (!state.overview) return;
    ensureForm();
    const example = currentExample();
    fillInputs(example.inputs, example.entity_id);
    showFormError("");
  });
}

async function render() {
  showView();
  markControls();
  updateSubtitle();
  // Client views require an authenticated session.
  if (state.role === "client" && !state.auth && state.view !== "login") {
    state.view = "login";
    showView();
  }
  if (state.view === "login") {
    setStatus("");
    document.getElementById("login-error").hidden = true;
    return;
  }

  try {
    if (state.view === "results") {
      renderResults();
      setStatus("");
    } else if (state.view === "cases") {
      await ensureCases();
      if (state.casesKey === `${state.domain}:${state.strategy}`) renderCaseList();
    } else if (state.view === "client") {
      await renderClientView();
    } else {
      if (state.role === "client") {
        if (!state.auth) { state.view = "login"; showView(); return; }
        const key = `${state.domain}:${state.clientId}`;
        if (state.clientOverviewKey !== key) {
          state.clientOverview = await api(
            `/api/client_overview?domain=${encodeURIComponent(state.domain)}&token=${authToken()}`
          );
          state.clientOverviewKey = key;
        }
      }
      const rebuilt = ensureForm();
      const panel = document.getElementById("predict-result");
      if (!rebuilt && panel.dataset.ready === "1") await submitPredict();
    }
  } catch (error) {
    setStatus(error.message, true);
  }
}

function currentDomain() {
  return state.overview.domains[state.domain];
}

function currentStrategy() {
  return currentDomain().strategies.find((item) => item.id === state.strategy);
}

function strategyLabel(id) {
  return STRATEGY_LABEL[id] || id;
}

function pretty(value) {
  if (value == null || value === "") return "—";
  return String(value).replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function pct(value) {
  return (Number(value) * 100).toFixed(2) + "%";
}

function esc(value) {
  return String(value).replace(/[&<>"]/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;"
  }[ch]));
}

function toneClass(label) {
  if (label === "low_risk" || label === "legitimate") return "tone-ok";
  if (label === "medium_risk") return "tone-warn";
  if (label === "high_risk" || label === "fraudulent") return "tone-bad";
  return "";
}

function formatValue(value) {
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return String(value);
    return value.toFixed(2);
  }
  return String(value);
}

function fieldLabel(name) {
  const field = currentDomain().fields.find((item) => item.name === name);
  return field ? field.label : name;
}

function setStatus(message, isError) {
  const el = document.getElementById("status");
  el.textContent = message || "";
  el.classList.toggle("error", Boolean(isError) && Boolean(message));
}

function showView() {
  for (const name of ["results", "cases", "client", "predict", "login"]) {
    const el = document.getElementById(`view-${name}`);
    if (el) el.hidden = name !== state.view;
  }
  const authed = Boolean(state.auth);
  document.querySelectorAll("nav button").forEach((button) => {
    const inRole = (button.dataset.roles || "").split(" ").includes(state.role);
    // Client tabs only appear once signed in.
    const visible = inRole && (state.role !== "client" || authed);
    button.hidden = !visible;
    if (visible && button.dataset.view === state.view) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
}

function markControls() {
  document.querySelectorAll("[data-domain]").forEach((button) => {
    button.setAttribute("aria-pressed", button.dataset.domain === state.domain ? "true" : "false");
  });
  document.querySelectorAll("[data-role]").forEach((button) => {
    button.setAttribute("aria-pressed", button.dataset.role === state.role ? "true" : "false");
  });
  const auth = document.getElementById("client-auth");
  if (auth) {
    const show = state.role === "client" && Boolean(state.auth);
    auth.hidden = !show;
    if (show) {
      document.getElementById("client-auth-label").textContent = `Signed in as Client ${state.auth.clientId}`;
    }
  }
}

function updateSubtitle() {
  if (!state.overview) return;
  const noun = state.domain === "healthcare" ? "disease-risk patients" : "card transactions";
  const prefix = state.role === "client"
    ? `Client ${state.clientId} view: only this participant's own ${noun}. `
    : "Coordinator view: global model and aggregate metrics only. ";
  document.getElementById("subtitle").textContent =
    `${prefix}Saved federated model, checked by symbolic rules before the explanation is written.`;
}

function fillStrategySelect() {
  const select = document.getElementById("strategy");
  select.innerHTML = currentDomain().strategies.map((item) =>
    `<option value="${esc(item.id)}">${esc(strategyLabel(item.id))}</option>`
  ).join("");
  select.value = state.strategy;
}

async function submitLogin() {
  const idEl = document.getElementById("login-client-id");
  const pwEl = document.getElementById("login-password");
  const errEl = document.getElementById("login-error");
  const clientId = idEl.value.trim();
  const password = pwEl.value;
  if (clientId === "" || password === "") {
    errEl.textContent = "Enter your client id and password.";
    errEl.hidden = false;
    return;
  }
  const button = document.querySelector("#login-form button[type=submit]");
  button.disabled = true;
  try {
    const data = await api("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ client_id: Number(clientId), password })
    });
    state.auth = { token: data.token, clientId: data.client_id };
    state.clientId = data.client_id;
    saveAuth();
    pwEl.value = "";
    errEl.hidden = true;
    state.clientOverviewKey = "";
    state.clientCasesKey = "";
    state.formDomain = null;
    resetClientDetail();
    state.view = "client";
    render();
  } catch (error) {
    errEl.textContent = error.message;
    errEl.hidden = false;
  } finally {
    button.disabled = false;
  }
}

async function doLogout() {
  const token = state.auth && state.auth.token;
  state.auth = null;
  saveAuth();
  state.clientOverview = null;
  state.clientOverviewKey = "";
  state.clientCases = [];
  state.clientCasesKey = "";
  state.formDomain = null;
  resetClientDetail();
  state.view = "login";
  render();
  if (token) {
    try {
      await api("/api/logout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token })
      });
    } catch (error) {
      /* best-effort */
    }
  }
}

function authToken() {
  return state.auth ? encodeURIComponent(state.auth.token) : "";
}

function resetClientDetail() {
  state.clientSelectedIndex = null;
  const detail = document.getElementById("client-case-detail");
  if (detail) {
    detail.innerHTML =
      `<p class="muted">Select one of your own records to see the neural label, the symbolic check, and the verified explanation.</p>`;
  }
}

async function renderClientView() {
  if (!state.auth) { state.view = "login"; showView(); return; }
  const key = `${state.domain}:${state.clientId}`;
  if (state.clientOverviewKey !== key) {
    setStatus("Loading your partition…");
    state.clientOverview = await api(
      `/api/client_overview?domain=${encodeURIComponent(state.domain)}&token=${authToken()}`
    );
    state.clientOverviewKey = key;
    setStatus("");
  }
  const info = state.clientOverview;
  const noun = state.domain === "healthcare" ? "patients" : "transactions";
  document.getElementById("client-summary").innerHTML = [
    ["You are", `Client ${info.client_id}`],
    ["Your records", String(info.n_samples)],
    ["Global model", strategyLabel(state.strategy)],
    ["Your label mix", ""]
  ].map(([label, value]) => {
    if (label === "Your label mix") {
      return `<article class="metric"><span>${esc(label)}</span>` +
        `<div class="mix" style="margin-top:6px">${stack(info.counts)}</div></article>`;
    }
    return `<article class="metric"><span>${esc(label)}</span><strong>${esc(value)}</strong></article>`;
  }).join("");

  document.getElementById("client-note").innerHTML =
    `Private view for <strong>Client ${info.client_id}</strong>: ${info.n_samples} ${noun} that never leave this participant. ` +
    `You can only see your own data and the global model you received — not any other client's records or counts. ` +
    info.class_names.map((name, index) => `${esc(pretty(name))}: ${info.counts[index]}`).join(" · ") + ".";

  await ensureClientCases();
  if (state.clientCasesKey === `${state.domain}:${state.strategy}:${state.clientId}`) renderClientCaseList();
}

async function ensureClientCases() {
  const key = `${state.domain}:${state.strategy}:${state.clientId}`;
  if (state.clientCasesKey === key) return;
  const token = ++state.clientCaseToken;
  setStatus("Scoring your records with the global model…");
  try {
    const data = await api(
      `/api/client_cases?domain=${encodeURIComponent(state.domain)}` +
      `&strategy=${encodeURIComponent(state.strategy)}&token=${authToken()}`
    );
    if (token !== state.clientCaseToken) return;
    state.clientCases = data.cases;
    state.clientCasesKey = key;
    resetClientDetail();
    setStatus("");
  } catch (error) {
    if (token !== state.clientCaseToken) return;
    throw error;
  }
}

function renderClientCaseList() {
  const list = document.getElementById("client-case-list");
  const records = state.clientCases;
  const scroll = list.scrollTop;
  if (!records.length) {
    list.innerHTML = `<p class="muted" style="padding:12px">This participant has no records.</p>`;
    return;
  }
  list.innerHTML = records.map((record) => {
    const tags = [];
    if (record.override_applied) tags.push('<span class="pill copper">Override</span>');
    else if (record.conflict) tags.push('<span class="pill rose">Conflict</span>');
    if (record.correct === false) tags.push('<span class="pill bad">Mismatch</span>');
    const active = record.index === state.clientSelectedIndex ? " active" : "";
    return `
      <button type="button" class="case${active}" data-index="${record.index}" aria-pressed="${record.index === state.clientSelectedIndex}">
        <span class="id">${esc(record.entity_id)}</span>
        <span class="tags">${tags.join("")}</span>
        <span class="meta">#${record.index} · True ${esc(pretty(record.true_label))} · Final ${esc(pretty(record.final_label))}</span>
      </button>`;
  }).join("");
  list.scrollTop = scroll;
  list.querySelectorAll(".case").forEach((button) => {
    button.addEventListener("click", () => {
      const index = Number(button.dataset.index);
      const record = state.clientCases.find((item) => item.index === index);
      state.clientSelectedIndex = index;
      renderClientCaseList();
      showDecision(document.getElementById("client-case-detail"), record, false);
      if (window.matchMedia("(max-width: 900px)").matches) {
        document.getElementById("client-case-detail").scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  });
}

function renderResults() {
  const domain = currentDomain();
  const selected = currentStrategy();
  const noun = state.domain === "healthcare" ? "patients" : "transactions";
  document.getElementById("metrics").innerHTML = [
    ["Neural accuracy", pct(selected.neural_accuracy)],
    ["Neuro-symbolic accuracy", pct(selected.neurosymbolic_accuracy)],
    ["Hallucination-free", pct(selected.hallucination_free_ratio)],
    ["Overrides", String(selected.overrides)],
    ["Conflicts", String(selected.conflicts)],
    ["Communication", `${selected.communication_mb.toFixed(2)} MB`]
  ].map(([label, value]) => `<article class="metric"><span>${esc(label)}</span><strong>${esc(value)}</strong></article>`).join("");

  document.getElementById("results-note").textContent =
    `${domain.title} test set: ${domain.test_size} ${noun}. ` +
    `Best neuro-symbolic strategy in the saved run: ${strategyLabel(domain.best_strategy)}. ` +
    `${strategyLabel(selected.id)} neural accuracy is ${pct(selected.neural_accuracy)}, ` +
    `neuro-symbolic accuracy is ${pct(selected.neurosymbolic_accuracy)}, ` +
    `with ${selected.overrides} overrides and ${selected.conflicts} conflicts. ` +
    `Hallucination-free explanations: ${pct(selected.hallucination_free_ratio)}.`;

  const table = document.getElementById("strategy-table");
  table.innerHTML = `
    <thead>
      <tr>
        <th>Strategy</th><th>Neural acc</th><th>NS acc</th><th>Neural F1</th><th>NS F1</th>
        <th>Conflicts</th><th>Overrides</th><th>MB</th><th>Seconds</th>
      </tr>
    </thead>
    <tbody>
      ${domain.strategies.map((item) => `
        <tr data-strategy="${esc(item.id)}" class="${item.id === state.strategy ? "active" : ""}" tabindex="0">
          <td>${esc(strategyLabel(item.id))}${item.id === domain.best_strategy ? '<span class="best-tag">best NS</span>' : ""}</td>
          <td>${pct(item.neural_accuracy)}</td>
          <td>${pct(item.neurosymbolic_accuracy)}</td>
          <td>${pct(item.neural_f1)}</td>
          <td>${pct(item.neurosymbolic_f1)}</td>
          <td>${item.conflicts}</td>
          <td>${item.overrides}</td>
          <td>${item.communication_mb.toFixed(2)}</td>
          <td>${item.wall_time_sec.toFixed(1)}</td>
        </tr>`).join("")}
    </tbody>`;
  table.querySelectorAll("tr[data-strategy]").forEach((row) => {
    const choose = () => {
      state.strategy = row.dataset.strategy;
      document.getElementById("strategy").value = state.strategy;
      resetDetail();
      render();
    };
    row.addEventListener("click", choose);
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        choose();
      }
    });
  });

  renderChart(domain.strategies);
  document.getElementById("matrix-neural").innerHTML = matrixTable(selected.confusion_neural, domain.class_names);
  document.getElementById("matrix-ns").innerHTML = matrixTable(selected.confusion_neurosymbolic, domain.class_names);
  document.getElementById("client-title").textContent =
    `Client partitions (${domain.num_clients} clients, Dirichlet alpha ${domain.non_iid_alpha})`;
  document.getElementById("clients").innerHTML = domain.clients.map((client) => `
    <div class="bar-row">
      <span>Client ${client.client_id}</span>
      <div class="stackbar">${stack(client.counts)}</div>
      <span>${client.n_samples}</span>
    </div>`).join("") + legend(domain.class_names);

  const total = domain.test_counts.reduce((sum, count) => sum + count, 0);
  document.getElementById("test-mix").innerHTML = `
    <div class="mix-row"><span>Test</span><div class="mix">${stack(domain.test_counts)}</div><span>${total}</span></div>
    <p class="hint">${domain.class_names.map((name, index) => `${esc(pretty(name))}: ${domain.test_counts[index]}`).join(" · ")}</p>
    ${legend(domain.class_names)}`;

  document.getElementById("plots").innerHTML = [
    figure(domain.plots.accuracy, "Saved accuracy curves", `${domain.title} accuracy by round`),
    figure(domain.plots.compare, "Saved neural and neuro-symbolic bars", "Neural and neuro-symbolic accuracy")
  ].join("");
}

function figure(name, caption, alt) {
  if (!name) return "";
  return `<figure class="panel"><img src="/plots/${encodeURIComponent(name)}" alt="${esc(alt)}"><figcaption>${esc(caption)}</figcaption></figure>`;
}

function stack(counts) {
  const total = counts.reduce((sum, count) => sum + count, 0) || 1;
  return counts.map((count, index) => {
    const width = (count / total) * 100;
    const color = CLASS_COLORS[index] || "#243656";
    return `<span style="width:${width}%;background:${color}"></span>`;
  }).join("");
}

function legend(classNames) {
  return `<div class="legend-dots">${classNames.map((name, index) =>
    `<span><b style="background:${CLASS_COLORS[index] || "#243656"}"></b>${esc(pretty(name))}</span>`
  ).join("")}</div>`;
}

function matrixTable(matrix, classNames) {
  const max = Math.max(1, ...matrix.flat());
  const head = `<tr><th>True \\ predicted</th>${classNames.map((name) => `<th>${esc(pretty(name))}</th>`).join("")}</tr>`;
  const body = matrix.map((row, rowIndex) => {
    const cells = row.map((count) => {
      const weight = count / max;
      const background = `rgba(12, 92, 86, ${(0.08 + 0.82 * weight).toFixed(3)})`;
      const color = weight > 0.55 ? "#fffdf8" : "#1a2332";
      return `<td style="background:${background};color:${color}">${count}</td>`;
    }).join("");
    return `<tr><th>${esc(pretty(classNames[rowIndex]))}</th>${cells}</tr>`;
  }).join("");
  return `<table class="matrix"><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

function renderChart(strategies) {
  const host = document.getElementById("chart");
  const series = strategies.filter((item) => item.history && item.history.length);
  if (!series.length) {
    host.textContent = "No round history was saved.";
    return;
  }
  const width = 720;
  const height = 280;
  const pad = { l: 48, r: 16, t: 16, b: 36 };
  const values = series.flatMap((item) => item.history.map((point) => point.accuracy));
  let lo = Math.min(...values) - 0.04;
  let hi = Math.max(...values) + 0.03;
  lo = Math.max(0, lo);
  hi = Math.min(1, hi);
  if (hi - lo < 0.1) {
    const mid = (hi + lo) / 2;
    lo = Math.max(0, mid - 0.05);
    hi = Math.min(1, mid + 0.05);
  }
  const n = series[0].history.length;
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;
  const x = (index) => pad.l + (n === 1 ? innerW / 2 : (index * innerW) / (n - 1));
  const y = (value) => pad.t + ((hi - value) / (hi - lo)) * innerH;
  let grid = "";
  for (let tick = 0; tick <= 4; tick += 1) {
    const value = lo + ((hi - lo) * tick) / 4;
    const yy = y(value);
    grid += `<line x1="${pad.l}" y1="${yy.toFixed(1)}" x2="${width - pad.r}" y2="${yy.toFixed(1)}" class="grid"/>`;
    grid += `<text x="${pad.l - 8}" y="${(yy + 4).toFixed(1)}" text-anchor="end" class="tick">${(value * 100).toFixed(0)}%</text>`;
  }
  let labels = "";
  for (let index = 0; index < n; index += 1) {
    labels += `<text x="${x(index).toFixed(1)}" y="${height - 12}" text-anchor="middle" class="tick">${index + 1}</text>`;
  }
  const lines = series.map((item) => {
    const path = item.history.map((point, index) =>
      `${index === 0 ? "M" : "L"} ${x(index).toFixed(1)} ${y(point.accuracy).toFixed(1)}`
    ).join(" ");
    const width = item.id === state.strategy ? 2.8 : 1.6;
    return `<path d="${path}" fill="none" stroke="${STRATEGY_COLOR[item.id] || "#243656"}" stroke-width="${width}"/>`;
  }).join("");
  const legendRow = series.map((item) =>
    `<span class="legend"><i style="background:${STRATEGY_COLOR[item.id] || "#243656"}"></i>${esc(strategyLabel(item.id))}</span>`
  ).join("");
  host.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Test accuracy by federated round">
      ${grid}${labels}${lines}
    </svg>
    <div class="legend-row">${legendRow}</div>`;
}

async function ensureCases() {
  const key = `${state.domain}:${state.strategy}`;
  if (state.casesKey === key) return;
  const token = ++state.caseToken;
  setStatus("Scoring the test set…");
  try {
    const data = await api(`/api/cases?domain=${encodeURIComponent(state.domain)}&strategy=${encodeURIComponent(state.strategy)}`);
    if (token !== state.caseToken) return;
    state.cases = data.cases;
    state.casesKey = key;
    resetDetail();
    setStatus("");
  } catch (error) {
    if (token !== state.caseToken) return;
    throw error;
  }
}

function filteredCases() {
  const query = state.query.trim().toLowerCase();
  return state.cases.filter((record) => {
    if (state.filter === "override" && !record.override_applied) return false;
    if (state.filter === "conflict" && !record.conflict) return false;
    if (state.filter === "wrong" && record.correct !== false) return false;
    if (state.filter === "correct" && record.correct !== true) return false;
    if (!query) return true;
    const blob = [record.entity_id, record.true_label, record.final_label, record.neural_label, record.symbolic_label]
      .join(" ")
      .toLowerCase();
    return blob.includes(query);
  });
}

function renderCaseList() {
  const rows = filteredCases();
  const list = document.getElementById("case-list");
  const scroll = list.scrollTop;
  const overrides = state.cases.filter((record) => record.override_applied).length;
  document.getElementById("case-count").textContent =
    `Showing ${rows.length} of ${state.cases.length}. Overrides in this test set: ${overrides}.`;
  if (!rows.length) {
    list.innerHTML = `<p class="muted" style="padding:12px">No cases match this filter.</p>`;
    return;
  }
  list.innerHTML = rows.map((record) => {
    const tags = [];
    if (record.override_applied) tags.push('<span class="pill copper">Override</span>');
    else if (record.conflict) tags.push('<span class="pill rose">Conflict</span>');
    if (record.correct === false) tags.push('<span class="pill bad">Mismatch</span>');
    const active = record.index === state.selectedIndex ? " active" : "";
    return `
      <button type="button" class="case${active}" data-index="${record.index}" aria-pressed="${record.index === state.selectedIndex}">
        <span class="id">${esc(record.entity_id)}</span>
        <span class="tags">${tags.join("")}</span>
        <span class="meta">#${record.index} · True ${esc(pretty(record.true_label))} · Final ${esc(pretty(record.final_label))}</span>
      </button>`;
  }).join("");
  list.scrollTop = scroll;
  list.querySelectorAll(".case").forEach((button) => {
    button.addEventListener("click", () => {
      const index = Number(button.dataset.index);
      const record = state.cases.find((item) => item.index === index);
      state.selectedIndex = index;
      renderCaseList();
      showDecision(document.getElementById("case-detail"), record, false);
      if (window.matchMedia("(max-width: 900px)").matches) {
        document.getElementById("case-detail").scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  });
}

function resetDetail() {
  state.selectedIndex = null;
  document.getElementById("case-detail").innerHTML =
    `<p class="muted">Select a case to see the neural label, the symbolic check, and the verified explanation.</p>`;
}

function showDecision(container, record, editable) {
  container.innerHTML = renderDecision(record, editable);
  const button = container.querySelector("[data-edit]");
  if (button) button.addEventListener("click", () => useInPredict(record));
}

function renderDecision(record, editable) {
  const probs = record.probabilities.map((item) => {
    const width = Math.max(0, Math.min(100, item.value * 100));
    return `
      <div class="prob">
        <span>${esc(pretty(item.label))}</span>
        <div class="track"><div class="fill" style="width:${width.toFixed(1)}%"></div></div>
        <span>${(item.value * 100).toFixed(1)}%</span>
      </div>`;
  }).join("");
  const rules = record.fired_rules.length
    ? `<ul class="rules">${record.fired_rules.map((rule) => {
        const sign = Number(rule.points) < 0 ? "neg" : "pos";
        const prefix = Number(rule.points) > 0 ? "+" : "";
        return `<li><span class="rule-name">${esc(rule.rule)}</span><span class="points ${sign}">${prefix}${Number(rule.points).toFixed(1)}</span><p>${esc(rule.message)}</p></li>`;
      }).join("")}</ul>`
    : `<p class="muted">No symbolic rules fired.</p>`;
  let banner;
  if (record.warnings.length) {
    const kind = record.override_applied ? "copper" : "rose";
    banner = record.warnings.map((warning) => `<p class="banner ${kind}">${esc(warning)}</p>`).join("");
  } else {
    banner = `<p class="banner ok">Neural and symbolic layers agree.</p>`;
  }
  const guard = record.hallucination_guard.passed
    ? `<p class="guard ok">Hallucination guard passed. The explanation names this entity and only rules that fired.</p>`
    : `<p class="guard bad">Hallucination guard failed${record.hallucination_guard.illegal_rules.length ? ": " + esc(record.hallucination_guard.illegal_rules.join(", ")) : ""}.</p>`;
  const match = record.correct === true
    ? `<span class="pill ok">Matches test label</span>`
    : record.correct === false
      ? `<span class="pill bad">Differs from test label</span>`
      : "";
  const features = Object.entries(record.features).map(([name, value]) =>
    `<div><dt>${esc(fieldLabel(name))}</dt><dd>${esc(formatValue(value))}</dd></div>`
  ).join("");
  const edit = editable ? `<button type="button" class="ghost" data-edit>Edit this case</button>` : "";
  return `
    <div class="decision-head">
      <div>
        <p class="eyebrow">${esc(pretty(record.domain))} · ${esc(strategyLabel(record.strategy))}</p>
        <h2>${esc(record.entity_id)}</h2>
      </div>
      ${match}
    </div>
    <div class="verdict">
      <div class="cell"><span>True label</span><strong class="${toneClass(record.true_label)}">${esc(pretty(record.true_label))}</strong><small>${record.true_label ? "Held-out test set" : "No test label"}</small></div>
      <div class="cell"><span>Neural</span><strong class="${toneClass(record.neural_label)}">${esc(pretty(record.neural_label))}</strong><small>${(record.neural_confidence * 100).toFixed(1)}% confidence</small></div>
      <div class="cell"><span>Symbolic</span><strong class="${toneClass(record.symbolic_label)}">${esc(pretty(record.symbolic_label))}</strong><small>Score ${Number(record.symbolic_score).toFixed(1)}</small></div>
      <div class="cell${record.override_applied ? " override" : ""}"><span>Final</span><strong class="${toneClass(record.final_label)}">${esc(pretty(record.final_label))}</strong><small>${(record.confidence * 100).toFixed(1)}% confidence</small></div>
    </div>
    ${banner}
    <h3>Neural probabilities</h3>
    <div class="probs">${probs}</div>
    <h3>Fired rules</h3>
    ${rules}
    <h3>Verified explanation</h3>
    <p class="explanation">${esc(record.explanation)}</p>
    ${guard}
    <h3>Features used</h3>
    <dl class="features">${features}</dl>
    <p class="model-file">Model file: ${esc(record.model_file)}</p>
    ${edit}`;
}

function currentExample() {
  if (state.role === "client" && state.clientOverview &&
      state.clientOverviewKey === `${state.domain}:${state.clientId}`) {
    return state.clientOverview.example;
  }
  return currentDomain().example;
}

function ensureForm() {
  const key = `${state.domain}:${state.role}:${state.clientId}`;
  if (state.formDomain === key) return false;
  buildForm();
  state.formDomain = key;
  clearPredictResult();
  return true;
}

function buildForm() {
  const domain = currentDomain();
  document.getElementById("fields").innerHTML = domain.fields.map((field) => {
    if (field.kind === "bool") {
      return `<label class="check"><input type="checkbox" name="f-${esc(field.name)}"><span>${esc(field.label)}</span></label>`;
    }
    const unit = field.unit ? ` <small>(${esc(field.unit)})</small>` : "";
    const step = field.kind === "int" ? "1" : "any";
    return `<label>${esc(field.label)}${unit}<input type="number" name="f-${esc(field.name)}" step="${step}" inputmode="decimal"></label>`;
  }).join("");
  const example = currentExample();
  fillInputs(example.inputs, example.entity_id);
  showFormError("");
}

function fillInputs(inputs, entityId) {
  document.getElementById("entity-id").value = entityId || "CUSTOM";
  for (const field of currentDomain().fields) {
    const el = document.querySelector(`[name="f-${field.name}"]`);
    if (!el) continue;
    const value = inputs[field.name];
    if (field.kind === "bool") el.checked = Boolean(value);
    else el.value = value;
  }
}

function readInputs() {
  const inputs = {};
  for (const field of currentDomain().fields) {
    const el = document.querySelector(`[name="f-${field.name}"]`);
    if (field.kind === "bool") {
      inputs[field.name] = el.checked;
      continue;
    }
    if (el.value.trim() === "" || !Number.isFinite(Number(el.value))) {
      showFormError(`Enter a valid number for ${field.label}.`);
      el.focus();
      return null;
    }
    inputs[field.name] = field.kind === "int" ? Math.round(Number(el.value)) : Number(el.value);
  }
  const entity = document.getElementById("entity-id").value.trim() || "CUSTOM";
  if (!/^[A-Za-z0-9_-]{1,40}$/.test(entity)) {
    showFormError("Use 1–40 letters, numbers, underscores, or hyphens for the entity id.");
    return null;
  }
  return inputs;
}

function showFormError(message) {
  const el = document.getElementById("form-error");
  el.hidden = !message;
  el.textContent = message || "";
}

function clearPredictResult() {
  const panel = document.getElementById("predict-result");
  panel.dataset.ready = "0";
  panel.innerHTML = `<p class="muted">Run verification to score this record with the selected model.</p>`;
}

async function submitPredict() {
  showFormError("");
  const inputs = readInputs();
  if (!inputs) return;
  const button = document.querySelector("#predict-form button[type=submit]");
  button.disabled = true;
  setStatus("Running verification…");
  try {
    const record = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        domain: state.domain,
        strategy: state.strategy,
        entity_id: document.getElementById("entity-id").value.trim() || "CUSTOM",
        inputs
      })
    });
    const panel = document.getElementById("predict-result");
    showDecision(panel, record, false);
    panel.dataset.ready = "1";
    setStatus("");
  } catch (error) {
    showFormError(error.message);
    setStatus(error.message, true);
  } finally {
    button.disabled = false;
  }
}

async function useInPredict(record) {
  state.view = "predict";
  showView();
  ensureForm();
  fillInputs(record.inputs, record.entity_id);
  await submitPredict();
}

async function api(url, options) {
  const response = await fetch(url, options);
  let data = {};
  try {
    data = await response.json();
  } catch (error) {
    data = {};
  }
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
