/* ---------------------------------------------------------------------
   Tabular Modelling Pipeline - Training Dashboard
   --------------------------------------------------------------------- */

const PALETTE = {
  catboost:       "#5b9dff",
  xgboost:        "#7c5cff",
  cann:           "#4ade80",
  cann_gbm:       "#10b981",
  ft_transformer: "#fbbf24",
  tabm:           "#fb923c",
  localglmnet:    "#a78bfa",
  drn:            "#f472b6",
  stacked_ensemble: "#f4f6fb",
};

const ARCH_ORDER = [
  "catboost", "xgboost", "cann", "cann_gbm",
  "ft_transformer", "tabm", "localglmnet", "drn", "stacked_ensemble",
];

const DATASET_LABEL = {
  house_prices: "House Prices",
  bike_sharing: "Bike Sharing",
  allstate:     "Allstate Claims",
  net_premium:  "Net Premium (legacy)",
};

let DATA = null;
let CURRENT_VIEW = "compare";
let CURRENT_RUN_ID = null;

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

window.addEventListener("DOMContentLoaded", async () => {
  try {
    const resp = await fetch("runs.json", { cache: "no-store" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    DATA = await resp.json();
  } catch (err) {
    document.getElementById("content").innerHTML =
      `<div class="loading">Failed to load runs.json: ${err.message}<br><br>` +
      `Run <code>python scripts/build_dashboard.py</code> first.</div>`;
    return;
  }

  populateMeta();
  populateRunList();
  wireSidebar();
  renderView("compare");
});

function populateMeta() {
  const nRuns = DATA.n_runs || (DATA.runs || []).length;
  const datasets = new Set((DATA.runs || []).map((r) => r.dataset));
  document.getElementById("meta-n-runs").textContent = `${nRuns} runs`;
  document.getElementById("meta-datasets").textContent =
    `${datasets.size} datasets`;
  document.getElementById("meta-best-model").textContent =
    `schema v${DATA.schema_version || "?"}`;
}

function populateRunList() {
  const list = document.getElementById("run-list");
  list.innerHTML = "";
  // Group by dataset
  const grouped = {};
  (DATA.runs || []).forEach((r) => {
    grouped[r.dataset] = grouped[r.dataset] || [];
    grouped[r.dataset].push(r);
  });
  const orderedDatasets = ["house_prices", "bike_sharing", "allstate", "net_premium"];
  orderedDatasets.forEach((ds) => {
    if (!grouped[ds]) return;
    const heading = document.createElement("li");
    heading.className = "sidebar-heading-inline";
    heading.style.cssText =
      "padding: 8px 16px 4px 16px; font-size: 10px; text-transform: uppercase; " +
      "letter-spacing: 0.08em; color: var(--text-dim); cursor: default;";
    heading.textContent = DATASET_LABEL[ds] || ds;
    list.appendChild(heading);

    grouped[ds].forEach((r) => {
      const li = document.createElement("li");
      li.className = "run-item";
      li.dataset.runId = r.run_id;
      const badges = [];
      if (r.tuned) badges.push(`<span class="badge badge-tuned">tuned</span>`);
      if (r.has_interpretability) badges.push(`<span class="badge badge-interp">interp</span>`);
      li.innerHTML =
        `<span>${r.label}${badges.join("")}</span>` +
        `<span class="run-item-sub">${r.run_id}</span>`;
      li.addEventListener("click", () => {
        CURRENT_RUN_ID = r.run_id;
        renderView("run");
      });
      list.appendChild(li);
    });
  });
}

function wireSidebar() {
  document.querySelectorAll(".view-item").forEach((el) => {
    el.addEventListener("click", () => {
      CURRENT_VIEW = el.dataset.view;
      CURRENT_RUN_ID = null;
      renderView(CURRENT_VIEW);
    });
  });
}

function setActiveSidebar(view, runId) {
  document.querySelectorAll(".view-item").forEach((el) => {
    el.classList.toggle("active", view !== "run" && el.dataset.view === view);
  });
  document.querySelectorAll(".run-item").forEach((el) => {
    el.classList.toggle("active", el.dataset.runId === runId);
  });
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

function renderView(view) {
  setActiveSidebar(view, CURRENT_RUN_ID);
  const content = document.getElementById("content");
  content.innerHTML = "";

  if (view === "compare") {
    renderCompareView(content);
  } else if (view === "kaggle") {
    renderKaggleView(content);
  } else if (view === "run") {
    const run = (DATA.runs || []).find((r) => r.run_id === CURRENT_RUN_ID);
    if (!run) {
      content.innerHTML = `<div class="empty">Run not found: ${CURRENT_RUN_ID}</div>`;
      return;
    }
    renderRunView(content, run);
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmt(val, decimals = 4) {
  if (val == null || isNaN(val)) return "-";
  if (Math.abs(val) >= 100000) return val.toFixed(0);
  if (Math.abs(val) >= 100) return val.toFixed(1);
  return val.toFixed(decimals);
}

function metricCell(val, decimals = 4, isBest = false, isWorst = false) {
  const cls = isBest ? "metric-best" : (isWorst ? "metric-worst" : "");
  return `<td class="${cls}">${fmt(val, decimals)}</td>`;
}

function findBestWorst(rows, key, higherIsBetter = true) {
  const vals = rows.map((r) => r[key]).filter((v) => v != null && !isNaN(v));
  if (!vals.length) return { best: null, worst: null };
  return {
    best: higherIsBetter ? Math.max(...vals) : Math.min(...vals),
    worst: higherIsBetter ? Math.min(...vals) : Math.max(...vals),
  };
}

function metricTable(rows, opts = {}) {
  const cols = opts.cols || [
    { key: "gini_test", label: "Test Gini", decimals: 4, higherIsBetter: true },
    { key: "gini_train", label: "Train Gini", decimals: 4, higherIsBetter: true },
    { key: "mae", label: "Test MAE", decimals: 1, higherIsBetter: false },
    { key: "rmse", label: "Test RMSE", decimals: 1, higherIsBetter: false },
    { key: "ae_ratio", label: "A/E", decimals: 3, higherIsBetter: false, distFromOne: true },
    { key: "n_params", label: "n params", decimals: 0, higherIsBetter: null },
    { key: "training_time", label: "Train time (s)", decimals: 1, higherIsBetter: null },
  ];

  // Determine best/worst for each col
  const bw = {};
  cols.forEach((c) => {
    if (c.distFromOne) {
      // A/E ratio - closest to 1 is best
      const vals = rows.map((r) => r[c.key]).filter((v) => v != null);
      let bestDelta = Infinity;
      let best = null;
      vals.forEach((v) => {
        const d = Math.abs(v - 1);
        if (d < bestDelta) {
          bestDelta = d;
          best = v;
        }
      });
      bw[c.key] = { best, worst: null };
    } else if (c.higherIsBetter !== null) {
      bw[c.key] = findBestWorst(rows, c.key, c.higherIsBetter);
    } else {
      bw[c.key] = { best: null, worst: null };
    }
  });

  // Sort by test Gini desc
  const sorted = [...rows].sort((a, b) => (b.gini_test ?? -1e9) - (a.gini_test ?? -1e9));

  const header = `<tr><th>Model</th>${cols
    .map((c) => `<th>${c.label}</th>`)
    .join("")}</tr>`;
  const body = sorted
    .map((row) => {
      const cells = cols
        .map((c) => {
          const val = row[c.key];
          const isBest = bw[c.key].best != null && val === bw[c.key].best;
          const isWorst = bw[c.key].worst != null && val === bw[c.key].worst;
          return metricCell(val, c.decimals, isBest, isWorst);
        })
        .join("");
      return `<tr><td>${row.model}</td>${cells}</tr>`;
    })
    .join("");
  return `<table class="metrics"><thead>${header}</thead><tbody>${body}</tbody></table>`;
}

// ---------------------------------------------------------------------------
// View: cross-run comparison
// ---------------------------------------------------------------------------

function renderCompareView(root) {
  root.innerHTML = `
    <h2 class="section-title">Cross-run comparison</h2>
    <p class="section-sub">Test-set performance of the best model per run, grouped by dataset.</p>
    <div id="kpi-row" class="kpi-row"></div>
    <div class="section-block">
      <h3>Best test Gini per architecture × run</h3>
      <div id="chart-gini" class="chart"></div>
    </div>
    <div class="section-block">
      <h3>Test MAE per architecture × run</h3>
      <div id="chart-mae" class="chart"></div>
    </div>
    <div class="section-block">
      <h3>Optuna tuning lift (House Prices)</h3>
      <div id="tuning-table"></div>
    </div>
  `;

  // KPI cards: best Gini per dataset
  const kpiContainer = document.getElementById("kpi-row");
  const byDataset = {};
  (DATA.runs || []).forEach((r) => {
    (byDataset[r.dataset] = byDataset[r.dataset] || []).push(r);
  });
  ["house_prices", "bike_sharing", "allstate", "net_premium"].forEach((ds) => {
    if (!byDataset[ds]) return;
    let best = -Infinity;
    let bestModel = null;
    let bestRun = null;
    byDataset[ds].forEach((r) => {
      r.models.forEach((m) => {
        if (m.gini_test != null && m.gini_test > best) {
          best = m.gini_test;
          bestModel = m.model;
          bestRun = r.label;
        }
      });
    });
    if (bestModel) {
      const card = document.createElement("div");
      card.className = "kpi-card";
      card.innerHTML = `
        <div class="kpi-label">${DATASET_LABEL[ds]}</div>
        <div class="kpi-value">${best.toFixed(4)}</div>
        <div class="kpi-sub">${bestModel} · ${bestRun}</div>
      `;
      kpiContainer.appendChild(card);
    }
  });

  renderGiniByRun("chart-gini");
  renderMAEByRun("chart-mae");
  renderTuningLift("tuning-table");
}

function renderGiniByRun(divId) {
  const datasets = ["house_prices", "bike_sharing", "allstate", "net_premium"];
  const traces = [];
  ARCH_ORDER.forEach((arch) => {
    const x = [];
    const y = [];
    (DATA.runs || []).forEach((r) => {
      if (!datasets.includes(r.dataset)) return;
      const m = r.models.find((mm) => mm.model === arch);
      if (m && m.gini_test != null) {
        x.push(`${DATASET_LABEL[r.dataset]}<br>${r.label}`);
        y.push(m.gini_test);
      }
    });
    if (x.length) {
      traces.push({
        x, y,
        type: "bar",
        name: arch,
        marker: { color: PALETTE[arch] || "#888" },
      });
    }
  });
  Plotly.newPlot(divId, traces, plotlyLayout({
    barmode: "group",
    yaxis: { title: "Test Gini" },
    height: 480,
  }), { responsive: true, displayModeBar: false });
}

function renderMAEByRun(divId) {
  const datasets = ["house_prices", "bike_sharing", "allstate"];
  const traces = [];
  ARCH_ORDER.forEach((arch) => {
    const x = [];
    const y = [];
    (DATA.runs || []).forEach((r) => {
      if (!datasets.includes(r.dataset)) return;
      const m = r.models.find((mm) => mm.model === arch);
      if (m && m.mae != null) {
        x.push(`${DATASET_LABEL[r.dataset]}<br>${r.label}`);
        y.push(m.mae);
      }
    });
    if (x.length) {
      traces.push({
        x, y,
        type: "bar",
        name: arch,
        marker: { color: PALETTE[arch] || "#888" },
      });
    }
  });
  Plotly.newPlot(divId, traces, plotlyLayout({
    barmode: "group",
    yaxis: { title: "Test MAE", type: "log" },
    height: 480,
  }), { responsive: true, displayModeBar: false });
}

function renderTuningLift(divId) {
  const hpUntuned = (DATA.runs || []).find(
    (r) => r.dataset === "house_prices" && r.run_id === "house_prices_8arch"
  );
  const hpTuned = (DATA.runs || []).find(
    (r) => r.dataset === "house_prices" && r.tuned
  );

  if (!hpUntuned || !hpTuned) {
    document.getElementById(divId).innerHTML =
      `<div class="empty">No tuned vs untuned pair available yet for House Prices. Run with --n-trials N.</div>`;
    return;
  }

  const rows = ARCH_ORDER.filter((a) => a !== "stacked_ensemble").map((arch) => {
    const u = hpUntuned.models.find((m) => m.model === arch);
    const t = hpTuned.models.find((m) => m.model === arch);
    return {
      arch,
      untuned: u ? u.gini_test : null,
      tuned: t ? t.gini_test : null,
      delta: u && t ? t.gini_test - u.gini_test : null,
    };
  });

  let html = `<table class="metrics"><thead><tr>
    <th>Architecture</th><th>Untuned Gini</th><th>Tuned Gini</th><th>Δ Gini</th>
  </tr></thead><tbody>`;
  rows.forEach((r) => {
    const deltaCls = r.delta == null ? "" :
      r.delta > 0.005 ? "metric-best" :
      r.delta < -0.005 ? "metric-worst" : "";
    const deltaStr = r.delta == null ? "-" : (r.delta > 0 ? "+" : "") + r.delta.toFixed(4);
    html += `<tr>
      <td>${r.arch}</td>
      <td>${fmt(r.untuned)}</td>
      <td>${fmt(r.tuned)}</td>
      <td class="${deltaCls}">${deltaStr}</td>
    </tr>`;
  });
  html += `</tbody></table>`;
  document.getElementById(divId).innerHTML = html;
}

// ---------------------------------------------------------------------------
// View: Kaggle leaderboard
// ---------------------------------------------------------------------------

function renderKaggleView(root) {
  const ref = DATA.kaggle_reference || {};
  let html = `
    <h2 class="section-title">Kaggle leaderboard comparison</h2>
    <p class="section-sub">Our best test-set metrics against publicly known Kaggle leaderboard ranges. Note: we score on an 80/20 random split of train.csv; Kaggle scores on held-out test.csv, so the comparison is approximate.</p>
  `;

  ["house_prices", "bike_sharing", "allstate"].forEach((ds) => {
    const r = ref[ds];
    if (!r) return;
    html += `<div class="section-block">
      <h3>${DATASET_LABEL[ds]} (metric: ${r.kaggle_metric.toUpperCase()})</h3>
      <table class="metrics"><thead><tr>
        <th>Source</th><th>${r.kaggle_metric.toUpperCase()}</th>
      </tr></thead><tbody>`;
    Object.entries(r).forEach(([key, val]) => {
      if (key === "kaggle_metric" || key === "competition_url" || key === "note") return;
      html += `<tr><td>${key.replace(/_/g, " ")}</td><td>${val}</td></tr>`;
    });
    // Our best
    const runs = (DATA.runs || []).filter((rr) => rr.dataset === ds);
    if (runs.length) {
      let bestMae = Infinity, bestModel = null, bestRun = null;
      runs.forEach((rr) => {
        rr.models.forEach((m) => {
          if (m.mae != null && m.mae < bestMae) {
            bestMae = m.mae;
            bestModel = m.model;
            bestRun = rr.label;
          }
        });
      });
      if (bestModel) {
        html += `<tr><td><strong>Our best (MAE)</strong></td><td><strong>${fmt(bestMae)}</strong> (${bestModel}, ${bestRun})</td></tr>`;
      }
    }
    html += `</tbody></table>`;
    if (r.note) {
      html += `<p class="section-sub" style="margin-top:8px"><em>${r.note}</em></p>`;
    }
    if (r.competition_url) {
      html += `<p class="section-sub"><a href="${r.competition_url}" target="_blank">Kaggle competition →</a></p>`;
    }
    html += `</div>`;
  });

  root.innerHTML = html;
}

// ---------------------------------------------------------------------------
// View: single run detail
// ---------------------------------------------------------------------------

function renderRunView(root, run) {
  const tunedBadge = run.tuned ? `<span class="badge badge-tuned">tuned</span>` : "";
  const interpBadge = run.has_interpretability ? `<span class="badge badge-interp">interpretability</span>` : "";
  root.innerHTML = `
    <h2 class="section-title">${DATASET_LABEL[run.dataset]}: ${run.label} ${tunedBadge}${interpBadge}</h2>
    <p class="section-sub">
      <span class="tag">${run.run_id}</span>
      ${run.timestamp ? `<span class="tag">${run.timestamp}</span>` : ""}
      ${run.config?.cv_folds ? `<span class="tag">${run.config.cv_folds}-fold CV</span>` : ""}
      ${run.config?.n_ensemble ? `<span class="tag">${run.config.n_ensemble}-seed DL ensemble</span>` : ""}
    </p>

    <div class="section-block">
      <h3>Per-model metrics (sorted by Test Gini)</h3>
      <div id="run-metrics-table"></div>
    </div>

    <div class="section-block">
      <h3>Test Gini vs n_params</h3>
      <div id="run-chart-scatter" class="chart"></div>
    </div>

    <div class="section-block">
      <h3>Training time per architecture</h3>
      <div id="run-chart-time" class="chart"></div>
    </div>

    <div class="section-block" id="run-fi-block">
      <h3>Feature importance</h3>
      <div id="run-fi-chart" class="chart"></div>
    </div>

    <div class="section-block" id="run-coef-block">
      <h3>LocalGLMnet coefficient stability</h3>
      <div id="run-coef-table"></div>
    </div>
  `;

  document.getElementById("run-metrics-table").innerHTML = metricTable(run.models);
  renderRunScatter("run-chart-scatter", run);
  renderRunTime("run-chart-time", run);

  if (run.feature_importance) {
    renderFeatureImportance("run-fi-chart", run.feature_importance);
  } else {
    document.getElementById("run-fi-block").innerHTML =
      `<h3>Feature importance</h3><div class="empty">Not available - re-run without --skip-interpretability to produce feature_importance.csv</div>`;
  }
  if (run.localglmnet_coef_summary) {
    renderLocalGLMnetTable("run-coef-table", run.localglmnet_coef_summary);
  } else {
    document.getElementById("run-coef-block").innerHTML =
      `<h3>LocalGLMnet coefficient stability</h3><div class="empty">Not available.</div>`;
  }
}

function renderRunScatter(divId, run) {
  const x = [], y = [], text = [], colors = [];
  run.models.forEach((m) => {
    if (m.gini_test != null && m.n_params != null) {
      x.push(m.n_params);
      y.push(m.gini_test);
      text.push(m.model);
      colors.push(PALETTE[m.model] || "#888");
    }
  });
  const trace = {
    x, y, text, mode: "markers+text",
    type: "scatter",
    textposition: "top center",
    marker: { size: 12, color: colors },
  };
  Plotly.newPlot(divId, [trace], plotlyLayout({
    xaxis: { title: "n_params", type: "log" },
    yaxis: { title: "Test Gini" },
    height: 360,
    showlegend: false,
  }), { responsive: true, displayModeBar: false });
}

function renderRunTime(divId, run) {
  const sorted = [...run.models]
    .filter((m) => m.training_time != null)
    .sort((a, b) => b.training_time - a.training_time);
  const x = sorted.map((m) => m.model);
  const y = sorted.map((m) => m.training_time);
  const colors = sorted.map((m) => PALETTE[m.model] || "#888");
  Plotly.newPlot(divId, [{
    x, y,
    type: "bar",
    marker: { color: colors },
  }], plotlyLayout({
    xaxis: { title: "" },
    yaxis: { title: "Training time (s)", type: "log" },
    height: 320,
    showlegend: false,
  }), { responsive: true, displayModeBar: false });
}

function renderFeatureImportance(divId, fi) {
  // Top 15 features per model
  const traces = Object.entries(fi).map(([model, imps]) => {
    const top = Object.entries(imps).slice(0, 15);
    return {
      y: top.map(([f, _]) => f).reverse(),
      x: top.map(([_, v]) => v).reverse(),
      type: "bar",
      orientation: "h",
      name: model,
      marker: { color: PALETTE[model] || "#888" },
    };
  });
  Plotly.newPlot(divId, traces, plotlyLayout({
    barmode: "group",
    height: Math.max(360, 24 * 15),
    margin: { l: 140 },
    xaxis: { title: "Importance" },
  }), { responsive: true, displayModeBar: false });
}

function renderLocalGLMnetTable(divId, summary) {
  let html = `<table class="metrics"><thead><tr>
    <th>Feature</th><th>Mean coef</th><th>Std</th><th>Sign stability</th>
  </tr></thead><tbody>`;
  Object.entries(summary).forEach(([feat, s]) => {
    const stabilityCls = s.sign_stability > 0.9 ? "metric-best" :
      s.sign_stability < 0.6 ? "metric-worst" : "";
    html += `<tr>
      <td>${feat}</td>
      <td>${s.mean.toExponential(2)}</td>
      <td>${s.std.toExponential(2)}</td>
      <td class="${stabilityCls}">${(s.sign_stability * 100).toFixed(0)}%</td>
    </tr>`;
  });
  html += `</tbody></table>
    <p class="section-sub" style="margin-top:8px;"><em>Sign stability = fraction of test rows where the coefficient has the same sign as the mean. Close to 100% means the model is confident about that feature's direction. Closer to 50% means the effect flips across the input space - which is exactly what LocalGLMnet was designed to detect.</em></p>`;
  document.getElementById(divId).innerHTML = html;
}

// ---------------------------------------------------------------------------
// Plotly layout helper
// ---------------------------------------------------------------------------

function plotlyLayout(extra) {
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: {
      family: "-apple-system, sans-serif",
      color: "#d6dbe5",
      size: 11,
    },
    xaxis: { gridcolor: "#2d3548", linecolor: "#2d3548" },
    yaxis: { gridcolor: "#2d3548", linecolor: "#2d3548" },
    legend: { orientation: "h", y: -0.2 },
    margin: { t: 24, r: 16, b: 80, l: 60 },
  }, extra);
}
