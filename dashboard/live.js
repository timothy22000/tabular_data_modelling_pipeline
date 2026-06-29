/* ---------------------------------------------------------------------
   Tabular Modelling Pipeline - Live Training Monitor
   --------------------------------------------------------------------- */

const POLL_INTERVAL_MS = 2000;
const PALETTE = {
  catboost:       "#5b9dff",
  xgboost:        "#7c5cff",
  cann:           "#4ade80",
  cann_gbm:       "#10b981",
  ft_transformer: "#fbbf24",
  tabm:           "#fb923c",
  localglmnet:    "#a78bfa",
  drn:            "#f472b6",
};

let CURRENT_PATH = null;
let POLL_TIMER = null;
let START_TIME_MS = null;

// Per-architecture state
const PARSED = {
  curves: {}, // {arch: {memberN: [{epoch, train_loss, val_loss}, ...]}}
  optuna: {}, // {arch: [{trial, score}, ...]}
  completed: [], // [{arch, gini, mae, ae, time_sec}]
  active_arch: null,
  active_member: null,
  latest_epoch: null,
  best_val: null,
  pipeline_finished: false,
  total_elapsed_sec: null,
};

document.getElementById("connect-btn").addEventListener("click", () => {
  const path = document.getElementById("log-path-input").value.trim();
  if (!path) return;
  startPolling(path);
});

function startPolling(path) {
  CURRENT_PATH = path;
  resetState();
  START_TIME_MS = Date.now();
  if (POLL_TIMER) clearInterval(POLL_TIMER);
  document.getElementById("curve-sub").textContent =
    `Tailing ${path} (poll every ${POLL_INTERVAL_MS}ms)`;
  poll();
  POLL_TIMER = setInterval(poll, POLL_INTERVAL_MS);
}

function resetState() {
  PARSED.curves = {};
  PARSED.optuna = {};
  PARSED.completed = [];
  PARSED.active_arch = null;
  PARSED.active_member = null;
  PARSED.latest_epoch = null;
  PARSED.best_val = null;
  PARSED.pipeline_finished = false;
  PARSED.total_elapsed_sec = null;
}

// ---------------------------------------------------------------------------
// Polling
// ---------------------------------------------------------------------------

async function poll() {
  try {
    // Try the SSE-style endpoint first, fall back to direct file fetch.
    // The simplest setup is to serve the repo root via python -m http.server,
    // in which case the log file is accessible at /<path> from the dashboard.
    // Dashboard lives at /dashboard/ so we need to go up one level.
    const url = `../${CURRENT_PATH}?_t=${Date.now()}`;
    const resp = await fetch(url, { cache: "no-store" });
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status} for ${url}`);
    }
    const text = await resp.text();
    parseLogContent(text);
    render();
  } catch (err) {
    document.getElementById("meta-status").textContent = `error`;
    document.getElementById("meta-status").style.color = "var(--danger)";
    document.getElementById("log-tail").textContent =
      `Failed to read log: ${err.message}\n\n` +
      `Make sure you started the server from the repo root, e.g.:\n` +
      `  cd ${"$"}{repo_root} && python -m http.server 8000\n` +
      `then visit http://localhost:8000/dashboard/live.html`;
  }
}

// ---------------------------------------------------------------------------
// Log parsers
// ---------------------------------------------------------------------------

// e.g. "  [ft_transformer_m2] Epoch  70/300 - train_loss=9.294189  val_loss=8.787351  lr=4.57e-04"
const EPOCH_RE = /\[([a-z_]+)_m(\d+)\]\s+Epoch\s+(\d+)\/\d+\s+.*?train_loss=([\d.eE+-]+)\s+val_loss=([\d.eE+-]+)/;

// e.g. "  Training: CATBOOST"
const ARCH_START_RE = /\s+Training:\s+([A-Z_]+)/;

// e.g. "  [catboost] DONE in 4.4s - Test Gini=0.2061 | MAE=16868 | A/E=1.0250"
const ARCH_DONE_RE = /\[([a-z_]+)\]\s+DONE in\s+([\d.]+)s\s+-\s+Test Gini=([\d.-]+)\s+\|\s+MAE=([\d.-]+)\s+\|\s+A\/E=([\d.-]+)/;

// e.g. "  xgboost best trial #22 (RMSE=27629.2207): {...}"
const OPTUNA_BEST_RE = /([a-z_]+)\s+best trial\s+#(\d+)\s+\((?:RMSE|val_deviance)=([\d.-]+)\)/;

// e.g. "Pipeline finished in 568.3s (9.5 min)"
const FINISH_RE = /Pipeline finished in\s+([\d.]+)s/;

function parseLogContent(text) {
  // Reset transient curves but keep completed list; we re-parse the whole
  // file every poll to keep state consistent if the user reconnects to a
  // partial run.
  PARSED.curves = {};
  PARSED.optuna = {};
  PARSED.completed = [];
  PARSED.active_arch = null;
  PARSED.active_member = null;
  PARSED.latest_epoch = null;
  PARSED.best_val = null;
  PARSED.pipeline_finished = false;
  PARSED.total_elapsed_sec = null;

  const lines = text.split("\n");
  for (const line of lines) {
    let m;
    if ((m = line.match(EPOCH_RE))) {
      const [, arch, member, epoch, train, val] = m;
      const memberKey = `m${member}`;
      PARSED.curves[arch] = PARSED.curves[arch] || {};
      PARSED.curves[arch][memberKey] = PARSED.curves[arch][memberKey] || [];
      PARSED.curves[arch][memberKey].push({
        epoch: parseInt(epoch),
        train_loss: parseFloat(train),
        val_loss: parseFloat(val),
      });
      PARSED.active_arch = arch;
      PARSED.active_member = memberKey;
      PARSED.latest_epoch = parseInt(epoch);
      const valNum = parseFloat(val);
      if (PARSED.best_val == null || valNum < PARSED.best_val) {
        PARSED.best_val = valNum;
      }
    } else if ((m = line.match(ARCH_START_RE))) {
      PARSED.active_arch = m[1].toLowerCase();
      PARSED.active_member = null;
      PARSED.latest_epoch = null;
      PARSED.best_val = null;
    } else if ((m = line.match(ARCH_DONE_RE))) {
      const [, arch, time, gini, mae, ae] = m;
      PARSED.completed.push({
        arch,
        gini: parseFloat(gini),
        mae: parseFloat(mae),
        ae: parseFloat(ae),
        time_sec: parseFloat(time),
      });
      // Clear active state for this arch
      if (PARSED.active_arch === arch) {
        PARSED.active_arch = null;
        PARSED.active_member = null;
      }
    } else if ((m = line.match(OPTUNA_BEST_RE))) {
      const [, arch, trial, score] = m;
      PARSED.optuna[arch] = PARSED.optuna[arch] || [];
      PARSED.optuna[arch].push({
        trial: parseInt(trial),
        score: parseFloat(score),
      });
    } else if ((m = line.match(FINISH_RE))) {
      PARSED.pipeline_finished = true;
      PARSED.total_elapsed_sec = parseFloat(m[1]);
    }
  }

  // Store the tail for raw display
  PARSED.tail = lines.slice(-40).join("\n");
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

function render() {
  // Status
  const statusEl = document.getElementById("meta-status");
  if (PARSED.pipeline_finished) {
    statusEl.textContent = "complete";
    statusEl.style.color = "var(--success)";
  } else if (PARSED.active_arch) {
    statusEl.textContent = "training";
    statusEl.style.color = "var(--accent)";
  } else {
    statusEl.textContent = "tailing";
    statusEl.style.color = "var(--warn)";
  }

  // Elapsed (since connect)
  const elapsedSec = (Date.now() - START_TIME_MS) / 1000;
  document.getElementById("meta-elapsed").textContent =
    `${Math.floor(elapsedSec / 60)}m ${Math.floor(elapsedSec % 60)}s`;

  // Current arch
  document.getElementById("meta-arch").textContent =
    PARSED.active_arch ? PARSED.active_arch : (PARSED.pipeline_finished ? "done" : "starting");

  // Sidebar status
  document.getElementById("active-arch").textContent = PARSED.active_arch || "-";
  document.getElementById("active-member").textContent = PARSED.active_member || "-";
  document.getElementById("latest-epoch").textContent =
    PARSED.latest_epoch != null ? PARSED.latest_epoch : "-";
  document.getElementById("best-val").textContent =
    PARSED.best_val != null ? PARSED.best_val.toExponential(3) : "-";

  // Completed list
  const completedEl = document.getElementById("completed-list");
  if (PARSED.completed.length === 0) {
    completedEl.innerHTML = `<div class="empty" style="padding:8px 0;">None yet</div>`;
  } else {
    completedEl.innerHTML = PARSED.completed
      .map((c) =>
        `<div style="margin-bottom:8px;">
          <strong style="color:${PALETTE[c.arch] || '#888'};">${c.arch}</strong><br>
          <span style="color:var(--text-dim);">
            Gini=${c.gini.toFixed(4)} · MAE=${c.mae.toFixed(0)} · ${c.time_sec.toFixed(1)}s
          </span>
        </div>`
      ).join("");
  }

  // Loss chart
  renderLossChart();
  renderOptunaChart();

  // Raw tail
  document.getElementById("log-tail").textContent = PARSED.tail || "(empty)";
}

function renderLossChart() {
  const traces = [];
  Object.entries(PARSED.curves).forEach(([arch, members]) => {
    Object.entries(members).forEach(([member, pts]) => {
      const color = PALETTE[arch] || "#888";
      // Train loss (solid)
      traces.push({
        x: pts.map((p) => p.epoch),
        y: pts.map((p) => p.train_loss),
        type: "scatter",
        mode: "lines",
        name: `${arch}/${member}/train`,
        line: { color, width: 1, dash: "solid" },
        showlegend: false,
      });
      // Val loss (dashed, more visible)
      traces.push({
        x: pts.map((p) => p.epoch),
        y: pts.map((p) => p.val_loss),
        type: "scatter",
        mode: "lines",
        name: `${arch} (${member})`,
        line: { color, width: 2, dash: "dash" },
      });
    });
  });

  Plotly.react("loss-chart", traces, plotlyLayout({
    yaxis: { title: "Loss", type: "log" },
    xaxis: { title: "Epoch" },
    height: 400,
    legend: { orientation: "v" },
  }), { responsive: true, displayModeBar: false });
}

function renderOptunaChart() {
  const traces = [];
  Object.entries(PARSED.optuna).forEach(([arch, trials]) => {
    if (trials.length < 2) return;
    traces.push({
      x: trials.map((t) => t.trial),
      y: trials.map((t) => t.score),
      type: "scatter",
      mode: "lines+markers",
      name: arch,
      line: { color: PALETTE[arch] || "#888" },
    });
  });
  if (traces.length === 0) {
    Plotly.react("optuna-chart", [], plotlyLayout({
      annotations: [{
        text: "No Optuna trials parsed yet",
        xref: "paper", yref: "paper",
        x: 0.5, y: 0.5, showarrow: false,
        font: { color: "#8b94a8" },
      }],
      height: 320,
    }), { responsive: true, displayModeBar: false });
    return;
  }
  Plotly.react("optuna-chart", traces, plotlyLayout({
    yaxis: { title: "Best trial score (lower is better)" },
    xaxis: { title: "Trial #" },
    height: 320,
  }), { responsive: true, displayModeBar: false });
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
    margin: { t: 24, r: 16, b: 60, l: 60 },
  }, extra);
}
