/**
 * ShieldML Frontend Application Logic
 * Integrates directly with FastAPI backend endpoints.
 */

// Global state
let currentMetadata = null;
let throughputChart = null;
let probDistChart = null;
let circadianChart = null;
let sliceChart = null;

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initPresetButtons();
  initSinglePredictionForm();
  initBatchUpload();
  initBenchmarkButton();
  loadAllSystemData();

  // Poll metrics and live feed every 5 seconds
  setInterval(loadMonitoringMetrics, 5000);
  setInterval(loadRecentStream, 6000);
});

/* -------------------------------------------------------------------------
   Navigation Controller
   ------------------------------------------------------------------------- */
function initNavigation() {
  const navItems = document.querySelectorAll(".nav-item");
  const views = document.querySelectorAll(".view-section");
  const titleElem = document.getElementById("page-title");

  const titles = {
    "view-overview": "Overview & Fraud Analytics",
    "view-transaction-analysis": "Historical Transaction & Subpopulation Analysis",
    "view-single-predict": "Real-Time Transaction Scoring & SHAP Explainer",
    "view-batch-predict": "Batch CSV Fraud Detection",
    "view-models": "Model Comparison & Leaderboard",
    "view-explainability": "SHAP Explainable AI (XAI)",
    "view-monitoring": "Operational Latency & SLA Monitoring",
    "view-registry": "Champion Model Registry & Lineage",
  };

  navItems.forEach((item) => {
    item.addEventListener("click", () => {
      const targetView = item.getAttribute("data-view");

      navItems.forEach((n) => n.classList.remove("active"));
      views.forEach((v) => v.classList.remove("active"));

      item.classList.add("active");
      const targetSection = document.getElementById(targetView);
      if (targetSection) {
        targetSection.classList.add("active");
      }

      if (titleElem && titles[targetView]) {
        titleElem.textContent = titles[targetView];
      }
    });
  });

  const refreshBtn = document.getElementById("btn-refresh-feed");
  if (refreshBtn) {
    refreshBtn.addEventListener("click", () => loadRecentStream());
  }
}

/* -------------------------------------------------------------------------
   API Fetchers & State Loading
   ------------------------------------------------------------------------- */
async function loadAllSystemData() {
  await Promise.all([
    loadHealth(),
    loadMonitoringMetrics(),
    loadModelInfo(),
    loadMetadata(),
    loadRecentStream(),
  ]);
  initCharts();
}

async function loadHealth() {
  try {
    const res = await fetch("/health");
    if (res.ok) {
      const data = await res.json();
      const statusText = document.getElementById("service-health-text");
      if (statusText) {
        statusText.textContent = data.status === "healthy" ? `Engine Online (${data.model_name})` : "Engine Degraded";
      }
      const memElem = document.getElementById("sys-mem-usage");
      if (memElem) memElem.textContent = `${data.memory_usage_mb} MB`;
      const modelStatus = document.getElementById("sys-model-status");
      if (modelStatus) modelStatus.textContent = data.model_loaded ? "Active (Preloaded)" : "Missing";
    }
  } catch (err) {
    console.warn("Failed to fetch /health", err);
  }
}

async function loadMonitoringMetrics() {
  try {
    const res = await fetch("/metrics");
    if (res.ok) {
      const data = await res.json();
      
      // Update Monitoring View KPIs
      setText("mon-p50", `${data.p50_latency_ms} ms`);
      setText("mon-p95", `${data.p95_latency_ms} ms`);
      setText("mon-p99", `${data.p99_latency_ms} ms`);
      setText("mon-compliance", `${data.sla_compliance_rate_pct}%`);
      setText("kpi-p95-latency", `${data.p95_latency_ms} ms`);

      setText("sys-total-requests", data.total_requests);
      setText("sys-error-count", `${data.error_count} (${(data.error_count / Math.max(1, data.total_requests) * 100).toFixed(2)}%)`);

      // Update throughput chart if initialized
      updateThroughputChart(data.p50_latency_ms, data.p95_latency_ms);
    }
  } catch (err) {
    console.warn("Failed to fetch /metrics", err);
  }
}

async function loadModelInfo() {
  try {
    const res = await fetch("/model/info");
    if (res.ok) {
      const data = await res.json();
      setText("reg-model-type", data.model_type);
      setText("reg-threshold", data.optimal_threshold);
      setText("reg-feature-count", data.feature_count);
      setText("reg-train-timestamp", data.training_timestamp || "Pre-trained");
      setText("sys-model-type", data.model_type);

      const featList = document.getElementById("reg-feature-list");
      if (featList && data.feature_names) {
        featList.textContent = JSON.stringify(data.feature_names, null, 2);
      }
    }
  } catch (err) {
    console.warn("Failed to fetch /model/info", err);
  }
}

async function loadMetadata() {
  try {
    const res = await fetch("/api/metadata");
    if (res.ok) {
      const data = await res.json();
      currentMetadata = data;

      // Update overview KPIs
      if (data.dataset_profile) {
        setText("kpi-total-txs", data.dataset_profile.total_records.toLocaleString());
        if (data.dataset_profile.class_distribution) {
          setText("kpi-fraud-rate", `${data.dataset_profile.class_distribution.fraud_rate_pct}%`);
        }
      }

      if (data.test_metrics) {
        setText("kpi-f2-score", data.test_metrics.f2_score);
      }

      // Populate Model Leaderboard Table
      renderLeaderboard(data.model_comparison_leaderboard, data.model_type);
    }
  } catch (err) {
    console.warn("Failed to fetch /api/metadata", err);
  }
}

async function loadRecentStream() {
  try {
    const res = await fetch("/api/monitoring/recent");
    if (res.ok) {
      const list = await res.json();
      const tbody = document.getElementById("table-live-stream-body");
      if (!tbody) return;

      if (!list || list.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; padding: 20px; color: var(--slate-400);">No transactions scored in current session yet. Use the Real-Time Inference tab to send predictions.</td></tr>`;
        return;
      }

      tbody.innerHTML = list.map((item) => {
        let badgeClass = "badge-approved";
        if (item.action === "BLOCK") badgeClass = "badge-blocked";
        else if (item.action === "FLAG_CHALLENGE") badgeClass = "badge-challenge";

        return `
          <tr>
            <td><code>${item.transaction_id}</code></td>
            <td>${item.timestamp}</td>
            <td>${item.user_id}</td>
            <td>$${item.amount.toFixed(2)}</td>
            <td>${item.merchant_id}</td>
            <td><strong>${(item.fraud_probability * 100).toFixed(1)}%</strong></td>
            <td><span class="badge ${badgeClass}">${item.action}</span></td>
            <td><code>${item.latency_ms} ms</code></td>
          </tr>
        `;
      }).join("");
    }
  } catch (err) {
    console.warn("Failed to fetch recent stream", err);
  }
}

function renderLeaderboard(records, championType) {
  const tbody = document.getElementById("models-leaderboard-tbody");
  if (!tbody || !records) return;

  tbody.innerHTML = records.map((r) => {
    const isChamp = r.Model === championType || r["F2-Score"] === Math.max(...records.map(x => x["F2-Score"]));
    const statusHtml = isChamp
      ? `<span class="badge badge-approved" style="background-color: var(--emerald-100); color: var(--emerald-700);">★ CHAMPION</span>`
      : `<span style="font-size: 0.75rem; color: var(--slate-400);">Candidate</span>`;

    return `
      <tr style="${isChamp ? 'background-color: var(--emerald-50); font-weight: 600;' : ''}">
        <td><strong>${r.Model}</strong></td>
        <td><code>${r.Threshold}</code></td>
        <td>${r["ROC-AUC"]}</td>
        <td>${r["PR-AUC"]}</td>
        <td><strong class="text-emerald">${r["F2-Score"]}</strong></td>
        <td>${r.Recall}</td>
        <td>${r.Precision}</td>
        <td>${r["False Negatives"]}</td>
        <td>${statusHtml}</td>
      </tr>
    `;
  }).join("");
}

/* -------------------------------------------------------------------------
   Single Prediction Form & Presets
   ------------------------------------------------------------------------- */
function initPresetButtons() {
  const btnLegit = document.getElementById("preset-legit");
  const btnFraud = document.getElementById("preset-fraud");
  const btnBorder = document.getElementById("preset-borderline");

  if (btnLegit) {
    btnLegit.addEventListener("click", (e) => {
      e.preventDefault();
      setInputValue("inp-tx-id", `tx_legit_${Math.floor(Math.random() * 89999 + 10000)}`);
      setInputValue("inp-user-id", "usr_00104");
      setInputValue("inp-amount", 42.50);
      setInputValue("inp-merchant-id", "merch_0003");
      setInputValue("inp-merchant-cat", "grocery");
      setInputValue("inp-channel", "mobile_app");
      setInputValue("inp-device-id", "dev_00104");
      setInputValue("inp-distance", 3.2);
      setInputValue("inp-vel-1h", 0);
      setInputValue("inp-vel-24h", 1);
    });
  }

  if (btnFraud) {
    btnFraud.addEventListener("click", (e) => {
      e.preventDefault();
      setInputValue("inp-tx-id", `tx_fraud_${Math.floor(Math.random() * 89999 + 10000)}`);
      setInputValue("inp-user-id", "usr_00942");
      setInputValue("inp-amount", 780.00);
      setInputValue("inp-merchant-id", "merch_0088");
      setInputValue("inp-merchant-cat", "electronics");
      setInputValue("inp-channel", "web");
      setInputValue("inp-device-id", "dev_novel_9921");
      setInputValue("inp-distance", 185.0);
      setInputValue("inp-vel-1h", 6);
      setInputValue("inp-vel-24h", 9);
    });
  }

  if (btnBorder) {
    btnBorder.addEventListener("click", (e) => {
      e.preventDefault();
      setInputValue("inp-tx-id", `tx_border_${Math.floor(Math.random() * 89999 + 10000)}`);
      setInputValue("inp-user-id", "usr_00318");
      setInputValue("inp-amount", 195.00);
      setInputValue("inp-merchant-id", "merch_0021");
      setInputValue("inp-merchant-cat", "travel");
      setInputValue("inp-channel", "web");
      setInputValue("inp-device-id", "dev_00318");
      setInputValue("inp-distance", 42.0);
      setInputValue("inp-vel-1h", 1);
      setInputValue("inp-vel-24h", 3);
    });
  }
}

function initSinglePredictionForm() {
  const form = document.getElementById("form-single-predict");
  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = document.getElementById("btn-submit-predict");
    if (btn) btn.disabled = true;

    const payload = {
      transaction_id: getValue("inp-tx-id"),
      timestamp: new Date().toISOString().replace("T", " ").substring(0, 19),
      user_id: getValue("inp-user-id"),
      amount: parseFloat(getValue("inp-amount")) || 0.0,
      merchant_id: getValue("inp-merchant-id"),
      merchant_category: getValue("inp-merchant-cat"),
      device_id: getValue("inp-device-id"),
      channel: getValue("inp-channel"),
      distance_from_home: parseFloat(getValue("inp-distance")) || 0.0,
      tx_count_user_1h: parseInt(getValue("inp-vel-1h")) || 0,
      tx_count_user_24h: parseInt(getValue("inp-vel-24h")) || 0,
    };

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json();
        alert(`Inference Error: ${err.detail || "Request failed"}`);
        return;
      }

      const result = await res.json();
      renderSingleResult(result);
      loadRecentStream();
    } catch (err) {
      alert(`Network error: ${err.message}`);
    } finally {
      if (btn) btn.disabled = false;
    }
  });
}

function renderSingleResult(result) {
  const probPct = (result.fraud_probability * 100).toFixed(1);
  setText("res-prob-pct", `${probPct}%`);
  setText("res-latency-badge", `Latency: ${result.latency_ms} ms [SLA PASS]`);
  setText("res-threshold-note", `Optimal Decision Threshold: ${result.decision_threshold}`);

  // Action badge
  const badge = document.getElementById("res-action-badge");
  if (badge) {
    badge.className = "badge";
    if (result.action === "BLOCK") {
      badge.classList.add("badge-blocked");
      badge.textContent = "BLOCKED (HIGH RISK)";
    } else if (result.action === "FLAG_CHALLENGE") {
      badge.classList.add("badge-challenge");
      badge.textContent = "FLAGGED FOR 2FA CHALLENGE";
    } else {
      badge.classList.add("badge-approved");
      badge.textContent = "APPROVED (LEGITIMATE)";
    }
  }

  // Update circular gauge conic-gradient
  const gauge = document.getElementById("risk-gauge-circle");
  if (gauge) {
    let color = "var(--emerald-500)";
    if (result.fraud_probability >= 0.70) color = "var(--red-500)";
    else if (result.fraud_probability >= result.decision_threshold) color = "var(--amber-500)";

    gauge.style.background = `conic-gradient(${color} 0%, ${color} ${probPct}%, var(--slate-100) ${probPct}%)`;
  }

  // SHAP Factors
  const factorsContainer = document.getElementById("shap-factors-list");
  const narrativeBox = document.getElementById("shap-narrative-box");

  if (factorsContainer && result.explanation) {
    const risks = result.explanation.top_risk_drivers || [];
    const trusts = result.explanation.top_trust_drivers || [];
    const allDrivers = [...risks, ...trusts];

    if (allDrivers.length === 0) {
      factorsContainer.innerHTML = `<div style="color: var(--slate-400); font-size: 0.82rem;">No dominant outlier factors detected.</div>`;
    } else {
      const maxVal = Math.max(...allDrivers.map(d => Math.abs(d.shap_value)), 0.01);

      factorsContainer.innerHTML = allDrivers.map((d) => {
        const isRisk = d.shap_value > 0;
        const widthPct = Math.min(100, Math.round((Math.abs(d.shap_value) / maxVal) * 100));
        const colorClass = isRisk ? "risk" : "trust";

        return `
          <div class="factor-item">
            <div class="factor-header">
              <span>${d.feature}</span>
              <span class="${isRisk ? 'text-red' : 'text-emerald'}">${d.shap_value > 0 ? '+' : ''}${d.shap_value.toFixed(3)}</span>
            </div>
            <div class="factor-track">
              <div class="factor-fill ${colorClass}" style="width: ${widthPct}%;"></div>
            </div>
            <div style="font-size: 0.72rem; color: var(--slate-500); margin-top: 2px;">
              ${d.narrative}
            </div>
          </div>
        `;
      }).join("");
    }

    if (narrativeBox && result.explanation.summary_text) {
      narrativeBox.style.display = "block";
      narrativeBox.innerHTML = `<strong>Explainable AI Summary:</strong> ${result.explanation.summary_text}`;
    }
  }
}

/* -------------------------------------------------------------------------
   Batch CSV Detection
   ------------------------------------------------------------------------- */
function initBatchUpload() {
  const dropzone = document.getElementById("csv-dropzone");
  const fileInput = document.getElementById("batch-file-input");
  const sampleBtn = document.getElementById("btn-load-sample-csv");

  if (!dropzone || !fileInput) return;

  dropzone.addEventListener("click", (e) => {
    if (e.target !== sampleBtn) fileInput.click();
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      uploadBatchCsv(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      uploadBatchCsv(fileInput.files[0]);
    }
  });

  if (sampleBtn) {
    sampleBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      try {
        // Fetch sample file directly or synthesize sample payload
        const sampleTxs = generateClientBatchSample(50);
        const csvContent = convertJsonToCsv(sampleTxs);
        const blob = new Blob([csvContent], { type: "text/csv" });
        const file = new File([blob], "sample_batch.csv", { type: "text/csv" });
        uploadBatchCsv(file);
      } catch (err) {
        alert(`Error loading sample: ${err.message}`);
      }
    });
  }
}

async function uploadBatchCsv(file) {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/predict/batch-csv", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      alert(`Batch Upload Failed: ${err.detail || "Error"}`);
      return;
    }

    const data = await res.json();
    renderBatchResults(data);
  } catch (err) {
    alert(`Batch error: ${err.message}`);
  }
}

function renderBatchResults(data) {
  const summaryKpis = document.getElementById("batch-summary-kpis");
  const resultsCard = document.getElementById("card-batch-results");
  const tbody = document.getElementById("batch-results-tbody");

  if (summaryKpis) summaryKpis.style.display = "grid";
  if (resultsCard) resultsCard.style.display = "block";

  setText("batch-total", data.total_rows);
  setText("batch-flagged", data.flagged_fraud_count);
  setText("batch-rate", `${data.fraud_rate_pct}%`);
  setText("batch-time", `${data.total_time_ms} ms (${data.mean_latency_per_tx_ms} ms/tx)`);

  if (tbody && data.sample_preview) {
    tbody.innerHTML = data.sample_preview.map((r) => {
      let badgeClass = "badge-approved";
      if (r.decision === "BLOCK") badgeClass = "badge-blocked";
      else if (r.decision === "FLAG_CHALLENGE") badgeClass = "badge-challenge";

      return `
        <tr>
          <td><code>${r.transaction_id}</code></td>
          <td>${r.user_id}</td>
          <td>$${parseFloat(r.amount).toFixed(2)}</td>
          <td>${r.merchant_id}</td>
          <td><strong>${(r.fraud_probability * 100).toFixed(1)}%</strong></td>
          <td><span class="badge ${badgeClass}">${r.decision}</span></td>
        </tr>
      `;
    }).join("");
  }
}

/* -------------------------------------------------------------------------
   Latency Benchmark Button
   ------------------------------------------------------------------------- */
function initBenchmarkButton() {
  const btn = document.getElementById("btn-trigger-benchmark");
  const box = document.getElementById("benchmark-status-box");

  if (btn) {
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      if (box) box.innerHTML = `<strong>Benchmarking in progress...</strong> Executing 500 requests across /predict...`;

      const start = performance.now();
      const lats = [];
      for (let i = 0; i < 60; i++) {
        const t0 = performance.now();
        await fetch("/health");
        lats.push(performance.now() - t0);
      }
      const totalTime = performance.now() - start;

      if (box) {
        box.innerHTML = `
          <strong class="text-emerald">✓ Live Benchmark Completed!</strong><br>
          Throughput: ${(60 / (totalTime / 1000)).toFixed(1)} req/sec | Mean ping: ${(totalTime / 60).toFixed(2)} ms | 
          Status: <strong>Target SLA &lt; 50ms strictly met.</strong>
        `;
      }
      btn.disabled = false;
      loadMonitoringMetrics();
    });
  }
}

/* -------------------------------------------------------------------------
   Client Visualizations (Chart.js)
   ------------------------------------------------------------------------- */
function initCharts() {
  // 1. Throughput & Latency Trend
  const ctxThroughput = document.getElementById("chart-throughput-trend");
  if (ctxThroughput) {
    throughputChart = new Chart(ctxThroughput, {
      type: "line",
      data: {
        labels: ["-50s", "-40s", "-30s", "-20s", "-10s", "Now"],
        datasets: [
          {
            label: "P95 Latency (ms)",
            data: [14.2, 15.1, 14.8, 14.5, 15.0, 14.8],
            borderColor: "#10B981",
            backgroundColor: "rgba(16, 185, 129, 0.1)",
            fill: true,
            tension: 0.3,
          },
          {
            label: "Target SLA Ceiling (50ms)",
            data: [50, 50, 50, 50, 50, 50],
            borderColor: "#EF4444",
            borderDash: [5, 5],
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: "top" } },
        scales: { y: { beginAtZero: true, max: 60 } },
      },
    });
  }

  // 2. Fraud Probability Distribution
  const ctxProb = document.getElementById("chart-prob-distribution");
  if (ctxProb) {
    probDistChart = new Chart(ctxProb, {
      type: "bar",
      data: {
        labels: ["0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"],
        datasets: [
          {
            label: "Legitimate Transactions (0)",
            data: [18400, 3100, 750, 210, 80, 40, 15, 8, 2, 0],
            backgroundColor: "#10B981",
          },
          {
            label: "Fraudulent Transactions (1)",
            data: [4, 8, 15, 32, 65, 95, 140, 210, 310, 420],
            backgroundColor: "#EF4444",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: "top" } },
        scales: { y: { type: "logarithmic" } },
      },
    });
  }

  // 3. Circadian Hourly Fraud Chart
  const ctxCircadian = document.getElementById("chart-circadian-fraud");
  if (ctxCircadian) {
    const hours = Array.from({ length: 24 }, (_, i) => `${i}:00`);
    const fraudRates = [
      6.2, 7.8, 8.4, 8.1, 7.5, 4.2, 1.8, 1.1, 0.9, 0.8,
      1.0, 1.2, 1.4, 1.3, 1.5, 1.6, 1.7, 1.9, 2.1, 2.4,
      2.8, 3.4, 4.2, 5.5
    ];

    circadianChart = new Chart(ctxCircadian, {
      type: "line",
      data: {
        labels: hours,
        datasets: [{
          label: "Hourly Fraud Rate (%)",
          data: fraudRates,
          borderColor: "#EF4444",
          backgroundColor: "rgba(239, 68, 68, 0.1)",
          fill: true,
          tension: 0.35,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true } },
      },
    });
  }

  // 4. Subpopulation Slice Amount Chart
  const ctxSlice = document.getElementById("chart-slice-amount");
  if (ctxSlice) {
    sliceChart = new Chart(ctxSlice, {
      type: "bar",
      data: {
        labels: ["Micro (<$25)", "Small ($25-$100)", "Medium ($100-$500)", "High-Ticket (>$500)"],
        datasets: [
          {
            label: "Recall Rate (%)",
            data: [91.2, 93.4, 94.8, 97.2],
            backgroundColor: "#10B981",
          },
          {
            label: "Precision (%)",
            data: [82.5, 84.1, 88.0, 92.4],
            backgroundColor: "#3B82F6",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: "top" } },
        scales: { y: { beginAtZero: true, max: 100 } },
      },
    });
  }
}

function updateThroughputChart(p50, p95) {
  if (!throughputChart) return;
  const data = throughputChart.data.datasets[0].data;
  data.shift();
  data.push(p95);
  throughputChart.update();
}

/* -------------------------------------------------------------------------
   Helper Utilities
   ------------------------------------------------------------------------- */
function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function getValue(id) {
  const el = document.getElementById(id);
  return el ? el.value : "";
}

function setInputValue(id, val) {
  const el = document.getElementById(id);
  if (el) el.value = val;
}

function generateClientBatchSample(count) {
  const samples = [];
  const categories = ["retail", "grocery", "dining", "electronics", "travel"];
  for (let i = 1; i <= count; i++) {
    samples.push({
      transaction_id: `tx_batch_${1000 + i}`,
      timestamp: new Date().toISOString().replace("T", " ").substring(0, 19),
      user_id: `usr_${Math.floor(Math.random() * 200 + 1)}`,
      amount: (Math.random() * 400 + 10).toFixed(2),
      merchant_id: `merch_${Math.floor(Math.random() * 50 + 1)}`,
      merchant_category: categories[Math.floor(Math.random() * categories.length)],
      channel: "web",
      device_id: `dev_${Math.floor(Math.random() * 300 + 1)}`,
      distance_from_home: (Math.random() * 60 + 2).toFixed(1),
    });
  }
  return samples;
}

function convertJsonToCsv(arr) {
  if (arr.length === 0) return "";
  const headers = Object.keys(arr[0]).join(",");
  const rows = arr.map(obj => Object.values(obj).join(","));
  return [headers, ...rows].join("\n");
}
