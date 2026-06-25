const state = {
  mode: "summary",
  job: null,
  pollTimer: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const metricNames = [
  "shapeError",
  "shapeVariability",
  "lengthError",
  "sizeError",
  "bendingError",
  "bendingVariability",
  "timeError",
  "timeVariability",
  "velocityError",
  "velocityVariability",
  "strokeError",
  "strokeOrderError",
  "dtwDistance",
  "ldtwDistance",
  "ddtwDistance",
  "wdtwDistance",
  "wddtwDistance",
];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function compactNumber(value) {
  if (value === null || value === undefined || value === "") return "-";
  const number = Number(value);
  if (Number.isNaN(number)) return escapeHtml(value);
  if (Math.abs(number) >= 1000) return number.toLocaleString(undefined, { maximumFractionDigits: 1 });
  return number.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function setStatus(status, text) {
  const dot = $("#statusDot");
  dot.className = "status-dot";
  if (status === "running" || status === "queued") dot.classList.add("running");
  else if (status === "completed") dot.classList.add("done");
  else if (status === "failed") dot.classList.add("failed");
  else dot.classList.add("idle");
  $("#statusText").textContent = text;
}

function selectedConfig() {
  const valueOrNull = (id) => {
    const value = $(id).value.trim();
    return value === "" ? null : Number(value);
  };
  return {
    mode: state.mode,
    summary: $("#summary").value,
    alignment: Number($("#alignment").value),
    popular: $("#popular").checked,
    strict: $("#strict").checked,
    rate: valueOrNull("#rate"),
    roundPrecision: Number($("#roundPrecision").value || 3),
    dtwWindow: valueOrNull("#dtwWindow"),
    exactDtw: $("#exactDtw").checked,
    groupBy: $("#groupBy").value,
  };
}

function setMode(mode) {
  state.mode = mode;
  $$(".seg").forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
  $$(".distribution-only").forEach((el) => el.classList.toggle("hidden", mode !== "distribution"));
  $$(".direct-only").forEach((el) => el.classList.toggle("hidden", mode !== "direct"));
}

function setActiveTab(tab) {
  $$(".tab").forEach((button) => button.classList.toggle("active", button.dataset.tab === tab));
  $$(".tab-pane").forEach((pane) => pane.classList.toggle("active", pane.id === tab));
}

function updateFileLabel(inputId, labelId) {
  const input = $(inputId);
  const label = $(labelId);
  const file = input.files[0];
  label.textContent = file ? `${file.name} (${Math.ceil(file.size / 1024)} KB)` : "No file selected";
}

async function runEvaluation() {
  const reference = $("#referenceZip").files[0];
  const candidate = $("#candidateZip").files[0];
  if (!reference || !candidate) {
    setStatus("failed", "Select two zip files");
    return;
  }

  $("#runButton").disabled = true;
  setStatus("queued", "Uploading");
  $("#progressBar").style.width = "4%";
  $("#issueList").innerHTML = '<p class="muted">Uploading and validating archives.</p>';

  const body = new FormData();
  body.append("reference_zip", reference);
  body.append("candidate_zip", candidate);
  body.append("config", JSON.stringify(selectedConfig()));

  try {
    const response = await fetch("/api/jobs", { method: "POST", body });
    if (!response.ok) throw new Error(`Upload failed (${response.status})`);
    const job = await response.json();
    renderJob(job);
    if (job.status === "queued" || job.status === "running") {
      pollJob(job.id);
    } else {
      $("#runButton").disabled = false;
    }
  } catch (error) {
    setStatus("failed", error.message);
    $("#issueList").innerHTML = `<div class="issue error"><strong>error</strong><span>${escapeHtml(error.message)}</span></div>`;
    $("#runButton").disabled = false;
  }
}

function pollJob(jobId) {
  clearTimeout(state.pollTimer);
  state.pollTimer = setTimeout(async () => {
    try {
      const response = await fetch(`/api/jobs/${jobId}`);
      const job = await response.json();
      renderJob(job);
      if (job.status === "queued" || job.status === "running") {
        pollJob(jobId);
      } else {
        $("#runButton").disabled = false;
      }
    } catch (error) {
      setStatus("failed", error.message);
      $("#runButton").disabled = false;
    }
  }, 650);
}

function renderJob(job) {
  state.job = job;
  $("#jobId").textContent = job.id ? `job ${job.id}` : "no run";
  $("#progressBar").style.width = `${job.progress || 0}%`;
  setStatus(job.status, job.phase || job.status);
  renderValidation(job.validation || {});

  if (job.result) {
    renderResults(job.result, job.id, job.config?.mode);
  } else if (job.error) {
    $("#overview").innerHTML = `<div class="empty-state">${escapeHtml(job.error)}</div>`;
  }
}

function renderValidation(validation) {
  const reference = validation.reference || {};
  const candidate = validation.candidate || {};
  const mode = validation.mode || {};
  const classCount = Object.keys(reference.classCounts || {}).length;
  const modeCount = mode.matchedPairCount ?? mode.validClassCount ?? mode.candidateCount ?? "-";

  $("#validationStats").innerHTML = `
    <div><span>Reference files</span><strong>${compactNumber(reference.fileCount)}</strong></div>
    <div><span>Candidate files</span><strong>${compactNumber(candidate.fileCount)}</strong></div>
    <div><span>Classes</span><strong>${compactNumber(classCount)}</strong></div>
    <div><span>Pairs / valid classes</span><strong>${compactNumber(modeCount)}</strong></div>
  `;

  const issues = validation.issues || [];
  if (!issues.length && validation.ok) {
    $("#issueList").innerHTML = '<div class="issue"><strong>ok</strong><span>Validation passed.</span></div>';
    return;
  }
  if (!issues.length) {
    $("#issueList").innerHTML = '<p class="muted">Upload two archives and run an evaluation.</p>';
    return;
  }
  $("#issueList").innerHTML = issues
    .slice(0, 8)
    .map((issue) => `
      <div class="issue ${escapeHtml(issue.severity)}">
        <strong>${escapeHtml(issue.severity)}</strong>
        <span>${escapeHtml(issue.scope)}: ${escapeHtml(issue.message)}${issue.path ? `<br><code>${escapeHtml(issue.path)}</code>` : ""}</span>
      </div>
    `)
    .join("");
}

function renderResults(result, jobId, mode) {
  renderOverview(result, mode);
  renderMetrics(result, mode);
  renderDistributions(result, mode);
  renderFiles(state.job.validation || {});
  renderOverlayControls(result, jobId);
  $("#jsonExport").href = `/api/jobs/${jobId}/exports/json`;
  $("#csvExport").href = `/api/jobs/${jobId}/exports/csv`;
  $("#jsonExport").classList.remove("disabled");
  $("#csvExport").classList.remove("disabled");
}

function renderOverview(result, mode) {
  const meta = result.metadata || {};
  const pairCount = meta.pairCount ?? "-";
  const validClassCount = meta.validClassCount ?? "-";
  const dtw = meta.exactDtw ? "exact" : (meta.dtwWindow === null || meta.dtwWindow === undefined ? "exact/auto" : `window ${meta.dtwWindow}`);
  const rows = mode === "distribution" ? result.results?.overall || [] : result.pairs || [];
  const shifted = topRows(rows, mode);
  $("#overview").innerHTML = `
    <div class="summary-grid">
      <div class="summary-card"><span>Mode</span><strong>${escapeHtml(meta.comparisonMode || mode)}</strong></div>
      <div class="summary-card"><span>Pairs</span><strong>${compactNumber(pairCount)}</strong></div>
      <div class="summary-card"><span>Valid classes</span><strong>${compactNumber(validClassCount)}</strong></div>
      <div class="summary-card"><span>DTW</span><strong>${escapeHtml(dtw)}</strong></div>
    </div>
    <h3 class="section-title">Highest movement deltas</h3>
    ${renderTopList(shifted, mode)}
  `;
}

function topRows(rows, mode) {
  if (mode === "distribution") {
    return rows
      .map((row) => ({
        label: row.gestureMetric,
        value: row.distributionMetrics?.wassersteinDistance ?? 0,
        detail: `candidate mean ${compactNumber(row.candidateStats?.mean)}`
      }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 6);
  }
  return rows
    .map((row) => ({
      label: row.pairKey,
      value: row.shapeError ?? 0,
      detail: `DTW ${compactNumber(row.dtwDistance)}`
    }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 6);
}

function renderTopList(rows) {
  if (!rows.length) return '<div class="empty-state">No completed metric rows yet.</div>';
  return `
    <div class="table-wrap">
      <table>
        <thead><tr><th>Item</th><th>Primary value</th><th>Detail</th></tr></thead>
        <tbody>
          ${rows.map((row) => `
            <tr>
              <td>${escapeHtml(row.label)}</td>
              <td class="metric-value">${compactNumber(row.value)}</td>
              <td>${escapeHtml(row.detail)}</td>
            </tr>`).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderMetrics(result, mode) {
  if (mode === "distribution") {
    const rows = result.results?.overall || [];
    $("#metrics").innerHTML = renderDistributionTable(rows);
    return;
  }
  const rows = result.pairs || [];
  const visibleMetrics = ["shapeError", "lengthError", "timeError", "velocityError", "strokeError", "dtwDistance", "ldtwDistance"];
  $("#metrics").innerHTML = `
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>Pair</th><th>Label</th>${visibleMetrics.map((name) => `<th>${name}</th>`).join("")}</tr>
        </thead>
        <tbody>
          ${rows.map((row) => `
            <tr>
              <td>${escapeHtml(row.pairKey)}</td>
              <td>${escapeHtml(row.label)}</td>
              ${visibleMetrics.map((name) => `<td class="metric-value">${compactNumber(row[name])}</td>`).join("")}
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderDistributionTable(rows) {
  return `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Metric</th><th>Baseline mean</th><th>Candidate mean</th><th>Wasserstein</th><th>Energy</th><th>KS stat</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map((row) => `
            <tr>
              <td>${escapeHtml(row.gestureMetric)}</td>
              <td class="metric-value">${compactNumber(row.baselineStats?.mean)}</td>
              <td class="metric-value">${compactNumber(row.candidateStats?.mean)}</td>
              <td class="metric-value">${compactNumber(row.distributionMetrics?.wassersteinDistance)}</td>
              <td class="metric-value">${compactNumber(row.distributionMetrics?.energyDistance)}</td>
              <td class="metric-value">${compactNumber(row.distributionMetrics?.ksStatistic)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderDistributions(result, mode) {
  if (mode !== "distribution") {
    const rows = result.pairs || [];
    const means = metricNames.slice(0, 8).map((name) => ({
      gestureMetric: name,
      baselineStats: { mean: 0 },
      candidateStats: { mean: average(rows.map((row) => row[name])) },
      distributionMetrics: { wassersteinDistance: average(rows.map((row) => row[name])) },
    }));
    $("#distributions").innerHTML = renderCharts(means);
    return;
  }
  $("#distributions").innerHTML = renderCharts((result.results?.overall || []).slice(0, 10));
}

function average(values) {
  const numbers = values.map(Number).filter((value) => Number.isFinite(value));
  if (!numbers.length) return 0;
  return numbers.reduce((a, b) => a + b, 0) / numbers.length;
}

function renderCharts(rows) {
  if (!rows.length) return '<div class="empty-state">No distribution rows available.</div>';
  return `
    <div class="chart-list">
      ${rows.map((row) => {
        const baseline = Number(row.baselineStats?.mean || 0);
        const candidate = Number(row.candidateStats?.mean || 0);
        const max = Math.max(baseline, candidate, 1);
        const baseWidth = Math.max(2, (baseline / max) * 270);
        const candWidth = Math.max(2, (candidate / max) * 270);
        return `
          <article class="chart-card">
            <header>
              <h3>${escapeHtml(row.gestureMetric)}</h3>
              <span class="muted">W ${compactNumber(row.distributionMetrics?.wassersteinDistance)}</span>
            </header>
            <svg class="mini-chart" viewBox="0 0 340 74" role="img" aria-label="${escapeHtml(row.gestureMetric)} mean comparison">
              <line x1="54" y1="19" x2="${54 + baseWidth}" y2="19" stroke="#126a73" stroke-width="12" stroke-linecap="round"></line>
              <line x1="54" y1="51" x2="${54 + candWidth}" y2="51" stroke="#b94722" stroke-width="12" stroke-linecap="round"></line>
              <text x="0" y="23" font-size="11" fill="#66706b">human</text>
              <text x="0" y="55" font-size="11" fill="#66706b">generated</text>
              <text x="${62 + baseWidth}" y="23" font-size="11" fill="#111514">${compactNumber(baseline)}</text>
              <text x="${62 + candWidth}" y="55" font-size="11" fill="#111514">${compactNumber(candidate)}</text>
            </svg>
          </article>
        `;
      }).join("")}
    </div>
  `;
}

function renderFiles(validation) {
  const reference = validation.reference || {};
  const candidate = validation.candidate || {};
  const refExamples = reference.examples || [];
  const candExamples = candidate.examples || [];
  $("#files").innerHTML = `
    <div class="summary-grid">
      <div class="summary-card"><span>Reference points</span><strong>${compactNumber(reference.pointCount)}</strong></div>
      <div class="summary-card"><span>Candidate points</span><strong>${compactNumber(candidate.pointCount)}</strong></div>
      <div class="summary-card"><span>Reference classes</span><strong>${compactNumber(Object.keys(reference.classCounts || {}).length)}</strong></div>
      <div class="summary-card"><span>Candidate classes</span><strong>${compactNumber(Object.keys(candidate.classCounts || {}).length)}</strong></div>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Reference examples</th><th>Candidate examples</th></tr></thead>
        <tbody>
          ${Array.from({ length: Math.max(refExamples.length, candExamples.length, 1) }).map((_, index) => `
            <tr>
              <td>${escapeHtml(refExamples[index] || "")}</td>
              <td>${escapeHtml(candExamples[index] || "")}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderOverlayControls(result, jobId) {
  const keys = result.metadata?.overlayKeys || [];
  const select = $("#overlaySelect");
  select.innerHTML = keys.map((key) => `<option value="${escapeHtml(key)}">${escapeHtml(key)}</option>`).join("");
  if (!keys.length) {
    $("#overlayFrame").innerHTML = '<div class="empty-state">No overlay keys available.</div>';
    return;
  }
  select.onchange = () => loadOverlay(jobId, select.value);
  loadOverlay(jobId, keys[0]);
}

function loadOverlay(jobId, key) {
  $("#overlayFrame").innerHTML = `<img alt="Trajectory overlay for ${escapeHtml(key)}" src="/api/jobs/${jobId}/overlay?key=${encodeURIComponent(key)}&t=${Date.now()}" />`;
}

function init() {
  $$(".seg").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
  $$(".tab").forEach((button) => button.addEventListener("click", () => setActiveTab(button.dataset.tab)));
  $("#referenceZip").addEventListener("change", () => updateFileLabel("#referenceZip", "#referenceName"));
  $("#candidateZip").addEventListener("change", () => updateFileLabel("#candidateZip", "#candidateName"));
  $("#runButton").addEventListener("click", runEvaluation);
  setMode("summary");
}

init();
