import * as echarts from "echarts";
import { useEffect, useMemo, useRef, useState } from "react";

const API = "/api";
const DEFAULT_METRICS = ["shapeError", "velocityError", "dtwDistance"];
const METRIC_FAMILIES = {
  core: ["shapeError", "lengthError", "bendingError", "velocityError", "dtwDistance", "curvature"],
  shape: ["shapeError", "shapeVariability", "lengthError", "sizeError", "bendingError", "bendingVariability"],
  timing: ["timeError", "timeVariability", "velocityError", "velocityVariability", "meanStrokeDuration"],
  movement: ["cornerSlowdown", "twoThirdsPowerLawR2", "highFrequencyRatio", "curvature"],
  stroke: ["strokeError", "strokeOrderError", "strokeLengthStd"],
  dtw: ["dtwDistance", "ldtwDistance", "ddtwDistance", "wdtwDistance", "wddtwDistance"],
};

const COLORS = ["#166b6b", "#c53d32", "#222222", "#6e5b2e", "#5f6f52", "#7c4d5f"];

function formatNumber(value, digits = 3) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  if (Math.abs(number) >= 1000) return number.toLocaleString(undefined, { maximumFractionDigits: 1 });
  return number.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function compactMetric(metric) {
  return metric.replace(/([a-z])([A-Z])/g, "$1 $2").replace(/Distance/g, "dist.");
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

async function fetchJsonWithBody(url, body) {
  const response = await fetch(url, { method: "POST", body });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function paramsFor(filters) {
  const params = new URLSearchParams();
  if (filters.source) {
    const [source, variant] = filters.source.split("/");
    params.set("source", source);
    if (variant && !filters.variant) params.set("variant", variant);
  }
  if (filters.dataset) params.set("dataset", filters.dataset);
  if (filters.variant) params.set("variant", filters.variant);
  if (filters.classKey) params.set("class", filters.classKey);
  if (filters.metrics?.length) params.set("metrics", filters.metrics.join(","));
  if (filters.metric) params.set("metric", filters.metric);
  if (filters.metricFamily) params.set("metric_family", filters.metricFamily);
  if (filters.distributionMetric) params.set("distribution_metric", filters.distributionMetric);
  return params;
}

function EChart({ option, onPointClick }) {
  const ref = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!ref.current) return undefined;
    const chart = echarts.init(ref.current, null, { renderer: "svg" });
    chartRef.current = chart;
    const resizeObserver = new ResizeObserver(() => chart.resize());
    resizeObserver.observe(ref.current);
    return () => {
      resizeObserver.disconnect();
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!chartRef.current) return;
    chartRef.current.setOption(option, true);
  }, [option]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || !onPointClick) return undefined;
    const handler = (params) => onPointClick(params.data || params);
    chart.on("click", handler);
    return () => chart.off("click", handler);
  }, [onPointClick]);

  return <div className="chart-canvas" ref={ref} />;
}

function useChart(reportId, kind, filters) {
  const [payload, setPayload] = useState(null);
  const [error, setError] = useState("");
  const queryKey = JSON.stringify(filters);

  useEffect(() => {
    if (!reportId) return undefined;
    let alive = true;
    setError("");
    const query = paramsFor(filters);
    fetchJson(`${API}/reports/${reportId}/chart/${kind}?${query}`)
      .then((data) => alive && setPayload(data))
      .catch((err) => alive && setError(err.message));
    return () => {
      alive = false;
    };
  }, [reportId, kind, queryKey]);

  return { payload, error };
}

function baseChart() {
  return {
    backgroundColor: "transparent",
    textStyle: { color: "#1e211f", fontFamily: "Inter, system-ui, sans-serif" },
    animationDuration: 350,
  };
}

function rankingOption(payload) {
  const rows = payload?.rows || [];
  return {
    ...baseChart(),
    grid: { left: 110, right: 24, top: 18, bottom: 44 },
    tooltip: {
      trigger: "axis",
      valueFormatter: (value) => formatNumber(value),
    },
    xAxis: {
      type: "value",
      name: payload?.metric || "distribution score",
      nameLocation: "middle",
      nameGap: 32,
      splitLine: { lineStyle: { color: "#dddddc" } },
      axisLabel: { color: "#555" },
    },
    yAxis: {
      type: "category",
      inverse: true,
      data: rows.map((row) => row.source),
      axisLabel: { color: "#222", width: 98, overflow: "truncate" },
    },
    series: [
      {
        type: "bar",
        data: rows.map((row, index) => ({
          ...row,
          value: row.score,
          itemStyle: { color: COLORS[index % COLORS.length] },
        })),
        barMaxWidth: 22,
        label: {
          show: true,
          position: "right",
          formatter: (params) => formatNumber(params.value, 3),
          color: "#333",
        },
      },
    ],
  };
}

function heatmapOption(payload) {
  const sources = payload?.sources || [];
  const metrics = payload?.metrics || [];
  const min = Number.isFinite(payload?.min) ? payload.min : 0;
  const max = Number.isFinite(payload?.max) && payload.max > min ? payload.max : min + 1;
  return {
    ...baseChart(),
    tooltip: {
      formatter: (params) => {
        const data = params.data;
        return `${data.source}<br/>${data.metric}<br/>${payload?.metric || "score"} ${formatNumber(data.raw)}<br/>rows ${data.rowCount}`;
      },
    },
    grid: { left: 106, right: 24, top: 44, bottom: 84 },
    xAxis: { type: "category", data: metrics, axisLabel: { rotate: 32, color: "#333" } },
    yAxis: { type: "category", data: sources, axisLabel: { color: "#222", width: 96, overflow: "truncate" } },
    visualMap: {
      min,
      max,
      calculable: true,
      orient: "horizontal",
      left: "center",
      bottom: 6,
      text: ["higher", "lower"],
      inRange: { color: ["#f8faf8", "#d8ece8", "#7cb7ae", "#c87460", "#aa352e"] },
    },
    series: [
      {
        type: "heatmap",
        data: (payload?.values || [])
          .filter((item) => item.raw !== null && item.raw !== undefined)
          .map((item) => ({ ...item, value: [item.metricIndex, item.sourceIndex, item.value] })),
        label: {
          show: true,
          formatter: (params) => formatNumber(params.data.raw, 2),
          color: "#1f1f1f",
          fontSize: 10,
        },
        emphasis: { itemStyle: { borderColor: "#111", borderWidth: 1 } },
      },
    ],
  };
}

function scatterOption(payload) {
  const points = payload?.points || [];
  const sources = [...new Set(points.map((point) => point.source))];
  const maxValue = Math.max(1, ...points.flatMap((point) => [point.humanMean || 0, point.generatedMean || 0]));
  return {
    ...baseChart(),
    tooltip: {
      formatter: (params) => {
        const data = params.data;
        return `${data.source}<br/>${data.dataset} / ${data.classKey}<br/>${data.metric}<br/>human ${formatNumber(data.humanMean)}<br/>candidate ${formatNumber(data.generatedMean)}<br/>ratio ${formatNumber(data.ratio, 2)}`;
      },
    },
    legend: { top: 4, type: "scroll", textStyle: { color: "#333" } },
    grid: { left: 64, right: 24, top: 56, bottom: 56 },
    xAxis: {
      name: "human summary mean",
      nameLocation: "middle",
      nameGap: 34,
      min: 0,
      splitLine: { lineStyle: { color: "#dddddc" } },
    },
    yAxis: {
      name: "candidate mean",
      nameLocation: "middle",
      nameGap: 42,
      min: 0,
      splitLine: { lineStyle: { color: "#dddddc" } },
    },
    series: [
      ...sources.map((source, index) => ({
        type: "scatter",
        name: source,
        symbolSize: (value, data) => Math.max(5, Math.min(16, Math.sqrt(data?.n || 1) * 1.4)),
        itemStyle: { color: COLORS[index % COLORS.length], opacity: 0.72 },
        data: points
          .filter((point) => point.source === source)
          .map((point) => ({ ...point, value: [point.humanMean, point.generatedMean] })),
      })),
      {
        type: "line",
        name: "equal",
        symbol: "none",
        lineStyle: { color: "#222", type: "dashed", width: 1 },
        tooltip: { show: false },
        data: [
          [0, 0],
          [maxValue, maxValue],
        ],
      },
    ],
  };
}

function pairwiseOption(payload) {
  const rows = payload?.rows || [];
  const metrics = [...new Set(rows.map((row) => row.metric))];
  const sources = [...new Set(rows.map((row) => row.source))];
  return {
    ...baseChart(),
    tooltip: {
      trigger: "axis",
      formatter: (items) => items.map((item) => `${item.seriesName}: ${formatNumber(item.value, 2)}x`).join("<br/>"),
    },
    legend: { top: 4, type: "scroll" },
    grid: { left: 60, right: 18, top: 58, bottom: 82 },
    xAxis: { type: "category", data: metrics.map(compactMetric), axisLabel: { rotate: 32 } },
    yAxis: {
      type: "value",
      name: "candidate / human",
      splitLine: { lineStyle: { color: "#dddddc" } },
    },
    series: sources.map((source, index) => ({
      type: "bar",
      name: source,
      itemStyle: { color: COLORS[index % COLORS.length], opacity: 0.82 },
      data: metrics.map((metric) => rows.find((row) => row.source === source && row.metric === metric)?.ratio ?? null),
      barMaxWidth: 18,
    })),
  };
}

function distributionOption(payload) {
  const rows = payload?.rows || [];
  const visibleRows = rows.slice(0, 36);
  const labels = visibleRows.map((row) => `${row.source} / ${compactMetric(row.metric)}`);
  return {
    ...baseChart(),
    tooltip: { trigger: "axis", valueFormatter: (value) => formatNumber(value) },
    legend: { top: 4 },
    grid: { left: 64, right: 22, top: 58, bottom: 104 },
    xAxis: { type: "category", data: labels, axisLabel: { rotate: 45, fontSize: 10 } },
    yAxis: { type: "value", name: "mean metric value", splitLine: { lineStyle: { color: "#dddddc" } } },
    series: [
      {
        name: "within reference",
        type: "bar",
        data: visibleRows.map((row) => row.withinReferenceMean),
        itemStyle: { color: "#222", opacity: 0.72 },
        barMaxWidth: 14,
      },
      {
        name: "within comparison",
        type: "bar",
        data: visibleRows.map((row) => row.withinComparisonMean),
        itemStyle: { color: "#166b6b", opacity: 0.82 },
        barMaxWidth: 14,
      },
      {
        name: "between groups",
        type: "bar",
        data: visibleRows.map((row) => row.betweenGroupsMean),
        itemStyle: { color: "#c53d32", opacity: 0.82 },
        barMaxWidth: 14,
      },
    ],
  };
}

function histogramOption(payload) {
  const series = payload?.series || [];
  const bins = series[0]?.bins || [];
  const labels = bins.map((bin) => `${formatNumber(bin.x0, 1)}-${formatNumber(bin.x1, 1)}`);
  return {
    ...baseChart(),
    tooltip: {
      trigger: "axis",
      formatter: (items) => items.map((item) => `${item.seriesName}: ${item.value}`).join("<br/>"),
    },
    legend: { top: 4 },
    grid: { left: 56, right: 18, top: 54, bottom: 82 },
    xAxis: { type: "category", data: labels, axisLabel: { rotate: 45, fontSize: 10 } },
    yAxis: { type: "value", name: "pair count", splitLine: { lineStyle: { color: "#dddddc" } } },
    series: series.map((item, index) => ({
      type: "bar",
      name: `${item.name} (n=${item.n})`,
      data: item.bins.map((bin) => bin.count),
      itemStyle: { color: COLORS[index % COLORS.length], opacity: 0.72 },
      barGap: "-30%",
    })),
  };
}

function ChartPanel({ title, kicker, option, children, onPointClick, exportHref, summary }) {
  return (
    <section className="panel chart-panel">
      <div className="panel-title">
        <div>
          <span>{kicker}</span>
          <h2>{title}</h2>
        </div>
        <details className="chart-actions">
          <summary aria-label={`${title} actions`}>...</summary>
          <div>
            {exportHref && <a href={exportHref}>Open chart JSON</a>}
            {summary && <p>{summary}</p>}
          </div>
        </details>
      </div>
      {children || <EChart option={option} onPointClick={onPointClick} />}
    </section>
  );
}

function Select({ label, value, options, onChange, allLabel = "All" }) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="">{allLabel}</option>
        {options.map((option) => (
          <option value={option} key={option}>{option}</option>
        ))}
      </select>
    </label>
  );
}

function Toggle({ label, checked, onChange }) {
  return (
    <button type="button" className={checked ? "toggle is-on" : "toggle"} aria-pressed={checked} onClick={() => onChange(!checked)}>
      <span>{label}</span>
    </button>
  );
}

function MetricPicker({ metrics, selected, family, onFamily, onToggle }) {
  return (
    <section className="metric-picker">
      <label className="field">
        <span>Metric family</span>
        <select value={family} onChange={(event) => onFamily(event.target.value)}>
          {Object.keys(METRIC_FAMILIES).map((name) => <option key={name} value={name}>{name}</option>)}
          <option value="all">all</option>
        </select>
      </label>
      <div className="metric-chip-grid">
        {metrics.map((metric) => (
          <button
            type="button"
            className={selected.includes(metric) ? "metric-chip selected" : "metric-chip"}
            key={metric}
            onClick={() => onToggle(metric)}
          >
            {metric}
          </button>
        ))}
      </div>
    </section>
  );
}

function UploadGroups({ groups, setGroups }) {
  function update(index, patch) {
    setGroups(groups.map((group, itemIndex) => (itemIndex === index ? { ...group, ...patch } : group)));
  }
  return (
    <div className="upload-groups">
      {groups.map((group, index) => (
        <div className="zip-row" key={group.id}>
          <label className="zip-name">
            <span>Comparison label</span>
            <input
              aria-label="Comparison group label"
              value={group.name}
              onChange={(event) => update(index, { name: event.target.value })}
            />
          </label>
          <label className="zip-file">
            <input
              type="file"
              accept=".zip,application/zip"
              onChange={(event) => update(index, { file: event.target.files?.[0] || null })}
            />
            <b>{group.file ? group.file.name : "choose zip"}</b>
          </label>
        </div>
      ))}
      <button
        type="button"
        className="ghost"
        onClick={() => setGroups([...groups, { id: crypto.randomUUID(), name: `comparison-${groups.length + 1}`, file: null }])}
      >
        Add comparison zip
      </button>
    </div>
  );
}

function DataTable({ title, rows, columns }) {
  return (
    <section className="panel data-panel">
      <div className="panel-title">
        <div>
          <span>Exact values</span>
          <h2>{title}</h2>
        </div>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>{columns.map((column) => <th key={column.key}>{column.label}</th>)}</tr>
          </thead>
          <tbody>
            {rows.slice(0, 18).map((row, index) => (
              <tr key={`${row.source}-${row.metric}-${index}`}>
                {columns.map((column) => <td key={column.key}>{column.format ? column.format(row[column.key]) : row[column.key]}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function JobPanel({ job, result }) {
  if (!job) return null;
  const pairs = result?.pairs || [];
  const groups = result?.metadata?.comparisonGroups || [];
  return (
    <section className="job-panel">
      <h3>Ad-hoc evaluation</h3>
      <dl>
        <dt>Status</dt><dd>{job.phase || job.status}</dd>
        <dt>Progress</dt><dd>{job.progress || 0}%</dd>
        <dt>Groups</dt><dd>{groups.length ? groups.join(", ") : "-"}</dd>
        <dt>Pairs</dt><dd>{pairs.length || result?.metadata?.pairCount || 0}</dd>
      </dl>
      {pairs.length > 0 && (
        <div className="job-preview">
          {pairs.slice(0, 5).map((pair) => (
            <div key={pair.pairKey}>
              <strong>{pair.comparisonGroup || "candidate"}</strong>
              <span>{pair.pairKey}</span>
              <em>shape {formatNumber(pair.shapeError)} / dtw {formatNumber(pair.dtwDistance)}</em>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

export default function App() {
  const [reports, setReports] = useState([]);
  const [selectedReportId, setSelectedReportId] = useState("");
  const [summary, setSummary] = useState(null);
  const [filterOptions, setFilterOptions] = useState({ sources: [], datasets: [], variants: [], classes: [], classByDataset: {}, metrics: [], distributionMetrics: [], metricFamilies: [] });
  const [analysis, setAnalysis] = useState({
    source: "",
    dataset: "",
    classKey: "",
    metricFamily: "core",
    metrics: DEFAULT_METRICS,
    distributionMetric: "normalizedWassersteinDistance",
  });
  const [overlay, setOverlay] = useState({
    source: "",
    dataset: "",
    classKey: "",
    showReference: true,
    showComparison: true,
    showSummary: true,
    summarySource: "reference",
    sampleCount: 18,
  });
  const [status, setStatus] = useState("Loading cache");
  const [referenceZip, setReferenceZip] = useState(null);
  const [comparisonGroups, setComparisonGroups] = useState([{ id: "first", name: "candidate", file: null }]);
  const [overlaySvg, setOverlaySvg] = useState("");
  const [overlayLoading, setOverlayLoading] = useState(false);
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [job, setJob] = useState(null);
  const [jobResult, setJobResult] = useState(null);

  useEffect(() => {
    let alive = true;
    fetchJson(`${API}/reports`)
      .then((data) => {
        if (!alive) return;
        setReports(data.reports || []);
        setSelectedReportId(data.reports?.[0]?.id || "");
        setStatus(data.reports?.length ? "Cache ready" : "No cached reports");
      })
      .catch((error) => alive && setStatus(error.message));
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (!selectedReportId) return undefined;
    let alive = true;
    Promise.all([
      fetchJson(`${API}/reports/${selectedReportId}/summary`),
      fetchJson(`${API}/reports/${selectedReportId}/filters`),
    ])
      .then(([summaryPayload, filtersPayload]) => {
        if (!alive) return;
        setSummary(summaryPayload);
        setFilterOptions(filtersPayload);
        const datasets = filtersPayload.datasets || [];
        const sources = filtersPayload.sources || [];
        const metrics = filtersPayload.metrics || [];
        const dataset = datasets[0] || "";
        const classKey = filtersPayload.classByDataset?.[dataset]?.[0] || filtersPayload.classes?.[0] || "";
        const defaults = DEFAULT_METRICS.filter((metric) => metrics.includes(metric));
        const selectedMetrics = defaults.length ? defaults : metrics.slice(0, 3);
        setAnalysis({
          source: "",
          dataset,
          classKey: "",
          metricFamily: "core",
          metrics: selectedMetrics,
          distributionMetric: "normalizedWassersteinDistance",
        });
        setOverlay((current) => ({
          ...current,
          source: sources.includes(current.source) ? current.source : sources[0] || "",
          dataset: datasets.includes(current.dataset) ? current.dataset : dataset,
          classKey: (filtersPayload.classByDataset?.[current.dataset] || []).includes(current.classKey) ? current.classKey : classKey,
        }));
      })
      .catch((error) => alive && setStatus(error.message));
    return () => {
      alive = false;
    };
  }, [selectedReportId]);

  const chartFilters = useMemo(() => ({
    source: analysis.source,
    dataset: analysis.dataset,
    classKey: analysis.classKey,
    metricFamily: analysis.metricFamily,
    metrics: analysis.metrics,
    distributionMetric: analysis.distributionMetric,
  }), [analysis]);
  const histogramFilters = useMemo(() => ({
    source: overlay.source,
    dataset: overlay.dataset,
    classKey: overlay.classKey,
    metrics: [analysis.metrics[0] || "shapeError"],
  }), [overlay.source, overlay.dataset, overlay.classKey, analysis.metrics]);

  const ranking = useChart(selectedReportId, "ranking", chartFilters);
  const heatmap = useChart(selectedReportId, "heatmap", chartFilters);
  const scatter = useChart(selectedReportId, "scatter", chartFilters);
  const pairwise = useChart(selectedReportId, "pairwise", chartFilters);
  const distribution = useChart(selectedReportId, "distribution", chartFilters);
  const histogram = useChart(selectedReportId, "histogram", histogramFilters);

  useEffect(() => {
    if (!selectedReportId || !overlay.source || !overlay.dataset || !overlay.classKey) return undefined;
    let alive = true;
    setOverlayLoading(true);
    const query = new URLSearchParams({
      source: overlay.source,
      dataset: overlay.dataset,
      class: overlay.classKey,
      sample_count: String(overlay.sampleCount),
      summary: summary?.summary || "medoid",
      show_reference: String(overlay.showReference),
      show_comparison: String(overlay.showComparison),
      show_summary: String(overlay.showSummary),
      summary_source: overlay.summarySource,
    });
    fetch(`${API}/reports/${selectedReportId}/overlay?${query}`)
      .then((response) => (response.ok ? response.text() : ""))
      .then((svg) => {
        if (!alive) return;
        setOverlaySvg(svg);
        setOverlayLoading(false);
      })
      .catch(() => {
        if (!alive) return;
        setOverlaySvg("");
        setOverlayLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [selectedReportId, overlay, summary?.summary]);

  useEffect(() => {
    if (!job || !["queued", "running"].includes(job.status)) return undefined;
    const timer = setInterval(async () => {
      try {
        const next = await fetchJson(`${API}/jobs/${job.id}`);
        setJob(next);
        setStatus(next.phase || next.status);
        if (next.status === "completed") {
          const result = await fetchJson(`${API}/jobs/${job.id}/results`);
          setJobResult(result);
        }
      } catch (error) {
        setStatus(error.message);
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [job]);

  const selectedReport = reports.find((report) => report.id === selectedReportId);
  const sourceCount = summary?.sources?.length || selectedReport?.sources?.length || 0;
  const datasetCount = summary?.datasets?.length || selectedReport?.datasets?.length || 0;
  const classCount = summary?.classCount || selectedReport?.classCount || 0;
  const warningCount = summary?.warningCount || selectedReport?.warningCount || 0;
  const classesForAnalysisDataset = filterOptions.classByDataset?.[analysis.dataset] || filterOptions.classes || [];
  const classesForOverlayDataset = filterOptions.classByDataset?.[overlay.dataset] || filterOptions.classes || [];
  const distributionMetricOptions = ["normalizedWassersteinDistance", ...(filterOptions.distributionMetrics || [])];
  const cacheLabel = selectedReport ? `${selectedReport.name} (${formatNumber((selectedReport.sizeBytes || 0) / 1024 / 1024, 1)} MB)` : "no cache selected";

  function updateAnalysis(patch) {
    setAnalysis((current) => ({ ...current, ...patch }));
  }

  function updateOverlay(patch) {
    setOverlay((current) => ({ ...current, ...patch }));
  }

  function setAnalysisDataset(dataset) {
    updateAnalysis({ dataset, classKey: "" });
  }

  function setOverlayDataset(dataset) {
    const nextClasses = filterOptions.classByDataset?.[dataset] || filterOptions.classes || [];
    updateOverlay({ dataset, classKey: nextClasses[0] || "" });
  }

  function setMetricFamily(metricFamily) {
    const allMetrics = filterOptions.metrics || [];
    const selected = metricFamily === "all" ? allMetrics : (METRIC_FAMILIES[metricFamily] || []).filter((metric) => allMetrics.includes(metric));
    updateAnalysis({ metricFamily, metrics: selected.length ? selected : allMetrics.slice(0, 3) });
  }

  function toggleMetric(metric) {
    const next = analysis.metrics.includes(metric)
      ? analysis.metrics.filter((item) => item !== metric)
      : [...analysis.metrics, metric];
    updateAnalysis({ metrics: next.length ? next : [metric] });
  }

  function handlePoint(point) {
    if (!point?.source) return;
    setSelectedPoint(point);
    updateAnalysis({
      source: point.source || analysis.source,
      dataset: point.dataset || analysis.dataset,
      classKey: point.classKey || analysis.classKey,
      metrics: point.metric ? [point.metric] : analysis.metrics,
    });
    updateOverlay({
      source: point.source || overlay.source,
      dataset: point.dataset || overlay.dataset,
      classKey: point.classKey || overlay.classKey,
    });
  }

  async function runEvaluation() {
    const groupsWithFiles = comparisonGroups.filter((group) => group.file);
    if (!referenceZip && !groupsWithFiles.length && selectedReport) {
      setStatus("Loaded cached report");
      return;
    }
    if (!referenceZip || !groupsWithFiles.length) {
      setStatus("Upload reference and at least one comparison zip, or choose a cached report.");
      return;
    }
    const body = new FormData();
    body.append("reference_zip", referenceZip);
    body.append("comparison_names", JSON.stringify(groupsWithFiles.map((group) => group.name)));
    groupsWithFiles.forEach((group) => body.append("comparison_zips", group.file));
    body.append(
      "config",
      JSON.stringify({
        mode: "summary",
        summary: summary?.summary || "medoid",
        rate: summary?.rate || 24,
        alignment: summary?.alignment || 0,
        strict: true,
      }),
    );
    setStatus("Uploading");
    setJob(null);
    setJobResult(null);
    try {
      const nextJob = await fetchJsonWithBody(`${API}/jobs`, body);
      setJob(nextJob);
      setStatus(nextJob.phase || nextJob.status);
    } catch (error) {
      setStatus(error.message);
    }
  }

  const query = paramsFor(chartFilters);
  const histogramQuery = paramsFor(histogramFilters);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-mark"><span></span><h1>RELACC WORKBENCH</h1></div>
        <div className="status-block"><span>Run status</span><strong>{status}</strong></div>
        <div className="status-block"><span>Cache status</span><strong>{selectedReport ? "Ready" : "Empty"}</strong></div>
        <label className="report-picker">
          <span>Selected report folder</span>
          <select value={selectedReportId} onChange={(event) => setSelectedReportId(event.target.value)}>
            {reports.map((report) => <option value={report.id} key={report.id}>{report.name}</option>)}
          </select>
        </label>
        <button className="evaluate" type="button" onClick={runEvaluation}>
          {referenceZip || comparisonGroups.some((group) => group.file) ? "Evaluate Zips" : "Load Cache"}
        </button>
      </header>

      <main className="workbench-grid">
        <aside className="source-rail">
          <h2>Inputs</h2>
          <label className="field">
            <span>Report cache</span>
            <select value={selectedReportId} onChange={(event) => setSelectedReportId(event.target.value)}>
              {reports.map((report) => <option value={report.id} key={report.id}>{report.name}</option>)}
            </select>
            <small>{cacheLabel}</small>
          </label>

          <section className="rail-section">
            <h3>Reference zip</h3>
            <label className="dropzone">
              <input type="file" accept=".zip,application/zip" onChange={(event) => setReferenceZip(event.target.files?.[0] || null)} />
              <strong>{referenceZip ? referenceZip.name : "Drop reference zip here"}</strong>
              <span>Uploads are evaluated only when files are selected. Otherwise cached reports are loaded.</span>
            </label>
          </section>

          <section className="rail-section">
            <h3>Comparison zip groups</h3>
            <p className="helper-copy">The label names each candidate group in multi-zip evaluation results.</p>
            <UploadGroups groups={comparisonGroups} setGroups={setComparisonGroups} />
          </section>

          <JobPanel job={job} result={jobResult} />
        </aside>

        <section className="main-stage">
          <section className="kpi-strip">
            <div><span>Cached</span><strong>{selectedReport ? "Yes" : "No"}</strong><em>{summary?.summary || "summary"} summary</em></div>
            <div><span>Sources</span><strong>{sourceCount}</strong><em>candidate generators</em></div>
            <div><span>Datasets</span><strong>{datasetCount}</strong><em>{summary?.rate ? `rate ${summary.rate}` : "rate"}</em></div>
            <div><span>Classes</span><strong>{formatNumber(classCount, 0)}</strong><em>gesture classes</em></div>
            <div><span>Warnings</span><strong className={warningCount ? "danger" : ""}>{warningCount}</strong><em>planning warnings</em></div>
          </section>

          <section className="analysis-controls">
            <div className="control-row">
              <Select label="Dataset" value={analysis.dataset} options={filterOptions.datasets || []} onChange={setAnalysisDataset} />
              <Select label="Source" value={analysis.source} options={filterOptions.sources || []} onChange={(source) => updateAnalysis({ source })} />
              <Select label="Class" value={analysis.classKey} options={classesForAnalysisDataset} onChange={(classKey) => updateAnalysis({ classKey })} />
              <label className="field">
                <span>Distribution score</span>
                <select value={analysis.distributionMetric} onChange={(event) => updateAnalysis({ distributionMetric: event.target.value })}>
                  {[...new Set(distributionMetricOptions)].map((metric) => <option value={metric} key={metric}>{metric}</option>)}
                </select>
              </label>
            </div>
            <MetricPicker
              metrics={filterOptions.metrics || []}
              selected={analysis.metrics}
              family={analysis.metricFamily}
              onFamily={setMetricFamily}
              onToggle={toggleMetric}
            />
          </section>

          <div className="chart-grid">
            <ChartPanel
              title="Generator Ranking"
              kicker="Lower distribution distance is better"
              option={rankingOption(ranking.payload)}
              exportHref={selectedReportId ? `${API}/reports/${selectedReportId}/chart/ranking?${query}` : "#"}
              summary={`${ranking.payload?.rows?.length || 0} sources ranked across ${analysis.metrics.length} selected metrics.`}
              onPointClick={handlePoint}
            />
            <ChartPanel
              title="Metric Heatmap"
              kicker="Source by selected gesture metric"
              option={heatmapOption(heatmap.payload)}
              exportHref={selectedReportId ? `${API}/reports/${selectedReportId}/chart/heatmap?${query}` : "#"}
              summary="Cell values are aggregated distribution scores from the cached report."
              onPointClick={handlePoint}
            />
            <ChartPanel
              title="Candidate vs Human Summary"
              kicker="Pairwise metric means against the human/reference summary"
              option={pairwiseOption(pairwise.payload)}
              exportHref={selectedReportId ? `${API}/reports/${selectedReportId}/chart/pairwise?${query}` : "#"}
              summary="A value near 1.0 means the candidate mean is close to the human baseline mean."
            />
            <ChartPanel
              title="Class Scatter"
              kicker={`Human mean vs candidate mean for ${scatter.payload?.metric || analysis.metrics[0] || "metric"}`}
              option={scatterOption(scatter.payload)}
              exportHref={selectedReportId ? `${API}/reports/${selectedReportId}/chart/scatter?${query}` : "#"}
              summary={`${scatter.payload?.points?.length || 0} class-level points. Click a point to inspect its gesture overlay.`}
              onPointClick={handlePoint}
            />
            <ChartPanel
              title="Within and Between Distributions"
              kicker="Within-reference, within-comparison, and between-groups means"
              option={distributionOption(distribution.payload)}
              exportHref={selectedReportId ? `${API}/reports/${selectedReportId}/chart/distribution?${query}` : "#"}
              summary="Bars use distribution.csv summary fields from the cached report."
            />
            <ChartPanel
              title="Selected Class Histogram"
              kicker="Raw pair values for the inspector class"
              option={histogramOption(histogram.payload)}
              exportHref={selectedReportId ? `${API}/reports/${selectedReportId}/chart/histogram?${histogramQuery}` : "#"}
              summary="Histogram reads raw within-reference, within-comparison, and between-groups pair files for the selected class."
            />
          </div>

          <div className="table-grid">
            <DataTable
              title="Pairwise Metric Scores"
              rows={pairwise.payload?.rows || []}
              columns={[
                { key: "source", label: "Source" },
                { key: "metric", label: "Metric" },
                { key: "humanMean", label: "Human", format: formatNumber },
                { key: "candidateMean", label: "Candidate", format: formatNumber },
                { key: "ratio", label: "Ratio", format: (value) => `${formatNumber(value, 2)}x` },
                { key: "n", label: "N", format: (value) => formatNumber(value, 0) },
              ]}
            />
            <DataTable
              title="Distribution Metrics"
              rows={distribution.payload?.rows || []}
              columns={[
                { key: "source", label: "Source" },
                { key: "metric", label: "Metric" },
                { key: "withinReferenceMean", label: "Within ref", format: formatNumber },
                { key: "withinComparisonMean", label: "Within comp", format: formatNumber },
                { key: "betweenGroupsMean", label: "Between", format: formatNumber },
                { key: "normalizedWassersteinDistance", label: "Norm W", format: formatNumber },
                { key: "ratio", label: "Var ratio", format: (value) => `${formatNumber(value, 2)}x` },
              ]}
            />
          </div>
        </section>

        <aside className="inspector">
          <h2>Canvas Inspector</h2>
          <Select label="Overlay source" value={overlay.source} options={filterOptions.sources || []} onChange={(source) => updateOverlay({ source })} />
          <Select label="Overlay dataset" value={overlay.dataset} options={filterOptions.datasets || []} onChange={setOverlayDataset} />
          <Select label="Overlay class" value={overlay.classKey} options={classesForOverlayDataset} onChange={(classKey) => updateOverlay({ classKey })} />

          <section className="overlay-controls">
            <Toggle label="Reference samples" checked={overlay.showReference} onChange={(showReference) => updateOverlay({ showReference })} />
            <Toggle label="Comparison samples" checked={overlay.showComparison} onChange={(showComparison) => updateOverlay({ showComparison })} />
            <Toggle label="Summary line" checked={overlay.showSummary} onChange={(showSummary) => updateOverlay({ showSummary })} />
            <label className="field">
              <span>Summary source</span>
              <select value={overlay.summarySource} onChange={(event) => updateOverlay({ summarySource: event.target.value })}>
                <option value="reference">reference/human</option>
                <option value="comparison">comparison</option>
                <option value="all">all samples</option>
              </select>
              <small>Default is reference/human summary, matching the cached report pipeline.</small>
            </label>
          </section>

          <section className="class-details">
            <h3>Selection</h3>
            <dl>
              <dt>Dataset</dt><dd>{overlay.dataset || "-"}</dd>
              <dt>Class</dt><dd>{overlay.classKey || "-"}</dd>
              <dt>Metric</dt><dd>{analysis.metrics[0] || "-"}</dd>
              <dt>Point</dt><dd>{selectedPoint?.source || "none"}</dd>
              <dt>Summary</dt><dd>{summary?.summary || "-"}</dd>
            </dl>
          </section>

          <section className="overlay-card">
            <h3>Overlay Preview</h3>
            <div className="overlay-frame" dangerouslySetInnerHTML={{ __html: overlaySvg || `<p>${overlayLoading ? "Rendering overlay..." : "No overlay available for this selection."}</p>` }} />
            <div className="legend">
              <span><i className="reference"></i> Reference</span>
              <span><i className="candidate"></i> Comparison</span>
              <span><i className="summary"></i> Summary</span>
            </div>
          </section>

          <section className="actions">
            <a href={selectedReportId ? `${API}/reports/${selectedReportId}/table/distribution?${query}` : "#"}>Export Distribution JSON</a>
            <a href={selectedReportId ? `${API}/reports/${selectedReportId}/overlay?source=${encodeURIComponent(overlay.source)}&dataset=${encodeURIComponent(overlay.dataset)}&class=${encodeURIComponent(overlay.classKey)}` : "#"}>Open Overlay SVG</a>
          </section>
        </aside>
      </main>
    </div>
  );
}
