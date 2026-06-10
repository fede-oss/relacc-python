import * as echarts from "echarts";
import { useEffect, useRef, useState } from "react";

const API = "/api";
const DEFAULT_FILTERS = {
  source: "",
  dataset: "",
  variant: "",
  classKey: "",
  metric: "shapeError",
  metricFamily: "core",
};

function formatNumber(value, digits = 3) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  if (Math.abs(number) >= 1000) return number.toLocaleString(undefined, { maximumFractionDigits: 1 });
  return number.toLocaleString(undefined, { maximumFractionDigits: digits });
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
  if (filters.metric) params.set("metric", filters.metric);
  if (filters.metricFamily) params.set("metric_family", filters.metricFamily);
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
  }, [reportId, kind, filters]);

  return { payload, error };
}

function rankingOption(payload) {
  const rows = payload?.rows || [];
  return {
    grid: { left: 92, right: 32, top: 28, bottom: 46 },
    tooltip: { trigger: "axis", valueFormatter: (value) => formatNumber(value) },
    xAxis: {
      type: "value",
      name: "average normalized distance",
      axisLine: { lineStyle: { color: "#111" } },
      splitLine: { lineStyle: { color: "#d8d8d2" } },
    },
    yAxis: {
      type: "category",
      inverse: true,
      data: rows.map((row) => row.source),
      axisLabel: { color: "#111", fontFamily: "IBM Plex Mono, monospace" },
    },
    series: [
      {
        type: "bar",
        data: rows.map((row) => ({
          ...row,
          value: row.score,
          itemStyle: { color: row.score <= 0.5 ? "#0aa3a3" : "#e53935" },
        })),
        barWidth: 16,
      },
    ],
  };
}

function heatmapOption(payload) {
  const sources = payload?.sources || [];
  const metrics = payload?.metrics || [];
  return {
    tooltip: {
      formatter: (params) => {
        const data = params.data;
        return `${data.source}<br/>${data.metric}<br/>z ${formatNumber(data.value)}<br/>raw ${formatNumber(data.raw)}`;
      },
    },
    grid: { left: 100, right: 22, top: 44, bottom: 34 },
    xAxis: { type: "category", data: metrics, axisLabel: { rotate: 18, color: "#111" } },
    yAxis: { type: "category", data: sources, axisLabel: { color: "#111" } },
    visualMap: {
      min: -1.5,
      max: 1.5,
      calculable: false,
      orient: "horizontal",
      left: "center",
      bottom: 0,
      inRange: { color: ["#0aa3a3", "#f7f7f3", "#e53935"] },
    },
    series: [
      {
        type: "heatmap",
        data: (payload?.values || []).map((item) => ({
          ...item,
          value: [item.metricIndex, item.sourceIndex, item.value],
        })),
        label: {
          show: true,
          formatter: (params) => formatNumber(params.data.raw, 2),
          color: "#111",
          fontSize: 10,
        },
      },
    ],
  };
}

function scatterOption(payload) {
  const points = payload?.points || [];
  const sources = [...new Set(points.map((point) => point.source))];
  return {
    tooltip: {
      formatter: (params) => {
        const data = params.data;
        return `${data.source}<br/>${data.dataset} / ${data.classKey}<br/>${data.metric}<br/>human ${formatNumber(data.humanMedian)}<br/>generated ${formatNumber(data.generatedMedian)}`;
      },
    },
    legend: { top: 6, type: "scroll" },
    grid: { left: 54, right: 18, top: 54, bottom: 42 },
    xAxis: { name: "human median", splitLine: { lineStyle: { color: "#d8d8d2" } } },
    yAxis: { name: "generated median", splitLine: { lineStyle: { color: "#d8d8d2" } } },
    series: sources.map((source, index) => ({
      type: "scatter",
      name: source,
      symbolSize: 7,
      itemStyle: { color: ["#0aa3a3", "#e53935", "#111", "#666", "#b00020"][index % 5] },
      data: points
        .filter((point) => point.source === source)
        .map((point) => ({ ...point, value: [point.humanMedian, point.generatedMedian] })),
    })),
  };
}

function distributionOption(payload) {
  const rows = payload?.rows || [];
  const labels = rows.slice(0, 24).map((row) => `${row.source}:${row.classKey}`);
  return {
    tooltip: { trigger: "axis", valueFormatter: (value) => formatNumber(value) },
    legend: { top: 2 },
    grid: { left: 62, right: 18, top: 50, bottom: 80 },
    xAxis: { type: "category", data: labels, axisLabel: { rotate: 55, fontSize: 10 } },
    yAxis: { type: "value", name: payload?.metric || "metric" },
    series: [
      {
        name: "within reference",
        type: "line",
        data: rows.slice(0, 24).map((row) => row.withinReferenceMean),
        color: "#111",
        lineStyle: { type: "dashed" },
      },
      {
        name: "within comparison",
        type: "line",
        data: rows.slice(0, 24).map((row) => row.withinComparisonMean),
        color: "#0aa3a3",
      },
      {
        name: "between groups",
        type: "line",
        data: rows.slice(0, 24).map((row) => row.betweenGroupsMean),
        color: "#e53935",
      },
    ],
  };
}

function ChartPanel({ title, kicker, option, children, onPointClick }) {
  return (
    <section className="panel chart-panel">
      <div className="panel-title">
        <div>
          <span>{kicker}</span>
          <h2>{title}</h2>
        </div>
        <button className="icon-button" type="button" aria-label="Chart options">...</button>
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

function UploadGroups({ groups, setGroups }) {
  function update(index, patch) {
    setGroups(groups.map((group, itemIndex) => (itemIndex === index ? { ...group, ...patch } : group)));
  }
  return (
    <div className="upload-groups">
      {groups.map((group, index) => (
        <div className="zip-row" key={group.id}>
          <input
            aria-label="Comparison group name"
            value={group.name}
            onChange={(event) => update(index, { name: event.target.value })}
          />
          <label>
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

export default function App() {
  const [reports, setReports] = useState([]);
  const [selectedReportId, setSelectedReportId] = useState("");
  const [summary, setSummary] = useState(null);
  const [filterOptions, setFilterOptions] = useState({ sources: [], datasets: [], variants: [], classes: [], classByDataset: {}, metrics: [], metricFamilies: [] });
  const [filters, setFilters] = useState(DEFAULT_FILTERS);
  const [status, setStatus] = useState("Loading cache");
  const [referenceZip, setReferenceZip] = useState(null);
  const [comparisonGroups, setComparisonGroups] = useState([{ id: "first", name: "candidate", file: null }]);
  const [overlaySvg, setOverlaySvg] = useState("");
  const [selectedPoint, setSelectedPoint] = useState(null);

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
        setFilters((current) => {
          const sources = filtersPayload.sources || [];
          const datasets = filtersPayload.datasets || [];
          const metrics = filtersPayload.metrics || [];
          const metricFamilies = filtersPayload.metricFamilies || [];
          const source = sources.includes(current.source) ? current.source : sources[0] || "";
          const dataset = datasets.includes(current.dataset) ? current.dataset : datasets[0] || "";
          const datasetClasses = filtersPayload.classByDataset?.[dataset] || filtersPayload.classes || [];
          return {
            ...current,
            source,
            dataset,
            classKey: datasetClasses.includes(current.classKey) ? current.classKey : datasetClasses[0] || "",
            metric: metrics.includes(current.metric) ? current.metric : metrics[0] || "shapeError",
            metricFamily: metricFamilies.includes(current.metricFamily) ? current.metricFamily : "core",
          };
        });
      })
      .catch((error) => alive && setStatus(error.message));
    return () => {
      alive = false;
    };
  }, [selectedReportId]);

  const ranking = useChart(selectedReportId, "ranking", filters);
  const heatmap = useChart(selectedReportId, "heatmap", filters);
  const scatter = useChart(selectedReportId, "scatter", filters);
  const distribution = useChart(selectedReportId, "distribution", filters);

  async function runEvaluation() {
    if (selectedReport) {
      setStatus("Loaded cached report");
      return;
    }
    const groupsWithFiles = comparisonGroups.filter((group) => group.file);
    if (!referenceZip || !groupsWithFiles.length) {
      setStatus("Upload reference and comparison zips");
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
    try {
      const job = await fetchJsonWithBody(`${API}/jobs`, body);
      setStatus(job.phase || job.status);
    } catch (error) {
      setStatus(error.message);
    }
  }

  useEffect(() => {
    if (!selectedReportId || !filters.source || !filters.dataset || !filters.classKey) return undefined;
    let alive = true;
    const query = new URLSearchParams({
      source: filters.source,
      dataset: filters.dataset,
      class: filters.classKey,
      sample_count: "18",
      summary: summary?.summary || "medoid",
    });
    if (filters.variant) query.set("variant", filters.variant);
    fetch(`${API}/reports/${selectedReportId}/overlay?${query}`)
      .then((response) => (response.ok ? response.text() : ""))
      .then((svg) => alive && setOverlaySvg(svg))
      .catch(() => alive && setOverlaySvg(""));
    return () => {
      alive = false;
    };
  }, [selectedReportId, filters.source, filters.dataset, filters.variant, filters.classKey, summary?.summary]);

  const selectedReport = reports.find((report) => report.id === selectedReportId);
  const sourceCount = summary?.sources?.length || selectedReport?.sources?.length || 0;
  const datasetCount = summary?.datasets?.length || selectedReport?.datasets?.length || 0;
  const classCount = summary?.classCount || selectedReport?.classCount || 0;
  const warningCount = summary?.warningCount || selectedReport?.warningCount || 0;

  const filterPatch = (patch) => setFilters((current) => ({ ...current, ...patch }));
  const classesForDataset = filterOptions.classByDataset?.[filters.dataset] || filterOptions.classes || [];
  const setDatasetFilter = (dataset) => {
    const nextClasses = filterOptions.classByDataset?.[dataset] || filterOptions.classes || [];
    filterPatch({ dataset, classKey: nextClasses[0] || "" });
  };
  const handlePoint = (point) => {
    setSelectedPoint(point);
    filterPatch({
      source: point.source || filters.source,
      dataset: point.dataset || filters.dataset,
      classKey: point.classKey || filters.classKey,
      metric: point.metric || filters.metric,
    });
  };

  const cacheLabel = selectedReport ? `${selectedReport.name} (${formatNumber((selectedReport.sizeBytes || 0) / 1024 / 1024, 1)} MB)` : "no cache selected";

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-mark"><span></span><h1>RELACC WORKBENCH</h1></div>
        <div className="status-block"><span>RUN STATUS</span><strong>{status}</strong></div>
        <div className="status-block"><span>CACHE STATUS</span><strong>{selectedReport ? "UP TO DATE" : "EMPTY"}</strong></div>
        <label className="report-picker">
          <span>SELECTED REPORT FOLDER</span>
          <select value={selectedReportId} onChange={(event) => setSelectedReportId(event.target.value)}>
            {reports.map((report) => <option value={report.id} key={report.id}>{report.name}</option>)}
          </select>
        </label>
        <button className="evaluate" type="button" onClick={runEvaluation}>
          {selectedReport ? "Load Cache" : "Evaluate"}
        </button>
      </header>

      <main className="workbench-grid">
        <aside className="source-rail">
          <h2>Source Controls</h2>
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
              <span>Used only when cache is unavailable.</span>
            </label>
          </section>

          <section className="rail-section">
            <h3>Comparison zip groups</h3>
            <UploadGroups groups={comparisonGroups} setGroups={setComparisonGroups} />
          </section>

          <section className="rail-section compact-controls">
            <Select label="Metric family" value={filters.metricFamily} options={filterOptions.metricFamilies || []} onChange={(metricFamily) => filterPatch({ metricFamily })} allLabel="Core" />
            <Select label="Dataset filter" value={filters.dataset} options={filterOptions.datasets || []} onChange={setDatasetFilter} />
            <Select label="Class filter" value={filters.classKey} options={classesForDataset} onChange={(classKey) => filterPatch({ classKey })} />
            <Select label="Metric" value={filters.metric} options={filterOptions.metrics || []} onChange={(metric) => filterPatch({ metric })} />
          </section>
        </aside>

        <section className="main-stage">
          <section className="kpi-strip">
            <div><span>Cached</span><strong>{selectedReport ? "Yes" : "No"}</strong><em>{summary?.summary || "summary"}</em></div>
            <div><span>Sources</span><strong>{sourceCount}</strong><em>reference + candidates</em></div>
            <div><span>Datasets</span><strong>{datasetCount}</strong><em>{summary?.rate ? `rate ${summary.rate}` : "rate"}</em></div>
            <div><span>Classes</span><strong>{formatNumber(classCount, 0)}</strong><em>gesture classes</em></div>
            <div><span>Warnings</span><strong className={warningCount ? "danger" : ""}>{warningCount}</strong><em>see diagnostics</em></div>
          </section>

          <div className="chart-grid">
            <ChartPanel title="Overall Generator Ranking" kicker="Average normalized distribution distance" option={rankingOption(ranking.payload)} onPointClick={handlePoint} />
            <ChartPanel title="Metric Family Heatmap" kicker="Z-score by source and metric" option={heatmapOption(heatmap.payload)} onPointClick={handlePoint} />
            <ChartPanel title="Human Median vs Generated Median" kicker="Class and metric scatter" option={scatterOption(scatter.payload)} onPointClick={handlePoint} />
            <ChartPanel title="Score Distribution Comparison" kicker="Within reference / within comparison / between groups" option={distributionOption(distribution.payload)} onPointClick={handlePoint} />
          </div>
        </section>

        <aside className="inspector">
          <h2>Inspector</h2>
          <Select label="Source" value={filters.source} options={filterOptions.sources || []} onChange={(source) => filterPatch({ source })} />
          <Select label="Dataset" value={filters.dataset} options={filterOptions.datasets || []} onChange={setDatasetFilter} />
          <Select label="Class" value={filters.classKey} options={classesForDataset} onChange={(classKey) => filterPatch({ classKey })} />

          <section className="class-details">
            <h3>Class Details</h3>
            <dl>
              <dt>Dataset</dt><dd>{filters.dataset || "-"}</dd>
              <dt>Class</dt><dd>{filters.classKey || "-"}</dd>
              <dt>Metric</dt><dd>{filters.metric || "-"}</dd>
              <dt>Selected point</dt><dd>{selectedPoint?.source || "none"}</dd>
              <dt>Summary</dt><dd>{summary?.summary || "-"}</dd>
            </dl>
          </section>

          <section className="overlay-card">
            <h3>Overlay Preview</h3>
            <div className="overlay-frame" dangerouslySetInnerHTML={{ __html: overlaySvg || "<p>No overlay available for this selection.</p>" }} />
            <div className="legend">
              <span><i className="reference"></i> Reference</span>
              <span><i className="candidate"></i> Comparison</span>
              <span><i className="summary"></i> Summary</span>
            </div>
          </section>

          <section className="actions">
            <a href={selectedReportId ? `${API}/reports/${selectedReportId}/table/distribution?${paramsFor(filters)}` : "#"}>Export Data JSON</a>
            <a href={selectedReportId ? `${API}/reports/${selectedReportId}/overlay?source=${encodeURIComponent(filters.source)}&dataset=${encodeURIComponent(filters.dataset)}&class=${encodeURIComponent(filters.classKey)}` : "#"}>Open Overlay SVG</a>
          </section>
        </aside>
      </main>
    </div>
  );
}
