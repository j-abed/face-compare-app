import { useState, useCallback, useRef, useEffect } from 'react';
import {
  runBenchmark,
  type BenchmarkResults,
  type BenchmarkProgress,
  type PairResult,
  type RocPoint,
  type DetPoint,
} from './benchmark';
import './BenchmarkPanel.css';

type BenchmarkTab = 'summary' | 'roc' | 'det' | 'perPerson';

function formatPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatMs(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function MetricCard({
  label,
  value,
  colorClass,
}: {
  label: string;
  value: string;
  colorClass?: string;
}) {
  return (
    <div className="benchmark-metric-card">
      <span className="benchmark-metric-label">{label}</span>
      <span className={`benchmark-metric-value ${colorClass ?? ''}`}>{value}</span>
    </div>
  );
}

function ConfusionMatrix({ tp, tn, fp, fn }: { tp: number; tn: number; fp: number; fn: number }) {
  return (
    <div className="benchmark-confusion-section">
      <h4 className="benchmark-section-title">Confusion Matrix</h4>
      <div className="benchmark-confusion-grid">
        <div className="benchmark-confusion-corner" />
        <div className="benchmark-confusion-col-header">Pred Same</div>
        <div className="benchmark-confusion-col-header">Pred Diff</div>

        <div className="benchmark-confusion-row-header">Actual Same</div>
        <div className="benchmark-confusion-cell confusion-tp">
          {tp}
          <span>TP</span>
        </div>
        <div className="benchmark-confusion-cell confusion-fn">
          {fn}
          <span>FN</span>
        </div>

        <div className="benchmark-confusion-row-header">Actual Diff</div>
        <div className="benchmark-confusion-cell confusion-fp">
          {fp}
          <span>FP</span>
        </div>
        <div className="benchmark-confusion-cell confusion-tn">
          {tn}
          <span>TN</span>
        </div>
      </div>
    </div>
  );
}

const CHART_SIZE = 400;
const CHART_BG = '#232340';
const CHART_AXIS = '#e2e8f0';
const CHART_GRID = 'rgba(226, 232, 240, 0.2)';
const CHART_CURVE = '#6c63ff';
const CHART_DIAG = 'rgba(226, 232, 240, 0.5)';

function RocChart({ data, auc }: { data: RocPoint[]; auc: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio ?? 1;
    const size = CHART_SIZE;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    ctx.scale(dpr, dpr);

    const pad = { left: 50, right: 20, top: 20, bottom: 50 };
    const w = size - pad.left - pad.right;
    const h = size - pad.top - pad.bottom;

    ctx.fillStyle = CHART_BG;
    ctx.fillRect(0, 0, size, size);

    ctx.strokeStyle = CHART_GRID;
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 5; i++) {
      const x = pad.left + (w * i) / 5;
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, size - pad.bottom);
      ctx.stroke();
      const y = pad.top + (h * (5 - i)) / 5;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(size - pad.right, y);
      ctx.stroke();
    }

    ctx.strokeStyle = CHART_AXIS;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, size - pad.bottom);
    ctx.lineTo(size - pad.right, size - pad.bottom);
    ctx.stroke();

    ctx.fillStyle = CHART_AXIS;
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('False Positive Rate', size / 2, size - 10);
    ctx.save();
    ctx.translate(12, size / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('True Positive Rate', 0, 0);
    ctx.restore();

    ctx.font = '9px system-ui, sans-serif';
    for (let i = 0; i <= 5; i++) {
      const x = pad.left + (w * i) / 5;
      ctx.textAlign = 'center';
      ctx.fillText((i / 5).toFixed(1), x, size - pad.bottom + 14);
      const y = pad.top + (h * (5 - i)) / 5;
      ctx.textAlign = 'right';
      ctx.fillText((i / 5).toFixed(1), pad.left - 6, y + 4);
    }

    const toX = (fpr: number) => pad.left + fpr * w;
    const toY = (tpr: number) => pad.top + (1 - tpr) * h;

    ctx.strokeStyle = CHART_DIAG;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pad.left, size - pad.bottom);
    ctx.lineTo(size - pad.right, pad.top);
    ctx.stroke();
    ctx.setLineDash([]);

    const sorted = [...data].sort((a, b) => a.fpr - b.fpr);
    ctx.strokeStyle = CHART_CURVE;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(toX(sorted[0].fpr), toY(sorted[0].tpr));
    for (let i = 1; i < sorted.length; i++) {
      ctx.lineTo(toX(sorted[i].fpr), toY(sorted[i].tpr));
    }
    ctx.stroke();

    ctx.fillStyle = CHART_AXIS;
    ctx.font = '12px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`AUC = ${auc.toFixed(3)}`, pad.left + w - 80, pad.top + 14);
  }, [data, auc]);

  return <canvas ref={canvasRef} className="benchmark-chart-canvas" />;
}

function DetChart({ data }: { data: DetPoint[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio ?? 1;
    const size = CHART_SIZE;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    ctx.scale(dpr, dpr);

    const pad = { left: 55, right: 20, top: 20, bottom: 55 };
    const w = size - pad.left - pad.right;
    const h = size - pad.top - pad.bottom;

    const minRate = 0.0001;
    const maxRate = 1;
    const toLog = (r: number) => Math.log10(Math.max(minRate, Math.min(maxRate, r)));
    const logMin = toLog(minRate);
    const logMax = toLog(maxRate);
    const toX = (fpr: number) => pad.left + ((toLog(fpr) - logMin) / (logMax - logMin)) * w;
    const toY = (fnr: number) => pad.top + ((toLog(fnr) - logMin) / (logMax - logMin)) * h;

    ctx.fillStyle = CHART_BG;
    ctx.fillRect(0, 0, size, size);

    const ticks = [0.0001, 0.001, 0.01, 0.1, 1];
    ctx.strokeStyle = CHART_GRID;
    ctx.lineWidth = 0.5;
    for (const v of ticks) {
      const x = toX(v);
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, size - pad.bottom);
      ctx.stroke();
      const y = toY(v);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(size - pad.right, y);
      ctx.stroke();
    }

    ctx.strokeStyle = CHART_AXIS;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, size - pad.bottom);
    ctx.lineTo(size - pad.right, size - pad.bottom);
    ctx.stroke();

    ctx.fillStyle = CHART_AXIS;
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('False Positive Rate', size / 2, size - 12);
    ctx.save();
    ctx.translate(14, size / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('False Negative Rate', 0, 0);
    ctx.restore();

    ctx.font = '9px system-ui, sans-serif';
    for (const v of ticks) {
      const x = toX(v);
      ctx.textAlign = 'center';
      ctx.fillText(v === 1 ? '1' : v.toFixed(v >= 0.1 ? 1 : 3), x, size - pad.bottom + 14);
      const y = toY(v);
      ctx.textAlign = 'right';
      ctx.fillText(v === 1 ? '1' : v.toFixed(v >= 0.1 ? 1 : 3), pad.left - 6, y + 4);
    }

    const sorted = [...data].sort((a, b) => a.fpr - b.fpr);
    ctx.strokeStyle = CHART_CURVE;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(toX(sorted[0].fpr), toY(sorted[0].fnr));
    for (let i = 1; i < sorted.length; i++) {
      ctx.lineTo(toX(sorted[i].fpr), toY(sorted[i].fnr));
    }
    ctx.stroke();
  }, [data]);

  return <canvas ref={canvasRef} className="benchmark-chart-canvas" />;
}

function FailureCard({ pr }: { pr: PairResult }) {
  const r = pr.result;
  return (
    <div className="benchmark-failure-card">
      <div className="benchmark-failure-thumbs">
        <img src={pr.image1} alt="" loading="lazy" />
        <img src={pr.image2} alt="" loading="lazy" />
      </div>
      <div className="benchmark-failure-info">
        <div className="benchmark-failure-person">{pr.personInfo}</div>
        <div className="benchmark-failure-verdicts">
          <span className="benchmark-failure-truth">
            Truth: {pr.isSame ? 'same' : 'different'}
          </span>
          <span
            className={`benchmark-failure-predicted benchmark-failure-predicted--${r.verdict}`}
          >
            Pred: {r.verdict}
          </span>
        </div>
      </div>
      <div className="benchmark-failure-metrics">
        {r.euclideanDistance != null && (
          <div className="benchmark-failure-metric">
            <span className="benchmark-failure-metric-label">Dist</span>
            <span className="benchmark-failure-metric-value">
              {r.euclideanDistance.toFixed(3)}
            </span>
          </div>
        )}
        {r.cosineSimilarity != null && (
          <div className="benchmark-failure-metric">
            <span className="benchmark-failure-metric-label">Cosine</span>
            <span className="benchmark-failure-metric-value">
              {r.cosineSimilarity.toFixed(3)}
            </span>
          </div>
        )}
        {r.score != null && (
          <div className="benchmark-failure-metric">
            <span className="benchmark-failure-metric-label">Score</span>
            <span className="benchmark-failure-metric-value">
              {r.score.toFixed(3)}
            </span>
          </div>
        )}
        {r.confidence != null && (
          <div className="benchmark-failure-metric">
            <span className="benchmark-failure-metric-label">Conf</span>
            <span className="benchmark-failure-metric-value">
              {formatPct(r.confidence)}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

export default function BenchmarkPanel() {
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<BenchmarkProgress | null>(null);
  const [results, setResults] = useState<BenchmarkResults | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef(false);

  const handleRun = useCallback(async () => {
    setRunning(true);
    setResults(null);
    setError(null);
    setProgress(null);
    abortRef.current = false;

    try {
      const res = await runBenchmark((prog) => {
        setProgress({ ...prog });
      });
      setResults(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  }, []);

  const handleExport = useCallback(() => {
    if (!results) return;
    const stripped = {
      ...results,
      pairResults: results.pairResults.map((pr) => ({
        ...pr,
        result: {
          ...pr.result,
          annotatedImage1: undefined,
          annotatedImage2: undefined,
        },
      })),
    };
    const blob = new Blob([JSON.stringify(stripped, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `benchmark-results-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [results]);

  const failures = results?.pairResults.filter((pr) => !pr.correct) ?? [];
  const [activeTab, setActiveTab] = useState<BenchmarkTab>('summary');

  return (
    <div className="benchmark-panel">
      <h2>LFW Benchmark</h2>
      <p className="benchmark-subtitle">
        Run face comparisons against labeled pairs to measure accuracy.
      </p>

      <button
        type="button"
        className="benchmark-run-btn"
        onClick={handleRun}
        disabled={running}
      >
        {running ? 'Running\u2026' : 'Run Benchmark'}
      </button>

      {error && <div className="benchmark-error">{error}</div>}

      {running && progress && (
        <>
          <div className="benchmark-progress" aria-busy="true" aria-label="Benchmark in progress">
            <div className="benchmark-progress-header">
              <span className="benchmark-progress-label">Progress</span>
              <span className="benchmark-progress-count">
                {progress.current} / {progress.total}
              </span>
            </div>
            <div className="benchmark-progress-bar">
              <div
                className="benchmark-progress-fill"
                style={{ width: `${(progress.current / progress.total) * 100}%` }}
              />
            </div>
            <div className="benchmark-current-pair">
              <img src={progress.currentPair.image1} alt="" />
              <span className="benchmark-pair-vs">vs</span>
              <img src={progress.currentPair.image2} alt="" />
              <span
                className={`benchmark-pair-label benchmark-pair-label--${
                  progress.currentPair.isSame ? 'same' : 'different'
                }`}
              >
                {progress.currentPair.isSame ? 'Same' : 'Different'}
              </span>
            </div>
          </div>
          <div className="benchmark-results" style={{ marginTop: '1.5rem' }}>
            <div className="benchmark-results-header">
              <h3>Results</h3>
            </div>
            <div className="benchmark-metrics-grid">
              {Array.from({ length: 6 }).map((_, i) => (
                <div key={i} className="benchmark-metric-card skeleton">
                  <div className="skeleton-line skeleton-line--short" style={{ marginBottom: '0.25rem' }} />
                  <div className="skeleton-line skeleton-line--medium" />
                </div>
              ))}
            </div>
            <div className="benchmark-failures-section">
              <h4 className="benchmark-section-title">Failure Cases</h4>
              <div className="benchmark-failures-list">
                {Array.from({ length: 4 }).map((_, i) => (
                  <div key={i} className="benchmark-failure-card skeleton">
                    <div className="benchmark-failure-thumbs">
                      <div className="skeleton-block" style={{ width: 40, height: 40, borderRadius: 6 }} />
                      <div className="skeleton-block" style={{ width: 40, height: 40, borderRadius: 6 }} />
                    </div>
                    <div className="benchmark-failure-info" style={{ flex: 1 }}>
                      <div className="skeleton-line skeleton-line--medium" style={{ marginBottom: '0.25rem' }} />
                      <div className="skeleton-line skeleton-line--short" />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}

      {results && (
        <div className="benchmark-results">
          <div className="benchmark-results-header">
            <h3>Results</h3>
            <button
              type="button"
              className="benchmark-export-btn"
              onClick={handleExport}
            >
              Export JSON
            </button>
          </div>

          <div className="benchmark-tabs">
            <button
              type="button"
              className={`benchmark-tab ${activeTab === 'summary' ? 'benchmark-tab--active' : ''}`}
              onClick={() => setActiveTab('summary')}
            >
              Summary
            </button>
            <button
              type="button"
              className={`benchmark-tab ${activeTab === 'roc' ? 'benchmark-tab--active' : ''}`}
              onClick={() => setActiveTab('roc')}
            >
              ROC Curve
            </button>
            <button
              type="button"
              className={`benchmark-tab ${activeTab === 'det' ? 'benchmark-tab--active' : ''}`}
              onClick={() => setActiveTab('det')}
            >
              DET Curve
            </button>
            <button
              type="button"
              className={`benchmark-tab ${activeTab === 'perPerson' ? 'benchmark-tab--active' : ''}`}
              onClick={() => setActiveTab('perPerson')}
            >
              Per-Person
            </button>
          </div>

          {activeTab === 'summary' && (
            <div className="benchmark-tab-content">
              <div className="benchmark-metrics-grid">
                <MetricCard
                  label="Accuracy"
                  value={formatPct(results.accuracy)}
                  colorClass="benchmark-metric-value--accent"
                />
                <MetricCard
                  label="AUC"
                  value={results.auc.toFixed(3)}
                  colorClass="benchmark-metric-value--accent"
                />
                <MetricCard
                  label="Precision"
                  value={formatPct(results.precision)}
                />
                <MetricCard
                  label="Recall"
                  value={formatPct(results.recall)}
                />
                <MetricCard
                  label="F1 Score"
                  value={formatPct(results.f1)}
                  colorClass="benchmark-metric-value--green"
                />
                <MetricCard
                  label="False Positive Rate"
                  value={formatPct(results.falsePositiveRate)}
                  colorClass="benchmark-metric-value--red"
                />
                <MetricCard
                  label="False Negative Rate"
                  value={formatPct(results.falseNegativeRate)}
                  colorClass="benchmark-metric-value--red"
                />
              </div>

              <ConfusionMatrix
                tp={results.truePositives}
                tn={results.trueNegatives}
                fp={results.falsePositives}
                fn={results.falseNegatives}
              />

              <div className="benchmark-extra-stats">
                <span className="benchmark-stat-chip">
                  Total pairs: <strong>{results.totalPairs}</strong>
                </span>
                <span className="benchmark-stat-chip">
                  Same: <strong>{results.samePairs}</strong>
                </span>
                <span className="benchmark-stat-chip">
                  Different: <strong>{results.differentPairs}</strong>
                </span>
                <span className="benchmark-stat-chip">
                  Inconclusive: <strong>{results.inconclusiveCount}</strong>
                </span>
                <span className="benchmark-stat-chip">
                  Errors: <strong>{results.errors}</strong>
                </span>
                <span className="benchmark-stat-chip">
                  Avg time/pair: <strong>{formatMs(results.avgTimePerPairMs)}</strong>
                </span>
                <span className="benchmark-stat-chip">
                  Total time: <strong>{formatMs(results.totalTimeMs)}</strong>
                </span>
              </div>

              <div className="benchmark-failures-section">
                <h4 className="benchmark-section-title">
                  Failure Cases ({failures.length})
                </h4>
                {failures.length === 0 ? (
                  <div className="benchmark-no-failures">
                    All pairs classified correctly.
                  </div>
                ) : (
                  <div className="benchmark-failures-list">
                    {failures.map((pr) => (
                      <FailureCard key={pr.pairIndex} pr={pr} />
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'roc' && (
            <div className="benchmark-tab-content benchmark-chart-container">
              <RocChart data={results.rocData} auc={results.auc} />
            </div>
          )}

          {activeTab === 'det' && (
            <div className="benchmark-tab-content benchmark-chart-container">
              <DetChart data={results.detData} />
            </div>
          )}

          {activeTab === 'perPerson' && (
            <div className="benchmark-tab-content">
              <h4 className="benchmark-section-title">Per-Person Accuracy</h4>
              <div className="benchmark-per-person-table-wrap">
                <table className="benchmark-per-person-table">
                  <thead>
                    <tr>
                      <th>Person</th>
                      <th>Correct</th>
                      <th>Total</th>
                      <th>Accuracy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.perPersonAccuracy.map((p) => (
                      <tr
                        key={p.name}
                        className={
                          p.accuracy >= 1
                            ? 'benchmark-per-person--green'
                            : p.accuracy >= 0.5
                              ? 'benchmark-per-person--yellow'
                              : 'benchmark-per-person--red'
                        }
                      >
                        <td>{p.name}</td>
                        <td>{p.correct}</td>
                        <td>{p.total}</td>
                        <td>{formatPct(p.accuracy)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
