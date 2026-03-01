import { useState, useCallback, useRef } from 'react';
import {
  runBenchmark,
  type BenchmarkResults,
  type BenchmarkProgress,
  type PairResult,
} from './benchmark';
import './BenchmarkPanel.css';

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
        <div className="benchmark-progress">
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

          <div className="benchmark-metrics-grid">
            <MetricCard
              label="Accuracy"
              value={formatPct(results.accuracy)}
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
    </div>
  );
}
