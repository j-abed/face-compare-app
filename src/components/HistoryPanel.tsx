import { useState, useEffect, useCallback } from 'react'
import {
  getHistory,
  deleteComparison,
  clearHistory,
  type HistoryEntry,
} from '../historyStore'
import './HistoryPanel.css'

function formatRelativeTime(timestamp: number): string {
  const sec = Math.floor((Date.now() - timestamp) / 1000)
  if (sec < 60) return sec <= 1 ? 'just now' : `${sec} seconds ago`
  const min = Math.floor(sec / 60)
  if (min < 60) return min === 1 ? '1 minute ago' : `${min} minutes ago`
  const hr = Math.floor(min / 60)
  if (hr < 24) return hr === 1 ? '1 hour ago' : `${hr} hours ago`
  const day = Math.floor(hr / 24)
  return day === 1 ? '1 day ago' : `${day} days ago`
}

function HistoryEntryCard({
  entry,
  expanded,
  onToggle,
}: {
  entry: HistoryEntry
  expanded: boolean
  onToggle: () => void
}) {
  const r = entry.fullResult
  const verdictClass =
    entry.verdict === 'same'
      ? 'history-verdict--same'
      : entry.verdict === 'different'
        ? 'history-verdict--different'
        : 'history-verdict--inconclusive'

  return (
    <div
      className={`history-entry${expanded ? ' history-entry--expanded' : ''}`}
      onClick={onToggle}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          onToggle()
        }
      }}
      aria-expanded={expanded}
    >
      <div className="history-entry-header">
        <div className="history-thumbs">
          <img
            src={entry.image1Thumb}
            alt="Photo 1"
            className="history-thumb"
          />
          <img
            src={entry.image2Thumb}
            alt="Photo 2"
            className="history-thumb"
          />
        </div>
        <div className="history-entry-meta">
          <span className={`history-verdict ${verdictClass}`}>
            {entry.verdict === 'same'
              ? 'Same'
              : entry.verdict === 'different'
                ? 'Different'
                : 'Inconclusive'}
          </span>
          {entry.confidence != null && (
            <span className="history-confidence">
              {Math.round(entry.confidence * 100)}%
            </span>
          )}
          <span className="history-timestamp">
            {formatRelativeTime(entry.timestamp)}
          </span>
        </div>
        <span className="history-expand-icon" aria-hidden>
          {expanded ? '\u25B2' : '\u25BC'}
        </span>
      </div>

      <div className={`history-entry-details${expanded ? ' history-entry-details--open' : ''}`}>
        <div className="history-metrics">
          <div className="history-metric">
            <span className="history-metric-label">Score</span>
            <span className="history-metric-value">
              {(entry.score * 100).toFixed(1)}%
            </span>
          </div>
          {entry.cosineSimilarity != null && (
            <div className="history-metric">
              <span className="history-metric-label">Cosine similarity</span>
              <span className="history-metric-value">
                {entry.cosineSimilarity.toFixed(4)}
              </span>
            </div>
          )}
          {r.euclideanDistance != null && (
            <div className="history-metric">
              <span className="history-metric-label">Euclidean distance</span>
              <span className="history-metric-value">
                {r.euclideanDistance.toFixed(4)}
              </span>
            </div>
          )}
        </div>
        {r.qualityWarnings && r.qualityWarnings.length > 0 && (
          <div className="history-quality-warnings">
            <span className="history-quality-label">Quality warnings:</span>
            <ul>
              {r.qualityWarnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  )
}

function HistoryPanel() {
  const [entries, setEntries] = useState<HistoryEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [expandedId, setExpandedId] = useState<number | null>(null)

  const loadHistory = useCallback(async () => {
    setLoading(true)
    try {
      const list = await getHistory()
      setEntries(list)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadHistory()
  }, [loadHistory])

  const handleClearHistory = useCallback(() => {
    if (!window.confirm('Clear all comparison history? This cannot be undone.')) {
      return
    }
    clearHistory()
      .then(() => setEntries([]))
      .catch(console.error)
  }, [])

  const handleDeleteEntry = useCallback(
    (e: React.MouseEvent, id: number) => {
      e.stopPropagation()
      deleteComparison(id).then(() => {
        setEntries((prev) => prev.filter((e) => e.id !== id))
      })
    },
    []
  )

  if (loading) {
    return (
      <div className="history-panel tab-content">
        <div className="history-loading">
          <div className="progress-spinner" aria-hidden />
          <span>Loading history&hellip;</span>
        </div>
      </div>
    )
  }

  return (
    <div className="history-panel tab-content">
      <div className="history-header">
        <h2>Comparison History</h2>
        {entries.length > 0 && (
          <button
            type="button"
            className="history-clear-btn"
            onClick={handleClearHistory}
          >
            Clear History
          </button>
        )}
      </div>

      {entries.length === 0 ? (
        <div className="history-empty">
          <div className="history-empty-icon">&#128196;</div>
          <h3>No comparisons yet</h3>
          <p>
            Run a face comparison in the Compare tab to see your history here.
          </p>
        </div>
      ) : (
        <ul className="history-list">
          {entries.map((entry) => (
            <li key={entry.id} className="history-list-item">
              <HistoryEntryCard
                entry={entry}
                expanded={expandedId === entry.id}
                onToggle={() =>
                  setExpandedId((prev) =>
                    prev === entry.id ? null : entry.id
                  )
                }
              />
              <button
                type="button"
                className="history-delete-btn"
                onClick={(e) => handleDeleteEntry(e, entry.id)}
                aria-label="Delete this comparison"
                title="Delete"
              >
                &times;
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export default HistoryPanel
