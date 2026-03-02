import {
  useState,
  useCallback,
  useRef,
  useEffect,
  lazy,
  Suspense,
  type ReactNode,
  type DragEvent,
} from 'react'
import {
  compareFaces,
  loadModels,
  type CompareResult,
  type CompareProgressStep,
  type ProgressPayload,
  type FeatureRegions,
} from './faceComparison'
import { saveComparison } from './historyStore'
import { WebcamCapture } from './components/WebcamCapture'
import './App.css'

/* ─── Lazy-loaded BenchmarkPanel (may not exist yet) ───────────── */

const BenchmarkPanel = lazy(() =>
  import('./BenchmarkPanel').catch(() => ({
    default: () => (
      <div className="benchmark-fallback">
        <div className="benchmark-fallback-icon">&#128295;</div>
        <h3>Benchmark Coming Soon</h3>
        <p>The benchmark panel is still being built. Check back shortly.</p>
      </div>
    ),
  }))
)

const IdentifyPanel = lazy(() => import('./components/IdentifyPanel'))
const HistoryPanel = lazy(() => import('./components/HistoryPanel'))

/* ─── LFW Manifest types (matches lfwData.ts) ─────────────────── */

interface LfwPerson {
  name: string
  images: string[]
}

interface LfwManifest {
  source: string
  license: string
  people: LfwPerson[]
  pairs: {
    same: { person: string; image1: string; image2: string }[]
    different: {
      person1: string
      image1: string
      person2: string
      image2: string
    }[]
  }
}

/* ─── Toast system ─────────────────────────────────────────────── */

interface ToastItem {
  id: number
  message: string
  type: 'error' | 'info'
  exiting?: boolean
}

let toastIdCounter = 0

function ToastContainer({
  toasts,
  onDismiss,
}: {
  toasts: ToastItem[]
  onDismiss: (id: number) => void
}) {
  return (
    <div className="toast-container">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={`toast toast--${t.type}${t.exiting ? ' toast--exiting' : ''}`}
        >
          <span className="toast-icon">
            {t.type === 'error' ? '\u26D4' : '\u2139\uFE0F'}
          </span>
          <div className="toast-body">
            <p className="toast-message">{t.message}</p>
          </div>
          <button
            className="toast-dismiss"
            onClick={() => onDismiss(t.id)}
            aria-label="Dismiss"
          >
            &times;
          </button>
        </div>
      ))}
    </div>
  )
}

function useToasts() {
  const [toasts, setToasts] = useState<ToastItem[]>([])

  const addToast = useCallback((message: string, type: 'error' | 'info' = 'error') => {
    const id = ++toastIdCounter
    setToasts((prev) => [...prev, { id, message, type }])
    setTimeout(() => {
      setToasts((prev) =>
        prev.map((t) => (t.id === id ? { ...t, exiting: true } : t))
      )
      setTimeout(() => {
        setToasts((prev) => prev.filter((t) => t.id !== id))
      }, 300)
    }, 5000)
  }, [])

  const dismissToast = useCallback((id: number) => {
    setToasts((prev) =>
      prev.map((t) => (t.id === id ? { ...t, exiting: true } : t))
    )
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id))
    }, 300)
  }, [])

  return { toasts, addToast, dismissToast }
}

/* ─── AnimatedNumber ───────────────────────────────────────────── */

function AnimatedNumber({
  value,
  decimals = 0,
  suffix = '',
  duration = 500,
}: {
  value: number
  decimals?: number
  suffix?: string
  duration?: number
}) {
  const [display, setDisplay] = useState(0)
  const rafRef = useRef(0)

  useEffect(() => {
    const start = performance.now()
    const from = 0
    const to = value

    const tick = (now: number) => {
      const elapsed = now - start
      const progress = Math.min(elapsed / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3)
      setDisplay(from + (to - from) * eased)
      if (progress < 1) {
        rafRef.current = requestAnimationFrame(tick)
      }
    }

    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [value, duration])

  return (
    <>
      {display.toFixed(decimals)}
      {suffix}
    </>
  )
}

/* ─── CollapsibleSection ───────────────────────────────────────── */

function CollapsibleSection({
  title,
  children,
  defaultOpen = false,
}: {
  title: string
  children: ReactNode
  defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <div className="collapsible-section" data-open={open}>
      <button
        type="button"
        className="collapsible-header"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <span className="collapsible-title">{title}</span>
        <span className={`collapsible-chevron${open ? ' collapsible-chevron--open' : ''}`} aria-hidden>
          &#9660;
        </span>
      </button>
      <div
        className={`collapsible-body${open ? ' collapsible-body--open' : ''}`}
        style={open ? { maxHeight: 3000 } : undefined}
      >
        <div className="collapsible-body-inner">{children}</div>
      </div>
    </div>
  )
}

/* ─── TabBar ───────────────────────────────────────────────────── */

type TabId = 'compare' | 'benchmark' | 'identify' | 'gallery' | 'history'

const TABS: { id: TabId; label: string }[] = [
  { id: 'compare', label: 'Compare' },
  { id: 'benchmark', label: 'Benchmark' },
  { id: 'identify', label: 'Identify' },
  { id: 'gallery', label: 'Gallery' },
  { id: 'history', label: 'History' },
]

function TabBar({
  active,
  onChange,
}: {
  active: TabId
  onChange: (tab: TabId) => void
}) {
  const tabListRef = useRef<HTMLDivElement>(null)

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent, index: number) => {
      if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault()
        const prev = (index - 1 + TABS.length) % TABS.length
        onChange(TABS[prev].id)
        ;(tabListRef.current?.querySelectorAll('button')[prev] as HTMLButtonElement)?.focus()
      } else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        e.preventDefault()
        const next = (index + 1) % TABS.length
        onChange(TABS[next].id)
        ;(tabListRef.current?.querySelectorAll('button')[next] as HTMLButtonElement)?.focus()
      }
    },
    [onChange]
  )

  return (
    <nav className="tab-bar" role="tablist" aria-label="Main navigation" ref={tabListRef}>
      {TABS.map((tab, index) => (
        <button
          key={tab.id}
          role="tab"
          aria-selected={active === tab.id}
          aria-label={`${tab.label} tab`}
          id={`tab-${tab.id}`}
          aria-controls={`tabpanel-${tab.id}`}
          tabIndex={active === tab.id ? 0 : -1}
          className={`tab-btn${active === tab.id ? ' tab-btn--active' : ''}`}
          onClick={() => onChange(tab.id)}
          onKeyDown={(e) => handleKeyDown(e, index)}
        >
          {tab.label}
        </button>
      ))}
    </nav>
  )
}

/* ─── DropZone ─────────────────────────────────────────────────── */

type QualityBadgeLevel = 'good' | 'minor' | 'major'

function getQualityBadgeLevel(warnings: string[] | undefined): QualityBadgeLevel {
  if (!warnings || warnings.length === 0) return 'good'
  const hasMajor = warnings.some(
    (w) =>
      /over-exposed|under-exposed|too bright|too dark|outside|occlusion|very small/i.test(w)
  )
  if (hasMajor) return 'major'
  return 'minor'
}

function DropZone({
  label,
  imageUrl,
  fileName,
  onFile,
  onRemove,
  onImageUrl,
  addToast,
  scanning,
  comparing,
  overlayCanvas,
  showCaption,
  qualityWarnings,
}: {
  label: string
  imageUrl: string | null
  fileName: string | null
  onFile: (file: File) => void
  onRemove: () => void
  onImageUrl?: (blobUrl: string, filename?: string) => void
  addToast?: (message: string, type?: 'error' | 'info') => void
  scanning?: boolean
  comparing?: boolean
  overlayCanvas?: ReactNode
  showCaption?: boolean
  qualityWarnings?: string[]
}) {
  const [dragOver, setDragOver] = useState(false)
  const [urlInput, setUrlInput] = useState('')
  const [urlLoading, setUrlLoading] = useState(false)
  const [showWebcam, setShowWebcam] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault()
      setDragOver(false)
      const file = e.dataTransfer.files?.[0]
      if (file && file.type.startsWith('image/')) {
        onFile(file)
      }
    },
    [onFile]
  )

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault()
    setDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: DragEvent) => {
    e.preventDefault()
    setDragOver(false)
  }, [])

  const handleClick = useCallback(() => {
    if (!imageUrl) {
      inputRef.current?.click()
    }
  }, [imageUrl])

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) onFile(file)
      e.target.value = ''
    },
    [onFile]
  )

  const handleLoadFromUrl = useCallback(async () => {
    const url = urlInput.trim()
    if (!url) return
    setUrlLoading(true)
    try {
      const res = await fetch(url, { mode: 'cors' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const blob = await res.blob()
      if (!blob.type.startsWith('image/')) {
        throw new Error('URL does not point to an image')
      }
      const file = new File([blob], 'image-from-url.jpg', { type: blob.type })
      onFile(file)
      setUrlInput('')
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      if (msg.includes('Failed to fetch') || msg.includes('CORS') || msg.includes('NetworkError')) {
        addToast?.('Could not load image from URL. The server may block cross-origin requests.', 'error')
      } else {
        addToast?.(msg, 'error')
      }
    } finally {
      setUrlLoading(false)
    }
  }, [urlInput, onFile, addToast])

  const handleWebcamCapture = useCallback(
    (blobUrl: string) => {
      onImageUrl?.(blobUrl, 'webcam-capture.jpg')
      setShowWebcam(false)
    },
    [onImageUrl]
  )

  useEffect(() => {
    const handlePaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items
      if (!items) return
      for (const item of items) {
        if (item.type.startsWith('image/')) {
          const file = item.getAsFile()
          if (file) {
            onFile(file)
            break
          }
        }
      }
    }
    window.addEventListener('paste', handlePaste)
    return () => window.removeEventListener('paste', handlePaste)
  }, [onFile])

  const dzClass = [
    'dropzone',
    dragOver && 'dropzone--dragover',
    imageUrl && 'dropzone--has-file',
    scanning && 'dropzone--scanning',
    comparing && 'dropzone--comparing',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <div className="upload-cell">
      <label className="upload-label">{label}</label>
      <div
        className={dzClass}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={handleClick}
        role={imageUrl ? undefined : 'button'}
        tabIndex={imageUrl ? undefined : 0}
        aria-label={`${label} drop zone. Drop an image or click to browse.`}
        onKeyDown={
          imageUrl
            ? undefined
            : (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault()
                  inputRef.current?.click()
                }
              }
        }
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={handleInputChange}
        />
        {imageUrl ? (
          <div className="dropzone-preview">
            <div className="dropzone-img-wrap">
              {qualityWarnings !== undefined && (
                <span
                  className={`quality-badge quality-badge--${getQualityBadgeLevel(qualityWarnings)}`}
                  title={qualityWarnings.length > 0 ? qualityWarnings.join('\n') : 'Good quality'}
                >
                  {qualityWarnings.length === 0 ? '\u2713' : '\u26A0'}
                </span>
              )}
              <div className="preview-wrap">
                <img src={imageUrl} alt={label} className="preview-img" />
              </div>
              {scanning && (
                <>
                  <div className="scan-brackets" aria-label="Scanning face in image" aria-live="polite">
                    <span />
                  </div>
                  <span className="preview-scan-badge" aria-hidden>Scanning</span>
                </>
              )}
              {overlayCanvas}
            </div>
            <div className="dropzone-footer">
              <span className="dropzone-filename">{fileName ?? 'Image'}</span>
              <button
                className="dropzone-remove"
                onClick={(e) => {
                  e.stopPropagation()
                  onRemove()
                }}
                aria-label="Remove image"
              >
                &times;
              </button>
            </div>
          </div>
        ) : (
          <>
            <span className="dropzone-icon">&#128247;</span>
            <span className="dropzone-text">
              Drop photo here or <em>click to browse</em>
            </span>
          </>
        )}
      </div>
      <div className="dropzone-url-row">
        <input
          type="url"
          className="dropzone-url-input"
          placeholder="Or paste image URL"
          value={urlInput}
          onChange={(e) => setUrlInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleLoadFromUrl()}
        />
        <button
          type="button"
          className="dropzone-url-load"
          onClick={handleLoadFromUrl}
          disabled={!urlInput.trim() || urlLoading}
        >
          {urlLoading ? 'Loading…' : 'Load'}
        </button>
        <button
          type="button"
          className="dropzone-camera-btn"
          onClick={(e) => {
            e.stopPropagation()
            setShowWebcam(true)
          }}
          aria-label="Capture from camera"
          title="Capture from camera"
        >
          <span className="dropzone-camera-icon" aria-hidden>📷</span>
        </button>
      </div>
      {showCaption && <p className="preview-caption">Detected face &amp; landmarks</p>}
      <WebcamCapture
        isOpen={showWebcam}
        onClose={() => setShowWebcam(false)}
        onCapture={handleWebcamCapture}
      />
    </div>
  )
}

/* ─── FeatureOverlayCanvas ─────────────────────────────────────── */

function FeatureOverlayCanvas({
  regions,
  imageDimensions,
}: {
  regions: FeatureRegions
  imageDimensions: { width: number; height: number }
}) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const wrap = wrapRef.current
    const canvas = canvasRef.current
    if (!wrap || !canvas || !wrap.parentElement) return
    const w = wrap.parentElement.offsetWidth
    const h = wrap.parentElement.offsetHeight
    if (w <= 0 || h <= 0) return
    const scaleX = w / imageDimensions.width
    const scaleY = h / imageDimensions.height
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const drawRegion = (
      points: { x: number; y: number }[],
      color: string,
      fillAlpha: number,
      label?: string
    ) => {
      if (points.length < 2) return
      const sx = (p: { x: number; y: number }) => p.x * scaleX
      const sy = (p: { x: number; y: number }) => p.y * scaleY

      ctx.beginPath()
      ctx.moveTo(sx(points[0]), sy(points[0]))
      for (let i = 1; i < points.length; i++) ctx.lineTo(sx(points[i]), sy(points[i]))
      ctx.closePath()

      ctx.fillStyle = color.replace(/[\d.]+\)$/, `${fillAlpha})`)
      ctx.fill()

      ctx.strokeStyle = color
      ctx.lineWidth = 1.5
      ctx.stroke()

      if (label) {
        const cx = points.reduce((s, p) => s + p.x, 0) / points.length
        const cy = points.reduce((s, p) => s + p.y, 0) / points.length
        ctx.fillStyle = color
        ctx.font = 'bold 9px Inter, system-ui, sans-serif'
        ctx.textAlign = 'center'
        ctx.fillText(label, cx * scaleX, cy * scaleY - 5)
      }
    }

    drawRegion(regions.eyes.left, 'rgba(96, 165, 250, 0.8)', 0.08, 'L')
    drawRegion(regions.eyes.right, 'rgba(96, 165, 250, 0.8)', 0.08, 'R')
    drawRegion(regions.nose, 'rgba(251, 191, 36, 0.8)', 0.06)
    drawRegion(regions.mouth, 'rgba(244, 114, 182, 0.8)', 0.08)
    if (regions.jaw && regions.jaw.length > 0) {
      drawRegion(regions.jaw, 'rgba(52, 211, 153, 0.5)', 0.03)
    }
  }, [regions, imageDimensions])

  return (
    <div ref={wrapRef} className="feature-overlay" aria-hidden>
      <canvas ref={canvasRef} className="feature-overlay-canvas" />
    </div>
  )
}

/* ─── Constants ────────────────────────────────────────────────── */

const PROGRESS_LABELS: Record<CompareProgressStep, string> = {
  loading_models: 'Loading face recognition models\u2026',
  loading_images: 'Loading images\u2026',
  detecting_face_1: 'Scanning face in photo 1\u2026',
  detecting_face_2: 'Scanning face in photo 2\u2026',
  comparing: 'Analyzing face descriptors\u2026',
}

const FEATURE_CHECKLIST = ['Eyes', 'Nose', 'Mouth', 'Jaw', 'Descriptor'] as const

function getSamplePhotoUrls(): string[] {
  return [
    '/lfw/Tony_Blair/Tony_Blair_0001.jpg',
    '/lfw/Tony_Blair/Tony_Blair_0003.jpg',
    '/lfw/Hugo_Chavez/Hugo_Chavez_0001.jpg',
    '/lfw/Hugo_Chavez/Hugo_Chavez_0005.jpg',
    '/lfw/David_Beckham/David_Beckham_0001.jpg',
    '/lfw/David_Beckham/David_Beckham_0004.jpg',
    '/lfw/Tiger_Woods/Tiger_Woods_0001.jpg',
    '/lfw/Tiger_Woods/Tiger_Woods_0005.jpg',
    '/lfw/Jelena_Dokic/Jelena_Dokic_0001.jpg',
    '/lfw/Jelena_Dokic/Jelena_Dokic_0003.jpg',
    '/lfw/Hans_Blix/Hans_Blix_0001.jpg',
    '/lfw/Pete_Sampras/Pete_Sampras_0001.jpg',
    '/lfw/Steven_Spielberg/Steven_Spielberg_0001.jpg',
    '/lfw/Liza_Minnelli/Liza_Minnelli_0001.jpg',
    '/lfw/Jean_Chretien/Jean_Chretien_0001.jpg',
    '/lfw/Silvio_Berlusconi/Silvio_Berlusconi_0001.jpg',
    '/lfw/Yasser_Arafat/Yasser_Arafat_0001.jpg',
    '/lfw/Paula_Radcliffe/Paula_Radcliffe_0001.jpg',
    '/lfw/Wen_Jiabao/Wen_Jiabao_0001.jpg',
    '/lfw/Gerry_Adams/Gerry_Adams_0001.jpg',
  ]
}

function pickTwoRandom<T>(arr: T[]): [T, T] {
  const copy = [...arr]
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[copy[i], copy[j]] = [copy[j], copy[i]]
  }
  return [copy[0], copy[1]]
}

/* ─── Ratio cell color helper ──────────────────────────────────── */

function ratioCellClass(a: number, b: number): string {
  if (a === 0 && b === 0) return ''
  const avg = (Math.abs(a) + Math.abs(b)) / 2
  if (avg === 0) return ''
  const diff = Math.abs(a - b) / avg
  if (diff <= 0.1) return 'cell-close'
  if (diff > 0.25) return 'cell-divergent'
  return ''
}

/* ─── SimilarityBar ────────────────────────────────────────────── */

function SimilarityBar({ value }: { value: number }) {
  const pct = Math.round(value * 100)
  const level = pct >= 70 ? 'high' : pct >= 40 ? 'mid' : 'low'
  return (
    <div className="similarity-bar-wrap">
      <div className="similarity-bar-label">
        <span>Similarity</span>
        <span>{pct}%</span>
      </div>
      <div className="similarity-bar">
        <div
          className={`similarity-bar-fill similarity-bar-fill--${level}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

/* ─── Gallery ──────────────────────────────────────────────────── */

function Gallery({
  onCompareSelected,
}: {
  onCompareSelected: (url1: string, url2: string) => void
}) {
  const [manifest, setManifest] = useState<LfwManifest | null>(null)
  const [loadError, setLoadError] = useState(false)
  const [loading, setLoading] = useState(true)
  const [selected, setSelected] = useState<string[]>([])

  useEffect(() => {
    let cancelled = false
    fetch('/lfw/manifest.json')
      .then((r) => {
        if (!r.ok) throw new Error('Not found')
        return r.json()
      })
      .then((data: LfwManifest) => {
        if (!cancelled) {
          setManifest(data)
          setLoading(false)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setLoadError(true)
          setLoading(false)
        }
      })
    return () => {
      cancelled = true
    }
  }, [])

  const toggleSelect = useCallback((url: string) => {
    setSelected((prev) => {
      if (prev.includes(url)) return prev.filter((u) => u !== url)
      if (prev.length >= 2) return [prev[1], url]
      return [...prev, url]
    })
  }, [])

  const handleRandomPair = useCallback(() => {
    if (!manifest?.people.length) return
    const allImages = manifest.people.flatMap((p) => p.images)
    if (allImages.length < 2) return
    const [a, b] = pickTwoRandom(allImages)
    setSelected([a, b])
  }, [manifest])

  if (loading) {
    return (
      <div className="gallery tab-content">
        <div className="gallery-header">
          <h2>LFW Sample Gallery</h2>
        </div>
        <div className="gallery-grid" aria-busy="true" aria-label="Loading gallery">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="gallery-person-card skeleton">
              <div className="gallery-person-name" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '0.25rem' }}>
                <div className="skeleton-line skeleton-line--medium" />
                <div className="skeleton-line skeleton-line--short" />
              </div>
              <div className="gallery-person-images">
                {Array.from({ length: 6 }).map((_, j) => (
                  <div key={j} className="skeleton-block" style={{ width: 60, height: 60, borderRadius: 8 }} />
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  if (loadError || !manifest) {
    return (
      <div className="gallery-empty">
        <div className="gallery-empty-icon">&#128444;&#65039;</div>
        <h3>Gallery Not Available Yet</h3>
        <p>
          The LFW sample gallery is still being prepared. The manifest file at{' '}
          <code>/lfw/manifest.json</code> is not available yet.
        </p>
      </div>
    )
  }

  return (
    <div className="gallery tab-content">
      <div className="gallery-header">
        <h2>LFW Sample Gallery</h2>
        <div className="gallery-actions">
          {selected.length > 0 && (
            <span className="gallery-selection-info">
              {selected.length}/2 selected
            </span>
          )}
          <button type="button" onClick={handleRandomPair} className="sample-btn">
            Random Pair
          </button>
          {selected.length === 2 && (
            <button
              type="button"
              className="compare-btn"
              onClick={() => onCompareSelected(selected[0], selected[1])}
            >
              Compare Selected
            </button>
          )}
        </div>
      </div>
      <div className="gallery-grid">
        {manifest.people.map((person) => {
          const MAX_THUMBS = 6
          const visible = person.images.slice(0, MAX_THUMBS)
          const remaining = person.images.length - MAX_THUMBS
          return (
            <div key={person.name} className="gallery-person-card">
              <div className="gallery-person-name">
                {person.name.replace(/_/g, ' ')}
                <span className="gallery-person-count">{person.images.length} photos</span>
              </div>
              <div className="gallery-person-images">
                {visible.map((img) => {
                  const url = img
                  const idx = selected.indexOf(url)
                  const isSelected = idx >= 0
                  return (
                    <div
                      key={img}
                      className={`gallery-thumb-wrap${isSelected ? ' gallery-thumb-wrap--selected' : ''}`}
                      onClick={() => toggleSelect(url)}
                    >
                      <img
                        src={url}
                        alt={person.name.replace(/_/g, ' ')}
                        className="gallery-thumb"
                        loading="lazy"
                      />
                      {isSelected && (
                        <span className="gallery-thumb-badge">{idx + 1}</span>
                      )}
                    </div>
                  )
                })}
                {remaining > 0 && (
                  <div className="gallery-thumb-more">+{remaining}</div>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

/* ─── Main App ─────────────────────────────────────────────────── */

const THEME_KEY = 'face-compare-theme'

function App() {
  const [activeTab, setActiveTab] = useState<TabId>('compare')
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    const saved = localStorage.getItem(THEME_KEY)
    return saved === 'light' ? 'light' : 'dark'
  })
  const [modelsLoaded, setModelsLoaded] = useState(false)
  const [image1, setImage1] = useState<string | null>(null)
  const [image2, setImage2] = useState<string | null>(null)
  const [fileName1, setFileName1] = useState<string | null>(null)
  const [fileName2, setFileName2] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [progressStep, setProgressStep] = useState<CompareProgressStep | null>(null)
  const [result, setResult] = useState<CompareResult | null>(null)
  const [liveAnnotated1, setLiveAnnotated1] = useState<string | null>(null)
  const [liveAnnotated2, setLiveAnnotated2] = useState<string | null>(null)
  const [comparingPayload, setComparingPayload] = useState<ProgressPayload | null>(null)
  const [checklistStep, setChecklistStep] = useState(-1)
  const [showAlignedOverlay, setShowAlignedOverlay] = useState(false)
  const { toasts, addToast, dismissToast } = useToasts()

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem(THEME_KEY, theme)
  }, [theme])

  const toggleTheme = useCallback(() => {
    setTheme((t) => (t === 'dark' ? 'light' : 'dark'))
  }, [])

  useEffect(() => {
    loadModels().then(() => setModelsLoaded(true)).catch(() => {})
  }, [])

  useEffect(() => {
    if (progressStep !== 'comparing') {
      setChecklistStep(-1)
      return
    }
    setChecklistStep(0)
    const timeouts: ReturnType<typeof setTimeout>[] = []
    for (let i = 1; i <= FEATURE_CHECKLIST.length; i++) {
      timeouts.push(setTimeout(() => setChecklistStep(i), i * 450))
    }
    return () => timeouts.forEach((t) => clearTimeout(t))
  }, [progressStep])

  const clearResults = useCallback(() => {
    setResult(null)
    setLiveAnnotated1(null)
    setLiveAnnotated2(null)
  }, [])

  const handleFile1 = useCallback(
    (file: File) => {
      clearResults()
      setFileName1(file.name)
      setImage1(URL.createObjectURL(file))
    },
    [clearResults]
  )

  const handleFile2 = useCallback(
    (file: File) => {
      clearResults()
      setFileName2(file.name)
      setImage2(URL.createObjectURL(file))
    },
    [clearResults]
  )

  const removeImage1 = useCallback(() => {
    clearResults()
    setImage1(null)
    setFileName1(null)
  }, [clearResults])

  const removeImage2 = useCallback(() => {
    clearResults()
    setImage2(null)
    setFileName2(null)
  }, [clearResults])

  const handleImageUrl1 = useCallback(
    (blobUrl: string, filename?: string) => {
      clearResults()
      setImage1((prev) => {
        if (prev?.startsWith('blob:')) URL.revokeObjectURL(prev)
        return blobUrl
      })
      setFileName1(filename ?? 'webcam-capture.jpg')
    },
    [clearResults]
  )

  const handleImageUrl2 = useCallback(
    (blobUrl: string, filename?: string) => {
      clearResults()
      setImage2((prev) => {
        if (prev?.startsWith('blob:')) URL.revokeObjectURL(prev)
        return blobUrl
      })
      setFileName2(filename ?? 'webcam-capture.jpg')
    },
    [clearResults]
  )

  const handleCompare = async () => {
    if (!image1 || !image2) return
    setLoading(true)
    setProgressStep(null)
    setResult(null)
    setLiveAnnotated1(null)
    setLiveAnnotated2(null)
    setComparingPayload(null)
    try {
      const res = await compareFaces(image1, image2, (step, payload) => {
        setProgressStep(step)
        if (payload?.annotatedImage1) setLiveAnnotated1(payload.annotatedImage1)
        if (payload?.annotatedImage2) setLiveAnnotated2(payload.annotatedImage2)
        if (step === 'comparing' && payload) setComparingPayload(payload)
      })
      setResult(res)
      if (!res.error) {
        saveComparison(res, image1, image2).catch(console.error)
      }
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err)
      setResult({ samePerson: false, verdict: 'inconclusive', score: 0, error: errMsg })
      addToast(errMsg, 'error')
    } finally {
      setLoading(false)
      setProgressStep(null)
    }
  }

  const canCompare = image1 && image2 && !loading

  const useSamplePhotos = () => {
    clearResults()
    const urls = getSamplePhotoUrls()
    const [url1, url2] = pickTwoRandom(urls)
    setImage1(url1)
    setImage2(url2)
    setFileName1(null)
    setFileName2(null)
  }

  const handleGalleryCompare = useCallback(
    (url1: string, url2: string) => {
      clearResults()
      setImage1(url1)
      setImage2(url2)
      setFileName1(null)
      setFileName2(null)
      setActiveTab('compare')
    },
    [clearResults]
  )

  const displayImage1 = liveAnnotated1 ?? result?.annotatedImage1 ?? image1
  const displayImage2 = liveAnnotated2 ?? result?.annotatedImage2 ?? image2

  const isScanning1 = progressStep === 'detecting_face_1'
  const isScanning2 = progressStep === 'detecting_face_2'
  const isComparing = progressStep === 'comparing'

  return (
    <div className="app">
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />

      <header className="app-header">
        <div className="app-header-inner">
          <h1>Face Comparison</h1>
          <p className="subtitle">
            Upload two photos to check if they show the same person.
          </p>
        </div>
        <div className="app-header-actions">
          {modelsLoaded ? (
            <span className="models-loaded" aria-label="Models loaded">
              <span aria-hidden>✓</span>
              <span>Ready</span>
            </span>
          ) : (
            <span className="models-loading" aria-label="Loading face recognition models">
              <span className="progress-spinner" aria-hidden />
              Loading models…
            </span>
          )}
          <button
            type="button"
            className="theme-toggle-btn"
            onClick={toggleTheme}
            aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
            title={theme === 'dark' ? 'Light mode' : 'Dark mode'}
          >
            <span aria-hidden>{theme === 'dark' ? '☀️' : '🌙'}</span>
          </button>
        </div>
      </header>

      <TabBar active={activeTab} onChange={setActiveTab} />

      {activeTab === 'compare' && (
        <div className="tab-content" key="compare" role="tabpanel" id="tabpanel-compare" aria-labelledby="tab-compare">
          <div className="actions-top">
            <button type="button" onClick={useSamplePhotos} className="sample-btn">
              Use sample photos
            </button>
          </div>

          <div className="upload-row">
            <DropZone
              label="Photo 1"
              imageUrl={displayImage1}
              fileName={fileName1}
              onFile={handleFile1}
              onRemove={removeImage1}
              onImageUrl={handleImageUrl1}
              addToast={addToast}
              scanning={isScanning1}
              comparing={isComparing}
              showCaption={!!(liveAnnotated1 || result?.annotatedImage1)}
              qualityWarnings={result ? (result.qualityWarnings1 ?? []) : undefined}
              overlayCanvas={
                loading &&
                isComparing &&
                comparingPayload?.featureRegions1 &&
                comparingPayload?.imageDimensions1 ? (
                  <FeatureOverlayCanvas
                    regions={comparingPayload.featureRegions1}
                    imageDimensions={comparingPayload.imageDimensions1}
                  />
                ) : undefined
              }
            />
            <DropZone
              label="Photo 2"
              imageUrl={displayImage2}
              fileName={fileName2}
              onFile={handleFile2}
              onRemove={removeImage2}
              onImageUrl={handleImageUrl2}
              addToast={addToast}
              scanning={isScanning2}
              comparing={isComparing}
              showCaption={!!(liveAnnotated2 || result?.annotatedImage2)}
              qualityWarnings={result ? (result.qualityWarnings2 ?? []) : undefined}
              overlayCanvas={
                loading &&
                isComparing &&
                comparingPayload?.featureRegions2 &&
                comparingPayload?.imageDimensions2 ? (
                  <FeatureOverlayCanvas
                    regions={comparingPayload.featureRegions2}
                    imageDimensions={comparingPayload.imageDimensions2}
                  />
                ) : undefined
              }
            />
          </div>

          {loading && progressStep && (
            <div className="progress-box">
              <div className="progress-header">
                <div className="progress-spinner" aria-hidden />
                <p className="progress-text">{PROGRESS_LABELS[progressStep]}</p>
              </div>
              {progressStep === 'comparing' && (
                <>
                  <p className="progress-comparing-desc">
                    Comparing face descriptors and landmark geometry.
                  </p>
                  <ul className="feature-checklist" aria-label="Features being compared">
                    {FEATURE_CHECKLIST.map((label, i) => (
                      <li
                        key={label}
                        className={
                          i <= checklistStep ? 'feature-checklist-item--done' : ''
                        }
                      >
                        {i < checklistStep
                          ? '\u2713'
                          : i === checklistStep
                            ? '\u2026'
                            : '\u25CB'}{' '}
                        {label}
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </div>
          )}

          <div className="actions">
            <button
              onClick={handleCompare}
              disabled={!canCompare}
              className="compare-btn"
            >
              {loading ? 'Comparing\u2026' : 'Compare faces'}
            </button>
          </div>

          {result && result.error && (
            <div className="result">
              <div className="quality-warnings">
                <p className="quality-warning">Error: {result.error}</p>
              </div>
            </div>
          )}

          {result && !result.error && (
            <div className="result">
              {result.qualityWarnings && result.qualityWarnings.length > 0 && (
                <div className="quality-warnings">
                  {result.qualityWarnings.map((w, i) => (
                    <p key={i} className="quality-warning">
                      {w}
                    </p>
                  ))}
                </div>
              )}

              <div className="result-verdict" aria-live="polite" aria-atomic="true">
                <div
                  className={`verdict-icon verdict-icon--${result.verdict ?? (result.samePerson ? 'same' : 'different')}`}
                  aria-label={
                    result.verdict === 'inconclusive'
                      ? 'Inconclusive result'
                      : result.verdict === 'same'
                        ? 'Same person'
                        : 'Different persons'
                  }
                >
                  {result.verdict === 'inconclusive'
                    ? '?'
                    : result.verdict === 'same'
                      ? '\u2713'
                      : '\u2717'}
                </div>
                <div>
                  <p className="result-message">
                    {result.verdict === 'inconclusive'
                      ? 'Inconclusive'
                      : result.verdict === 'same'
                        ? 'Same person'
                        : 'Different persons'}
                  </p>
                  {result.verdict === 'inconclusive' && (
                    <p className="verdict-explanation">
                      The signals are ambiguous or confidence is too low to make a
                      reliable determination.
                    </p>
                  )}
                  {result.confidence != null && (
                    <div
                      className={`confidence-badge confidence-${result.confidenceLabel?.toLowerCase().replace(' ', '-')}`}
                    >
                      Confidence:{' '}
                      <AnimatedNumber
                        value={Math.round(result.confidence * 100)}
                        suffix="%"
                      />{' '}
                      ({result.confidenceLabel})
                    </div>
                  )}
                </div>
              </div>

              <SimilarityBar value={result.score} />

              <div className="metrics-grid">
                {result.euclideanDistance != null && (
                  <div className="metric-card">
                    <span className="metric-label">Descriptor distance</span>
                    <span className="metric-value">
                      <AnimatedNumber value={result.euclideanDistance} decimals={4} />
                    </span>
                    <div className="metric-sub">Lower = more similar</div>
                  </div>
                )}
                {result.cosineSimilarity != null && (
                  <div className="metric-card">
                    <span className="metric-label">Cosine similarity</span>
                    <span className="metric-value">
                      <AnimatedNumber value={result.cosineSimilarity} decimals={4} />
                    </span>
                    <div className="metric-sub">Higher = more similar</div>
                  </div>
                )}
                {result.geometricSimilarity != null && (
                  <div className="metric-card">
                    <span className="metric-label">Geometric similarity</span>
                    <span className="metric-value">
                      <AnimatedNumber
                        value={Math.round(result.geometricSimilarity * 100)}
                        suffix="%"
                      />
                    </span>
                    <div className="metric-sub">Facial ratio comparison</div>
                  </div>
                )}
                {result.landmarkAlignment != null && (
                  <div className="metric-card">
                    <span className="metric-label">Landmark alignment</span>
                    <span className="metric-value">
                      <AnimatedNumber
                        value={Math.round(result.landmarkAlignment * 100)}
                        suffix="%"
                      />
                    </span>
                    <div className="metric-sub">Procrustes analysis</div>
                  </div>
                )}
              </div>

              {result.face1 && result.face2 && (
                <div className="features-section">
                  <CollapsibleSection key="feature-comparison" title="Feature comparison">
                    <p className="features-desc">
                      Geometric measures from detected landmarks. Absolute values in
                      pixels; normalized ratios are scale-invariant.
                    </p>
                    <table className="features-table">
                      <thead>
                        <tr>
                          <th>Feature</th>
                          <th>Photo 1</th>
                          <th>Photo 2</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td>Detection confidence</td>
                          <td
                            className={ratioCellClass(
                              result.face1!.detection.score,
                              result.face2!.detection.score
                            )}
                          >
                            {(result.face1!.detection.score * 100).toFixed(1)}%
                          </td>
                          <td
                            className={ratioCellClass(
                              result.face1!.detection.score,
                              result.face2!.detection.score
                            )}
                          >
                            {(result.face2!.detection.score * 100).toFixed(1)}%
                          </td>
                        </tr>
                        <tr>
                          <td>Face box (W&times;H)</td>
                          <td>
                            {Math.round(result.face1!.features.boxWidth)} &times;{' '}
                            {Math.round(result.face1!.features.boxHeight)} px
                          </td>
                          <td>
                            {Math.round(result.face2!.features.boxWidth)} &times;{' '}
                            {Math.round(result.face2!.features.boxHeight)} px
                          </td>
                        </tr>
                        <tr>
                          <td>Inter-eye distance</td>
                          <td>
                            {result.face1!.features.interEyeDistance.toFixed(1)} px
                          </td>
                          <td>
                            {result.face2!.features.interEyeDistance.toFixed(1)} px
                          </td>
                        </tr>
                        <tr>
                          <td>Face aspect ratio</td>
                          <td
                            className={ratioCellClass(
                              result.face1!.features.faceAspectRatio,
                              result.face2!.features.faceAspectRatio
                            )}
                          >
                            {result.face1!.features.faceAspectRatio.toFixed(3)}
                          </td>
                          <td
                            className={ratioCellClass(
                              result.face1!.features.faceAspectRatio,
                              result.face2!.features.faceAspectRatio
                            )}
                          >
                            {result.face2!.features.faceAspectRatio.toFixed(3)}
                          </td>
                        </tr>
                        <tr>
                          <td>Nose length</td>
                          <td>{result.face1!.features.noseLength.toFixed(1)} px</td>
                          <td>{result.face2!.features.noseLength.toFixed(1)} px</td>
                        </tr>
                        <tr>
                          <td>Mouth width</td>
                          <td>{result.face1!.features.mouthWidth.toFixed(1)} px</td>
                          <td>{result.face2!.features.mouthWidth.toFixed(1)} px</td>
                        </tr>
                        <tr>
                          <td>Jaw width</td>
                          <td>{result.face1!.features.jawWidth.toFixed(1)} px</td>
                          <td>{result.face2!.features.jawWidth.toFixed(1)} px</td>
                        </tr>
                        <tr>
                          <td>Eyebrow slope (L / R)</td>
                          <td>
                            {result.face1!.features.eyebrowSlopeLeft.toFixed(3)} /{' '}
                            {result.face1!.features.eyebrowSlopeRight.toFixed(3)}
                          </td>
                          <td>
                            {result.face2!.features.eyebrowSlopeLeft.toFixed(3)} /{' '}
                            {result.face2!.features.eyebrowSlopeRight.toFixed(3)}
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </CollapsibleSection>

                  <CollapsibleSection key="normalized-ratios" title="Normalized ratios (scale-invariant)">
                    <p className="features-desc">
                      Ratios normalized by inter-eye distance &mdash; directly
                      comparable regardless of image size.
                    </p>
                    <table className="features-table">
                      <thead>
                        <tr>
                          <th>Ratio</th>
                          <th>Photo 1</th>
                          <th>Photo 2</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td>Nose / inter-eye</td>
                          <td
                            className={ratioCellClass(
                              result.face1!.features.noseToInterEye,
                              result.face2!.features.noseToInterEye
                            )}
                          >
                            {result.face1!.features.noseToInterEye.toFixed(3)}
                          </td>
                          <td
                            className={ratioCellClass(
                              result.face1!.features.noseToInterEye,
                              result.face2!.features.noseToInterEye
                            )}
                          >
                            {result.face2!.features.noseToInterEye.toFixed(3)}
                          </td>
                        </tr>
                        <tr>
                          <td>Mouth / inter-eye</td>
                          <td
                            className={ratioCellClass(
                              result.face1!.features.mouthToInterEye,
                              result.face2!.features.mouthToInterEye
                            )}
                          >
                            {result.face1!.features.mouthToInterEye.toFixed(3)}
                          </td>
                          <td
                            className={ratioCellClass(
                              result.face1!.features.mouthToInterEye,
                              result.face2!.features.mouthToInterEye
                            )}
                          >
                            {result.face2!.features.mouthToInterEye.toFixed(3)}
                          </td>
                        </tr>
                        <tr>
                          <td>Jaw / inter-eye</td>
                          <td
                            className={ratioCellClass(
                              result.face1!.features.jawToInterEye,
                              result.face2!.features.jawToInterEye
                            )}
                          >
                            {result.face1!.features.jawToInterEye.toFixed(3)}
                          </td>
                          <td
                            className={ratioCellClass(
                              result.face1!.features.jawToInterEye,
                              result.face2!.features.jawToInterEye
                            )}
                          >
                            {result.face2!.features.jawToInterEye.toFixed(3)}
                          </td>
                        </tr>
                        <tr>
                          <td>Nose-to-mouth / inter-eye</td>
                          <td
                            className={ratioCellClass(
                              result.face1!.features.noseToMouthToInterEye,
                              result.face2!.features.noseToMouthToInterEye
                            )}
                          >
                            {result.face1!.features.noseToMouthToInterEye.toFixed(3)}
                          </td>
                          <td
                            className={ratioCellClass(
                              result.face1!.features.noseToMouthToInterEye,
                              result.face2!.features.noseToMouthToInterEye
                            )}
                          >
                            {result.face2!.features.noseToMouthToInterEye.toFixed(3)}
                          </td>
                        </tr>
                        <tr>
                          <td>Eye / jaw ratio</td>
                          <td
                            className={ratioCellClass(
                              result.face1!.features.eyeToJawRatio,
                              result.face2!.features.eyeToJawRatio
                            )}
                          >
                            {result.face1!.features.eyeToJawRatio.toFixed(3)}
                          </td>
                          <td
                            className={ratioCellClass(
                              result.face1!.features.eyeToJawRatio,
                              result.face2!.features.eyeToJawRatio
                            )}
                          >
                            {result.face2!.features.eyeToJawRatio.toFixed(3)}
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </CollapsibleSection>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {activeTab === 'benchmark' && (
        <div className="tab-content" key="benchmark" role="tabpanel" id="tabpanel-benchmark" aria-labelledby="tab-benchmark">
          <Suspense
            fallback={
              <div className="benchmark-loading">
                <div className="progress-spinner" />
                <span>Loading benchmark&hellip;</span>
              </div>
            }
          >
            <BenchmarkPanel />
          </Suspense>
        </div>
      )}

      {activeTab === 'identify' && (
        <div className="tab-content" key="identify" role="tabpanel" id="tabpanel-identify" aria-labelledby="tab-identify">
          <Suspense
            fallback={
              <div className="benchmark-loading">
                <div className="progress-spinner" />
                <span>Loading identify&hellip;</span>
              </div>
            }
          >
            <IdentifyPanel />
          </Suspense>
        </div>
      )}

      {activeTab === 'gallery' && (
        <div className="tab-content" key="gallery" role="tabpanel" id="tabpanel-gallery" aria-labelledby="tab-gallery">
          <Gallery onCompareSelected={handleGalleryCompare} />
        </div>
      )}

      {activeTab === 'history' && (
        <div className="tab-content" key="history">
          <Suspense
            fallback={
              <div className="benchmark-loading">
                <div className="progress-spinner" />
                <span>Loading history&hellip;</span>
              </div>
            }
          >
            <HistoryPanel />
          </Suspense>
        </div>
      )}
    </div>
  )
}

export default App
