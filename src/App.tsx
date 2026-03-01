import { useState, useCallback, useRef, useEffect } from 'react'
import {
  compareFaces,
  type CompareResult,
  type CompareProgressStep,
  type ProgressPayload,
  type FeatureRegions,
} from './faceComparison'
import './App.css'

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
      label?: string,
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

const PROGRESS_LABELS: Record<CompareProgressStep, string> = {
  loading_models: 'Loading face recognition models\u2026',
  loading_images: 'Loading images\u2026',
  detecting_face_1: 'Scanning face in photo 1\u2026',
  detecting_face_2: 'Scanning face in photo 2\u2026',
  comparing: 'Analyzing face descriptors\u2026',
}

const FEATURE_CHECKLIST = ['Eyes', 'Nose', 'Mouth', 'Jaw', 'Descriptor'] as const

function getSamplePhotoUrls(): string[] {
  const base = typeof window !== 'undefined' ? window.location.origin : ''
  return [
    `${base}/sample1.jpg`,
    `${base}/sample2.jpg`,
    `${base}/sample3.jpg`,
    `${base}/sample4.jpg`,
    `${base}/sample5.jpg`,
    `${base}/sample6.jpg`,
    `${base}/sample7.jpg`,
    `${base}/sample8.jpg`,
    `${base}/sample9.jpg`,
    `${base}/sample10.jpg`,
  ]
}

function pickTwoRandom<T>(arr: T[]): [T, T] {
  const copy = [...arr]
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]]
  }
  return [copy[0], copy[1]]
}

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

function App() {
  const [image1, setImage1] = useState<string | null>(null)
  const [image2, setImage2] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [progressStep, setProgressStep] = useState<CompareProgressStep | null>(null)
  const [result, setResult] = useState<CompareResult | null>(null)
  const [liveAnnotated1, setLiveAnnotated1] = useState<string | null>(null)
  const [liveAnnotated2, setLiveAnnotated2] = useState<string | null>(null)
  const [comparingPayload, setComparingPayload] = useState<ProgressPayload | null>(null)
  const [checklistStep, setChecklistStep] = useState(-1)

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

  const handleFile = useCallback(
    (file: File | null, setter: (url: string | null) => void) => {
      if (result) setResult(null)
      setLiveAnnotated1(null)
      setLiveAnnotated2(null)
      if (!file) {
        setter(null)
        return
      }
      const url = URL.createObjectURL(file)
      setter(url)
    },
    [result]
  )

  const onImage1Change = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleFile(e.target.files?.[0] ?? null, setImage1)
  }
  const onImage2Change = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleFile(e.target.files?.[0] ?? null, setImage2)
  }

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
    } catch (err) {
      setResult({
        samePerson: false,
        verdict: 'inconclusive',
        score: 0,
        error: err instanceof Error ? err.message : 'Models failed to load. Check console.',
      })
    } finally {
      setLoading(false)
      setProgressStep(null)
    }
  }

  const canCompare = image1 && image2 && !loading

  const useSamplePhotos = () => {
    setResult(null)
    setLiveAnnotated1(null)
    setLiveAnnotated2(null)
    const urls = getSamplePhotoUrls()
    const [url1, url2] = pickTwoRandom(urls)
    setImage1(url1)
    setImage2(url2)
  }

  const displayImage1 = liveAnnotated1 ?? result?.annotatedImage1 ?? image1
  const displayImage2 = liveAnnotated2 ?? result?.annotatedImage2 ?? image2

  const isScanning1 = progressStep === 'detecting_face_1'
  const isScanning2 = progressStep === 'detecting_face_2'
  const isComparing = progressStep === 'comparing'

  return (
    <div className="app">
      <header className="app-header">
        <h1>Face Comparison</h1>
        <p className="subtitle">Upload two photos to check if they show the same person.</p>
      </header>

      <div className="actions-top">
        <button type="button" onClick={useSamplePhotos} className="sample-btn">
          Use sample photos
        </button>
      </div>

      <div className="upload-row">
        <div className="upload-cell">
          <label className="upload-label">Photo 1</label>
          <input
            type="file"
            accept="image/jpeg,image/png,image/webp"
            onChange={onImage1Change}
            className="file-input"
          />
          <div
            className={`preview-wrap${isScanning1 ? ' preview-wrap--scanning' : ''}${isComparing ? ' preview-wrap--comparing' : ''}`}
          >
            {displayImage1 ? (
              <img src={displayImage1} alt="Photo 1" className="preview-img" />
            ) : (
              <span className="placeholder">Choose image</span>
            )}
            {isScanning1 && (
              <>
                <div className="scan-brackets"><span /></div>
                <span className="preview-scan-badge">Scanning</span>
              </>
            )}
            {loading && isComparing && comparingPayload?.featureRegions1 && comparingPayload?.imageDimensions1 && (
              <FeatureOverlayCanvas
                regions={comparingPayload.featureRegions1}
                imageDimensions={comparingPayload.imageDimensions1}
              />
            )}
          </div>
          {(liveAnnotated1 || result?.annotatedImage1) && (
            <p className="preview-caption">Detected face &amp; landmarks</p>
          )}
        </div>

        <div className="upload-cell">
          <label className="upload-label">Photo 2</label>
          <input
            type="file"
            accept="image/jpeg,image/png,image/webp"
            onChange={onImage2Change}
            className="file-input"
          />
          <div
            className={`preview-wrap${isScanning2 ? ' preview-wrap--scanning' : ''}${isComparing ? ' preview-wrap--comparing' : ''}`}
          >
            {displayImage2 ? (
              <img src={displayImage2} alt="Photo 2" className="preview-img" />
            ) : (
              <span className="placeholder">Choose image</span>
            )}
            {isScanning2 && (
              <>
                <div className="scan-brackets"><span /></div>
                <span className="preview-scan-badge">Scanning</span>
              </>
            )}
            {loading && isComparing && comparingPayload?.featureRegions2 && comparingPayload?.imageDimensions2 && (
              <FeatureOverlayCanvas
                regions={comparingPayload.featureRegions2}
                imageDimensions={comparingPayload.imageDimensions2}
              />
            )}
          </div>
          {(liveAnnotated2 || result?.annotatedImage2) && (
            <p className="preview-caption">Detected face &amp; landmarks</p>
          )}
        </div>
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
                    className={i <= checklistStep ? 'feature-checklist-item--done' : ''}
                  >
                    {i < checklistStep ? '\u2713' : i === checklistStep ? '\u2026' : '\u25CB'} {label}
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

      {result && (
        <div className={`result ${result.error ? 'result-error' : ''}`}>
          {result.error ? (
            <p className="result-message error">{result.error}</p>
          ) : (
            <>
              {result.qualityWarnings && result.qualityWarnings.length > 0 && (
                <div className="quality-warnings">
                  {result.qualityWarnings.map((w, i) => (
                    <p key={i} className="quality-warning">{w}</p>
                  ))}
                </div>
              )}

              <div className="result-verdict">
                <div className={`verdict-icon verdict-icon--${result.verdict ?? (result.samePerson ? 'same' : 'different')}`}>
                  {result.verdict === 'inconclusive' ? '?' : result.verdict === 'same' ? '\u2713' : '\u2717'}
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
                      The signals are ambiguous or confidence is too low to make a reliable determination.
                    </p>
                  )}
                  {result.confidence != null && (
                    <div className={`confidence-badge confidence-${result.confidenceLabel?.toLowerCase().replace(' ', '-')}`}>
                      Confidence: {Math.round(result.confidence * 100)}% ({result.confidenceLabel})
                    </div>
                  )}
                </div>
              </div>

              <SimilarityBar value={result.score} />

              <div className="metrics-grid">
                {result.euclideanDistance != null && (
                  <div className="metric-card">
                    <span className="metric-label">Descriptor distance</span>
                    <span className="metric-value">{result.euclideanDistance.toFixed(4)}</span>
                    <div className="metric-sub">Lower = more similar</div>
                  </div>
                )}
                {result.cosineSimilarity != null && (
                  <div className="metric-card">
                    <span className="metric-label">Cosine similarity</span>
                    <span className="metric-value">{result.cosineSimilarity.toFixed(4)}</span>
                    <div className="metric-sub">Higher = more similar</div>
                  </div>
                )}
                {result.geometricSimilarity != null && (
                  <div className="metric-card">
                    <span className="metric-label">Geometric similarity</span>
                    <span className="metric-value">{Math.round(result.geometricSimilarity * 100)}%</span>
                    <div className="metric-sub">Facial ratio comparison</div>
                  </div>
                )}
                {result.landmarkAlignment != null && (
                  <div className="metric-card">
                    <span className="metric-label">Landmark alignment</span>
                    <span className="metric-value">{Math.round(result.landmarkAlignment * 100)}%</span>
                    <div className="metric-sub">Procrustes analysis</div>
                  </div>
                )}
              </div>

              {result.face1 && result.face2 && (
                <div className="features-section">
                  <h3 className="features-title">Feature comparison</h3>
                  <p className="features-desc">
                    Geometric measures from detected landmarks. Absolute values in pixels; normalized ratios are scale-invariant.
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
                        <td>{(result.face1.detection.score * 100).toFixed(1)}%</td>
                        <td>{(result.face2.detection.score * 100).toFixed(1)}%</td>
                      </tr>
                      <tr>
                        <td>Face box (W&times;H)</td>
                        <td>{Math.round(result.face1.features.boxWidth)} &times; {Math.round(result.face1.features.boxHeight)} px</td>
                        <td>{Math.round(result.face2.features.boxWidth)} &times; {Math.round(result.face2.features.boxHeight)} px</td>
                      </tr>
                      <tr>
                        <td>Inter-eye distance</td>
                        <td>{result.face1.features.interEyeDistance.toFixed(1)} px</td>
                        <td>{result.face2.features.interEyeDistance.toFixed(1)} px</td>
                      </tr>
                      <tr>
                        <td>Face aspect ratio</td>
                        <td>{result.face1.features.faceAspectRatio.toFixed(3)}</td>
                        <td>{result.face2.features.faceAspectRatio.toFixed(3)}</td>
                      </tr>
                      <tr>
                        <td>Nose length</td>
                        <td>{result.face1.features.noseLength.toFixed(1)} px</td>
                        <td>{result.face2.features.noseLength.toFixed(1)} px</td>
                      </tr>
                      <tr>
                        <td>Mouth width</td>
                        <td>{result.face1.features.mouthWidth.toFixed(1)} px</td>
                        <td>{result.face2.features.mouthWidth.toFixed(1)} px</td>
                      </tr>
                      <tr>
                        <td>Jaw width</td>
                        <td>{result.face1.features.jawWidth.toFixed(1)} px</td>
                        <td>{result.face2.features.jawWidth.toFixed(1)} px</td>
                      </tr>
                      <tr>
                        <td>Eyebrow slope (L / R)</td>
                        <td>{result.face1.features.eyebrowSlopeLeft.toFixed(3)} / {result.face1.features.eyebrowSlopeRight.toFixed(3)}</td>
                        <td>{result.face2.features.eyebrowSlopeLeft.toFixed(3)} / {result.face2.features.eyebrowSlopeRight.toFixed(3)}</td>
                      </tr>
                    </tbody>
                  </table>

                  <h3 className="features-title" style={{ marginTop: '1.25rem' }}>Normalized ratios (scale-invariant)</h3>
                  <p className="features-desc">
                    Ratios normalized by inter-eye distance — directly comparable regardless of image size.
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
                        <td>{result.face1.features.noseToInterEye.toFixed(3)}</td>
                        <td>{result.face2.features.noseToInterEye.toFixed(3)}</td>
                      </tr>
                      <tr>
                        <td>Mouth / inter-eye</td>
                        <td>{result.face1.features.mouthToInterEye.toFixed(3)}</td>
                        <td>{result.face2.features.mouthToInterEye.toFixed(3)}</td>
                      </tr>
                      <tr>
                        <td>Jaw / inter-eye</td>
                        <td>{result.face1.features.jawToInterEye.toFixed(3)}</td>
                        <td>{result.face2.features.jawToInterEye.toFixed(3)}</td>
                      </tr>
                      <tr>
                        <td>Nose-to-mouth / inter-eye</td>
                        <td>{result.face1.features.noseToMouthToInterEye.toFixed(3)}</td>
                        <td>{result.face2.features.noseToMouthToInterEye.toFixed(3)}</td>
                      </tr>
                      <tr>
                        <td>Eye / jaw ratio</td>
                        <td>{result.face1.features.eyeToJawRatio.toFixed(3)}</td>
                        <td>{result.face2.features.eyeToJawRatio.toFixed(3)}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}

export default App
