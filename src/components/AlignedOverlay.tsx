import { useState, useEffect, useRef, useCallback } from 'react'
import { alignFace } from '../ml/faceAlign'
import './AlignedOverlay.css'

type ViewMode = 'overlay' | 'slider'

interface AlignedOverlayProps {
  imageUrl1: string
  imageUrl2: string
  landmarks1: [number, number][]
  landmarks2: [number, number][]
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error(`Failed to load image: ${src}`))
    img.src = src
  })
}

export function AlignedOverlay({
  imageUrl1,
  imageUrl2,
  landmarks1,
  landmarks2,
}: AlignedOverlayProps) {
  const [mode, setMode] = useState<ViewMode>('overlay')
  const [opacity, setOpacity] = useState(50)
  const [sliderPos, setSliderPos] = useState(50)
  const [aligned1, setAligned1] = useState<string | null>(null)
  const [aligned2, setAligned2] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const containerRef = useRef<HTMLDivElement>(null)
  const isDraggingRef = useRef(false)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)

    Promise.all([loadImage(imageUrl1), loadImage(imageUrl2)])
      .then(([img1, img2]) => {
        if (cancelled) return
        const canvas1 = alignFace(img1, landmarks1)
        const canvas2 = alignFace(img2, landmarks2)
        setAligned1(canvas1.toDataURL('image/jpeg', 0.92))
        setAligned2(canvas2.toDataURL('image/jpeg', 0.92))
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err))
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [imageUrl1, imageUrl2, landmarks1, landmarks2])

  const handleSliderMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    isDraggingRef.current = true
  }, [])

  const handleSliderTouchStart = useCallback((e: React.TouchEvent) => {
    e.stopPropagation()
    isDraggingRef.current = true
  }, [])

  useEffect(() => {
    const handleMove = (e: MouseEvent | TouchEvent) => {
      if (!isDraggingRef.current || !containerRef.current) return
      const clientX =
        'touches' in e && e.touches.length > 0
          ? e.touches[0].clientX
          : (e as MouseEvent).clientX
      const rect = containerRef.current.getBoundingClientRect()
      const x = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100))
      setSliderPos(x)
    }

    const handleEnd = () => {
      isDraggingRef.current = false
    }

    window.addEventListener('mousemove', handleMove)
    window.addEventListener('mouseup', handleEnd)
    window.addEventListener('touchmove', handleMove, { passive: true })
    window.addEventListener('touchend', handleEnd)

    return () => {
      window.removeEventListener('mousemove', handleMove)
      window.removeEventListener('mouseup', handleEnd)
      window.removeEventListener('touchmove', handleMove)
      window.removeEventListener('touchend', handleEnd)
    }
  }, [])

  const handleContainerClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current || mode !== 'slider') return
    const rect = containerRef.current.getBoundingClientRect()
    const x = Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100))
    setSliderPos(x)
  }, [mode])

  if (loading) {
    return (
      <div className="aligned-overlay aligned-overlay--loading">
        <div className="progress-spinner" />
        <span>Aligning faces&hellip;</span>
      </div>
    )
  }

  if (error || !aligned1 || !aligned2) {
    return (
      <div className="aligned-overlay aligned-overlay--error">
        {error ?? 'Failed to align faces'}
      </div>
    )
  }

  return (
    <div className="aligned-overlay">
      <div className="aligned-overlay-mode-toggle">
        <button
          type="button"
          className={`aligned-overlay-mode-btn${mode === 'overlay' ? ' aligned-overlay-mode-btn--active' : ''}`}
          onClick={() => setMode('overlay')}
        >
          Overlay
        </button>
        <button
          type="button"
          className={`aligned-overlay-mode-btn${mode === 'slider' ? ' aligned-overlay-mode-btn--active' : ''}`}
          onClick={() => setMode('slider')}
        >
          Slider
        </button>
      </div>

      <div
        ref={containerRef}
        className={`aligned-overlay-view aligned-overlay-view--${mode}`}
        onClick={mode === 'slider' ? handleContainerClick : undefined}
        role={mode === 'slider' ? 'button' : undefined}
        tabIndex={mode === 'slider' ? 0 : undefined}
        onKeyDown={
          mode === 'slider'
            ? (e) => {
                if (e.key === 'ArrowLeft') setSliderPos((p) => Math.max(0, p - 4))
                if (e.key === 'ArrowRight') setSliderPos((p) => Math.min(100, p + 4))
              }
            : undefined
        }
      >
        <div className="aligned-overlay-canvas-wrap">
          {mode === 'overlay' ? (
            <>
              <img
                src={aligned1}
                alt="Aligned face 1"
                className="aligned-overlay-img aligned-overlay-img--base"
              />
              <img
                src={aligned2}
                alt="Aligned face 2"
                className="aligned-overlay-img aligned-overlay-img--overlay"
                style={{ opacity: opacity / 100 }}
              />
            </>
          ) : (
            <>
              <img
                src={aligned1}
                alt="Aligned face 1"
                className="aligned-overlay-img aligned-overlay-img--left"
                style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
              />
              <img
                src={aligned2}
                alt="Aligned face 2"
                className="aligned-overlay-img aligned-overlay-img--right"
                style={{ clipPath: `inset(0 0 0 ${sliderPos}%)` }}
              />
              <div
                className="aligned-overlay-handle"
                style={{ left: `${sliderPos}%` }}
                onMouseDown={handleSliderMouseDown}
                onTouchStart={handleSliderTouchStart}
                role="slider"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(sliderPos)}
                aria-label="Before/after position"
              >
                <div className="aligned-overlay-handle-line" />
                <div className="aligned-overlay-handle-grab" />
              </div>
            </>
          )}
        </div>
      </div>

      {mode === 'overlay' && (
        <div className="aligned-overlay-opacity-control">
          <label htmlFor="aligned-opacity" className="aligned-overlay-opacity-label">
            Photo 2 opacity: {opacity}%
          </label>
          <input
            id="aligned-opacity"
            type="range"
            min={0}
            max={100}
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
            className="aligned-overlay-opacity-slider"
          />
        </div>
      )}

      <div className="aligned-overlay-sidebyside">
        <div className="aligned-overlay-thumb">
          <img src={aligned1} alt="Aligned face 1" />
          <span>Photo 1</span>
        </div>
        <div className="aligned-overlay-thumb">
          <img src={aligned2} alt="Aligned face 2" />
          <span>Photo 2</span>
        </div>
      </div>
    </div>
  )
}
