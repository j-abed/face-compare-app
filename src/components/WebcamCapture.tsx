import { useEffect, useRef, useState, useCallback } from 'react'
import './WebcamCapture.css'

interface WebcamCaptureProps {
  isOpen: boolean
  onClose: () => void
  onCapture: (blobUrl: string) => void
}

export function WebcamCapture({ isOpen, onClose, onCapture }: WebcamCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const stopStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
  }, [])

  useEffect(() => {
    if (!isOpen) return

    setError(null)
    setLoading(true)

    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        streamRef.current = stream
        const video = videoRef.current
        if (video) {
          video.srcObject = stream
          video.onloadedmetadata = () => {
            video.play().catch(() => {})
            setLoading(false)
          }
        } else {
          setLoading(false)
        }
      })
      .catch((err) => {
        setLoading(false)
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
          setError('Camera access was denied. Please allow camera permission and try again.')
        } else if (err.name === 'NotFoundError') {
          setError('No camera was found on this device.')
        } else if (err.name === 'NotReadableError') {
          setError('Camera is in use by another application.')
        } else {
          setError(err.message || 'Failed to access camera.')
        }
      })

    return () => {
      stopStream()
    }
  }, [isOpen, stopStream])

  const handleCapture = useCallback(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || !streamRef.current || video.readyState !== 4) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    ctx.drawImage(video, 0, 0)

    canvas.toBlob(
      (blob) => {
        if (blob) {
          const blobUrl = URL.createObjectURL(blob)
          onCapture(blobUrl)
          stopStream()
          onClose()
        }
      },
      'image/jpeg',
      0.92
    )
  }, [onCapture, onClose, stopStream])

  const handleClose = useCallback(() => {
    stopStream()
    onClose()
  }, [onClose, stopStream])

  if (!isOpen) return null

  return (
    <div className="webcam-overlay" role="dialog" aria-modal="true" aria-labelledby="webcam-title">
      <div className="webcam-backdrop" onClick={handleClose} aria-hidden />
      <div className="webcam-modal">
        <div className="webcam-header">
          <h2 id="webcam-title" className="webcam-title">
            Capture from camera
          </h2>
          <button
            type="button"
            className="webcam-close"
            onClick={handleClose}
            aria-label="Close"
          >
            &times;
          </button>
        </div>

        <div className="webcam-body">
          {error ? (
            <div className="webcam-error">
              <span className="webcam-error-icon">&#9888;&#65039;</span>
              <p>{error}</p>
            </div>
          ) : (
            <>
              <div className="webcam-video-wrap">
                <video
                  ref={videoRef}
                  className="webcam-video"
                  playsInline
                  muted
                  autoPlay
                />
                {loading && (
                  <div className="webcam-loading">
                    <div className="progress-spinner" />
                    <span>Starting camera&hellip;</span>
                  </div>
                )}
              </div>
              <canvas ref={canvasRef} className="webcam-canvas" aria-hidden />
            </>
          )}
        </div>

        <div className="webcam-footer">
          <button
            type="button"
            className="webcam-btn webcam-btn--secondary"
            onClick={handleClose}
          >
            Cancel
          </button>
          <button
            type="button"
            className="webcam-btn webcam-btn--primary"
            onClick={handleCapture}
            disabled={!!error || loading}
            aria-label="Capture photo"
          >
            <span className="webcam-capture-icon">&#128247;</span>
            Capture
          </button>
        </div>
      </div>
    </div>
  )
}
