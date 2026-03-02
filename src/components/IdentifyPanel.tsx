import { useState, useCallback, useRef, useEffect, type DragEvent } from 'react';
import { identifyFace, type IdentifyResult } from '../batchCompare';
import './IdentifyPanel.css';

function ProbeDropZone({
  imageUrl,
  fileName,
  onFile,
  onRemove,
}: {
  imageUrl: string | null;
  fileName: string | null;
  onFile: (file: File) => void;
  onRemove: () => void;
}) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files?.[0];
      if (file && file.type.startsWith('image/')) {
        onFile(file);
      }
    },
    [onFile]
  );

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
  }, []);

  const handleClick = useCallback(() => {
    if (!imageUrl) {
      inputRef.current?.click();
    }
  }, [imageUrl]);

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onFile(file);
      e.target.value = '';
    },
    [onFile]
  );

  const dzClass = [
    'identify-dropzone',
    'identify-dropzone--probe',
    dragOver && 'identify-dropzone--dragover',
    imageUrl && 'identify-dropzone--has-file',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <div className="identify-upload-cell">
      <label className="identify-upload-label">Probe (face to identify)</label>
      <div
        className={dzClass}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={handleClick}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={handleInputChange}
        />
        {imageUrl ? (
          <div className="identify-dropzone-preview">
            <img src={imageUrl} alt="Probe" className="identify-preview-img" />
            <div className="identify-dropzone-footer">
              <span className="identify-dropzone-filename">{fileName ?? 'Image'}</span>
              <button
                className="identify-dropzone-remove"
                onClick={(e) => {
                  e.stopPropagation();
                  onRemove();
                }}
                aria-label="Remove probe"
              >
                &times;
              </button>
            </div>
          </div>
        ) : (
          <>
            <span className="identify-dropzone-icon">&#128247;</span>
            <span className="identify-dropzone-text">
              Drop probe photo here or <em>click to browse</em>
            </span>
          </>
        )}
      </div>
    </div>
  );
}

function GalleryDropZone({
  files,
  onFiles,
  onRemove,
}: {
  files: File[];
  onFiles: (files: File[]) => void;
  onRemove: (index: number) => void;
}) {
  const [dragOver, setDragOver] = useState(false);
  const [galleryUrls, setGalleryUrls] = useState<string[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const urls = files.map((f) => URL.createObjectURL(f));
    setGalleryUrls(urls);
    return () => urls.forEach((u) => URL.revokeObjectURL(u));
  }, [files]);

  const handleDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const dropped = Array.from(e.dataTransfer.files ?? []).filter((f) =>
        f.type.startsWith('image/')
      );
      if (dropped.length > 0) {
        onFiles([...files, ...dropped]);
      }
    },
    [files, onFiles]
  );

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
  }, []);

  const handleClick = useCallback(() => {
    inputRef.current?.click();
  }, []);

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selected = Array.from(e.target.files ?? []).filter((f) =>
        f.type.startsWith('image/')
      );
      if (selected.length > 0) {
        onFiles([...files, ...selected]);
      }
      e.target.value = '';
    },
    [files, onFiles]
  );

  const dzClass = [
    'identify-dropzone',
    'identify-dropzone--gallery',
    dragOver && 'identify-dropzone--dragover',
    files.length > 0 && 'identify-dropzone--has-files',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <div className="identify-upload-cell">
      <label className="identify-upload-label">Gallery (candidates to match against)</label>
      <div
        className={dzClass}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={handleClick}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          multiple
          onChange={handleInputChange}
        />
        {files.length > 0 ? (
          <div className="identify-gallery-thumbs">
            {galleryUrls.map((url, i) => (
              <div key={`${url}-${i}`} className="identify-gallery-thumb-wrap">
                <img
                  src={url}
                  alt={files[i]?.name ?? ''}
                  className="identify-gallery-thumb"
                />
                <button
                  className="identify-gallery-thumb-remove"
                  onClick={(e) => {
                    e.stopPropagation();
                    onRemove(i);
                  }}
                  aria-label={`Remove ${files[i]?.name ?? 'image'}`}
                >
                  &times;
                </button>
              </div>
            ))}
          </div>
        ) : (
          <>
            <span className="identify-dropzone-icon">&#128444;&#65039;</span>
            <span className="identify-dropzone-text">
              Drop gallery photos here or <em>click to browse</em> (multiple)
            </span>
          </>
        )}
      </div>
    </div>
  );
}

export default function IdentifyPanel() {
  const [probeUrl, setProbeUrl] = useState<string | null>(null);
  const [probeFile, setProbeFile] = useState<File | null>(null);
  const [galleryFiles, setGalleryFiles] = useState<File[]>([]);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState<{ current: number; total: number } | null>(null);
  const [results, setResults] = useState<IdentifyResult[] | null>(null);
  const runUrlsRef = useRef<string[]>([]);

  const handleProbeFile = useCallback((file: File) => {
    setProbeUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(file);
    });
    setProbeFile(file);
    setResults(null);
  }, []);

  const removeProbe = useCallback(() => {
    if (probeUrl) URL.revokeObjectURL(probeUrl);
    setProbeUrl(null);
    setProbeFile(null);
    setResults(null);
  }, [probeUrl]);

  const handleGalleryFiles = useCallback((files: File[]) => {
    setGalleryFiles(files);
    setResults(null);
  }, []);

  const removeGalleryFile = useCallback((index: number) => {
    setGalleryFiles((prev) => prev.filter((_, i) => i !== index));
    setResults(null);
  }, []);

  const runIdentification = useCallback(async () => {
    if (!probeUrl || galleryFiles.length === 0) return;
    runUrlsRef.current.forEach((u) => URL.revokeObjectURL(u));
    runUrlsRef.current = [];
    const urls = galleryFiles.map((f) => URL.createObjectURL(f));
    runUrlsRef.current = urls;
    setLoading(true);
    setProgress({ current: 0, total: urls.length });
    setResults(null);
    try {
      const res = await identifyFace(probeUrl, urls, (current, total) => {
        setProgress({ current, total });
      });
      setResults(res);
    } catch (err) {
      console.error(err);
      setResults([]);
    } finally {
      setLoading(false);
      setProgress(null);
    }
  }, [probeUrl, galleryFiles]);

  useEffect(
    () => () => runUrlsRef.current.forEach((u) => URL.revokeObjectURL(u)),
    []
  );

  const canRun = probeUrl && galleryFiles.length > 0 && !loading;

  return (
    <div className="identify-panel tab-content">
      <h2 className="identify-title">1:N Face Identification</h2>
      <p className="identify-subtitle">
        Upload a probe face and a gallery of candidates. Results are ranked by match score.
      </p>

      <div className="identify-upload-row">
        <ProbeDropZone
          imageUrl={probeUrl}
          fileName={probeFile?.name ?? null}
          onFile={handleProbeFile}
          onRemove={removeProbe}
        />
        <GalleryDropZone
          files={galleryFiles}
          onFiles={handleGalleryFiles}
          onRemove={removeGalleryFile}
        />
      </div>

      <div className="identify-actions">
        <button
          className="identify-run-btn"
          onClick={runIdentification}
          disabled={!canRun}
        >
          {loading ? 'Identifying…' : 'Run Identification'}
        </button>
      </div>

      {loading && progress && (
        <div className="identify-progress">
          <div className="identify-progress-header">
            <span className="identify-progress-label">Comparing probe to gallery…</span>
            <span className="identify-progress-count">
              {progress.current} / {progress.total}
            </span>
          </div>
          <div className="identify-progress-bar">
            <div
              className="identify-progress-fill"
              style={{ width: `${(progress.current / progress.total) * 100}%` }}
            />
          </div>
        </div>
      )}

      {results && results.length > 0 && (
        <div className="identify-results">
          <h3 className="identify-results-title">Results (ranked by match score)</h3>
          <div className="identify-results-grid">
            {results.map((r, i) => (
              <div
                key={r.url}
                className={`identify-result-card${i === 0 ? ' identify-result-card--top' : ''}`}
              >
                <img src={r.url} alt="" className="identify-result-thumb" loading="lazy" />
                <div className="identify-result-meta">
                  <span className="identify-result-score">
                    {(r.score * 100).toFixed(1)}%
                  </span>
                  <span className="identify-result-cosine">
                    cosine: {r.cosine.toFixed(4)}
                  </span>
                  <span
                    className={`identify-result-verdict identify-result-verdict--${r.verdict}`}
                  >
                    {r.verdict}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {results && results.length === 0 && !loading && (
        <div className="identify-no-results">No results (error or empty gallery).</div>
      )}
    </div>
  );
}
