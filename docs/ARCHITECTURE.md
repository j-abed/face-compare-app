# Architecture & models

This document describes the face comparison pipeline, ONNX models, and how results are computed.

## Pipeline overview

1. **Load models** — SCRFD detector and ArcFace recognizer are loaded via ONNX Runtime Web (WASM). They live in `public/models/onnx/`.
2. **Load images** — User provides two images (upload, URL, or webcam). Images are decoded and optionally compressed for speed.
3. **Detect faces** — SCRFD runs on each image. The implementation uses progressive confidence thresholds and optional crop-and-upscale for small faces. Each detection yields a bounding box, score, and 5 landmarks (eyes, nose, mouth corners).
4. **Align & crop** — A 5-point similarity transform maps the detected face to a canonical 112×112 template. This aligned crop is the input to ArcFace.
5. **Embed** — ArcFace produces a 512-dimensional descriptor. Optionally, a horizontally flipped version is embedded and the two vectors are averaged (test-time augmentation).
6. **Compare** — Cosine similarity and Euclidean distance between the two descriptors are computed. Geometric similarity (facial ratios) and optional landmark alignment supplement the decision.
7. **Decide** — Verdict is “same”, “different”, or “inconclusive” using calibrated thresholds and a confidence score.

All steps run in the browser; no data is sent to a server.

## Models

| Model | File | Role |
|-------|------|------|
| SCRFD 500MF | `det_500m.onnx` | Face detection + 5-point landmarks. Input: RGB image (e.g. 640×640). Output: boxes, scores, landmarks. |
| ArcFace MobileFaceNet | `w600k_mbf.onnx` | Face recognition. Input: aligned 112×112 face. Output: 512-d embedding (L2-normalized). |

Source: [InsightFace buffalo_sc](https://huggingface.co/jungseoik/ai_genportrait) (Hugging Face). Downloaded via `npm run download-onnx-models`.

## Thresholds and calibration

- **Cosine similarity** — “Same” if cosine ≥ `COSINE_SAME_THRESHOLD` (default 0.16). Calibrated on LFW (see `public/lfw/calibration.json` and `scripts/calibrate-onnx.mjs`).
- **Euclidean distance** — “Same” if L2 distance ≤ `DISTANCE_SAME_THRESHOLD` (default 1.245). Also calibrated on LFW.
- **Confidence** — Derived from detection scores, distance of cosine to threshold, and geometric agreement. High confidence for clear same/different; lower for borderline or low-quality detections.

To re-calibrate thresholds (e.g. after changing models or LFW set):

```bash
node scripts/calibrate-onnx.mjs
```

This reads `public/lfw/manifest.json` and writes recommended values to `public/lfw/calibration.json`. The app reads thresholds from code; update `faceComparison.ts` if you adopt new values from calibration.

## Key source files

| Path | Purpose |
|------|---------|
| `src/ml/onnxLoader.ts` | Load SCRFD and ArcFace sessions (browser). |
| `src/ml/scrfdDetector.ts` | Run SCRFD, NMS, return boxes + 5 landmarks. |
| `src/ml/arcfaceEmbed.ts` | Run ArcFace on aligned face, optional TTA. |
| `src/ml/faceAlign.ts` | 5-point similarity transform → 112×112 crop. |
| `src/ml/faceQuality.ts` | Blur, pose, exposure, occlusion checks. |
| `src/faceComparison.ts` | Orchestrates detect → align → embed → compare, applies thresholds, computes confidence. |
| `src/benchmark.ts` | Runs LFW pairs through `compareFaces`, aggregates accuracy/ROC/DET. |
| `src/lfwData.ts` | Load manifest, resolve LFW image paths. |

## Data flow

- **Compare tab** — Two images → `compareFaces()` → `CompareResult` (verdict, score, confidence, metrics, optional face/feature data).
- **Benchmark tab** — `manifest.json` same/different pairs → for each pair `compareFaces()` → aggregate into `BenchmarkResults` (accuracy, F1, ROC, etc.).
- **Identify tab** — One probe image + gallery list → repeated `compareFaces(probe, gallery[i])` → rank by similarity.
- **History** — Each comparison can be saved to IndexedDB via `historyStore.ts`; History tab reads from the same store.

## LFW dataset

The app can use a curated subset of [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) for:

- **Benchmark** — Fixed set of same/different pairs to measure accuracy.
- **Gallery** — Browsable sample photos and “Use sample photos” in Compare.

Scripts:

1. `npm run download-lfw` — Fetches LFW to `/tmp/lfw-download/`.
2. `LFW_SUBDIR=lfw-deepfunneled npm run curate-lfw` — Copies a subset into `public/lfw/` and generates `public/lfw/manifest.json` (people, same pairs, different pairs).

The curate script is configurable (e.g. number of people/pairs) via constants in `scripts/curate-lfw.js`.
