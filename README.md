# Face Comparison App

A browser-based app that compares two face photos and determines whether they show the same person. Built with **React**, **Vite**, and **ONNX Runtime** (SCRFD face detection + ArcFace embeddings). All processing runs in the browser; no backend or API keys required.

## Features

- **Compare** — Upload two photos (drag-and-drop, file picker, URL, or webcam), then run a face comparison. Results include a verdict (same / different / inconclusive), confidence score, similarity metrics, and optional feature/ratio details.
- **Benchmark** — Run the built-in LFW (Labeled Faces in the Wild) benchmark to measure accuracy, precision, recall, and view ROC/DET curves.
- **Identify** — Compare one probe photo against a gallery of faces (1:N identification).
- **Gallery** — Browse and pick from curated LFW sample images.
- **History** — View past comparisons (stored in IndexedDB).

Additional behavior: face quality checks (blur, pose, exposure), optional aligned overlay view, dark/light theme, and keyboard-accessible UI.

## Tech stack

- **React 19** + **Vite 7** (TypeScript)
- **ONNX Runtime Web** — SCRFD 500MF for detection (5-point landmarks), ArcFace MobileFaceNet (w600k) for 512‑dim embeddings
- No server: models and logic run entirely in the browser

## Setup

### 1. Install dependencies

```bash
npm install
```

### 2. Download ONNX models (required)

Face detection and recognition require two ONNX models. Download them once:

```bash
npm run download-onnx-models
```

This writes to `public/models/onnx/`:

- `det_500m.onnx` — SCRFD detector (~2.4 MB)
- `w600k_mbf.onnx` — ArcFace recognizer (~13 MB)

Without these, the app cannot run comparisons.

### 3. (Optional) LFW dataset for Benchmark and Gallery

To use the **Benchmark** tab and **Gallery** with the full LFW set:

1. Download LFW (pairs + images, ~109 MB):

   ```bash
   npm run download-lfw
   ```

   Files go to `/tmp/lfw-download/` (pairs.txt + extracted `lfw-deepfunneled`).

2. Curate into the project (copies a subset into `public/lfw/` and builds `manifest.json`):

   ```bash
   LFW_SUBDIR=lfw-deepfunneled npm run curate-lfw
   ```

   Run from the **project root**. This creates 40 people and 80 pairs (40 same, 40 different) for the benchmark.

If you skip this, the app still works for **Compare** and **Identify**; Benchmark and Gallery will have no (or minimal) data until LFW is set up.

## Run

```bash
npm run dev
```

Open the URL shown (e.g. http://localhost:5173). Use **Use sample photos** to load two random LFW images (if curated), or upload/paste URLs/use webcam, then click **Compare faces**.

## Build

```bash
npm run build
```

Output is in `dist/`. Test the production build with:

```bash
npm run preview
```

## Scripts

| Script | Description |
|--------|-------------|
| `npm run dev` | Start Vite dev server |
| `npm run build` | TypeScript build + Vite production bundle |
| `npm run preview` | Serve `dist/` locally |
| `npm run lint` | Run ESLint |
| `npm run download-onnx-models` | Download SCRFD + ArcFace ONNX models to `public/models/onnx/` |
| `npm run download-lfw` | Download LFW dataset to `/tmp/lfw-download/` |
| `npm run curate-lfw` | Curate LFW into `public/lfw/` and generate `manifest.json` (run after `download-lfw`; use `LFW_SUBDIR=lfw-deepfunneled` if needed) |

## Project structure

```
├── public/
│   ├── models/onnx/     # SCRFD + ArcFace ONNX models (from download-onnx-models)
│   └── lfw/             # Curated LFW images + manifest.json (from curate-lfw)
├── scripts/
│   ├── download-onnx-models.sh
│   ├── download-lfw.sh
│   ├── curate-lfw.js
│   └── calibrate-onnx.mjs  # Node script to calibrate thresholds on LFW
├── src/
│   ├── ml/              # ONNX loader, SCRFD detector, ArcFace, alignment, quality
│   ├── components/      # WebcamCapture, IdentifyPanel, HistoryPanel, etc.
│   ├── faceComparison.ts # Core compare API and thresholds
│   ├── benchmark.ts      # LFW benchmark runner
│   ├── historyStore.ts   # IndexedDB persistence for history
│   └── App.tsx / App.css
├── docs/                # Additional documentation
└── README.md
```

## Documentation

- [Architecture & models](docs/ARCHITECTURE.md) — Pipeline, models, and calibration
- [Development](docs/DEVELOPMENT.md) — Local dev, benchmarking, and contributing

## License

See repository license file.
