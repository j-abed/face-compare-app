# Development

This document covers local development, running the benchmark, and contributing.

## Prerequisites

- **Node.js** 18+ (recommend 20+)
- **npm** (or compatible package manager)

## Local development

1. Clone the repo and install dependencies:

   ```bash
   git clone https://github.com/j-abed/face-compare-app.git
   cd face-compare-app
   npm install
   ```

2. Download ONNX models (required for Compare):

   ```bash
   npm run download-onnx-models
   ```

3. Start the dev server:

   ```bash
   npm run dev
   ```

   Open the URL in the browser (e.g. http://localhost:5173). Use Compare with uploads/URLs/webcam, or run the Benchmark if LFW is set up.

## LFW and benchmark

To work on benchmark accuracy or use the Gallery with real LFW data:

1. Download LFW (one-time, ~109 MB):

   ```bash
   npm run download-lfw
   ```

2. Curate into the project (from project root):

   ```bash
   LFW_SUBDIR=lfw-deepfunneled npm run curate-lfw
   ```

   This populates `public/lfw/` with images and `manifest.json`. The Benchmark tab will run 80 pairs (40 same, 40 different) by default.

3. (Optional) Calibrate thresholds on the current LFW set:

   ```bash
   node scripts/calibrate-onnx.mjs
   ```

   This updates `public/lfw/calibration.json`. To use the suggested values in the app, copy them into `src/faceComparison.ts` (e.g. `COSINE_SAME_THRESHOLD`, `DISTANCE_SAME_THRESHOLD`).

## Linting and type-checking

- **Lint:** `npm run lint`
- **Type-check:** `npx tsc --noEmit` (uses project `tsconfig.json`)

Fix lint/type errors before submitting changes.

## Project layout (summary)

- `src/` — React app and comparison/benchmark logic
  - `ml/` — ONNX loader, SCRFD, ArcFace, alignment, quality
  - `components/` — WebcamCapture, IdentifyPanel, HistoryPanel, etc.
  - `faceComparison.ts`, `benchmark.ts`, `historyStore.ts`, `lfwData.ts`
- `public/` — Static assets; `models/onnx/` and `lfw/` populated by scripts
- `scripts/` — Download and curate scripts; `calibrate-onnx.mjs` for threshold tuning
- `docs/` — This file and `ARCHITECTURE.md`

## Contributing

1. Open an issue or pick an existing one.
2. Create a branch, make changes, run `npm run lint` and `npx tsc --noEmit`.
3. Test Compare (and Benchmark if LFW is available) in the browser.
4. Open a PR with a short description and reference to the issue.

For deeper changes (e.g. new models, threshold logic), see [ARCHITECTURE.md](ARCHITECTURE.md).
