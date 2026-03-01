# Face comparison app

Upload two photos and compare faces to see if they are the same person. Built with React, Vite, and [face-api.js](https://github.com/justadudewhohacks/face-api.js) (TensorFlow.js). All processing runs in the browser; no backend or API keys required.

## Setup

1. Install dependencies:

   ```bash
   npm install
   ```

2. Download the face-api.js models (required for detection and recognition):

   ```bash
   npm run download-models
   ```

   This places the models in `public/models/`. If you skip this step, the app will fail when you click “Compare faces”.

3. (Optional) Ensure 5 sample photos exist for **Use sample photos** (creates `sample3.jpg`–`sample5.jpg` from the first two):

   ```bash
   npm run prepare-samples
   ```

   You need at least `public/sample1.jpg` and `public/sample2.jpg`; the script creates the rest. Without this, only two sample images are available.

## Run

```bash
npm run dev
```

Open the URL shown (e.g. http://localhost:5173). Use **Use sample photos** to load two random images from the 5 samples (run `npm run prepare-samples` first so all 5 exist), or upload your own images, then click **Compare faces**. You’ll see “Same person” or “Different persons” plus a similarity score. If no face is found in one or both photos, an error message is shown instead.

## Build

```bash
npm run build
```

Output is in `dist/`. Serve with `npm run preview` to test the production build.
