#!/usr/bin/env bash
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/public/models/onnx"
mkdir -p "$DEST"

# InsightFace buffalo_sc model pack (small + accurate)
# SCRFD-500MF detector (~2.4 MB) + MobileFaceNet ArcFace w600k (~13 MB)
BASE="https://huggingface.co/jungseoik/ai_genportrait/resolve/main/buffalo_sc"

if [ ! -f "$DEST/det_500m.onnx" ]; then
  echo "Downloading SCRFD det_500m.onnx (~2.4 MB)..."
  curl -L "$BASE/det_500m.onnx" -o "$DEST/det_500m.onnx"
else
  echo "det_500m.onnx already exists, skipping."
fi

if [ ! -f "$DEST/w600k_mbf.onnx" ]; then
  echo "Downloading ArcFace w600k_mbf.onnx (~13 MB)..."
  curl -L "$BASE/w600k_mbf.onnx" -o "$DEST/w600k_mbf.onnx"
else
  echo "w600k_mbf.onnx already exists, skipping."
fi

echo "Done. Models saved to $DEST"
ls -lh "$DEST"
