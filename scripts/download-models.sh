#!/usr/bin/env bash
# Download face-api.js models from GitHub into public/models/
set -e
BASE="https://raw.githubusercontent.com/justadudewhohacks/face-api.js-models/master"
MODELS_DIR="$(dirname "$0")/../public/models"
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

echo "Downloading ssd_mobilenetv1..."
curl -sSLO "$BASE/ssd_mobilenetv1/ssd_mobilenetv1_model-weights_manifest.json"
curl -sSLO "$BASE/ssd_mobilenetv1/ssd_mobilenetv1_model-shard1"
curl -sSLO "$BASE/ssd_mobilenetv1/ssd_mobilenetv1_model-shard2"

echo "Downloading face_landmark_68..."
curl -sSLO "$BASE/face_landmark_68/face_landmark_68_model-weights_manifest.json"
curl -sSLO "$BASE/face_landmark_68/face_landmark_68_model-shard1"

echo "Downloading face_recognition..."
curl -sSLO "$BASE/face_recognition/face_recognition_model-weights_manifest.json"
curl -sSLO "$BASE/face_recognition/face_recognition_model-shard1"
curl -sSLO "$BASE/face_recognition/face_recognition_model-shard2"

echo "Done. Models are in $MODELS_DIR"
