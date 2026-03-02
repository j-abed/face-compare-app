#!/usr/bin/env bash
# Download LFW dataset for benchmark expansion.
# Run this before: npm run curate-lfw
# Requires: curl, tar

set -euo pipefail

BASE="http://vis-www.cs.umass.edu/lfw"
DEST="/tmp/lfw-download"
TGZ="$DEST/lfw-deepfunneled.tgz"
mkdir -p "$DEST"

echo "Downloading pairs.txt..."
curl -sL "$BASE/pairs.txt" -o "$DEST/pairs.txt"
echo "  OK ($(wc -c < "$DEST/pairs.txt") bytes)"

echo "Downloading lfw-deepfunneled.tgz (~109 MB, may take a few minutes)..."
if [ -f "$TGZ" ]; then
  SIZE=$(wc -c < "$TGZ")
  if [ "$SIZE" -gt 100000000 ]; then
    echo "  Found existing file ($((SIZE / 1024 / 1024)) MB), skipping download."
  else
    rm -f "$TGZ"
    curl -# -L "$BASE/lfw-deepfunneled.tgz" -o "$TGZ"
  fi
else
  curl -# -L "$BASE/lfw-deepfunneled.tgz" -o "$TGZ"
fi
SIZE=$(wc -c < "$TGZ")
echo "  OK ($((SIZE / 1024 / 1024)) MB)"

echo "Extracting..."
tar -xzf "$TGZ" -C "$DEST"

if [ ! -d "$DEST/lfw-deepfunneled" ]; then
  echo "ERROR: Expected directory $DEST/lfw-deepfunneled after extract. Contents of $DEST:"
  ls -la "$DEST"
  exit 1
fi
COUNT=$(find "$DEST/lfw-deepfunneled" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "  OK ($COUNT person directories)"

echo ""
echo "Done. To curate with expanded benchmark (40 people, 80 pairs), run:"
echo "  LFW_SUBDIR=lfw-deepfunneled npm run curate-lfw"
