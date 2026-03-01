#!/usr/bin/env bash
# Create sample3, sample4, sample5 from sample1 and sample2 so we have 5 distinct files.
# Run from project root: npm run prepare-samples
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PUBLIC="${SCRIPT_DIR}/../public"
cd "$PUBLIC"

if [[ ! -f sample1.jpg || ! -f sample2.jpg ]]; then
  echo "Need sample1.jpg and sample2.jpg in public/. Run download-samples first or add them manually."
  exit 1
fi

# Create 3 more files (resized so they're distinct and all load)
if command -v sips &>/dev/null; then
  cp sample1.jpg sample3.jpg && sips -Z 240 sample3.jpg
  cp sample2.jpg sample4.jpg && sips -Z 260 sample4.jpg
  cp sample1.jpg sample5.jpg && sips -Z 180 sample5.jpg
  echo "Created sample3.jpg, sample4.jpg, sample5.jpg (resized variants)."
else
  # No sips (e.g. Linux): just copy so all 5 URLs exist and load
  cp sample1.jpg sample3.jpg
  cp sample2.jpg sample4.jpg
  cp sample1.jpg sample5.jpg
  echo "Created sample3.jpg, sample4.jpg, sample5.jpg (copies)."
fi
echo "You now have 5 sample images. Use 'Use sample photos' in the app."
