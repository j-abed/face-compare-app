#!/usr/bin/env bash
# Download additional sample face photos into public/
set -e
PUBLIC="$(dirname "$0")/../public"
cd "$PUBLIC"

# Sample face image URLs (Unsplash, small size for testing)
curl -sSL -o sample3.jpg "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=200"
curl -sSL -o sample4.jpg "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=200"
curl -sSL -o sample5.jpg "https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?w=200"

echo "Done. Sample photos are in $PUBLIC"
