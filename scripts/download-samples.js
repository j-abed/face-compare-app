#!/usr/bin/env node
/**
 * Downloads 3 additional sample face images into public/.
 * Run from project root: node scripts/download-samples.js
 * Requires network access.
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const publicDir = path.join(__dirname, '..', 'public');

const SAMPLES = [
  { file: 'sample3.jpg', url: 'https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=300' },
  { file: 'sample4.jpg', url: 'https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=300' },
  { file: 'sample5.jpg', url: 'https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?w=300' },
];

async function download() {
  for (const { file, url } of SAMPLES) {
    try {
      const res = await fetch(url, {
        headers: { 'User-Agent': 'FaceCompareApp/1.0 (sample photo fetcher)' },
      });
      if (!res.ok) throw new Error(`${res.status}`);
      const buf = await res.arrayBuffer();
      fs.writeFileSync(path.join(publicDir, file), Buffer.from(buf));
      console.log(`Wrote ${file}`);
    } catch (err) {
      console.error(`Failed ${file}: ${err.message}`);
    }
  }
  console.log('Done. Run the app and click "Use sample photos" to pick 2 from 5.');
}

download();
