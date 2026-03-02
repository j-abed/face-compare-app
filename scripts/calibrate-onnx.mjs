import ort from 'onnxruntime-node';
import sharp from 'sharp';
import { readFileSync } from 'fs';

const SIZE = 112;
const INPUT_SIZE = 640;
const STRIDES = [8, 16, 32];
const NUM_ANCHORS = 2;

const TEMPLATE = [
  [38.2946, 51.6963], [73.5318, 51.5014],
  [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041],
];

function cosine(a, b) {
  let dot = 0, ma = 0, mb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i]*b[i]; ma += a[i]*a[i]; mb += b[i]*b[i]; }
  return dot / (Math.sqrt(ma) * Math.sqrt(mb));
}

function euclidean(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; sum += d * d; }
  return Math.sqrt(sum);
}

function l2Normalize(vec) {
  let sum = 0;
  for (let i = 0; i < vec.length; i++) sum += vec[i]*vec[i];
  const norm = Math.sqrt(sum) || 1;
  return vec.map(v => v / norm);
}

function estimateTransform(src, dst) {
  const n = src.length;
  let sx = 0, sy = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) { sx += src[i][0]; sy += src[i][1]; dx += dst[i][0]; dy += dst[i][1]; }
  sx/=n; sy/=n; dx/=n; dy/=n;
  let srcVar = 0, a = 0, b = 0;
  for (let i = 0; i < n; i++) {
    const sxx = src[i][0]-sx, syy = src[i][1]-sy, dxx = dst[i][0]-dx, dyy = dst[i][1]-dy;
    srcVar += sxx*sxx + syy*syy;
    a += sxx*dxx + syy*dyy;
    b += sxx*dyy - syy*dxx;
  }
  a /= srcVar; b /= srcVar;
  return [a, -b, dx - a*sx + b*sy, b, a, dy - a*sy - b*sx];
}

async function detect(session, imagePath) {
  const img = sharp(imagePath);
  const meta = await img.metadata();
  const ratio = Math.min(INPUT_SIZE / meta.width, INPUT_SIZE / meta.height);
  const newW = Math.round(meta.width * ratio);
  const newH = Math.round(meta.height * ratio);
  const resized = await img.resize(newW, newH).removeAlpha().raw().toBuffer();
  const float32 = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  for (let y = 0; y < newH; y++) {
    for (let x = 0; x < newW; x++) {
      const si = (y*newW + x) * 3;
      const di = y*INPUT_SIZE + x;
      float32[di] = (resized[si] - 127.5) / 128.0;
      float32[INPUT_SIZE*INPUT_SIZE + di] = (resized[si+1] - 127.5) / 128.0;
      float32[2*INPUT_SIZE*INPUT_SIZE + di] = (resized[si+2] - 127.5) / 128.0;
    }
  }
  const tensor = new ort.Tensor('float32', float32, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const results = await session.run({ [session.inputNames[0]]: tensor });
  const outputNames = session.outputNames;
  const fmc = STRIDES.length;
  let bestScore = -1, bestKps = null;
  for (let si = 0; si < fmc; si++) {
    const stride = STRIDES[si];
    const scores = results[outputNames[si]].data;
    const bboxes = results[outputNames[si + fmc]].data;
    const kps = results[outputNames[si + fmc * 2]].data;
    const featH = Math.ceil(INPUT_SIZE / stride);
    const featW = Math.ceil(INPUT_SIZE / stride);
    for (let row = 0; row < featH; row++) {
      for (let col = 0; col < featW; col++) {
        for (let a = 0; a < NUM_ANCHORS; a++) {
          const idx = (row * featW + col) * NUM_ANCHORS + a;
          if (scores[idx] < 0.3) continue;
          const cx = col * stride, cy = row * stride;
          const kp = [];
          for (let k = 0; k < 5; k++) {
            kp.push([(cx + kps[idx*10 + k*2]*stride) / ratio, (cy + kps[idx*10 + k*2 + 1]*stride) / ratio]);
          }
          if (scores[idx] > bestScore) { bestScore = scores[idx]; bestKps = kp; }
        }
      }
    }
  }
  return bestKps;
}

async function embed(recSession, imagePath, landmarks) {
  const M = estimateTransform(landmarks, TEMPLATE);
  // Use sharp to apply the affine transform
  const meta = await sharp(imagePath).metadata();
  const raw = await sharp(imagePath).removeAlpha().raw().toBuffer();

  const aligned = Buffer.alloc(SIZE * SIZE * 3);
  for (let dy = 0; dy < SIZE; dy++) {
    for (let dx = 0; dx < SIZE; dx++) {
      // Inverse transform: canvas (dx,dy) -> source (sx,sy)
      const det = M[0]*M[4] - M[1]*M[3];
      const sx = (M[4]*(dx - M[2]) - M[1]*(dy - M[5])) / det;
      const sy = (-M[3]*(dx - M[2]) + M[0]*(dy - M[5])) / det;
      const ix = Math.round(sx), iy = Math.round(sy);
      if (ix >= 0 && ix < meta.width && iy >= 0 && iy < meta.height) {
        const si = (iy * meta.width + ix) * 3;
        const di = (dy * SIZE + dx) * 3;
        aligned[di] = raw[si]; aligned[di+1] = raw[si+1]; aligned[di+2] = raw[si+2];
      }
    }
  }

  const float32 = new Float32Array(3 * SIZE * SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    float32[i] = (aligned[i*3] - 127.5) / 128.0;
    float32[SIZE*SIZE + i] = (aligned[i*3 + 1] - 127.5) / 128.0;
    float32[2*SIZE*SIZE + i] = (aligned[i*3 + 2] - 127.5) / 128.0;
  }

  const tensor = new ort.Tensor('float32', float32, [1, 3, SIZE, SIZE]);
  const result = await recSession.run({ [recSession.inputNames[0]]: tensor });
  return l2Normalize(new Float32Array(result[recSession.outputNames[0]].data));
}

async function main() {
  const detSession = await ort.InferenceSession.create('public/models/onnx/det_500m.onnx');
  const recSession = await ort.InferenceSession.create('public/models/onnx/w600k_mbf.onnx');
  const manifest = JSON.parse(readFileSync('public/lfw/manifest.json', 'utf8'));

  const cache = new Map();
  async function getEmb(imgPath) {
    if (cache.has(imgPath)) return cache.get(imgPath);
    const fullPath = 'public' + imgPath;
    const kps = await detect(detSession, fullPath);
    if (!kps) { console.log('No face:', fullPath); return null; }
    const emb = await embed(recSession, fullPath, kps);
    cache.set(imgPath, emb);
    return emb;
  }

  let total = 0;
  const sameCosines = [], sameDistances = [];
  const diffCosines = [], diffDistances = [];

  for (const pair of manifest.pairs.same) {
    const emb1 = await getEmb(pair.image1);
    const emb2 = await getEmb(pair.image2);
    if (!emb1 || !emb2) continue;
    sameCosines.push(cosine(emb1, emb2));
    sameDistances.push(euclidean(emb1, emb2));
    total++;
  }

  for (const pair of manifest.pairs.different) {
    const emb1 = await getEmb(pair.image1);
    const emb2 = await getEmb(pair.image2);
    if (!emb1 || !emb2) continue;
    diffCosines.push(cosine(emb1, emb2));
    diffDistances.push(euclidean(emb1, emb2));
    total++;
  }

  const nSame = sameCosines.length;
  const nDiff = diffCosines.length;
  console.log(`\n=== LFW Benchmark (ONNX, ${total} pairs: ${nSame} same, ${nDiff} different) ===`);

  console.log(`\nSame-person cosines (${nSame}): min=${Math.min(...sameCosines).toFixed(4)} max=${Math.max(...sameCosines).toFixed(4)} mean=${(sameCosines.reduce((a,b)=>a+b,0)/nSame).toFixed(4)}`);
  console.log(`Same-person distances (${nSame}): min=${Math.min(...sameDistances).toFixed(4)} max=${Math.max(...sameDistances).toFixed(4)} mean=${(sameDistances.reduce((a,b)=>a+b,0)/nSame).toFixed(4)}`);
  console.log(`Diff-person cosines (${nDiff}): min=${Math.min(...diffCosines).toFixed(4)} max=${Math.max(...diffCosines).toFixed(4)} mean=${(diffCosines.reduce((a,b)=>a+b,0)/nDiff).toFixed(4)}`);
  console.log(`Diff-person distances (${nDiff}): min=${Math.min(...diffDistances).toFixed(4)} max=${Math.max(...diffDistances).toFixed(4)} mean=${(diffDistances.reduce((a,b)=>a+b,0)/nDiff).toFixed(4)}`);

  // Optimal cosine threshold: same if cos > T
  const allCos = [...sameCosines.map(c => ({ v: c, same: true })), ...diffCosines.map(c => ({ v: c, same: false }))];
  let bestAccCos = 0, bestThreshCos = 0;
  for (let t = -0.5; t <= 1; t += 0.01) {
    const acc = allCos.filter(x => x.same ? x.v > t : x.v <= t).length / allCos.length;
    if (acc > bestAccCos) { bestAccCos = acc; bestThreshCos = t; }
  }
  console.log(`\nOptimal cosine threshold: ${bestThreshCos.toFixed(2)} (same if cos > threshold) -> accuracy ${(100*bestAccCos).toFixed(1)}%`);

  // Optimal distance threshold: same if distance < T
  const allDist = [...sameDistances.map(d => ({ v: d, same: true })), ...diffDistances.map(d => ({ v: d, same: false }))];
  const distRange = [...new Set(sameDistances.concat(diffDistances))].sort((a, b) => a - b);
  let bestAccDist = 0, bestThreshDist = 1.3;
  for (let i = 0; i < distRange.length - 1; i++) {
    const t = (distRange[i] + distRange[i + 1]) / 2;
    const acc = allDist.filter(x => x.same ? x.v < t : x.v >= t).length / allDist.length;
    if (acc > bestAccDist) { bestAccDist = acc; bestThreshDist = t; }
  }
  console.log(`Optimal distance threshold: ${bestThreshDist.toFixed(3)} (same if distance < threshold) -> accuracy ${(100*bestAccDist).toFixed(1)}%`);

  // Combined: predict same only when both cosine and distance agree
  let correctBoth = 0;
  for (let i = 0; i < allCos.length; i++) {
    const cosSaySame = allCos[i].v > bestThreshCos;
    const distSaySame = allDist[i].v < bestThreshDist;
    const predictedSame = cosSaySame && distSaySame;
    if ((allCos[i].same && predictedSame) || (!allCos[i].same && !predictedSame)) correctBoth++;
  }
  console.log(`\nUsing both thresholds (agree): ${correctBoth}/${allCos.length} = ${(100*correctBoth/allCos.length).toFixed(1)}%`);

  // Write recommended thresholds for app
  const rec = {
    COSINE_SAME_THRESHOLD: Math.round(bestThreshCos * 100) / 100,
    DISTANCE_SAME_THRESHOLD: Math.round(bestThreshDist * 1000) / 1000,
    accuracyCosineOnly: Math.round(100 * bestAccCos) / 100,
    accuracyDistanceOnly: Math.round(100 * bestAccDist) / 100,
    pairsTotal: total,
    pairsSame: nSame,
    pairsDifferent: nDiff,
  };
  const { writeFileSync } = await import('fs');
  const outPath = 'public/lfw/calibration.json';
  writeFileSync(outPath, JSON.stringify(rec, null, 2));
  console.log(`\nWrote ${outPath} - update src/faceComparison.ts with COSINE_SAME_THRESHOLD and DISTANCE_SAME_THRESHOLD`);
}

main().catch(console.error);
