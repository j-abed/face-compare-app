import * as faceapi from 'face-api.js';

const MODEL_BASE = '/models';
const SAME_PERSON_THRESHOLD = 0.56;
const COSINE_GATE = 0.92;
const MIN_FACE_PIXELS = 40;
const UPSCALE_MIN = 200;

let modelsLoaded = false;

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load image: ${src}`));
    img.src = src;
  });
}

export async function loadModels(): Promise<void> {
  if (modelsLoaded) return;
  await Promise.all([
    faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_BASE),
    faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_BASE),
    faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_BASE),
  ]);
  modelsLoaded = true;
}

export type CompareProgressStep =
  | 'loading_models'
  | 'loading_images'
  | 'detecting_face_1'
  | 'detecting_face_2'
  | 'comparing';

export interface ProgressPayload {
  annotatedImage1?: string;
  annotatedImage2?: string;
  featureRegions1?: FeatureRegions;
  featureRegions2?: FeatureRegions;
  imageDimensions1?: { width: number; height: number };
  imageDimensions2?: { width: number; height: number };
}

export interface FeatureRegions {
  eyes: { left: { x: number; y: number }[]; right: { x: number; y: number }[] };
  nose: { x: number; y: number }[];
  mouth: { x: number; y: number }[];
  jaw?: { x: number; y: number }[];
}

export interface FaceFeatures {
  detectionConfidence: number;
  boxWidth: number;
  boxHeight: number;
  aspectRatio: number;
  interEyeDistance: number;
  noseLength: number;
  mouthWidth: number;
  faceWidth: number;
  faceHeight: number;
  eyeAspectRatio: number;
  jawWidth: number;
  noseToMouthDistance: number;
  eyebrowSlopeLeft: number;
  eyebrowSlopeRight: number;
  mouthHeight: number;
  noseToInterEye: number;
  mouthToInterEye: number;
  jawToInterEye: number;
  faceAspectRatio: number;
  mouthAspectRatio: number;
  noseToMouthToInterEye: number;
  eyeToJawRatio: number;
}

function pointDist(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return Math.hypot(b.x - a.x, b.y - a.y);
}

function centerPoint(points: Array<{ x: number; y: number }>): { x: number; y: number } {
  const n = points.length;
  const x = points.reduce((s, p) => s + p.x, 0) / n;
  const y = points.reduce((s, p) => s + p.y, 0) / n;
  return { x, y };
}

function eyeAspect(points: Array<{ x: number; y: number }>): number {
  if (points.length < 2) return 0;
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const width = Math.max(...xs) - Math.min(...xs);
  const height = Math.max(...ys) - Math.min(...ys);
  if (height === 0) return 0;
  return width / height;
}

function eyebrowSlope(points: Array<{ x: number; y: number }>): number {
  if (points.length < 2) return 0;
  const first = points[0];
  const last = points[points.length - 1];
  const dx = last.x - first.x;
  const dy = last.y - first.y;
  if (dx === 0) return 0;
  return dy / dx;
}

function extractFeatures(
  detection: faceapi.FaceDetection,
  landmarks: faceapi.FaceLandmarks68
): FaceFeatures {
  const box = detection.box;
  const w = box.width;
  const h = box.height;
  const leftEye = landmarks.getLeftEye();
  const rightEye = landmarks.getRightEye();
  const nose = landmarks.getNose();
  const mouth = landmarks.getMouth();
  const jaw = landmarks.getJawOutline();
  const leftBrow = landmarks.getLeftEyeBrow();
  const rightBrow = landmarks.getRightEyeBrow();
  const leftCenter = centerPoint(leftEye.map((p) => ({ x: p.x, y: p.y })));
  const rightCenter = centerPoint(rightEye.map((p) => ({ x: p.x, y: p.y })));
  const interEye = pointDist(leftCenter, rightCenter);
  const noseYs = nose.map((p) => p.y);
  const noseLength = Math.max(...noseYs) - Math.min(...noseYs);
  const mouthXs = mouth.map((p) => p.x);
  const mouthYs = mouth.map((p) => p.y);
  const mouthWidth = Math.max(...mouthXs) - Math.min(...mouthXs);
  const mouthHeight = Math.max(...mouthYs) - Math.min(...mouthYs);
  const noseBottom = Math.max(...noseYs);
  const mouthTop = Math.min(...mouthYs);
  const noseToMouth = mouthTop - noseBottom;
  const leftEyeAr = eyeAspect(leftEye.map((p) => ({ x: p.x, y: p.y })));
  const rightEyeAr = eyeAspect(rightEye.map((p) => ({ x: p.x, y: p.y })));
  const eyeAspectRatio = (leftEyeAr + rightEyeAr) / 2;
  const jawXs = jaw.map((p) => p.x);
  const jawWidth = Math.max(...jawXs) - Math.min(...jawXs);
  const eyebrowSlopeLeft = eyebrowSlope(leftBrow.map((p) => ({ x: p.x, y: p.y })));
  const eyebrowSlopeRight = eyebrowSlope(rightBrow.map((p) => ({ x: p.x, y: p.y })));
  const safeInterEye = interEye > 0 ? interEye : 1;
  return {
    detectionConfidence: detection.score,
    boxWidth: w,
    boxHeight: h,
    aspectRatio: w / h,
    interEyeDistance: interEye,
    noseLength,
    mouthWidth,
    faceWidth: w,
    faceHeight: h,
    eyeAspectRatio,
    jawWidth,
    noseToMouthDistance: noseToMouth,
    eyebrowSlopeLeft,
    eyebrowSlopeRight,
    mouthHeight,
    noseToInterEye: noseLength / safeInterEye,
    mouthToInterEye: mouthWidth / safeInterEye,
    jawToInterEye: jawWidth / safeInterEye,
    faceAspectRatio: w / h,
    mouthAspectRatio: mouthHeight > 0 ? mouthWidth / mouthHeight : 0,
    noseToMouthToInterEye: noseToMouth / safeInterEye,
    eyeToJawRatio: interEye / (jawWidth > 0 ? jawWidth : 1),
  };
}

export type Verdict = 'same' | 'different' | 'inconclusive';

export interface CompareResult {
  samePerson: boolean;
  verdict: Verdict;
  score: number;
  error?: string;
  euclideanDistance?: number;
  cosineSimilarity?: number;
  geometricSimilarity?: number;
  landmarkAlignment?: number;
  confidence?: number;
  confidenceLabel?: string;
  qualityWarnings?: string[];
  progressSteps?: CompareProgressStep[];
  annotatedImage1?: string;
  annotatedImage2?: string;
  face1?: {
    detection: { score: number; box: { x: number; y: number; width: number; height: number } };
    features: FaceFeatures;
  };
  face2?: {
    detection: { score: number; box: { x: number; y: number; width: number; height: number } };
    features: FaceFeatures;
  };
}

// ---------------------------------------------------------------------------
// Image manipulation helpers
// ---------------------------------------------------------------------------

type FaceInput = HTMLImageElement | HTMLCanvasElement;

function imgWidth(img: FaceInput): number {
  return img instanceof HTMLImageElement ? img.naturalWidth : img.width;
}
function imgHeight(img: FaceInput): number {
  return img instanceof HTMLImageElement ? img.naturalHeight : img.height;
}

function flipImageHorizontally(img: FaceInput): HTMLCanvasElement {
  const w = imgWidth(img);
  const h = imgHeight(img);
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d')!;
  ctx.translate(w, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(img, 0, 0);
  return canvas;
}

function enhanceBrightness(img: FaceInput, factor: number): HTMLCanvasElement {
  const w = imgWidth(img);
  const h = imgHeight(img);
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0);
  ctx.globalCompositeOperation = 'lighter';
  const alpha = Math.min(1, factor - 1);
  ctx.fillStyle = `rgba(128,128,128,${alpha})`;
  ctx.fillRect(0, 0, w, h);
  ctx.globalCompositeOperation = 'source-over';
  return canvas;
}

// ---------------------------------------------------------------------------
// Robust face detection
// ---------------------------------------------------------------------------

type DetectionResult = faceapi.WithFaceDescriptor<
  faceapi.WithFaceLandmarks<{ detection: faceapi.FaceDetection }>
>;

/**
 * Extract only the descriptor from a cropped + upscaled version of the face
 * region. Returns null if re-detection fails on the crop.
 */
async function extractDescriptorFromCrop(
  img: FaceInput,
  box: faceapi.Box,
): Promise<Float32Array | null> {
  const iw = imgWidth(img);
  const ih = imgHeight(img);

  const pad = Math.max(box.width, box.height) * 0.6;
  const cx = Math.max(0, box.x - pad);
  const cy = Math.max(0, box.y - pad);
  const cw = Math.min(iw - cx, box.width + pad * 2);
  const ch = Math.min(ih - cy, box.height + pad * 2);

  const minDim = Math.min(box.width, box.height);
  const scale = Math.max(1, UPSCALE_MIN / minDim);

  const canvas = document.createElement('canvas');
  canvas.width = Math.round(cw * scale);
  canvas.height = Math.round(ch * scale);
  const ctx = canvas.getContext('2d')!;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, cx, cy, cw, ch, 0, 0, canvas.width, canvas.height);

  const fi = canvas as unknown as HTMLImageElement;
  for (const minConfidence of [0.3, 0.15]) {
    const opts = new faceapi.SsdMobilenetv1Options({ minConfidence });
    const result = await faceapi
      .detectSingleFace(fi, opts)
      .withFaceLandmarks()
      .withFaceDescriptor();
    if (result) return result.descriptor;
  }
  return null;
}

/**
 * Robust face detection pipeline:
 *  1. Try progressively lower confidence thresholds.
 *  2. Fall back to detectAllFaces and pick the largest face.
 *  3. Try brightness-enhanced copy as a last resort.
 *
 * When a face is found but is small (< UPSCALE_MIN px), we crop + upscale
 * and re-extract only the descriptor. The detection box and landmarks
 * remain in the original image coordinate space so annotations stay correct.
 */
async function detectFaceRobust(
  img: FaceInput,
): Promise<DetectionResult | null> {
  const fi = img as unknown as HTMLImageElement;
  const thresholds = [0.5, 0.35, 0.2];

  for (const minConfidence of thresholds) {
    const opts = new faceapi.SsdMobilenetv1Options({ minConfidence });
    const result = await faceapi
      .detectSingleFace(fi, opts)
      .withFaceLandmarks()
      .withFaceDescriptor();
    if (result) return await maybeUpscaleDescriptor(img, result);
  }

  const allFaces = await faceapi
    .detectAllFaces(fi, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.15 }))
    .withFaceLandmarks()
    .withFaceDescriptors();

  if (allFaces.length > 0) {
    let best = allFaces[0];
    let bestArea = best.detection.box.width * best.detection.box.height;
    for (let i = 1; i < allFaces.length; i++) {
      const area = allFaces[i].detection.box.width * allFaces[i].detection.box.height;
      if (area > bestArea) {
        best = allFaces[i];
        bestArea = area;
      }
    }
    return await maybeUpscaleDescriptor(img, best);
  }

  const enhanced = enhanceBrightness(img, 1.5);
  const efi = enhanced as unknown as HTMLImageElement;
  for (const minConfidence of [0.3, 0.15]) {
    const opts = new faceapi.SsdMobilenetv1Options({ minConfidence });
    const result = await faceapi
      .detectSingleFace(efi, opts)
      .withFaceLandmarks()
      .withFaceDescriptor();
    if (result) return await maybeUpscaleDescriptor(img, result);
  }

  return null;
}

/**
 * If the detected face is small, crop + upscale to get a better descriptor
 * while keeping the original box/landmarks untouched for visualization.
 */
async function maybeUpscaleDescriptor(
  img: FaceInput,
  result: DetectionResult,
): Promise<DetectionResult> {
  const faceSize = Math.min(result.detection.box.width, result.detection.box.height);
  if (faceSize >= UPSCALE_MIN) return result;

  const betterDesc = await extractDescriptorFromCrop(img, result.detection.box);
  if (!betterDesc) return result;

  // Return a new object with the better descriptor but original box/landmarks
  return {
    detection: result.detection,
    landmarks: result.landmarks,
    descriptor: betterDesc,
    unshiftedLandmarks: result.unshiftedLandmarks,
    alignedRect: result.alignedRect,
  } as DetectionResult;
}

// ---------------------------------------------------------------------------
// Augmented descriptor (original + flip averaged)
// ---------------------------------------------------------------------------

async function getAugmentedDescriptor(
  img: FaceInput,
): Promise<Float32Array | null> {
  const result = await detectFaceRobust(img);
  if (!result) return null;

  const desc = result.descriptor;

  try {
    const flipped = flipImageHorizontally(img);
    const flippedResult = await detectFaceRobust(flipped);
    if (flippedResult) {
      const avg = new Float32Array(desc.length);
      for (let i = 0; i < desc.length; i++) {
        avg[i] = (desc[i] + flippedResult.descriptor[i]) / 2;
      }
      return avg;
    }
  } catch {
    // flip augmentation failed, use original
  }

  return desc;
}

// ---------------------------------------------------------------------------
// Annotation & regions
// ---------------------------------------------------------------------------

function drawAnnotatedCanvas(
  img: HTMLImageElement,
  result: { detection: faceapi.FaceDetection; landmarks: faceapi.FaceLandmarks68 }
): string {
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0);
  const displaySize = { width: img.naturalWidth, height: img.naturalHeight };
  const resized = faceapi.resizeResults(result, displaySize);
  faceapi.draw.drawDetections(canvas, resized);
  faceapi.draw.drawFaceLandmarks(canvas, resized.landmarks);
  return canvas.toDataURL('image/jpeg', 0.92);
}

function buildFeatureRegions(landmarks: faceapi.FaceLandmarks68): FeatureRegions {
  const toPoints = (points: Array<{ x: number; y: number }>) =>
    points.map((p) => ({ x: p.x, y: p.y }));
  return {
    eyes: {
      left: toPoints(landmarks.getLeftEye().map((p) => ({ x: p.x, y: p.y }))),
      right: toPoints(landmarks.getRightEye().map((p) => ({ x: p.x, y: p.y }))),
    },
    nose: toPoints(landmarks.getNose().map((p) => ({ x: p.x, y: p.y }))),
    mouth: toPoints(landmarks.getMouth().map((p) => ({ x: p.x, y: p.y }))),
    jaw: toPoints(landmarks.getJawOutline().map((p) => ({ x: p.x, y: p.y }))),
  };
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

function distanceToScore(distance: number): number {
  const sigma = 0.60;
  return Math.exp(-(distance * distance) / (2 * sigma * sigma));
}

function cosineSim(a: Float32Array, b: Float32Array): number {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  return denom === 0 ? 0 : dot / denom;
}

function geometricSimilarity(f1: FaceFeatures, f2: FaceFeatures): number {
  const ratios: [number, number][] = [
    [f1.noseToInterEye, f2.noseToInterEye],
    [f1.mouthToInterEye, f2.mouthToInterEye],
    [f1.jawToInterEye, f2.jawToInterEye],
    [f1.faceAspectRatio, f2.faceAspectRatio],
    [f1.mouthAspectRatio, f2.mouthAspectRatio],
    [f1.noseToMouthToInterEye, f2.noseToMouthToInterEye],
    [f1.eyeToJawRatio, f2.eyeToJawRatio],
    [f1.eyeAspectRatio, f2.eyeAspectRatio],
  ];
  let sum = 0;
  let count = 0;
  for (const [a, b] of ratios) {
    const maxVal = Math.max(Math.abs(a), Math.abs(b));
    if (maxVal === 0) continue;
    sum += 1 - Math.abs(a - b) / maxVal;
    count++;
  }
  const raw = count > 0 ? sum / count : 0;
  const floor = 0.85;
  return Math.max(0, Math.min(1, (raw - floor) / (1 - floor)));
}

function procrustesDistance(
  landmarks1: faceapi.FaceLandmarks68,
  landmarks2: faceapi.FaceLandmarks68,
): number {
  const pts1 = landmarks1.positions.map(p => ({ x: p.x, y: p.y }));
  const pts2 = landmarks2.positions.map(p => ({ x: p.x, y: p.y }));
  const n = Math.min(pts1.length, pts2.length);
  if (n < 3) return 0;

  const centroid = (pts: { x: number; y: number }[]) => {
    let sx = 0, sy = 0;
    for (const p of pts) { sx += p.x; sy += p.y; }
    return { x: sx / pts.length, y: sy / pts.length };
  };
  const c1 = centroid(pts1);
  const c2 = centroid(pts2);

  const a = pts1.map(p => ({ x: p.x - c1.x, y: p.y - c1.y }));
  const b = pts2.map(p => ({ x: p.x - c2.x, y: p.y - c2.y }));

  const norm = (pts: { x: number; y: number }[]) => {
    let s = 0;
    for (const p of pts) s += p.x * p.x + p.y * p.y;
    return Math.sqrt(s);
  };
  const n1 = norm(a) || 1;
  const n2 = norm(b) || 1;
  for (let i = 0; i < n; i++) { a[i].x /= n1; a[i].y /= n1; }
  for (let i = 0; i < n; i++) { b[i].x /= n2; b[i].y /= n2; }

  let sxx = 0, sxy = 0, syx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    sxx += a[i].x * b[i].x;
    sxy += a[i].x * b[i].y;
    syx += a[i].y * b[i].x;
    syy += a[i].y * b[i].y;
  }
  const num = syx - sxy;
  const den = sxx + syy;
  const theta = Math.atan2(num, den);
  const cosT = Math.cos(theta);
  const sinT = Math.sin(theta);

  let sse = 0;
  for (let i = 0; i < n; i++) {
    const rx = cosT * b[i].x - sinT * b[i].y;
    const ry = sinT * b[i].x + cosT * b[i].y;
    const dx = a[i].x - rx;
    const dy = a[i].y - ry;
    sse += dx * dx + dy * dy;
  }
  const mse = sse / n;
  const midpoint = 0.003;
  const steepness = 800;
  return 1 / (1 + Math.exp(steepness * (mse - midpoint)));
}

// ---------------------------------------------------------------------------
// Confidence
// ---------------------------------------------------------------------------

function computeConfidence(
  detConf1: number,
  detConf2: number,
  distance: number,
  cosine: number,
): { confidence: number; label: string } {
  const detectionQuality = Math.min(detConf1, detConf2);

  const gap = Math.abs(distance - SAME_PERSON_THRESHOLD);
  const sigma = 0.20;
  const decisiveness = 1 - Math.exp(-(gap * gap) / (2 * sigma * sigma));

  const isSame = distance < SAME_PERSON_THRESHOLD;
  const cosineStrength = isSame
    ? Math.min(1, cosine / 0.95)
    : Math.min(1, (1 - cosine) / 0.25);

  const cosineSaysSame = cosine > COSINE_GATE;
  const signalsAgree = isSame === cosineSaysSame ? 1.0 : 0.3 + 0.4 * (1 - Math.abs(distance - SAME_PERSON_THRESHOLD) / 0.3);

  const confidence =
    detectionQuality * 0.15 +
    decisiveness * 0.45 +
    cosineStrength * 0.20 +
    signalsAgree * 0.20;

  const clamped = Math.max(0, Math.min(1, confidence));
  let label: string;
  if (clamped >= 0.90) label = 'Very high';
  else if (clamped >= 0.75) label = 'High';
  else if (clamped >= 0.55) label = 'Moderate';
  else if (clamped >= 0.35) label = 'Low';
  else label = 'Very low';
  return { confidence: clamped, label };
}

// ---------------------------------------------------------------------------
// Main comparison
// ---------------------------------------------------------------------------

export async function compareFaces(
  image1Url: string,
  image2Url: string,
  onProgress?: (step: CompareProgressStep, payload?: ProgressPayload) => void
): Promise<CompareResult> {
  const steps: CompareProgressStep[] = [];
  const sameUrl = image1Url === image2Url;

  onProgress?.('loading_models');
  steps.push('loading_models');
  await loadModels();

  onProgress?.('loading_images');
  steps.push('loading_images');
  const img1 = await loadImage(image1Url);
  const img2 = sameUrl ? img1 : await loadImage(image2Url);

  onProgress?.('detecting_face_1');
  steps.push('detecting_face_1');
  const result1 = await detectFaceRobust(img1);

  let annotatedImage1: string | undefined;
  if (result1) {
    try {
      annotatedImage1 = drawAnnotatedCanvas(img1, { detection: result1.detection, landmarks: result1.landmarks });
      onProgress?.('detecting_face_1', { annotatedImage1 });
    } catch {
      // ignore
    }
  }

  let result2: typeof result1;
  let annotatedImage2: string | undefined;

  if (sameUrl) {
    result2 = result1;
    annotatedImage2 = annotatedImage1;
    onProgress?.('detecting_face_2', { annotatedImage2 });
  } else {
    onProgress?.('detecting_face_2');
    steps.push('detecting_face_2');
    result2 = await detectFaceRobust(img2);

    if (result2) {
      try {
        annotatedImage2 = drawAnnotatedCanvas(img2, { detection: result2.detection, landmarks: result2.landmarks });
        onProgress?.('detecting_face_2', { annotatedImage2 });
      } catch {
        // ignore
      }
    }
  }

  if (!result1) {
    return { samePerson: false, verdict: 'inconclusive', score: 0, error: 'No face detected in photo 1.', progressSteps: steps };
  }
  if (!result2) {
    return { samePerson: false, verdict: 'inconclusive', score: 0, error: 'No face detected in photo 2.', progressSteps: steps };
  }

  const featureRegions1 = buildFeatureRegions(result1.landmarks);
  const featureRegions2 = buildFeatureRegions(result2.landmarks);
  onProgress?.('comparing', {
    annotatedImage1: annotatedImage1 ?? drawAnnotatedCanvas(img1, { detection: result1.detection, landmarks: result1.landmarks }),
    annotatedImage2: annotatedImage2 ?? drawAnnotatedCanvas(img2, { detection: result2.detection, landmarks: result2.landmarks }),
    featureRegions1,
    featureRegions2,
    imageDimensions1: { width: img1.naturalWidth, height: img1.naturalHeight },
    imageDimensions2: { width: img2.naturalWidth, height: img2.naturalHeight },
  });
  steps.push('comparing');

  let desc1 = result1.descriptor as Float32Array;
  let desc2 = result2.descriptor as Float32Array;
  if (!sameUrl) {
    const [aug1, aug2] = await Promise.all([
      getAugmentedDescriptor(img1),
      getAugmentedDescriptor(img2),
    ]);
    if (aug1) desc1 = aug1;
    if (aug2) desc2 = aug2;
  }

  const distance = sameUrl ? 0 : faceapi.euclideanDistance(desc1, desc2);
  const cosine = sameUrl ? 1 : cosineSim(desc1, desc2);

  const features1 = extractFeatures(result1.detection, result1.landmarks);
  const features2 = extractFeatures(result2.detection, result2.landmarks);

  const geoSim = sameUrl ? 1 : geometricSimilarity(features1, features2);
  const landmarkAlign = sameUrl ? 1 : procrustesDistance(result1.landmarks, result2.landmarks);
  const descriptorScore = distanceToScore(distance);

  const score = descriptorScore * 0.75 + landmarkAlign * 0.15 + geoSim * 0.10;

  // Quality warnings
  const qualityWarnings: string[] = [];
  const box1 = result1.detection.box;
  const box2 = result2.detection.box;
  if (Math.min(box1.width, box1.height) < MIN_FACE_PIXELS) {
    qualityWarnings.push('Face in photo 1 is very small — results may be unreliable.');
  }
  if (Math.min(box2.width, box2.height) < MIN_FACE_PIXELS) {
    qualityWarnings.push('Face in photo 2 is very small — results may be unreliable.');
  }
  if (result1.detection.score < 0.85) {
    qualityWarnings.push('Low detection confidence in photo 1.');
  }
  if (result2.detection.score < 0.85) {
    qualityWarnings.push('Low detection confidence in photo 2.');
  }

  // Dual-gate: require both distance AND cosine to agree for "same"
  const distanceSaysSame = distance < SAME_PERSON_THRESHOLD;
  const cosineSaysSame = cosine > COSINE_GATE;
  const samePerson = distanceSaysSame && cosineSaysSame;

  const { confidence, label: confidenceLabel } = sameUrl
    ? { confidence: 1, label: 'Very high' }
    : computeConfidence(
        result1.detection.score,
        result2.detection.score,
        distance,
        cosine,
      );

  let verdict: Verdict;
  if (sameUrl) {
    verdict = 'same';
  } else if (distanceSaysSame !== cosineSaysSame) {
    verdict = 'inconclusive';
  } else if (confidence < 0.25) {
    verdict = 'inconclusive';
  } else {
    verdict = samePerson ? 'same' : 'different';
  }

  if (!annotatedImage1) {
    try {
      annotatedImage1 = drawAnnotatedCanvas(img1, { detection: result1.detection, landmarks: result1.landmarks });
    } catch {
      // ignore
    }
  }
  if (!annotatedImage2) {
    try {
      annotatedImage2 = drawAnnotatedCanvas(img2, { detection: result2.detection, landmarks: result2.landmarks });
    } catch {
      // ignore
    }
  }

  const boxToPlain = (box: faceapi.Box) => ({
    x: box.x,
    y: box.y,
    width: box.width,
    height: box.height,
  });

  return {
    samePerson,
    verdict,
    score,
    euclideanDistance: distance,
    cosineSimilarity: cosine,
    geometricSimilarity: geoSim,
    landmarkAlignment: landmarkAlign,
    confidence,
    confidenceLabel,
    qualityWarnings: qualityWarnings.length > 0 ? qualityWarnings : undefined,
    progressSteps: steps,
    annotatedImage1,
    annotatedImage2,
    face1: {
      detection: { score: result1.detection.score, box: boxToPlain(result1.detection.box) },
      features: features1,
    },
    face2: {
      detection: { score: result2.detection.score, box: boxToPlain(result2.detection.box) },
      features: features2,
    },
  };
}

export function isModelsLoaded(): boolean {
  return modelsLoaded;
}
