import { loadOnnxModels, isLoaded } from './ml/onnxLoader';
import { detectFaceRobust, type DetectedFace } from './ml/scrfdDetector';
import { getEmbedding } from './ml/arcfaceEmbed';
import { assessFaceQuality, cropFaceToCanvas } from './ml/faceQuality';

// ArcFace 512-dim thresholds (calibrated via: node scripts/calibrate-onnx.mjs)
// Uses public/lfw/calibration.json from last run (80 pairs: 40 same, 40 different).
const COSINE_SAME_THRESHOLD = 0.16;
const DISTANCE_SAME_THRESHOLD = 1.245;
const MIN_FACE_PIXELS = 40;

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
  qualityWarnings1?: string[];
  qualityWarnings2?: string[];
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
  landmarks1?: [number, number][];
  landmarks2?: [number, number][];
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load image: ${src}`));
    img.src = src;
  });
}

const DEFAULT_MAX_DIM = 1024;

function resizeImageIfNeeded(
  img: HTMLImageElement,
  maxDim: number = DEFAULT_MAX_DIM
): HTMLCanvasElement | HTMLImageElement {
  const w = img.naturalWidth;
  const h = img.naturalHeight;
  const max = Math.max(w, h);
  if (max <= maxDim) return img;

  const scale = maxDim / max;
  const nw = Math.round(w * scale);
  const nh = Math.round(h * scale);

  const canvas = document.createElement('canvas');
  canvas.width = nw;
  canvas.height = nh;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0, nw, nh);
  return canvas;
}

function getImageDimensions(img: HTMLImageElement | HTMLCanvasElement): { w: number; h: number } {
  if (img instanceof HTMLCanvasElement) {
    return { w: img.width, h: img.height };
  }
  return { w: img.naturalWidth, h: img.naturalHeight };
}

function drawAnnotatedCanvas(
  img: HTMLImageElement | HTMLCanvasElement,
  face: DetectedFace,
): string {
  const { w, h } = getImageDimensions(img);
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0);

  const { box, landmarks5 } = face;
  ctx.strokeStyle = '#00e5ff';
  ctx.lineWidth = Math.max(2, Math.round(w / 300));
  ctx.strokeRect(box.x, box.y, box.width, box.height);

  const dotRadius = Math.max(2, Math.round(w / 200));
  const colors = ['#60a5fa', '#60a5fa', '#fbbf24', '#f472b6', '#f472b6'];
  for (let i = 0; i < landmarks5.length; i++) {
    ctx.fillStyle = colors[i];
    ctx.beginPath();
    ctx.arc(landmarks5[i][0], landmarks5[i][1], dotRadius, 0, Math.PI * 2);
    ctx.fill();
  }

  return canvas.toDataURL('image/jpeg', 0.92);
}

function buildFeatureRegions(landmarks5: [number, number][]): FeatureRegions {
  const [le, re, nose, lm, rm] = landmarks5;
  return {
    eyes: { left: [{ x: le[0], y: le[1] }], right: [{ x: re[0], y: re[1] }] },
    nose: [{ x: nose[0], y: nose[1] }],
    mouth: [{ x: lm[0], y: lm[1] }, { x: rm[0], y: rm[1] }],
  };
}

function pointDist(a: [number, number], b: [number, number]): number {
  return Math.hypot(b[0] - a[0], b[1] - a[1]);
}

function extractFeatures(face: DetectedFace): FaceFeatures {
  const { box, score, landmarks5 } = face;
  const [le, re, nose, lm, rm] = landmarks5;
  const interEye = pointDist(le, re);
  const mouthWidth = pointDist(lm, rm);
  const noseLength = Math.abs(nose[1] - (le[1] + re[1]) / 2);
  const noseToMouth = Math.abs((lm[1] + rm[1]) / 2 - nose[1]);
  const safeInterEye = interEye > 0 ? interEye : 1;
  const approxJawWidth = interEye * 2.2;

  return {
    detectionConfidence: score,
    boxWidth: box.width,
    boxHeight: box.height,
    aspectRatio: box.width / box.height,
    interEyeDistance: interEye,
    noseLength,
    mouthWidth,
    faceWidth: box.width,
    faceHeight: box.height,
    eyeAspectRatio: 0,
    jawWidth: approxJawWidth,
    noseToMouthDistance: noseToMouth,
    eyebrowSlopeLeft: 0,
    eyebrowSlopeRight: 0,
    mouthHeight: 0,
    noseToInterEye: noseLength / safeInterEye,
    mouthToInterEye: mouthWidth / safeInterEye,
    jawToInterEye: approxJawWidth / safeInterEye,
    faceAspectRatio: box.width / box.height,
    mouthAspectRatio: 0,
    noseToMouthToInterEye: noseToMouth / safeInterEye,
    eyeToJawRatio: interEye / (approxJawWidth > 0 ? approxJawWidth : 1),
  };
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

function euclideanDist(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; sum += d * d; }
  return Math.sqrt(sum);
}

function geometricSimilarity(f1: FaceFeatures, f2: FaceFeatures): number {
  const ratios: [number, number][] = [
    [f1.noseToInterEye, f2.noseToInterEye],
    [f1.mouthToInterEye, f2.mouthToInterEye],
    [f1.jawToInterEye, f2.jawToInterEye],
    [f1.faceAspectRatio, f2.faceAspectRatio],
    [f1.noseToMouthToInterEye, f2.noseToMouthToInterEye],
    [f1.eyeToJawRatio, f2.eyeToJawRatio],
  ];
  let sum = 0, count = 0;
  for (const [a, b] of ratios) {
    const maxVal = Math.max(Math.abs(a), Math.abs(b));
    if (maxVal === 0) continue;
    sum += 1 - Math.abs(a - b) / maxVal;
    count++;
  }
  const raw = count > 0 ? sum / count : 0;
  return Math.max(0, Math.min(1, (raw - 0.85) / 0.15));
}

function computeConfidence(
  detConf1: number, detConf2: number, cosine: number, distance: number,
): { confidence: number; label: string } {
  const detQuality = Math.min(detConf1, detConf2);
  const gap = Math.abs(cosine - COSINE_SAME_THRESHOLD);
  // Use smaller sigma (0.08) so clearly different pairs (gap ~0.1) get high decisiveness
  const decisiveness = 1 - Math.exp(-(gap * gap) / (2 * 0.08 * 0.08));
  // For "different": use 0.08 divisor so cosine 0.07 → (0.16-0.07)/0.08 = 1.0 (clearly different)
  // For "same": keep 0.4 divisor so cosine 0.56 → 1.0
  const cosineStrength = cosine > COSINE_SAME_THRESHOLD
    ? Math.min(1, (cosine - COSINE_SAME_THRESHOLD) / 0.4)
    : Math.min(1, (COSINE_SAME_THRESHOLD - cosine) / 0.08);

  const distSaysSame = distance < DISTANCE_SAME_THRESHOLD;
  const cosSaysSame = cosine > COSINE_SAME_THRESHOLD;
  const signalsAgree = distSaysSame === cosSaysSame ? 1.0 : 0.3;

  const raw = detQuality * 0.10 + decisiveness * 0.40 + cosineStrength * 0.30 + signalsAgree * 0.20;
  const clamped = Math.max(0, Math.min(1, raw));

  let label: string;
  if (clamped >= 0.90) label = 'Very high';
  else if (clamped >= 0.75) label = 'High';
  else if (clamped >= 0.55) label = 'Moderate';
  else if (clamped >= 0.35) label = 'Low';
  else label = 'Very low';
  return { confidence: clamped, label };
}

export async function compareFaces(
  image1Url: string,
  image2Url: string,
  onProgress?: (step: CompareProgressStep, payload?: ProgressPayload) => void,
): Promise<CompareResult> {
  const steps: CompareProgressStep[] = [];
  const sameUrl = image1Url === image2Url;

  onProgress?.('loading_models');
  steps.push('loading_models');
  await loadOnnxModels();

  onProgress?.('loading_images');
  steps.push('loading_images');
  const img1Raw = await loadImage(image1Url);
  const img2Raw = sameUrl ? img1Raw : await loadImage(image2Url);
  const img1 = resizeImageIfNeeded(img1Raw);
  const img2 = sameUrl ? img1 : resizeImageIfNeeded(img2Raw);

  onProgress?.('detecting_face_1');
  steps.push('detecting_face_1');
  const face1 = await detectFaceRobust(img1);
  let annotatedImage1: string | undefined;
  if (face1) {
    annotatedImage1 = drawAnnotatedCanvas(img1, face1);
    onProgress?.('detecting_face_1', { annotatedImage1 });
  }

  let face2: DetectedFace | null;
  let annotatedImage2: string | undefined;
  if (sameUrl) {
    face2 = face1;
    annotatedImage2 = annotatedImage1;
    onProgress?.('detecting_face_2', { annotatedImage2 });
  } else {
    onProgress?.('detecting_face_2');
    steps.push('detecting_face_2');
    face2 = await detectFaceRobust(img2);
    if (face2) {
      annotatedImage2 = drawAnnotatedCanvas(img2, face2);
      onProgress?.('detecting_face_2', { annotatedImage2 });
    }
  }

  if (!face1) return { samePerson: false, verdict: 'inconclusive', score: 0, error: 'No face detected in photo 1.', progressSteps: steps };
  if (!face2) return { samePerson: false, verdict: 'inconclusive', score: 0, error: 'No face detected in photo 2.', progressSteps: steps };

  const featureRegions1 = buildFeatureRegions(face1.landmarks5);
  const featureRegions2 = buildFeatureRegions(face2.landmarks5);
  const dim1 = getImageDimensions(img1);
  const dim2 = getImageDimensions(img2);
  onProgress?.('comparing', {
    annotatedImage1, annotatedImage2,
    featureRegions1, featureRegions2,
    imageDimensions1: { width: dim1.w, height: dim1.h },
    imageDimensions2: { width: dim2.w, height: dim2.h },
  });
  steps.push('comparing');

  let emb1: Float32Array, emb2: Float32Array;
  if (sameUrl) {
    emb1 = await getEmbedding(img1, face1.landmarks5, false);
    emb2 = emb1;
  } else {
    emb1 = await getEmbedding(img1, face1.landmarks5);
    emb2 = await getEmbedding(img2, face2.landmarks5);
  }

  const distance = sameUrl ? 0 : euclideanDist(emb1, emb2);
  const cosine = sameUrl ? 1 : cosineSim(emb1, emb2);
  const features1 = extractFeatures(face1);
  const features2 = extractFeatures(face2);
  const geoSim = sameUrl ? 1 : geometricSimilarity(features1, features2);

  // ArcFace score: map cosine [-1, 1] to [0, 1]
  const score = sameUrl ? 1 : Math.max(0, Math.min(1, (cosine + 1) / 2));

  const qualityWarnings: string[] = [];
  let qualityWarnings1: string[] = [];
  let qualityWarnings2: string[] = [];

  if (Math.min(face1.box.width, face1.box.height) < MIN_FACE_PIXELS) {
    qualityWarnings.push('Face in photo 1 is very small — results may be unreliable.');
    qualityWarnings1.push('Face is very small — results may be unreliable.');
  }
  if (Math.min(face2.box.width, face2.box.height) < MIN_FACE_PIXELS) {
    qualityWarnings.push('Face in photo 2 is very small — results may be unreliable.');
    qualityWarnings2.push('Face is very small — results may be unreliable.');
  }
  if (face1.score < 0.85) {
    qualityWarnings.push('Low detection confidence in photo 1.');
    qualityWarnings1.push('Low detection confidence.');
  }
  if (face2.score < 0.85) {
    qualityWarnings.push('Low detection confidence in photo 2.');
    qualityWarnings2.push('Low detection confidence.');
  }

  const faceCrop1 = cropFaceToCanvas(img1, face1.box);
  const faceCrop2 = sameUrl ? faceCrop1 : cropFaceToCanvas(img2, face2.box);
  const q1 = assessFaceQuality(faceCrop1, face1.box, face1.landmarks5, 'Photo 1');
  const q2 = sameUrl ? q1 : assessFaceQuality(faceCrop2, face2.box, face2.landmarks5, 'Photo 2');
  qualityWarnings1 = [...qualityWarnings1, ...q1.warnings];
  qualityWarnings2 = [...qualityWarnings2, ...q2.warnings];
  qualityWarnings.push(...q1.warnings, ...(sameUrl ? [] : q2.warnings));

  const distSaysSame = distance < DISTANCE_SAME_THRESHOLD;
  const cosSaysSame = cosine > COSINE_SAME_THRESHOLD;
  const samePerson = cosSaysSame;

  const { confidence, label: confidenceLabel } = sameUrl
    ? { confidence: 1, label: 'Very high' }
    : computeConfidence(face1.score, face2.score, cosine, distance);

  let verdict: Verdict;
  if (sameUrl) {
    verdict = 'same';
  } else if (distSaysSame !== cosSaysSame && confidence < 0.30) {
    verdict = 'inconclusive';
  } else {
    verdict = samePerson ? 'same' : 'different';
  }

  return {
    samePerson, verdict, score,
    euclideanDistance: distance,
    cosineSimilarity: cosine,
    geometricSimilarity: geoSim,
    landmarkAlignment: sameUrl ? 1 : undefined,
    confidence, confidenceLabel,
    qualityWarnings: qualityWarnings.length > 0 ? qualityWarnings : undefined,
    qualityWarnings1: qualityWarnings1.length > 0 ? qualityWarnings1 : undefined,
    qualityWarnings2: qualityWarnings2.length > 0 ? qualityWarnings2 : undefined,
    progressSteps: steps,
    annotatedImage1, annotatedImage2,
    face1: { detection: { score: face1.score, box: face1.box }, features: features1 },
    face2: { detection: { score: face2.score, box: face2.box }, features: features2 },
    landmarks1: face1.landmarks5,
    landmarks2: face2.landmarks5,
  };
}

export async function loadModels(): Promise<void> {
  await loadOnnxModels();
}

export function isModelsLoaded(): boolean {
  return isLoaded();
}
