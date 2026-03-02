/**
 * Face quality assessment: blur, pose, exposure, occlusion.
 * Used to flag images that may produce unreliable comparison results.
 */

export interface FaceQualityResult {
  warnings: string[];
  overallScore: number;
}

/**
 * Compute Laplacian variance on the face crop.
 * Higher variance = sharper image.
 * Returns score 0-1 where higher = sharper. Flags if below 0.3.
 */
export function assessBlur(canvas: HTMLCanvasElement): { score: number; isBlurry: boolean } {
  const ctx = canvas.getContext('2d');
  if (!ctx) return { score: 0, isBlurry: true };

  const w = canvas.width;
  const h = canvas.height;
  const imgData = ctx.getImageData(0, 0, w, h);
  const data = imgData.data;

  // Laplacian kernel: [[0,-1,0],[-1,4,-1],[0,-1,0]]
  let sum = 0;
  let count = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = (y * w + x) * 4;
      const g = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
      const top = (data[((y - 1) * w + x) * 4] + data[((y - 1) * w + x) * 4 + 1] + data[((y - 1) * w + x) * 4 + 2]) / 3;
      const bottom = (data[((y + 1) * w + x) * 4] + data[((y + 1) * w + x) * 4 + 1] + data[((y + 1) * w + x) * 4 + 2]) / 3;
      const left = (data[(y * w + (x - 1)) * 4] + data[(y * w + (x - 1)) * 4 + 1] + data[(y * w + (x - 1)) * 4 + 2]) / 3;
      const right = (data[(y * w + (x + 1)) * 4] + data[(y * w + (x + 1)) * 4 + 1] + data[(y * w + (x + 1)) * 4 + 2]) / 3;
      const lap = Math.abs(4 * g - top - bottom - left - right);
      sum += lap * lap;
      count++;
    }
  }
  const variance = count > 0 ? sum / count : 0;
  // Normalize: typical sharp face ~100-500, blurry <50. Map to 0-1.
  const score = Math.min(1, Math.max(0, (variance - 20) / 200));
  return { score, isBlurry: score < 0.3 };
}

/**
 * Estimate yaw angle from 5-point landmarks.
 * Order: left eye, right eye, nose, left mouth corner, right mouth corner.
 * Warn if yaw > 30 degrees.
 */
export function assessPose(landmarks5: [number, number][]): { yawDeg: number; isProfile: boolean } {
  if (landmarks5.length < 5) return { yawDeg: 0, isProfile: false };

  const [le, re, nose] = landmarks5;
  const eyeCenterX = (le[0] + re[0]) / 2;
  const eyeCenterY = (le[1] + re[1]) / 2;
  const interEye = Math.hypot(re[0] - le[0], re[1] - le[1]);
  if (interEye < 1) return { yawDeg: 0, isProfile: false };

  // Nose offset from eye center (in image coords): positive = nose right of center = face turned left
  const noseOffsetX = nose[0] - eyeCenterX;
  // Approximate yaw: ~15px offset per 10 deg for typical face. Scale by inter-eye.
  const yawDeg = (noseOffsetX / interEye) * 25;
  return { yawDeg, isProfile: Math.abs(yawDeg) > 30 };
}

/**
 * Analyze pixel histogram. Warn if over-exposed (mean > 220) or under-exposed (mean < 35).
 */
export function assessExposure(canvas: HTMLCanvasElement): {
  mean: number;
  isOverExposed: boolean;
  isUnderExposed: boolean;
} {
  const ctx = canvas.getContext('2d');
  if (!ctx) return { mean: 128, isOverExposed: false, isUnderExposed: false };

  const w = canvas.width;
  const h = canvas.height;
  const imgData = ctx.getImageData(0, 0, w, h);
  const data = imgData.data;
  let sum = 0;
  const n = w * h;
  for (let i = 0; i < n; i++) {
    const idx = i * 4;
    sum += (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
  }
  const mean = n > 0 ? sum / n : 128;
  return {
    mean,
    isOverExposed: mean > 220,
    isUnderExposed: mean < 35,
  };
}

/**
 * Check if any key landmarks fall too close to or outside the bounding box edges.
 */
export function assessOcclusion(
  box: { x: number; y: number; width: number; height: number },
  landmarks5: [number, number][],
): { hasOcclusion: boolean; warnings: string[] } {
  const margin = Math.min(box.width, box.height) * 0.08;
  const left = box.x + margin;
  const right = box.x + box.width - margin;
  const top = box.y + margin;
  const bottom = box.y + box.height - margin;
  const warnings: string[] = [];

  const names = ['left eye', 'right eye', 'nose', 'left mouth', 'right mouth'];
  for (let i = 0; i < landmarks5.length && i < names.length; i++) {
    const [px, py] = landmarks5[i];
    if (px < left) warnings.push(`${names[i]} near or outside left edge`);
    if (px > right) warnings.push(`${names[i]} near or outside right edge`);
    if (py < top) warnings.push(`${names[i]} near or outside top edge`);
    if (py > bottom) warnings.push(`${names[i]} near or outside bottom edge`);
  }

  return { hasOcclusion: warnings.length > 0, warnings };
}

/**
 * Create a canvas crop of the face from the full image using the bounding box.
 */
export function cropFaceToCanvas(
  img: HTMLImageElement | HTMLCanvasElement,
  box: { x: number; y: number; width: number; height: number },
): HTMLCanvasElement {
  const srcW = img instanceof HTMLImageElement ? img.naturalWidth : img.width;
  const srcH = img instanceof HTMLImageElement ? img.naturalHeight : img.height;

  const x = Math.max(0, Math.floor(box.x));
  const y = Math.max(0, Math.floor(box.y));
  const w = Math.min(srcW - x, Math.ceil(box.width));
  const h = Math.min(srcH - y, Math.ceil(box.height));

  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, w);
  canvas.height = Math.max(1, h);
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, x, y, w, h, 0, 0, w, h);
  return canvas;
}

/**
 * Run all quality checks and return combined warnings and overall score.
 */
export function assessFaceQuality(
  canvas: HTMLCanvasElement,
  box: { x: number; y: number; width: number; height: number },
  landmarks5: [number, number][],
  photoLabel: string,
): FaceQualityResult {
  const warnings: string[] = [];

  const { score: blurScore, isBlurry } = assessBlur(canvas);
  if (isBlurry) warnings.push(`${photoLabel}: Image is blurry (sharpness ${(blurScore * 100).toFixed(0)}%).`);

  const { yawDeg, isProfile } = assessPose(landmarks5);
  if (isProfile) warnings.push(`${photoLabel}: Face angle is too extreme (yaw ~${Math.round(Math.abs(yawDeg))}°).`);

  const { isOverExposed, isUnderExposed } = assessExposure(canvas);
  if (isOverExposed) warnings.push(`${photoLabel}: Image is over-exposed (too bright).`);
  if (isUnderExposed) warnings.push(`${photoLabel}: Image is under-exposed (too dark).`);

  const { warnings: occWarnings } = assessOcclusion(box, landmarks5);
  for (const w of occWarnings) {
    warnings.push(`${photoLabel}: ${w}.`);
  }

  // Overall score: blend blur (0.3), pose (0.2), exposure (0.25), occlusion (0.25)
  const poseScore = Math.max(0, 1 - Math.abs(yawDeg) / 45);
  const exposureScore = isOverExposed || isUnderExposed ? 0.3 : 1;
  const occlusionScore = occWarnings.length === 0 ? 1 : Math.max(0, 1 - occWarnings.length * 0.25);
  const overallScore =
    blurScore * 0.3 + poseScore * 0.2 + exposureScore * 0.25 + occlusionScore * 0.25;

  return { warnings, overallScore: Math.max(0, Math.min(1, overallScore)) };
}
