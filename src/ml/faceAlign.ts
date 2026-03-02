/**
 * Affine alignment of a detected face to a canonical 112x112 template
 * using 5-point landmarks (left eye, right eye, nose, left mouth, right mouth).
 */

const TEMPLATE_112: [number, number][] = [
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041],
];

const ALIGNED_SIZE = 112;

/**
 * Estimate a 2x3 similarity-transform matrix mapping src points to dst points.
 * Uses least-squares over all point pairs.
 */
function estimateSimilarityTransform(
  src: [number, number][],
  dst: [number, number][],
): [number, number, number, number, number, number] {
  const n = src.length;
  let srcMeanX = 0, srcMeanY = 0, dstMeanX = 0, dstMeanY = 0;
  for (let i = 0; i < n; i++) {
    srcMeanX += src[i][0];
    srcMeanY += src[i][1];
    dstMeanX += dst[i][0];
    dstMeanY += dst[i][1];
  }
  srcMeanX /= n; srcMeanY /= n;
  dstMeanX /= n; dstMeanY /= n;

  let srcVar = 0;
  let a = 0, b = 0;
  for (let i = 0; i < n; i++) {
    const sx = src[i][0] - srcMeanX;
    const sy = src[i][1] - srcMeanY;
    const dx = dst[i][0] - dstMeanX;
    const dy = dst[i][1] - dstMeanY;
    srcVar += sx * sx + sy * sy;
    a += sx * dx + sy * dy;
    b += sx * dy - sy * dx;
  }
  a /= srcVar;
  b /= srcVar;

  const tx = dstMeanX - a * srcMeanX + b * srcMeanY;
  const ty = dstMeanY - a * srcMeanY - b * srcMeanX;

  return [a, -b, tx, b, a, ty];
}

/**
 * Align a face to a 112x112 canvas using 5-point landmarks.
 * Returns the aligned canvas ready for ArcFace embedding extraction.
 */
export function alignFace(
  img: HTMLImageElement | HTMLCanvasElement,
  landmarks5: [number, number][],
): HTMLCanvasElement {
  const M = estimateSimilarityTransform(landmarks5, TEMPLATE_112);

  const canvas = document.createElement('canvas');
  canvas.width = ALIGNED_SIZE;
  canvas.height = ALIGNED_SIZE;
  const ctx = canvas.getContext('2d')!;

  ctx.setTransform(M[0], M[3], M[1], M[4], M[2], M[5]);
  ctx.drawImage(img, 0, 0);
  ctx.resetTransform();

  return canvas;
}
