import * as ort from 'onnxruntime-web/wasm';
import { getDetSession } from './onnxLoader';

export interface DetectedFace {
  box: { x: number; y: number; width: number; height: number };
  score: number;
  landmarks5: [number, number][];
}

const INPUT_SIZE = 640;
const STRIDES = [8, 16, 32];
const NUM_ANCHORS = 2;
const NMS_THRESH = 0.4;

function generateAnchors(
  height: number,
  width: number,
  stride: number,
): Float32Array {
  const total = height * width * NUM_ANCHORS * 2;
  const centers = new Float32Array(total);
  let idx = 0;
  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      const cx = col * stride;
      const cy = row * stride;
      for (let a = 0; a < NUM_ANCHORS; a++) {
        centers[idx++] = cx;
        centers[idx++] = cy;
      }
    }
  }
  return centers;
}

function distance2bbox(
  cx: number,
  cy: number,
  pred: Float32Array,
  offset: number,
  stride: number,
): [number, number, number, number] {
  const left = pred[offset] * stride;
  const top = pred[offset + 1] * stride;
  const right = pred[offset + 2] * stride;
  const bottom = pred[offset + 3] * stride;
  return [cx - left, cy - top, cx + right, cy + bottom];
}

function distance2kps(
  cx: number,
  cy: number,
  pred: Float32Array,
  offset: number,
  stride: number,
): [number, number][] {
  const pts: [number, number][] = [];
  for (let i = 0; i < 5; i++) {
    pts.push([
      cx + pred[offset + i * 2] * stride,
      cy + pred[offset + i * 2 + 1] * stride,
    ]);
  }
  return pts;
}

function nms(
  dets: { x1: number; y1: number; x2: number; y2: number; score: number; idx: number }[],
  thresh: number,
): number[] {
  if (dets.length === 0) return [];
  const sorted = [...dets].sort((a, b) => b.score - a.score);
  const keep: number[] = [];
  const suppressed = new Set<number>();

  for (let i = 0; i < sorted.length; i++) {
    if (suppressed.has(i)) continue;
    keep.push(sorted[i].idx);
    const a = sorted[i];
    const aArea = (a.x2 - a.x1 + 1) * (a.y2 - a.y1 + 1);

    for (let j = i + 1; j < sorted.length; j++) {
      if (suppressed.has(j)) continue;
      const b = sorted[j];
      const xx1 = Math.max(a.x1, b.x1);
      const yy1 = Math.max(a.y1, b.y1);
      const xx2 = Math.min(a.x2, b.x2);
      const yy2 = Math.min(a.y2, b.y2);
      const w = Math.max(0, xx2 - xx1 + 1);
      const h = Math.max(0, yy2 - yy1 + 1);
      const inter = w * h;
      const bArea = (b.x2 - b.x1 + 1) * (b.y2 - b.y1 + 1);
      if (inter / (aArea + bArea - inter) > thresh) {
        suppressed.add(j);
      }
    }
  }
  return keep;
}

function preprocessImage(
  img: HTMLImageElement | HTMLCanvasElement,
): { tensor: ort.Tensor; scale: number; inputH: number; inputW: number } {
  const srcW = img instanceof HTMLImageElement ? img.naturalWidth : img.width;
  const srcH = img instanceof HTMLImageElement ? img.naturalHeight : img.height;

  const ratio = Math.min(INPUT_SIZE / srcW, INPUT_SIZE / srcH);
  const newW = Math.round(srcW * ratio);
  const newH = Math.round(srcH * ratio);

  const canvas = document.createElement('canvas');
  canvas.width = INPUT_SIZE;
  canvas.height = INPUT_SIZE;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  ctx.drawImage(img, 0, 0, newW, newH);

  const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const { data } = imageData;
  const float32 = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);

  for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
    float32[i] = (data[i * 4] - 127.5) / 128.0;                       // R
    float32[INPUT_SIZE * INPUT_SIZE + i] = (data[i * 4 + 1] - 127.5) / 128.0; // G
    float32[2 * INPUT_SIZE * INPUT_SIZE + i] = (data[i * 4 + 2] - 127.5) / 128.0; // B
  }

  return {
    tensor: new ort.Tensor('float32', float32, [1, 3, INPUT_SIZE, INPUT_SIZE]),
    scale: ratio,
    inputH: INPUT_SIZE,
    inputW: INPUT_SIZE,
  };
}

export async function detectFaces(
  img: HTMLImageElement | HTMLCanvasElement,
  threshold = 0.5,
): Promise<DetectedFace[]> {
  const session = getDetSession();
  const { tensor, scale, inputH, inputW } = preprocessImage(img);

  const inputName = session.inputNames[0];
  const feeds: Record<string, ort.Tensor> = { [inputName]: tensor };
  const results = await session.run(feeds);
  const outputNames = session.outputNames;

  const fmc = STRIDES.length;
  const allDets: {
    x1: number; y1: number; x2: number; y2: number;
    score: number; kps: [number, number][]; idx: number;
  }[] = [];

  let globalIdx = 0;

  for (let strideIdx = 0; strideIdx < fmc; strideIdx++) {
    const stride = STRIDES[strideIdx];
    const scoreData = results[outputNames[strideIdx]].data as Float32Array;
    const bboxData = results[outputNames[strideIdx + fmc]].data as Float32Array;
    const kpsData = results[outputNames[strideIdx + fmc * 2]].data as Float32Array;

    const featH = Math.ceil(inputH / stride);
    const featW = Math.ceil(inputW / stride);
    const anchors = generateAnchors(featH, featW, stride);
    const numCells = featH * featW * NUM_ANCHORS;

    for (let i = 0; i < numCells; i++) {
      const sc = scoreData[i];
      if (sc < threshold) continue;

      const cx = anchors[i * 2];
      const cy = anchors[i * 2 + 1];
      const bbox = distance2bbox(cx, cy, bboxData, i * 4, stride);
      const kps = distance2kps(cx, cy, kpsData, i * 10, stride);

      allDets.push({
        x1: bbox[0] / scale,
        y1: bbox[1] / scale,
        x2: bbox[2] / scale,
        y2: bbox[3] / scale,
        score: sc,
        kps: kps.map(([x, y]) => [x / scale, y / scale] as [number, number]),
        idx: globalIdx++,
      });
    }
  }

  const keepIndices = nms(allDets, NMS_THRESH);
  const keepSet = new Set(keepIndices);

  return allDets
    .filter((d) => keepSet.has(d.idx))
    .sort((a, b) => b.score - a.score)
    .map((d) => ({
      box: {
        x: d.x1,
        y: d.y1,
        width: d.x2 - d.x1,
        height: d.y2 - d.y1,
      },
      score: d.score,
      landmarks5: d.kps,
    }));
}

export async function detectFaceRobust(
  img: HTMLImageElement | HTMLCanvasElement,
): Promise<DetectedFace | null> {
  for (const thresh of [0.5, 0.3, 0.15]) {
    const faces = await detectFaces(img, thresh);
    if (faces.length > 0) {
      let best = faces[0];
      let bestArea = best.box.width * best.box.height;
      for (let i = 1; i < faces.length; i++) {
        const area = faces[i].box.width * faces[i].box.height;
        if (area > bestArea) {
          best = faces[i];
          bestArea = area;
        }
      }
      return best;
    }
  }
  return null;
}
