import * as ort from 'onnxruntime-web/wasm';
import { getRecSession } from './onnxLoader';
import { alignFace } from './faceAlign';

const SIZE = 112;

function canvasToTensor(canvas: HTMLCanvasElement): ort.Tensor {
  const ctx = canvas.getContext('2d')!;
  const { data } = ctx.getImageData(0, 0, SIZE, SIZE);
  const float32 = new Float32Array(3 * SIZE * SIZE);
  const pixelCount = SIZE * SIZE;

  for (let i = 0; i < pixelCount; i++) {
    float32[i] = (data[i * 4] - 127.5) / 128.0;
    float32[pixelCount + i] = (data[i * 4 + 1] - 127.5) / 128.0;
    float32[2 * pixelCount + i] = (data[i * 4 + 2] - 127.5) / 128.0;
  }

  return new ort.Tensor('float32', float32, [1, 3, SIZE, SIZE]);
}

function l2Normalize(vec: Float32Array): Float32Array {
  let sum = 0;
  for (let i = 0; i < vec.length; i++) sum += vec[i] * vec[i];
  const norm = Math.sqrt(sum) || 1;
  const result = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) result[i] = vec[i] / norm;
  return result;
}

function flipCanvasHorizontally(canvas: HTMLCanvasElement): HTMLCanvasElement {
  const flipped = document.createElement('canvas');
  flipped.width = canvas.width;
  flipped.height = canvas.height;
  const ctx = flipped.getContext('2d')!;
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(canvas, 0, 0);
  return flipped;
}

async function extractEmbedding(canvas: HTMLCanvasElement): Promise<Float32Array> {
  const session = getRecSession();
  const inputMeta = session.inputNames[0];
  const outputMeta = session.outputNames[0];
  const tensor = canvasToTensor(canvas);
  const result = await session.run({ [inputMeta]: tensor });
  return l2Normalize(new Float32Array(result[outputMeta].data as Float32Array));
}

export async function getEmbedding(
  img: HTMLImageElement | HTMLCanvasElement,
  landmarks5: [number, number][],
  augment = true,
): Promise<Float32Array> {
  const aligned = alignFace(img, landmarks5);
  const embedding = await extractEmbedding(aligned);

  if (!augment) return embedding;

  try {
    const flipped = flipCanvasHorizontally(aligned);
    const flippedEmb = await extractEmbedding(flipped);
    const avg = new Float32Array(embedding.length);
    for (let i = 0; i < embedding.length; i++) {
      avg[i] = (embedding[i] + flippedEmb[i]) / 2;
    }
    return l2Normalize(avg);
  } catch {
    return embedding;
  }
}
