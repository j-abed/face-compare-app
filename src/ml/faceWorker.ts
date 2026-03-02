import { loadOnnxModels } from './onnxLoader';
import { detectFaceRobust, type DetectedFace } from './scrfdDetector';
import { getEmbedding } from './arcfaceEmbed';

export type WorkerRequest =
  | { type: 'loadModels'; id: number }
  | { type: 'detectFace'; id: number; imageData: ImageData; width: number; height: number }
  | {
      type: 'compareFaces';
      id: number;
      image1Data: ImageData;
      image2Data: ImageData;
      width1: number;
      height1: number;
      width2: number;
      height2: number;
      sameImage: boolean;
    };

export type WorkerResponse =
  | { type: 'loadModels'; id: number; success: boolean; error?: string }
  | { type: 'detectFace'; id: number; face: DetectedFace | null; error?: string }
  | {
      type: 'compareFaces';
      id: number;
      face1: DetectedFace | null;
      face2: DetectedFace | null;
      embedding1: number[] | null;
      embedding2: number[] | null;
      error?: string;
    }
  | { type: 'progress'; id: number; step: string };

function imageDataToCanvas(data: ImageData, width: number, height: number): HTMLCanvasElement {
  const canvas = new OffscreenCanvas(width, height) as unknown as HTMLCanvasElement;
  const ctx = (canvas as unknown as OffscreenCanvas).getContext('2d')!;
  ctx.putImageData(data, 0, 0);
  return canvas;
}

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;

  try {
    switch (msg.type) {
      case 'loadModels': {
        await loadOnnxModels();
        self.postMessage({ type: 'loadModels', id: msg.id, success: true } satisfies WorkerResponse);
        break;
      }

      case 'detectFace': {
        const canvas = imageDataToCanvas(msg.imageData, msg.width, msg.height);
        const face = await detectFaceRobust(canvas);
        self.postMessage({ type: 'detectFace', id: msg.id, face } satisfies WorkerResponse);
        break;
      }

      case 'compareFaces': {
        self.postMessage({ type: 'progress', id: msg.id, step: 'loading_models' } satisfies WorkerResponse);
        await loadOnnxModels();

        self.postMessage({ type: 'progress', id: msg.id, step: 'detecting_face_1' } satisfies WorkerResponse);
        const canvas1 = imageDataToCanvas(msg.image1Data, msg.width1, msg.height1);
        const face1 = await detectFaceRobust(canvas1);

        let face2: DetectedFace | null;
        let canvas2: HTMLCanvasElement;
        if (msg.sameImage) {
          face2 = face1;
          canvas2 = canvas1;
        } else {
          self.postMessage({ type: 'progress', id: msg.id, step: 'detecting_face_2' } satisfies WorkerResponse);
          canvas2 = imageDataToCanvas(msg.image2Data, msg.width2, msg.height2);
          face2 = await detectFaceRobust(canvas2);
        }

        let embedding1: number[] | null = null;
        let embedding2: number[] | null = null;

        if (face1 && face2) {
          self.postMessage({ type: 'progress', id: msg.id, step: 'comparing' } satisfies WorkerResponse);
          if (msg.sameImage) {
            const emb = await getEmbedding(canvas1, face1.landmarks5, false);
            embedding1 = Array.from(emb);
            embedding2 = embedding1;
          } else {
            const [emb1, emb2] = await Promise.all([
              getEmbedding(canvas1, face1.landmarks5),
              getEmbedding(canvas2, face2.landmarks5),
            ]);
            embedding1 = Array.from(emb1);
            embedding2 = Array.from(emb2);
          }
        }

        self.postMessage({
          type: 'compareFaces',
          id: msg.id,
          face1,
          face2,
          embedding1,
          embedding2,
        } satisfies WorkerResponse);
        break;
      }
    }
  } catch (err) {
    self.postMessage({
      type: msg.type,
      id: msg.id,
      error: err instanceof Error ? err.message : String(err),
    } as WorkerResponse);
  }
};
