import type { WorkerRequest, WorkerResponse } from './faceWorker';

let worker: Worker | null = null;
let requestId = 0;
const pendingRequests = new Map<number, {
  resolve: (value: WorkerResponse) => void;
  reject: (reason: Error) => void;
  onProgress?: (step: string) => void;
}>();

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL('./faceWorker.ts', import.meta.url), { type: 'module' });
    worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const msg = e.data;
      if (msg.type === 'progress') {
        const pending = pendingRequests.get(msg.id);
        pending?.onProgress?.(msg.step);
        return;
      }
      const pending = pendingRequests.get(msg.id);
      if (pending) {
        pendingRequests.delete(msg.id);
        if ('error' in msg && msg.error) {
          pending.reject(new Error(msg.error));
        } else {
          pending.resolve(msg);
        }
      }
    };
    worker.onerror = (e) => {
      for (const [id, pending] of pendingRequests) {
        pending.reject(new Error(e.message || 'Worker error'));
        pendingRequests.delete(id);
      }
    };
  }
  return worker;
}

function sendRequest(
  req: Omit<WorkerRequest, 'id'>,
  onProgress?: (step: string) => void,
): Promise<WorkerResponse> {
  const id = ++requestId;
  const w = getWorker();
  return new Promise((resolve, reject) => {
    pendingRequests.set(id, { resolve, reject, onProgress });
    w.postMessage({ ...req, id } as WorkerRequest);
  });
}

export async function loadModelsInWorker(): Promise<void> {
  const res = await sendRequest({ type: 'loadModels' });
  if (res.type === 'loadModels' && !res.success) {
    throw new Error(res.error || 'Failed to load models in worker');
  }
}

export function imageToImageData(
  img: HTMLImageElement,
): { imageData: ImageData; width: number; height: number } {
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0);
  return {
    imageData: ctx.getImageData(0, 0, canvas.width, canvas.height),
    width: canvas.width,
    height: canvas.height,
  };
}

export { sendRequest };
