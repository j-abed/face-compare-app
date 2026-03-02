import * as ort from 'onnxruntime-web/wasm';

ort.env.wasm.numThreads = 1;

const MODELS_BASE = '/models/onnx';

let detSession: ort.InferenceSession | null = null;
let recSession: ort.InferenceSession | null = null;
let loading: Promise<void> | null = null;

export async function loadOnnxModels(): Promise<void> {
  if (detSession && recSession) return;
  if (loading) return loading;

  loading = (async () => {
    const [det, rec] = await Promise.all([
      ort.InferenceSession.create(`${MODELS_BASE}/det_500m.onnx`, {
        executionProviders: ['wasm'],
      }),
      ort.InferenceSession.create(`${MODELS_BASE}/w600k_mbf.onnx`, {
        executionProviders: ['wasm'],
      }),
    ]);
    detSession = det;
    recSession = rec;
  })();

  return loading;
}

export function getDetSession(): ort.InferenceSession {
  if (!detSession) throw new Error('ONNX models not loaded. Call loadOnnxModels() first.');
  return detSession;
}

export function getRecSession(): ort.InferenceSession {
  if (!recSession) throw new Error('ONNX models not loaded. Call loadOnnxModels() first.');
  return recSession;
}

export function isLoaded(): boolean {
  return detSession !== null && recSession !== null;
}
