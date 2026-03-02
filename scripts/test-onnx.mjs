import ort from 'onnxruntime-node';
import sharp from 'sharp';
import { readFileSync } from 'fs';

const SIZE = 112;
const INPUT_SIZE = 640;
const STRIDES = [8, 16, 32];
const NUM_ANCHORS = 2;

function cosine(a, b) {
  let dot = 0, ma = 0, mb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i]*b[i]; ma += a[i]*a[i]; mb += b[i]*b[i]; }
  return dot / (Math.sqrt(ma) * Math.sqrt(mb));
}

function l2Normalize(vec) {
  let sum = 0;
  for (let i = 0; i < vec.length; i++) sum += vec[i]*vec[i];
  const norm = Math.sqrt(sum) || 1;
  return vec.map(v => v / norm);
}

async function preprocess640(imagePath) {
  const img = sharp(imagePath);
  const meta = await img.metadata();
  const ratio = Math.min(INPUT_SIZE / meta.width, INPUT_SIZE / meta.height);
  const newW = Math.round(meta.width * ratio);
  const newH = Math.round(meta.height * ratio);

  const resized = await img.resize(newW, newH).removeAlpha().raw().toBuffer();
  const float32 = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);

  for (let y = 0; y < newH; y++) {
    for (let x = 0; x < newW; x++) {
      const srcIdx = (y * newW + x) * 3;
      const dstIdx = y * INPUT_SIZE + x;
      float32[dstIdx] = (resized[srcIdx] - 127.5) / 128.0;
      float32[INPUT_SIZE*INPUT_SIZE + dstIdx] = (resized[srcIdx + 1] - 127.5) / 128.0;
      float32[2*INPUT_SIZE*INPUT_SIZE + dstIdx] = (resized[srcIdx + 2] - 127.5) / 128.0;
    }
  }
  return { tensor: new ort.Tensor('float32', float32, [1, 3, INPUT_SIZE, INPUT_SIZE]), scale: ratio };
}

async function detect(session, imagePath) {
  const { tensor, scale } = await preprocess640(imagePath);
  const inputName = session.inputNames[0];
  const results = await session.run({ [inputName]: tensor });
  const outputNames = session.outputNames;
  const fmc = STRIDES.length;

  let bestScore = -1;
  let bestBox = null;
  let bestKps = null;

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
          const sc = scores[idx];
          if (sc < 0.3) continue;

          const cx = col * stride;
          const cy = row * stride;
          const x1 = (cx - bboxes[idx*4]*stride) / scale;
          const y1 = (cy - bboxes[idx*4+1]*stride) / scale;
          const x2 = (cx + bboxes[idx*4+2]*stride) / scale;
          const y2 = (cy + bboxes[idx*4+3]*stride) / scale;

          const kp = [];
          for (let k = 0; k < 5; k++) {
            kp.push([
              (cx + kps[idx*10 + k*2]*stride) / scale,
              (cy + kps[idx*10 + k*2 + 1]*stride) / scale
            ]);
          }

          if (sc > bestScore) {
            bestScore = sc;
            bestBox = { x1, y1, x2, y2 };
            bestKps = kp;
          }
        }
      }
    }
  }

  return { score: bestScore, box: bestBox, kps: bestKps };
}

async function cropAndEmbed(recSession, imagePath, box) {
  const padX = (box.x2 - box.x1) * 0.3;
  const padY = (box.y2 - box.y1) * 0.3;
  const x = Math.max(0, Math.round(box.x1 - padX));
  const y = Math.max(0, Math.round(box.y1 - padY));
  const w = Math.round(box.x2 - box.x1 + 2*padX);
  const h = Math.round(box.y2 - box.y1 + 2*padY);

  const cropped = await sharp(imagePath)
    .extract({ left: x, top: y, width: w, height: h })
    .resize(SIZE, SIZE)
    .removeAlpha()
    .raw()
    .toBuffer();

  const float32 = new Float32Array(3 * SIZE * SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    float32[i] = (cropped[i*3] - 127.5) / 128.0;
    float32[SIZE*SIZE + i] = (cropped[i*3 + 1] - 127.5) / 128.0;
    float32[2*SIZE*SIZE + i] = (cropped[i*3 + 2] - 127.5) / 128.0;
  }

  const tensor = new ort.Tensor('float32', float32, [1, 3, SIZE, SIZE]);
  const result = await recSession.run({ [recSession.inputNames[0]]: tensor });
  const raw = new Float32Array(result[recSession.outputNames[0]].data);
  return l2Normalize(raw);
}

async function main() {
  const detSession = await ort.InferenceSession.create('public/models/onnx/det_500m.onnx');
  const recSession = await ort.InferenceSession.create('public/models/onnx/w600k_mbf.onnx');

  const images = [
    'public/lfw/Tony_Blair/Tony_Blair_0001.jpg',
    'public/lfw/Tony_Blair/Tony_Blair_0003.jpg',
    'public/lfw/Hugo_Chavez/Hugo_Chavez_0001.jpg',
  ];

  const embeddings = [];
  for (const img of images) {
    const det = await detect(detSession, img);
    if (!det.box) { console.log('No face in', img); continue; }
    console.log(`${img}: score=${det.score.toFixed(3)} box=[${Object.values(det.box).map(v=>v.toFixed(0))}]`);
    const emb = await cropAndEmbed(recSession, img, det.box);
    embeddings.push({ img, emb });
  }

  for (let i = 0; i < embeddings.length; i++) {
    for (let j = i + 1; j < embeddings.length; j++) {
      const cos = cosine(embeddings[i].emb, embeddings[j].emb);
      const name1 = embeddings[i].img.split('/').pop();
      const name2 = embeddings[j].img.split('/').pop();
      console.log(`${name1} vs ${name2}: cos=${cos.toFixed(4)}`);
    }
  }
}

main().catch(console.error);
