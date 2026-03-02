import { compareFaces, type CompareResult } from './faceComparison';
import {
  loadLfwManifest,
  type LfwSamePair,
  type LfwDifferentPair,
} from './lfwData';

export interface PairResult {
  pairIndex: number;
  isSame: boolean;
  image1: string;
  image2: string;
  personInfo: string;
  result: CompareResult;
  correct: boolean;
  error?: string;
}

export interface RocPoint {
  threshold: number;
  tpr: number;
  fpr: number;
}

export interface DetPoint {
  threshold: number;
  fnr: number;
  fpr: number;
}

export interface PerPersonAccuracy {
  name: string;
  accuracy: number;
  total: number;
  correct: number;
}

export interface BenchmarkResults {
  totalPairs: number;
  samePairs: number;
  differentPairs: number;
  completed: number;
  errors: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  falsePositiveRate: number;
  falseNegativeRate: number;
  inconclusiveCount: number;
  truePositives: number;
  trueNegatives: number;
  falsePositives: number;
  falseNegatives: number;
  pairResults: PairResult[];
  totalTimeMs: number;
  avgTimePerPairMs: number;
  rocData: RocPoint[];
  detData: DetPoint[];
  auc: number;
  perPersonAccuracy: PerPersonAccuracy[];
}

export type BenchmarkProgress = {
  current: number;
  total: number;
  currentPair: { image1: string; image2: string; isSame: boolean };
  pairResult?: PairResult;
};

interface TestPair {
  image1: string;
  image2: string;
  isSame: boolean;
  personInfo: string;
}

function buildPairs(
  samePairs: LfwSamePair[],
  differentPairs: LfwDifferentPair[],
): TestPair[] {
  const pairs: TestPair[] = [];
  for (const sp of samePairs) {
    pairs.push({
      image1: sp.image1,
      image2: sp.image2,
      isSame: true,
      personInfo: sp.person,
    });
  }
  for (const dp of differentPairs) {
    pairs.push({
      image1: dp.image1,
      image2: dp.image2,
      isSame: false,
      personInfo: `${dp.person1} vs ${dp.person2}`,
    });
  }
  return pairs;
}

function safeDiv(numerator: number, denominator: number): number {
  return denominator === 0 ? 0 : numerator / denominator;
}

export async function runBenchmark(
  onProgress?: (progress: BenchmarkProgress) => void,
): Promise<BenchmarkResults> {
  const manifest = await loadLfwManifest();
  const pairs = buildPairs(manifest.pairs.same, manifest.pairs.different);
  const total = pairs.length;

  const pairResults: PairResult[] = [];
  let errors = 0;
  let inconclusiveCount = 0;
  let tp = 0;
  let tn = 0;
  let fp = 0;
  let fn = 0;

  const t0 = performance.now();

  for (let i = 0; i < total; i++) {
    const pair = pairs[i];

    onProgress?.({
      current: i + 1,
      total,
      currentPair: {
        image1: pair.image1,
        image2: pair.image2,
        isSame: pair.isSame,
      },
    });

    let result: CompareResult;
    let hadError = false;
    try {
      result = await compareFaces(pair.image1, pair.image2);
    } catch (err) {
      hadError = true;
      result = {
        samePerson: false,
        verdict: 'inconclusive',
        score: 0,
        error: err instanceof Error ? err.message : String(err),
      };
    }

    const isInconclusive = result.verdict === 'inconclusive';
    if (isInconclusive) inconclusiveCount++;
    if (hadError || result.error) errors++;

    let correct: boolean;
    if (pair.isSame) {
      if (result.verdict === 'same') {
        correct = true;
        tp++;
      } else {
        correct = false;
        fn++;
      }
    } else {
      if (result.verdict === 'different') {
        correct = true;
        tn++;
      } else {
        correct = false;
        fp++;
      }
    }

    const pr: PairResult = {
      pairIndex: i,
      isSame: pair.isSame,
      image1: pair.image1,
      image2: pair.image2,
      personInfo: pair.personInfo,
      result,
      correct,
      error: result.error,
    };
    pairResults.push(pr);

    onProgress?.({
      current: i + 1,
      total,
      currentPair: {
        image1: pair.image1,
        image2: pair.image2,
        isSame: pair.isSame,
      },
      pairResult: pr,
    });
  }

  const totalTimeMs = performance.now() - t0;
  const completed = pairResults.length;
  const evaluated = completed - errors;

  const accuracy = safeDiv(tp + tn, evaluated);
  const precision = safeDiv(tp, tp + fp);
  const recall = safeDiv(tp, tp + fn);
  const f1 = safeDiv(2 * precision * recall, precision + recall);
  const falsePositiveRate = safeDiv(fp, fp + tn);
  const falseNegativeRate = safeDiv(fn, fn + tp);

  // Get similarity score in [-1, 1] for threshold sweep
  function getSimilarity(pr: PairResult): number {
    if (pr.result.cosineSimilarity != null) return pr.result.cosineSimilarity;
    // score is [0,1], map to [-1,1]
    return 2 * (pr.result.score ?? 0) - 1;
  }

  const rocData: RocPoint[] = [];
  const detData: DetPoint[] = [];

  for (let t = -1.0; t <= 1.0; t += 0.02) {
    const threshold = Math.round(t * 100) / 100;
    let tp_t = 0, fn_t = 0, fp_t = 0, tn_t = 0;
    for (const pr of pairResults) {
      const sim = getSimilarity(pr);
      const predSame = sim >= threshold;
      if (pr.isSame) {
        if (predSame) tp_t++;
        else fn_t++;
      } else {
        if (predSame) fp_t++;
        else tn_t++;
      }
    }
    const tpr = safeDiv(tp_t, tp_t + fn_t);
    const fpr = safeDiv(fp_t, fp_t + tn_t);
    const fnr = safeDiv(fn_t, fn_t + tp_t);
    rocData.push({ threshold, tpr, fpr });
    detData.push({ threshold, fnr, fpr });
  }

  // AUC via trapezoidal rule (ROC: integrate TPR over FPR, sorted by FPR ascending)
  const rocSorted = [...rocData].sort((a, b) => a.fpr - b.fpr);
  let auc = 0;
  for (let i = 1; i < rocSorted.length; i++) {
    const dx = rocSorted[i].fpr - rocSorted[i - 1].fpr;
    const yAvg = (rocSorted[i].tpr + rocSorted[i - 1].tpr) / 2;
    auc += dx * yAvg;
  }

  // Per-person accuracy (same-pairs only)
  const personStats = new Map<string, { correct: number; total: number }>();
  for (const pr of pairResults) {
    if (!pr.isSame) continue;
    const name = pr.personInfo;
    const stats = personStats.get(name) ?? { correct: 0, total: 0 };
    stats.total++;
    if (pr.correct) stats.correct++;
    personStats.set(name, stats);
  }
  const perPersonAccuracy: PerPersonAccuracy[] = Array.from(personStats.entries())
    .map(([name, { correct, total }]) => ({
      name,
      accuracy: safeDiv(correct, total),
      total,
      correct,
    }))
    .sort((a, b) => a.accuracy - b.accuracy);

  return {
    totalPairs: total,
    samePairs: manifest.pairs.same.length,
    differentPairs: manifest.pairs.different.length,
    completed,
    errors,
    accuracy,
    precision,
    recall,
    f1,
    falsePositiveRate,
    falseNegativeRate,
    inconclusiveCount,
    truePositives: tp,
    trueNegatives: tn,
    falsePositives: fp,
    falseNegatives: fn,
    pairResults,
    totalTimeMs,
    avgTimePerPairMs: safeDiv(totalTimeMs, completed),
    rocData,
    detData,
    auc,
    perPersonAccuracy,
  };
}
