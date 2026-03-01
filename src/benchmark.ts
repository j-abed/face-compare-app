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
  };
}
