import { compareFaces } from './faceComparison';

export interface IdentifyResult {
  url: string;
  score: number;
  cosine: number;
  verdict: string;
}

export async function identifyFace(
  probeUrl: string,
  galleryUrls: string[],
  onProgress?: (current: number, total: number) => void
): Promise<IdentifyResult[]> {
  const total = galleryUrls.length;
  const results: IdentifyResult[] = [];

  for (let i = 0; i < galleryUrls.length; i++) {
    onProgress?.(i + 1, total);
    const res = await compareFaces(probeUrl, galleryUrls[i]);
    results.push({
      url: galleryUrls[i],
      score: res.score,
      cosine: res.cosineSimilarity ?? 0,
      verdict: res.verdict,
    });
  }

  return results.sort((a, b) => b.score - a.score);
}
