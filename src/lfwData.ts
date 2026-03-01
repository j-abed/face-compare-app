export interface LfwSamePair {
  person: string;
  image1: string;
  image2: string;
}

export interface LfwDifferentPair {
  person1: string;
  image1: string;
  person2: string;
  image2: string;
}

export interface LfwPerson {
  name: string;
  images: string[];
}

export interface LfwManifest {
  source: string;
  license: string;
  people: LfwPerson[];
  pairs: {
    same: LfwSamePair[];
    different: LfwDifferentPair[];
  };
}

let cachedManifest: LfwManifest | null = null;

export async function loadLfwManifest(): Promise<LfwManifest> {
  if (cachedManifest) return cachedManifest;

  const response = await fetch('/lfw/manifest.json');
  if (!response.ok) {
    throw new Error(`Failed to load LFW manifest: ${response.statusText}`);
  }
  cachedManifest = (await response.json()) as LfwManifest;
  return cachedManifest;
}

export function getRandomPair(manifest: LfwManifest): {
  image1: string;
  image2: string;
  isSame: boolean;
} {
  const allPairs = [
    ...manifest.pairs.same.map((p) => ({
      image1: p.image1,
      image2: p.image2,
      isSame: true,
    })),
    ...manifest.pairs.different.map((p) => ({
      image1: p.image1,
      image2: p.image2,
      isSame: false,
    })),
  ];

  const index = Math.floor(Math.random() * allPairs.length);
  return allPairs[index];
}

export function getAllImageUrls(manifest: LfwManifest): string[] {
  return manifest.people.flatMap((person) => person.images);
}
