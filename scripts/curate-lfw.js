import { readFileSync, mkdirSync, copyFileSync, writeFileSync, readdirSync, existsSync } from 'fs';
import { join, basename } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const PROJECT_ROOT = join(__dirname, '..');

// Configurable paths (set LFW_SUBDIR env to override, e.g. lfw-deepfunneled or lfw_funneled)
const LFW_BASE = '/tmp/lfw-download';
const LFW_SUBDIR = process.env.LFW_SUBDIR || 'lfw_funneled';
let LFW_EXTRACTED_DIR = join(LFW_BASE, LFW_SUBDIR);
const PAIRS_FILE = join(LFW_BASE, 'pairs.txt');

function resolveLfwExtractedDir() {
  if (existsSync(LFW_EXTRACTED_DIR)) {
    const subdirs = readdirSync(LFW_EXTRACTED_DIR, { withFileTypes: true })
      .filter((d) => d.isDirectory())
      .map((d) => d.name);
    if (subdirs.length > 0) return LFW_EXTRACTED_DIR;
  }
  // Auto-detect: try common names (both hyphen and underscore)
  for (const name of ['lfw-deepfunneled', 'lfw_funneled', 'lfw-funneled', 'lfw']) {
    const candidate = join(LFW_BASE, name);
    if (existsSync(candidate)) {
      const subdirs = readdirSync(candidate, { withFileTypes: true })
        .filter((d) => d.isDirectory())
        .map((d) => d.name);
      if (subdirs.length > 0) {
        LFW_EXTRACTED_DIR = candidate;
        return LFW_EXTRACTED_DIR;
      }
    }
  }
  return null;
}
const OUTPUT_DIR = join(PROJECT_ROOT, 'public', 'lfw');
const MANIFEST_PATH = join(OUTPUT_DIR, 'manifest.json');

// How many people to select (larger = less overfitting to benchmark)
const TARGET_PEOPLE = 40;
const TARGET_SAME_PAIRS = 40;
const TARGET_DIFF_PAIRS = 40;

function parsePairs(pairsPath) {
  const content = readFileSync(pairsPath, 'utf-8').trim();
  const lines = content.split('\n');

  // First line: "num_folds\tpairs_per_fold" or just "pairs_per_fold"
  const header = lines[0].trim().split('\t');
  const numFolds = header.length > 1 ? parseInt(header[0]) : 10;
  const pairsPerFold = header.length > 1 ? parseInt(header[1]) : parseInt(header[0]);

  const samePairs = [];
  const diffPairs = [];

  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].trim().split('\t');
    if (parts.length === 0 || parts[0] === '') continue;

    if (parts.length === 3) {
      // Same person pair: Name imgNum1 imgNum2
      samePairs.push({
        person: parts[0],
        img1: parseInt(parts[1]),
        img2: parseInt(parts[2]),
      });
    } else if (parts.length === 4) {
      // Different person pair: Name1 imgNum1 Name2 imgNum2
      diffPairs.push({
        person1: parts[0],
        img1: parseInt(parts[1]),
        person2: parts[2],
        img2: parseInt(parts[3]),
      });
    }
  }

  return { samePairs, diffPairs, numFolds, pairsPerFold };
}

function imgNumToFilename(personName, imgNum) {
  return `${personName}_${String(imgNum).padStart(4, '0')}.jpg`;
}

function getPersonImageCount(personName) {
  const personDir = join(LFW_EXTRACTED_DIR, personName);
  if (!existsSync(personDir)) return 0;
  return readdirSync(personDir).filter(f => f.endsWith('.jpg')).length;
}

function selectDiversePeople(samePairs, diffPairs) {
  // Build a set of people who appear in same-person pairs (they have 2+ photos by definition)
  const samePersonCounts = new Map();
  for (const pair of samePairs) {
    samePersonCounts.set(pair.person, (samePersonCounts.get(pair.person) || 0) + 1);
  }

  // Also collect people from different pairs
  const allPairedPeople = new Set();
  for (const pair of samePairs) allPairedPeople.add(pair.person);
  for (const pair of diffPairs) {
    allPairedPeople.add(pair.person1);
    allPairedPeople.add(pair.person2);
  }

  // Get people with 2+ images who appear in same-person pairs, sorted by pair count
  const candidates = [...samePersonCounts.entries()]
    .map(([name, pairCount]) => ({
      name,
      pairCount,
      imageCount: getPersonImageCount(name),
    }))
    .filter(p => p.imageCount >= 2)
    .sort((a, b) => b.pairCount - a.pairCount);

  console.log(`Found ${candidates.length} candidates with 2+ images in same-person pairs`);

  // Select a diverse subset:
  // - Pick people with varying numbers of photos (some with many, some with few)
  // - Spread across different "first letters" for name diversity
  const selected = [];
  const usedFirstLetters = new Set();

  // Strategy: pick from different ranges of image counts for diversity
  // First, pick some with moderate image counts (3-10) from different letters
  const moderate = candidates.filter(c => c.imageCount >= 3 && c.imageCount <= 15);
  const high = candidates.filter(c => c.imageCount > 15 && c.imageCount <= 50);
  const veryHigh = candidates.filter(c => c.imageCount > 50);

  function pickFromPool(pool, maxPicks) {
    let picked = 0;
    for (const person of pool) {
      if (picked >= maxPicks) break;
      if (selected.find(s => s.name === person.name)) continue;
      const firstLetter = person.name[0];
      // Allow some letter reuse but prefer diversity
      if (usedFirstLetters.has(firstLetter) && picked < maxPicks - 2) {
        // Try to skip if we haven't used up most of our picks
        const sameLetterCount = selected.filter(s => s.name[0] === firstLetter).length;
        if (sameLetterCount >= 2) continue;
      }
      selected.push(person);
      usedFirstLetters.add(firstLetter);
      picked++;
    }
    return picked;
  }

  // Pick people with many photos (good for benchmarking)
  pickFromPool(veryHigh, 6);
  pickFromPool(high, 10);
  pickFromPool(moderate, TARGET_PEOPLE - selected.length);

  // If still under target, pick from any remaining candidates
  if (selected.length < TARGET_PEOPLE) {
    const remaining = candidates.filter(c => !selected.find(s => s.name === c.name));
    pickFromPool(remaining, TARGET_PEOPLE - selected.length);
  }

  return selected;
}

function buildManifest(selectedPeople, samePairs, diffPairs) {
  const selectedNames = new Set(selectedPeople.map(p => p.name));

  // Build people array with all their images
  const people = selectedPeople.map(person => {
    const personDir = join(LFW_EXTRACTED_DIR, person.name);
    const images = readdirSync(personDir)
      .filter(f => f.endsWith('.jpg'))
      .sort()
      .map(f => `/lfw/${person.name}/${f}`);
    return { name: person.name, images };
  });

  // Filter same pairs to those in our selected set
  const filteredSame = samePairs
    .filter(p => selectedNames.has(p.person))
    .map(p => ({
      person: p.person,
      image1: `/lfw/${p.person}/${imgNumToFilename(p.person, p.img1)}`,
      image2: `/lfw/${p.person}/${imgNumToFilename(p.person, p.img2)}`,
    }));

  // Deduplicate and limit same pairs
  const seenSame = new Set();
  const uniqueSame = filteredSame.filter(p => {
    const key = `${p.image1}|${p.image2}`;
    if (seenSame.has(key)) return false;
    seenSame.add(key);
    return true;
  });

  // Take up to TARGET_SAME_PAIRS, trying to spread across people
  const samePairsByPerson = new Map();
  for (const p of uniqueSame) {
    if (!samePairsByPerson.has(p.person)) samePairsByPerson.set(p.person, []);
    samePairsByPerson.get(p.person).push(p);
  }

  const finalSamePairs = [];
  let round = 0;
  while (finalSamePairs.length < TARGET_SAME_PAIRS && finalSamePairs.length < uniqueSame.length) {
    for (const [person, pairs] of samePairsByPerson) {
      if (round < pairs.length && finalSamePairs.length < TARGET_SAME_PAIRS) {
        finalSamePairs.push(pairs[round]);
      }
    }
    round++;
  }

  // Filter different pairs where both people are in our set
  const filteredDiff = diffPairs
    .filter(p => selectedNames.has(p.person1) && selectedNames.has(p.person2))
    .map(p => ({
      person1: p.person1,
      image1: `/lfw/${p.person1}/${imgNumToFilename(p.person1, p.img1)}`,
      person2: p.person2,
      image2: `/lfw/${p.person2}/${imgNumToFilename(p.person2, p.img2)}`,
    }));

  // Deduplicate and limit
  const seenDiff = new Set();
  const uniqueDiff = filteredDiff.filter(p => {
    const key = `${p.image1}|${p.image2}`;
    if (seenDiff.has(key)) return false;
    seenDiff.add(key);
    return true;
  });

  const finalDiffPairs = uniqueDiff.slice(0, TARGET_DIFF_PAIRS);

  // If we don't have enough different pairs from the official pairs,
  // generate some from our selected people
  if (finalDiffPairs.length < TARGET_DIFF_PAIRS) {
    const selectedList = [...selectedNames];
    for (let i = 0; i < selectedList.length && finalDiffPairs.length < TARGET_DIFF_PAIRS; i++) {
      for (let j = i + 1; j < selectedList.length && finalDiffPairs.length < TARGET_DIFF_PAIRS; j++) {
        const p1 = selectedList[i];
        const p2 = selectedList[j];
        const p1Dir = join(LFW_EXTRACTED_DIR, p1);
        const p2Dir = join(LFW_EXTRACTED_DIR, p2);
        const p1Imgs = readdirSync(p1Dir).filter(f => f.endsWith('.jpg')).sort();
        const p2Imgs = readdirSync(p2Dir).filter(f => f.endsWith('.jpg')).sort();
        if (p1Imgs.length > 0 && p2Imgs.length > 0) {
          const candidate = {
            person1: p1,
            image1: `/lfw/${p1}/${p1Imgs[0]}`,
            person2: p2,
            image2: `/lfw/${p2}/${p2Imgs[0]}`,
          };
          const key = `${candidate.image1}|${candidate.image2}`;
          if (!seenDiff.has(key)) {
            seenDiff.add(key);
            finalDiffPairs.push(candidate);
          }
        }
      }
    }
  }

  return {
    source: 'Labeled Faces in the Wild (LFW)',
    license: 'Research use - see http://vis-www.cs.umass.edu/lfw/',
    people,
    pairs: {
      same: finalSamePairs,
      different: finalDiffPairs.slice(0, TARGET_DIFF_PAIRS),
    },
  };
}

function copyImages(manifest) {
  let totalCopied = 0;
  for (const person of manifest.people) {
    const destDir = join(OUTPUT_DIR, person.name);
    mkdirSync(destDir, { recursive: true });

    for (const imgPath of person.images) {
      const filename = basename(imgPath);
      const srcPath = join(LFW_EXTRACTED_DIR, person.name, filename);
      const destPath = join(destDir, filename);
      if (existsSync(srcPath)) {
        copyFileSync(srcPath, destPath);
        totalCopied++;
      } else {
        console.warn(`WARNING: Source image not found: ${srcPath}`);
      }
    }
  }
  return totalCopied;
}

// Main
console.log('=== LFW Dataset Curation ===\n');

const resolvedDir = resolveLfwExtractedDir();
if (!resolvedDir) {
  console.error(`ERROR: No LFW image directory found.`);
  console.error(`  Looked at: ${LFW_EXTRACTED_DIR}`);
  if (existsSync(LFW_BASE)) {
    const contents = readdirSync(LFW_BASE);
    console.error(`  Contents of ${LFW_BASE}: ${contents.join(', ') || '(empty)'}`);
    const hasTgz = contents.some((f) => f.endsWith('.tgz'));
    if (hasTgz) {
      console.error('\n  You have a .tgz file but it may not be extracted. Extract it with:');
      console.error(`  cd ${LFW_BASE} && tar -xzf <filename>.tgz`);
      console.error('  Then from your project directory run: npm run curate-lfw');
    }
    if (contents.length > 0 && !hasTgz) {
      console.error('\n  Try: LFW_SUBDIR=<folder-name> npm run curate-lfw');
    }
  } else {
    console.error(`  Directory ${LFW_BASE} does not exist.`);
  }
  console.error('\n  To download LFW: npm run download-lfw');
  process.exit(1);
}
LFW_EXTRACTED_DIR = resolvedDir;
console.log(`Using: ${LFW_EXTRACTED_DIR}`);
console.log(`Pairs: ${PAIRS_FILE}\n`);

if (!existsSync(PAIRS_FILE)) {
  console.error(`ERROR: pairs.txt not found at ${PAIRS_FILE}`);
  console.error('  Run: npm run download-lfw');
  process.exit(1);
}

console.log('Parsing pairs.txt...');
const { samePairs, diffPairs } = parsePairs(PAIRS_FILE);
console.log(`  Found ${samePairs.length} same-person pairs`);
console.log(`  Found ${diffPairs.length} different-person pairs\n`);

console.log('Selecting diverse subset of people...');
const selectedPeople = selectDiversePeople(samePairs, diffPairs);
console.log(`  Selected ${selectedPeople.length} people:\n`);
for (const p of selectedPeople) {
  console.log(`    ${p.name} (${p.imageCount} images, ${p.pairCount} same-pairs)`);
}

console.log('\nBuilding manifest...');
const manifest = buildManifest(selectedPeople, samePairs, diffPairs);

console.log('\nCopying images...');
mkdirSync(OUTPUT_DIR, { recursive: true });
const totalCopied = copyImages(manifest);

console.log(`  Copied ${totalCopied} images`);

if (manifest.people.length === 0 || totalCopied === 0) {
  console.error('\nERROR: No images found. Ensure LFW is downloaded and extracted:');
  console.error('  npm run download-lfw');
  console.error('  LFW_SUBDIR=lfw-deepfunneled npm run curate-lfw');
  process.exit(1);
}

console.log('\nWriting manifest.json...');
writeFileSync(MANIFEST_PATH, JSON.stringify(manifest, null, 2));

// Summary
const totalImages = manifest.people.reduce((sum, p) => sum + p.images.length, 0);
console.log('\n=== Summary ===');
console.log(`People selected: ${manifest.people.length}`);
console.log(`Total images: ${totalImages}`);
console.log(`Same-person pairs: ${manifest.pairs.same.length}`);
console.log(`Different-person pairs: ${manifest.pairs.different.length}`);
console.log(`\nPeople: ${manifest.people.map(p => p.name).join(', ')}`);
console.log(`\nManifest written to: ${MANIFEST_PATH}`);
console.log(`Images copied to: ${OUTPUT_DIR}`);
