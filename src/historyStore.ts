import type { CompareResult } from './faceComparison'

const DB_NAME = 'face-compare-history'
const DB_VERSION = 1
const STORE_NAME = 'comparisons'

export interface HistoryEntry {
  id: number
  timestamp: number
  image1Thumb: string
  image2Thumb: string
  verdict: CompareResult['verdict']
  score: number
  confidence: number | undefined
  cosineSimilarity: number | undefined
  fullResult: CompareResult
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION)
    req.onerror = () => reject(req.error)
    req.onsuccess = () => resolve(req.result)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'id', autoIncrement: true })
      }
    }
  })
}

/**
 * Resizes an image to ~maxSize and returns a data URL.
 */
export async function generateThumbnail(
  imageUrl: string,
  maxSize: number = 100
): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      const w = img.naturalWidth
      const h = img.naturalHeight
      const scale = Math.min(maxSize / w, maxSize / h, 1)
      const tw = Math.round(w * scale)
      const th = Math.round(h * scale)
      const canvas = document.createElement('canvas')
      canvas.width = tw
      canvas.height = th
      const ctx = canvas.getContext('2d')
      if (!ctx) {
        reject(new Error('Could not get canvas context'))
        return
      }
      ctx.drawImage(img, 0, 0, tw, th)
      try {
        resolve(canvas.toDataURL('image/jpeg', 0.7))
      } catch (e) {
        reject(e)
      }
    }
    img.onerror = () => reject(new Error('Failed to load image for thumbnail'))
    img.src = imageUrl
  })
}

/**
 * Saves a comparison to history with thumbnail generation.
 * Returns the assigned id.
 */
export async function saveComparison(
  result: CompareResult,
  image1Url: string,
  image2Url: string
): Promise<number> {
  const [image1Thumb, image2Thumb] = await Promise.all([
    generateThumbnail(image1Url),
    generateThumbnail(image2Url),
  ])

  const entry: Omit<HistoryEntry, 'id'> = {
    timestamp: Date.now(),
    image1Thumb,
    image2Thumb,
    verdict: result.verdict,
    score: result.score,
    confidence: result.confidence,
    cosineSimilarity: result.cosineSimilarity,
    fullResult: result,
  }

  const db = await openDb()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    const store = tx.objectStore(STORE_NAME)
    const req = store.add(entry)
    req.onerror = () => reject(req.error)
    req.onsuccess = () => resolve(req.result as number)
    tx.oncomplete = () => db.close()
  })
}

/**
 * Returns all history entries sorted by timestamp descending.
 */
export async function getHistory(): Promise<HistoryEntry[]> {
  const db = await openDb()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly')
    const store = tx.objectStore(STORE_NAME)
    const req = store.getAll()
    req.onerror = () => reject(req.error)
    req.onsuccess = () => {
      const entries = (req.result as HistoryEntry[]).sort(
        (a, b) => b.timestamp - a.timestamp
      )
      resolve(entries)
    }
    tx.oncomplete = () => db.close()
  })
}

/**
 * Deletes a single comparison by id.
 */
export async function deleteComparison(id: number): Promise<void> {
  const db = await openDb()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    const store = tx.objectStore(STORE_NAME)
    const req = store.delete(id)
    req.onerror = () => reject(req.error)
    req.onsuccess = () => resolve()
    tx.oncomplete = () => db.close()
  })
}

/**
 * Clears all history.
 */
export async function clearHistory(): Promise<void> {
  const db = await openDb()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    const store = tx.objectStore(STORE_NAME)
    const req = store.clear()
    req.onerror = () => reject(req.error)
    req.onsuccess = () => resolve()
    tx.oncomplete = () => db.close()
  })
}
