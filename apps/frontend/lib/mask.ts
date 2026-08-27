import type { ImageDimensions } from "@/types/api";

const MASK_THRESHOLD = 128;

/** Decode a mask PNG blob into a boolean H×W grid (true = inpaint). */
export async function decodeMaskPng(
  blob: Blob,
  expected?: ImageDimensions
): Promise<Uint8Array> {
  const bitmap = await createImageBitmap(blob);
  const width = bitmap.width;
  const height = bitmap.height;
  if (
    expected &&
    (width !== expected.width || height !== expected.height)
  ) {
    bitmap.close();
    throw new Error(
      `Mask size ${width}×${height} does not match image ${expected.width}×${expected.height}.`
    );
  }

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    bitmap.close();
    throw new Error("Could not create mask canvas.");
  }
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();

  const { data } = ctx.getImageData(0, 0, width, height);
  const mask = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i += 1) {
    mask[i] = data[i * 4] >= MASK_THRESHOLD ? 1 : 0;
  }
  return mask;
}

/** Create an empty mask (all preserve). */
export function createEmptyMask(width: number, height: number): Uint8Array {
  return new Uint8Array(width * height);
}

/** Deep copy of a mask buffer. */
export function cloneMask(mask: Uint8Array): Uint8Array {
  return new Uint8Array(mask);
}

/** Morphological dilation (expand mask). */
export function dilateMask(
  mask: Uint8Array,
  width: number,
  height: number,
  iterations: number
): Uint8Array {
  if (iterations <= 0) return cloneMask(mask);
  let out = cloneMask(mask);
  for (let n = 0; n < iterations; n += 1) {
    const next = new Uint8Array(out.length);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const idx = y * width + x;
        if (out[idx]) {
          next[idx] = 1;
          continue;
        }
        let on = false;
        for (let dy = -1; dy <= 1 && !on; dy += 1) {
          for (let dx = -1; dx <= 1; dx += 1) {
            const nx = x + dx;
            const ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
            if (out[ny * width + nx]) {
              on = true;
              break;
            }
          }
        }
        next[idx] = on ? 1 : 0;
      }
    }
    out = next;
  }
  return out;
}

/** Morphological erosion (shrink mask). */
export function erodeMask(
  mask: Uint8Array,
  width: number,
  height: number,
  iterations: number
): Uint8Array {
  if (iterations <= 0) return cloneMask(mask);
  const inverted = new Uint8Array(mask.length);
  for (let i = 0; i < mask.length; i += 1) {
    inverted[i] = mask[i] ? 0 : 1;
  }
  const erodedInv = dilateMask(inverted, width, height, iterations);
  const out = new Uint8Array(mask.length);
  for (let i = 0; i < mask.length; i += 1) {
    out[i] = erodedInv[i] ? 0 : 1;
  }
  return out;
}

/** Paint a circular brush stroke onto the mask. */
export function paintBrush(
  mask: Uint8Array,
  width: number,
  height: number,
  centerX: number,
  centerY: number,
  radius: number,
  mode: "brush" | "erase"
): void {
  const r2 = radius * radius;
  const minX = Math.max(0, Math.floor(centerX - radius));
  const maxX = Math.min(width - 1, Math.ceil(centerX + radius));
  const minY = Math.max(0, Math.floor(centerY - radius));
  const maxY = Math.min(height - 1, Math.ceil(centerY + radius));
  const value = mode === "brush" ? 1 : 0;

  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      const dx = x - centerX;
      const dy = y - centerY;
      if (dx * dx + dy * dy <= r2) {
        mask[y * width + x] = value;
      }
    }
  }
}

/** Encode boolean mask to PNG blob (255 = inpaint, 0 = preserve). */
export async function encodeMaskPng(
  mask: Uint8Array,
  width: number,
  height: number
): Promise<Blob> {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Could not encode mask.");
  }
  const imageData = ctx.createImageData(width, height);
  for (let i = 0; i < mask.length; i += 1) {
    const v = mask[i] ? 255 : 0;
    const idx = i * 4;
    imageData.data[idx] = v;
    imageData.data[idx + 1] = v;
    imageData.data[idx + 2] = v;
    imageData.data[idx + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  const blob = await new Promise<Blob | null>((resolve) => {
    canvas.toBlob((b) => resolve(b), "image/png");
  });
  if (!blob) {
    throw new Error("Failed to encode mask PNG.");
  }
  return blob;
}

/** Render mask overlay onto a canvas context (translucent highlight). */
export function drawMaskOverlay(
  ctx: CanvasRenderingContext2D,
  mask: Uint8Array,
  width: number,
  height: number,
  layout: { offsetX: number; offsetY: number; scale: number; renderedWidth: number; renderedHeight: number },
  options?: { featherRadius?: number }
): void {
  const overlay = document.createElement("canvas");
  overlay.width = width;
  overlay.height = height;
  const octx = overlay.getContext("2d");
  if (!octx) return;

  const feather = options?.featherRadius ?? 0;
  if (feather > 0) {
    const imageData = octx.createImageData(width, height);
    const dist = computeFeatherDistances(mask, width, height, feather);
    for (let i = 0; i < mask.length; i += 1) {
      if (!mask[i] && dist[i] <= 0) continue;
      const alpha = mask[i]
        ? 180
        : Math.max(0, Math.round(140 * (1 - dist[i] / feather)));
      if (alpha <= 0) continue;
      const idx = i * 4;
      imageData.data[idx] = 56;
      imageData.data[idx + 1] = 189;
      imageData.data[idx + 2] = 248;
      imageData.data[idx + 3] = alpha;
    }
    octx.putImageData(imageData, 0, 0);
  } else {
    const imageData = octx.createImageData(width, height);
    for (let i = 0; i < mask.length; i += 1) {
      if (!mask[i]) continue;
      const idx = i * 4;
      imageData.data[idx] = 56;
      imageData.data[idx + 1] = 189;
      imageData.data[idx + 2] = 248;
      imageData.data[idx + 3] = 140;
    }
    octx.putImageData(imageData, 0, 0);
  }

  ctx.drawImage(
    overlay,
    layout.offsetX,
    layout.offsetY,
    layout.renderedWidth,
    layout.renderedHeight
  );
}

/** Distance from each pixel to the nearest inpaint pixel (for feather preview). */
function computeFeatherDistances(
  mask: Uint8Array,
  width: number,
  height: number,
  maxRadius: number
): Float32Array {
  const dist = new Float32Array(mask.length);
  dist.fill(maxRadius + 1);
  const queue: number[] = [];
  for (let i = 0; i < mask.length; i += 1) {
    if (mask[i]) {
      dist[i] = 0;
      queue.push(i);
    }
  }
  let head = 0;
  while (head < queue.length) {
    const idx = queue[head];
    head += 1;
    const x = idx % width;
    const y = Math.floor(idx / width);
    const base = dist[idx];
    if (base >= maxRadius) continue;
    const neighbors = [
      [x - 1, y],
      [x + 1, y],
      [x, y - 1],
      [x, y + 1],
    ];
    for (const [nx, ny] of neighbors) {
      if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
      const nidx = ny * width + nx;
      const next = base + 1;
      if (next < dist[nidx]) {
        dist[nidx] = next;
        queue.push(nidx);
      }
    }
  }
  return dist;
}

/** Check whether any inpaint pixels are set. */
export function maskHasInpaint(mask: Uint8Array): boolean {
  for (let i = 0; i < mask.length; i += 1) {
    if (mask[i]) return true;
  }
  return false;
}

/** Render mask alone as a grayscale preview data URL. */
export function maskPreviewUrl(
  mask: Uint8Array,
  width: number,
  height: number,
  options?: { featherRadius?: number }
): string {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";
  const imageData = ctx.createImageData(width, height);
  const feather = options?.featherRadius ?? 0;
  if (feather > 0) {
    const dist = computeFeatherDistances(mask, width, height, feather);
    for (let i = 0; i < mask.length; i += 1) {
      let v = 0;
      if (mask[i]) {
        v = 255;
      } else if (dist[i] > 0 && dist[i] <= feather) {
        v = Math.round(255 * (1 - dist[i] / feather));
      }
      const idx = i * 4;
      imageData.data[idx] = v;
      imageData.data[idx + 1] = v;
      imageData.data[idx + 2] = v;
      imageData.data[idx + 3] = 255;
    }
  } else {
    for (let i = 0; i < mask.length; i += 1) {
      const v = mask[i] ? 255 : 0;
      const idx = i * 4;
      imageData.data[idx] = v;
      imageData.data[idx + 1] = v;
      imageData.data[idx + 2] = v;
      imageData.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL("image/png");
}

/** Bounded undo/redo history for mask edits. */
export class MaskEditHistory {
  private undoStack: Uint8Array[] = [];
  private redoStack: Uint8Array[] = [];

  constructor(private readonly maxEntries = 50) {}

  canUndo(): boolean {
    return this.undoStack.length > 0;
  }

  canRedo(): boolean {
    return this.redoStack.length > 0;
  }

  clear(): void {
    this.undoStack = [];
    this.redoStack = [];
  }

  push(mask: Uint8Array): void {
    this.undoStack.push(cloneMask(mask));
    this.redoStack = [];
    if (this.undoStack.length > this.maxEntries) {
      this.undoStack.shift();
    }
  }

  undo(current: Uint8Array): Uint8Array | null {
    if (!this.canUndo()) return null;
    this.redoStack.push(cloneMask(current));
    return this.undoStack.pop() ?? null;
  }

  redo(current: Uint8Array): Uint8Array | null {
    if (!this.canRedo()) return null;
    this.undoStack.push(cloneMask(current));
    return this.redoStack.pop() ?? null;
  }
}
