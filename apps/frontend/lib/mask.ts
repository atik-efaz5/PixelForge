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
    // White / bright pixels = inpaint.
    mask[i] = data[i * 4] >= MASK_THRESHOLD ? 1 : 0;
  }
  return mask;
}

/** Create an empty mask (all preserve). */
export function createEmptyMask(width: number, height: number): Uint8Array {
  return new Uint8Array(width * height);
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
  layout: { offsetX: number; offsetY: number; scale: number; renderedWidth: number; renderedHeight: number }
): void {
  const overlay = document.createElement("canvas");
  overlay.width = width;
  overlay.height = height;
  const octx = overlay.getContext("2d");
  if (!octx) return;

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

  ctx.drawImage(
    overlay,
    layout.offsetX,
    layout.offsetY,
    layout.renderedWidth,
    layout.renderedHeight
  );
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
  height: number
): string {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";
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
  return canvas.toDataURL("image/png");
}
