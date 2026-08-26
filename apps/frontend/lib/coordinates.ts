import type { ImageDimensions } from "@/types/api";

export interface DisplayLayout {
  scale: number;
  offsetX: number;
  offsetY: number;
  renderedWidth: number;
  renderedHeight: number;
}

/** Letterboxed layout for an image inside a container. */
export function computeDisplayLayout(
  container: ImageDimensions,
  image: ImageDimensions
): DisplayLayout {
  const scale = Math.min(
    container.width / image.width,
    container.height / image.height
  );
  const renderedWidth = image.width * scale;
  const renderedHeight = image.height * scale;
  return {
    scale,
    offsetX: (container.width - renderedWidth) / 2,
    offsetY: (container.height - renderedHeight) / 2,
    renderedWidth,
    renderedHeight,
  };
}

/** Map pointer position inside the canvas to source-image pixel coordinates. */
export function pointerToImageCoords(
  pointerX: number,
  pointerY: number,
  layout: DisplayLayout,
  image: ImageDimensions
): { x: number; y: number } | null {
  const relX = pointerX - layout.offsetX;
  const relY = pointerY - layout.offsetY;
  if (
    relX < 0 ||
    relY < 0 ||
    relX > layout.renderedWidth ||
    relY > layout.renderedHeight
  ) {
    return null;
  }
  const x = Math.min(
    Math.max(0, Math.floor(relX / layout.scale)),
    image.width - 1
  );
  const y = Math.min(
    Math.max(0, Math.floor(relY / layout.scale)),
    image.height - 1
  );
  return { x, y };
}
