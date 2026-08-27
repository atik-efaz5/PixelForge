import type { ImageDimensions } from "@/types/api";
import {
  computeDisplayLayout,
  type DisplayLayout,
} from "@/lib/coordinates";

export interface CanvasViewport {
  zoom: number;
  panX: number;
  panY: number;
}

export const MIN_ZOOM = 0.25;
export const MAX_ZOOM = 8;
export const ZOOM_STEP = 1.2;

export function fitViewport(): CanvasViewport {
  return { zoom: 1, panX: 0, panY: 0 };
}

export function clampZoom(zoom: number): number {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom));
}

/** Letterbox fit with optional zoom and pan offsets. */
export function computeViewLayout(
  container: ImageDimensions,
  image: ImageDimensions,
  viewport: CanvasViewport
): DisplayLayout {
  const base = computeDisplayLayout(container, image);
  const scale = base.scale * viewport.zoom;
  const renderedWidth = image.width * scale;
  const renderedHeight = image.height * scale;
  return {
    scale,
    offsetX: (container.width - renderedWidth) / 2 + viewport.panX,
    offsetY: (container.height - renderedHeight) / 2 + viewport.panY,
    renderedWidth,
    renderedHeight,
  };
}

export function zoomIn(viewport: CanvasViewport): CanvasViewport {
  return { ...viewport, zoom: clampZoom(viewport.zoom * ZOOM_STEP) };
}

export function zoomOut(viewport: CanvasViewport): CanvasViewport {
  return { ...viewport, zoom: clampZoom(viewport.zoom / ZOOM_STEP) };
}

/** Zoom toward a pointer position inside the canvas container. */
export function zoomViewportAtPoint(
  viewport: CanvasViewport,
  factor: number,
  pointerX: number,
  pointerY: number,
  layout: DisplayLayout,
  container: ImageDimensions,
  image: ImageDimensions
): CanvasViewport {
  const nextZoom = clampZoom(viewport.zoom * factor);
  if (nextZoom === viewport.zoom) {
    return viewport;
  }

  const imgX = (pointerX - layout.offsetX) / layout.scale;
  const imgY = (pointerY - layout.offsetY) / layout.scale;
  const baseScale = computeDisplayLayout(container, image).scale;
  const newScale = baseScale * nextZoom;
  const renderedWidth = image.width * newScale;
  const renderedHeight = image.height * newScale;
  const centeredX = (container.width - renderedWidth) / 2;
  const centeredY = (container.height - renderedHeight) / 2;

  return {
    zoom: nextZoom,
    panX: pointerX - imgX * newScale - centeredX,
    panY: pointerY - imgY * newScale - centeredY,
  };
}
