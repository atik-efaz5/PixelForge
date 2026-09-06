/** Bitmap backing-store size for a CSS-sized canvas stage. */
export function canvasBitmapSize(
  stageWidth: number,
  stageHeight: number,
  dpr: number
): { width: number; height: number } {
  const safeDpr = Number.isFinite(dpr) && dpr > 0 ? dpr : 1;
  return {
    width: Math.max(0, Math.floor(Math.max(0, stageWidth) * safeDpr)),
    height: Math.max(0, Math.floor(Math.max(0, stageHeight) * safeDpr)),
  };
}

export type StageSize = { width: number; height: number };

/** Skip redraw when the stage has not changed by a full CSS pixel. */
export function shouldSkipStageRedraw(
  previous: StageSize | null,
  next: StageSize,
  epsilon = 1
): boolean {
  if (!previous) return false;
  return (
    Math.abs(previous.width - next.width) < epsilon &&
    Math.abs(previous.height - next.height) < epsilon
  );
}

export function readStageSize(stage: { clientWidth: number; clientHeight: number }): StageSize {
  return {
    width: stage.clientWidth,
    height: stage.clientHeight,
  };
}
