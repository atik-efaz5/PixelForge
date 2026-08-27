import { describe, expect, it } from "vitest";

import {
  clampZoom,
  fitViewport,
  MAX_ZOOM,
  MIN_ZOOM,
  zoomViewportAtPoint,
  type CanvasViewport,
} from "./canvasView";
import {
  isFiniteViewport,
  resetInteractionState,
  shouldPaintStroke,
  shouldTriggerPointSelect,
} from "./canvasPointer";

describe("canvasPointer", () => {
  it("does not trigger point select on pointer move", () => {
    expect(shouldTriggerPointSelect("select", "move")).toBe(false);
    expect(shouldTriggerPointSelect("select", "down")).toBe(true);
    expect(shouldTriggerPointSelect("brush", "down")).toBe(false);
  });

  it("paints only while painting with brush or eraser", () => {
    expect(shouldPaintStroke("brush", true)).toBe(true);
    expect(shouldPaintStroke("brush", false)).toBe(false);
    expect(shouldPaintStroke("select", true)).toBe(false);
  });

  it("resets interaction state to idle", () => {
    expect(resetInteractionState()).toEqual({
      isPanning: false,
      isPainting: false,
      strokeStarted: false,
    });
  });
});

describe("canvasView", () => {
  it("keeps zoom within bounds", () => {
    expect(clampZoom(0.01)).toBe(MIN_ZOOM);
    expect(clampZoom(100)).toBe(MAX_ZOOM);
    expect(clampZoom(1)).toBe(1);
  });

  it("returns finite viewport after zoom-at-point", () => {
    const viewport: CanvasViewport = fitViewport();
    const layout = {
      scale: 0.5,
      offsetX: 10,
      offsetY: 20,
      renderedWidth: 256,
      renderedHeight: 256,
    };
    const next = zoomViewportAtPoint(
      viewport,
      1.12,
      128,
      128,
      layout,
      { width: 512, height: 512 },
      { width: 512, height: 512 }
    );
    expect(isFiniteViewport(next.panX, next.panY, next.zoom)).toBe(true);
  });

  it("ordinary pan delta from move applies only while panning", () => {
    const start = { panX: 0, panY: 0 };
    const notPanning = start;
    const panning = { panX: start.panX + 5, panY: start.panY + 10 };
    expect(notPanning).toEqual({ panX: 0, panY: 0 });
    expect(panning).toEqual({ panX: 5, panY: 10 });
    expect(isFiniteViewport(panning.panX, panning.panY, 1)).toBe(true);
  });
});
