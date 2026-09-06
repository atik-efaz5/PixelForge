import { describe, expect, it } from "vitest";

import {
  canvasBitmapSize,
  shouldSkipStageRedraw,
} from "./canvasStage";

describe("canvasBitmapSize", () => {
  it("floors stage CSS pixels times devicePixelRatio", () => {
    expect(canvasBitmapSize(400, 300, 2)).toEqual({ width: 800, height: 600 });
    expect(canvasBitmapSize(100.9, 50.2, 1)).toEqual({ width: 100, height: 50 });
  });

  it("never returns a negative bitmap size", () => {
    expect(canvasBitmapSize(-10, -4, 2)).toEqual({ width: 0, height: 0 });
    expect(canvasBitmapSize(10, 10, 0)).toEqual({ width: 10, height: 10 });
  });
});

describe("shouldSkipStageRedraw", () => {
  it("redraws when there is no previous size", () => {
    expect(shouldSkipStageRedraw(null, { width: 100, height: 80 })).toBe(false);
  });

  it("skips when the stage size is unchanged", () => {
    expect(
      shouldSkipStageRedraw(
        { width: 640, height: 480 },
        { width: 640, height: 480 }
      )
    ).toBe(true);
  });

  it("skips sub-pixel jitter under 1px", () => {
    expect(
      shouldSkipStageRedraw(
        { width: 640, height: 480 },
        { width: 640.4, height: 479.7 }
      )
    ).toBe(true);
  });

  it("redraws when the stage changes by 2px", () => {
    expect(
      shouldSkipStageRedraw(
        { width: 640, height: 480 },
        { width: 640, height: 478 }
      )
    ).toBe(false);
  });
});
