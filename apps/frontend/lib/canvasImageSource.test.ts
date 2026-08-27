import { describe, expect, it } from "vitest";

import { shouldReloadCanvasImage } from "./canvasImageSource";

describe("shouldReloadCanvasImage", () => {
  it("reloads when source URL changes", () => {
    expect(shouldReloadCanvasImage("blob:a", "blob:b", true)).toBe(true);
  });

  it("reloads when no image has been decoded yet", () => {
    expect(shouldReloadCanvasImage("blob:a", "blob:a", false)).toBe(true);
  });

  it("skips reload when URL is unchanged and image is already decoded", () => {
    expect(shouldReloadCanvasImage("blob:a", "blob:a", true)).toBe(false);
  });
});
