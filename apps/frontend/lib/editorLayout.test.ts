import { describe, expect, it } from "vitest";

import { EDITOR_CANVAS_MIN_HEIGHT_PX } from "./editorLayout";

describe("editorLayout", () => {
  it("reserves a minimum canvas height floor", () => {
    expect(EDITOR_CANVAS_MIN_HEIGHT_PX).toBeGreaterThanOrEqual(400);
  });
});
