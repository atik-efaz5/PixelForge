import { describe, expect, it } from "vitest";

import { EMPTY_MASK_REFINE_MESSAGE, requireInpaintMask } from "./maskRefine";

describe("requireInpaintMask", () => {
  it("errors on a missing or empty mask", () => {
    expect(requireInpaintMask(null)).toBe(EMPTY_MASK_REFINE_MESSAGE);
    expect(requireInpaintMask(new Uint8Array([0, 0, 0]))).toBe(
      EMPTY_MASK_REFINE_MESSAGE
    );
  });

  it("allows refine when any inpaint pixel is set", () => {
    expect(requireInpaintMask(new Uint8Array([0, 1, 0]))).toBeNull();
  });
});
