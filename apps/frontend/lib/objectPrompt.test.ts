import { describe, expect, it } from "vitest";

import { canAttemptLocalRemove, extractObjectPrompt } from "./objectPrompt";

describe("extractObjectPrompt", () => {
  it("strips polite removal phrasing", () => {
    expect(extractObjectPrompt("please remove the black bag")).toBe("black bag");
    expect(extractObjectPrompt("Remove the black bag.")).toBe("black bag");
  });

  it("keeps a plain object phrase", () => {
    expect(extractObjectPrompt("black bag")).toBe("black bag");
  });
});

describe("canAttemptLocalRemove", () => {
  it("allows fill when a mask already exists", () => {
    expect(
      canAttemptLocalRemove({
        hasImage: true,
        hasInpaintMask: true,
        instruction: "",
        selectPrompt: "",
      })
    ).toBe(true);
  });

  it("is false without an image", () => {
    expect(
      canAttemptLocalRemove({
        hasImage: false,
        hasInpaintMask: true,
        instruction: "black bag",
        selectPrompt: "",
      })
    ).toBe(false);
  });

  it("allows fill from an instruction or select prompt", () => {
    expect(
      canAttemptLocalRemove({
        hasImage: true,
        hasInpaintMask: false,
        instruction: "please remove the black bag",
        selectPrompt: "",
      })
    ).toBe(true);
  });
});
