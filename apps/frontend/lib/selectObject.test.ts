import { describe, expect, it } from "vitest";

import { canSubmitFindObject, selectionModeAfterTyping } from "./selectObject";

describe("selectionModeAfterTyping", () => {
  it("switches Point to Text when the user types a prompt", () => {
    expect(selectionModeAfterTyping("point", "black bag")).toBe("text");
  });

  it("switches Smart to Text when the user types a prompt", () => {
    expect(selectionModeAfterTyping("smart", "black bag")).toBe("text");
  });

  it("keeps Text unchanged", () => {
    expect(selectionModeAfterTyping("text", "black bag")).toBe("text");
  });
});

describe("canSubmitFindObject", () => {
  it("allows Enter/Find when there is a prompt and the editor is idle", () => {
    expect(canSubmitFindObject(false, "black bag")).toBe(true);
  });

  it("blocks empty prompts, busy state, and missing image", () => {
    expect(canSubmitFindObject(false, "  ")).toBe(false);
    expect(canSubmitFindObject(true, "black bag")).toBe(false);
    expect(canSubmitFindObject(false, "black bag", false)).toBe(false);
  });
});
