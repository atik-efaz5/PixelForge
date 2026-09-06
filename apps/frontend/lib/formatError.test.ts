import { describe, expect, it } from "vitest";

import { toUserFacingError } from "./formatError";

describe("toUserFacingError", () => {
  it("points to Localized Fill when InstructPix2Pix is unavailable", () => {
    const err = toUserFacingError(
      new Error(
        "Requested backend 'instruct_pix2pix' is not available for global_instruction_edit."
      )
    );
    expect(err.message).toMatch(/instruct_pix2pix/i);
    expect(err.recovery).toMatch(/Localized Fill/i);
  });

  it("preserves plain string errors", () => {
    const err = toUserFacingError("Select or brush a region first.");
    expect(err.message).toBe("Select or brush a region first.");
    expect(err.recovery).toMatch(/Localized Fill/i);
  });

  it("points to click-to-select when Grounding DINO fails to load", () => {
    const err = toUserFacingError(new Error("Grounding DINO failed to load."));
    expect(err.message).toMatch(/Grounding DINO/i);
    expect(err.recovery).toMatch(/Click the object/i);
    expect(err.recovery).not.toMatch(/refine the mask/i);
  });

  it("points to Fill retry when Moebius fails to load", () => {
    const err = toUserFacingError(new Error("Moebius failed to load."));
    expect(err.recovery).toMatch(/Fill selected region/i);
    expect(err.recovery).not.toMatch(/refine the mask/i);
  });

  it("explains when the hosted UI cannot reach the API", () => {
    const err = toUserFacingError(new Error("Failed to fetch"));
    expect(err.message).toMatch(/cannot reach/i);
    expect(err.recovery).toMatch(/tunnel/i);
  });

  it("keeps generic unavailable recovery for other backends", () => {
    const err = toUserFacingError(new Error("SAM 2 is not available."));
    expect(err.recovery).toMatch(/local models/i);
  });
});
