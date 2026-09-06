import { maskHasInpaint } from "./mask";

export const EMPTY_MASK_REFINE_MESSAGE = "Select or brush a region first.";

export function requireInpaintMask(mask: Uint8Array | null): string | null {
  if (!mask || !maskHasInpaint(mask)) {
    return EMPTY_MASK_REFINE_MESSAGE;
  }
  return null;
}
