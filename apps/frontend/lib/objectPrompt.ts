/** Turn a removal-style instruction into a Grounding DINO / Find Object query. */
export function extractObjectPrompt(instruction: string): string {
  let text = instruction.trim();
  if (!text) return "";
  text = text.replace(/^(please\s+)+/i, "");
  text = text.replace(/^(can you|could you)\s+/i, "");
  text = text.replace(/^(remove|delete|erase|inpaint|fill)\s+/i, "");
  text = text.replace(/^the\s+/i, "");
  text = text.replace(/\s+(from the image|from this (photo|picture)|please)\.?$/i, "");
  text = text.replace(/\.+$/g, "");
  return text.trim();
}

export function canAttemptLocalRemove(options: {
  hasImage: boolean;
  hasInpaintMask: boolean;
  instruction: string;
  selectPrompt: string;
}): boolean {
  if (!options.hasImage) return false;
  if (options.hasInpaintMask) return true;
  return Boolean(
    extractObjectPrompt(options.instruction) || options.selectPrompt.trim()
  );
}
