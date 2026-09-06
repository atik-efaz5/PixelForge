import type { SelectionMode } from "@/types/api";

/** Typing a prompt switches to Text so Find Object is not ignored in Smart/Point. */
export function selectionModeAfterTyping(
  current: SelectionMode,
  prompt: string
): SelectionMode {
  if (prompt.trim() && current !== "text") return "text";
  return current;
}

export function canSubmitFindObject(
  busy: boolean,
  prompt: string,
  hasImage = true
): boolean {
  return !busy && hasImage && Boolean(prompt.trim());
}
