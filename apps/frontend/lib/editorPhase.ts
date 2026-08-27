import type { EditorStatus, EditorTool } from "@/types/api";

export type EditorPhase =
  | "idle"
  | "uploading"
  | "segmenting"
  | "mask_ready"
  | "refining"
  | "generating"
  | "result_ready"
  | "error";

export function deriveEditorPhase(input: {
  status: EditorStatus;
  error: string | null;
  hasMask: boolean;
  hasPendingResult: boolean;
  tool: EditorTool;
}): EditorPhase {
  if (input.error) return "error";
  if (input.status === "uploading") return "uploading";
  if (input.status === "segmenting" || input.status === "grounding") {
    return "segmenting";
  }
  if (input.status === "generating" || input.status === "instruction_editing") {
    return "generating";
  }
  if (input.hasPendingResult) return "result_ready";
  if (input.tool === "brush" || input.tool === "erase") return "refining";
  if (input.hasMask) return "mask_ready";
  return "idle";
}

export const PHASE_LABELS: Record<EditorPhase, string> = {
  idle: "Ready",
  uploading: "Uploading image…",
  segmenting: "Segmenting…",
  mask_ready: "Mask ready — refine or generate",
  refining: "Refining mask",
  generating: "Generating result…",
  result_ready: "Result ready — accept or discard",
  error: "Error",
};
