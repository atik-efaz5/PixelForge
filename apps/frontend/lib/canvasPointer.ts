import type { EditorTool } from "@/types/api";

/** Point selection runs on pointer down only — never on hover/move. */
export function shouldTriggerPointSelect(
  tool: EditorTool,
  phase: "down" | "move"
): boolean {
  return tool === "select" && phase === "down";
}

export function shouldPaintStroke(tool: EditorTool, isPainting: boolean): boolean {
  return isPainting && (tool === "brush" || tool === "erase");
}

export interface InteractionState {
  isPanning: boolean;
  isPainting: boolean;
  strokeStarted: boolean;
}

export const IDLE_INTERACTION: InteractionState = {
  isPanning: false,
  isPainting: false,
  strokeStarted: false,
};

/** Clear all transient pointer interaction flags. */
export function resetInteractionState(): InteractionState {
  return { ...IDLE_INTERACTION };
}

export function isFiniteViewport(panX: number, panY: number, zoom: number): boolean {
  return Number.isFinite(panX) && Number.isFinite(panY) && Number.isFinite(zoom) && zoom > 0;
}
