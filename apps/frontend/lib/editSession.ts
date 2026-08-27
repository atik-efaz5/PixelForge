import type { InpaintBackend } from "@/types/api";
import { cloneMask } from "@/lib/mask";

export type EditOperation =
  | "ORIGINAL"
  | "SELECT"
  | "MASK_EDIT"
  | "MASK_RESET"
  | "INPAINT"
  | "RESULT_ACCEPTED"
  | "RESULT_REJECTED";

export interface SelectionMetadata {
  method: "point" | "text";
  prompt?: string;
  label?: string;
  detectionIndex?: number;
}

export interface InpaintSessionMetadata {
  backend: InpaintBackend | string;
  model?: string;
  latencyMs?: number;
}

export interface EditSessionSnapshot {
  id: string;
  operation: EditOperation;
  label: string;
  timestamp: number;
  /** Immutable original image object URL for this session. */
  originalUrl: string;
  mask: Uint8Array | null;
  aiMask: Uint8Array | null;
  resultUrl: string | null;
  selection?: SelectionMetadata;
  inpaint?: InpaintSessionMetadata;
}

export interface AppendSnapshotInput {
  operation: EditOperation;
  label: string;
  mask?: Uint8Array | null;
  aiMask?: Uint8Array | null;
  resultUrl?: string | null;
  selection?: SelectionMetadata;
  inpaint?: InpaintSessionMetadata;
}

function cloneMaskOrNull(mask: Uint8Array | null | undefined): Uint8Array | null {
  if (!mask) return null;
  return cloneMask(mask);
}

function newId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `snap-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

/** Session-level non-destructive edit history. */
export class EditSessionHistory {
  private entries: EditSessionSnapshot[] = [];
  private index = -1;
  private originalUrl: string | null = null;
  private readonly maxEntries: number;
  private readonly ownedResultUrls = new Set<string>();

  constructor(maxEntries = 30) {
    this.maxEntries = Math.max(1, maxEntries);
  }

  get snapshotEntries(): readonly EditSessionSnapshot[] {
    return this.entries;
  }

  get currentIndex(): number {
    return this.index;
  }

  get currentOriginalUrl(): string | null {
    return this.originalUrl;
  }

  canUndo(): boolean {
    return this.index > 0;
  }

  canRedo(): boolean {
    return this.index >= 0 && this.index < this.entries.length - 1;
  }

  current(): EditSessionSnapshot | null {
    if (this.index < 0 || this.index >= this.entries.length) return null;
    return this.entries[this.index];
  }

  reset(originalUrl: string): EditSessionSnapshot {
    this.disposeOwnedResults();
    this.entries = [];
    this.index = -1;
    this.originalUrl = originalUrl;
    return this.append({
      operation: "ORIGINAL",
      label: "Original",
      mask: null,
      aiMask: null,
      resultUrl: null,
    });
  }

  append(input: AppendSnapshotInput): EditSessionSnapshot {
    if (!this.originalUrl) {
      throw new Error("Edit session not initialized.");
    }

    if (this.index < this.entries.length - 1) {
      this.truncateForwardBranch();
    }

    const entry: EditSessionSnapshot = {
      id: newId(),
      operation: input.operation,
      label: input.label,
      timestamp: Date.now(),
      originalUrl: this.originalUrl,
      mask: cloneMaskOrNull(input.mask ?? null),
      aiMask: cloneMaskOrNull(input.aiMask ?? null),
      resultUrl: input.resultUrl ?? null,
      selection: input.selection,
      inpaint: input.inpaint,
    };

    if (entry.resultUrl?.startsWith("blob:")) {
      this.ownedResultUrls.add(entry.resultUrl);
    }

    this.entries.push(entry);
    this.index = this.entries.length - 1;

    if (this.entries.length > this.maxEntries) {
      const dropped = this.entries.shift();
      this.index -= 1;
      if (dropped?.resultUrl) {
        this.revokeResultUrl(dropped.resultUrl);
      }
    }

    return entry;
  }

  undo(): EditSessionSnapshot | null {
    if (!this.canUndo()) return null;
    this.index -= 1;
    return this.current();
  }

  redo(): EditSessionSnapshot | null {
    if (!this.canRedo()) return null;
    this.index += 1;
    return this.current();
  }

  goTo(index: number): EditSessionSnapshot | null {
    if (index < 0 || index >= this.entries.length) return null;
    this.index = index;
    return this.current();
  }

  previousResultSnapshot(): EditSessionSnapshot | null {
    for (let idx = this.index - 1; idx >= 0; idx -= 1) {
      const entry = this.entries[idx];
      if (
        entry.resultUrl &&
        (entry.operation === "INPAINT" || entry.operation === "RESULT_ACCEPTED")
      ) {
        return entry;
      }
    }
    return null;
  }

  releaseResult(url: string): void {
    this.revokeResultUrl(url);
    for (const entry of this.entries) {
      if (entry.resultUrl === url) {
        entry.resultUrl = null;
      }
    }
  }

  dispose(): void {
    this.disposeOwnedResults();
    this.entries = [];
    this.index = -1;
    this.originalUrl = null;
  }

  private truncateForwardBranch(): void {
    const removed = this.entries.slice(this.index + 1);
    for (const entry of removed) {
      if (entry.resultUrl) {
        this.revokeResultUrl(entry.resultUrl);
      }
    }
    this.entries = this.entries.slice(0, this.index + 1);
  }

  private disposeOwnedResults(): void {
    for (const url of this.ownedResultUrls) {
      URL.revokeObjectURL(url);
    }
    this.ownedResultUrls.clear();
  }

  private revokeResultUrl(url: string): void {
    if (!this.ownedResultUrls.has(url)) return;
    URL.revokeObjectURL(url);
    this.ownedResultUrls.delete(url);
  }
}

export function operationLabel(operation: EditOperation): string {
  switch (operation) {
    case "ORIGINAL":
      return "Original";
    case "SELECT":
      return "Object selected";
    case "MASK_EDIT":
      return "Mask refined";
    case "MASK_RESET":
      return "Mask reset";
    case "INPAINT":
      return "Moebius generated";
    case "RESULT_ACCEPTED":
      return "Result accepted";
    case "RESULT_REJECTED":
      return "Result discarded";
    default:
      return operation;
  }
}
