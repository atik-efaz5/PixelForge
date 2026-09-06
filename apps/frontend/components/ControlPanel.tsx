"use client";

import type { DetectionInfo, EditorTool, InpaintBackend, SelectionMode } from "@/types/api";
import { canSubmitFindObject, selectionModeAfterTyping } from "@/lib/selectObject";

interface ControlPanelProps {
  hasImage: boolean;
  onNewSession: () => void;
  selectionMode: SelectionMode;
  onSelectionModeChange: (mode: SelectionMode) => void;
  tool: EditorTool;
  backend: InpaintBackend;
  onBackendChange: (backend: InpaintBackend) => void;
  brushRadius: number;
  eraserRadius: number;
  morphAmount: number;
  featherRadius: number;
  showMaskOverlay: boolean;
  showMaskOnly: boolean;
  hasAiMask: boolean;
  canUndo: boolean;
  canRedo: boolean;
  busy: boolean;
  canGenerate: boolean;
  generateTwoCandidates: boolean;
  onGenerateTwoCandidatesChange: (enabled: boolean) => void;
  textPrompt: string;
  detections: DetectionInfo[];
  detectionIndex: number;
  onUpload: (file: File) => void;
  onToolChange: (tool: EditorTool) => void;
  onBrushRadiusChange: (radius: number) => void;
  onEraserRadiusChange: (radius: number) => void;
  onMorphAmountChange: (amount: number) => void;
  onFeatherRadiusChange: (radius: number) => void;
  onShowMaskOverlayChange: (show: boolean) => void;
  onShowMaskOnlyChange: (show: boolean) => void;
  onExpandMask: () => void;
  onShrinkMask: () => void;
  onUndo: () => void;
  onRedo: () => void;
  onResetToAiMask: () => void;
  onClearMask: () => void;
  onGenerate: () => void;
  onTextPromptChange: (value: string) => void;
  onFindObject: () => void;
  onDetectionIndexChange: (index: number) => void;
  editInstruction: string;
  canApplyInstruction: boolean;
  instructionEditAvailable: boolean;
  canLocalRemove: boolean;
  emphasizeGenerate: boolean;
  onEditInstructionChange: (value: string) => void;
  onApplyInstruction: () => void;
  onLocalRemove: () => void;
}

export function ControlPanel({
  hasImage,
  onNewSession,
  selectionMode,
  onSelectionModeChange,
  tool,
  backend,
  onBackendChange,
  brushRadius,
  eraserRadius,
  morphAmount,
  featherRadius,
  showMaskOverlay,
  showMaskOnly,
  hasAiMask,
  canUndo,
  canRedo,
  busy,
  canGenerate,
  generateTwoCandidates,
  onGenerateTwoCandidatesChange,
  textPrompt,
  detections,
  detectionIndex,
  onUpload,
  onToolChange,
  onBrushRadiusChange,
  onEraserRadiusChange,
  onMorphAmountChange,
  onFeatherRadiusChange,
  onShowMaskOverlayChange,
  onShowMaskOnlyChange,
  onExpandMask,
  onShrinkMask,
  onUndo,
  onRedo,
  onResetToAiMask,
  onClearMask,
  onGenerate,
  onTextPromptChange,
  onFindObject,
  onDetectionIndexChange,
  editInstruction,
  canApplyInstruction,
  instructionEditAvailable,
  canLocalRemove,
  emphasizeGenerate,
  onEditInstructionChange,
  onApplyInstruction,
  onLocalRemove,
}: ControlPanelProps) {
  return (
    <aside
      aria-label="Editor controls"
      className="editor-sidebar"
      style={{
        width: 300,
        flexShrink: 0,
        display: "flex",
        flexDirection: "column",
        gap: 18,
        padding: 20,
        background: "#151820",
        borderLeft: "1px solid #2a2f3a",
        overflowY: "auto",
        maxHeight: "100vh",
      }}
    >
      <section>
        <h2 style={sectionTitleStyle}>Session</h2>
        <button
          type="button"
          disabled={busy}
          onClick={onNewSession}
          style={secondaryButtonStyle}
        >
          New image / New session
        </button>
        {hasImage ? (
          <p style={hintStyle}>Clears the current image, mask, result, and history.</p>
        ) : null}
      </section>

      <section>
        <h2 style={sectionTitleStyle}>Upload Image</h2>
        <label style={labelStyle}>
          <span className="sr-only">Choose image file</span>
          <input
            type="file"
            accept="image/png,image/jpeg,image/webp"
            disabled={busy}
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) onUpload(file);
            }}
            style={{ width: "100%" }}
          />
        </label>
      </section>

      <section>
        <h2 style={sectionTitleStyle}>Selection</h2>
        <label style={labelStyle}>
          Selection mode
          <select
            value={selectionMode}
            disabled={busy}
            onChange={(event) =>
              onSelectionModeChange(event.target.value as SelectionMode)
            }
            style={{ width: "100%", marginTop: 6 }}
          >
            <option value="smart">Smart</option>
            <option value="point">Point</option>
            <option value="text">Text</option>
          </select>
        </label>
        <label style={labelStyle}>
          Select object
          <input
            type="text"
            value={textPrompt}
            disabled={busy}
            placeholder="dog, red car, person..."
            onChange={(event) => {
              const value = event.target.value;
              onTextPromptChange(value);
              const nextMode = selectionModeAfterTyping(selectionMode, value);
              if (nextMode !== selectionMode) {
                onSelectionModeChange(nextMode);
              }
            }}
            onKeyDown={(event) => {
              if (event.key !== "Enter") return;
              event.preventDefault();
              if (!canSubmitFindObject(busy, textPrompt, hasImage)) return;
              const nextMode = selectionModeAfterTyping(selectionMode, textPrompt);
              if (nextMode !== selectionMode) {
                onSelectionModeChange(nextMode);
              }
              onFindObject();
            }}
            style={{ width: "100%", marginTop: 6, padding: "8px 10px" }}
          />
        </label>
        <button
          type="button"
          disabled={!canSubmitFindObject(busy, textPrompt, hasImage)}
          onClick={() => {
            const nextMode = selectionModeAfterTyping(selectionMode, textPrompt);
            if (nextMode !== selectionMode) {
              onSelectionModeChange(nextMode);
            }
            onFindObject();
          }}
          aria-label="Find object by text description"
          style={{ ...secondaryButtonStyle, marginTop: 8 }}
        >
          Find Object
        </button>
        {detections.length > 1 ? (
          <label style={{ ...labelStyle, marginTop: 10 }}>
            Detection
            <select
              value={detectionIndex}
              disabled={busy}
              onChange={(event) => onDetectionIndexChange(Number(event.target.value))}
              style={{ width: "100%", marginTop: 6 }}
            >
              {detections.map((det) => (
                <option key={det.index} value={det.index}>
                  {det.label} ({det.confidence.toFixed(2)})
                </option>
              ))}
            </select>
          </label>
        ) : null}
        <div style={{ ...buttonRowStyle, marginTop: 10 }}>
          <button
            type="button"
            aria-pressed={tool === "select"}
            disabled={busy}
            aria-label="Click to select object on image"
            onClick={() => onToolChange("select")}
            style={toolButtonStyle(tool === "select")}
          >
            Click object
          </button>
        </div>
        <p style={hintStyle}>
          Smart ranks SAM2 / Grounding candidates. Point = click only. Text = prompt only.
        </p>
        <p style={hintStyle}>
          Shortcuts: B brush · E eraser · +/- zoom · 0 fit · Cmd/Ctrl+Z undo
        </p>
      </section>

      <section>
        <h2 style={sectionTitleStyle}>Mask refine</h2>
        <div style={buttonRowStyle}>
          <button
            type="button"
            aria-pressed={tool === "brush"}
            disabled={busy}
            onClick={() => onToolChange("brush")}
            style={toolButtonStyle(tool === "brush")}
          >
            Brush
          </button>
          <button
            type="button"
            aria-pressed={tool === "erase"}
            disabled={busy}
            onClick={() => onToolChange("erase")}
            style={toolButtonStyle(tool === "erase")}
          >
            Eraser
          </button>
        </div>
        <label style={labelStyle}>
          Brush size
          <input
            type="range"
            min={4}
            max={80}
            value={brushRadius}
            disabled={busy}
            onChange={(event) => onBrushRadiusChange(Number(event.target.value))}
            style={{ width: "100%" }}
          />
        </label>
        <label style={labelStyle}>
          Eraser size
          <input
            type="range"
            min={4}
            max={80}
            value={eraserRadius}
            disabled={busy}
            onChange={(event) => onEraserRadiusChange(Number(event.target.value))}
            style={{ width: "100%" }}
          />
        </label>
        <label style={labelStyle}>
          Expand / shrink amount
          <input
            type="range"
            min={1}
            max={8}
            value={morphAmount}
            disabled={busy}
            onChange={(event) => onMorphAmountChange(Number(event.target.value))}
            style={{ width: "100%" }}
          />
        </label>
        <div style={buttonRowStyle}>
          <button
            type="button"
            disabled={busy}
            onClick={onExpandMask}
            style={toolButtonStyle(false)}
          >
            Expand
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={onShrinkMask}
            style={toolButtonStyle(false)}
          >
            Shrink
          </button>
        </div>
        <label style={labelStyle}>
          Feather preview
          <input
            type="range"
            min={0}
            max={24}
            value={featherRadius}
            disabled={busy}
            onChange={(event) => onFeatherRadiusChange(Number(event.target.value))}
            style={{ width: "100%" }}
          />
        </label>
        <p style={hintStyle}>
          Expand / shrink change the mask overlay only — they do not delete the
          object. Feather is a preview. Use Localized Fill to remove the region.
        </p>
        <div style={buttonRowStyle}>
          <button type="button" disabled={busy || !canUndo} onClick={onUndo} style={toolButtonStyle(false)}>
            Undo
          </button>
          <button type="button" disabled={busy || !canRedo} onClick={onRedo} style={toolButtonStyle(false)}>
            Redo
          </button>
        </div>
        <button
          type="button"
          disabled={busy || !hasAiMask}
          onClick={onResetToAiMask}
          style={secondaryButtonStyle}
        >
          Reset to AI mask
        </button>
        <button type="button" disabled={busy} onClick={onClearMask} style={secondaryButtonStyle}>
          Clear mask
        </button>
        <label style={checkboxLabelStyle}>
          <input
            type="checkbox"
            checked={showMaskOverlay}
            disabled={busy}
            onChange={(event) => onShowMaskOverlayChange(event.target.checked)}
          />
          Show mask overlay
        </label>
        <label style={checkboxLabelStyle}>
          <input
            type="checkbox"
            checked={showMaskOnly}
            disabled={busy}
            onChange={(event) => onShowMaskOnlyChange(event.target.checked)}
          />
          Show mask only
        </label>
      </section>

      <section>
        <h2 style={sectionTitleStyle}>Backend</h2>
        <label style={labelStyle}>
          Inpainting model
          <select
            value={backend}
            disabled={busy}
            onChange={(event) =>
              onBackendChange(event.target.value as InpaintBackend)
            }
            style={{ width: "100%", marginTop: 6 }}
          >
            <option value="auto">Automatic (router)</option>
            <option value="moebius">Moebius (LOCAL_MPS)</option>
          </select>
        </label>
        <p style={hintStyle}>
          Automatic uses capability-aware routing. Manual selection overrides for
          debugging.
        </p>
      </section>

      <section>
        <h2 style={sectionTitleStyle}>Instruction Edit</h2>
        <label style={labelStyle}>
          Instruction
          <textarea
            value={editInstruction}
            disabled={busy}
            placeholder="Make the sky look like sunset"
            rows={3}
            onChange={(event) => onEditInstructionChange(event.target.value)}
            style={{
              width: "100%",
              marginTop: 6,
              padding: "8px 10px",
              resize: "vertical",
              fontFamily: "inherit",
            }}
          />
        </label>
        {instructionEditAvailable ? (
          <button
            type="button"
            disabled={busy || !canApplyInstruction}
            onClick={onApplyInstruction}
            style={{ ...secondaryButtonStyle, marginTop: 8, fontWeight: 600 }}
          >
            {busy ? "Applying…" : "Apply Instruction"}
          </button>
        ) : (
          <button
            type="button"
            disabled={busy || !canLocalRemove}
            onClick={onLocalRemove}
            style={{ ...primaryButtonStyle, marginTop: 8, fontWeight: 600 }}
          >
            Remove with local fill (Moebius)
          </button>
        )}
        <p style={hintStyle}>
          {instructionEditAvailable
            ? "Global full-frame edit via InstructPix2Pix (cloud). Does not use the mask above."
            : "Cloud InstructPix2Pix is not configured. Object removal uses the current mask, or selects from this instruction, then Moebius fill."}
        </p>
      </section>

      <section>
        <h2 style={sectionTitleStyle}>Localized Fill</h2>
        <label style={checkboxLabelStyle}>
          <input
            type="checkbox"
            checked={generateTwoCandidates}
            disabled={busy}
            onChange={(event) => onGenerateTwoCandidatesChange(event.target.checked)}
          />
          Generate 2 candidates
        </label>
        <button
          type="button"
          disabled={busy || !canGenerate}
          onClick={onGenerate}
          style={{
            ...primaryButtonStyle,
            boxShadow: emphasizeGenerate ? "0 0 0 2px #38bdf8" : undefined,
          }}
          aria-busy={busy}
          aria-label="Fill selected region with Moebius inpainting"
        >
          {busy ? "Generating…" : "Fill selected region"}
        </button>
        <p style={hintStyle}>
          {emphasizeGenerate
            ? "Next step: fill the masked region with Moebius to remove or replace it."
            : "Sends the current edited bool mask to Moebius. Original image is never modified."}
        </p>
      </section>
    </aside>
  );
}

const sectionTitleStyle: React.CSSProperties = {
  margin: "0 0 10px",
  fontSize: 13,
  fontWeight: 600,
  letterSpacing: "0.04em",
  textTransform: "uppercase",
  color: "#9aa3b2",
};

const labelStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 6,
  fontSize: 14,
  color: "#cbd5e1",
};

const checkboxLabelStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 8,
  fontSize: 13,
  color: "#cbd5e1",
  marginTop: 8,
};

const hintStyle: React.CSSProperties = {
  margin: "8px 0 0",
  fontSize: 12,
  color: "#7b8494",
  lineHeight: 1.4,
};

const buttonRowStyle: React.CSSProperties = {
  display: "flex",
  gap: 8,
  flexWrap: "wrap",
};

function toolButtonStyle(active: boolean): React.CSSProperties {
  return {
    flex: 1,
    padding: "8px 10px",
    borderRadius: 6,
    border: active ? "1px solid #38bdf8" : "1px solid #3a4150",
    background: active ? "#0c4a6e" : "#1f2430",
    color: "#e8eaed",
  };
}

const primaryButtonStyle: React.CSSProperties = {
  width: "100%",
  padding: "12px 16px",
  borderRadius: 8,
  border: "none",
  background: "#2563eb",
  color: "#fff",
  fontWeight: 600,
};

const secondaryButtonStyle: React.CSSProperties = {
  marginTop: 8,
  width: "100%",
  padding: "8px 12px",
  borderRadius: 6,
  border: "1px solid #3a4150",
  background: "#1f2430",
  color: "#e8eaed",
};
