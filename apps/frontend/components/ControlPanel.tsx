"use client";

import type { DetectionInfo, EditorTool, InpaintBackend } from "@/types/api";

interface ControlPanelProps {
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
  onEditInstructionChange: (value: string) => void;
  onApplyInstruction: () => void;
}

export function ControlPanel({
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
  onEditInstructionChange,
  onApplyInstruction,
}: ControlPanelProps) {
  return (
    <aside
      aria-label="Editor controls"
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
          Select object
          <input
            type="text"
            value={textPrompt}
            disabled={busy}
            placeholder="dog, red car, person..."
            onChange={(event) => onTextPromptChange(event.target.value)}
            style={{ width: "100%", marginTop: 6, padding: "8px 10px" }}
          />
        </label>
        <button
          type="button"
          disabled={busy || !textPrompt.trim()}
          onClick={onFindObject}
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
          Text finds an object; click segments on the image. AI mask is stored separately from edits.
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
          Feather affects overlay and preview only. Inpainting still uses a hard bool mask.
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
        <button
          type="button"
          disabled={busy || !canApplyInstruction}
          onClick={onApplyInstruction}
          style={{ ...secondaryButtonStyle, marginTop: 8, fontWeight: 600 }}
        >
          {busy ? "Applying…" : "Apply Instruction"}
        </button>
        <p style={hintStyle}>
          Global full-frame edit via InstructPix2Pix (cloud). Does not use the mask above.
        </p>
      </section>

      <section>
        <h2 style={sectionTitleStyle}>Localized Fill</h2>
        <button
          type="button"
          disabled={busy || !canGenerate}
          onClick={onGenerate}
          style={primaryButtonStyle}
          aria-busy={busy}
          aria-label="Fill selected region with Moebius inpainting"
        >
          {busy ? "Generating…" : "Fill selected region"}
        </button>
        <p style={hintStyle}>
          Sends the current edited bool mask to Moebius. Original image is never modified.
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
