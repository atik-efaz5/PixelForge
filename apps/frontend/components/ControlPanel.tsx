"use client";

import type { DetectionInfo, EditorTool, InpaintBackend } from "@/types/api";

interface ControlPanelProps {
  tool: EditorTool;
  backend: InpaintBackend;
  brushRadius: number;
  busy: boolean;
  canGenerate: boolean;
  textPrompt: string;
  detections: DetectionInfo[];
  detectionIndex: number;
  onUpload: (file: File) => void;
  onToolChange: (tool: EditorTool) => void;
  onBrushRadiusChange: (radius: number) => void;
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
  brushRadius,
  busy,
  canGenerate,
  textPrompt,
  detections,
  detectionIndex,
  onUpload,
  onToolChange,
  onBrushRadiusChange,
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
        width: 280,
        flexShrink: 0,
        display: "flex",
        flexDirection: "column",
        gap: 20,
        padding: 20,
        background: "#151820",
        borderLeft: "1px solid #2a2f3a",
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
            onClick={() => onToolChange("select")}
            style={toolButtonStyle(tool === "select")}
          >
            Click object
          </button>
        </div>
        <p style={hintStyle}>
          Enter text to find an object, or click directly on the image.
        </p>
      </section>

      <section>
        <h2 style={sectionTitleStyle}>Mask</h2>
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
            Erase
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
        <button
          type="button"
          disabled={busy}
          onClick={onClearMask}
          style={secondaryButtonStyle}
        >
          Clear mask
        </button>
      </section>

      <section>
        <h2 style={sectionTitleStyle}>Backend</h2>
        <label style={labelStyle}>
          Inpainting model
          <select value={backend} disabled style={{ width: "100%", marginTop: 6 }}>
            <option value="moebius">Moebius (LOCAL_MPS)</option>
          </select>
        </label>
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
          Global full-frame edit via InstructPix2Pix (cloud). Does not use the mask
          from selection above.
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
        >
          {busy ? "Generating…" : "Fill selected region"}
        </button>
        <p style={hintStyle}>
          Mask-conditioned inpainting only. Text instructions for object replacement are not
          supported by the current backend.
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
