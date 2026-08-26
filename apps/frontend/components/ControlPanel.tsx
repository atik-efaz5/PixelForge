"use client";

import type { EditorTool, InpaintBackend } from "@/types/api";

interface ControlPanelProps {
  tool: EditorTool;
  backend: InpaintBackend;
  brushRadius: number;
  busy: boolean;
  canGenerate: boolean;
  onUpload: (file: File) => void;
  onToolChange: (tool: EditorTool) => void;
  onBrushRadiusChange: (radius: number) => void;
  onClearMask: () => void;
  onGenerate: () => void;
}

export function ControlPanel({
  tool,
  backend,
  brushRadius,
  busy,
  canGenerate,
  onUpload,
  onToolChange,
  onBrushRadiusChange,
  onClearMask,
  onGenerate,
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
        <div style={buttonRowStyle}>
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
        <p style={hintStyle}>Click on the object you want to remove or edit.</p>
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
        <h2 style={sectionTitleStyle}>Actions</h2>
        <button
          type="button"
          disabled={busy || !canGenerate}
          onClick={onGenerate}
          style={primaryButtonStyle}
          aria-busy={busy}
        >
          {busy ? "Generating…" : "Generate"}
        </button>
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
