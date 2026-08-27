"use client";

import { useState } from "react";
import { downloadImage } from "@/lib/exportImage";
import { GenerationResultMeta } from "@/components/GenerationStatus";

export type ResultViewMode =
  | "original"
  | "mask"
  | "result"
  | "side_by_side"
  | "compare";

interface ResultPanelProps {
  originalUrl: string | null;
  maskPreviewUrl: string | null;
  resultUrl: string | null;
  latencyMs: number | null;
  maskLabel?: string;
  hasAiMask?: boolean;
  resultModel?: string;
  resultBackend?: string;
  requestedBackend?: string;
}

export function ResultPanel({
  originalUrl,
  maskPreviewUrl,
  resultUrl,
  latencyMs,
  maskLabel = "Edited mask",
  hasAiMask = false,
  resultModel,
  resultBackend,
  requestedBackend,
}: ResultPanelProps) {
  const [viewMode, setViewMode] = useState<ResultViewMode>("side_by_side");
  const [comparePos, setComparePos] = useState(50);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);

  const handleExport = async (format: "png" | "jpeg") => {
    if (!resultUrl) return;
    setExporting(true);
    setExportError(null);
    try {
      await downloadImage(resultUrl, "pixelforge-result", format);
    } catch (err) {
      setExportError(
        err instanceof Error ? err.message : "Export failed."
      );
    } finally {
      setExporting(false);
    }
  };

  return (
    <section aria-label="Results" className="result-panel" style={{ padding: "16px 0 0" }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 8,
          flexWrap: "wrap",
          marginBottom: 12,
        }}
      >
        <h2
          style={{
            margin: 0,
            fontSize: 13,
            fontWeight: 600,
            letterSpacing: "0.04em",
            textTransform: "uppercase",
            color: "#9aa3b2",
          }}
        >
          Preview
        </h2>
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
          {VIEW_MODES.map((mode) => (
            <button
              key={mode.id}
              type="button"
              disabled={!canShowMode(mode.id, { originalUrl, maskPreviewUrl, resultUrl })}
              onClick={() => setViewMode(mode.id)}
              style={tabStyle(viewMode === mode.id)}
            >
              {mode.label}
            </button>
          ))}
        </div>
      </div>

      {hasAiMask ? (
        <p style={{ margin: "0 0 8px", fontSize: 12, color: "#7b8494" }}>
          AI mask is stored. Use Reset to AI mask to discard manual edits.
        </p>
      ) : null}

      <GenerationResultMeta
        model={resultModel}
        backend={resultBackend}
        latencyMs={latencyMs}
        requestedBackend={requestedBackend}
      />

      <div style={{ marginBottom: 12 }}>{renderView()}</div>

      {resultUrl ? (
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
          <button
            type="button"
            disabled={exporting}
            onClick={() => handleExport("png")}
            style={exportButtonStyle}
          >
            Download PNG
          </button>
          <button
            type="button"
            disabled={exporting}
            onClick={() => handleExport("jpeg")}
            style={exportButtonStyle}
          >
            Download JPEG
          </button>
          {exportError ? (
            <span style={{ fontSize: 12, color: "#fca5a5" }}>{exportError}</span>
          ) : null}
        </div>
      ) : null}
    </section>
  );

  function renderView() {
    if (viewMode === "compare" && originalUrl && resultUrl) {
      return (
        <div
          style={{
            position: "relative",
            aspectRatio: "16 / 10",
            maxHeight: 360,
            background: "#1a1d24",
            border: "1px solid #2a2f3a",
            borderRadius: 8,
            overflow: "hidden",
            userSelect: "none",
          }}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={originalUrl}
            alt="Original"
            style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "contain" }}
          />
          <div
            style={{
              position: "absolute",
              inset: 0,
              clipPath: `inset(0 ${100 - comparePos}% 0 0)`,
            }}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={resultUrl}
              alt="Result"
              style={{ width: "100%", height: "100%", objectFit: "contain" }}
            />
          </div>
          <input
            type="range"
            min={0}
            max={100}
            value={comparePos}
            onChange={(e) => setComparePos(Number(e.target.value))}
            aria-label="Before and after comparison"
            style={{
              position: "absolute",
              left: "5%",
              right: "5%",
              bottom: 12,
              width: "90%",
            }}
          />
          <div
            style={{
              position: "absolute",
              top: 0,
              bottom: 0,
              left: `${comparePos}%`,
              width: 2,
              background: "#38bdf8",
              pointerEvents: "none",
            }}
          />
        </div>
      );
    }

    if (viewMode === "side_by_side") {
      return (
        <div className="result-grid-side">
          <PreviewTile title="Original" url={originalUrl} alt="Original uploaded image" />
          <PreviewTile title="Result" url={resultUrl} alt="Generated inpainting result" />
        </div>
      );
    }

    if (viewMode === "original") {
      return <PreviewTile title="Original" url={originalUrl} alt="Original uploaded image" large />;
    }
    if (viewMode === "mask") {
      return (
        <PreviewTile title={maskLabel} url={maskPreviewUrl} alt="Current edited mask preview" large />
      );
    }
    return <PreviewTile title="Result" url={resultUrl} alt="Generated inpainting result" large />;
  }
}

const VIEW_MODES: { id: ResultViewMode; label: string }[] = [
  { id: "side_by_side", label: "Side by side" },
  { id: "compare", label: "Compare" },
  { id: "original", label: "Original" },
  { id: "mask", label: "Mask" },
  { id: "result", label: "Result" },
];

function canShowMode(
  mode: ResultViewMode,
  urls: { originalUrl: string | null; maskPreviewUrl: string | null; resultUrl: string | null }
): boolean {
  if (mode === "original") return Boolean(urls.originalUrl);
  if (mode === "mask") return Boolean(urls.maskPreviewUrl);
  if (mode === "result") return Boolean(urls.resultUrl);
  if (mode === "compare") return Boolean(urls.originalUrl && urls.resultUrl);
  if (mode === "side_by_side") return Boolean(urls.originalUrl);
  return true;
}

function tabStyle(active: boolean): React.CSSProperties {
  return {
    padding: "5px 10px",
    borderRadius: 6,
    border: active ? "1px solid #38bdf8" : "1px solid #3a4150",
    background: active ? "#0c4a6e" : "#1f2430",
    color: "#e8eaed",
    fontSize: 12,
  };
}

const exportButtonStyle: React.CSSProperties = {
  padding: "8px 12px",
  borderRadius: 6,
  border: "1px solid #3a4150",
  background: "#1f2430",
  color: "#e8eaed",
  fontSize: 13,
};

function PreviewTile({
  title,
  url,
  alt,
  large = false,
}: {
  title: string;
  url: string | null;
  alt: string;
  large?: boolean;
}) {
  return (
    <figure style={{ margin: 0 }}>
      <figcaption
        style={{
          marginBottom: 6,
          fontSize: 12,
          color: "#9aa3b2",
          textAlign: "center",
        }}
      >
        {title}
      </figcaption>
      <div
        style={{
          aspectRatio: large ? "16 / 10" : "1",
          maxHeight: large ? 360 : undefined,
          background: "#1a1d24",
          border: "1px solid #2a2f3a",
          borderRadius: 6,
          overflow: "hidden",
          display: "grid",
          placeItems: "center",
        }}
      >
        {url ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={url}
            alt={alt}
            style={{
              width: "100%",
              height: "100%",
              objectFit: "contain",
            }}
          />
        ) : (
          <span style={{ fontSize: 11, color: "#5c6573" }}>—</span>
        )}
      </div>
    </figure>
  );
}
