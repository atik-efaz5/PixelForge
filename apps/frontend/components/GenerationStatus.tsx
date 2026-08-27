"use client";

import type { EditorStatus, InpaintBackend } from "@/types/api";
import { formatBackendName, formatModelName } from "@/lib/modelLabels";

interface GenerationStatusProps {
  status: EditorStatus;
  backend?: InpaintBackend | string;
  model?: string;
  elapsedMs: number;
}

const STATUS_LABELS: Partial<Record<EditorStatus, string>> = {
  uploading: "Uploading image…",
  segmenting: "Segmenting selection…",
  grounding: "Finding object…",
  generating: "Generating…",
  instruction_editing: "Applying instruction…",
};

export function GenerationStatus({
  status,
  backend,
  model,
  elapsedMs,
}: GenerationStatusProps) {
  const label = STATUS_LABELS[status];
  if (!label) return null;

  const backendLabel =
    backend === "auto" || !backend
      ? "Automatic"
      : formatModelName(String(backend));
  const modelLabel = model ? formatModelName(model) : null;
  const elapsed = elapsedMs > 0 ? `${(elapsedMs / 1000).toFixed(1)}s` : null;

  return (
    <div
      aria-live="polite"
      aria-busy="true"
      style={{
        marginBottom: 12,
        padding: "10px 14px",
        borderRadius: 8,
        background: "#172554",
        border: "1px solid #1e40af",
        color: "#bfdbfe",
        fontSize: 13,
        display: "flex",
        flexWrap: "wrap",
        alignItems: "center",
        gap: "8px 16px",
      }}
    >
      <span style={{ fontWeight: 600 }}>{label}</span>
      {(status === "generating" || status === "instruction_editing") && (
        <>
          <span style={{ color: "#93c5fd" }}>Backend: {backendLabel}</span>
          {modelLabel ? (
            <span style={{ color: "#93c5fd" }}>Model: {modelLabel}</span>
          ) : null}
        </>
      )}
      {status === "segmenting" ? (
        <span style={{ color: "#93c5fd" }}>Model: SAM 2</span>
      ) : null}
      {status === "grounding" ? (
        <span style={{ color: "#93c5fd" }}>Model: Grounding DINO</span>
      ) : null}
      {elapsed ? <span style={{ color: "#7dd3fc" }}>Elapsed: {elapsed}</span> : null}
    </div>
  );
}

interface GenerationResultMetaProps {
  model?: string;
  backend?: string;
  latencyMs?: number | null;
  requestedBackend?: string;
}

export function GenerationResultMeta({
  model,
  backend,
  latencyMs,
  requestedBackend,
}: GenerationResultMetaProps) {
  if (!model && latencyMs == null) return null;

  const isAuto =
    requestedBackend === "auto" ||
    !requestedBackend ||
    requestedBackend === "automatic";
  const modelLabel = formatModelName(model);
  const backendLabel = formatBackendName(backend);

  return (
    <p style={{ margin: "0 0 12px", fontSize: 12, color: "#9aa3b2", lineHeight: 1.5 }}>
      {isAuto ? (
        <>
          Generated with <strong style={{ color: "#cbd5e1" }}>Automatic → {modelLabel}</strong>
        </>
      ) : (
        <>
          Generated with <strong style={{ color: "#cbd5e1" }}>{modelLabel}</strong>
        </>
      )}
      {backendLabel ? <> · {backendLabel}</> : null}
      {latencyMs != null ? <> · {(latencyMs / 1000).toFixed(1)}s</> : null}
    </p>
  );
}
