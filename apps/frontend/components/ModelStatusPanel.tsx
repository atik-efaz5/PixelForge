"use client";

import type { ModelLine } from "@/lib/modelLabels";

interface ModelStatusPanelProps {
  selection: ModelLine | null;
  inpainting: ModelLine | null;
}

export function ModelStatusPanel({ selection, inpainting }: ModelStatusPanelProps) {
  if (!selection && !inpainting) {
    return null;
  }

  return (
    <section
      aria-label="Model status"
      style={{
        marginBottom: 12,
        padding: "10px 12px",
        borderRadius: 8,
        background: "#151820",
        border: "1px solid #2a2f3a",
        display: "flex",
        flexWrap: "wrap",
        gap: "12px 24px",
        fontSize: 13,
      }}
    >
      {selection ? <StatusLine line={selection} /> : null}
      {inpainting ? <StatusLine line={inpainting} /> : null}
    </section>
  );
}

function StatusLine({ line }: { line: ModelLine }) {
  const detail = line.route ?? `${line.model} · ${line.backend}`;
  return (
    <div style={{ minWidth: 0 }}>
      <span style={{ color: "#7b8494", marginRight: 6 }}>{line.label}:</span>
      <span style={{ color: "#e8eaed", fontWeight: 500 }}>{detail}</span>
      {line.confidenceTier ? (
        <span
          style={{
            marginLeft: 8,
            fontSize: 11,
            fontWeight: 600,
            color: tierColor(line.confidenceTier),
            textTransform: "capitalize",
          }}
        >
          {line.confidenceTier.toLowerCase()} confidence
        </span>
      ) : null}
    </div>
  );
}

function tierColor(tier: string): string {
  if (tier === "HIGH") return "#4ade80";
  if (tier === "MEDIUM") return "#fbbf24";
  return "#f87171";
}
