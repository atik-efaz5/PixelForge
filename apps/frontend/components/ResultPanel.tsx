"use client";

interface ResultPanelProps {
  originalUrl: string | null;
  maskPreviewUrl: string | null;
  resultUrl: string | null;
  latencyMs: number | null;
  maskLabel?: string;
  hasAiMask?: boolean;
}

export function ResultPanel({
  originalUrl,
  maskPreviewUrl,
  resultUrl,
  latencyMs,
  maskLabel = "Edited mask",
  hasAiMask = false,
}: ResultPanelProps) {
  return (
    <section aria-label="Results" style={{ padding: "16px 20px 20px" }}>
      <h2
        style={{
          margin: "0 0 12px",
          fontSize: 13,
          fontWeight: 600,
          letterSpacing: "0.04em",
          textTransform: "uppercase",
          color: "#9aa3b2",
        }}
      >
        Preview
      </h2>
      {hasAiMask ? (
        <p style={{ margin: "0 0 8px", fontSize: 12, color: "#7b8494" }}>
          AI mask is stored. Use Reset to AI mask to discard manual edits.
        </p>
      ) : null}
      {latencyMs !== null ? (
        <p style={{ margin: "0 0 12px", fontSize: 12, color: "#7b8494" }}>
          Inpainting latency: {latencyMs.toFixed(1)} ms
        </p>
      ) : null}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
          gap: 12,
        }}
      >
        <PreviewTile title="Original" url={originalUrl} alt="Original uploaded image" />
        <PreviewTile title={maskLabel} url={maskPreviewUrl} alt="Current edited mask preview" />
        <PreviewTile title="Result" url={resultUrl} alt="Generated inpainting result" />
      </div>
    </section>
  );
}

function PreviewTile({
  title,
  url,
  alt,
}: {
  title: string;
  url: string | null;
  alt: string;
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
          aspectRatio: "1",
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
