"use client";

import type { UserFacingError } from "@/lib/formatError";

interface ErrorBannerProps {
  error: UserFacingError;
  onDismiss: () => void;
  onRetry?: () => void;
}

export function ErrorBanner({ error, onDismiss, onRetry }: ErrorBannerProps) {
  return (
    <div
      role="alert"
      style={{
        marginBottom: 12,
        padding: "12px 14px",
        borderRadius: 8,
        background: "#450a0a",
        border: "1px solid #b91c1c",
        color: "#fecaca",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
        <div>
          <strong style={{ display: "block", marginBottom: 4, fontSize: 14 }}>
            {error.message}
          </strong>
          {error.recovery ? (
            <p style={{ margin: 0, fontSize: 13, color: "#fca5a5", lineHeight: 1.4 }}>
              {error.recovery}
            </p>
          ) : null}
        </div>
        <div style={{ display: "flex", gap: 6, flexShrink: 0 }}>
          {onRetry ? (
            <button type="button" onClick={onRetry} style={actionStyle}>
              Retry
            </button>
          ) : null}
          <button type="button" onClick={onDismiss} style={actionStyle}>
            Dismiss
          </button>
        </div>
      </div>
    </div>
  );
}

const actionStyle: React.CSSProperties = {
  padding: "6px 10px",
  borderRadius: 6,
  border: "1px solid #7f1d1d",
  background: "#7f1d1d",
  color: "#fff",
  fontSize: 12,
  whiteSpace: "nowrap",
};
