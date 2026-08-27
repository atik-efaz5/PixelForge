"use client";

import type { EditSessionSnapshot } from "@/lib/editSession";
import { operationLabel } from "@/lib/editSession";

interface HistoryPanelProps {
  entries: readonly EditSessionSnapshot[];
  currentIndex: number;
  canUndo: boolean;
  canRedo: boolean;
  busy: boolean;
  hasPendingResult: boolean;
  onSelectEntry: (index: number) => void;
  onSessionUndo: () => void;
  onSessionRedo: () => void;
  onUseResult: () => void;
  onDiscardResult: () => void;
  onRestorePrevious: () => void;
}

export function HistoryPanel({
  entries,
  currentIndex,
  canUndo,
  canRedo,
  busy,
  hasPendingResult,
  onSelectEntry,
  onSessionUndo,
  onSessionRedo,
  onUseResult,
  onDiscardResult,
  onRestorePrevious,
}: HistoryPanelProps) {
  return (
    <section aria-label="Edit history" style={{ padding: "0 20px 16px" }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 8,
          marginBottom: 8,
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
          Session history
        </h2>
        <div style={{ display: "flex", gap: 6 }}>
          <button
            type="button"
            disabled={busy || !canUndo}
            onClick={onSessionUndo}
            style={smallButtonStyle}
          >
            Undo
          </button>
          <button
            type="button"
            disabled={busy || !canRedo}
            onClick={onSessionRedo}
            style={smallButtonStyle}
          >
            Redo
          </button>
        </div>
      </div>

      <ol
        style={{
          listStyle: "none",
          margin: 0,
          padding: 0,
          display: "flex",
          flexDirection: "column",
          gap: 4,
          maxHeight: 140,
          overflowY: "auto",
        }}
      >
        {entries.length === 0 ? (
          <li style={emptyStyle}>Upload an image to start history.</li>
        ) : (
          entries.map((entry, index) => {
            const active = index === currentIndex;
            return (
              <li key={entry.id}>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => onSelectEntry(index)}
                  style={{
                    width: "100%",
                    textAlign: "left",
                    padding: "6px 10px",
                    borderRadius: 6,
                    border: active ? "1px solid #38bdf8" : "1px solid #2a2f3a",
                    background: active ? "#0c4a6e" : "#1a1d24",
                    color: "#e8eaed",
                    fontSize: 12,
                    cursor: busy ? "not-allowed" : "pointer",
                  }}
                >
                  <span style={{ fontWeight: active ? 600 : 400 }}>
                    {entry.label || operationLabel(entry.operation)}
                  </span>
                  <span style={{ color: "#7b8494", marginLeft: 8 }}>
                    {new Date(entry.timestamp).toLocaleTimeString()}
                  </span>
                </button>
              </li>
            );
          })
        )}
      </ol>

      {hasPendingResult ? (
        <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 6 }}>
          <p style={{ margin: 0, fontSize: 12, color: "#9aa3b2" }}>
            New result ready — original image is unchanged.
          </p>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            <button type="button" disabled={busy} onClick={onUseResult} style={actionStyle}>
              Use result
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={onDiscardResult}
              style={actionStyle}
            >
              Discard result
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={onRestorePrevious}
              style={actionStyle}
            >
              Restore previous
            </button>
          </div>
        </div>
      ) : null}
    </section>
  );
}

const smallButtonStyle: React.CSSProperties = {
  padding: "4px 8px",
  borderRadius: 4,
  border: "1px solid #3a4150",
  background: "#1f2430",
  color: "#e8eaed",
  fontSize: 12,
};

const actionStyle: React.CSSProperties = {
  flex: 1,
  minWidth: 100,
  padding: "6px 8px",
  borderRadius: 6,
  border: "1px solid #3a4150",
  background: "#1f2430",
  color: "#e8eaed",
  fontSize: 12,
};

const emptyStyle: React.CSSProperties = {
  fontSize: 12,
  color: "#5c6573",
  padding: "6px 0",
};
