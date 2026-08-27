"use client";

export interface CandidateOption {
  id: string;
  url: string;
  label: string;
  score?: number;
  selected: boolean;
}

interface CandidatePickerProps {
  options: CandidateOption[];
  disabled?: boolean;
  onSelect: (id: string) => void;
}

export function CandidatePicker({
  options,
  disabled = false,
  onSelect,
}: CandidatePickerProps) {
  if (options.length <= 1) return null;

  return (
    <section aria-label="Candidate selection" style={{ padding: "0 20px 12px" }}>
      <h2
        style={{
          margin: "0 0 8px",
          fontSize: 13,
          fontWeight: 600,
          letterSpacing: "0.04em",
          textTransform: "uppercase",
          color: "#9aa3b2",
        }}
      >
        Choose result
      </h2>
      <p style={{ margin: "0 0 10px", fontSize: 12, color: "#7b8494" }}>
        Two candidates were generated. Pick the one to preview before accepting.
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
        {options.map((option) => (
          <button
            key={option.id}
            type="button"
            disabled={disabled}
            onClick={() => onSelect(option.id)}
            style={{
              padding: 8,
              borderRadius: 8,
              border: option.selected ? "2px solid #38bdf8" : "1px solid #2a2f3a",
              background: option.selected ? "#0c4a6e" : "#1a1d24",
              color: "#e8eaed",
              cursor: disabled ? "not-allowed" : "pointer",
              textAlign: "left",
            }}
          >
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>
              {option.label}
              {option.score !== undefined ? (
                <span style={{ color: "#9aa3b2", fontWeight: 400, marginLeft: 6 }}>
                  score {option.score.toFixed(3)}
                </span>
              ) : null}
            </div>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={option.url}
              alt={option.label}
              style={{
                width: "100%",
                aspectRatio: "1",
                objectFit: "contain",
                background: "#111318",
                borderRadius: 4,
              }}
            />
          </button>
        ))}
      </div>
    </section>
  );
}
