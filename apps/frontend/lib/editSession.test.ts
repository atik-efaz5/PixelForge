import { describe, expect, it } from "vitest";

import { EditSessionHistory } from "./editSession";

describe("EditSessionHistory lifecycle", () => {
  it("creates ORIGINAL entry on reset", () => {
    const session = new EditSessionHistory();
    const entry = session.reset("blob:original-a");

    expect(entry.operation).toBe("ORIGINAL");
    expect(entry.originalUrl).toBe("blob:original-a");
    expect(session.snapshotEntries).toHaveLength(1);
    expect(session.currentIndex).toBe(0);
    expect(session.current()?.operation).toBe("ORIGINAL");
  });

  it("rejects append before reset", () => {
    const session = new EditSessionHistory();

    expect(() =>
      session.append({
        operation: "SELECT",
        label: "Object selected",
      })
    ).toThrow(/Edit session not initialized/i);
  });

  it("replaces session when reset with a new image URL", () => {
    const session = new EditSessionHistory();
    session.reset("blob:image-a");
    session.append({
      operation: "SELECT",
      label: "First selection",
      mask: new Uint8Array([1]),
    });

    const replaced = session.reset("blob:image-b");

    expect(replaced.operation).toBe("ORIGINAL");
    expect(replaced.originalUrl).toBe("blob:image-b");
    expect(session.snapshotEntries).toHaveLength(1);
    expect(session.current()?.originalUrl).toBe("blob:image-b");
  });

  it("simulates upload lifecycle without disposing session on URL change", () => {
    const session = new EditSessionHistory();

    // First upload
    session.reset("blob:first");
    expect(session.currentOriginalUrl).toBe("blob:first");

    // Replacing image disposes owned results but must re-init via reset, not dispose()
    session.reset("blob:second");
    expect(session.currentOriginalUrl).toBe("blob:second");
    expect(session.snapshotEntries).toHaveLength(1);

    // append must work immediately after replacement
    expect(() =>
      session.append({
        operation: "SELECT",
        label: "Selected",
        mask: new Uint8Array([255]),
      })
    ).not.toThrow();
  });

  it("dispose clears session until reset", () => {
    const session = new EditSessionHistory();
    session.reset("blob:upload");

    session.dispose();

    expect(session.currentOriginalUrl).toBeNull();
    expect(session.current()).toBeNull();
    expect(() =>
      session.append({
        operation: "MASK_EDIT",
        label: "Should fail",
      })
    ).toThrow(/Edit session not initialized/i);

    const fresh = session.reset("blob:after-reload");
    expect(fresh.operation).toBe("ORIGINAL");
  });
});
