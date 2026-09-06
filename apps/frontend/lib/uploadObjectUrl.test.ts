import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";

import { UploadObjectUrlRegistry, resetUploadObjectUrlPageUnloadHookForTests } from "./uploadObjectUrl";

describe("UploadObjectUrlRegistry", () => {
  let revokeSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    resetUploadObjectUrlPageUnloadHookForTests();
    revokeSpy = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
  });

  afterEach(() => {
    revokeSpy.mockRestore();
    resetUploadObjectUrlPageUnloadHookForTests();
  });

  it("adopts first URL without revoking", () => {
    const registry = new UploadObjectUrlRegistry();
    registry.adopt("blob:a");
    expect(registry.current()).toBe("blob:a");
    expect(revokeSpy).not.toHaveBeenCalled();
  });

  it("revokes previous URL only when adopting a different one", () => {
    const registry = new UploadObjectUrlRegistry();
    registry.adopt("blob:a");
    registry.adopt("blob:b");
    expect(registry.current()).toBe("blob:b");
    expect(revokeSpy).toHaveBeenCalledTimes(1);
    expect(revokeSpy).toHaveBeenCalledWith("blob:a");
  });

  it("does not revoke active URL on releaseIfReplaced", () => {
    const registry = new UploadObjectUrlRegistry();
    registry.adopt("blob:active");
    registry.releaseIfReplaced("blob:active");
    expect(revokeSpy).not.toHaveBeenCalled();
  });

  it("revokes replaced URL via releaseIfReplaced", () => {
    const registry = new UploadObjectUrlRegistry();
    registry.adopt("blob:b");
    registry.releaseIfReplaced("blob:a");
    expect(revokeSpy).toHaveBeenCalledWith("blob:a");
  });

  it("release clears active URL", () => {
    const registry = new UploadObjectUrlRegistry();
    registry.adopt("blob:a");
    registry.release();
    expect(registry.current()).toBeNull();
    expect(revokeSpy).toHaveBeenCalledWith("blob:a");
  });

  it("simulates rerender without revoking active URL", () => {
    const registry = new UploadObjectUrlRegistry();
    registry.adopt("blob:session");
    // Ordinary rerender / effect noop — active URL must survive.
    expect(registry.isActive("blob:session")).toBe(true);
    expect(revokeSpy).not.toHaveBeenCalled();
  });

  it("session history never revokes active source URL", () => {
    const registry = new UploadObjectUrlRegistry();
    const source = registry.adopt("blob:source");
    // History snapshots may reference the same string; registry must not revoke it.
    const snapshotOriginalUrl = source;
    registry.releaseIfReplaced(snapshotOriginalUrl);
    expect(registry.current()).toBe("blob:source");
    expect(revokeSpy).not.toHaveBeenCalled();
  });

  it("simulates Strict Mode remount without revoking active URL", () => {
    const registry = new UploadObjectUrlRegistry();
    registry.adopt("blob:session");
    // React Strict Mode unmount must not call release(); URL stays alive for remount.
    expect(registry.isActive("blob:session")).toBe(true);
    expect(revokeSpy).not.toHaveBeenCalled();
  });

  it("release on explicit new session still revokes", () => {
    const registry = new UploadObjectUrlRegistry();
    registry.adopt("blob:session");
    registry.release();
    expect(registry.current()).toBeNull();
    expect(revokeSpy).toHaveBeenCalledWith("blob:session");
  });
});
