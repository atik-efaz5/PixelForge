/**
 * Tracks the single active uploaded-image object URL for an editor session.
 * Only this registry may revoke the source upload URL.
 */

export class UploadObjectUrlRegistry {
  private active: string | null = null;

  /** Replace the active upload URL, revoking the previous one when different. */
  adopt(next: string): string {
    if (this.active && this.active !== next) {
      URL.revokeObjectURL(this.active);
    }
    this.active = next;
    return next;
  }

  /** Release the active upload URL (new session or unmount). */
  release(): void {
    if (!this.active) return;
    URL.revokeObjectURL(this.active);
    this.active = null;
  }

  current(): string | null {
    return this.active;
  }

  isActive(url: string): boolean {
    return this.active === url;
  }

  /** Revoke only when the URL is not the active source (e.g. replaced upload). */
  releaseIfReplaced(url: string | null | undefined): void {
    if (!url || url === this.active) return;
    URL.revokeObjectURL(url);
  }
}

export function isObjectUrl(url: string | null | undefined): url is string {
  return typeof url === "string" && url.startsWith("blob:");
}
