export interface UserFacingError {
  message: string;
  recovery?: string;
}

/** Strip stack traces and return a clean user-facing error. */
export function toUserFacingError(raw: unknown): UserFacingError {
  let message =
    raw instanceof Error ? raw.message : "Something went wrong. Please try again.";

  if (message.includes("Traceback") || message.includes("File \"")) {
    return {
      message: "The server returned an unexpected error.",
      recovery: "Try again, or start a new session with a different image.",
    };
  }

  message = message.split("\n")[0]?.trim() || message;
  message = message.replace(/^Error:\s*/i, "");

  const lower = message.toLowerCase();
  if (lower.includes("not available") || lower.includes("unavailable")) {
    return {
      message,
      recovery: "Check that local models are installed, or choose a different backend.",
    };
  }
  if (lower.includes("failed") || lower.includes("timeout")) {
    return {
      message,
      recovery: "Try again. If the problem persists, refine the mask or use a smaller image.",
    };
  }
  if (lower.includes("empty") || lower.includes("mask")) {
    return {
      message,
      recovery: "Select or draw a mask region before generating.",
    };
  }
  if (lower.includes("edit session not initialized") || lower.includes("edit session could not")) {
    return {
      message,
      recovery: "Upload the image again or click New session to reset the editor.",
    };
  }
  return { message, recovery: "You can dismiss this and continue editing." };
}
