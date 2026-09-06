import { EMPTY_MASK_REFINE_MESSAGE } from "./maskRefine";

export interface UserFacingError {
  message: string;
  recovery?: string;
}

/** Strip stack traces and return a clean user-facing error. */
export function toUserFacingError(raw: unknown): UserFacingError {
  let message =
    raw instanceof Error
      ? raw.message
      : typeof raw === "string"
        ? raw
        : "Something went wrong. Please try again.";

  if (message.includes("Traceback") || message.includes("File \"")) {
    return {
      message: "The server returned an unexpected error.",
      recovery: "Try again, or start a new session with a different image.",
    };
  }

  message = message.split("\n")[0]?.trim() || message;
  message = message.replace(/^Error:\s*/i, "");

  const lower = message.toLowerCase();
  if (
    message === EMPTY_MASK_REFINE_MESSAGE ||
    lower.includes("select or brush")
  ) {
    return {
      message,
      recovery: "Find an object or paint a mask, then use Localized Fill to remove it.",
    };
  }
  if (
    lower.includes("grounding") ||
    lower.includes("grounding dino") ||
    (lower.includes("failed to load") && lower.includes("dino"))
  ) {
    return {
      message,
      recovery:
        "Click the object on the image (SAM 2), or retry Find Object.",
    };
  }
  if (
    lower.includes("moebius") ||
    lower.includes("model_load_failed") ||
    (lower.includes("inpaint") &&
      (lower.includes("worker") || lower.includes("load") || lower.includes("503")))
  ) {
    return {
      message,
      recovery:
        "The mask is still there. Retry Fill selected region — Moebius takes about 20 seconds.",
    };
  }
  if (
    lower.includes("instruct_pix2pix") ||
    lower.includes("global_instruction_edit")
  ) {
    return {
      message,
      recovery:
        "Instruction edit needs a cloud InstructPix2Pix endpoint. To remove an object locally, select it and use Localized Fill (Moebius).",
    };
  }
  if (lower.includes("not available") || lower.includes("unavailable")) {
    return {
      message,
      recovery: "Check that local models are installed, or choose a different backend.",
    };
  }
  if (
    lower.includes("failed to fetch") ||
    lower.includes("networkerror") ||
    lower.includes("load failed") ||
    lower.includes("err_connection") ||
    lower.includes("mixed content")
  ) {
    return {
      message: "Cannot reach the PixelForge API from this page.",
      recovery:
        "The hosted UI needs the public API URL (Cloudflare tunnel), not localhost. Keep the Mac backend and tunnel running, then retry.",
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
