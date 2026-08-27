import type {
  ApiErrorBody,
  EditByInstructionMetadata,
  HealthResponse,
  InpaintBackend,
  InpaintMetadata,
  ModelsResponse,
  PngWithMetadata,
  RemoveObjectMetadata,
  SelectByTextMetadata,
  SegmentMetadata,
} from "@/types/api";

const DEFAULT_BASE_URL = "http://127.0.0.1:8000";

export function apiBaseUrl(): string {
  return process.env.NEXT_PUBLIC_API_BASE_URL?.trim() || DEFAULT_BASE_URL;
}

function pfHeaderKey(field: string): string {
  return `x-pf-${field.replace(/_/g, "-")}`;
}

function parseHeaderMetadata<T extends object>(
  headers: Headers,
  fields: (keyof T)[]
): T {
  const meta = {} as T;
  for (const field of fields) {
    const raw = headers.get(pfHeaderKey(String(field)));
    if (raw === null || raw === "") continue;
    if (field === "metadata" || field === "detections" || field === "selected_box_xyxy") {
      try {
        (meta as Record<string, unknown>)[field as string] = JSON.parse(raw);
      } catch {
        (meta as Record<string, unknown>)[field as string] =
          field === "detections" ? [] : field === "selected_box_xyxy" ? [] : {};
      }
      continue;
    }
    if (
      field === "confidence" ||
      field === "latency_ms" ||
      field === "memory_mb" ||
      field === "detection_index" ||
      field === "detection_count"
    ) {
      const num = Number(raw);
      (meta as Record<string, unknown>)[field as string] = Number.isFinite(num)
        ? num
        : null;
      continue;
    }
    (meta as Record<string, unknown>)[field as string] = raw;
  }
  return meta;
}

async function parseErrorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as ApiErrorBody;
    if (
      body.error &&
      typeof body.error === "object" &&
      "message" in body.error &&
      body.error.message
    ) {
      return body.error.message;
    }
    if (body.message) return body.message;
    if (typeof body.error === "string") return body.error;
  } catch {
    // Response body is not JSON.
  }
  return `Request failed (${response.status})`;
}

async function readPngResponse<T extends object>(
  response: Response,
  fields: (keyof T)[]
): Promise<PngWithMetadata<T>> {
  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }
  const blob = await response.blob();
  if (!blob.type.includes("png") && blob.size === 0) {
    throw new Error("Server returned an empty response.");
  }
  return {
    blob,
    metadata: parseHeaderMetadata<T>(response.headers, fields),
  };
}

export async function health(): Promise<HealthResponse> {
  const response = await fetch(`${apiBaseUrl()}/health`);
  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }
  return (await response.json()) as HealthResponse;
}

export async function models(): Promise<ModelsResponse> {
  const response = await fetch(`${apiBaseUrl()}/models`);
  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }
  return (await response.json()) as ModelsResponse;
}

export async function segment(
  image: File,
  x: number,
  y: number
): Promise<PngWithMetadata<SegmentMetadata>> {
  const form = new FormData();
  form.append("image", image, image.name || "image.png");
  form.append("x", String(x));
  form.append("y", String(y));

  const response = await fetch(`${apiBaseUrl()}/segment`, {
    method: "POST",
    body: form,
  });

  return readPngResponse<SegmentMetadata>(response, [
    "confidence",
    "model",
    "backend",
    "method",
    "metadata",
  ]);
}

export async function inpaint(
  image: File,
  mask: Blob,
  backend: InpaintBackend = "moebius"
): Promise<PngWithMetadata<InpaintMetadata>> {
  const form = new FormData();
  form.append("image", image, image.name || "image.png");
  form.append("mask", mask, "mask.png");
  form.append("backend", backend);

  const response = await fetch(`${apiBaseUrl()}/inpaint`, {
    method: "POST",
    body: form,
  });

  return readPngResponse<InpaintMetadata>(response, [
    "model",
    "backend",
    "latency_ms",
    "memory_mb",
    "metadata",
  ]);
}

export async function selectByText(
  image: File,
  prompt: string,
  detectionIndex = 0
): Promise<PngWithMetadata<SelectByTextMetadata>> {
  const form = new FormData();
  form.append("image", image, image.name || "image.png");
  form.append("prompt", prompt);
  form.append("detection_index", String(detectionIndex));
  form.append("grounding_backend", "grounding_dino");

  const response = await fetch(`${apiBaseUrl()}/select-by-text`, {
    method: "POST",
    body: form,
  });

  return readPngResponse<SelectByTextMetadata>(response, [
    "prompt",
    "model",
    "segmentation_model",
    "grounding_backend",
    "confidence",
    "method",
    "detection_index",
    "detection_count",
    "selected_label",
    "selected_box_xyxy",
    "detections",
    "metadata",
  ]);
}

export async function removeObject(
  image: File,
  x: number,
  y: number,
  backend: InpaintBackend = "moebius",
  mask?: Blob
): Promise<PngWithMetadata<RemoveObjectMetadata>> {
  const form = new FormData();
  form.append("image", image, image.name || "image.png");
  form.append("x", String(x));
  form.append("y", String(y));
  form.append("backend", backend);
  if (mask) {
    // Backend remove-object uses point segmentation; mask refinement is server-side.
    // Exposed for API parity; MVP UI uses segment + inpaint instead.
    void mask;
  }

  const response = await fetch(`${apiBaseUrl()}/remove-object`, {
    method: "POST",
    body: form,
  });

  return readPngResponse<RemoveObjectMetadata>(response, [
    "model",
    "backend",
    "segmentation_model",
    "latency_ms",
    "segmentation_ms",
    "inpainting_ms",
    "metadata",
  ]);
}

export async function editByInstruction(
  image: File,
  instruction: string,
  backend = "instruct_pix2pix"
): Promise<PngWithMetadata<EditByInstructionMetadata>> {
  const form = new FormData();
  form.append("image", image, image.name || "image.png");
  form.append("instruction", instruction);
  form.append("backend", backend);

  const response = await fetch(`${apiBaseUrl()}/edit-by-instruction`, {
    method: "POST",
    body: form,
  });

  return readPngResponse<EditByInstructionMetadata>(response, [
    "model",
    "backend",
    "instruction",
    "latency_ms",
    "memory_mb",
    "metadata",
  ]);
}
