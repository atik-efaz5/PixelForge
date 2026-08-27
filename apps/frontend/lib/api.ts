import type {
  ApiErrorBody,
  EditByInstructionMetadata,
  HealthResponse,
  InpaintBackend,
  InpaintCandidateResult,
  InpaintCandidatesMetadata,
  InpaintCandidatesResponse,
  InpaintMetadata,
  ModelsResponse,
  RoutingResponse,
  PngWithMetadata,
  RemoveObjectMetadata,
  SelectByTextMetadata,
  SelectSmartMetadata,
  SegmentMetadata,
  SelectionMode,
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

function parseMultipartBoundary(contentType: string): string | null {
  const match = /boundary=([^;]+)/i.exec(contentType);
  if (!match) return null;
  return match[1].trim().replace(/^"|"$/g, "");
}

async function readMultipartInpaintCandidates(
  response: Response
): Promise<InpaintCandidatesResponse> {
  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }
  const contentType = response.headers.get("content-type") ?? "";
  const boundary = parseMultipartBoundary(contentType);
  if (!boundary) {
    throw new Error("Expected multipart candidate response.");
  }

  const raw = new Uint8Array(await response.arrayBuffer());
  const textDecoder = new TextDecoder();
  const boundaryToken = new TextEncoder().encode(`--${boundary}`);
  const crlfcrlf = new Uint8Array([0x0d, 0x0a, 0x0d, 0x0a]);

  const findSequence = (haystack: Uint8Array, needle: Uint8Array, from = 0): number => {
    outer: for (let i = from; i <= haystack.length - needle.length; i += 1) {
      for (let j = 0; j < needle.length; j += 1) {
        if (haystack[i + j] !== needle[j]) continue outer;
      }
      return i;
    }
    return -1;
  };

  let metadata: InpaintCandidatesMetadata | null = null;
  const blobs = new Map<string, Blob>();
  let offset = findSequence(raw, boundaryToken, 0);

  while (offset >= 0) {
    let partStart = offset + boundaryToken.length;
    if (raw[partStart] === 0x0d && raw[partStart + 1] === 0x0a) {
      partStart += 2;
    }
    const nextBoundary = findSequence(raw, boundaryToken, partStart);
    const partEnd = nextBoundary >= 0 ? nextBoundary - 2 : raw.length;
    const part = raw.slice(partStart, partEnd);
    const headerEnd = findSequence(part, crlfcrlf, 0);
    if (headerEnd >= 0) {
      const headers = textDecoder.decode(part.slice(0, headerEnd));
      const body = part.slice(headerEnd + 4);
      const nameMatch = /name="([^"]+)"/.exec(headers);
      if (nameMatch) {
        const name = nameMatch[1];
        if (name === "metadata") {
          metadata = JSON.parse(textDecoder.decode(body)) as InpaintCandidatesMetadata;
        } else {
          blobs.set(name, new Blob([body], { type: "image/png" }));
        }
      }
    }
    if (nextBoundary < 0) break;
    offset = nextBoundary;
    if (raw[offset + boundaryToken.length] === 0x2d && raw[offset + boundaryToken.length + 1] === 0x2d) {
      break;
    }
  }

  if (!metadata) {
    throw new Error("Candidate response missing metadata part.");
  }

  const candidates: InpaintCandidateResult[] = metadata.candidates.map((info) => {
    const blob = blobs.get(info.candidate_id);
    if (!blob) {
      throw new Error(`Missing PNG for ${info.candidate_id}.`);
    }
    return {
      id: info.candidate_id,
      blob,
      url: URL.createObjectURL(blob),
      rank: info.rank,
      score: info.score,
      seed: info.seed,
    };
  });

  return { metadata, candidates };
}

export type InpaintResponse =
  | { mode: "single"; blob: Blob; metadata: InpaintMetadata }
  | { mode: "candidates"; response: InpaintCandidatesResponse };

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

export async function routing(): Promise<RoutingResponse> {
  const response = await fetch(`${apiBaseUrl()}/routing`);
  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }
  return (await response.json()) as RoutingResponse;
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
  backend: InpaintBackend = "moebius",
  options?: { candidateCount?: 1 | 2 }
): Promise<InpaintResponse> {
  const form = new FormData();
  form.append("image", image, image.name || "image.png");
  form.append("mask", mask, "mask.png");
  form.append("backend", backend);
  const candidateCount = options?.candidateCount ?? 1;
  form.append("candidate_count", String(candidateCount));

  const response = await fetch(`${apiBaseUrl()}/inpaint`, {
    method: "POST",
    body: form,
  });

  if (candidateCount > 1) {
    return {
      mode: "candidates",
      response: await readMultipartInpaintCandidates(response),
    };
  }

  const single = await readPngResponse<InpaintMetadata>(response, [
    "model",
    "backend",
    "latency_ms",
    "memory_mb",
    "metadata",
  ]);
  return { mode: "single", blob: single.blob, metadata: single.metadata };
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

export async function selectSmart(
  image: File,
  options: {
    selectionMode?: SelectionMode;
    x?: number;
    y?: number;
    prompt?: string;
    detectionIndex?: number;
  }
): Promise<PngWithMetadata<SelectSmartMetadata>> {
  const form = new FormData();
  form.append("image", image, image.name || "image.png");
  form.append("selection_mode", options.selectionMode ?? "smart");
  if (options.x !== undefined) form.append("x", String(options.x));
  if (options.y !== undefined) form.append("y", String(options.y));
  if (options.prompt) form.append("prompt", options.prompt);
  if (options.detectionIndex !== undefined) {
    form.append("detection_index", String(options.detectionIndex));
  }
  form.append("grounding_backend", "grounding_dino");

  const response = await fetch(`${apiBaseUrl()}/select-smart`, {
    method: "POST",
    body: form,
  });

  return readPngResponse<SelectSmartMetadata>(response, [
    "selection_mode",
    "method",
    "confidence_tier",
    "model",
    "segmentation_model",
    "grounding_backend",
    "confidence",
    "prompt",
    "point_xy",
    "detection_index",
    "detection_count",
    "selected_label",
    "selected_box_xyxy",
    "detections",
    "ranking",
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
