export interface ModelLine {
  label: string;
  model: string;
  backend: string;
  route?: string;
  confidenceTier?: string;
}

const MODEL_NAMES: Record<string, string> = {
  sam2: "SAM 2",
  grounding_dino: "Grounding DINO",
  moebius: "Moebius",
  pixelhacker: "PixelHacker",
  instruct_pix2pix: "InstructPix2Pix",
};

const BACKEND_NAMES: Record<string, string> = {
  LOCAL_MPS: "Local",
  CLOUD_GPU: "Cloud",
  CPU: "CPU",
  UNAVAILABLE: "Unavailable",
};

export function formatModelName(id: string | undefined): string {
  if (!id) return "—";
  const key = id.trim().toLowerCase();
  return MODEL_NAMES[key] ?? id;
}

export function formatBackendName(backend: string | undefined): string {
  if (!backend) return "—";
  const normalized = backend.trim().toUpperCase().replace(/\s+/g, "_");
  return BACKEND_NAMES[normalized] ?? backend;
}

export function formatInpaintLine(input: {
  requestedBackend?: string;
  model?: string;
  backend?: string;
  routing?: { model?: string; backend?: string };
}): ModelLine {
  const resolvedModel =
    input.routing?.model ?? input.model ?? input.requestedBackend ?? "—";
  const resolvedBackend =
    input.routing?.backend ?? input.backend ?? "—";
  const requested = (input.requestedBackend ?? "").trim().toLowerCase();
  const isAuto = requested === "auto" || requested === "automatic" || !requested;

  return {
    label: "Inpainting",
    model: formatModelName(resolvedModel),
    backend: formatBackendName(resolvedBackend),
    route: isAuto
      ? `Automatic → ${formatModelName(resolvedModel)}`
      : undefined,
  };
}

export function formatSmartSelectionLine(input: {
  selectionMode: string;
  method: "point" | "text";
  confidenceTier: string;
}): ModelLine {
  const modeLabel =
    input.selectionMode === "smart"
      ? "Smart"
      : input.selectionMode === "point"
        ? "Point"
        : "Text";
  const path = input.method === "text" ? "Grounding DINO → SAM 2" : "SAM 2";
  return {
    label: "Selection",
    model: `${modeLabel} → ${path}`,
    backend: input.method === "text" ? "CPU → Local" : "Local",
    confidenceTier: input.confidenceTier,
  };
}

export function formatSelectionLine(input: {
  method?: "point" | "text";
  model?: string;
  backend?: string;
  segmentationModel?: string;
  groundingBackend?: string;
}): ModelLine {
  if (input.method === "text") {
    const seg = formatModelName(input.segmentationModel ?? "sam2");
    const ground = formatModelName(input.model ?? "grounding_dino");
    const groundBackend = formatBackendName(input.groundingBackend ?? "CPU");
    const segBackend = formatBackendName(input.backend ?? "LOCAL_MPS");
    return {
      label: "Selection",
      model: `${ground} → ${seg}`,
      backend: `${groundBackend} → ${segBackend}`,
    };
  }
  return {
    label: "Selection",
    model: formatModelName(input.model ?? "sam2"),
    backend: formatBackendName(input.backend ?? "LOCAL_MPS"),
  };
}
