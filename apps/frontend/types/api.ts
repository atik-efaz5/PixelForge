export type BackendType =
  | "LOCAL_MPS"
  | "CLOUD_GPU"
  | "CPU"
  | "UNAVAILABLE";

export type ModelStatus = "READY" | "UNAVAILABLE" | "LOADING" | "ERROR";

export interface HealthResponse {
  status: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  role: string;
  backend: BackendType;
  available: boolean;
  loaded: boolean;
  status: ModelStatus;
}

export interface ModelsResponse {
  models: ModelInfo[];
}

export interface SegmentMetadata {
  confidence: number | null;
  model: string;
  backend: string;
  method: string;
  metadata: Record<string, unknown>;
}

export interface InpaintMetadata {
  model: string;
  backend: string;
  latency_ms: number;
  memory_mb: number | null;
  metadata: Record<string, unknown>;
}

export interface RemoveObjectMetadata {
  model: string;
  backend: string;
  segmentation_model: string | null;
  latency_ms: number;
  segmentation_ms: number | null;
  inpainting_ms: number | null;
  metadata: Record<string, unknown>;
}

export interface ApiErrorBody {
  error: string;
  message: string;
  details?: unknown;
}

export interface PngWithMetadata<T> {
  blob: Blob;
  metadata: T;
}

export type InpaintBackend = "moebius";

export type EditorTool = "select" | "brush" | "erase";

export interface ImageDimensions {
  width: number;
  height: number;
}
