export type BackendType =
  | "LOCAL_MPS"
  | "CLOUD_GPU"
  | "CPU"
  | "UNAVAILABLE";

export type ModelStatus = "READY" | "UNAVAILABLE" | "LOADING" | "ERROR";

export interface HealthResponse {
  status: string;
  version?: string;
  max_upload_bytes?: number;
  max_concurrent_generations?: number;
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

export interface RoutingCapabilityEntry {
  capability: string;
  models: Array<{
    model: string;
    backend: BackendType;
    available: boolean;
    runtime_validated: boolean;
    status: ModelStatus;
    display_name: string;
  }>;
}

export interface RoutingResponse {
  capabilities: RoutingCapabilityEntry[];
  operations: Array<{ operation: string; required_capability: string }>;
  execution_preferences: string[];
  automatic_backend_aliases: string[];
  known_models: string[];
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

export interface InpaintCandidateInfo {
  candidate_id: string;
  rank: number;
  score: number;
  seed: number | null;
  latency_ms: number;
  memory_mb: number | null;
  output_hash: string;
  validity_status: string;
  generation_params: Record<string, unknown>;
  score_components: Record<string, unknown>;
}

export interface InpaintCandidatesMetadata {
  model: string;
  backend: string;
  candidate_count: number;
  selected_candidate_id: string;
  candidates: InpaintCandidateInfo[];
  ranking: Record<string, unknown>;
  latency_ms: number;
  memory_mb: number | null;
  metadata: Record<string, unknown>;
}

export interface InpaintCandidateResult {
  id: string;
  blob: Blob;
  url: string;
  rank: number;
  score: number;
  seed: number | null;
}

export interface InpaintCandidatesResponse {
  metadata: InpaintCandidatesMetadata;
  candidates: InpaintCandidateResult[];
}

export interface EditByInstructionMetadata {
  model: string;
  backend: string;
  instruction: string;
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
  error: string | { code: string; message: string };
  code?: string;
  message: string;
  details?: unknown;
}

export interface PngWithMetadata<T> {
  blob: Blob;
  metadata: T;
}

export interface SelectByTextMetadata {
  prompt: string;
  model: string;
  segmentation_model: string;
  grounding_backend: string;
  confidence: number | null;
  method: string;
  detection_index: number;
  detection_count: number;
  selected_label: string;
  selected_box_xyxy: number[];
  detections: DetectionInfo[];
  metadata: Record<string, unknown>;
}

export interface DetectionInfo {
  index: number;
  label: string;
  confidence: number;
  box_xyxy: number[];
}

export type SelectionMode = "smart" | "point" | "text";

export type ConfidenceTier = "HIGH" | "MEDIUM" | "LOW";

export interface SelectSmartMetadata {
  selection_mode: string;
  method: string;
  confidence_tier: ConfidenceTier;
  model: string;
  segmentation_model: string;
  grounding_backend?: string | null;
  confidence: number | null;
  prompt?: string | null;
  point_xy?: number[] | null;
  detection_index?: number | null;
  detection_count?: number | null;
  selected_label?: string | null;
  selected_box_xyxy?: number[] | null;
  detections: DetectionInfo[];
  ranking: Record<string, unknown>;
  metadata: Record<string, unknown>;
}

export type InpaintBackend = "auto" | "moebius";

export type EditorTool = "select" | "brush" | "erase";

export type EditorStatus =
  | "idle"
  | "uploading"
  | "segmenting"
  | "grounding"
  | "generating"
  | "instruction_editing";

export interface ImageDimensions {
  width: number;
  height: number;
}
