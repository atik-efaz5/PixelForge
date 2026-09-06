import type { RoutingResponse } from "@/types/api";

export interface EditingCapabilityEntry {
  backend_id: string;
  localized_inpaint: boolean;
  semantic_replace: boolean;
  global_instruction_edit: boolean;
  mask_conditioned_edit: boolean;
  accepts_text_instruction: boolean;
  accepts_reference_image: boolean;
  notes: string;
}

export interface EditingCapabilitiesResponse {
  capabilities: EditingCapabilityEntry[];
}

/** True only when InstructPix2Pix is declared AND actually available. */
export function isInstructionEditAvailable(
  routing: RoutingResponse | null,
  capabilities: EditingCapabilitiesResponse | null
): boolean {
  const declared = capabilities?.capabilities.some(
    (row) => row.backend_id === "instruct_pix2pix" && row.global_instruction_edit
  );
  if (declared === false) return false;
  const cap = routing?.capabilities.find(
    (entry) => entry.capability === "global_instruction_edit"
  );
  return Boolean(
    cap?.models.some((model) => model.model === "instruct_pix2pix" && model.available)
  );
}
