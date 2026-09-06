import { describe, expect, it } from "vitest";

import type { RoutingResponse } from "@/types/api";
import {
  isInstructionEditAvailable,
  type EditingCapabilitiesResponse,
} from "./editorCapabilities";

const routingUnavailable: RoutingResponse = {
  capabilities: [
    {
      capability: "global_instruction_edit",
      models: [
        {
          model: "instruct_pix2pix",
          backend: "CLOUD_GPU",
          available: false,
          runtime_validated: false,
          status: "UNAVAILABLE",
          display_name: "InstructPix2Pix",
        },
      ],
    },
  ],
  operations: [],
  execution_preferences: [],
  automatic_backend_aliases: [],
  known_models: ["instruct_pix2pix"],
};

const capabilitiesDeclared: EditingCapabilitiesResponse = {
  capabilities: [
    {
      backend_id: "instruct_pix2pix",
      localized_inpaint: false,
      semantic_replace: false,
      global_instruction_edit: true,
      mask_conditioned_edit: false,
      accepts_text_instruction: true,
      accepts_reference_image: false,
      notes: "",
    },
  ],
};

describe("isInstructionEditAvailable", () => {
  it("is false when InstructPix2Pix is declared but not available", () => {
    expect(
      isInstructionEditAvailable(routingUnavailable, capabilitiesDeclared)
    ).toBe(false);
  });

  it("is false when capabilities declare instruction edit unsupported", () => {
    const capabilities: EditingCapabilitiesResponse = {
      capabilities: [
        {
          ...capabilitiesDeclared.capabilities[0],
          global_instruction_edit: false,
        },
      ],
    };
    const routingAvailable = {
      ...routingUnavailable,
      capabilities: [
        {
          capability: "global_instruction_edit",
          models: [
            {
              ...routingUnavailable.capabilities[0].models[0],
              available: true,
              status: "READY" as const,
            },
          ],
        },
      ],
    };
    expect(isInstructionEditAvailable(routingAvailable, capabilities)).toBe(false);
  });

  it("is true only when the cloud model is available", () => {
    const routing = {
      ...routingUnavailable,
      capabilities: [
        {
          capability: "global_instruction_edit",
          models: [
            {
              ...routingUnavailable.capabilities[0].models[0],
              available: true,
              status: "READY" as const,
            },
          ],
        },
      ],
    };
    expect(isInstructionEditAvailable(routing, capabilitiesDeclared)).toBe(true);
  });
});
