"""Pipeline-level validation errors. Adapter failures use ``models.errors``."""


class PipelineValidationError(ValueError):
    """Invalid pipeline input or mask state."""


class PipelineBackendError(ValueError):
    """Unsupported or unavailable inpainting backend was requested explicitly."""


class PipelinePromptError(ValueError):
    """Invalid or empty text prompt."""
