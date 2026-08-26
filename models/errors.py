"""Application-facing model errors. Avoid leaking research-repo tracebacks."""


class PixelForgeModelError(Exception):
    """Base class for adapter-layer failures."""


class ModelUnavailableError(PixelForgeModelError):
    """The model cannot be used in this process (missing weights, device, or config)."""


class ModelLoadError(PixelForgeModelError):
    """Weights or runtime failed to load. The adapter is not ready."""


class ModelInferenceError(PixelForgeModelError):
    """A loaded model failed while producing an output."""
