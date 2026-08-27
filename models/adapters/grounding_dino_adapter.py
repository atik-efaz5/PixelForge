"""Grounding DINO adapter — open-vocabulary box grounding from text.

Imports the vendored GroundingDINO tree inside ``research/upstream/Grounded-Segment-Anything``
inside ``load()`` only. Upstream stays READ-ONLY.

Phase 11 audit classification: **CPU** (custom CUDA ops; no validated Apple MPS path).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from models.adapters._runtime import prepare_upstream_import
from models.adapters.base import GroundingAdapter
from models.config import model_entry, project_root, resolve_path
from models.errors import ModelInferenceError, ModelLoadError, ModelUnavailableError
from models.types import BackendType, BoundingBox, GroundingResult, validate_image


def preprocess_caption(caption: str) -> str:
    text = caption.strip().lower()
    if not text:
        return ""
    if not text.endswith("."):
        text = f"{text}."
    return text


def boxes_to_xyxy(
    boxes_cxcywh_norm: np.ndarray, width: int, height: int
) -> list[tuple[float, float, float, float]]:
    """Convert normalized cxcywh boxes to pixel XYXY."""
    scaled = boxes_cxcywh_norm * np.array([width, height, width, height], dtype=np.float32)
    xyxy: list[tuple[float, float, float, float]] = []
    for cx, cy, w, h in scaled:
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        xyxy.append((float(x1), float(y1), float(x2), float(y2)))
    return xyxy


class GroundingDINOAdapter(GroundingAdapter):
    """Grounding DINO SwinT OGC. Runs on CPU unless overridden."""

    def __init__(
        self,
        *,
        checkpoint_path: Path | None = None,
        config_file: Path | None = None,
        upstream_dir: Path | None = None,
        device: str | None = None,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
        enabled: bool | None = None,
    ) -> None:
        super().__init__()
        cfg = model_entry("grounding_dino")
        env_ckpt = os.environ.get("PIXELFORGE_GROUNDING_DINO_CHECKPOINT")
        self._checkpoint = (
            Path(env_ckpt).expanduser()
            if env_ckpt
            else checkpoint_path or resolve_path(cfg.get("checkpoint"))
        )
        self._config_file = config_file or resolve_path(
            cfg.get("config_file")
            or "research/upstream/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        )
        self._upstream = upstream_dir or resolve_path(cfg.get("upstream_dir"))
        self._grounding_root = (
            self._upstream / "GroundingDINO" if self._upstream is not None else None
        )
        self._device = device or str(cfg.get("device") or "cpu")
        self._box_threshold = float(
            box_threshold if box_threshold is not None else cfg.get("box_threshold", 0.3)
        )
        self._text_threshold = float(
            text_threshold if text_threshold is not None else cfg.get("text_threshold", 0.25)
        )
        self._enabled = cfg.get("enabled", True) if enabled is None else enabled
        self._display = str(cfg.get("display_name") or "Grounding DINO SwinT OGC")
        self._model: Any = None
        self._transform: Any = None

    @property
    def model_name(self) -> str:
        return self._display

    @property
    def backend_type(self) -> BackendType:
        return BackendType.CPU

    def is_available(self) -> bool:
        if not self._enabled:
            return False
        if self._checkpoint is None or not self._checkpoint.is_file():
            return False
        if self._config_file is None or not self._config_file.is_file():
            return False
        if self._grounding_root is None or not (self._grounding_root / "groundingdino").is_dir():
            return False
        return True

    def load(self) -> None:
        if self._loaded:
            return
        if not self.is_available():
            raise ModelUnavailableError("Grounding DINO is not available.")
        self._loading = True
        self._error = None
        try:
            if self._grounding_root is None:
                raise ModelUnavailableError("Grounding DINO upstream path is missing.")
            prepare_upstream_import(self._grounding_root)
            from groundingdino.datasets.transforms import Compose, Normalize, RandomResize, ToTensor
            from groundingdino.models import build_model
            from groundingdino.util.slconfig import SLConfig
            from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

            self._get_phrases_from_posmap = get_phrases_from_posmap
            args = SLConfig.fromfile(str(self._config_file))
            args.device = self._device
            model = build_model(args)
            import torch

            checkpoint = torch.load(str(self._checkpoint), map_location="cpu")
            model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
            model.eval()
            self._model = model.to(self._device)
            self._transform = Compose(
                [
                    RandomResize([800], max_size=1333),
                    ToTensor(),
                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            self._loaded = True
        except (ModelLoadError, ModelUnavailableError):
            raise
        except Exception as exc:
            self._error = type(exc).__name__
            raise ModelLoadError("Grounding DINO failed to load.") from exc
        finally:
            self._loading = False

    def unload(self) -> None:
        self._model = None
        self._transform = None
        self._loaded = False
        self._loading = False

    def ground(self, image: np.ndarray, text_prompt: str) -> GroundingResult:
        return self.infer(image, text_prompt)

    def infer(self, image: np.ndarray, text_prompt: str, /) -> GroundingResult:
        image = validate_image(image)
        caption = preprocess_caption(text_prompt)
        if not caption:
            raise ModelInferenceError("Text prompt is empty.")
        if not self._loaded:
            self.load()
        if self._model is None or self._transform is None:
            raise ModelLoadError("Grounding DINO is not loaded.")

        from PIL import Image

        import torch

        h, w = image.shape[:2]
        pil = Image.fromarray(image, mode="RGB")
        tensor, _ = self._transform(pil, None)
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                outputs = self._model(tensor[None].to(self._device), captions=[caption])
            logits = outputs["pred_logits"].cpu().sigmoid()[0]
            boxes = outputs["pred_boxes"].cpu()[0]
            filt = logits.max(dim=1)[0] > self._box_threshold
            logits_filt = logits[filt]
            boxes_filt = boxes[filt]
            tokenizer = self._model.tokenizer
            tokenized = tokenizer(caption)
            detections: list[BoundingBox] = []
            xyxy_list = boxes_to_xyxy(boxes_filt.numpy(), w, h)
            for logit_row, box_xyxy in zip(logits_filt, xyxy_list):
                phrase = self._get_phrases_from_posmap(
                    logit_row > self._text_threshold, tokenized, tokenizer
                )
                label = phrase.replace(".", "").strip() or text_prompt.strip()
                confidence = float(logit_row.max().item())
                x1, y1, x2, y2 = box_xyxy
                detections.append(
                    BoundingBox(
                        x1=max(0.0, x1),
                        y1=max(0.0, y1),
                        x2=min(float(w - 1), x2),
                        y2=min(float(h - 1), y2),
                        confidence=confidence,
                        label=label,
                    )
                )
        except Exception as exc:
            self._error = type(exc).__name__
            raise ModelInferenceError("Grounding DINO inference failed.") from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return GroundingResult(
            detections=detections,
            model=self.model_name,
            prompt=text_prompt.strip(),
            backend=self.backend_type,
            metadata={
                "device": self._device,
                "latency_ms": round(latency_ms, 3),
                "box_threshold": self._box_threshold,
                "text_threshold": self._text_threshold,
                "checkpoint": str(self._checkpoint),
                "project_root": str(project_root()),
            },
        )
