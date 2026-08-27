"""Registry lookup, import-time laziness, and availability probes (no inference)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from models.adapters.grounding_dino_adapter import GroundingDINOAdapter
from models.adapters.moebius_adapter import MoebiusAdapter
from models.adapters.pixelhacker_adapter import PixelHackerAdapter
from models.adapters.sam2_adapter import SAM2Adapter
from models.errors import ModelInferenceError, ModelUnavailableError
from models.registry import get_adapter, known_models, reset_registry
from models.types import BackendType, ModelStatus


class TestAdapterRegistry(unittest.TestCase):
    def setUp(self) -> None:
        reset_registry()
        self._endpoint_env = os.environ.pop("PIXELFORGE_PIXELHACKER_ENDPOINT", None)

    def tearDown(self) -> None:
        reset_registry()
        if self._endpoint_env is None:
            os.environ.pop("PIXELFORGE_PIXELHACKER_ENDPOINT", None)
        else:
            os.environ["PIXELFORGE_PIXELHACKER_ENDPOINT"] = self._endpoint_env

    def test_known_models(self) -> None:
        self.assertEqual(
            known_models(),
            ("sam2", "moebius", "pixelhacker", "instruct_pix2pix", "grounding_dino"),
        )

    def test_registry_lookup(self) -> None:
        sam2 = get_adapter("sam2")
        moebius = get_adapter("moebius")
        pixelhacker = get_adapter("pixelhacker")
        instruct = get_adapter("instruct_pix2pix")
        grounding_dino = get_adapter("grounding_dino")
        self.assertEqual(sam2.model_name, "SAM 2.1 Hiera-Tiny")
        self.assertIs(sam2.backend_type, BackendType.LOCAL_MPS)
        self.assertEqual(moebius.model_name, "Moebius")
        self.assertIs(moebius.backend_type, BackendType.LOCAL_MPS)
        self.assertEqual(pixelhacker.model_name, "PixelHacker")
        self.assertIs(pixelhacker.backend_type, BackendType.CLOUD_GPU)
        self.assertEqual(instruct.model_name, "InstructPix2Pix")
        self.assertIs(instruct.backend_type, BackendType.CLOUD_GPU)
        self.assertEqual(grounding_dino.model_name, "Grounding DINO SwinT OGC")
        self.assertIs(grounding_dino.backend_type, BackendType.CPU)

    def test_registry_is_case_insensitive_and_cached(self) -> None:
        self.assertIs(get_adapter("SAM2"), get_adapter("sam2"))

    def test_unknown_model(self) -> None:
        with self.assertRaisesRegex(ModelUnavailableError, "Unknown model"):
            get_adapter("brushnet")

    def test_import_and_lookup_do_not_load_research_models(self) -> None:
        get_adapter("sam2")
        get_adapter("moebius")
        get_adapter("pixelhacker")
        get_adapter("instruct_pix2pix")
        get_adapter("grounding_dino")
        leaked = [
            name
            for name in sys.modules
            if name == "sam2"
            or name.startswith("sam2.")
            or name == "removal"
            or name.startswith("removal.")
            or name.startswith("gla_model")
            or name == "model_lib"
        ]
        self.assertEqual(leaked, [])
        self.assertFalse(get_adapter("sam2").is_loaded)
        self.assertFalse(get_adapter("moebius").is_loaded)
        self.assertIs(get_adapter("sam2").status, ModelStatus.UNAVAILABLE)

    def test_sam2_unavailable_without_checkpoint(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        adapter = SAM2Adapter(checkpoint_path=tmp / "missing.pt", upstream_dir=tmp)
        adapter._mps_is_available = lambda: True  # type: ignore[method-assign]
        self.assertFalse(adapter.is_available())

    def test_sam2_available_with_checkpoint_and_mps(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        ckpt = tmp / "sam2.1_hiera_tiny.pt"
        ckpt.write_bytes(b"placeholder")
        (tmp / "sam2").mkdir()
        adapter = SAM2Adapter(checkpoint_path=ckpt, upstream_dir=tmp)
        adapter._mps_is_available = lambda: True  # type: ignore[method-assign]
        self.assertTrue(adapter.is_available())
        self.assertFalse(adapter.is_loaded)

    def test_sam2_unavailable_without_mps(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        ckpt = tmp / "sam2.1_hiera_tiny.pt"
        ckpt.write_bytes(b"placeholder")
        (tmp / "sam2").mkdir()
        adapter = SAM2Adapter(checkpoint_path=ckpt, upstream_dir=tmp)
        adapter._mps_is_available = lambda: False  # type: ignore[method-assign]
        self.assertFalse(adapter.is_available())

    def test_moebius_unavailable_without_checkpoints(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        adapter = MoebiusAdapter(
            checkpoint_path=tmp / "missing.bin",
            vae_dir=tmp / "vae",
            upstream_dir=tmp,
        )
        adapter._mps_is_available = lambda: True  # type: ignore[method-assign]
        self.assertFalse(adapter.is_available())

    def test_moebius_available_with_files_and_mps(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        student = tmp / "student.bin"
        student.write_bytes(b"placeholder")
        vae = tmp / "vae"
        vae.mkdir()
        (vae / "config.json").write_text("{}")
        (vae / "diffusion_pytorch_model.bin").write_bytes(b"placeholder")
        adapter = MoebiusAdapter(
            checkpoint_path=student,
            vae_dir=vae,
            upstream_dir=tmp,
        )
        adapter._mps_is_available = lambda: True  # type: ignore[method-assign]
        self.assertTrue(adapter.is_available())
        self.assertFalse(adapter.is_loaded)

    def test_pixelhacker_unavailable_without_endpoint(self) -> None:
        adapter = PixelHackerAdapter(endpoint="")
        self.assertFalse(adapter.is_available())
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=bool)
        with self.assertRaisesRegex(ModelUnavailableError, "not configured"):
            adapter.infer(image, mask)

    def test_pixelhacker_available_with_endpoint_but_infer_not_implemented(self) -> None:
        os.environ["PIXELFORGE_PIXELHACKER_ENDPOINT"] = "https://example.invalid/inpaint"
        adapter = PixelHackerAdapter()
        self.assertTrue(adapter.is_available())
        self.assertIs(adapter.backend_type, BackendType.CLOUD_GPU)
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=bool)
        with self.assertRaisesRegex(ModelInferenceError, "not implemented"):
            adapter.infer(image, mask)

    def test_pixelhacker_request_shape(self) -> None:
        adapter = PixelHackerAdapter(endpoint="https://example.invalid/inpaint")
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=bool)
        payload = adapter.build_remote_request(image, mask)
        self.assertEqual(payload["endpoint"], "https://example.invalid/inpaint")
        self.assertEqual(payload["image_shape"], [8, 8, 3])
        self.assertEqual(payload["num_steps"], 20)
        self.assertEqual(payload["guidance_scale"], 4.5)
        self.assertFalse(payload["paste"])

    def test_grounding_dino_unavailable_without_checkpoint(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        adapter = GroundingDINOAdapter(
            checkpoint_path=tmp / "missing.pth",
            config_file=tmp / "config.py",
            upstream_dir=tmp,
        )
        self.assertFalse(adapter.is_available())
        self.assertIs(adapter.backend_type, BackendType.CPU)


if __name__ == "__main__":
    unittest.main()
