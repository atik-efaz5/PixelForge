"""Unit tests for production hardening controls."""

from __future__ import annotations

import io
import json
import threading
import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from apps.backend.concurrency import generation_slot, reset_generation_slots
from apps.backend.errors import InvalidInputError, ServiceBusyError, error_body, register_exception_handlers
from apps.backend.main import create_app
from apps.backend.settings import get_settings, reset_settings_cache
from apps.backend.validation import (
    normalize_upload_image,
    sanitize_filename,
    validate_text_field,
    validate_upload_bytes,
    verify_image_magic,
)


def _png_bytes(size: tuple[int, int] = (16, 16)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=(10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


class TestSettings(unittest.TestCase):
    def tearDown(self) -> None:
        reset_settings_cache()
        reset_generation_slots()

    def test_settings_load_defaults(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            reset_settings_cache()
            settings = get_settings()
        self.assertEqual(settings.max_upload_bytes, 25 * 1024 * 1024)
        self.assertEqual(settings.max_concurrent_generations, 1)

    def test_settings_env_override(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {"PIXELFORGE_MAX_UPLOAD_BYTES": "1024", "PIXELFORGE_MAX_CONCURRENT_GENERATIONS": "2"},
            clear=True,
        ):
            reset_settings_cache()
            reset_generation_slots()
            settings = get_settings()
        self.assertEqual(settings.max_upload_bytes, 1024)
        self.assertEqual(settings.max_concurrent_generations, 2)


class TestUploadValidation(unittest.TestCase):
    def test_rejects_oversized_bytes(self) -> None:
        with self.assertRaises(InvalidInputError) as ctx:
            validate_upload_bytes(b"x" * 20, max_bytes=10, label="Image upload")
        self.assertIn("maximum size", str(ctx.exception).lower())

    def test_rejects_non_image_magic(self) -> None:
        with self.assertRaises(InvalidInputError):
            verify_image_magic(b"not-an-image")

    def test_rejects_oversized_decode(self) -> None:
        raw = _png_bytes()
        self.assertGreater(len(raw), 50)
        with mock.patch.dict(
            "os.environ",
            {"PIXELFORGE_MAX_UPLOAD_BYTES": "50"},
            clear=True,
        ):
            reset_settings_cache()
            with self.assertRaises(InvalidInputError):
                validate_upload_bytes(raw, max_bytes=50, label="Image upload")


class TestFilenameSanitization(unittest.TestCase):
    def test_strips_path_traversal(self) -> None:
        self.assertEqual(
            sanitize_filename("../../etc/passwd"),
            "passwd",
        )
        self.assertEqual(
            sanitize_filename("..\\..\\evil.png"),
            "evil.png",
        )


class TestTextValidation(unittest.TestCase):
    def test_rejects_overlong_prompt(self) -> None:
        with self.assertRaises(InvalidInputError):
            validate_text_field("x" * 600, field_name="prompt", max_length=512)


class TestConcurrency(unittest.TestCase):
    def tearDown(self) -> None:
        reset_generation_slots()

    def test_second_generation_rejected_when_busy(self) -> None:
        barrier = threading.Barrier(2, timeout=5)
        errors: list[Exception] = []

        def hold_slot() -> None:
            try:
                with generation_slot(operation="test", request_id="a"):
                    barrier.wait()
                    barrier.wait()
            except Exception as exc:
                errors.append(exc)

        t = threading.Thread(target=hold_slot)
        t.start()
        barrier.wait()
        try:
            with generation_slot(operation="test", request_id="b"):
                pass
        except ServiceBusyError:
            pass
        else:
            self.fail("expected ServiceBusyError")
        barrier.wait()
        t.join(timeout=5)


class TestErrorSchema(unittest.TestCase):
    def test_error_body_includes_request_id(self) -> None:
        body = error_body("invalid_input", "bad", request_id="abc123")
        self.assertEqual(body["request_id"], "abc123")
        self.assertEqual(body["error"]["request_id"], "abc123")


class TestHealthEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app())

    def test_health_returns_limits(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("max_upload_bytes", payload)


class TestWorkerMalformedResponse(unittest.TestCase):
    def test_persistent_worker_rejects_oversized_json(self) -> None:
        from pathlib import Path

        from apps.backend.persistent_worker import LineJsonWorkerClient
        from models.errors import ModelInferenceError

        client = LineJsonWorkerClient(
            name="test",
            python=Path("/fake/python"),
            script=Path("/fake/script.py"),
        )
        huge_line = "{" + ("x" * (8 * 1024 * 1024 + 10)) + "}"
        proc = mock.Mock()
        proc.poll.return_value = None
        stdout = mock.Mock()
        stdout.readline.return_value = huge_line
        stdout.fileno.return_value = 1
        proc.stdout = stdout
        client._proc = proc  # noqa: SLF001

        with mock.patch("select.select", return_value=([stdout], [], [])):
            with self.assertRaises(ModelInferenceError):
                client._read_line_locked(1.0)  # noqa: SLF001


if __name__ == "__main__":
    unittest.main()
