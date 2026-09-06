"""Unit tests for persistent worker IPC and lifecycle."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from apps.backend.isolated_runner import (
    persistent_moebius_enabled,
    shutdown_persistent_workers,
)
from apps.backend.persistent_worker import LineJsonWorkerClient, WorkerTimeoutError
from models.errors import ModelInferenceError


class TestPersistentWorkerConfig(unittest.TestCase):
    def tearDown(self) -> None:
        from apps.backend.settings import reset_settings_cache

        reset_settings_cache()

    def test_enabled_by_default(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            from apps.backend.settings import reset_settings_cache

            reset_settings_cache()
            self.assertTrue(persistent_moebius_enabled())

    def test_disabled_with_zero(self) -> None:
        with mock.patch.dict("os.environ", {"PIXELFORGE_MOEBIUS_PERSISTENT_WORKER": "0"}):
            from apps.backend.settings import reset_settings_cache

            reset_settings_cache()
            self.assertFalse(persistent_moebius_enabled())


class TestLineJsonWorkerClient(unittest.TestCase):
    def tearDown(self) -> None:
        shutdown_persistent_workers()

    def test_start_returns_ready_payload(self) -> None:
        client = LineJsonWorkerClient(
            name="test",
            python=Path("/fake/python"),
            script=Path("/fake/script.py"),
        )
        with (
            mock.patch.object(client, "_spawn"),
            mock.patch.object(
                client,
                "_read_line_locked",
                return_value={"event": "ready", "load_ms": 12.3},
            ),
        ):
            client._proc = mock.Mock()  # noqa: SLF001
            client._proc.poll.return_value = None
            ready = client.start()
        self.assertEqual(ready["event"], "ready")

    def test_request_delegates_to_locked_io(self) -> None:
        client = LineJsonWorkerClient(
            name="test",
            python=Path("/fake/python"),
            script=Path("/fake/script.py"),
        )
        client._proc = mock.Mock()  # noqa: SLF001
        client._proc.poll.return_value = None

        with (
            mock.patch.object(client, "_write_line_locked") as write_mock,
            mock.patch.object(
                client,
                "_read_line_locked",
                return_value={"id": "1", "ok": True, "value": 7},
            ),
        ):
            out = client.request({"id": "1", "cmd": "ping"})
        self.assertEqual(out["value"], 7)
        write_mock.assert_called_once()

    def test_request_raises_on_worker_error(self) -> None:
        client = LineJsonWorkerClient(
            name="test",
            python=Path("/fake/python"),
            script=Path("/fake/script.py"),
        )
        client._proc = mock.Mock()  # noqa: SLF001
        client._proc.poll.return_value = None

        with (
            mock.patch.object(client, "_write_line_locked"),
            mock.patch.object(
                client,
                "_read_line_locked",
                return_value={"id": "1", "ok": False, "error": "boom"},
            ),
        ):
            with self.assertRaises(ModelInferenceError):
                client.request({"id": "1", "cmd": "inpaint"})

    def test_shutdown_calls_terminate_when_handshake_fails(self) -> None:
        client = LineJsonWorkerClient(
            name="test",
            python=Path("/fake/python"),
            script=Path("/fake/script.py"),
        )
        proc = mock.Mock()
        proc.poll.return_value = None
        proc.wait.return_value = 0
        client._proc = proc  # noqa: SLF001

        with (
            mock.patch.object(client, "_write_line_locked"),
            mock.patch.object(client, "_read_line_locked", side_effect=RuntimeError("no reply")),
        ):
            client.shutdown()
        proc.terminate.assert_called_once()

    def test_timeout_raises_worker_timeout_error(self) -> None:
        client = LineJsonWorkerClient(
            name="test",
            python=Path("/fake/python"),
            script=Path("/fake/script.py"),
            request_timeout_sec=0.01,
        )
        proc = mock.Mock()
        proc.poll.return_value = None
        proc.stdout = mock.Mock()
        client._proc = proc  # noqa: SLF001

        with mock.patch("select.select", return_value=([], [], [])):
            with self.assertRaises(WorkerTimeoutError):
                client._read_line_locked(0.01)  # noqa: SLF001


class TestIsolatedRunnerRouting(unittest.TestCase):
    def test_fastapi_module_has_no_moebius_import(self) -> None:
        import ast

        source = Path("apps/backend/main.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        import_from = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertNotIn("moebius", imports)
        self.assertFalse(any("Moebius" in (m or "") for m in import_from))

    @mock.patch("apps.backend.isolated_runner.inpaint_via_persistent_worker")
    @mock.patch("apps.backend.isolated_runner.inpaint_via_oneshot_subprocess")
    def test_routes_to_persistent_by_default(
        self, oneshot_mock: mock.Mock, persistent_mock: mock.Mock
    ) -> None:
        import numpy as np

        from apps.backend.isolated_runner import inpaint_via_isolated_env

        image = np.zeros((4, 4, 3), dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=bool)
        persistent_mock.return_value = mock.Mock()

        with mock.patch(
            "apps.backend.isolated_runner.persistent_moebius_enabled",
            return_value=True,
        ):
            inpaint_via_isolated_env(image, mask)
        persistent_mock.assert_called_once()
        oneshot_mock.assert_not_called()

    @mock.patch("apps.backend.isolated_runner.inpaint_via_persistent_worker")
    @mock.patch("apps.backend.isolated_runner.inpaint_via_oneshot_subprocess")
    def test_falls_back_to_oneshot_when_persistent_disabled(
        self, oneshot_mock: mock.Mock, persistent_mock: mock.Mock
    ) -> None:
        import numpy as np

        from apps.backend.isolated_runner import inpaint_via_isolated_env

        image = np.zeros((4, 4, 3), dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=bool)
        oneshot_mock.return_value = mock.Mock()

        with mock.patch(
            "apps.backend.isolated_runner.persistent_moebius_enabled",
            return_value=False,
        ):
            inpaint_via_isolated_env(image, mask)
        oneshot_mock.assert_called_once()
        persistent_mock.assert_not_called()

    @mock.patch("apps.backend.isolated_runner.inpaint_via_persistent_worker")
    @mock.patch("apps.backend.isolated_runner.inpaint_via_oneshot_subprocess")
    def test_falls_back_to_oneshot_on_model_inference_error(
        self, oneshot_mock: mock.Mock, persistent_mock: mock.Mock
    ) -> None:
        import numpy as np

        from apps.backend.isolated_runner import inpaint_via_isolated_env

        image = np.zeros((4, 4, 3), dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=bool)
        persistent_mock.side_effect = ModelInferenceError("Moebius inference failed.")
        oneshot_mock.return_value = mock.Mock()

        with mock.patch(
            "apps.backend.isolated_runner.persistent_moebius_enabled",
            return_value=True,
        ):
            inpaint_via_isolated_env(image, mask)
        persistent_mock.assert_called_once()
        oneshot_mock.assert_called_once()

    def test_worker_error_includes_type_and_cause(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "moebius_persistent_worker",
            Path("scripts/moebius_persistent_worker.py"),
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        try:
            raise RuntimeError("shape mismatch")
        except RuntimeError as inner:
            try:
                raise ModelInferenceError("Moebius inference failed.") from inner
            except ModelInferenceError as outer:
                message = module._error_message(outer)

        self.assertIn("ModelInferenceError", message)
        self.assertIn("Moebius inference failed.", message)
        self.assertIn("RuntimeError", message)
        self.assertIn("shape mismatch", message)


if __name__ == "__main__":
    unittest.main()
