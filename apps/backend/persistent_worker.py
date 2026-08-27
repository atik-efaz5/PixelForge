"""Persistent subprocess worker client (newline-delimited JSON IPC)."""

from __future__ import annotations

import atexit
import json
import logging
import os
import select
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from apps.backend.settings import get_settings
from models.config import project_root
from models.errors import ModelInferenceError, ModelLoadError

logger = logging.getLogger(__name__)

_MAX_JSON_LINE_BYTES = 8 * 1024 * 1024


class WorkerCrashedError(RuntimeError):
    """Worker process exited unexpectedly."""


class WorkerTimeoutError(TimeoutError):
    """Worker did not respond within the allotted time."""


class LineJsonWorkerClient:
    """Manage a long-lived worker subprocess with one-request-at-a-time IPC."""

    def __init__(
        self,
        *,
        name: str,
        python: Path,
        script: Path,
        startup_timeout_sec: float | None = None,
        request_timeout_sec: float | None = None,
    ) -> None:
        self.name = name
        self.python = python
        self.script = script
        settings = get_settings()
        self.startup_timeout_sec = startup_timeout_sec or settings.worker_startup_timeout_sec
        self.request_timeout_sec = request_timeout_sec or settings.worker_request_timeout_sec
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._started = False
        self._stderr_thread: threading.Thread | None = None
        self._stderr_tail = ""

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> dict[str, Any]:
        """Start the worker and wait for the ready event."""
        with self._lock:
            if self.is_running:
                return {"event": "ready", "reused": True}
            self._spawn()
            assert self._proc is not None
            ready = self._read_line_locked(self.startup_timeout_sec)
            if ready.get("event") != "ready":
                self._terminate_locked()
                raise ModelLoadError(
                    f"{self.name} worker failed to start: {ready!r}"
                )
            self._started = True
            logger.info(
                "%s worker ready pid=%s load_ms=%s",
                self.name,
                self._proc.pid,
                ready.get("load_ms"),
            )
            return ready

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one JSON request and return the JSON response."""
        with self._lock:
            if not self.is_running:
                self._spawn()
                ready = self._read_line_locked(self.startup_timeout_sec)
                if ready.get("event") != "ready":
                    self._terminate_locked()
                    raise ModelLoadError(
                        f"{self.name} worker failed to restart: {ready!r}"
                    )

            request_id = payload.setdefault("id", uuid.uuid4().hex)
            self._write_line_locked(payload)
            response = self._read_line_locked(self.request_timeout_sec)
            if response.get("id") not in (None, request_id):
                raise ModelInferenceError(
                    f"{self.name} worker response id mismatch"
                )
            if self._proc is not None and self._proc.poll() is not None:
                raise WorkerCrashedError(f"{self.name} worker exited during request")
            if not response.get("ok", False):
                message = str(response.get("error") or "worker request failed")
                raise ModelInferenceError(message)
            return response

    def shutdown(self) -> None:
        with self._lock:
            if not self.is_running:
                self._proc = None
                return
            try:
                self._write_line_locked({"cmd": "shutdown", "id": "shutdown"})
                self._read_line_locked(30.0)
            except Exception:
                logger.warning("%s worker shutdown handshake failed", self.name)
            self._terminate_locked()

    def _spawn(self) -> None:
        if not self.python.is_file():
            raise ModelLoadError(f"{self.name} python not found: {self.python}")
        if not self.script.is_file():
            raise ModelLoadError(f"{self.name} worker script not found: {self.script}")

        env = os.environ.copy()
        env.setdefault("PYTHONPATH", str(project_root()))
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

        self._proc = subprocess.Popen(
            [str(self.python), str(self.script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(project_root()),
            env=env,
        )
        self._start_stderr_drain()
        logger.info("%s worker spawned pid=%s", self.name, self._proc.pid)

    def _start_stderr_drain(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return

        def _drain() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                self._stderr_tail = (self._stderr_tail + line)[-4000:]
                logger.debug("%s worker stderr: %s", self.name, line.rstrip())

        self._stderr_thread = threading.Thread(
            target=_drain,
            name=f"{self.name}-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

    def _write_line_locked(self, payload: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise WorkerCrashedError(f"{self.name} worker is not running")
        line = json.dumps(payload, separators=(",", ":"))
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()

    def _read_line_locked(self, timeout_sec: float) -> dict[str, Any]:
        if self._proc is None or self._proc.stdout is None:
            raise WorkerCrashedError(f"{self.name} worker is not running")

        deadline = time.monotonic() + timeout_sec
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._terminate_locked()
                raise WorkerTimeoutError(
                    f"{self.name} worker timed out after {timeout_sec:.0f}s"
                )
            ready, _, _ = select.select([self._proc.stdout], [], [], min(remaining, 1.0))
            if not ready:
                if self._proc.poll() is not None:
                    code = self._proc.returncode
                    stderr = self._stderr_tail[-500:]
                    self._proc = None
                    raise WorkerCrashedError(
                        f"{self.name} worker exited (code {code}). {stderr}"
                    )
                continue
            line = self._proc.stdout.readline()
            if not line:
                if self._proc.poll() is not None:
                    self._proc = None
                    raise WorkerCrashedError(f"{self.name} worker closed stdout")
                continue
            line = line.strip()
            if not line.startswith("{"):
                continue
            if len(line.encode("utf-8")) > _MAX_JSON_LINE_BYTES:
                raise ModelInferenceError(f"{self.name} worker response too large")
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    def _terminate_locked(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


_clients: list[LineJsonWorkerClient] = []


def register_worker_client(client: LineJsonWorkerClient) -> LineJsonWorkerClient:
    _clients.append(client)
    return client


def shutdown_all_workers() -> None:
    for client in reversed(_clients):
        try:
            client.shutdown()
        except Exception:
            logger.exception("failed to shut down worker %s", client.name)


atexit.register(shutdown_all_workers)
