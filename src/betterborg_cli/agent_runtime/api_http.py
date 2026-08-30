"""Abortable urllib requests isolated behind a spawned process boundary."""

from __future__ import annotations

import contextlib
import http.client
import multiprocessing
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Protocol, TypeAlias

from betterborg_cli.agent_runtime.base import (
    CancellationDeliveryError,
    CancellationRegistration,
    CancellationRegistrationRejected,
    CancellationToken,
    ForceTarget,
)

_IPC_POLL_SECONDS = 0.05
_RESPONSE_CHUNK_SIZE = 64 * 1024


@dataclass(frozen=True, slots=True)
class UrlRequestSpec:
    """Serializable inputs for one standard-library URL request."""

    url: str
    method: str
    headers: Mapping[str, str]
    body: bytes | None
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not self.url:
            raise ValueError("request URL must not be empty")
        if not self.method:
            raise ValueError("request method must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("request timeout must be positive")
        if self.body is not None and not isinstance(self.body, bytes):
            raise TypeError("request body must be bytes or None")
        headers = dict(self.headers)
        if not all(
            isinstance(name, str) and isinstance(value, str)
            for name, value in headers.items()
        ):
            raise TypeError("request headers must contain strings")
        object.__setattr__(self, "headers", headers)


@dataclass(frozen=True, slots=True)
class UrlResponse:
    """Complete non-streaming response captured by the request worker."""

    status_code: int
    reason: str
    body: bytes


class UrlTransportError(RuntimeError):
    """A provider-neutral URL transport failure."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


_ResponseMessage: TypeAlias = tuple[str, int, str, bytes]
_ErrorMessage: TypeAlias = tuple[str, str, str]
_WorkerMessage: TypeAlias = _ResponseMessage | _ErrorMessage


class _ReadableResponse(Protocol):
    def read(self, size: int = -1) -> bytes: ...


class MultiprocessUrlRequest:
    """Run one urllib request in a cancellable, attempt-local child process."""

    def __init__(
        self,
        spec: UrlRequestSpec,
        cancel: CancellationToken | None,
    ) -> None:
        self.spec = spec
        self.cancel = cancel

    def run(self) -> UrlResponse:
        """Return a captured response or raise a typed transport failure."""
        if self.cancel is not None and self.cancel.is_set():
            raise UrlTransportError("cancelled", "URL request was cancelled")

        window = None
        process: BaseProcess | None = None
        registration: CancellationRegistration | None = None
        force_target: ForceTarget | None = None
        receiver: Connection | None = None
        sender: Connection | None = None
        completed_normally = False

        try:
            try:
                window = (
                    self.cancel.registration_window()
                    if self.cancel is not None
                    else None
                )
            except CancellationRegistrationRejected as error:
                raise UrlTransportError(
                    "cancelled", "URL request was cancelled"
                ) from error

            try:
                context = multiprocessing.get_context("spawn")
                receiver, sender = context.Pipe(duplex=False)
                process = context.Process(
                    target=_url_request_worker,
                    args=(self.spec, sender),
                    name="betterborg-url-request",
                    daemon=False,
                )
                process.start()
            except BaseException:
                if window is not None:
                    window.no_resource()
                raise

            if window is not None:
                window.resource_created()
            if process.pid is None:
                raise RuntimeError("request child has no process identity")
            force_target = ForceTarget(process.pid)
            sender.close()
            sender = None

            if window is not None:
                try:
                    registration = window.register(
                        lambda: _terminate_process(process),
                        lambda: _force_process(process),
                        force_target=force_target,
                    )
                except CancellationDeliveryError as error:
                    registration = error.registration
                    raise

            message = self._receive(process, receiver)
            completed_normally = True
            if self.cancel is not None and self.cancel.is_set():
                raise UrlTransportError("cancelled", "URL request was cancelled")
            if message[0] == "response":
                _, status_code, reason, body = message
                return UrlResponse(status_code, reason, body)
            _, kind, detail = message
            raise UrlTransportError(kind, detail)
        finally:
            if sender is not None:
                sender.close()
            if receiver is not None:
                receiver.close()
            if process is not None and process.pid is not None:
                deadline = (
                    self.cancel.force_deadline
                    if self.cancel is not None and self.cancel.is_set()
                    else None
                )
                _cleanup_process(
                    process,
                    terminate=not completed_normally,
                    deadline=deadline,
                )
                if registration is not None:
                    registration.unregister()
                elif window is not None and not window.is_settled:
                    if force_target is None:
                        force_target = ForceTarget(process.pid)
                    window.publish_cleaned_resource(
                        lambda: _terminate_process(process),
                        lambda: _force_process(process),
                        force_target=force_target,
                    )
                process.close()

    def _receive(
        self,
        process: BaseProcess,
        receiver: Connection,
    ) -> _WorkerMessage:
        while True:
            if self.cancel is not None and self.cancel.is_set():
                raise UrlTransportError("cancelled", "URL request was cancelled")
            if receiver.poll(_IPC_POLL_SECONDS):
                try:
                    message = receiver.recv()
                except EOFError as error:
                    if self.cancel is not None and self.cancel.is_set():
                        raise UrlTransportError(
                            "cancelled", "URL request was cancelled"
                        ) from error
                    raise UrlTransportError(
                        "worker", "URL request worker closed without a result"
                    ) from error
                if _valid_worker_message(message):
                    return message
                raise UrlTransportError(
                    "worker", "URL request worker returned an invalid result"
                )
            if not process.is_alive():
                if receiver.poll():
                    continue
                if self.cancel is not None and self.cancel.is_set():
                    raise UrlTransportError(
                        "cancelled", "URL request was cancelled"
                    )
                raise UrlTransportError(
                    "worker",
                    f"URL request worker exited with code {process.exitcode}",
                )


def _url_request_worker(spec: UrlRequestSpec, sender: Connection) -> None:
    """Execute urllib entirely in the spawned child and send one result."""
    try:
        request = urllib.request.Request(
            spec.url,
            data=spec.body,
            headers=dict(spec.headers),
            method=spec.method,
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=spec.timeout_seconds,
            ) as response:
                body = _read_body(response)
                message: _WorkerMessage = (
                    "response",
                    int(response.status),
                    str(response.reason),
                    body,
                )
        except urllib.error.HTTPError as error:
            try:
                body = _read_body(error)
            except (OSError, http.client.HTTPException) as body_error:
                message = ("error", "response", str(body_error))
            else:
                message = (
                    "response",
                    int(error.code),
                    str(error.reason),
                    body,
                )
            finally:
                error.close()
        except urllib.error.URLError as error:
            message = ("error", "network", str(error.reason))
        except TimeoutError as error:
            message = ("error", "timeout", str(error))
        except (OSError, http.client.HTTPException) as error:
            message = ("error", "response", str(error))
        sender.send(message)
    except BaseException as error:
        with contextlib.suppress(BrokenPipeError, EOFError, OSError):
            sender.send(("error", "worker", str(error)))
    finally:
        sender.close()


def _read_body(response: _ReadableResponse) -> bytes:
    chunks: list[bytes] = []
    while chunk := response.read(_RESPONSE_CHUNK_SIZE):
        chunks.append(chunk)
    return b"".join(chunks)


def _valid_worker_message(message: object) -> bool:
    if not isinstance(message, tuple) or not message:
        return False
    if message[0] == "response":
        return (
            len(message) == 4
            and isinstance(message[1], int)
            and isinstance(message[2], str)
            and isinstance(message[3], bytes)
        )
    if message[0] == "error":
        return (
            len(message) == 3
            and isinstance(message[1], str)
            and isinstance(message[2], str)
        )
    return False


def _terminate_process(process: BaseProcess) -> None:
    with contextlib.suppress(ProcessLookupError, OSError, ValueError):
        if process.is_alive():
            process.terminate()


def _kill_process(process: BaseProcess) -> None:
    with contextlib.suppress(ProcessLookupError, OSError, ValueError):
        if process.is_alive():
            process.kill()


def _force_process(process: BaseProcess) -> None:
    """Kill and reap a request child before force delivery is complete."""
    _kill_process(process)
    process.join(CancellationToken.DEFAULT_GRACE_SECONDS)
    if process.is_alive():
        raise TimeoutError(f"URL request worker {process.pid} could not be joined")


def _cleanup_process(
    process: BaseProcess,
    *,
    terminate: bool,
    deadline: float | None,
) -> None:
    if terminate:
        _terminate_process(process)
    cleanup_deadline = (
        time.monotonic() + CancellationToken.DEFAULT_GRACE_SECONDS
        if deadline is None
        else deadline
    )
    while process.is_alive():
        remaining = cleanup_deadline - time.monotonic()
        if remaining <= 0:
            break
        process.join(min(_IPC_POLL_SECONDS, remaining))
    if process.is_alive():
        _kill_process(process)
    process.join(CancellationToken.DEFAULT_GRACE_SECONDS)
    if process.is_alive():
        raise TimeoutError(f"URL request worker {process.pid} could not be joined")
