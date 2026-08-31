"""Shared provider API adapter contract harness."""

from __future__ import annotations

import contextlib
import json
import socket
import ssl
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from betterborg_cli.agent_runtime import (
    AbortableApiRequest,
    AgentActivity,
    AgentAdapter,
    AgentRunSpec,
    AnthropicAdapter,
    AnthropicApiError,
    ApiAgentRole,
    CancellationToken,
    OpenAIAdapter,
    OpenAIApiError,
    UrlRequestSpec,
    UrlResponse,
)
from betterborg_cli.host_execution.service import _ExecutionActivityBinding

Response = Mapping[str, Any]
QueuedResponse = Response | Exception | Callable[[CancellationToken | None], Response]
Provider = Literal["anthropic", "openai"]
RunProvider = Literal["anthropic", "openai", "claude", "codex"]


def redacting_execution_activity_sink(
    secret: str, received: list[AgentActivity]
) -> Callable[[AgentActivity], None]:
    """Return the execution-owned sink shape used by provider tests."""
    binding = _ExecutionActivityBinding(
        (secret,), lambda _task_id, activity: received.append(activity)
    )
    task_id = uuid4()
    return lambda activity: binding.emit(task_id, activity)


@dataclass(frozen=True, slots=True)
class HttpRequestRecord:
    method: str
    path: str
    headers: Mapping[str, str]
    body: bytes


HttpResponder = Callable[
    [HttpRequestRecord],
    tuple[int, Mapping[str, str], bytes],
]


@dataclass
class LocalHttpServer:
    """Threaded local HTTP endpoint for spawned urllib contract tests."""

    responder: HttpResponder
    requests: list[HttpRequestRecord] = field(default_factory=list)
    body_started: threading.Event | None = None
    body_release: threading.Event | None = None
    tls_context: ssl.SSLContext | None = field(default=None, repr=False)
    _server: ThreadingHTTPServer = field(init=False, repr=False)
    _thread: threading.Thread = field(init=False, repr=False)

    def __enter__(self) -> LocalHttpServer:
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:
                self._respond()

            def do_POST(self) -> None:
                self._respond()

            def _respond(self) -> None:
                length = int(self.headers.get("content-length", "0"))
                record = HttpRequestRecord(
                    self.command,
                    self.path,
                    dict(self.headers.items()),
                    self.rfile.read(length),
                )
                fixture.requests.append(record)
                status, headers, body = fixture.responder(record)
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    if fixture.body_started is not None and body:
                        self.wfile.write(body[:1])
                        self.wfile.flush()
                        fixture.body_started.set()
                        if fixture.body_release is not None:
                            fixture.body_release.wait()
                        body = body[1:]
                    self.wfile.write(body)

            def log_message(self, _format: str, *_args: object) -> None:
                return None

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        if self.tls_context is not None:
            self._server.socket = self.tls_context.wrap_socket(
                self._server.socket,
                server_side=True,
            )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="betterborg-test-http",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def url(self, path: str = "/") -> str:
        host, port = self._server.server_address
        scheme = "https" if self.tls_context is not None else "http"
        return f"{scheme}://{host}:{port}{path}"


@dataclass
class LocalTlsServer:
    """Local HTTPS endpoint with an explicit self-signed trust anchor."""

    root: Path
    responder: HttpResponder
    certificate_path: Path = field(init=False)
    _server: LocalHttpServer = field(init=False, repr=False)

    def __enter__(self) -> LocalTlsServer:
        self.certificate_path = self.root / "localhost-ca.pem"
        private_key_path = self.root / "localhost-key.pem"
        _write_localhost_certificate(self.certificate_path, private_key_path)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(self.certificate_path, private_key_path)
        self._server = LocalHttpServer(self.responder, tls_context=context)
        self._server.__enter__()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._server.__exit__(*exc_info)

    @property
    def requests(self) -> list[HttpRequestRecord]:
        return self._server.requests

    def url(self, path: str = "/") -> str:
        return self._server.url(path)


def _write_localhost_certificate(certificate_path: Path, key_path: Path) -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.now(UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ip_address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        )
        .sign(private_key, hashes.SHA256())
    )
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )


@dataclass
class BlockingTcpServer:
    """Accept one connection and hold it for TLS-handshake cancellation."""

    connected: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    _socket: socket.socket = field(init=False, repr=False)
    _thread: threading.Thread = field(init=False, repr=False)

    def __enter__(self) -> BlockingTcpServer:
        self._socket = socket.socket()
        self._socket.bind(("127.0.0.1", 0))
        self._socket.listen()
        self._socket.settimeout(0.1)

        def accept() -> None:
            while not self.release.is_set():
                try:
                    connection, _address = self._socket.accept()
                except TimeoutError:
                    continue
                break
            else:
                return
            with connection:
                self.connected.set()
                self.release.wait()

        self._thread = threading.Thread(
            target=accept,
            name="betterborg-test-tcp",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.release.set()
        self._socket.close()
        self._thread.join()

    @property
    def url(self) -> str:
        host, port = self._socket.getsockname()
        return f"https://{host}:{port}/"


def native_event_stream(*events: str | Mapping[str, Any]) -> str:
    """Serialize native provider events as a JSONL stream."""
    return "\n".join(
        event if isinstance(event, str) else json.dumps(event) for event in events
    )

def write_native_output(
    log_path: Path,
    text: str,
    on_line: Callable[[str], None] | None,
) -> None:
    """Persist native output and present the same text to its line observer."""
    log_path.write_text(text, encoding="utf-8")
    if on_line is not None:
        for line in text.splitlines(keepends=True):
            on_line(line)


@dataclass
class FakeApiTransport:
    """Queue responses while recording JSON-safe copies of provider requests."""

    responses: list[QueuedResponse]
    payloads: list[dict[str, Any]] = field(default_factory=list)
    api_keys: list[str] = field(default_factory=list, repr=False)
    requests: list[FakeApiRequest] = field(default_factory=list)

    def create_message(
        self,
        payload: Mapping[str, Any],
        *,
        api_key: str,
        cancel: CancellationToken | None = None,
    ) -> AbortableApiRequest:
        return self._create(payload, api_key=api_key, cancel=cancel)

    def create_response(
        self,
        payload: Mapping[str, Any],
        *,
        api_key: str,
        cancel: CancellationToken | None = None,
    ) -> AbortableApiRequest:
        return self._create(payload, api_key=api_key, cancel=cancel)

    def _create(
        self,
        payload: Mapping[str, Any],
        *,
        api_key: str,
        cancel: CancellationToken | None,
    ) -> FakeApiRequest:
        self.payloads.append(json.loads(json.dumps(payload)))
        self.api_keys.append(api_key)
        response = self.responses.pop(0)
        request = FakeApiRequest(response, cancel)
        self.requests.append(request)
        return request


@dataclass
class FakeApiRequest:
    """Attempt-local request handle for provider-neutral adapter tests."""

    response: QueuedResponse
    cancel: CancellationToken | None
    abort_calls: int = 0
    force_calls: int = 0
    executed: bool = False

    def execute(self) -> Mapping[str, Any]:
        self.executed = True
        response = self.response
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(self.cancel)
        return response

    def abort(self) -> None:
        self.abort_calls += 1

    def force(self) -> None:
        self.force_calls += 1


@dataclass
class FakeUrlRequestFactory:
    """Record URL envelopes and return attempt-local transport handles."""

    responses: list[UrlResponse | Exception]
    specs: list[UrlRequestSpec] = field(default_factory=list)
    cancels: list[CancellationToken | None] = field(default_factory=list)

    def __call__(
        self,
        spec: UrlRequestSpec,
        cancel: CancellationToken | None,
    ) -> FakeUrlRequest:
        self.specs.append(spec)
        self.cancels.append(cancel)
        return FakeUrlRequest(self.responses.pop(0))


@dataclass
class FakeUrlRequest:
    response: UrlResponse | Exception
    abort_calls: int = 0
    force_calls: int = 0

    def execute(self) -> UrlResponse:
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def abort(self) -> None:
        self.abort_calls += 1

    def force(self) -> None:
        self.force_calls += 1


class ChunkedHttpResponse:
    """Small urllib response double that can fail between body chunks."""

    def __init__(self, chunks: list[bytes | Exception]) -> None:
        self.chunks = iter(chunks)

    def __enter__(self) -> ChunkedHttpResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _size: int = -1) -> bytes:
        chunk = next(self.chunks)
        if isinstance(chunk, Exception):
            raise chunk
        return chunk

    def close(self) -> None:
        return None


def make_run_spec(
    tmp_path: Path,
    provider: RunProvider,
    **changes: Any,
) -> AgentRunSpec:
    """Build the shared structured-result run contract for one provider."""
    model = "gpt-test" if provider in {"openai", "codex"} else "claude-test"
    values: dict[str, Any] = {
        "system_prompt": "Inspect the repository and complete the task.",
        "user_prompt": "Read the version, then submit the result.",
        "schema": {
            "type": "object",
            "required": ["status", "version"],
            "properties": {
                "status": {"const": "completed"},
                "version": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "cwd": tmp_path,
        "model": model,
        "log_path": tmp_path / f"{provider}.jsonl",
        "result_path": tmp_path / "result.json",
    }
    if provider in {"claude", "codex"}:
        values["billing_mode"] = "subscription"
    values.update(changes)
    return AgentRunSpec(**values)


def anthropic_spec(tmp_path: Path, **changes: Any) -> AgentRunSpec:
    return make_run_spec(tmp_path, "anthropic", **changes)


def openai_spec(tmp_path: Path, **changes: Any) -> AgentRunSpec:
    return make_run_spec(tmp_path, "openai", **changes)


def claude_spec(tmp_path: Path, **changes: Any) -> AgentRunSpec:
    return make_run_spec(tmp_path, "claude", **changes)


def codex_spec(tmp_path: Path, **changes: Any) -> AgentRunSpec:
    return make_run_spec(tmp_path, "codex", **changes)


def anthropic_message(
    content: list[dict[str, Any]],
    *,
    model: str = "claude-test-20260801",
    input_tokens: int = 10,
    output_tokens: int = 4,
    cache_read: int = 0,
    cache_write: int = 0,
    stop_reason: str = "tool_use",
) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_write,
        },
    }


def openai_function_call(
    name: str,
    arguments: Mapping[str, Any],
    *,
    call_id: str,
) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
        "status": "completed",
    }


def openai_response(
    output: list[dict[str, Any]],
    *,
    response_id: str = "resp_test",
    model: str = "gpt-test-2026-08-01",
    input_tokens: int = 10,
    output_tokens: int = 4,
    cache_read: int = 0,
    cache_write: int = 0,
    status: str = "completed",
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "status": status,
        "model": model,
        "output": output,
        "usage": {
            # Responses reports an inclusive input total; the harness arguments
            # describe the provider-neutral, mutually exclusive buckets.
            "input_tokens": input_tokens + cache_read + cache_write,
            "output_tokens": output_tokens,
            "input_tokens_details": {
                "cached_tokens": cache_read,
                "cache_write_tokens": cache_write,
            },
        },
    }


@dataclass(frozen=True, slots=True)
class ApiAdapterHarness:
    """Translate provider-neutral contract cases to each provider wire shape."""

    provider: Provider

    @property
    def resolved_model(self) -> str:
        if self.provider == "anthropic":
            return "claude-test-20260801"
        return "gpt-test-2026-08-01"

    @property
    def credential(self) -> str:
        if self.provider == "anthropic":
            return "sk-ant-api03-contract-secret"
        return "sk-proj-contract-secret"

    def spec(self, tmp_path: Path, **changes: Any) -> AgentRunSpec:
        return make_run_spec(tmp_path, self.provider, **changes)

    def tool_call(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        call_id: str,
    ) -> dict[str, Any]:
        if self.provider == "anthropic":
            return {
                "type": "tool_use",
                "id": call_id,
                "name": name,
                "input": dict(arguments),
            }
        return openai_function_call(name, arguments, call_id=call_id)

    def response(
        self,
        calls: list[dict[str, Any]],
        *,
        response_id: str = "response",
        input_tokens: int = 10,
        output_tokens: int = 4,
        cache_read: int = 0,
        cache_write: int = 0,
    ) -> dict[str, Any]:
        if self.provider == "anthropic":
            return anthropic_message(
                calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read=cache_read,
                cache_write=cache_write,
            )
        return openai_response(
            calls,
            response_id=response_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read=cache_read,
            cache_write=cache_write,
        )

    def adapter(
        self,
        role: ApiAgentRole,
        *,
        transport: FakeApiTransport,
        api_key: str | None = "key",
        workspace_trusted: bool = False,
        **options: Any,
    ) -> AgentAdapter:
        adapter_type = (
            AnthropicAdapter if self.provider == "anthropic" else OpenAIAdapter
        )
        return adapter_type(
            role,
            api_key=api_key,
            workspace_trusted=workspace_trusted,
            transport=transport,
            **options,
        )

    def transient_error(self, message: str = "temporarily overloaded") -> Exception:
        if self.provider == "anthropic":
            return AnthropicApiError(
                message,
                status_code=529,
                error_type="overloaded_error",
            )
        return OpenAIApiError(
            message,
            status_code=503,
            error_type="server_error",
        )

    def api_error(self, message: str) -> Exception:
        if self.provider == "anthropic":
            return AnthropicApiError(message, status_code=401)
        return OpenAIApiError(message, status_code=401)

    def extract_tool_output(self, payload: Mapping[str, Any]) -> str:
        if self.provider == "anthropic":
            return payload["messages"][-1]["content"][0]["content"]
        return payload["input"][0]["output"]


API_ADAPTER_HARNESSES = (
    ApiAdapterHarness("anthropic"),
    ApiAdapterHarness("openai"),
)
