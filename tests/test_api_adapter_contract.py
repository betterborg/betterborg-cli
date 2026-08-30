"""Provider-neutral behavioral contract for contained API adapters."""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest
from conftest import RealProcessHarness
from test_adapter_harness import (
    API_ADAPTER_HARNESSES,
    ApiAdapterHarness,
    BlockingTcpServer,
    FakeApiTransport,
    FakeUrlRequestFactory,
    LocalHttpServer,
    LocalTlsServer,
)

import betterborg_cli.agent_runtime.api_http as api_http
from betterborg_cli.agent_runtime import (
    AgentActivity,
    AgentActivityKind,
    AgentStatus,
    AgentUsage,
    ApiAgentRole,
    BillingMode,
    CancellationToken,
    MultiprocessUrlRequest,
    UrllibAnthropicTransport,
    UrllibOpenAITransport,
    UrlRequestSpec,
    UrlResponse,
    UrlTransportError,
)

_API_TOOL_ACTIVITY_CASES = (
    (
        "read_file",
        {"path": "version.txt"},
        AgentActivity(AgentActivityKind.READING, "version.txt"),
    ),
    (
        "search_text",
        {"query": "activity needle"},
        AgentActivity(AgentActivityKind.SEARCHING, "activity needle"),
    ),
    (
        "list_files",
        {},
        AgentActivity(AgentActivityKind.SEARCHING, "."),
    ),
    (
        "run_command",
        {"argv": [sys.executable, "-c", "raise SystemExit(0)"]},
        AgentActivity(
            AgentActivityKind.COMMAND,
            f"{sys.executable} -c 'raise SystemExit(0)'",
        ),
    ),
    (
        "apply_patch",
        {
            "patch": (
                "*** Begin Patch\n"
                "*** Add File: activity.txt\n"
                "+translated\n"
                "*** End Patch"
            )
        },
        AgentActivity(AgentActivityKind.WRITING, "activity.txt"),
    ),
)


@pytest.fixture(params=API_ADAPTER_HARNESSES, ids=lambda harness: harness.provider)
def harness(request: pytest.FixtureRequest) -> ApiAdapterHarness:
    return request.param


@pytest.mark.parametrize("failure_point", ["pipe", "process"])
def test_url_request_settles_window_when_prestart_setup_fails(
    failure_point: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connections = []

    class StubConnection:
        closed = False

        def close(self) -> None:
            self.closed = True

    class FailingContext:
        def Pipe(self, *, duplex: bool):
            assert duplex is False
            if failure_point == "pipe":
                raise OSError("injected pipe failure")
            receiver = StubConnection()
            sender = StubConnection()
            connections.extend((receiver, sender))
            return receiver, sender

        def Process(self, **_kwargs):
            raise OSError("injected process failure")

    monkeypatch.setattr(
        api_http.multiprocessing,
        "get_context",
        lambda method: FailingContext(),
    )
    cancel = CancellationToken()

    with pytest.raises(OSError, match=f"injected {failure_point} failure"):
        MultiprocessUrlRequest(
            UrlRequestSpec("http://127.0.0.1/", "GET", {}, None),
            cancel,
        ).run()

    assert cancel.active_windows == ()
    assert all(connection.closed for connection in connections)


def test_url_request_abort_and_force_are_idempotent_before_execute() -> None:
    request = MultiprocessUrlRequest(
        UrlRequestSpec("http://127.0.0.1/", "GET", {}, None),
        CancellationToken(),
    )

    request.abort()
    request.abort()
    request.force()
    request.force()

    with pytest.raises(UrlTransportError, match="cancelled") as captured:
        request.execute()

    assert captured.value.kind == "cancelled"


def test_multiprocess_url_request_preserves_http_behavior() -> None:
    def respond(request):
        if request.path == "/redirect":
            return 302, {"location": "/final"}, b""
        if request.path == "/error":
            return 429, {"content-type": "application/json"}, b'{"retry":true}'
        if request.path == "/final":
            return 200, {}, b"redirected"
        assert request.method == "POST"
        headers = {name.casefold(): value for name, value in request.headers.items()}
        assert headers["x-contract"] == "preserved"
        assert request.body == b"request-body"
        return 201, {"content-type": "application/octet-stream"}, b"response-body"

    with LocalHttpServer(respond) as server:
        response = MultiprocessUrlRequest(
            UrlRequestSpec(
                server.url("/request"),
                "POST",
                {"x-contract": "preserved"},
                b"request-body",
            ),
            CancellationToken(),
        ).run()
        redirected = MultiprocessUrlRequest(
            UrlRequestSpec(server.url("/redirect"), "GET", {}, None),
            CancellationToken(),
        ).run()
        error = MultiprocessUrlRequest(
            UrlRequestSpec(server.url("/error"), "GET", {}, None),
            CancellationToken(),
        ).run()

    assert response.status_code == 201
    assert response.reason == "Created"
    assert response.body == b"response-body"
    assert redirected.status_code == 200
    assert redirected.body == b"redirected"
    assert error.status_code == 429
    assert error.body == b'{"retry":true}'


def test_multiprocess_url_request_uses_default_proxy_opener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def respond(request):
        assert request.path == "http://api.invalid/provider"
        return 200, {}, b"proxied"

    with LocalHttpServer(respond) as proxy:
        monkeypatch.setenv("http_proxy", proxy.url())
        monkeypatch.delenv("no_proxy", raising=False)
        monkeypatch.delenv("NO_PROXY", raising=False)
        response = MultiprocessUrlRequest(
            UrlRequestSpec("http://api.invalid/provider", "GET", {}, None),
            CancellationToken(),
        ).run()

    assert response.body == b"proxied"
    assert len(proxy.requests) == 1


def test_multiprocess_url_request_preserves_default_tls_trust(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")

    with LocalTlsServer(
        tmp_path,
        lambda _request: (200, {}, b"trusted"),
    ) as server:
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)
        with pytest.raises(UrlTransportError) as untrusted:
            MultiprocessUrlRequest(
                UrlRequestSpec(server.url(), "GET", {}, None),
                CancellationToken(),
            ).run()

        monkeypatch.setenv("SSL_CERT_FILE", str(server.certificate_path))
        trusted = MultiprocessUrlRequest(
            UrlRequestSpec(server.url(), "GET", {}, None),
            CancellationToken(),
        ).run()

    assert untrusted.value.kind == "network"
    assert "CERTIFICATE_VERIFY_FAILED" in untrusted.value.message
    assert trusted.status_code == 200
    assert trusted.body == b"trusted"


@pytest.mark.parametrize("phase", ["headers", "body", "proxy", "tls"])
def test_multiprocess_url_request_cancels_blocked_network_phase(
    phase: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    cancel = CancellationToken()

    def respond(_request):
        started.set()
        if phase != "body":
            release.wait()
        return 200, {}, b"too late"

    if phase == "tls":
        server_context = BlockingTcpServer()
    elif phase == "body":
        server_context = LocalHttpServer(
            respond,
            body_started=threading.Event(),
            body_release=release,
        )
    else:
        server_context = LocalHttpServer(respond)
    with server_context as server:
        if phase == "tls":
            url = server.url
            marker = server.connected
        elif phase == "proxy":
            monkeypatch.setenv("http_proxy", server.url())
            monkeypatch.delenv("no_proxy", raising=False)
            monkeypatch.delenv("NO_PROXY", raising=False)
            url = "http://api.invalid/blocked"
            marker = started
        elif phase == "body":
            url = server.url("/blocked")
            assert server.body_started is not None
            marker = server.body_started
        else:
            url = server.url("/blocked")
            marker = started

        def cancel_when_blocked() -> None:
            assert marker.wait(2)
            cancel.cancel()

        canceller = threading.Thread(target=cancel_when_blocked)
        canceller.start()
        before = time.monotonic()
        try:
            with pytest.raises(UrlTransportError, match="cancelled") as captured:
                MultiprocessUrlRequest(
                    UrlRequestSpec(url, "GET", {}, None), cancel
                ).run()
        finally:
            release.set()
            if phase == "tls":
                server.release.set()
            canceller.join()

    assert time.monotonic() - before < 1.5
    assert captured.value.kind == "cancelled"
    assert cancel.active_windows == ()


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals required")
def test_url_request_cancels_blocked_dns_before_socket_within_deadline(
    real_process_harness: RealProcessHarness,
) -> None:
    process = real_process_harness.launch_blocked_url_wrapper(
        "http://127.0.0.1:9/",
        name="url-blocked-dns",
    )
    real_process_harness.wait_for_marker("url-blocked-dns.dns-gate")
    child_pid = int(
        real_process_harness.wait_for_marker("url-blocked-dns.request.pid")
    )

    started = time.monotonic()
    real_process_harness.signal(process, signal.SIGINT)

    assert real_process_harness.wait_for_exit(
        process,
        timeout=CancellationToken.DEFAULT_GRACE_SECONDS,
    ) == 130
    joined_at = float(
        real_process_harness.wait_for_marker("url-blocked-dns.request-joined")
    )
    assert joined_at - started <= CancellationToken.DEFAULT_GRACE_SECONDS
    assert (
        real_process_harness.wait_for_marker("url-blocked-dns.active-windows")
        == "0"
    )
    real_process_harness.assert_pid_absent(child_pid, timeout=0.1)


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals required")
def test_url_request_force_kills_and_joins_resistant_child(
    real_process_harness: RealProcessHarness,
) -> None:
    process = real_process_harness.launch_blocked_url_wrapper(
        "http://127.0.0.1:9/",
        name="url-resistant-dns",
        resistant=True,
    )
    real_process_harness.wait_for_marker("url-resistant-dns.dns-gate")
    child_pid = int(
        real_process_harness.wait_for_marker("url-resistant-dns.request.pid")
    )

    real_process_harness.signal(process, signal.SIGINT)
    cancelled_at = float(
        real_process_harness.wait_for_marker("url-resistant-dns.cancelled")
    )
    real_process_harness.signal(process, signal.SIGINT)

    assert real_process_harness.wait_for_exit(process, timeout=1) == 130
    killed_at = float(
        real_process_harness.wait_for_marker("url-resistant-dns.kill")
    )
    joined_at = float(
        real_process_harness.wait_for_marker("url-resistant-dns.force-joined")
    )
    assert killed_at >= cancelled_at
    assert joined_at >= killed_at
    real_process_harness.assert_pid_absent(child_pid, timeout=0.1)


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals required")
def test_url_request_joins_resistant_child_by_production_deadline(
    real_process_harness: RealProcessHarness,
) -> None:
    process = real_process_harness.launch_blocked_url_wrapper(
        "http://127.0.0.1:9/",
        name="url-deadline-dns",
        resistant=True,
    )
    real_process_harness.wait_for_marker("url-deadline-dns.dns-gate")
    child_pid = int(
        real_process_harness.wait_for_marker("url-deadline-dns.request.pid")
    )

    real_process_harness.signal(process, signal.SIGINT)
    cancelled_at = float(
        real_process_harness.wait_for_marker("url-deadline-dns.cancelled")
    )

    assert real_process_harness.wait_for_exit(process, timeout=1.5) == 130
    killed_at = float(
        real_process_harness.wait_for_marker("url-deadline-dns.kill")
    )
    joined_at = float(
        real_process_harness.wait_for_marker("url-deadline-dns.request-joined")
    )
    assert killed_at - cancelled_at <= CancellationToken.DEFAULT_GRACE_SECONDS + 0.1
    assert joined_at - cancelled_at <= CancellationToken.DEFAULT_GRACE_SECONDS + 0.1
    real_process_harness.assert_pid_absent(child_pid, timeout=0.1)


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals required")
def test_url_request_sigint_during_registration_joins_exact_child(
    real_process_harness: RealProcessHarness,
) -> None:
    release_response = threading.Event()

    def respond(_request):
        release_response.wait()
        return 200, {}, b"too late"

    with LocalHttpServer(respond) as server:
        process = real_process_harness.launch_url_registration_wrapper(
            server.url("/blocked"),
            name="url-late-signal",
        )
        real_process_harness.wait_for_marker("url-late-signal.registration-gate")
        child_pid = int(
            real_process_harness.wait_for_marker("url-late-signal.request.pid")
        )
        command_line_path = Path("/proc") / str(child_pid) / "cmdline"
        if command_line_path.exists():
            assert b"request-private-value" not in command_line_path.read_bytes()

        real_process_harness.signal(process, signal.SIGINT)
        real_process_harness.wait_for_marker("url-late-signal.cancelled")
        real_process_harness.release("release-url-late-signal")

        assert real_process_harness.wait_for_exit(process) == 130
        real_process_harness.assert_pid_absent(child_pid, timeout=1)
        release_response.set()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process cleanup required")
def test_url_request_registration_failure_joins_exact_child(
    real_process_harness: RealProcessHarness,
) -> None:
    release_response = threading.Event()

    def respond(_request):
        release_response.wait()
        return 200, {}, b"too late"

    with LocalHttpServer(respond) as server:
        process = real_process_harness.launch_url_registration_wrapper(
            server.url("/blocked"),
            name="url-registration-error",
            fail_registration=True,
        )
        real_process_harness.wait_for_marker(
            "url-registration-error.registration-gate"
        )
        child_pid = int(
            real_process_harness.wait_for_marker(
                "url-registration-error.request.pid"
            )
        )
        real_process_harness.release("release-url-registration-error")

        assert real_process_harness.wait_for_exit(process) == 73
        assert (
            real_process_harness.wait_for_marker("url-registration-error.error")
            == "injected registration failure"
        )
        assert (
            real_process_harness.wait_for_marker(
                "url-registration-error.active-windows"
            )
            == "0"
        )
        real_process_harness.assert_pid_absent(child_pid, timeout=1)
        release_response.set()


def test_structured_result_persists_usage_and_metadata(
    tmp_path: Path,
    harness: ApiAdapterHarness,
) -> None:
    transport = FakeApiTransport(
        [
            harness.response(
                [
                    harness.tool_call(
                        "submit_result",
                        {"status": "completed", "version": "1.2.3"},
                        call_id="submit",
                    )
                ],
                input_tokens=20,
                output_tokens=5,
                cache_read=7,
                cache_write=2,
            )
        ]
    )

    result = harness.adapter(
        ApiAgentRole.ANALYSIS,
        transport=transport,
    ).run(harness.spec(tmp_path))

    assert result.status == AgentStatus.COMPLETED
    assert result.payload == {"status": "completed", "version": "1.2.3"}
    assert result.provider == harness.provider
    assert result.model == harness.resolved_model
    assert result.billing_mode == BillingMode.API
    assert result.duration_seconds >= 0
    assert result.attempts == 1
    assert result.usage == AgentUsage(
        tokens_input=20,
        tokens_output=5,
        tokens_cache_read=7,
        tokens_cache_write=2,
        num_turns=1,
    )
    assert json.loads(result.result_path.read_text(encoding="utf-8")) == result.payload


@pytest.mark.parametrize(
    ("tool_name", "arguments", "expected"),
    _API_TOOL_ACTIVITY_CASES,
    ids=("read", "search", "list", "command", "patch"),
)
def test_contained_tools_emit_equal_provider_neutral_activity(
    tmp_path: Path,
    harness: ApiAdapterHarness,
    tool_name: str,
    arguments: dict[str, object],
    expected: AgentActivity,
) -> None:
    (tmp_path / "version.txt").write_text(
        "activity needle\n", encoding="utf-8"
    )
    activities: list[AgentActivity] = []
    transport = FakeApiTransport(
        [
            harness.response(
                [harness.tool_call(tool_name, arguments, call_id="activity")],
                response_id="activity_response",
            ),
            harness.response(
                [
                    harness.tool_call(
                        "submit_result",
                        {"status": "completed", "version": "activity"},
                        call_id="submit",
                    )
                ]
            ),
        ]
    )

    result = harness.adapter(
        ApiAgentRole.CODING,
        transport=transport,
        workspace_trusted=True,
    ).run(harness.spec(tmp_path, activity_sink=activities.append))

    assert result.status == AgentStatus.COMPLETED
    assert activities == [
        AgentActivity(AgentActivityKind.THINKING),
        expected,
        AgentActivity(AgentActivityKind.THINKING),
    ]


def test_submit_result_emits_no_tool_activity(
    tmp_path: Path,
    harness: ApiAdapterHarness,
) -> None:
    activities: list[AgentActivity] = []
    transport = FakeApiTransport(
        [
            harness.response(
                [
                    harness.tool_call(
                        "submit_result",
                        {"status": "completed", "version": "silent"},
                        call_id="submit",
                    )
                ]
            )
        ]
    )

    result = harness.adapter(
        ApiAgentRole.ANALYSIS,
        transport=transport,
    ).run(harness.spec(tmp_path, activity_sink=activities.append))

    assert result.status == AgentStatus.COMPLETED
    assert activities == [AgentActivity(AgentActivityKind.THINKING)]


@pytest.mark.parametrize("call_kind", ["malformed", "disallowed"])
def test_rejected_calls_emit_no_provider_payload_activity(
    tmp_path: Path,
    harness: ApiAdapterHarness,
    call_kind: str,
) -> None:
    provider_payload = "provider-payload-must-stay-hidden"
    if call_kind == "disallowed":
        call = harness.tool_call(provider_payload, {}, call_id="rejected")
    else:
        call = harness.tool_call("read_file", {}, call_id="rejected")
        if harness.provider == "anthropic":
            call["input"] = provider_payload
        else:
            call["arguments"] = provider_payload
    activities: list[AgentActivity] = []
    transport = FakeApiTransport(
        [
            harness.response([call], response_id="rejected_response"),
            harness.response(
                [
                    harness.tool_call(
                        "submit_result",
                        {"status": "completed", "version": "safe"},
                        call_id="submit",
                    )
                ]
            ),
        ]
    )

    result = harness.adapter(
        ApiAgentRole.ANALYSIS,
        transport=transport,
    ).run(harness.spec(tmp_path, activity_sink=activities.append))

    assert result.status == AgentStatus.COMPLETED
    assert activities == [AgentActivity(AgentActivityKind.THINKING)] * 2
    assert provider_payload not in repr(activities)


def test_api_tool_activity_detail_is_credential_redacted(
    tmp_path: Path,
    harness: ApiAdapterHarness,
) -> None:
    credential = harness.credential
    activities: list[AgentActivity] = []
    transport = FakeApiTransport(
        [
            harness.response(
                [
                    harness.tool_call(
                        "read_file",
                        {"path": credential},
                        call_id="credential-path",
                    )
                ],
                response_id="credential_response",
            ),
            harness.response(
                [
                    harness.tool_call(
                        "submit_result",
                        {"status": "completed", "version": "redacted"},
                        call_id="submit",
                    )
                ]
            ),
        ]
    )

    result = harness.adapter(
        ApiAgentRole.ANALYSIS,
        api_key=credential,
        transport=transport,
    ).run(harness.spec(tmp_path, activity_sink=activities.append))

    assert result.status == AgentStatus.COMPLETED
    assert activities == [
        AgentActivity(AgentActivityKind.THINKING),
        AgentActivity(AgentActivityKind.READING, "[REDACTED]"),
        AgentActivity(AgentActivityKind.THINKING),
    ]


def test_standard_transport_rejects_malformed_response_body(
    tmp_path: Path,
    harness: ApiAdapterHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = FakeUrlRequestFactory([UrlResponse(200, "OK", b"{not-json")])
    module = f"betterborg_cli.agent_runtime.{harness.provider}"
    monkeypatch.setattr(f"{module}.MultiprocessUrlRequest", requests)
    transport = (
        UrllibAnthropicTransport()
        if harness.provider == "anthropic"
        else UrllibOpenAITransport()
    )

    result = harness.adapter(
        ApiAgentRole.ANALYSIS,
        transport=transport,
    ).run(harness.spec(tmp_path))

    assert result.status == AgentStatus.FAILED
    assert result.error == (
        f"{'Anthropic' if harness.provider == 'anthropic' else 'OpenAI'} "
        "returned malformed JSON"
    )
    assert len(requests.specs) == 1


def test_standard_transport_cancellation_joins_request_before_return(
    tmp_path: Path,
    harness: ApiAdapterHarness,
) -> None:
    request_started = threading.Event()
    release_response = threading.Event()
    cancel = CancellationToken()

    def respond(_request):
        request_started.set()
        release_response.wait()
        return 200, {}, b"too late"

    with LocalHttpServer(respond) as server:
        transport = (
            UrllibAnthropicTransport(server.url("/messages"))
            if harness.provider == "anthropic"
            else UrllibOpenAITransport(server.url("/responses"))
        )
        adapter = harness.adapter(
            ApiAgentRole.ANALYSIS,
            transport=transport,
        )

        def cancel_when_started() -> None:
            assert request_started.wait(2)
            cancel.cancel()

        canceller = threading.Thread(target=cancel_when_started)
        canceller.start()
        try:
            result = adapter.run(harness.spec(tmp_path), cancel=cancel)
        finally:
            release_response.set()
            canceller.join()

    assert result.status == AgentStatus.CANCELLED
    assert cancel.active_windows == ()
    assert not (tmp_path / "result.json").exists()


def test_execution_role_advertises_command_only_after_trust(
    tmp_path: Path,
    harness: ApiAdapterHarness,
) -> None:
    response = harness.response(
        [
            harness.tool_call(
                "submit_result",
                {"status": "completed", "version": "one"},
                call_id="submit",
            )
        ]
    )
    untrusted_transport = FakeApiTransport([response])
    trusted_transport = FakeApiTransport([response])

    harness.adapter(
        ApiAgentRole.CODING,
        transport=untrusted_transport,
    ).run(harness.spec(tmp_path))
    harness.adapter(
        ApiAgentRole.CODING,
        transport=trusted_transport,
        workspace_trusted=True,
    ).run(harness.spec(tmp_path))

    untrusted_names = {
        tool["name"] for tool in untrusted_transport.payloads[0]["tools"]
    }
    trusted_names = {
        tool["name"] for tool in trusted_transport.payloads[0]["tools"]
    }
    assert "run_command" not in untrusted_names
    assert "run_command" in trusted_names


def test_execution_role_commands_receive_run_environment(
    tmp_path: Path,
    harness: ApiAdapterHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variable = "BETTERBORG_API_AGENT_ENV_TEST"
    monkeypatch.setenv(variable, "ambient-value")
    transport = FakeApiTransport(
        [
            harness.response(
                [
                    harness.tool_call(
                        "run_command",
                        {
                            "argv": [
                                sys.executable,
                                "-c",
                                (
                                    "import os; "
                                    f"print(os.environ[{variable!r}])"
                                ),
                            ]
                        },
                        call_id="command",
                    )
                ]
            ),
            harness.response(
                [
                    harness.tool_call(
                        "submit_result",
                        {"status": "completed", "version": "environment"},
                        call_id="submit",
                    )
                ]
            ),
        ]
    )

    result = harness.adapter(
        ApiAgentRole.CODING,
        transport=transport,
        workspace_trusted=True,
    ).run(harness.spec(tmp_path, env={variable: "spec-value"}))

    assert result.status == AgentStatus.COMPLETED
    tool_output = json.loads(harness.extract_tool_output(transport.payloads[1]))
    assert tool_output["returncode"] == 0
    assert tool_output["stdout"] == "spec-value\n"


def test_cancellation_after_in_flight_response_wins(
    tmp_path: Path,
    harness: ApiAdapterHarness,
) -> None:
    def cancel_during_request(cancel: CancellationToken | None):
        assert cancel is not None
        cancel.cancel()
        return harness.response(
            [
                harness.tool_call(
                    "submit_result",
                    {"status": "completed", "version": "ignored"},
                    call_id="submit",
                )
            ]
        )

    cancel = CancellationToken()
    result = harness.adapter(
        ApiAgentRole.ANALYSIS,
        transport=FakeApiTransport([cancel_during_request]),
    ).run(harness.spec(tmp_path), cancel=cancel)

    assert result.status == AgentStatus.CANCELLED
    assert result.resumable
    assert result.attempts == 1
    assert not (tmp_path / "result.json").exists()


def test_cancellation_interrupts_blocked_transport(
    tmp_path: Path,
    harness: ApiAdapterHarness,
) -> None:
    started = threading.Event()
    finished = threading.Event()

    def block_request(request_cancel: CancellationToken | None):
        assert request_cancel is not None
        started.set()
        assert request_cancel.wait(1)
        finished.set()
        return harness.response([])

    cancel = CancellationToken()
    transport = FakeApiTransport([block_request])
    adapter = harness.adapter(
        ApiAgentRole.ANALYSIS,
        transport=transport,
    )

    def cancel_when_started() -> None:
        assert started.wait(1)
        cancel.cancel()

    canceller = threading.Thread(target=cancel_when_started)
    canceller.start()
    before = time.monotonic()
    result = adapter.run(harness.spec(tmp_path), cancel=cancel)
    canceller.join()

    assert time.monotonic() - before < 1
    assert result.status == AgentStatus.CANCELLED
    assert result.resumable
    assert finished.is_set()
    assert transport.requests[0].abort_calls >= 1


def test_forced_cancellation_reaches_owned_request_handle(
    tmp_path: Path,
    harness: ApiAdapterHarness,
) -> None:
    started = threading.Event()
    cancel = CancellationToken()

    def block_request(request_cancel: CancellationToken | None):
        assert request_cancel is cancel
        started.set()
        assert request_cancel.wait_for_force(1)
        return harness.response([])

    transport = FakeApiTransport([block_request])

    def force_when_started() -> None:
        assert started.wait(1)
        cancel.force()

    force_worker = threading.Thread(target=force_when_started)
    force_worker.start()
    result = harness.adapter(
        ApiAgentRole.ANALYSIS,
        transport=transport,
    ).run(harness.spec(tmp_path), cancel=cancel)
    force_worker.join()

    assert result.status == AgentStatus.CANCELLED
    assert transport.requests[0].abort_calls >= 1
    assert transport.requests[0].force_calls == 1
    assert cancel.force_targets == ()


def test_cancellation_terminates_in_flight_command(
    tmp_path: Path,
    harness: ApiAdapterHarness,
) -> None:
    started_path = tmp_path / "command-started"
    cancel = CancellationToken()
    transport = FakeApiTransport(
        [
            harness.response(
                [
                    harness.tool_call(
                        "run_command",
                        {
                            "argv": [
                                sys.executable,
                                "-c",
                                (
                                    "from pathlib import Path; import time; "
                                    "Path('command-started').write_text('yes'); "
                                    "time.sleep(3)"
                                ),
                            ]
                        },
                        call_id="command",
                    )
                ]
            )
        ]
    )
    adapter = harness.adapter(
        ApiAgentRole.CODING,
        transport=transport,
        workspace_trusted=True,
    )

    def cancel_when_started() -> None:
        deadline = time.monotonic() + 1
        while not started_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert started_path.exists()
        cancel.cancel()

    canceller = threading.Thread(target=cancel_when_started)
    canceller.start()
    before = time.monotonic()
    result = adapter.run(harness.spec(tmp_path), cancel=cancel)
    canceller.join()

    assert time.monotonic() - before < 2
    assert result.status == AgentStatus.CANCELLED
    assert result.resumable
    assert len(transport.payloads) == 1


def test_transient_failure_retries_same_turn_and_then_completes(
    tmp_path: Path,
    harness: ApiAdapterHarness,
) -> None:
    transport = FakeApiTransport(
        [
            harness.transient_error(),
            harness.response(
                [
                    harness.tool_call(
                        "submit_result",
                        {"status": "completed", "version": "retry"},
                        call_id="submit",
                    )
                ]
            ),
        ]
    )
    adapter = harness.adapter(
        ApiAgentRole.ANALYSIS,
        transport=transport,
        transient_backoff_seconds=0,
    )

    result = adapter.run(harness.spec(tmp_path))

    assert result.status == AgentStatus.COMPLETED
    assert result.attempts == 2
    assert transport.payloads[0] == transport.payloads[1]
    assert len(transport.requests) == 2
    assert transport.requests[0] is not transport.requests[1]


def test_transient_exhaustion_is_cancelled_and_resumable(
    tmp_path: Path,
    harness: ApiAdapterHarness,
) -> None:
    transport = FakeApiTransport(
        [harness.transient_error("rate limited") for _attempt in range(2)]
    )
    adapter = harness.adapter(
        ApiAgentRole.ANALYSIS,
        transport=transport,
        transient_backoff_seconds=0,
        transient_max_attempts=2,
    )

    result = adapter.run(harness.spec(tmp_path, resume_token="operation-123"))

    assert result.status == AgentStatus.CANCELLED
    assert result.retryable and result.resumable
    assert result.resume_token == "operation-123"
    assert result.attempts == 2
    assert "transient retry exhausted" in (result.error or "")


def test_schema_invalid_submission_fails_without_persisting_result(
    tmp_path: Path,
    harness: ApiAdapterHarness,
) -> None:
    transport = FakeApiTransport(
        [
            harness.response(
                [
                    harness.tool_call(
                        "submit_result",
                        {"status": "completed"},
                        call_id="submit",
                    )
                ]
            )
        ]
    )

    result = harness.adapter(
        ApiAgentRole.ANALYSIS,
        transport=transport,
    ).run(harness.spec(tmp_path))

    assert result.status == AgentStatus.FAILED
    assert "missing required property 'version'" in (result.error or "")
    assert not (tmp_path / "result.json").exists()


def test_credentials_are_redacted_from_errors_and_logs(
    tmp_path: Path,
    harness: ApiAdapterHarness,
) -> None:
    credential = harness.credential
    transport = FakeApiTransport(
        [harness.api_error(f"Authorization: {credential} rejected")]
    )
    adapter = harness.adapter(
        ApiAgentRole.ANALYSIS,
        api_key=credential,
        transport=transport,
    )

    result = adapter.run(harness.spec(tmp_path))

    log = (tmp_path / f"{harness.provider}.jsonl").read_text(encoding="utf-8")
    assert result.status == AgentStatus.FAILED
    assert credential not in (result.error or "")
    assert credential not in log
    assert "[REDACTED]" in (result.error or "")
    assert "[REDACTED]" in log


def test_adapter_requires_an_injected_credential(
    tmp_path: Path,
    harness: ApiAdapterHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variable = (
        "ANTHROPIC_API_KEY"
        if harness.provider == "anthropic"
        else "OPENAI_API_KEY"
    )
    monkeypatch.setenv(variable, "ambient-secret")
    transport = FakeApiTransport(
        [
            harness.response(
                [
                    harness.tool_call(
                        "submit_result",
                        {"status": "completed", "version": "unexpected"},
                        call_id="submit",
                    )
                ]
            )
        ]
    )

    result = harness.adapter(
        ApiAgentRole.ANALYSIS,
        api_key=None,
        transport=transport,
    ).run(harness.spec(tmp_path, env={variable: "spec-secret"}))

    assert result.status == AgentStatus.FAILED
    provider_name = "Anthropic" if harness.provider == "anthropic" else "OpenAI"
    assert result.error == f"{provider_name} API credential is not configured"
    assert transport.payloads == []


def test_credentials_are_redacted_from_tools_and_completed_payload(
    tmp_path: Path,
    harness: ApiAdapterHarness,
) -> None:
    credential = harness.credential
    (tmp_path / "credential.txt").write_text(credential, encoding="utf-8")
    transport = FakeApiTransport(
        [
            harness.response(
                [
                    harness.tool_call(
                        "read_file",
                        {"path": "credential.txt"},
                        call_id="read",
                    )
                ],
                response_id="read_response",
            ),
            harness.response(
                [
                    harness.tool_call(
                        "submit_result",
                        {"status": "completed", "version": credential},
                        call_id="submit",
                    )
                ]
            ),
        ]
    )

    result = harness.adapter(
        ApiAgentRole.ANALYSIS,
        api_key=credential,
        transport=transport,
    ).run(harness.spec(tmp_path))

    tool_output = harness.extract_tool_output(transport.payloads[1])
    persisted = (tmp_path / "result.json").read_text(encoding="utf-8")
    assert result.status == AgentStatus.COMPLETED
    assert json.loads(tool_output) == {"content": "[REDACTED]"}
    assert result.payload == {"status": "completed", "version": "[REDACTED]"}
    assert credential not in persisted
    assert json.loads(persisted) == result.payload
