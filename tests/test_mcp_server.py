"""Protocol contracts for Betterborg's typed MCP workflow surface."""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import anyio
import pytest
from conftest import blocked_dns_url_request_worker
from mcp import types as mcp_types
from mcp.client.session import ClientSession
from mcp.server.session import ServerSession
from mcp.shared.exceptions import McpError
from mcp.shared.memory import create_connected_server_and_client_session
from test_adapter_harness import (
    LocalHttpServer,
    openai_function_call,
    openai_response,
)

from betterborg_cli import cli as cli_module
from betterborg_cli import mcp_server
from betterborg_cli.agent_runtime import (
    AgentRunSpec,
    AgentStatus,
    AgentUsage,
    ApiAgentRole,
    BillingMode,
    OpenAIAdapter,
    SelectedAgent,
    UrllibOpenAITransport,
    api_http,
)
from betterborg_cli.agent_runtime.mock import MockAdapter, MockResponse
from betterborg_cli.agent_runtime.process import run_captured
from betterborg_cli.host_execution import (
    HostExecutionResult,
    HostPreflightPlan,
    HostSchedulerConfig,
    HostSchedulerResult,
    HostTaskScheduler,
    ScheduledTaskContext,
)
from betterborg_cli.planning import TaskPublisher, build_plan_element_catalog
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.repository_config import AgentStage
from betterborg_cli.run_control import DEFAULT_FORCE_GRACE_SECONDS
from betterborg_cli.store import (
    AgentAttempt,
    Borg,
    BorgState,
    ExecutionEvent,
    ExecutionRun,
    ExecutionRunStatus,
    PlanApproval,
    PlanningAttempt,
    PlanningAttemptStatus,
    SqliteStore,
    TaskClaim,
    TaskRuntime,
    TaskRuntimeStatus,
)
from betterborg_cli.store.models import utcnow
from betterborg_cli.workspace_trust import TrustStore, WorkspaceIdentity


def _call_tool(
    name: str,
    arguments: dict | None = None,
    *,
    answers: tuple[str, ...] = (),
    approve: bool = True,
    elicitation: bool = True,
    requests: list | None = None,
):
    async def call():
        supplied = iter(answers)

        async def elicit(_context, params):
            if requests is not None:
                requests.append(params)
            properties = params.requestedSchema["properties"]
            if "approved" in properties:
                content = {"approved": approve}
            else:
                content = {"answer": next(supplied)}
            return mcp_types.ElicitResult(action="accept", content=content)

        async with create_connected_server_and_client_session(
            mcp_server.server,
            raise_exceptions=True,
            elicitation_callback=elicit if elicitation else None,
        ) as session:
            return await session.call_tool(name, arguments or {})

    return anyio.run(call)


def _list_tools():
    async def list_tools():
        async with create_connected_server_and_client_session(
            mcp_server.server,
            raise_exceptions=True,
        ) as session:
            return await session.list_tools()

    return anyio.run(list_tools).tools


def _list_resource_templates():
    async def list_resource_templates():
        async with create_connected_server_and_client_session(
            mcp_server.server,
            raise_exceptions=True,
        ) as session:
            return await session.list_resource_templates()

    return anyio.run(list_resource_templates).resourceTemplates


def _read_resource(uri: str) -> dict:
    async def read_resource():
        async with create_connected_server_and_client_session(
            mcp_server.server,
            raise_exceptions=True,
        ) as session:
            return await session.read_resource(uri)

    result = anyio.run(read_resource)
    assert len(result.contents) == 1
    content = result.contents[0]
    assert isinstance(content, mcp_types.TextResourceContents)
    return json.loads(content.text)


def _structured(result) -> dict:
    assert result.isError is False
    assert result.structuredContent is not None
    return result.structuredContent


def test_run_cancellable_forces_and_waits_for_worker_completion() -> None:
    started = threading.Event()
    worker_completed = threading.Event()
    order: list[str] = []
    cancellation_requested: list[float] = []

    def workflow(*, cancel):
        order.append("token-received")
        started.set()
        try:
            assert cancel.wait(timeout=2)
            order.append("cancelled")
            assert cancel.force_deadline is not None
            assert cancel.wait_for_force(timeout=2)
            order.append("forced")
        finally:
            order.append("worker-finally")
            worker_completed.set()
        if not cancel.is_set():
            order.append("forbidden-post-cancel-mutation")
        return "late-result"

    async def cancel_when_started(scope: anyio.CancelScope) -> None:
        await anyio.to_thread.run_sync(started.wait)
        cancellation_requested.append(time.monotonic())
        scope.cancel()

    async def run() -> None:
        with anyio.CancelScope() as scope:
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(cancel_when_started, scope)
                await mcp_server._run_cancellable(workflow)
        order.append("request-returned")

    anyio.run(run)

    assert worker_completed.is_set()
    assert time.monotonic() - cancellation_requested[0] >= 0.9
    assert order == [
        "token-received",
        "cancelled",
        "forced",
        "worker-finally",
        "request-returned",
    ]


def test_protocol_cancellation_waits_for_worker_completion(monkeypatch) -> None:
    worker_started = threading.Event()
    cancellation_received = threading.Event()
    release_worker = threading.Event()
    worker_completed = threading.Event()
    order: list[str] = []

    def workflow(_io, *, cancel) -> None:
        worker_started.set()
        try:
            assert cancel.wait(timeout=2)
            order.append("token-cancelled")
            cancellation_received.set()
            assert release_worker.wait(timeout=2)
        finally:
            order.append("worker-completed")
            worker_completed.set()

    async def wait_for_thread_event(event: threading.Event) -> None:
        while not event.is_set():
            await anyio.sleep(0.01)

    async def run() -> list[McpError]:
        call_finished = anyio.Event()
        errors: list[McpError] = []

        async def elicit(_context, _params):
            raise AssertionError("analyze must not elicit")

        async with create_connected_server_and_client_session(
            mcp_server.server,
            raise_exceptions=True,
            elicitation_callback=elicit,
        ) as session:
            request_id = session._request_id

            async def call() -> None:
                try:
                    await session.call_tool("analyze", {})
                except McpError as error:
                    errors.append(error)
                finally:
                    order.append("request-returned")
                    call_finished.set()

            async with anyio.create_task_group() as tasks:
                tasks.start_soon(call)
                await wait_for_thread_event(worker_started)
                await session.send_notification(
                    mcp_types.CancelledNotification(
                        params=mcp_types.CancelledNotificationParams(
                            requestId=request_id,
                            reason="test cancellation",
                        )
                    )
                )
                await wait_for_thread_event(cancellation_received)
                with anyio.move_on_after(0.05):
                    await call_finished.wait()
                assert call_finished.is_set() is False
                release_worker.set()
                with anyio.fail_after(2):
                    await call_finished.wait()

        return errors

    monkeypatch.setattr(mcp_server, "_analyze", workflow)
    try:
        errors = anyio.run(run)
    finally:
        release_worker.set()

    assert worker_completed.is_set()
    assert len(errors) == 1
    assert str(errors[0]) == "Request cancelled"
    assert order == [
        "token-cancelled",
        "worker-completed",
        "request-returned",
    ]


def test_run_cancellable_preserves_success_and_failure_results() -> None:
    def succeed(value: str, *, cancel) -> str:
        assert cancel.is_set() is False
        return value

    def fail(*, cancel) -> None:
        assert cancel.is_set() is False
        raise ValueError("worker failed")

    assert anyio.run(mcp_server._run_cancellable, succeed, "compatible") == (
        "compatible"
    )
    with pytest.raises(ValueError, match="worker failed"):
        anyio.run(mcp_server._run_cancellable, fail)


def test_run_cancellable_starts_worker_under_pending_cancellation() -> None:
    worker_finished = threading.Event()

    def workflow(*, cancel) -> None:
        try:
            cancel.wait(timeout=2)
        finally:
            worker_finished.set()

    async def run() -> None:
        with anyio.CancelScope() as scope:
            scope.cancel()
            await mcp_server._run_cancellable(workflow)

    anyio.run(run)

    assert worker_finished.is_set()


def test_run_cancellable_ignores_exhausted_default_thread_limiter() -> None:
    default_worker_started = threading.Event()
    release_default_worker = threading.Event()
    workflow_started = threading.Event()
    workflow_finished = threading.Event()
    rescue_started = threading.Event()
    request_finished_before_rescue: list[bool] = []

    def occupy_default_worker() -> None:
        default_worker_started.set()
        assert release_default_worker.wait(timeout=4)

    def workflow(*, cancel) -> None:
        workflow_started.set()
        try:
            assert cancel.wait(timeout=2)
            assert cancel.wait_for_force(timeout=2)
        finally:
            workflow_finished.set()

    async def wait_for_event(event: threading.Event) -> None:
        while not event.is_set():
            await anyio.sleep(0.01)

    async def rescue_default_limiter(limiter, original_tokens: int) -> None:
        await anyio.sleep(1.5)
        rescue_started.set()
        limiter.total_tokens = original_tokens

    async def cancel_when_started(scope: anyio.CancelScope) -> None:
        await wait_for_event(workflow_started)
        scope.cancel()

    async def run() -> None:
        limiter = anyio.to_thread.current_default_thread_limiter()
        original_tokens = limiter.total_tokens
        limiter.total_tokens = 1
        try:
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(anyio.to_thread.run_sync, occupy_default_worker)
                await wait_for_event(default_worker_started)
                tasks.start_soon(
                    rescue_default_limiter,
                    limiter,
                    original_tokens,
                )
                try:
                    with anyio.CancelScope() as scope:
                        async with anyio.create_task_group() as cancellation_tasks:
                            cancellation_tasks.start_soon(cancel_when_started, scope)
                            await mcp_server._run_cancellable(workflow)
                    request_finished_before_rescue.append(not rescue_started.is_set())
                finally:
                    release_default_worker.set()
        finally:
            limiter.total_tokens = original_tokens

    anyio.run(run)

    assert workflow_finished.is_set()
    assert request_finished_before_rescue == [True]


def test_run_cancellable_bounds_workflow_and_cleanup_threads(monkeypatch) -> None:
    started_lock = threading.Lock()
    started_count = 0
    completed_count = 0
    force_count = 0
    workers_saturated = threading.Event()
    cleanup_saturated = threading.Event()
    release_cleanup = threading.Event()
    worker_limit = mcp_server._MCP_MAX_WORKERS
    request_count = worker_limit + 44

    class TrackingCancellation:
        def __init__(self) -> None:
            self._cancelled = threading.Event()
            self._forced = threading.Event()
            self.force_deadline: float | None = None

        def cancel(self) -> None:
            self.force_deadline = time.monotonic() + 0.05
            self._cancelled.set()

        def force(self) -> None:
            nonlocal force_count
            with started_lock:
                force_count += 1
                if force_count == worker_limit:
                    cleanup_saturated.set()
            release_cleanup.wait(timeout=4)
            self._forced.set()

        def wait(self, timeout: float | None = None) -> bool:
            return self._cancelled.wait(timeout)

        def wait_for_force(self, timeout: float | None = None) -> bool:
            return self._forced.wait(timeout)

    class TrackingRunControl:
        def __init__(self) -> None:
            self.cancellation = TrackingCancellation()

    def workflow(*, cancel) -> None:
        nonlocal started_count, completed_count
        with started_lock:
            started_count += 1
            assert started_count <= worker_limit
            if started_count == worker_limit:
                workers_saturated.set()
        assert cancel.wait(timeout=2)
        assert cancel.wait_for_force(timeout=2)
        with started_lock:
            completed_count += 1

    async def wait_for_event(event: threading.Event) -> None:
        while not event.is_set():
            await anyio.sleep(0.01)

    async def run() -> None:
        with anyio.fail_after(4):
            with anyio.CancelScope() as scope:
                async with anyio.create_task_group() as tasks:
                    for _ in range(request_count):
                        tasks.start_soon(mcp_server._run_cancellable, workflow)
                    await wait_for_event(workers_saturated)
                    await anyio.sleep(0.05)
                    assert started_count == worker_limit
                    scope.cancel()
                    with anyio.CancelScope(shield=True):
                        await wait_for_event(cleanup_saturated)
                        worker_threads = [
                            thread
                            for thread in threading.enumerate()
                            if thread.name == "AnyIO worker thread"
                        ]
                        assert len(worker_threads) <= mcp_server._MCP_THREAD_CAPACITY
                        release_cleanup.set()

    monkeypatch.setattr(mcp_server, "RunControl", TrackingRunControl)
    try:
        anyio.run(run)
    finally:
        release_cleanup.set()

    assert started_count == worker_limit
    assert force_count == worker_limit
    assert completed_count == worker_limit


def test_run_cancellable_reaps_resistant_local_descendants_before_return(
    real_process_harness,
) -> None:
    process_name = "mcp-local-command"
    worker_finished = threading.Event()

    def workflow(*, cancel) -> None:
        try:
            run_captured(
                real_process_harness.resistant_argv(process_name),
                cwd=real_process_harness.root,
                input="",
                cancel=cancel,
            )
        finally:
            worker_finished.set()

    async def cancel_after_descendant(scope: anyio.CancelScope) -> None:
        await anyio.to_thread.run_sync(
            real_process_harness.wait_for_marker,
            f"{process_name}.child.pid",
        )
        scope.cancel()

    async def run() -> None:
        with anyio.CancelScope() as scope:
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(cancel_after_descendant, scope)
                await mcp_server._run_cancellable(workflow)

    anyio.run(run)

    assert worker_finished.is_set()
    real_process_harness.assert_tree_absent(process_name, timeout=0.1)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process cleanup required")
def test_api_backed_mcp_cancellation_joins_provider_before_request_return(
    real_process_harness,
    monkeypatch,
) -> None:
    request_name = "mcp-provider-request"
    root = real_process_harness.root
    target = root / "late-tool-target.txt"
    original_execute = api_http.MultiprocessUrlRequest.execute
    request_join_lock = threading.Lock()

    def marked_execute(request):
        try:
            return original_execute(request)
        finally:
            child_pid = int(
                real_process_harness.wait_for_marker(
                    f"{request_name}.request.pid"
                )
            )
            real_process_harness.assert_pid_absent(child_pid, timeout=0.1)
            with request_join_lock:
                marker = root / f"{request_name}.request-joined"
                if not marker.exists():
                    marker.write_text(str(time.monotonic()), encoding="utf-8")

    monkeypatch.setenv("BETTERBORG_TEST_REQUEST_ROOT", str(root))
    monkeypatch.setenv("BETTERBORG_TEST_REQUEST_NAME", request_name)
    monkeypatch.setenv("BETTERBORG_TEST_REQUEST_RESISTANT", "1")
    monkeypatch.setattr(api_http, "_url_request_worker", blocked_dns_url_request_worker)
    monkeypatch.setattr(api_http.MultiprocessUrlRequest, "execute", marked_execute)

    provider_response = openai_response(
        [
            openai_function_call(
                "apply_patch",
                {
                    "patch": (
                        "*** Begin Patch\n"
                        "*** Add File: late-tool-target.txt\n"
                        "+late mutation\n"
                        "*** End Patch"
                    )
                },
                call_id="late-tool",
            )
        ]
    )

    def respond(_request):
        return (
            200,
            {"content-type": "application/json"},
            json.dumps(provider_response).encode(),
        )

    adapter_statuses: list[AgentStatus] = []

    def workflow(_io, *, cancel) -> None:
        try:
            result = OpenAIAdapter(
                "analysis",
                api_key="test-key",
                transport=UrllibOpenAITransport(server.url()),
            ).run(
                AgentRunSpec(
                    system_prompt="Do not mutate after cancellation.",
                    user_prompt="Wait for the provider.",
                    schema={
                        "type": "object",
                        "properties": {"status": {"type": "string"}},
                    },
                    cwd=root,
                    model="gpt-test",
                    log_path=root / "provider.jsonl",
                    result_path=root / "provider-result.json",
                    allowed_tools=("apply_patch",),
                ),
                cancel=cancel,
            )
            adapter_statuses.append(result.status)
            (root / f"{request_name}.adapter-finished").write_text(
                str(time.monotonic()),
                encoding="utf-8",
            )
        finally:
            (root / f"{request_name}.worker-finished").write_text(
                str(time.monotonic()),
                encoding="utf-8",
            )

    request_teardown: list[float] = []
    request_finished: list[float] = []
    session_closed: list[float] = []
    cancellation_requested: list[float] = []
    protocol_errors: list[McpError] = []
    original_run_cancellable = mcp_server._run_cancellable

    async def marked_run_cancellable(function, *args):
        try:
            return await original_run_cancellable(function, *args)
        finally:
            request_teardown.append(time.monotonic())

    async def run() -> None:
        async def elicit(_context, _params):
            raise AssertionError("analyze must not elicit")

        async with create_connected_server_and_client_session(
            mcp_server.server,
            raise_exceptions=True,
            elicitation_callback=elicit,
        ) as session:
            request_id = session._request_id
            call_finished = anyio.Event()

            async def call() -> None:
                try:
                    await session.call_tool("analyze", {})
                except McpError as error:
                    protocol_errors.append(error)
                finally:
                    request_finished.append(time.monotonic())
                    call_finished.set()

            async with anyio.create_task_group() as tasks:
                tasks.start_soon(call)
                await anyio.to_thread.run_sync(
                    real_process_harness.wait_for_marker,
                    f"{request_name}.dns-gate",
                )
                cancellation_requested.append(time.monotonic())
                await session.send_notification(
                    mcp_types.CancelledNotification(
                        params=mcp_types.CancelledNotificationParams(
                            requestId=request_id,
                            reason="test cancellation",
                        )
                    )
                )
                with anyio.fail_after(DEFAULT_FORCE_GRACE_SECONDS + 0.5):
                    await call_finished.wait()
        session_closed.append(time.monotonic())

    monkeypatch.setattr(mcp_server, "_analyze", workflow)
    monkeypatch.setattr(mcp_server, "_run_cancellable", marked_run_cancellable)
    with LocalHttpServer(respond) as server:
        anyio.run(run)

    child_pid = int(
        real_process_harness.wait_for_marker(f"{request_name}.request.pid")
    )
    joined_at = float(
        real_process_harness.wait_for_marker(f"{request_name}.request-joined")
    )
    adapter_finished_at = float(
        real_process_harness.wait_for_marker(f"{request_name}.adapter-finished")
    )
    worker_finished_at = float(
        real_process_harness.wait_for_marker(f"{request_name}.worker-finished")
    )
    assert adapter_statuses == [AgentStatus.CANCELLED]
    assert len(protocol_errors) == 1
    assert str(protocol_errors[0]) == "Request cancelled"
    assert (
        cancellation_requested[0]
        <= joined_at
        <= adapter_finished_at
        <= worker_finished_at
        <= request_teardown[0]
        <= request_finished[0]
        <= session_closed[0]
    )
    assert (
        joined_at - cancellation_requested[0]
        <= DEFAULT_FORCE_GRACE_SECONDS + 0.1
    )
    real_process_harness.assert_pid_absent(child_pid, timeout=0.1)
    (root / f"release-{request_name}").write_text("released", encoding="utf-8")
    time.sleep(0.05)
    assert target.exists() is False
    assert server.requests == []


def test_cancelling_active_elicitation_unwinds_coroutine_and_worker(
    monkeypatch,
) -> None:
    order: list[str] = []
    worker_finished = threading.Event()
    received_tokens = []
    protocol_errors: list[McpError] = []
    tool_results: list[object] = []
    cancellation_requested: list[float] = []
    request_finished: list[float] = []

    def blocked_create(_name, _source, io, *, cancel):
        received_tokens.append(cancel)
        try:
            io.prompt("Wait for an answer")
        finally:
            order.append("worker-finished")
            worker_finished.set()
        raise AssertionError("cancelled elicitation fabricated a tool result")

    monkeypatch.setattr(mcp_server, "_create", blocked_create)

    # The SDK's default in-memory client runs request callbacks in its receive
    # loop, which prevents it from receiving cancellation while a callback is
    # blocked. Real protocol clients dispatch incoming requests concurrently;
    # make the test peer do the same so this exercises the cancellation
    # notification instead of relying on session teardown.
    received_request = ClientSession._received_request

    async def dispatch_request(self, responder) -> None:
        callback_ready = anyio.Event()
        callback_finished = anyio.Event()
        callback_scopes: list[anyio.CancelScope] = []

        async def handle_request() -> None:
            try:
                await received_request(self, responder)
            finally:
                callback_finished.set()

        async def host_request() -> None:
            async with anyio.create_task_group() as callbacks:
                callback_scopes.append(callbacks.cancel_scope)
                callback_ready.set()
                callbacks.start_soon(handle_request)

        async def cancel_after_callback() -> None:
            await callback_ready.wait()
            callback_scopes[0].cancel()
            await callback_finished.wait()
            responder._completed = True
            responder._on_complete(responder)
            await responder._session._send_response(
                request_id=responder.request_id,
                response=mcp_types.ErrorData(
                    code=0,
                    message="Request cancelled",
                    data=None,
                ),
            )

        # A cancellation response is the peer's acknowledgement that its
        # callback has unwound. The SDK sends that response before its default
        # callback task exits, so this protocol peer tightens the ordering that
        # the Betterborg parent-response fence relies on and verifies.
        responder.cancel = cancel_after_callback
        self._task_group.start_soon(host_request)
        await callback_ready.wait()

    monkeypatch.setattr(ClientSession, "_received_request", dispatch_request)

    async def run() -> None:
        elicitation_started = anyio.Event()
        elicitation_unwound = anyio.Event()
        call_finished = anyio.Event()

        async def elicit(_context, _params):
            elicitation_started.set()
            try:
                await anyio.sleep_forever()
            finally:
                order.append("elicitation-unwound")
                elicitation_unwound.set()

        async with create_connected_server_and_client_session(
            mcp_server.server,
            raise_exceptions=True,
            elicitation_callback=elicit,
        ) as session:
            request_id = session._request_id

            async def call() -> None:
                try:
                    tool_results.append(
                        await session.call_tool(
                            "create",
                            {"name": "cancel-me"},
                        )
                    )
                except McpError as error:
                    protocol_errors.append(error)
                finally:
                    order.append("request-finished")
                    request_finished.append(time.monotonic())
                    call_finished.set()

            async with anyio.create_task_group() as tasks:
                tasks.start_soon(call)
                await elicitation_started.wait()
                cancellation_requested.append(time.monotonic())
                await session.send_notification(
                    mcp_types.CancelledNotification(
                        params=mcp_types.CancelledNotificationParams(
                            requestId=request_id,
                            reason="test cancellation",
                        )
                    )
                )
                with anyio.fail_after(DEFAULT_FORCE_GRACE_SECONDS + 0.5):
                    await call_finished.wait()

                assert elicitation_unwound.is_set()
                assert worker_finished.is_set()
                assert len((await session.list_tools()).tools) > 0

    anyio.run(run)

    assert tool_results == []
    assert len(protocol_errors) == 1
    assert str(protocol_errors[0]) == "Request cancelled"
    assert order == [
        "elicitation-unwound",
        "worker-finished",
        "request-finished",
    ]
    assert (
        request_finished[0] - cancellation_requested[0]
        <= DEFAULT_FORCE_GRACE_SECONDS + 0.1
    )
    assert len(received_tokens) == 1
    assert received_tokens[0].is_set() is True


def test_disconnected_elicitation_cancellation_still_fences_worker(
    monkeypatch,
) -> None:
    order: list[str] = []
    worker_unwinding = threading.Event()
    release_worker = threading.Event()
    worker_finished = threading.Event()
    notification_failed = threading.Event()
    protocol_errors: list[McpError] = []

    def blocked_create(_name, _source, io, *, cancel):
        try:
            io.prompt("Wait for an answer")
        finally:
            worker_unwinding.set()
            assert release_worker.wait(timeout=2.0)
            order.append("worker-finished")
            worker_finished.set()
        raise AssertionError("cancelled elicitation fabricated a tool result")

    monkeypatch.setattr(mcp_server, "_create", blocked_create)

    received_request = ClientSession._received_request

    async def dispatch_request(self, responder) -> None:
        async def handle_request() -> None:
            await received_request(self, responder)

        self._task_group.start_soon(handle_request)

    monkeypatch.setattr(ClientSession, "_received_request", dispatch_request)

    original_send_notification = ServerSession.send_notification

    async def fail_nested_cancellation(
        self,
        notification,
        related_request_id=None,
    ) -> None:
        if isinstance(notification.root, mcp_types.CancelledNotification):
            notification_failed.set()
            raise anyio.BrokenResourceError
        await original_send_notification(
            self,
            notification,
            related_request_id=related_request_id,
        )

    monkeypatch.setattr(ServerSession, "send_notification", fail_nested_cancellation)

    async def run() -> None:
        elicitation_started = anyio.Event()
        call_finished = anyio.Event()

        async def elicit(_context, _params):
            elicitation_started.set()
            await anyio.sleep_forever()

        async with create_connected_server_and_client_session(
            mcp_server.server,
            raise_exceptions=True,
            elicitation_callback=elicit,
        ) as session:
            request_id = session._request_id

            async def call() -> None:
                try:
                    await session.call_tool("create", {"name": "cancel-me"})
                except McpError as error:
                    protocol_errors.append(error)
                finally:
                    order.append("request-finished")
                    call_finished.set()

            async with anyio.create_task_group() as tasks:
                tasks.start_soon(call)
                await elicitation_started.wait()
                await session.send_notification(
                    mcp_types.CancelledNotification(
                        params=mcp_types.CancelledNotificationParams(
                            requestId=request_id,
                            reason="test cancellation",
                        )
                    )
                )
                with anyio.fail_after(0.5):
                    while not notification_failed.is_set():
                        await anyio.sleep(0.01)
                    while not worker_unwinding.is_set():
                        await anyio.sleep(0.01)
                assert call_finished.is_set() is False
                release_worker.set()
                with anyio.fail_after(DEFAULT_FORCE_GRACE_SECONDS + 0.5):
                    await call_finished.wait()
                assert worker_finished.is_set()
                assert len((await session.list_tools()).tools) > 0

    anyio.run(run)

    assert len(protocol_errors) == 1
    assert str(protocol_errors[0]) == "Request cancelled"
    assert order == ["worker-finished", "request-finished"]


def test_cancelled_execute_is_durable_before_request_returns(
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    real_process_harness,
    monkeypatch,
) -> None:
    repository, paths = planning_cli_repository(
        committed_git_repo,
        "mcp-cancel-execute",
    )
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "mcp-cancel-execute")
        assert borg is not None
        borg = store.compare_and_set_borg_state(
            borg.id,
            expected_state=borg.state,
            expected_version=borg.state_version,
            new_state=BorgState.READY_TO_EXECUTE,
        )
        approval = PlanApproval(
            borg_id=borg.id,
            plan_digest="sha256:mcp-cancel-execute",
            manifest={},
        )
        store.append_plan_approval(approval)
        second_task = {
            **_task_body(),
            "stem": "02-cancellable-runtime",
            "title": "Cancellable runtime task",
        }
        current = approved_task_generation(
            store,
            borg,
            approval,
            body=[_task_body(), second_task],
            round_number=1,
        )
        TaskPublisher(repository, store).publish(current.generation.id)
        records = store.list_task_records(current.generation.id)

    process_name = "mcp-execute-local-command"
    host_finished = threading.Event()
    first_task, active_task = records

    def behavior(context: ScheduledTaskContext) -> TaskRuntimeStatus | None:
        if context.claim.task_id == first_task.id:
            context.transition(
                TaskRuntimeStatus.CLAIMED,
                TaskRuntimeStatus.DONE,
            )
            return TaskRuntimeStatus.DONE
        assert context.claim.task_id == active_task.id
        try:
            run_captured(
                real_process_harness.resistant_argv(process_name),
                cwd=real_process_harness.root,
                input="",
                cancel=context.cancel,
            )
        except KeyboardInterrupt:
            return None
        raise AssertionError("resistant command returned without cancellation")

    def invoke(
        _paths,
        store,
        _config,
        _repository_id,
        _borg_id,
        _generation_id,
        *,
        cancel,
        progress,
    ):
        assert progress is None
        try:
            scheduler = HostTaskScheduler(
                store,
                behavior,
                config=HostSchedulerConfig(
                    jobs=1,
                    poll_interval_seconds=0.005,
                ),
            ).run(
                borg.id,
                current.generation.id,
                cancel=cancel,
            )
        finally:
            host_finished.set()
        return HostExecutionResult(
            preflight=HostPreflightPlan(
                repository_root=paths.root,
                commands=(),
                prepare_commands=(),
                materialize_commands=(),
                environment_files=(),
                executables=(),
                required_secret_names=(),
                compose_files=(),
                services=(),
            ),
            scheduler=scheduler,
        )

    monkeypatch.setattr(
        mcp_server,
        "_paths",
        lambda *, trusted, io=None, cancel=None: paths,
    )
    monkeypatch.setattr(cli_module, "_invoke_host_execution", invoke)
    protocol_errors: list[McpError] = []

    async def run() -> None:
        async def elicit(_context, _params):
            return mcp_types.ElicitResult(
                action="accept",
                content={"approved": True},
            )

        async with create_connected_server_and_client_session(
            mcp_server.server,
            raise_exceptions=True,
            elicitation_callback=elicit,
        ) as session:
            request_id = session._request_id
            call_finished = anyio.Event()

            async def call() -> None:
                try:
                    await session.call_tool("execute", {"name": borg.name})
                except McpError as error:
                    protocol_errors.append(error)
                finally:
                    call_finished.set()

            async with anyio.create_task_group() as tasks:
                tasks.start_soon(call)
                await anyio.to_thread.run_sync(
                    real_process_harness.wait_for_marker,
                    f"{process_name}.child.pid",
                )
                await session.send_notification(
                    mcp_types.CancelledNotification(
                        params=mcp_types.CancelledNotificationParams(
                            requestId=request_id,
                            reason="test cancellation",
                        )
                    )
                )
                with anyio.fail_after(4):
                    await call_finished.wait()

    anyio.run(run)

    assert host_finished.is_set()
    assert len(protocol_errors) == 1
    assert str(protocol_errors[0]) == "Request cancelled"
    real_process_harness.assert_tree_absent(process_name, timeout=0.1)
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        operation = store.list_execution_runs(borg.id)[-1]
        retained = store.get_task_runtime(first_task.id)
        unfinished = store.get_task_runtime(active_task.id)
        claims = store.list_task_claims(operation.id)
        event_ids = [event.id for event in store.list_execution_events(operation.id)]
        assert operation.status is ExecutionRunStatus.CANCELLED
        assert retained is not None
        assert retained.status is TaskRuntimeStatus.DONE
        assert unfinished is not None
        assert unfinished.status is TaskRuntimeStatus.PENDING
        assert len(claims) == 2
        assert all(claim.released_at is not None for claim in claims)

    time.sleep(0.05)
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        assert [
            event.id for event in store.list_execution_events(operation.id)
        ] == event_ids


def test_empty_elicitation_capability_supports_legacy_form_mode() -> None:
    params = mcp_types.InitializeRequestParams.model_validate(
        {
            "protocolVersion": "2025-06-18",
            "capabilities": {"elicitation": {}},
            "clientInfo": {"name": "legacy-form-client", "version": "1.0"},
        }
    )
    context = SimpleNamespace(
        session=SimpleNamespace(client_params=params),
    )

    assert mcp_server.McpInteractiveIO.supported(context) is True


def test_url_only_elicitation_capability_does_not_support_forms() -> None:
    params = mcp_types.InitializeRequestParams.model_validate(
        {
            "protocolVersion": mcp_types.LATEST_PROTOCOL_VERSION,
            "capabilities": {"elicitation": {"url": {}}},
            "clientInfo": {"name": "url-only-client", "version": "1.0"},
        }
    )
    context = SimpleNamespace(
        session=SimpleNamespace(client_params=params),
    )

    assert mcp_server.McpInteractiveIO.supported(context) is False


def _task_body() -> dict:
    return {
        "stage": "01-foundation",
        "stem": "01-runtime",
        "title": "Project runtime task status",
        "why": "Consumers need one runtime projection.",
        "scope": ["Expose runtime state."],
        "implementation_notes": [],
        "acceptance_criteria": ["Runtime state is exact."],
        "tests": ["Exercise runtime projection."],
        "dependencies": [],
        "out_of_scope": [],
        "plan_refs": ["P1.deliverable.1"],
        "estimate_complexity": "small",
    }


def _pm_tasks(plan: dict) -> dict:
    refs = [
        element.ref for element in build_plan_element_catalog(plan) if element.required
    ]
    return {
        "summary": "One task covers the approved plan.",
        "tasks": [
            {
                "stage": "01-release-workflow",
                "stem": "01-document-release",
                "title": "Document the release workflow",
                "why": "The approved workflow needs an executable task.",
                "scope": ["Document the release path."],
                "implementation_notes": [],
                "acceptance_criteria": ["The release path is documented."],
                "tests": ["Assert the documented public workflow."],
                "dependencies": [],
                "out_of_scope": [],
                "plan_refs": refs,
                "estimate_complexity": "small",
            }
        ],
    }


def _seed_plan_awaiting_approval(
    paths: RepoPaths,
    repository,
    name: str,
    plan: dict,
) -> None:
    """Persist one completed architect plan and gate its Borg on approval."""
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, name)
        assert borg is not None
        attempt = PlanningAttempt(
            borg_id=borg.id,
            phase="architect_plan",
            round=1,
            adapter="mock",
            model="test-model",
        )
        store.append_planning_attempt(attempt)
        store.complete_planning_attempt(
            attempt.id,
            status=PlanningAttemptStatus.COMPLETED,
            result=plan,
            summary="Ready for approval.",
        )
        store.compare_and_set_borg_state(
            borg.id,
            expected_state=borg.state,
            expected_version=borg.state_version,
            new_state=BorgState.PLAN_APPROVAL_PENDING,
        )


def _published_runtime(
    root: Path,
    planning_cli_repository,
    approved_task_generation,
):
    repository, paths = planning_cli_repository(root, "mcp-runtime")
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "mcp-runtime")
        assert borg is not None
        borg = store.compare_and_set_borg_state(
            borg.id,
            expected_state=borg.state,
            expected_version=borg.state_version,
            new_state=BorgState.READY_TO_EXECUTE,
        )
        approval = PlanApproval(
            borg_id=borg.id,
            plan_digest="sha256:mcp-approved-plan",
            manifest={},
        )
        store.append_plan_approval(approval)
        current = approved_task_generation(
            store,
            borg,
            approval,
            body=_task_body(),
            round_number=1,
            task_ref="T-MCP-1",
        )
        publication = TaskPublisher(repository, store).publish(current.generation.id)

        started_at = utcnow()
        run = ExecutionRun(
            borg_id=borg.id,
            generation_id=current.generation.id,
            started_at=started_at,
            heartbeat_at=started_at,
            lease_expires_at=started_at + timedelta(minutes=5),
        )
        runtime = TaskRuntime(
            generation_id=current.generation.id,
            task_id=current.task.id,
            status=TaskRuntimeStatus.FIX,
            state_reason="review requested changes",
            review_round=2,
        )
        claim = TaskClaim(
            run_id=run.id,
            task_id=current.task.id,
            resume_phase="fix",
            claimed_at=started_at,
            lease_expires_at=started_at + timedelta(minutes=2),
        )
        store.add_execution_run(run)
        store.add_task_runtime(runtime)
        store.append_task_claim(claim)
        for index, (billing, cost, duration) in enumerate(
            (
                (BillingMode.API, 0.75, 4.0),
                (BillingMode.SUBSCRIPTION, 8.0, 6.0),
            ),
            start=1,
        ):
            store.append_agent_attempt(
                AgentAttempt(
                    run_id=run.id,
                    claim_id=claim.id,
                    task_id=current.task.id,
                    phase=f"phase-{index}",
                    attempt_number=1,
                    adapter="mock",
                    model="test-model",
                    billing_mode=billing,
                    status=AgentStatus.COMPLETED,
                    log_path=f"artifacts/{index}.log",
                    duration_seconds=duration,
                    usage=AgentUsage(cost_usd=cost),
                    started_at=started_at,
                    finished_at=started_at + timedelta(seconds=duration),
                ),
                run.owner_token,
                claim.claim_token,
                now=started_at,
            )
    return paths, borg, current, publication


def test_tool_inventory_has_typed_results_and_no_removed_gates() -> None:
    tools = _list_tools()
    schemas = {tool.name: tool.outputSchema for tool in tools}

    assert [tool.name for tool in tools] == [
        "init",
        "analyze",
        "create",
        "plan",
        "task_list",
        "execute",
    ]
    assert all(tool.outputSchema is not None for tool in tools)
    assert {"approve_task", "task_approve", "decompose"}.isdisjoint(
        tool.name for tool in tools
    )
    assert {
        name: schema["title"] for name, schema in schemas.items() if schema is not None
    } == {
        "init": "InitializeResult",
        "analyze": "AnalyzeResult",
        "create": "CreateResult",
        "plan": "PlanResult",
        "task_list": "TaskListResult",
        "execute": "ExecuteResult",
    }
    assert schemas["init"]["$defs"]["InitializeData"]["properties"][
        "repository_id"
    ]["format"] == "uuid"
    assert schemas["analyze"]["$defs"]["AnalyzeData"]["properties"]["score"] == {
        "maximum": 5,
        "minimum": 0,
        "title": "Score",
        "type": "number",
    }
    assert schemas["create"]["$defs"]["CreateData"]["properties"]["borg_id"][
        "format"
    ] == "uuid"
    assert schemas["plan"]["$defs"]["PlanDocument"][
        "additionalProperties"
    ] is False
    assert schemas["execute"]["$defs"]["ExecutionEstimateData"][
        "additionalProperties"
    ] is False
    assert not any(
        value is True
        for schema in schemas.values()
        for value in _additional_properties(schema)
    )
    plan_schema = next(tool.inputSchema for tool in tools if tool.name == "plan")
    assert plan_schema["properties"]["action"]["enum"] == [
        "start",
        "show",
        "change",
        "approve",
    ]
    assert "answers" not in plan_schema["properties"]
    create_schema = next(tool.inputSchema for tool in tools if tool.name == "create")
    assert "confirmed" not in create_schema["properties"]
    execute_schema = next(tool.inputSchema for tool in tools if tool.name == "execute")
    assert "auto_execute" not in execute_schema["properties"]


def test_progress_resources_reconnect_and_resume_durable_execution_events(
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch,
) -> None:
    paths, borg, current, _publication = _published_runtime(
        committed_git_repo,
        planning_cli_repository,
        approved_task_generation,
    )
    monkeypatch.setattr(
        mcp_server,
        "_paths",
        lambda *, trusted, io=None, cancel=None: paths,
    )

    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        operation = store.list_execution_runs(borg.id)[0]
        attempt = next(
            item
            for item in store.list_agent_attempts(current.task.id)
            if item.phase == "phase-1"
        )
        started = operation.started_at
        acquired = ExecutionEvent(
            id=UUID("00000000-0000-0000-0000-000000000101"),
            run_id=operation.id,
            kind="run.acquired",
            payload={"message": "Execution started"},
            created_at=started,
        )
        attempted = ExecutionEvent(
            id=UUID("00000000-0000-0000-0000-000000000102"),
            run_id=operation.id,
            task_id=current.task.id,
            attempt_id=attempt.id,
            kind="agent.progress",
            payload={"summary": "Coding attempt completed"},
            created_at=started + timedelta(seconds=1),
        )
        store.append_execution_event(acquired)
        store.append_execution_event(attempted)

    templates = _list_resource_templates()
    assert [(item.name, item.uriTemplate) for item in templates] == [
        ("operation_progress", "betterborg://progress/{operation_id}"),
        (
            "operation_progress_after",
            "betterborg://progress/{operation_id}/after/{event_id}",
        ),
    ]

    initial = _read_resource(f"betterborg://progress/{operation.id}")
    assert initial == {
        "events": [
            {
                "event_id": str(acquired.id),
                "operation_id": str(operation.id),
                "borg": "mcp-runtime",
                "task": None,
                "phase": "execution",
                "message": "Execution started",
                "completed": 0,
                "total": 1,
            },
            {
                "event_id": str(attempted.id),
                "operation_id": str(operation.id),
                "borg": "mcp-runtime",
                "task": "T-MCP-1",
                "phase": "phase-1",
                "message": "Coding attempt completed",
                "completed": 0,
                "total": 1,
            },
        ]
    }

    completed = ExecutionEvent(
        id=UUID("00000000-0000-0000-0000-000000000103"),
        run_id=operation.id,
        task_id=current.task.id,
        kind="task.phase_transitioned",
        payload={
            "from": "merging",
            "to": "done",
            "message": "Task completed",
        },
        created_at=started + timedelta(seconds=2),
    )
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        store.append_execution_event(completed)

    resumed = _read_resource(
        f"betterborg://progress/{operation.id}/after/{attempted.id}"
    )
    assert resumed == {
        "events": [
            {
                "event_id": str(completed.id),
                "operation_id": str(operation.id),
                "borg": "mcp-runtime",
                "task": "T-MCP-1",
                "phase": "done",
                "message": "Task completed",
                "completed": 1,
                "total": 1,
            }
        ]
    }
    reconnected = _read_resource(f"betterborg://progress/{operation.id}")
    assert [event["event_id"] for event in reconnected["events"]] == [
        str(acquired.id),
        str(attempted.id),
        str(completed.id),
    ]


def _additional_properties(value) -> list[object]:
    if isinstance(value, dict):
        current = (
            [value["additionalProperties"]]
            if "additionalProperties" in value
            else []
        )
        return current + [
            item
            for child in value.values()
            for item in _additional_properties(child)
        ]
    if isinstance(value, list):
        return [item for child in value for item in _additional_properties(child)]
    return []


def test_init_and_analyze_use_repository_service_with_typed_actions(
    committed_git_repo: Path,
    monkeypatch,
) -> None:
    paths = RepoPaths.discover(committed_git_repo)
    paths.state_dir.mkdir(parents=True)
    score = paths.score_report
    prompt = paths.prompts_dir / "coding.md"
    improvement = paths.improvement_prds_dir / "runtime.md"
    repository = SimpleNamespace(id=uuid4())
    analysis = SimpleNamespace(id=uuid4(), overall_score=4.5, score_delta=0.5)
    previous = SimpleNamespace(overall_score=4.0)
    calls: list[str] = []
    selected_stages: list[AgentStage] = []

    class FakeRepositoryService:
        def __init__(self, service_paths, _store, factory, *, cancel) -> None:
            assert service_paths == paths
            assert cancel is not None
            factory(object())

        def initialize(self):
            calls.append("init")
            return SimpleNamespace(
                initialized=False,
                repository=repository,
                analysis=analysis,
                score_path=score,
                prompts=(SimpleNamespace(role="coding", path=prompt),),
                improvement_prds=(
                    SimpleNamespace(
                        path=improvement,
                        suggested_borg_name="runtime-fix",
                    ),
                ),
            )

        def analyze(self):
            calls.append("analyze")
            return SimpleNamespace(
                repository=repository,
                analysis=analysis,
                previous_analysis=previous,
                score_path=score,
                prompts=(SimpleNamespace(role="coding", path=prompt),),
                improvement_prds=(
                    SimpleNamespace(
                        path=improvement,
                        suggested_borg_name="runtime-fix",
                    ),
                ),
            )

    monkeypatch.setattr(
        mcp_server,
        "_paths",
        lambda *, trusted, io=None, cancel=None: paths,
    )
    monkeypatch.setattr(mcp_server, "RepositoryService", FakeRepositoryService)
    monkeypatch.setattr(
        mcp_server,
        "select_agent",
        lambda _config, stage, _paths, **_kwargs: selected_stages.append(stage),
    )

    initialized = _structured(_call_tool("init"))
    analyzed = _structured(_call_tool("analyze"))

    assert calls == ["init", "analyze"]
    assert selected_stages == [AgentStage.ANALYSIS, AgentStage.ANALYSIS]
    assert initialized["status"] == "already_initialized"
    assert initialized["artifacts"] == [
        {"kind": "score", "path": ".betterborg/score.md"},
        {"kind": "coding_prompt", "path": ".betterborg/prompts/coding.md"},
        {"kind": "improvement_prd", "path": ".betterborg/prds/improvements/runtime.md"},
    ]
    assert initialized["next_actions"] == [
        {
            "tool": "create",
            "arguments": {
                "name": "runtime-fix",
                "source": ".betterborg/prds/improvements/runtime.md",
            },
        }
    ]
    assert analyzed["status"] == "completed"
    assert analyzed["data"]["previous_score"] == 4.0
    assert analyzed["data"]["delta"] == 0.5


def test_create_and_plan_approval_are_service_backed_and_typed(
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch,
) -> None:
    paths, borg, _current, _publication = _published_runtime(
        committed_git_repo,
        planning_cli_repository,
        approved_task_generation,
    )
    prd_path = paths.tracked_dir / "prds" / "new-borg.md"
    created_borg = Borg(repository_id=borg.repository_id, name="new-borg")
    create_calls: list[tuple[str, Path | None]] = []
    selected_stages: list[AgentStage] = []

    class FakeCreateService:
        def __init__(
            self,
            repository,
            _store,
            _agent,
            *,
            io,
            interactive: bool,
            cancel,
        ) -> None:
            assert repository.id == borg.repository_id
            assert io is not None
            assert interactive is True
            assert cancel is not None

        def create(self, name, source):
            create_calls.append((name, source))
            return SimpleNamespace(
                borg=created_borg,
                confirmed=True,
                questions=(),
                body_md="# New Borg\n",
                prd_path=prd_path,
            )

    monkeypatch.chdir(committed_git_repo)
    monkeypatch.setattr(
        mcp_server,
        "_paths",
        lambda *, trusted, io=None, cancel=None: paths,
    )
    monkeypatch.setattr(
        mcp_server,
        "select_agent",
        lambda _config, stage, _paths, **_kwargs: selected_stages.append(stage),
    )
    monkeypatch.setattr(mcp_server, "CreateService", FakeCreateService)

    created = _structured(
        _call_tool(
            "create",
            {"name": "new-borg", "source": "source.md"},
        )
    )

    assert create_calls == [("new-borg", paths.root / "source.md")]
    assert selected_stages == [AgentStage.REQUIREMENTS]
    assert created["status"] == "confirmed"
    assert created["artifacts"] == [
        {"kind": "prd", "path": ".betterborg/prds/new-borg.md"}
    ]
    assert created["next_actions"] == [
        {"tool": "plan", "arguments": {"name": "new-borg", "action": "start"}}
    ]


def test_plan_show_carries_an_assumption_the_architect_made(
    planning_plan_response,
) -> None:
    """An unattended plan must survive the surface a person approves it on.

    The plan document is closed to unknown fields, so an assumption that the
    Architect schema allows but this model does not know is not merely dropped
    here: it makes the plan unreadable, and unapprovable, over MCP.
    """
    plan = planning_plan_response()
    plan["assumptions"] = [
        {"question": "Which platforms are required?", "assumption": "Linux only."}
    ]

    document = mcp_server.PlanDocument.model_validate(plan)

    assert document.assumptions[0].question == "Which platforms are required?"
    assert document.assumptions[0].assumption == "Linux only."
    shown = mcp_server.PlanShowData(borg="unattended", plan=document).model_dump()
    assert shown["plan"]["assumptions"] == (
        {"question": "Which platforms are required?", "assumption": "Linux only."},
    )


def test_plan_start_recovers_questions_injects_answers_and_shows_plan(
    committed_git_repo: Path,
    planning_cli_repository,
    planning_plan_response,
    tech_lead_approval_response,
    monkeypatch,
) -> None:
    repository, paths = planning_cli_repository(committed_git_repo, "mcp-start")
    plan = planning_plan_response(summary="MCP plan is ready.")
    architect = MockAdapter(name="openai").queue(
        MockResponse(
            payload={
                "decision": "ask_more",
                "questions": [
                    {
                        "id": "q1",
                        "question": "Which platforms are required?",
                        "why": "The answer controls the test matrix.",
                    }
                ],
            }
        )
    ).queue(MockResponse(payload={"decision": "ready_to_plan"})).queue(
        MockResponse(payload=plan)
    )
    tech_lead = MockAdapter(name="openai").queue(
        MockResponse(payload=tech_lead_approval_response())
    )
    selected_stages: list[AgentStage] = []
    continuations: list[tuple[str, str | None]] = []
    continue_planning = cli_module._continue_planning

    def select(_config, stage, _paths, **_kwargs):
        selected_stages.append(stage)
        return {
            AgentStage.ARCHITECT: architect,
            AgentStage.TECH_LEAD: tech_lead,
        }[stage]

    def continue_spy(selected_paths, name, *, change_note=None, io=None, cancel=None):
        continuations.append((name, change_note))
        return continue_planning(
            selected_paths,
            name,
            change_note=change_note,
            io=io,
            cancel=cancel,
        )

    monkeypatch.chdir(committed_git_repo)
    monkeypatch.setattr(
        mcp_server,
        "_paths",
        lambda *, trusted, io=None, cancel=None: paths,
    )
    monkeypatch.setattr(cli_module, "select_agent", select)
    monkeypatch.setattr(cli_module, "_continue_planning", continue_spy)

    requests: list = []
    started = _structured(
        _call_tool(
            "plan",
            {"name": "mcp-start", "action": "start"},
            answers=("Linux, macOS, and Windows.",),
            requests=requests,
        )
    )
    shown = _structured(
        _call_tool("plan", {"name": "mcp-start", "action": "show"})
    )

    assert started["status"] == BorgState.PLAN_APPROVAL_PENDING.value
    assert continuations == [("mcp-start", None)]
    assert selected_stages == [AgentStage.ARCHITECT, AgentStage.TECH_LEAD]
    assert len(architect.calls) == 3
    assert len(tech_lead.calls) == 1
    assert [action["arguments"]["action"] for action in started["next_actions"]] == [
        "show",
        "approve",
    ]
    assert shown["status"] == BorgState.PLAN_APPROVAL_PENDING.value
    assert shown["data"]["borg"] == "mcp-start"
    assert shown["data"]["plan"]["summary"] == "MCP plan is ready."
    assert shown["data"]["plan"]["phases"][0]["name"] == "01-release-workflow"
    assert len(requests) == 1
    assert "Which platforms are required?" in requests[0].message
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "mcp-start")
        assert borg is not None
        questions = store.list_planning_questions(borg.id)
    assert questions[0].answers == [
        {"q_id": "q1", "answer": "Linux, macOS, and Windows."}
    ]


def test_plan_change_validates_note_and_preserves_service_history(
    committed_git_repo: Path,
    planning_cli_repository,
    planning_plan_response,
    tech_lead_approval_response,
    monkeypatch,
) -> None:
    repository, paths = planning_cli_repository(committed_git_repo, "mcp-change")
    original = planning_plan_response(summary="Original MCP plan.")
    revised = planning_plan_response(summary="Revised MCP plan.")
    architect = MockAdapter(name="openai")
    for payload in (
        {"decision": "ready_to_plan"},
        original,
    ):
        architect.queue(MockResponse(payload=payload))
    tech_lead = MockAdapter(name="openai").queue(
        MockResponse(payload=tech_lead_approval_response())
    )
    selected_stages: list[AgentStage] = []
    continuations: list[tuple[str, str | None]] = []
    continue_planning = cli_module._continue_planning

    def select(_config, stage, _paths, **_kwargs):
        selected_stages.append(stage)
        return {
            AgentStage.ARCHITECT: architect,
            AgentStage.TECH_LEAD: tech_lead,
        }[stage]

    def continue_spy(selected_paths, name, *, change_note=None, io=None, cancel=None):
        continuations.append((name, change_note))
        return continue_planning(
            selected_paths,
            name,
            change_note=change_note,
            io=io,
            cancel=cancel,
        )

    monkeypatch.chdir(committed_git_repo)
    monkeypatch.setattr(
        mcp_server,
        "_paths",
        lambda *, trusted, io=None, cancel=None: paths,
    )
    monkeypatch.setattr(cli_module, "select_agent", select)
    monkeypatch.setattr(cli_module, "_continue_planning", continue_spy)

    started = _structured(
        _call_tool("plan", {"name": "mcp-change", "action": "start"})
    )
    assert started["status"] == BorgState.PLAN_APPROVAL_PENDING.value

    invalid = _call_tool(
        "plan",
        {"name": "mcp-change", "action": "change", "note": "   "},
    )
    assert invalid.isError is True
    assert "plan change note must not be empty" in invalid.content[0].text

    architect.queue(MockResponse(payload=revised))
    tech_lead.queue(MockResponse(payload=tech_lead_approval_response()))
    changed = _structured(
        _call_tool(
            "plan",
            {
                "name": "mcp-change",
                "action": "change",
                "note": "  Add staged rollout checks.  ",
            },
        )
    )
    shown = _structured(
        _call_tool("plan", {"name": "mcp-change", "action": "show"})
    )

    assert changed["status"] == BorgState.PLAN_APPROVAL_PENDING.value
    assert continuations == [
        ("mcp-change", None),
        ("mcp-change", "Add staged rollout checks."),
    ]
    assert selected_stages == [
        AgentStage.ARCHITECT,
        AgentStage.TECH_LEAD,
        AgentStage.ARCHITECT,
        AgentStage.TECH_LEAD,
    ]
    assert len(architect.calls) == 3
    assert len(tech_lead.calls) == 2
    assert shown["data"]["plan"]["summary"] == "Revised MCP plan."
    assert shown["data"]["plan"]["code_pointers"] == revised["code_pointers"]
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "mcp-change")
        assert borg is not None
        attempts = store.list_planning_attempts(borg.id)
        requests = store.list_plan_change_requests(borg.id)
    assert [item.note for item in requests] == ["Add staged rollout checks."]
    assert [
        item.result["summary"]
        for item in attempts
        if item.phase == "architect_plan" and item.result is not None
    ] == ["Original MCP plan.", "Revised MCP plan."]


def test_plan_approval_automatically_decomposes_without_another_gate(
    committed_git_repo: Path,
    planning_cli_repository,
    planning_plan_response,
    monkeypatch,
) -> None:
    plan = planning_plan_response()
    repository, paths = planning_cli_repository(committed_git_repo, "mcp-plan")
    _seed_plan_awaiting_approval(paths, repository, "mcp-plan", plan)

    project_manager = MockAdapter(name="openai").queue(
        MockResponse(payload=_pm_tasks(plan))
    )
    supervisor = MockAdapter(name="openai").queue(
        MockResponse(
            payload={
                "decision": "approve",
                "summary": "The task is ready.",
                "findings": [],
            }
        )
    )
    selected_stages: list[AgentStage] = []

    def select(_config, stage, _paths, **_kwargs):
        selected_stages.append(stage)
        return {
            AgentStage.PM: project_manager,
            AgentStage.SUPERVISOR: supervisor,
        }[stage]

    monkeypatch.chdir(committed_git_repo)
    monkeypatch.setattr(
        mcp_server,
        "_paths",
        lambda *, trusted, io=None, cancel=None: paths,
    )
    monkeypatch.setattr(
        mcp_server,
        "select_agent",
        select,
    )

    requests: list = []
    result = _structured(
        _call_tool(
            "plan",
            {"name": "mcp-plan", "action": "approve"},
            requests=requests,
        )
    )

    assert result["status"] == BorgState.READY_TO_EXECUTE.value
    assert selected_stages == [AgentStage.PM, AgentStage.SUPERVISOR]
    assert len(project_manager.calls) == 1
    assert len(supervisor.calls) == 1
    assert [artifact["kind"] for artifact in result["artifacts"]] == [
        "approved_plan",
        "task",
    ]
    assert [action["tool"] for action in result["next_actions"]] == [
        "task_list",
        "execute",
    ]
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        stored = store.get_borg_by_name(repository.id, "mcp-plan")
        assert stored is not None
        generations = store.list_task_generations(stored.id)
    assert stored.state is BorgState.READY_TO_EXECUTE
    assert len(generations) == 1
    assert len(requests) == 1
    assert "Approve the current plan" in requests[0].message
    assert not hasattr(mcp_server, "approve_task")
    assert not hasattr(mcp_server, "decompose")


def test_plan_approval_reuses_repository_trust_for_its_managed_worktree(
    committed_git_repo: Path,
    planning_cli_repository,
    planning_plan_response,
    host_capable_adapter,
    monkeypatch,
) -> None:
    plan = planning_plan_response()
    repository, paths = planning_cli_repository(committed_git_repo, "mcp-trust")
    _seed_plan_awaiting_approval(paths, repository, "mcp-trust", plan)
    state_home = committed_git_repo.parent / "mcp-machine-state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
    TrustStore().trust(WorkspaceIdentity.discover(paths))

    project_manager = host_capable_adapter().queue(
        MockResponse(payload=_pm_tasks(plan))
    )
    supervisor = host_capable_adapter().queue(
        MockResponse(
            payload={
                "decision": "approve",
                "summary": "The task is ready.",
                "findings": [],
            }
        )
    )
    adapters = {
        AgentStage.PM: project_manager,
        AgentStage.SUPERVISOR: supervisor,
    }

    def select(_config, stage, selected_paths, **policy):
        return SelectedAgent(
            role=ApiAgentRole.PLANNING,
            adapter=adapters[stage],
            paths=selected_paths,
            **policy,
        )

    monkeypatch.chdir(committed_git_repo)
    monkeypatch.setattr(
        mcp_server,
        "_paths",
        lambda *, trusted, io=None, cancel=None: paths,
    )
    monkeypatch.setattr(mcp_server, "select_agent", select)

    result = _structured(
        _call_tool("plan", {"name": "mcp-trust", "action": "approve"})
    )

    assert result["status"] == BorgState.READY_TO_EXECUTE.value
    planning_root = paths.worktrees_dir / "planning"
    worktrees = [
        call.cwd for call in (*project_manager.calls, *supervisor.calls)
    ]
    assert len(worktrees) == 2
    assert all(worktree.is_relative_to(planning_root) for worktree in worktrees)
    trusted = json.loads(
        (state_home / "betterborg" / "trusted-workspaces.json").read_text(
            encoding="utf-8"
        )
    )
    assert [
        entry["repository_path"] for entry in trusted["workspaces"].values()
    ] == [str(paths.root)]


def test_task_list_matches_runtime_projection_and_execute_uses_host_service(
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch,
) -> None:
    paths, borg, current, publication = _published_runtime(
        committed_git_repo,
        planning_cli_repository,
        approved_task_generation,
    )
    operation_id = uuid4()
    invoked: list[tuple] = []
    selected_stages: list[AgentStage] = []
    selected_settings = {
        AgentStage.CODING: ("mcp-coding-model", "mcp-coding-effort"),
        AgentStage.REVIEW: ("mcp-review-model", "mcp-review-effort"),
        AgentStage.MERGE: ("mcp-merge-model", "mcp-merge-effort"),
    }
    invoke_host_execution = cli_module._invoke_host_execution

    def select(_config, stage, selected_paths, **_kwargs):
        selected_stages.append(stage)
        model, effort = selected_settings[stage]
        return SelectedAgent(
            role=ApiAgentRole(stage.value),
            adapter=MockAdapter(name="openai"),
            paths=selected_paths,
            model=model,
            effort=effort,
        )

    def invoke(*args, **kwargs):
        invoked.append((args, kwargs))
        return invoke_host_execution(*args, **kwargs)

    def run(service, *_args, **_kwargs):
        coding_config = service._runtime._coding._config
        assert (coding_config.model, coding_config.effort) == (
            "mcp-coding-model",
            "mcp-coding-effort",
        )
        review_config = service._runtime._review_fix._config
        assert (
            review_config.review_model,
            review_config.review_effort,
            review_config.fix_effort,
        ) == (
            "mcp-review-model",
            "mcp-review-effort",
            "mcp-review-effort",
        )
        merge_config = service._runtime._merge._config
        assert (merge_config.model, merge_config.effort) == (
            "mcp-merge-model",
            "mcp-merge-effort",
        )
        return HostExecutionResult(
            preflight=service._runtime.plan,
            scheduler=HostSchedulerResult(
                operation_id=operation_id,
                acquired=True,
                status=ExecutionRunStatus.COMPLETED,
                total=1,
                done=1,
                failed=0,
                blocked=0,
                pending=0,
            ),
        )

    monkeypatch.chdir(committed_git_repo)
    monkeypatch.setattr(
        mcp_server,
        "_paths",
        lambda *, trusted, io=None, cancel=None: paths,
    )
    monkeypatch.setattr(cli_module, "_invoke_host_execution", invoke)
    monkeypatch.setattr(cli_module, "select_agent", select)
    monkeypatch.setattr(cli_module.HostExecutionService, "run", run)
    monkeypatch.setattr(
        cli_module.HostPreflight,
        "validate",
        lambda _preflight, _plan, **_kwargs: HostPreflightPlan(
            repository_root=paths.root,
            commands=(),
            prepare_commands=(),
            materialize_commands=(),
            environment_files=(),
            executables=(),
            required_secret_names=(),
            compose_files=(),
            services=(),
        ),
    )

    listed = _structured(_call_tool("task_list", {"name": borg.name}))
    requests: list = []
    executed = _structured(
        _call_tool("execute", {"name": borg.name}, requests=requests)
    )

    assert listed["generation_id"] == str(current.generation.id)
    assert listed["generation_digest"] == publication.generation.digest
    assert listed["approved_plan_digest"] == "sha256:mcp-approved-plan"
    assert listed["tasks"] == [
        {
            "generation_id": str(current.generation.id),
            "task_id": str(current.task.id),
            "task_ref": "T-MCP-1",
            "stage": "01-foundation",
            "stem": "01-runtime",
            "position": 1,
            "title": "Project runtime task status",
            "complexity": "small",
            "status": "fix",
            "state_reason": "review requested changes",
            "review_round": 2,
            "attempt_count": 2,
            "duration_seconds": 10.0,
            "cost": {
                "api_spend_usd": 0.75,
                "api_spend_unknown": False,
                "subscription_included": True,
            },
        }
    ]
    assert len(invoked) == 1
    assert selected_stages == [
        AgentStage.CODING,
        AgentStage.REVIEW,
        AgentStage.MERGE,
    ]
    assert len(requests) == 1
    assert "Approve this estimate" in requests[0].message
    invoked_args, invoked_kwargs = invoked[0]
    (
        invoked_paths,
        invoked_store,
        invoked_config,
        repository_id,
        borg_id,
        generation_id,
    ) = invoked_args
    assert invoked_paths == paths
    assert isinstance(invoked_store, SqliteStore)
    assert invoked_config.repository_id == borg.repository_id
    assert (repository_id, borg_id, generation_id) == (
        borg.repository_id,
        borg.id,
        current.generation.id,
    )
    assert invoked_kwargs["cancel"] is not None
    assert invoked_kwargs["progress"] is None
    assert executed["status"] == "completed"
    assert executed["data"]["operation_id"] == str(operation_id)
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        decision = store.get_current_execution_decision(borg.id)
    assert decision is not None
    assert decision.source == "mcp_elicitation"
    assert decision.decision == "approved"


@pytest.mark.parametrize(
    ("tool", "arguments", "command"),
    [
        ("init", {}, "betterborg init"),
        ("analyze", {}, "betterborg analyze"),
        (
            "create",
            {"name": "new-borg", "source": "docs/my prd.md"},
            "betterborg create new-borg --prd 'docs/my prd.md'",
        ),
        (
            "plan",
            {"name": "new-borg", "action": "start"},
            "betterborg plan start new-borg",
        ),
        (
            "plan",
            {
                "name": "new-borg",
                "action": "change",
                "note": "ship safely",
            },
            "betterborg plan change new-borg --note 'ship safely'",
        ),
        (
            "plan",
            {"name": "new-borg", "action": "approve"},
            "betterborg plan approve new-borg",
        ),
        (
            "execute",
            {"name": "new-borg"},
            "betterborg execute new-borg",
        ),
    ],
)
def test_interactive_tools_require_elicitation_before_workflow_mutation(
    tool: str,
    arguments: dict,
    command: str,
    monkeypatch,
) -> None:
    touched: list[str] = []

    def unexpected_paths(**_kwargs):
        touched.append("paths")
        raise AssertionError("workflow reached before capability gate")

    def unexpected_io(*_args, **_kwargs):
        touched.append("prompt")
        raise AssertionError("interactive IO constructed without elicitation")

    monkeypatch.setattr(mcp_server, "_paths", unexpected_paths)
    monkeypatch.setattr(mcp_server.McpInteractiveIO, "__init__", unexpected_io)

    result = _structured(
        _call_tool(tool, arguments, elicitation=False)
    )

    assert result == {
        "status": "setup_required",
        "artifacts": [],
        "next_actions": [],
        "data": {"cli_command": command},
    }
    assert touched == []
    serialized = json.dumps(result)
    assert "continuation" not in serialized
    assert "resume" not in serialized


def test_workspace_trust_confirmation_uses_elicitation(
    committed_git_repo: Path,
    monkeypatch,
) -> None:
    paths = RepoPaths.discover(committed_git_repo)
    paths.state_dir.mkdir(parents=True)
    repository = SimpleNamespace(id=uuid4())
    analysis = SimpleNamespace(id=uuid4(), overall_score=4.0)

    class FakeRepositoryService:
        def __init__(self, service_paths, _store, _factory, *, cancel) -> None:
            assert service_paths == paths
            assert cancel is not None

        def initialize(self):
            return SimpleNamespace(
                initialized=False,
                repository=repository,
                analysis=analysis,
                score_path=paths.score_report,
                prompts=(),
                improvement_prds=(),
            )

    state_home = committed_git_repo.parent / "mcp-machine-state"
    monkeypatch.chdir(committed_git_repo)
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
    monkeypatch.setattr(mcp_server, "RepositoryService", FakeRepositoryService)
    requests: list = []

    result = _structured(_call_tool("init", requests=requests))

    identity = WorkspaceIdentity.discover(paths)
    assert result["status"] == "already_initialized"
    assert TrustStore().is_trusted(identity)
    assert len(requests) == 1
    assert "host-capable agents may read and modify files" in requests[0].message
    assert requests[0].requestedSchema["properties"]["approved"]["default"] is False


def test_init_elicits_onboarding_door_theme_and_name(
    committed_git_repo: Path,
    planning_cli_repository,
    monkeypatch,
) -> None:
    repository, paths = planning_cli_repository(committed_git_repo, "existing")
    score = paths.score_report
    theme = SimpleNamespace(
        title="Repair CI",
        predicted_impact=1.5,
        effort="small",
        suggested_borg_name="repair-ci",
        path=paths.improvement_prds_dir / "repair-ci.md",
    )
    analysis = SimpleNamespace(id=uuid4(), overall_score=3.5)
    created = Borg(repository_id=repository.id, name="repair-ci")
    selected_stages: list[AgentStage] = []

    class FakeRepositoryService:
        def __init__(self, service_paths, _store, factory, *, cancel) -> None:
            assert service_paths == paths
            assert cancel is not None
            factory(object())

        def initialize(self):
            return SimpleNamespace(
                initialized=True,
                repository=repository,
                analysis=analysis,
                score_path=score,
                prompts=(),
                improvement_prds=(theme,),
            )

    class FakeCreateService:
        def __init__(
            self,
            _repository,
            _store,
            _agent,
            *,
            io,
            interactive,
            cancel,
        ):
            assert io is not None
            assert interactive is True
            assert cancel is not None

        def create(self, name, source):
            assert (name, source) == ("repair-ci", theme.path)
            return SimpleNamespace(
                borg=created,
                confirmed=True,
                questions=(),
                body_md="# Repair CI\n",
                prd_path=paths.tracked_dir / "prds" / "repair-ci.md",
            )

    monkeypatch.chdir(committed_git_repo)
    monkeypatch.setattr(
        mcp_server,
        "_paths",
        lambda *, trusted, io=None, cancel=None: paths,
    )
    monkeypatch.setattr(mcp_server, "RepositoryService", FakeRepositoryService)
    monkeypatch.setattr(mcp_server, "CreateService", FakeCreateService)
    monkeypatch.setattr(
        mcp_server,
        "select_agent",
        lambda _config, stage, _paths, **_kwargs: selected_stages.append(stage),
    )
    requests: list = []

    result = _structured(
        _call_tool(
            "init",
            answers=("1", "1", ""),
            requests=requests,
        )
    )

    assert result["status"] == "initialized"
    assert selected_stages == [AgentStage.ANALYSIS, AgentStage.REQUIREMENTS]
    assert result["artifacts"][-1] == {
        "kind": "prd",
        "path": ".betterborg/prds/repair-ci.md",
    }
    assert result["next_actions"] == [
        {"tool": "plan", "arguments": {"name": "repair-ci", "action": "start"}}
    ]
    assert len(requests) == 3
    assert "Choose a door" in requests[0].message
    assert "Choose a theme" in requests[1].message
    assert "Borg name [repair-ci]" in requests[2].message


def test_create_elicits_prd_question_and_final_confirmation(
    committed_git_repo: Path,
    planning_cli_repository,
    monkeypatch,
) -> None:
    repository, paths = planning_cli_repository(committed_git_repo, "existing")
    adapter = MockAdapter(name="openai").queue(
        MockResponse(
            payload={
                "questions": ["Which operating systems must be supported?"],
                "prd_markdown": None,
            }
        )
    ).queue(
        MockResponse(
            payload={
                "questions": [],
                "prd_markdown": "# Portable CLI\n\nSupport all required systems.\n",
            }
        )
    )
    monkeypatch.chdir(committed_git_repo)
    monkeypatch.setattr(
        mcp_server,
        "_paths",
        lambda *, trusted, io=None, cancel=None: paths,
    )
    monkeypatch.setattr(mcp_server, "select_agent", lambda *_args, **_kwargs: adapter)
    requests: list = []

    result = _structured(
        _call_tool(
            "create",
            {"name": "portable-cli"},
            answers=("Linux, macOS, and Windows.",),
            requests=requests,
        )
    )

    assert result["status"] == "confirmed"
    assert paths.tracked_dir.joinpath("prds", "portable-cli.md").is_file()
    assert len(requests) == 2
    assert "Which operating systems" in requests[0].message
    assert "# Portable CLI" in requests[1].message
    assert "Create Borg 'portable-cli'" in requests[1].message
    assert requests[1].requestedSchema["properties"]["approved"]["default"] is False


def test_stdio_stdout_contains_only_protocol_json() -> None:
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "1"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ]

    process = subprocess.Popen(
        [str(Path(sys.executable).with_name("betterborg")), "mcp"],
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    responses = []
    try:
        process.stdin.write(json.dumps(messages[0]) + "\n")
        process.stdin.flush()
        ready, _, _ = select.select([process.stdout], [], [], 5)
        assert ready, "MCP server did not answer initialize"
        responses.append(json.loads(process.stdout.readline()))

        process.stdin.write(json.dumps(messages[1]) + "\n")
        process.stdin.write(json.dumps(messages[2]) + "\n")
        process.stdin.flush()
        ready, _, _ = select.select([process.stdout], [], [], 5)
        assert ready, "MCP server did not answer tools/list"
        responses.append(json.loads(process.stdout.readline()))
        process.stdin.close()
        returncode = process.wait(timeout=5)
        stderr = process.stderr.read()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    assert returncode == 0, stderr
    assert [response["id"] for response in responses] == [1, 2]
    assert all(response["jsonrpc"] == "2.0" for response in responses)
    assert "Processing request" not in "\n".join(map(json.dumps, responses))
    assert "Processing request" in stderr


def test_stdio_stdout_contains_only_protocol_json_for_api_backed_tool(
    tmp_path: Path,
) -> None:
    provider_response = openai_response(
        [
            openai_function_call(
                "submit_result",
                {"status": "completed", "version": "stdio-clean"},
                call_id="submit",
            )
        ]
    )

    def respond(_request):
        return (
            200,
            {"content-type": "application/json"},
            json.dumps(provider_response).encode(),
        )

    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {"elicitation": {}},
                "clientInfo": {"name": "pytest", "version": "1"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "analyze", "arguments": {}},
        },
    ]
    server_script = r'''
from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

from rich.console import Console

from betterborg_cli import mcp_server
from betterborg_cli.agent_runtime import (
    AgentRunSpec,
    AgentStatus,
    OpenAIAdapter,
    UrllibOpenAITransport,
)

root = Path(sys.argv[2])
Console


def analyze(_io, *, cancel):
    result = OpenAIAdapter(
        "analysis",
        api_key="stdio-key",
        transport=UrllibOpenAITransport(sys.argv[1]),
    ).run(
        AgentRunSpec(
            system_prompt="Return a structured result.",
            user_prompt="Submit the result.",
            schema={
                "type": "object",
                "required": ["status", "version"],
                "properties": {
                    "status": {"const": "completed"},
                    "version": {"type": "string"},
                },
                "additionalProperties": False,
            },
            cwd=root,
            model="gpt-test",
            log_path=root / "stdio-provider.jsonl",
            result_path=root / "stdio-result.json",
        ),
        cancel=cancel,
    )
    assert result.status is AgentStatus.COMPLETED
    return mcp_server.AnalyzeResult(
        status="completed",
        data=mcp_server.AnalyzeData(
            repository_id=uuid4(),
            analysis_id=uuid4(),
            score=5,
            previous_score=4,
            delta=1,
        ),
    )


mcp_server._analyze = analyze
mcp_server.run_stdio_server()
'''

    with LocalHttpServer(respond) as server:
        process = subprocess.Popen(
            [sys.executable, "-c", server_script, server.url(), str(tmp_path)],
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        protocol_lines: list[str] = []
        try:
            process.stdin.write(json.dumps(messages[0]) + "\n")
            process.stdin.flush()
            ready, _, _ = select.select([process.stdout], [], [], 5)
            assert ready, "MCP server did not answer initialize"
            protocol_lines.append(process.stdout.readline())

            process.stdin.write(json.dumps(messages[1]) + "\n")
            for message in messages[2:]:
                process.stdin.write(json.dumps(message) + "\n")
                process.stdin.flush()
                ready, _, _ = select.select([process.stdout], [], [], 5)
                assert ready, f"MCP server did not answer request {message['id']}"
                protocol_lines.append(process.stdout.readline())
            process.stdin.close()
            returncode = process.wait(timeout=5)
            protocol_lines.extend(process.stdout.readlines())
            stderr = process.stderr.read()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    assert returncode == 0, stderr
    raw_stdout = "".join(protocol_lines)
    assert "\x1b" not in raw_stdout
    responses = [json.loads(line) for line in protocol_lines if line.strip()]
    assert [response["id"] for response in responses] == [1, 2, 3]
    assert all(response["jsonrpc"] == "2.0" for response in responses)
    assert responses[-1]["result"]["structuredContent"]["status"] == "completed"
    assert "Processing request" not in "\n".join(map(json.dumps, responses))
    assert "Processing request" in stderr
