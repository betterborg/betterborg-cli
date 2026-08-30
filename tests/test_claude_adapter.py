"""Claude CLI behavior over the shared adapter run contract."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from test_adapter_harness import (
    claude_spec,
    native_event_stream,
    write_native_output,
)

from betterborg_cli.agent_runtime import (
    AgentActivity,
    AgentActivityKind,
    AgentArtifact,
    AgentStatus,
    AgentUsage,
    ApiAgentRole,
    BillingMode,
    CancellationToken,
    ClaudeAdapter,
)


def _envelope(
    result: Mapping[str, Any] | str,
    *,
    is_error: bool = False,
    api_error_status: int | None = None,
    usage: Mapping[str, Any] | None = None,
    cost: float | None = None,
    turns: int | None = None,
) -> str:
    if not isinstance(result, str):
        result = json.dumps(result)
    payload: dict[str, Any] = {
        "type": "result",
        "subtype": "success",
        "result": result,
    }
    if is_error:
        payload["is_error"] = True
    if api_error_status is not None:
        payload["api_error_status"] = api_error_status
    if usage is not None:
        payload["usage"] = dict(usage)
    if cost is not None:
        payload["total_cost_usd"] = cost
    if turns is not None:
        payload["num_turns"] = turns
    return json.dumps(payload)


def _tool_event(name: str, tool_input: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "assistant",
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "name": name,
                    "input": dict(tool_input),
                }
            ]
        },
    }


@pytest.mark.parametrize("role", list(ApiAgentRole))
def test_every_role_discloses_native_host_capability(
    role: ApiAgentRole,
) -> None:
    adapter = ClaudeAdapter(role, proc_runner=lambda *_args: 0)

    assert adapter.capabilities.host_capable
    assert adapter.capabilities.supports_billing(BillingMode.SUBSCRIPTION)
    assert not adapter.capabilities.supports_billing(BillingMode.API)


@pytest.mark.parametrize(
    ("tool_name", "tool_input", "expected"),
    (
        (
            "Read",
            {"file_path": "src/betterborg_cli/cli.py"},
            AgentActivity(AgentActivityKind.READING, "src/betterborg_cli/cli.py"),
        ),
        (
            "Glob",
            {"pattern": "tests/test_*.py"},
            AgentActivity(AgentActivityKind.SEARCHING, "tests/test_*.py"),
        ),
        (
            "Grep",
            {"pattern": "NativeInvocation", "path": "src"},
            AgentActivity(AgentActivityKind.SEARCHING, "NativeInvocation"),
        ),
        (
            "Bash",
            {"command": "make test"},
            AgentActivity(AgentActivityKind.COMMAND, "make test"),
        ),
        (
            "Edit",
            {"file_path": "src/betterborg_cli/agent_runtime/claude.py"},
            AgentActivity(
                AgentActivityKind.WRITING,
                "src/betterborg_cli/agent_runtime/claude.py",
            ),
        ),
        (
            "Write",
            {"file_path": "tests/test_claude_adapter.py"},
            AgentActivity(
                AgentActivityKind.WRITING,
                "tests/test_claude_adapter.py",
            ),
        ),
    ),
)
def test_native_tool_events_emit_neutral_activity_without_changing_logs(
    tmp_path: Path,
    tool_name: str,
    tool_input: Mapping[str, Any],
    expected: AgentActivity,
) -> None:
    activities: list[AgentActivity] = []
    transcript = native_event_stream(
        _tool_event(tool_name, tool_input),
        _envelope(
            {"status": "completed", "version": "activity"},
            usage={"input_tokens": 4, "output_tokens": 2},
        ),
    )

    def runner(
        _command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
        on_line: Callable[[str], None] | None,
    ) -> int:
        write_native_output(log_path, transcript, on_line)
        return 0

    spec = claude_spec(tmp_path, activity_sink=activities.append)
    result = ClaudeAdapter(ApiAgentRole.CODING, proc_runner=runner).run(spec)

    assert result.status == AgentStatus.COMPLETED
    assert result.usage == AgentUsage(tokens_input=4, tokens_output=2)
    assert spec.log_path.read_text(encoding="utf-8") == transcript
    assert activities == [
        AgentActivity(AgentActivityKind.THINKING),
        expected,
        AgentActivity(AgentActivityKind.THINKING),
    ]


def test_fragmented_large_tool_event_emits_activity_without_changing_log(
    tmp_path: Path,
) -> None:
    activities: list[AgentActivity] = []
    transcript = native_event_stream(
        _tool_event(
            "Write",
            {
                "file_path": "large-output.txt",
                "content": "x" * (64 * 1024),
            },
        ),
        _envelope({"status": "completed", "version": "fragmented"}),
    )
    process_chunk_size = 64 * 1024

    def runner(
        _command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
        on_line: Callable[[str], None] | None,
    ) -> int:
        log_path.write_text(transcript, encoding="utf-8")
        assert on_line is not None
        for offset in range(0, len(transcript), process_chunk_size):
            on_line(transcript[offset : offset + process_chunk_size])
        return 0

    spec = claude_spec(tmp_path, activity_sink=activities.append)
    result = ClaudeAdapter(ApiAgentRole.CODING, proc_runner=runner).run(spec)

    assert result.status == AgentStatus.COMPLETED
    assert result.payload == {"status": "completed", "version": "fragmented"}
    assert spec.log_path.read_text(encoding="utf-8") == transcript
    assert activities == [
        AgentActivity(AgentActivityKind.THINKING),
        AgentActivity(AgentActivityKind.WRITING, "large-output.txt"),
        AgentActivity(AgentActivityKind.THINKING),
    ]


def test_unknown_malformed_result_and_usage_events_fall_back_to_thinking(
    tmp_path: Path,
) -> None:
    activities: list[AgentActivity] = []
    transcript = native_event_stream(
        "not-json",
        _tool_event("UnknownProviderTool", {"provider_detail": "secret"}),
        _tool_event("Read", {"path": "missing-file-path"}),
        {"type": "usage", "usage": {"input_tokens": 999}},
        _envelope({"status": "completed", "version": "fallback"}),
    )

    def runner(
        _command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
        on_line: Callable[[str], None] | None,
    ) -> int:
        write_native_output(log_path, transcript, on_line)
        return 0

    spec = claude_spec(tmp_path, activity_sink=activities.append)
    result = ClaudeAdapter(ApiAgentRole.ANALYSIS, proc_runner=runner).run(spec)

    assert result.status == AgentStatus.COMPLETED
    assert result.payload == {"status": "completed", "version": "fallback"}
    assert spec.log_path.read_text(encoding="utf-8") == transcript
    assert activities == [
        AgentActivity(AgentActivityKind.THINKING)
    ] * 6


def test_activity_callback_failure_does_not_change_native_result(
    tmp_path: Path,
) -> None:
    callback_calls = 0
    transcript = native_event_stream(
        _tool_event("Read", {"file_path": "README.md"}),
        _envelope({"status": "completed", "version": "callback"}),
    )

    def fail_activity(_activity: AgentActivity) -> None:
        nonlocal callback_calls
        callback_calls += 1
        raise RuntimeError("progress renderer failed")

    def runner(
        _command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
        on_line: Callable[[str], None] | None,
    ) -> int:
        write_native_output(log_path, transcript, on_line)
        return 0

    spec = claude_spec(tmp_path, activity_sink=fail_activity)
    result = ClaudeAdapter(ApiAgentRole.ANALYSIS, proc_runner=runner).run(spec)

    assert result.status == AgentStatus.COMPLETED
    assert result.payload == {"status": "completed", "version": "callback"}
    assert spec.log_path.read_text(encoding="utf-8") == transcript
    assert callback_calls == 3


def test_native_command_validates_and_persists_result_metadata(
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    artifact = AgentArtifact("artifact://claude/transcript", kind="transcript")

    def runner(
        command: Sequence[str],
        cwd: Path,
        stdin_text: str,
        log_path: Path,
        cancel: CancellationToken | None,
        env: Mapping[str, str] | None,
        on_line: Callable[[str], None] | None,
    ) -> int:
        prompt_path = Path(command[-1])
        captured.update(
            command=list(command),
            cwd=cwd,
            stdin=stdin_text,
            env=env,
            system_prompt=prompt_path.read_text(encoding="utf-8"),
            prompt_path=prompt_path,
        )
        write_native_output(
            log_path,
            native_event_stream(
                {"type": "system", "subtype": "init", "session_id": "test"},
                {"type": "assistant", "message": {"content": [{"type": "text"}]}},
                _envelope(
                    {"status": "completed", "version": "1.2.3"},
                    usage={
                        "input_tokens": 20,
                        "output_tokens": 5,
                        "cache_read_input_tokens": 7,
                        "cache_creation_input_tokens": 2,
                    },
                    cost=0.25,
                    turns=3,
                ),
            ),
            on_line,
        )
        return 0

    adapter = ClaudeAdapter(
        ApiAgentRole.ANALYSIS,
        binary="claude-custom",
        proc_runner=runner,
        artifacts=(artifact,),
    )
    spec = claude_spec(
        tmp_path,
        model="claude-model-override",
        effort="high",
        allowed_tools=("Read", "Edit"),
        env={"BETTERBORG_TEST_ENV": "present"},
    )

    result = adapter.run(spec)

    command = captured["command"]
    system_prompt = captured["system_prompt"]
    assert command == [
        "claude-custom",
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        "claude-model-override",
        "--effort",
        "high",
        "--dangerously-skip-permissions",
        "--allowed-tools",
        "Read,Edit",
        "--system-prompt-file",
        str(captured["prompt_path"]),
    ]
    assert captured["cwd"] == tmp_path
    assert captured["env"]["BETTERBORG_TEST_ENV"] == "present"
    assert "JSON Schema" in system_prompt
    assert '"version"' in system_prompt
    assert captured["stdin"] == spec.user_prompt
    assert not captured["prompt_path"].exists()
    assert result.status == AgentStatus.COMPLETED
    assert result.provider == "claude"
    assert result.model == "claude-model-override"
    assert result.billing_mode == BillingMode.SUBSCRIPTION
    assert result.artifacts == (artifact,)
    assert result.usage == AgentUsage(
        cost_usd=0.25,
        tokens_input=20,
        tokens_output=5,
        tokens_cache_read=7,
        tokens_cache_write=2,
        num_turns=3,
    )
    assert json.loads(spec.result_path.read_text(encoding="utf-8")) == result.payload


def test_read_only_tool_allowlist_uses_plan_mode(tmp_path: Path) -> None:
    captured_command: list[str] = []

    def runner(
        command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
        on_line: Callable[[str], None] | None,
    ) -> int:
        captured_command.extend(command)
        write_native_output(
            log_path,
            _envelope({"status": "completed", "version": "1.2.3"}),
            on_line,
        )
        return 0

    spec = claude_spec(
        tmp_path,
        allowed_tools=("list_files", "read_file", "search_text"),
    )

    result = ClaudeAdapter(ApiAgentRole.PLANNING, proc_runner=runner).run(spec)

    assert result.status == AgentStatus.COMPLETED
    assert "--dangerously-skip-permissions" not in captured_command
    assert captured_command[
        captured_command.index("--permission-mode") + 1
    ] == "plan"
    assert captured_command[
        captured_command.index("--allowed-tools") + 1
    ] == "Glob,Read,Grep"


def test_schema_invalid_result_fails_without_persisting_result(
    tmp_path: Path,
) -> None:
    def runner(
        _command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
        on_line: Callable[[str], None] | None,
    ) -> int:
        write_native_output(
            log_path,
            _envelope({"status": "completed"}),
            on_line,
        )
        return 0

    spec = claude_spec(tmp_path)
    result = ClaudeAdapter(ApiAgentRole.ANALYSIS, proc_runner=runner).run(spec)

    assert result.status == AgentStatus.FAILED
    assert "missing required property 'version'" in (result.error or "")
    assert not spec.result_path.exists()


def test_cancellation_is_forwarded_and_preserves_artifacts(tmp_path: Path) -> None:
    cancel = CancellationToken()
    artifact = AgentArtifact(tmp_path / "partial.log", kind="log")

    def runner(
        _command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        _log_path: Path,
        runner_cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
        _on_line: Callable[[str], None] | None,
    ) -> int:
        assert runner_cancel is cancel
        cancel.cancel()
        return -1

    result = ClaudeAdapter(
        ApiAgentRole.CODING,
        proc_runner=runner,
        artifacts=(artifact,),
    ).run(claude_spec(tmp_path), cancel=cancel)

    assert result.status == AgentStatus.CANCELLED
    assert result.retryable and result.resumable
    assert result.artifacts == (artifact,)
    assert result.attempts == 1


def test_transient_error_retries_same_native_invocation(tmp_path: Path) -> None:
    calls: list[tuple[list[str], str]] = []
    activities: list[AgentActivity] = []
    responses = [
        _envelope(
            "model overloaded",
            is_error=True,
            api_error_status=529,
            usage={"input_tokens": 2},
        ),
        _envelope(
            {"status": "completed", "version": "retry"},
            usage={"input_tokens": 3},
        ),
    ]

    def runner(
        command: Sequence[str],
        _cwd: Path,
        stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
        on_line: Callable[[str], None] | None,
    ) -> int:
        calls.append((list(command), stdin_text))
        write_native_output(log_path, responses.pop(0), on_line)
        return 0

    result = ClaudeAdapter(
        ApiAgentRole.PLANNING,
        proc_runner=runner,
        transient_backoff_seconds=0,
    ).run(claude_spec(tmp_path, activity_sink=activities.append))

    assert result.status == AgentStatus.COMPLETED
    assert result.attempts == 2
    assert result.usage == AgentUsage(tokens_input=5)
    assert calls[0] == calls[1]
    assert activities == [
        AgentActivity(AgentActivityKind.THINKING)
    ] * 4


def test_process_spawn_failure_returns_failed_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_spawn(*_args: Any, **_kwargs: Any) -> None:
        raise FileNotFoundError("configured executable disappeared")

    monkeypatch.setattr(
        "betterborg_cli.agent_runtime.native_cli.shutil.which",
        lambda _binary: "/installed/claude-custom",
    )
    monkeypatch.setattr(
        "betterborg_cli.agent_runtime.process.subprocess.Popen",
        fail_spawn,
    )

    result = ClaudeAdapter(ApiAgentRole.CODING, binary="claude-custom").run(
        claude_spec(tmp_path)
    )

    assert result.status == AgentStatus.FAILED
    assert result.attempts == 1
    assert result.exit_code is None
    assert "unable to start Claude process" in (result.error or "")
    assert "configured executable disappeared" in (result.error or "")


def test_transient_exhaustion_is_resumable(tmp_path: Path) -> None:
    def runner(
        _command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
        on_line: Callable[[str], None] | None,
    ) -> int:
        write_native_output(
            log_path,
            _envelope("rate limited", is_error=True, api_error_status=429),
            on_line,
        )
        return 1

    result = ClaudeAdapter(
        ApiAgentRole.REVIEW,
        proc_runner=runner,
        transient_backoff_seconds=0,
        transient_max_attempts=2,
    ).run(claude_spec(tmp_path, resume_token="session-123"))

    assert result.status == AgentStatus.CANCELLED
    assert result.retryable and result.resumable
    assert result.resume_token == "session-123"
    assert result.attempts == 2
    assert "transient retry exhausted" in (result.error or "")


def test_stream_json_and_prose_wrapped_result_are_supported(
    tmp_path: Path,
) -> None:
    transcript = native_event_stream(
        {"type": "system", "subtype": "init"},
        {"type": "assistant", "message": {"content": []}},
        _envelope(
            'Result:\n```json\n{"status":"completed","version":"2.x"}\n```'
        ),
    )

    def runner(
        _command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
        on_line: Callable[[str], None] | None,
    ) -> int:
        write_native_output(log_path, transcript, on_line)
        return 0

    result = ClaudeAdapter(ApiAgentRole.ANALYSIS, proc_runner=runner).run(
        claude_spec(tmp_path)
    )

    assert result.status == AgentStatus.COMPLETED
    assert result.payload == {"status": "completed", "version": "2.x"}
