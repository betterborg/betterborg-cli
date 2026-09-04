"""Codex CLI behavior over the shared adapter run contract."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest
from test_adapter_harness import (
    codex_spec,
    native_event_stream,
    redacting_execution_activity_sink,
    write_native_output,
)

import betterborg_cli.agent_runtime.retry as agent_retry
from betterborg_cli.agent_runtime import (
    DEFAULT_SCHEMA_MAX_ATTEMPTS,
    AgentActivity,
    AgentActivityKind,
    AgentArtifact,
    AgentStatus,
    AgentUsage,
    ApiAgentRole,
    BillingMode,
    CancellationToken,
    CodexAdapter,
)


def _usage_event(
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    reasoning_output_tokens: int = 0,
    cache_write_input_tokens: int = 0,
) -> str:
    return json.dumps(
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": input_tokens,
                "cached_input_tokens": cached_input_tokens,
                "output_tokens": output_tokens,
                "reasoning_output_tokens": reasoning_output_tokens,
                "cache_write_input_tokens": cache_write_input_tokens,
            },
        }
    )


def _write_invocation_result(command: Sequence[str], payload: Any) -> None:
    result_path = Path(command[command.index("-o") + 1])
    result_path.write_text(json.dumps(payload), encoding="utf-8")


def _item_event(
    event_type: str,
    item_type: str,
    **item: Any,
) -> dict[str, Any]:
    return {"type": event_type, "item": {"type": item_type, **item}}


@pytest.mark.parametrize("role", list(ApiAgentRole))
def test_every_role_discloses_native_host_capability(role: ApiAgentRole) -> None:
    adapter = CodexAdapter(role, proc_runner=lambda *_args: 0)

    assert adapter.capabilities.host_capable
    assert adapter.capabilities.supports_billing(BillingMode.SUBSCRIPTION)
    assert not adapter.capabilities.supports_billing(BillingMode.API)
    assert not adapter.capabilities.tool_allowlist
    assert adapter.capabilities.read_only_sandbox


@pytest.mark.parametrize(
    ("event_type", "item_type", "item", "expected"),
    (
        (
            "item.started",
            "command_execution",
            {"command": "cat README.md"},
            AgentActivity(AgentActivityKind.READING, "cat README.md"),
        ),
        (
            "item.completed",
            "command_execution",
            {"command": "/bin/bash -lc 'rg NativeInvocation src'"},
            AgentActivity(
                AgentActivityKind.SEARCHING,
                "/bin/bash -lc 'rg NativeInvocation src'",
            ),
        ),
        (
            "item.started",
            "command_execution",
            {"command": "make test"},
            AgentActivity(AgentActivityKind.COMMAND, "make test"),
        ),
        (
            "item.completed",
            "file_change",
            {"changes": [{"path": "src/betterborg_cli/agent_runtime/codex.py"}]},
            AgentActivity(
                AgentActivityKind.WRITING,
                "src/betterborg_cli/agent_runtime/codex.py",
            ),
        ),
        (
            "item.started",
            "web_search",
            {"query": "Codex JSONL events"},
            AgentActivity(AgentActivityKind.SEARCHING, "Codex JSONL events"),
        ),
        (
            "item.started",
            "mcp_tool_call",
            {"tool": "filesystem.read_file", "arguments": {"path": "pyproject.toml"}},
            AgentActivity(AgentActivityKind.READING, "pyproject.toml"),
        ),
        (
            "item.completed",
            "mcp_tool_call",
            {
                "tool": "filesystem.search_files",
                "arguments": {"pattern": "test_*.py"},
            },
            AgentActivity(AgentActivityKind.SEARCHING, "test_*.py"),
        ),
        (
            "item.started",
            "mcp_tool_call",
            {"tool": "filesystem.write_file", "arguments": {"path": "result.json"}},
            AgentActivity(AgentActivityKind.WRITING, "result.json"),
        ),
    ),
)
def test_native_item_events_emit_neutral_activity_without_changing_logs(
    tmp_path: Path,
    event_type: str,
    item_type: str,
    item: Mapping[str, Any],
    expected: AgentActivity,
) -> None:
    activities: list[AgentActivity] = []
    transcript = native_event_stream(
        _item_event(event_type, item_type, **item),
        _usage_event(10, 4, 3),
    )

    def runner(
        command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
        on_line: Callable[[str], None] | None,
    ) -> int:
        write_native_output(log_path, transcript, on_line)
        _write_invocation_result(
            command, {"status": "completed", "version": "activity"}
        )
        return 0

    spec = codex_spec(tmp_path, activity_sink=activities.append)
    result = CodexAdapter(ApiAgentRole.CODING, proc_runner=runner).run(spec)

    assert result.status == AgentStatus.COMPLETED
    assert result.usage == AgentUsage(
        tokens_input=6,
        tokens_output=3,
        tokens_cache_read=4,
        tokens_cache_write=0,
        num_turns=1,
    )
    assert spec.log_path.read_text(encoding="utf-8") == transcript
    assert activities == [
        AgentActivity(AgentActivityKind.THINKING),
        expected,
        AgentActivity(AgentActivityKind.THINKING),
    ]


def test_generic_command_activity_is_single_line_and_bounded(tmp_path: Path) -> None:
    activities: list[AgentActivity] = []
    command_text = "python -c '" + ("x" * 200) + "'\nwith another line"
    transcript = json.dumps(
        _item_event("item.started", "command_execution", command=command_text)
    )

    def runner(
        command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
        on_line: Callable[[str], None] | None,
    ) -> int:
        write_native_output(log_path, transcript, on_line)
        _write_invocation_result(
            command, {"status": "completed", "version": "bounded"}
        )
        return 0

    result = CodexAdapter(ApiAgentRole.CODING, proc_runner=runner).run(
        codex_spec(tmp_path, activity_sink=activities.append)
    )

    assert result.status == AgentStatus.COMPLETED
    command_activity = activities[1]
    assert command_activity.kind is AgentActivityKind.COMMAND
    assert command_activity.detail is not None
    assert len(command_activity.detail) == 160
    assert "\n" not in command_activity.detail
    assert command_activity.detail.endswith("…")


def test_native_activity_uses_execution_secret_redaction_without_changing_result(
    tmp_path: Path,
) -> None:
    secret = 'native"secret/slash?x=1'
    escaped = json.dumps(secret)[1:-1]
    encoded = quote(secret, safe="")
    detail = f"{secret} {escaped} {encoded}"
    activities: list[AgentActivity] = []
    transcript = native_event_stream(
        _item_event("item.started", "web_search", query=detail),
        _usage_event(3, 1, 2),
    )

    def runner(
        command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
        on_line: Callable[[str], None] | None,
    ) -> int:
        write_native_output(log_path, transcript, on_line)
        _write_invocation_result(
            command, {"status": "completed", "version": "unchanged"}
        )
        return 0

    result = CodexAdapter(ApiAgentRole.CODING, proc_runner=runner).run(
        codex_spec(
            tmp_path,
            activity_sink=redacting_execution_activity_sink(secret, activities),
        )
    )

    assert result.status is AgentStatus.COMPLETED
    assert result.payload == {"status": "completed", "version": "unchanged"}
    assert activities[1].detail == "[REDACTED] [REDACTED] [REDACTED]"
    assert all(value not in repr(activities) for value in (secret, escaped, encoded))


def test_unknown_malformed_result_and_usage_events_fall_back_to_thinking(
    tmp_path: Path,
) -> None:
    activities: list[AgentActivity] = []
    transcript = native_event_stream(
        "not-json",
        {"type": "item.started", "item": "malformed"},
        _item_event("item.started", "unknown_provider_item", provider_name="secret"),
        _item_event(
            "item.started",
            "mcp_tool_call",
            tool="provider_specific_tool",
            arguments={"secret": "value"},
        ),
        _item_event("item.completed", "command_execution", command=42),
        _item_event("item.completed", "agent_message", text="final result"),
        _usage_event(8, 3, 2),
    )

    def runner(
        command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
        on_line: Callable[[str], None] | None,
    ) -> int:
        write_native_output(log_path, transcript, on_line)
        _write_invocation_result(
            command, {"status": "completed", "version": "fallback"}
        )
        return 0

    spec = codex_spec(tmp_path, activity_sink=activities.append)
    result = CodexAdapter(ApiAgentRole.ANALYSIS, proc_runner=runner).run(spec)

    assert result.status == AgentStatus.COMPLETED
    assert result.payload == {"status": "completed", "version": "fallback"}
    assert result.usage == AgentUsage(
        tokens_input=5,
        tokens_output=2,
        tokens_cache_read=3,
        tokens_cache_write=0,
        num_turns=1,
    )
    assert spec.log_path.read_text(encoding="utf-8") == transcript
    assert activities == [AgentActivity(AgentActivityKind.THINKING)] * 8


def test_native_command_validates_and_persists_result_metadata(
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    artifact = AgentArtifact("artifact://codex/transcript", kind="transcript")

    def runner(
        command: Sequence[str],
        cwd: Path,
        stdin_text: str,
        log_path: Path,
        cancel: CancellationToken | None,
        env: Mapping[str, str] | None,
        _on_line: Callable[[str], None] | None,
    ) -> int:
        schema_path = Path(command[command.index("--output-schema") + 1])
        invocation_result_path = Path(command[command.index("-o") + 1])
        captured.update(
            command=list(command),
            cwd=cwd,
            stdin=stdin_text,
            env=env,
            schema=json.loads(schema_path.read_text(encoding="utf-8")),
            schema_path=schema_path,
            invocation_result_path=invocation_result_path,
            cancel=cancel,
        )
        log_path.write_text(
            "\n".join(
                (
                    json.dumps({"type": "thread.started", "thread_id": "test"}),
                    _usage_event(20, 7, 5, 2, 4),
                    _usage_event(18, 10, 3, cache_write_input_tokens=6),
                )
            ),
            encoding="utf-8",
        )
        _write_invocation_result(
            command, {"status": "completed", "version": "1.2.3"}
        )
        return 0

    adapter = CodexAdapter(
        ApiAgentRole.ANALYSIS,
        binary="codex-custom",
        proc_runner=runner,
        artifacts=(artifact,),
    )
    spec = codex_spec(
        tmp_path,
        model="gpt-model-override",
        effort="high",
        env={"BETTERBORG_TEST_ENV": "present"},
    )

    result = adapter.run(spec)

    command = captured["command"]
    assert command == [
        "codex-custom",
        "exec",
        "--json",
        "-C",
        str(tmp_path),
        "-m",
        "gpt-model-override",
        "-s",
        "danger-full-access",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--ephemeral",
        "--output-schema",
        str(captured["schema_path"]),
        "-o",
        str(captured["invocation_result_path"]),
        "-c",
        "model_reasoning_effort=high",
        "-",
    ]
    assert captured["cwd"] == tmp_path
    assert captured["env"]["BETTERBORG_TEST_ENV"] == "present"
    assert captured["schema"] == spec.schema
    assert captured["stdin"].startswith(spec.system_prompt)
    assert f"<stdin>\n{spec.user_prompt}\n</stdin>" in captured["stdin"]
    assert not captured["schema_path"].exists()
    assert not captured["invocation_result_path"].exists()
    assert result.status == AgentStatus.COMPLETED
    assert result.provider == "codex"
    assert result.model == "gpt-model-override"
    assert result.billing_mode == BillingMode.SUBSCRIPTION
    assert result.artifacts == (artifact,)
    assert result.usage == AgentUsage(
        tokens_input=11,
        tokens_output=8,
        tokens_cache_read=17,
        tokens_cache_write=10,
        num_turns=2,
    )
    assert json.loads(spec.result_path.read_text(encoding="utf-8")) == result.payload


def test_read_only_tool_allowlist_uses_read_only_sandbox(tmp_path: Path) -> None:
    captured_command: list[str] = []

    def runner(
        command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
        _on_line: Callable[[str], None] | None,
    ) -> int:
        captured_command.extend(command)
        log_path.write_text("{}\n", encoding="utf-8")
        _write_invocation_result(
            command, {"status": "completed", "version": "1.2.3"}
        )
        return 0

    spec = codex_spec(
        tmp_path,
        allowed_tools=("list_files", "read_file", "search_text"),
    )

    result = CodexAdapter(ApiAgentRole.PLANNING, proc_runner=runner).run(spec)

    assert result.status == AgentStatus.COMPLETED
    assert captured_command[captured_command.index("-s") + 1] == "read-only"


def test_schema_invalid_result_fails_after_bounded_attempts(
    tmp_path: Path,
) -> None:
    calls = 0

    def runner(
        command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
        _on_line: Callable[[str], None] | None,
    ) -> int:
        nonlocal calls
        calls += 1
        log_path.write_text("{}\n", encoding="utf-8")
        _write_invocation_result(command, {"status": "completed"})
        return 0

    spec = codex_spec(tmp_path)
    result = CodexAdapter(ApiAgentRole.ANALYSIS, proc_runner=runner).run(spec)

    assert result.status == AgentStatus.FAILED
    assert "missing required property 'version'" in (result.error or "")
    assert not spec.result_path.exists()
    assert calls == DEFAULT_SCHEMA_MAX_ATTEMPTS
    assert result.attempts == DEFAULT_SCHEMA_MAX_ATTEMPTS


def test_schema_miss_is_retried_with_the_validating_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits: list[float] = []
    monkeypatch.setattr(agent_retry.time, "sleep", waits.append)
    prompts: list[str] = []
    payloads: list[dict[str, str]] = [
        {"status": "completed"},
        {"status": "completed", "version": "corrected"},
    ]

    def runner(
        command: Sequence[str],
        _cwd: Path,
        stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
        _on_line: Callable[[str], None] | None,
    ) -> int:
        prompts.append(stdin_text)
        log_path.write_text(_usage_event(10, 0, 5), encoding="utf-8")
        _write_invocation_result(command, payloads.pop(0))
        return 0

    spec = codex_spec(tmp_path)
    result = CodexAdapter(ApiAgentRole.PLANNING, proc_runner=runner).run(spec)

    assert result.status == AgentStatus.COMPLETED
    assert result.attempts == 2
    assert result.payload == {"status": "completed", "version": "corrected"}
    assert spec.user_prompt in prompts[0]
    assert prompts[1].startswith(prompts[0])
    assert "missing required property 'version'" in prompts[1]
    assert not waits
    assert json.loads(
        spec.result_path.read_text(encoding="utf-8")
    ) == result.payload


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

    result = CodexAdapter(
        ApiAgentRole.CODING,
        proc_runner=runner,
        artifacts=(artifact,),
    ).run(codex_spec(tmp_path), cancel=cancel)

    assert result.status == AgentStatus.CANCELLED
    assert result.retryable and result.resumable
    assert result.artifacts == (artifact,)
    assert result.attempts == 1


def test_transient_error_retries_and_accumulates_jsonl_usage(
    tmp_path: Path,
) -> None:
    calls: list[tuple[list[str], str]] = []

    def runner(
        command: Sequence[str],
        _cwd: Path,
        stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
        _on_line: Callable[[str], None] | None,
    ) -> int:
        calls.append((list(command), stdin_text))
        if len(calls) == 1:
            log_path.write_text(
                _usage_event(100, 25, 10, 5, 7)
                + "\n"
                + json.dumps(
                    {
                        "type": "error",
                        "message": "status 503: model is overloaded",
                    }
                ),
                encoding="utf-8",
            )
            return 1
        log_path.write_text(
            _usage_event(300, 50, 30, 15, 8), encoding="utf-8"
        )
        _write_invocation_result(
            command, {"status": "completed", "version": "retry"}
        )
        return 0

    result = CodexAdapter(
        ApiAgentRole.PLANNING,
        proc_runner=runner,
        transient_backoff_seconds=0,
    ).run(codex_spec(tmp_path))

    assert result.status == AgentStatus.COMPLETED
    assert result.attempts == 2
    assert result.usage == AgentUsage(
        tokens_input=310,
        tokens_output=40,
        tokens_cache_read=75,
        tokens_cache_write=15,
        num_turns=2,
    )
    assert calls[0] == calls[1]


def test_schema_invalid_partial_result_does_not_suppress_transient_retry(
    tmp_path: Path,
) -> None:
    calls = 0

    def runner(
        command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
        _on_line: Callable[[str], None] | None,
    ) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            log_path.write_text(
                '{"type":"error","message":"status 503: overloaded"}\n',
                encoding="utf-8",
            )
            _write_invocation_result(command, {"status": "completed"})
            return 1
        log_path.write_text("Codex completed retry\n", encoding="utf-8")
        _write_invocation_result(
            command, {"status": "completed", "version": "retry"}
        )
        return 0

    result = CodexAdapter(
        ApiAgentRole.PLANNING,
        proc_runner=runner,
        transient_backoff_seconds=0,
    ).run(codex_spec(tmp_path))

    assert result.status == AgentStatus.COMPLETED
    assert result.attempts == 2
    assert result.payload == {"status": "completed", "version": "retry"}
    assert calls == 2


def test_process_spawn_failure_returns_failed_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_spawn(*_args: Any, **_kwargs: Any) -> None:
        raise FileNotFoundError("configured executable disappeared")

    monkeypatch.setattr(
        "betterborg_cli.agent_runtime.native_cli.shutil.which",
        lambda _binary: "/installed/codex-custom",
    )
    monkeypatch.setattr(
        "betterborg_cli.agent_runtime.process.subprocess.Popen",
        fail_spawn,
    )

    result = CodexAdapter(ApiAgentRole.CODING, binary="codex-custom").run(
        codex_spec(tmp_path)
    )

    assert result.status == AgentStatus.FAILED
    assert result.attempts == 1
    assert result.exit_code is None
    assert "unable to start Codex process" in (result.error or "")
    assert "configured executable disappeared" in (result.error or "")


def test_directory_preparation_failure_returns_failed_result(tmp_path: Path) -> None:
    non_directory = tmp_path / "not-a-directory"
    non_directory.write_text("occupied", encoding="utf-8")
    spec = codex_spec(
        tmp_path,
        log_path=non_directory / "codex.jsonl",
    )

    def runner(*_args: Any) -> int:
        pytest.fail("process runner must not be called when preparation fails")

    result = CodexAdapter(ApiAgentRole.CODING, proc_runner=runner).run(spec)

    assert result.status == AgentStatus.FAILED
    assert result.exit_code is None
    assert result.payload is None
    assert "unable to prepare Codex process" in (result.error or "")
    assert not spec.result_path.exists()


def test_transient_exhaustion_is_resumable(tmp_path: Path) -> None:
    def runner(
        _command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
        _on_line: Callable[[str], None] | None,
    ) -> int:
        log_path.write_text(
            '{"type":"error","message":"status 429: rate limit"}\n',
            encoding="utf-8",
        )
        return 1

    result = CodexAdapter(
        ApiAgentRole.REVIEW,
        proc_runner=runner,
        transient_backoff_seconds=0,
        transient_max_attempts=2,
    ).run(codex_spec(tmp_path, resume_token="thread-123"))

    assert result.status == AgentStatus.CANCELLED
    assert result.retryable and result.resumable
    assert result.resume_token == "thread-123"
    assert result.attempts == 2
    assert "transient retry exhausted" in (result.error or "")


def test_nonzero_exit_with_fresh_valid_result_completes(tmp_path: Path) -> None:
    def runner(
        command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
        _on_line: Callable[[str], None] | None,
    ) -> int:
        log_path.write_text("Codex completed before exiting\n", encoding="utf-8")
        _write_invocation_result(
            command, {"status": "completed", "version": "nonzero"}
        )
        return 2

    spec = codex_spec(tmp_path)
    result = CodexAdapter(ApiAgentRole.MERGE, proc_runner=runner).run(spec)

    assert result.status == AgentStatus.COMPLETED
    assert result.exit_code == 2
    assert result.payload == {"status": "completed", "version": "nonzero"}
    assert result.error is None
    assert json.loads(spec.result_path.read_text(encoding="utf-8")) == result.payload


def test_nonzero_exit_with_invalid_result_still_fails(tmp_path: Path) -> None:
    def runner(
        command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
        _on_line: Callable[[str], None] | None,
    ) -> int:
        log_path.write_text("Codex exited after a partial result\n", encoding="utf-8")
        _write_invocation_result(command, {"status": "completed"})
        return 2

    spec = codex_spec(tmp_path)
    result = CodexAdapter(ApiAgentRole.MERGE, proc_runner=runner).run(spec)

    assert result.status == AgentStatus.FAILED
    assert result.exit_code == 2
    assert result.payload is None
    assert "missing required property 'version'" in (result.error or "")
    assert not spec.result_path.exists()


def test_valid_result_prevents_transient_retry_despite_nonzero_exit(
    tmp_path: Path,
) -> None:
    calls = 0

    def runner(
        command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
        _on_line: Callable[[str], None] | None,
    ) -> int:
        nonlocal calls
        calls += 1
        log_path.write_text(
            '{"type":"error","message":"status 503: overloaded"}\n',
            encoding="utf-8",
        )
        _write_invocation_result(
            command, {"status": "completed", "version": "fresh"}
        )
        return 1

    result = CodexAdapter(
        ApiAgentRole.REVIEW,
        proc_runner=runner,
        transient_backoff_seconds=0,
    ).run(codex_spec(tmp_path))

    assert result.status == AgentStatus.COMPLETED
    assert result.attempts == 1
    assert result.exit_code == 1
    assert calls == 1


def test_optional_schema_fields_are_normalized_for_strict_transport(
    tmp_path: Path,
) -> None:
    captured_schema: dict[str, Any] = {}
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["status"],
        "properties": {
            "status": {"const": "completed"},
            "summary": {
                "type": "string",
                "minLength": 3,
                "pattern": "^[a-z]+$",
            },
            "comment": {"type": ["string", "null"]},
        },
        "additionalProperties": False,
    }

    def runner(
        command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
        _on_line: Callable[[str], None] | None,
    ) -> int:
        schema_path = Path(command[command.index("--output-schema") + 1])
        captured_schema.update(json.loads(schema_path.read_text(encoding="utf-8")))
        log_path.write_text("ok\n", encoding="utf-8")
        _write_invocation_result(
            command,
            {"status": "completed", "summary": None, "comment": None},
        )
        return 0

    spec = codex_spec(tmp_path, schema=schema)
    result = CodexAdapter(ApiAgentRole.ANALYSIS, proc_runner=runner).run(spec)

    assert captured_schema == {
        "type": "object",
        "required": ["status", "summary", "comment"],
        "properties": {
            "status": {"const": "completed"},
            "summary": {"type": ["string", "null"]},
            "comment": {"type": ["string", "null"]},
        },
        "additionalProperties": False,
    }
    assert result.status == AgentStatus.COMPLETED
    assert result.payload == {"status": "completed", "comment": None}
    assert schema["properties"]["summary"]["type"] == "string"


@pytest.mark.parametrize(
    ("property_schema", "expected_transport_schema"),
    (
        (
            {"type": "string", "enum": ["approve", "reject"]},
            {
                "type": ["string", "null"],
                "enum": ["approve", "reject", None],
            },
        ),
        (
            {"type": "string", "const": "approve"},
            {
                "anyOf": [
                    {"type": "string", "const": "approve"},
                    {"type": "null"},
                ]
            },
        ),
    ),
)
def test_optional_typed_enum_and_const_accept_null_transport_placeholders(
    tmp_path: Path,
    property_schema: dict[str, Any],
    expected_transport_schema: dict[str, Any],
) -> None:
    captured_schema: dict[str, Any] = {}
    schema = {
        "type": "object",
        "required": ["status"],
        "properties": {
            "status": {"const": "completed"},
            "decision": property_schema,
        },
        "additionalProperties": False,
    }

    def runner(
        command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
        _on_line: Callable[[str], None] | None,
    ) -> int:
        schema_path = Path(command[command.index("--output-schema") + 1])
        captured_schema.update(json.loads(schema_path.read_text(encoding="utf-8")))
        log_path.write_text("ok\n", encoding="utf-8")
        _write_invocation_result(
            command, {"status": "completed", "decision": None}
        )
        return 0

    result = CodexAdapter(ApiAgentRole.ANALYSIS, proc_runner=runner).run(
        codex_spec(tmp_path, schema=schema)
    )

    assert captured_schema["properties"]["decision"] == expected_transport_schema
    assert result.status == AgentStatus.COMPLETED
    assert result.payload == {"status": "completed"}
    assert schema["properties"]["decision"] == property_schema


def test_unconstrained_optional_null_is_preserved(tmp_path: Path) -> None:
    schema = {
        "type": "object",
        "required": ["status"],
        "properties": {
            "status": {"const": "completed"},
            "metadata": {},
        },
        "additionalProperties": False,
    }

    def runner(
        command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
        _on_line: Callable[[str], None] | None,
    ) -> int:
        log_path.write_text("ok\n", encoding="utf-8")
        _write_invocation_result(
            command, {"status": "completed", "metadata": None}
        )
        return 0

    spec = codex_spec(tmp_path, schema=schema)
    result = CodexAdapter(ApiAgentRole.ANALYSIS, proc_runner=runner).run(spec)

    expected = {"status": "completed", "metadata": None}
    assert result.status == AgentStatus.COMPLETED
    assert result.payload == expected
    assert json.loads(spec.result_path.read_text(encoding="utf-8")) == expected


def test_unrepresentable_strict_schema_falls_back_to_prompt_and_local_validation(
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    schema = {
        "type": "object",
        "required": ["status", "details"],
        "properties": {
            "status": {"const": "completed"},
            "details": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
        },
        "additionalProperties": False,
    }

    def runner(
        command: Sequence[str],
        _cwd: Path,
        stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
        _on_line: Callable[[str], None] | None,
    ) -> int:
        captured["command"] = list(command)
        captured["stdin"] = stdin_text
        log_path.write_text("ok\n", encoding="utf-8")
        _write_invocation_result(
            command, {"status": "completed", "details": {"owner": "borg"}}
        )
        return 0

    result = CodexAdapter(ApiAgentRole.PLANNING, proc_runner=runner).run(
        codex_spec(tmp_path, schema=schema)
    )

    assert "--output-schema" not in captured["command"]
    assert json.dumps(schema, indent=2, sort_keys=True) in captured["stdin"]
    assert result.status == AgentStatus.COMPLETED
    assert result.payload == {
        "status": "completed",
        "details": {"owner": "borg"},
    }


def test_constraints_removed_from_transport_remain_authoritative_locally(
    tmp_path: Path,
) -> None:
    schema = {
        "type": "object",
        "required": ["status", "summary"],
        "properties": {
            "status": {"const": "completed"},
            "summary": {"type": "string", "pattern": "^[a-z]+$"},
        },
        "additionalProperties": False,
    }

    def runner(
        command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
        _on_line: Callable[[str], None] | None,
    ) -> int:
        log_path.write_text("ok\n", encoding="utf-8")
        _write_invocation_result(
            command, {"status": "completed", "summary": "not valid"}
        )
        return 0

    spec = codex_spec(tmp_path, schema=schema)
    result = CodexAdapter(ApiAgentRole.ANALYSIS, proc_runner=runner).run(spec)

    assert result.status == AgentStatus.FAILED
    assert "string does not match pattern" in (result.error or "")
    assert not spec.result_path.exists()
