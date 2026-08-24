"""Codex CLI behavior over the shared adapter run contract."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from test_adapter_harness import codex_spec

from betterborg_cli.agent_runtime import (
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


@pytest.mark.parametrize("role", list(ApiAgentRole))
def test_every_role_discloses_native_host_capability(role: ApiAgentRole) -> None:
    adapter = CodexAdapter(role, proc_runner=lambda *_args: 0)

    assert adapter.capabilities.host_capable
    assert adapter.capabilities.supports_billing(BillingMode.SUBSCRIPTION)
    assert not adapter.capabilities.supports_billing(BillingMode.API)
    assert not adapter.capabilities.tool_allowlist


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
                    _usage_event(12, 10, 3, cache_write_input_tokens=6),
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
        tokens_input=32,
        tokens_output=8,
        tokens_cache_read=17,
        tokens_cache_write=10,
        num_turns=2,
    )
    assert json.loads(spec.result_path.read_text(encoding="utf-8")) == result.payload


def test_schema_invalid_result_fails_without_persisting_result(
    tmp_path: Path,
) -> None:
    def runner(
        command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
    ) -> int:
        log_path.write_text("{}\n", encoding="utf-8")
        _write_invocation_result(command, {"status": "completed"})
        return 0

    spec = codex_spec(tmp_path)
    result = CodexAdapter(ApiAgentRole.ANALYSIS, proc_runner=runner).run(spec)

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
            _write_invocation_result(
                command, {"status": "completed", "version": "transient"}
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
        tokens_input=400,
        tokens_output=40,
        tokens_cache_read=75,
        tokens_cache_write=15,
        num_turns=2,
    )
    assert calls[0] == calls[1]


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


def test_nonzero_exit_with_valid_result_fails(tmp_path: Path) -> None:
    def runner(
        command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
    ) -> int:
        log_path.write_text("Codex completed before exiting\n", encoding="utf-8")
        _write_invocation_result(
            command, {"status": "completed", "version": "nonzero"}
        )
        return 2

    spec = codex_spec(tmp_path)
    result = CodexAdapter(ApiAgentRole.MERGE, proc_runner=runner).run(spec)

    assert result.status == AgentStatus.FAILED
    assert result.exit_code == 2
    assert result.payload is None
    assert "Codex exited 2" in (result.error or "")
    assert not spec.result_path.exists()
