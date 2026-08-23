"""Claude CLI behavior over the shared adapter run contract."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from test_adapter_harness import claude_spec

from betterborg_cli.agent_runtime import (
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


def _unpack_invocation(
    command: Sequence[str], stdin_text: str
) -> tuple[list[str], str, str]:
    assert command[:2] == ["sh", "-c"]
    assert command[3] == "betterborg-claude-system-prompt"
    assert '--system-prompt-file "$prompt_file"' in command[2]
    encoded_size = int(command[4])
    system_prompt = base64.b64decode(stdin_text[:encoded_size]).decode()
    return list(command[5:]), system_prompt, stdin_text[encoded_size:]


@pytest.mark.parametrize("role", list(ApiAgentRole))
def test_every_role_discloses_native_host_capability(
    role: ApiAgentRole,
) -> None:
    adapter = ClaudeAdapter(role, proc_runner=lambda *_args: 0)

    assert adapter.capabilities.host_capable
    assert adapter.capabilities.supports_billing(BillingMode.SUBSCRIPTION)
    assert not adapter.capabilities.supports_billing(BillingMode.API)


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
    ) -> int:
        captured.update(command=list(command), cwd=cwd, stdin=stdin_text, env=env)
        log_path.write_text(
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
            encoding="utf-8",
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

    command, system_prompt, user_prompt = _unpack_invocation(
        captured["command"], captured["stdin"]
    )
    assert command == [
        "claude-custom",
        "-p",
        "--output-format",
        "json",
        "--model",
        "claude-model-override",
        "--effort",
        "high",
        "--dangerously-skip-permissions",
        "--allowed-tools",
        "Read,Edit",
    ]
    assert captured["cwd"] == tmp_path
    assert captured["env"]["BETTERBORG_TEST_ENV"] == "present"
    assert "JSON Schema" in system_prompt
    assert '"version"' in system_prompt
    assert user_prompt == spec.user_prompt
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
    ) -> int:
        log_path.write_text(_envelope({"status": "completed"}), encoding="utf-8")
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
    ) -> int:
        calls.append((list(command), stdin_text))
        log_path.write_text(responses.pop(0), encoding="utf-8")
        return 0

    result = ClaudeAdapter(
        ApiAgentRole.PLANNING,
        proc_runner=runner,
        transient_backoff_seconds=0,
    ).run(claude_spec(tmp_path))

    assert result.status == AgentStatus.COMPLETED
    assert result.attempts == 2
    assert result.usage == AgentUsage(tokens_input=5)
    assert calls[0] == calls[1]


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
            _envelope("rate limited", is_error=True, api_error_status=429),
            encoding="utf-8",
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


def test_transcript_array_and_prose_wrapped_result_are_supported(
    tmp_path: Path,
) -> None:
    transcript = json.dumps(
        [
            {"type": "system", "subtype": "init"},
            json.loads(
                _envelope(
                    'Result:\n```json\n{"status":"completed","version":"2.x"}\n```'
                )
            ),
        ]
    )

    def runner(
        _command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        log_path: Path,
        _cancel: CancellationToken | None,
        _env: Mapping[str, str] | None,
    ) -> int:
        log_path.write_text(transcript, encoding="utf-8")
        return 0

    result = ClaudeAdapter(ApiAgentRole.ANALYSIS, proc_runner=runner).run(
        claude_spec(tmp_path)
    )

    assert result.status == AgentStatus.COMPLETED
    assert result.payload == {"status": "completed", "version": "2.x"}
