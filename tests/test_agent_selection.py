"""Integration contracts for TTY-aware agent selection."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from betterborg_cli.agent_runtime import (
    AgentRunSpec,
    AgentSelectionError,
    AgentStatus,
    AgentUsage,
    ApiAgentRole,
    CancellationToken,
    OpenAIAdapter,
    select_agent,
)
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.repository_config import (
    AgentChoice,
    AgentChoices,
    RepositoryConfig,
)


def _config(**choices: AgentChoice) -> RepositoryConfig:
    return RepositoryConfig(
        version=1,
        repository_id=UUID("bd3b21f9-693b-4c58-b7cf-a90417809e1f"),
        default_branch="main",
        agents=AgentChoices(**choices),
    )


def _spec(tmp_path: Path, **changes: Any) -> AgentRunSpec:
    values: dict[str, Any] = {
        "system_prompt": "Complete the task.",
        "user_prompt": "Return the version.",
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
        "model": "caller-model",
        "effort": "low",
        "log_path": tmp_path / "agent.jsonl",
        "result_path": tmp_path / "result.json",
    }
    values.update(changes)
    return AgentRunSpec(**values)


@pytest.mark.parametrize(
    ("installed", "credentials", "expected"),
    [
        (
            {"claude", "codex"},
            {"ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "o"},
            "claude",
        ),
        ({"codex"}, {"ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "o"}, "codex"),
        (set(), {"ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "o"}, "anthropic"),
        (set(), {"OPENAI_API_KEY": "o"}, "openai"),
    ],
)
def test_interactive_default_precedence(
    git_repo: Path,
    installed: set[str],
    credentials: Mapping[str, str],
    expected: str,
) -> None:
    selected = select_agent(
        _config(),
        ApiAgentRole.CODING,
        RepoPaths.discover(git_repo),
        interactive=True,
        credentials=credentials,
        executable_lookup=lambda binary: (
            f"/bin/{binary}" if binary in installed else None
        ),
    )

    assert selected.name == expected


def test_noninteractive_selection_ignores_installed_native_clis(
    git_repo: Path,
) -> None:
    selected = select_agent(
        _config(),
        ApiAgentRole.REVIEW,
        RepoPaths.discover(git_repo),
        interactive=False,
        credentials={"ANTHROPIC_API_KEY": "owner-secret"},
        executable_lookup=lambda binary: f"/bin/{binary}",
    )

    assert selected.name == "anthropic"


def test_configured_native_role_applies_overrides_after_trust_before_spawn(
    git_repo: Path,
) -> None:
    events: list[str] = []
    captured: dict[str, Any] = {}

    def require_trust(paths: RepoPaths, **_kwargs: Any) -> None:
        assert paths.root == git_repo.resolve()
        events.append("trust")

    def runner(
        command: Sequence[str],
        cwd: Path,
        stdin_text: str,
        log_path: Path,
        _cancel: object,
        _env: Mapping[str, str] | None,
    ) -> int:
        events.append("spawn")
        captured.update(command=list(command), cwd=cwd, stdin=stdin_text)
        log_path.write_text(
            json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "result": json.dumps(
                        {"status": "completed", "version": "configured"}
                    ),
                    "usage": {"input_tokens": 4, "output_tokens": 2},
                    "num_turns": 1,
                }
            ),
            encoding="utf-8",
        )
        return 0

    selected = select_agent(
        _config(
            coding=AgentChoice(
                adapter="claude", model="role-model", effort="high"
            )
        ),
        ApiAgentRole.CODING,
        RepoPaths.discover(git_repo),
        interactive=True,
        credentials={},
        executable_lookup=lambda _binary: "/bin/claude",
        trust_requirement=require_trust,
    )
    selected.adapter.proc_runner = runner  # type: ignore[attr-defined]

    result = selected.run(_spec(git_repo))

    assert events == ["trust", "spawn"]
    assert captured["cwd"] == git_repo
    assert captured["command"][:8] == [
        "claude",
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        "role-model",
        "--effort",
    ]
    assert "high" in captured["command"]
    assert "--dangerously-skip-permissions" in captured["command"]
    assert result.status == AgentStatus.COMPLETED
    assert result.model == "role-model"
    assert result.usage == AgentUsage(tokens_input=4, tokens_output=2, num_turns=1)


def test_untrusted_native_selection_never_spawns(git_repo: Path) -> None:
    spawned = False

    def reject_trust(_paths: RepoPaths, **_kwargs: Any) -> None:
        raise RuntimeError("workspace trust rejected")

    def runner(*_args: Any, **_kwargs: Any) -> int:
        nonlocal spawned
        spawned = True
        return 0

    selected = select_agent(
        _config(coding=AgentChoice(adapter="codex")),
        ApiAgentRole.CODING,
        RepoPaths.discover(git_repo),
        interactive=True,
        credentials={},
        executable_lookup=lambda _binary: "/bin/codex",
        trust_requirement=reject_trust,
    )
    selected.adapter.proc_runner = runner  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="trust rejected"):
        selected.run(_spec(git_repo))

    assert not spawned


def test_pre_cancelled_native_run_does_not_request_trust_or_spawn(
    git_repo: Path,
) -> None:
    selected = select_agent(
        _config(coding=AgentChoice(adapter="codex")),
        ApiAgentRole.CODING,
        RepoPaths.discover(git_repo),
        interactive=True,
        credentials={},
        executable_lookup=lambda _binary: "/bin/codex",
        trust_requirement=lambda *_args, **_kwargs: pytest.fail("unexpected trust"),
    )
    selected.adapter.proc_runner = (  # type: ignore[attr-defined]
        lambda *_args, **_kwargs: pytest.fail("unexpected spawn")
    )
    cancel = CancellationToken()
    cancel.cancel()

    result = selected.run(_spec(git_repo), cancel=cancel)

    assert result.status == AgentStatus.CANCELLED
    assert result.attempts == 0


class _OpenAITransport:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.keys: list[str] = []
        self.payloads: list[Mapping[str, Any]] = []

    def create_response(
        self,
        payload: Mapping[str, Any],
        *,
        api_key: str,
        cancel: object = None,
    ) -> Mapping[str, Any]:
        self.events.append("request")
        self.keys.append(api_key)
        self.payloads.append(payload)
        return {
            "id": "response",
            "object": "response",
            "status": "completed",
            "model": "resolved-openai-model",
            "output": [
                {
                    "type": "function_call",
                    "id": "call",
                    "call_id": "submit",
                    "name": "submit_result",
                    "arguments": json.dumps(
                        {"status": "completed", "version": "api"}
                    ),
                    "status": "completed",
                }
            ],
            "usage": {"input_tokens": 7, "output_tokens": 3},
        }


def test_api_execution_discloses_host_capability_and_trusts_before_request(
    git_repo: Path,
) -> None:
    events: list[str] = []
    secret = "sk-proj-owner-only-secret"
    selected = select_agent(
        _config(review=AgentChoice(adapter="openai", model="review-model")),
        ApiAgentRole.REVIEW,
        RepoPaths.discover(git_repo),
        interactive=False,
        credentials={"OPENAI_API_KEY": secret},
        trust_requirement=lambda _paths, **_kwargs: events.append("trust"),
    )
    transport = _OpenAITransport(events)
    assert isinstance(selected.adapter, OpenAIAdapter)
    selected.adapter.transport = transport

    result = selected.run(_spec(git_repo))

    assert selected.capabilities.host_capable
    assert selected.capabilities.tool_allowlist
    assert events == ["trust", "request"]
    assert transport.keys == [secret]
    assert transport.payloads[0]["model"] == "review-model"
    assert transport.payloads[0]["reasoning"] == {"effort": "low"}
    assert result.status == AgentStatus.COMPLETED
    assert result.model == "resolved-openai-model"
    assert secret not in repr(selected)
    assert secret not in repr(selected.adapter)


def test_api_analysis_remains_contained_without_workspace_trust(
    git_repo: Path,
) -> None:
    selected = select_agent(
        _config(),
        ApiAgentRole.ANALYSIS,
        RepoPaths.discover(git_repo),
        interactive=False,
        credentials={"OPENAI_API_KEY": "key"},
        trust_requirement=lambda *_args, **_kwargs: pytest.fail("unexpected trust"),
    )

    assert not selected.capabilities.host_capable


@pytest.mark.parametrize(
    "choice",
    [
        AgentChoice(),
        AgentChoice(adapter="missing"),
        AgentChoice(adapter="codex"),
    ],
)
def test_missing_setup_and_unusable_overrides_name_every_setup_option(
    git_repo: Path,
    choice: AgentChoice,
) -> None:
    secret = "must-not-appear"

    with pytest.raises(AgentSelectionError) as captured:
        select_agent(
            _config(coding=choice),
            ApiAgentRole.CODING,
            RepoPaths.discover(git_repo),
            interactive=False,
            credentials={"UNRELATED": secret},
        )

    message = str(captured.value)
    for setup_name in (
        "claude",
        "codex",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    ):
        assert setup_name in message
    assert secret not in message


def test_effort_override_skips_unsupported_anthropic_default(
    git_repo: Path,
) -> None:
    selected = select_agent(
        _config(merge=AgentChoice(effort="high")),
        ApiAgentRole.MERGE,
        RepoPaths.discover(git_repo),
        interactive=False,
        credentials={"ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "o"},
    )

    assert selected.name == "openai"
    assert selected.effort == "high"
