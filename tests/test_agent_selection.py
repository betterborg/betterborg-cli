"""Integration contracts for native-first agent selection."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from test_adapter_harness import (
    FakeApiTransport,
    openai_function_call,
    openai_response,
)

from betterborg_cli.agent_runtime import (
    READ_ONLY_API_TOOLS,
    AgentRunSpec,
    AgentSelectionError,
    AgentStatus,
    AgentUsage,
    ApiAgentRole,
    CancellationToken,
    OpenAIAdapter,
    SelectedAgent,
    select_agent,
)
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.repository_config import (
    AgentChoice,
    AgentChoices,
    RepositoryConfig,
)
from betterborg_cli.workspace_trust import UntrustedWorkspaceError


def _refusing_trust(*_args: Any, **_kwargs: Any) -> None:
    raise UntrustedWorkspaceError("workspace is not trusted on this machine")


def _spawn_recording(runner: Any, events: list[str]) -> Any:
    def recording(*args: Any, **kwargs: Any) -> int:
        events.append("spawn")
        return runner(*args, **kwargs)

    return recording


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


@pytest.mark.parametrize("interactive", [True, False])
@pytest.mark.parametrize("role", list(ApiAgentRole))
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
def test_every_role_prefers_an_installed_native_cli_over_a_credential(
    git_repo: Path,
    interactive: bool,
    role: ApiAgentRole,
    installed: set[str],
    credentials: Mapping[str, str],
    expected: str,
) -> None:
    selected = select_agent(
        _config(),
        role,
        RepoPaths.discover(git_repo),
        interactive=interactive,
        credentials=credentials,
        executable_lookup=lambda binary: (
            f"/bin/{binary}" if binary in installed else None
        ),
    )

    assert selected.name == expected


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
        _on_line: Callable[[str], None] | None,
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


def _selected_codex(git_repo: Path, **changes: Any) -> SelectedAgent:
    return select_agent(
        _config(coding=AgentChoice(adapter="codex")),
        ApiAgentRole.CODING,
        RepoPaths.discover(git_repo),
        interactive=True,
        credentials={},
        executable_lookup=lambda _binary: "/bin/codex",
        **changes,
    )


def test_contained_run_rejects_a_host_capable_adapter_without_read_only_tools(
    git_repo: Path,
) -> None:
    selected = _selected_codex(git_repo)
    selected.adapter.proc_runner = (  # type: ignore[attr-defined]
        lambda *_args, **_kwargs: pytest.fail("unexpected spawn")
    )

    with pytest.raises(AgentSelectionError, match="read-only tool set"):
        selected.run_contained(_spec(git_repo, allowed_tools=()))


def test_contained_run_rejects_a_host_capable_adapter_granted_a_write_tool(
    git_repo: Path,
) -> None:
    selected = _selected_codex(git_repo)
    selected.adapter.proc_runner = (  # type: ignore[attr-defined]
        lambda *_args, **_kwargs: pytest.fail("unexpected spawn")
    )

    with pytest.raises(AgentSelectionError, match="read-only tool set"):
        selected.run_contained(
            _spec(git_repo, allowed_tools=(*READ_ONLY_API_TOOLS, "apply_patch"))
        )


def test_contained_run_reports_cancellation_before_trust_or_spawn(
    git_repo: Path,
) -> None:
    selected = _selected_codex(git_repo, trust_requirement=_refusing_trust)
    selected.adapter.proc_runner = (  # type: ignore[attr-defined]
        lambda *_args, **_kwargs: pytest.fail("unexpected spawn")
    )
    cancel = CancellationToken()
    cancel.cancel()

    result = selected.run_contained(
        _spec(git_repo, allowed_tools=READ_ONLY_API_TOOLS), cancel=cancel
    )

    assert result.status == AgentStatus.CANCELLED


def test_contained_run_requires_workspace_trust_for_a_native_cli(
    git_repo: Path,
) -> None:
    selected = _selected_codex(
        git_repo,
        trust_requirement=_refusing_trust,
    )
    selected.adapter.proc_runner = (  # type: ignore[attr-defined]
        lambda *_args, **_kwargs: pytest.fail("unexpected spawn")
    )

    with pytest.raises(UntrustedWorkspaceError):
        selected.run_contained(
            _spec(git_repo, allowed_tools=READ_ONLY_API_TOOLS)
        )


def test_contained_run_sandboxes_a_native_cli_under_read_only_tools(
    git_repo: Path,
) -> None:
    commands: list[Sequence[str]] = []

    def runner(
        command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        log_path: Path,
        _cancel: Any,
        _env: Any,
        _on_line: Callable[[str], None] | None,
    ) -> int:
        commands.append(command)
        log_path.write_text("", encoding="utf-8")
        Path(command[command.index("-o") + 1]).write_text(
            json.dumps({"status": "completed", "version": "1"}), encoding="utf-8"
        )
        return 0

    events: list[str] = []

    def require_trust(paths: RepoPaths, **_kwargs: Any) -> None:
        assert paths.root == git_repo.resolve()
        events.append("trust")

    selected = _selected_codex(git_repo, trust_requirement=require_trust)
    selected.adapter.proc_runner = _spawn_recording(  # type: ignore[attr-defined]
        runner, events
    )

    result = selected.run_contained(
        _spec(git_repo, allowed_tools=READ_ONLY_API_TOOLS)
    )

    assert result.status == AgentStatus.COMPLETED
    assert events == ["trust", "spawn"]
    assert commands[0][commands[0].index("-s") + 1] == "read-only"


def test_run_rejects_a_cwd_from_a_different_repository_before_trust_or_spawn(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    other_repo = tmp_path / "other"
    other_repo.mkdir()
    subprocess.run(["git", "init", "--quiet", str(other_repo)], check=True)
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

    with pytest.raises(AgentSelectionError, match="different repository"):
        selected.run(_spec(other_repo))


def test_run_accepts_a_managed_linked_worktree_and_trusts_that_workspace(
    git_repo: Path,
) -> None:
    marker = git_repo / "tracked.txt"
    marker.write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(git_repo), "add", marker.name], check=True)
    subprocess.run(
        ["git", "-C", str(git_repo), "commit", "--quiet", "-m", "test"],
        check=True,
    )
    paths = RepoPaths.discover(git_repo)
    worktree = paths.worktrees_dir / "task"
    worktree.parent.mkdir(parents=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(git_repo),
            "worktree",
            "add",
            "--quiet",
            "--detach",
            str(worktree),
            "HEAD",
        ],
        check=True,
    )
    trusted: list[Path] = []
    transport = FakeApiTransport(
        [
            openai_response(
                [
                    openai_function_call(
                        "submit_result",
                        {"status": "completed", "version": "worktree"},
                        call_id="submit",
                    )
                ]
            )
        ]
    )
    selected = select_agent(
        _config(review=AgentChoice(adapter="openai")),
        ApiAgentRole.REVIEW,
        paths,
        interactive=False,
        credentials={"OPENAI_API_KEY": "key"},
        trust_requirement=lambda run_paths, **_kwargs: trusted.append(
            run_paths.root
        ),
    )
    assert isinstance(selected.adapter, OpenAIAdapter)
    selected.adapter.transport = transport

    result = selected.run(_spec(worktree))

    assert result.status == AgentStatus.COMPLETED
    assert trusted == [worktree.resolve()]


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
    def respond(_cancel: CancellationToken | None) -> Mapping[str, Any]:
        events.append("request")
        return openai_response(
            [
                openai_function_call(
                    "submit_result",
                    {"status": "completed", "version": "api"},
                    call_id="submit",
                )
            ],
            model="resolved-openai-model",
            input_tokens=7,
            output_tokens=3,
        )

    transport = FakeApiTransport([respond])
    assert isinstance(selected.adapter, OpenAIAdapter)
    selected.adapter.transport = transport

    result = selected.run(_spec(git_repo))

    assert selected.capabilities.host_capable
    assert selected.capabilities.tool_allowlist
    assert events == ["trust", "request"]
    assert transport.api_keys == [secret]
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
        executable_lookup=lambda _binary: None,
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
            executable_lookup=lambda _binary: None,
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
        executable_lookup=lambda _binary: None,
    )

    assert selected.name == "openai"
    assert selected.effort == "high"
