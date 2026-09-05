"""Integration contracts for native-first agent selection."""

from __future__ import annotations

import ast
import json
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, replace
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
    MockAdapter,
    MockResponse,
    OpenAIAdapter,
    SandboxSettingError,
    SelectedAgent,
    run_captured,
    select_agent,
)
from betterborg_cli.agent_runtime.selection import _STAGE_ROLES
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.repository_config import (
    AgentChoice,
    AgentChoices,
    AgentStage,
    RepositoryConfig,
)
from betterborg_cli.workspace_trust import UntrustedWorkspaceError, WorkspaceIdentity


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


def test_production_selectors_use_stage_identities() -> None:
    package_root = Path(__file__).parents[1] / "src" / "betterborg_cli"
    violations: list[str] = []
    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else None
            )
            if called_name != "select_agent":
                continue
            stage_arguments = node.args[1:2]
            stage_arguments.extend(
                keyword.value
                for keyword in node.keywords
                if keyword.arg == "stage"
            )
            if any(
                isinstance(stage, ast.Attribute)
                and isinstance(stage.value, ast.Name)
                and stage.value.id == "ApiAgentRole"
                for stage in stage_arguments
            ):
                violations.append(f"{path.relative_to(package_root)}:{node.lineno}")

    assert violations == []


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
@pytest.mark.parametrize("stage", list(AgentStage))
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
def test_every_stage_uses_native_first_provider_defaults_and_security_role(
    git_repo: Path,
    interactive: bool,
    stage: AgentStage,
    installed: set[str],
    credentials: Mapping[str, str],
    expected: str,
) -> None:
    selected = select_agent(
        _config(),
        stage,
        RepoPaths.discover(git_repo),
        interactive=interactive,
        credentials=credentials,
        executable_lookup=lambda binary: (
            f"/bin/{binary}" if binary in installed else None
        ),
    )

    assert selected.name == expected
    assert selected.role is _STAGE_ROLES[stage]
    assert selected.model == (
        "claude-opus-5"
        if expected in {"claude", "anthropic"}
        else "gpt-5.6-sol"
    )
    assert selected.effort == "high"


def test_stage_catalog_configuration_and_security_roles_have_structural_parity(
) -> None:
    choice_fields = {item.name for item in fields(AgentChoices)} - {"defaults"}

    assert choice_fields == {stage.value for stage in AgentStage}
    assert set(_STAGE_ROLES) == set(AgentStage)


@pytest.mark.parametrize(
    ("stage_choice", "expected_model", "expected_effort"),
    [
        (AgentChoice(effort="high"), "default-model", "high"),
        (AgentChoice(model="stage-model"), "stage-model", "low"),
    ],
)
def test_stage_overrides_inherit_other_selection_fields(
    git_repo: Path,
    stage_choice: AgentChoice,
    expected_model: str,
    expected_effort: str,
) -> None:
    selected = select_agent(
        _config(
            defaults=AgentChoice(
                adapter="codex", model="default-model", effort="low"
            ),
            architect=stage_choice,
        ),
        AgentStage.ARCHITECT,
        RepoPaths.discover(git_repo),
        credentials={},
        executable_lookup=lambda binary: (
            "/bin/codex" if binary == "codex" else None
        ),
    )

    assert selected.name == "codex"
    assert selected.model == expected_model
    assert selected.effort == expected_effort


@pytest.mark.parametrize(
    ("choice", "problem"),
    [
        (AgentChoice(adapter="missing"), "is unknown"),
        (AgentChoice(adapter="codex"), "executable was not found"),
        (AgentChoice(adapter="anthropic"), "ANTHROPIC_API_KEY is not set"),
    ],
)
def test_configured_stage_adapter_failure_names_stage_and_never_falls_back(
    git_repo: Path,
    choice: AgentChoice,
    problem: str,
) -> None:
    with pytest.raises(AgentSelectionError) as captured:
        select_agent(
            _config(architect=choice),
            AgentStage.ARCHITECT,
            RepoPaths.discover(git_repo),
            credentials={"OPENAI_API_KEY": "available"},
            executable_lookup=lambda binary: (
                "/bin/claude" if binary == "claude" else None
            ),
        )

    message = str(captured.value)
    assert "stage 'architect'" in message
    assert problem in message


@pytest.mark.parametrize("effort", ["high", "low"])
def test_configured_anthropic_stage_accepts_effort(
    git_repo: Path, effort: str
) -> None:
    selected = select_agent(
        _config(
            analysis=AgentChoice(adapter="anthropic", effort=effort)
        ),
        AgentStage.ANALYSIS,
        RepoPaths.discover(git_repo),
        credentials={"ANTHROPIC_API_KEY": "available"},
        executable_lookup=lambda _binary: None,
    )

    assert selected.name == "anthropic"
    assert selected.model == "claude-opus-5"
    assert selected.effort == effort


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
        AgentStage.CODING,
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


def test_selected_agent_forwards_one_token_through_discovery_and_trust(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = RepoPaths.discover(git_repo)
    cancel = CancellationToken()
    events: list[tuple[str, object]] = []
    original_paths_discover = RepoPaths.discover.__func__
    original_identity_discover = WorkspaceIdentity.discover.__func__

    def discover_paths(cls, start=None, *, cancel=None, command_runner=None):
        events.append(("paths", cancel))
        runner = run_captured if command_runner is None else command_runner
        return original_paths_discover(cls, start, command_runner=runner)

    def discover_identity(cls, discovered, *, cancel=None, command_runner=None):
        events.append(("identity", cancel))
        runner = run_captured if command_runner is None else command_runner
        return original_identity_discover(
            cls,
            discovered,
            command_runner=runner,
        )

    def trust(_paths: RepoPaths, **kwargs: Any) -> None:
        events.append(("trust", kwargs["cancel"]))
        assert callable(kwargs["command_runner"])

    monkeypatch.setattr(RepoPaths, "discover", classmethod(discover_paths))
    monkeypatch.setattr(
        WorkspaceIdentity,
        "discover",
        classmethod(discover_identity),
    )
    adapter = MockAdapter(
        capabilities=replace(MockAdapter().capabilities, host_capable=True)
    ).queue(MockResponse(payload={"status": "completed", "version": "1"}))
    selected = SelectedAgent(
        role=ApiAgentRole.CODING,
        adapter=adapter,
        paths=paths,
        trust_requirement=trust,
    )

    result = selected.run(_spec(git_repo), cancel=cancel)

    assert result.status is AgentStatus.COMPLETED
    assert events == [
        ("paths", cancel),
        ("identity", cancel),
        ("identity", cancel),
        ("trust", cancel),
    ]


def test_contained_selected_agent_forwards_token_to_compatible_trust(
    git_repo: Path,
) -> None:
    cancel = CancellationToken()
    observed: list[object] = []

    def trust(_paths: RepoPaths, **kwargs: Any) -> None:
        observed.append(kwargs["cancel"])
        assert callable(kwargs["command_runner"])

    adapter = MockAdapter(
        capabilities=replace(MockAdapter().capabilities, host_capable=True)
    ).queue(MockResponse(payload={"status": "completed", "version": "1"}))
    selected = SelectedAgent(
        role=ApiAgentRole.CODING,
        adapter=adapter,
        paths=RepoPaths.discover(git_repo),
        trust_requirement=trust,
    )

    result = selected.run_contained(
        _spec(git_repo, allowed_tools=READ_ONLY_API_TOOLS),
        cancel=cancel,
    )

    assert result.status is AgentStatus.COMPLETED
    assert observed == [cancel]


def test_contained_agent_cancellation_reaps_trust_probe_before_adapter(
    git_repo: Path,
    real_process_harness: Any,
) -> None:
    cancel = CancellationToken(grace_seconds=0.05)
    errors: list[BaseException] = []

    def trust(_paths: RepoPaths, **kwargs: Any) -> None:
        assert kwargs["cancel"] is cancel
        result = kwargs["command_runner"](
            real_process_harness.resistant_argv("contained-trust"),
            cancel=kwargs["cancel"],
        )
        if result.returncode != 0:
            raise ValueError("contained trust discovery cancelled")

    adapter = MockAdapter(
        capabilities=replace(MockAdapter().capabilities, host_capable=True)
    ).queue(MockResponse(payload={"status": "completed", "version": "1"}))
    selected = SelectedAgent(
        role=ApiAgentRole.CODING,
        adapter=adapter,
        paths=RepoPaths.discover(git_repo),
        trust_requirement=trust,
    )

    def invoke() -> None:
        try:
            selected.run_contained(
                _spec(git_repo, allowed_tools=READ_ONLY_API_TOOLS),
                cancel=cancel,
            )
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=invoke)
    worker.start()
    real_process_harness.wait_for_marker("contained-trust.parent.pid")
    real_process_harness.wait_for_marker("contained-trust.child.pid")
    cancel.cancel()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    assert str(errors[0]) == "contained trust discovery cancelled"
    assert adapter.calls == []
    real_process_harness.assert_tree_absent("contained-trust")


@pytest.mark.parametrize(
    "blocked_probe",
    ["run-root", "selected-identity", "run-identity", "trust"],
)
def test_selected_agent_cancellation_reaps_each_discovery_probe_before_adapter(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    real_process_harness: Any,
    blocked_probe: str,
) -> None:
    paths = RepoPaths.discover(git_repo)
    cancel = CancellationToken(grace_seconds=0.05)
    errors: list[BaseException] = []
    original_paths_discover = RepoPaths.discover.__func__
    original_identity_discover = WorkspaceIdentity.discover.__func__
    identity_calls = 0

    def blocked_runner(
        _command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        return run_captured(
            real_process_harness.resistant_argv(blocked_probe),
            cancel=kwargs["cancel"],
            check=kwargs["check"],
        )

    def discover_paths(cls, start=None, *, cancel=None, command_runner=None):
        runner = (
            blocked_runner
            if blocked_probe == "run-root"
            else (run_captured if command_runner is None else command_runner)
        )
        return original_paths_discover(
            cls,
            start,
            cancel=cancel,
            command_runner=runner,
        )

    def discover_identity(cls, discovered, *, cancel=None, command_runner=None):
        nonlocal identity_calls
        identity_calls += 1
        target_call = 1 if blocked_probe == "selected-identity" else 2
        runner = (
            blocked_runner
            if blocked_probe in {"selected-identity", "run-identity"}
            and identity_calls == target_call
            else (run_captured if command_runner is None else command_runner)
        )
        return original_identity_discover(
            cls,
            discovered,
            cancel=cancel,
            command_runner=runner,
        )

    def trust(_paths: RepoPaths, **kwargs: Any) -> None:
        if blocked_probe != "trust":
            return
        result = blocked_runner([], check=True, cancel=kwargs["cancel"])
        if result.returncode != 0:
            raise ValueError("trust discovery cancelled")

    monkeypatch.setattr(RepoPaths, "discover", classmethod(discover_paths))
    monkeypatch.setattr(
        WorkspaceIdentity,
        "discover",
        classmethod(discover_identity),
    )
    adapter = MockAdapter(
        capabilities=replace(MockAdapter().capabilities, host_capable=True)
    ).queue(MockResponse(payload={"status": "completed", "version": "1"}))
    selected = SelectedAgent(
        role=ApiAgentRole.CODING,
        adapter=adapter,
        paths=paths,
        trust_requirement=trust,
    )

    def invoke() -> None:
        try:
            selected.run(_spec(git_repo), cancel=cancel)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=invoke)
    worker.start()
    real_process_harness.wait_for_marker(f"{blocked_probe}.parent.pid")
    real_process_harness.wait_for_marker(f"{blocked_probe}.child.pid")
    cancel.cancel()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert adapter.calls == []
    real_process_harness.assert_tree_absent(blocked_probe)


def test_custom_trust_type_error_is_not_treated_as_keyword_incompatibility(
    git_repo: Path,
) -> None:
    def broken_trust(
        _paths: RepoPaths,
        *,
        store: object,
        explicit: bool,
        interactive: bool,
        confirm: object,
    ) -> None:
        raise TypeError("trust implementation failed")

    selected = _selected_codex(git_repo, trust_requirement=broken_trust)
    selected.adapter.proc_runner = (  # type: ignore[attr-defined]
        lambda *_args, **_kwargs: pytest.fail("unexpected spawn")
    )

    with pytest.raises(TypeError, match="trust implementation failed"):
        selected.run(_spec(git_repo))


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
        AgentStage.CODING,
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


def test_unrecognised_sandbox_declaration_stops_agent_selection(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selecting Codex refuses a malformed declaration, as a ValueError.

    The base class is load-bearing rather than incidental: every CLI command
    that selects an agent catches ``(OSError, RuntimeError, ValueError)``, so
    it is what turns this into a message naming the accepted values instead of
    a traceback.
    """
    monkeypatch.setenv("BETTERBORG_SANDBOX", "hsot")

    with pytest.raises(ValueError, match="accepted values are auto, host") as error:
        select_agent(
            _config(coding=AgentChoice(adapter="codex")),
            AgentStage.CODING,
            RepoPaths.discover(git_repo),
            interactive=True,
            credentials={},
            executable_lookup=lambda _binary: "/bin/codex",
        )

    assert isinstance(error.value, SandboxSettingError)
    assert "BETTERBORG_SANDBOX='hsot'" in str(error.value)


def _selected_codex(git_repo: Path, **changes: Any) -> SelectedAgent:
    return select_agent(
        _config(coding=AgentChoice(adapter="codex")),
        AgentStage.CODING,
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
        AgentStage.CODING,
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
        AgentStage.REVIEW,
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
        AgentStage.CODING,
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
        AgentStage.REVIEW,
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
    assert transport.payloads[0]["reasoning"] == {"effort": "high"}
    assert result.status == AgentStatus.COMPLETED
    assert result.model == "resolved-openai-model"
    assert secret not in repr(selected)
    assert secret not in repr(selected.adapter)


def test_api_analysis_remains_contained_without_workspace_trust(
    git_repo: Path,
) -> None:
    selected = select_agent(
        _config(),
        AgentStage.ANALYSIS,
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
            AgentStage.CODING,
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
