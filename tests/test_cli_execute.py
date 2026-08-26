"""CLI contracts for the generation-bound host execution gate."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from click.testing import CliRunner

from betterborg_cli import cli as cli_module
from betterborg_cli.agent_runtime import MockAdapter
from betterborg_cli.cli import cli
from betterborg_cli.host_execution import HostExecutionResult, HostPreflightPlan
from betterborg_cli.planning import TaskPublisher
from betterborg_cli.store import (
    BorgState,
    ExecutionRunStatus,
    PlanApproval,
    SqliteStore,
)


def _task_body(round_number: int) -> dict[str, object]:
    return {
        "stage": "08-estimate-publish",
        "stem": f"{round_number:02d}-execute-gate",
        "title": f"Execute generation {round_number}",
        "why": "The approved generation needs a host execution gate.",
        "scope": ["Exercise the generation-bound gate."],
        "implementation_notes": [],
        "acceptance_criteria": ["The generation is gated."],
        "tests": ["Verify the execution decision."],
        "dependencies": [],
        "out_of_scope": [],
        "plan_refs": ["P1.deliverable.1"],
        "estimate_complexity": "small",
    }


def _seed_executable_generation(
    root: Path,
    planning_cli_repository,
    approved_task_generation,
    *,
    name: str = "execute-gate",
    round_number: int = 1,
    approval: PlanApproval | None = None,
):
    if approval is None:
        repository, paths = planning_cli_repository(root, name)
    else:
        paths = cli_module.RepoPaths.discover(root)
        config = cli_module.load_repository_config(paths)
        with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
            repository = store.get_repository(config.repository_id)
        assert repository is not None

    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, name)
        assert borg is not None
        if borg.state is not BorgState.READY_TO_EXECUTE:
            borg = store.compare_and_set_borg_state(
                borg.id,
                expected_state=borg.state,
                expected_version=borg.state_version,
                new_state=BorgState.READY_TO_EXECUTE,
            )
        if approval is None:
            approval = PlanApproval(
                borg_id=borg.id,
                plan_digest="sha256:approved-plan",
                manifest={},
            )
            store.append_plan_approval(approval)
        fixture = approved_task_generation(
            store,
            borg,
            approval,
            body=_task_body(round_number),
            round_number=round_number,
            task_ref=f"T-{round_number}",
        )
        publication = TaskPublisher(repository, store).publish(
            fixture.generation.id
        )
    return repository, paths, borg, approval, fixture, publication


def _execution_result(
    status: ExecutionRunStatus = ExecutionRunStatus.COMPLETED,
):
    return SimpleNamespace(
        preflight=HostPreflightPlan(
            repository_root=Path.cwd(),
            commands=(),
            prepare_commands=(),
            materialize_commands=(),
            environment_files=(),
            executables=(),
            required_secret_names=(),
            compose_files=(),
            services=(),
        ),
        active_operation_id=None,
        operation_id=uuid4(),
        status=status,
    )


def _trust(
    cli_runner: CliRunner,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(root)
    monkeypatch.setenv("XDG_STATE_HOME", str(root.parent / "machine-state"))
    trusted = cli_runner.invoke(cli, ["trust", "--yes"])
    assert trusted.exit_code == 0, trusted.output


def test_execute_requires_trust_then_approves_resumes_and_regates_generation(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, paths, borg, approval, first, _publication = (
        _seed_executable_generation(
            committed_git_repo,
            planning_cli_repository,
            approved_task_generation,
        )
    )
    monkeypatch.chdir(committed_git_repo)
    monkeypatch.setenv(
        "XDG_STATE_HOME", str(committed_git_repo.parent / "machine-state")
    )
    calls: list[object] = []
    config_calls: list[Path] = []
    load_config = cli_module.load_repository_config

    def observed_load_config(observed_paths):
        config_calls.append(observed_paths.root)
        return load_config(observed_paths)

    def invoke_host(
        _paths, store, _config, _repository_id, borg_id, generation_id
    ):
        # Host execution owns run acquisition, so observing the immutable row
        # here proves the gate commits before any claim can occur.
        decision = store.get_current_execution_decision(borg_id)
        assert decision is not None
        assert decision.generation_id == generation_id
        calls.append(decision)
        return _execution_result()

    monkeypatch.setattr(cli_module, "_invoke_host_execution", invoke_host)
    monkeypatch.setattr(cli_module, "load_repository_config", observed_load_config)

    untrusted = cli_runner.invoke(cli, ["execute", "execute-gate", "--auto-execute"])
    assert untrusted.exit_code == 1
    assert "workspace is not trusted" in untrusted.output
    assert calls == []
    assert config_calls == []

    _trust(cli_runner, committed_git_repo, monkeypatch)
    approved = cli_runner.invoke(
        cli,
        ["execute", "execute-gate"],
        input="y\n",
    )
    assert approved.exit_code == 0, approved.output
    assert approved.output.startswith("DUMMY DATA")
    assert "Recorded execution estimate approved" in approved.output
    assert len(calls) == 1
    assert config_calls == [committed_git_repo]
    assert calls[0].decision == "approved"
    assert calls[0].source == "interactive"

    resumed = cli_runner.invoke(cli, ["execute", "execute-gate"])
    assert resumed.exit_code == 0, resumed.output
    assert "Using recorded execution decision" in resumed.output
    assert "Approve this estimate" not in resumed.output
    assert len(calls) == 2
    assert calls[1].id == calls[0].id

    _repository, _paths, _borg, _approval, second, _publication = (
        _seed_executable_generation(
            committed_git_repo,
            planning_cli_repository,
            approved_task_generation,
            round_number=2,
            approval=approval,
        )
    )
    regated = cli_runner.invoke(
        cli,
        ["execute", "execute-gate"],
        input="y\n",
    )
    assert regated.exit_code == 0, regated.output
    assert "Approve this estimate and begin host execution?" in regated.output
    assert len(calls) == 3
    assert calls[2].generation_id == second.generation.id
    assert calls[2].id != calls[0].id

    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        assert store.get_execution_decision(borg.id, first.generation.id) == calls[0]
        assert store.get_current_execution_decision(borg.id) == calls[2]


def test_auto_execute_records_bypass_without_skipping_workspace_trust(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repository, paths, borg, _approval, fixture, _publication = (
        _seed_executable_generation(
            committed_git_repo,
            planning_cli_repository,
            approved_task_generation,
            name="auto-execute",
        )
    )
    _trust(cli_runner, committed_git_repo, monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "_invoke_host_execution",
        lambda *_args: _execution_result(),
    )

    result = cli_runner.invoke(
        cli, ["execute", "auto-execute", "--auto-execute"]
    )

    assert result.exit_code == 0, result.output
    assert "Approve this estimate" not in result.output
    assert "Recorded execution estimate bypassed" in result.output
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        decision = store.get_current_execution_decision(borg.id)
    assert decision is not None
    assert decision.generation_id == fixture.generation.id
    assert decision.decision == "bypassed"
    assert decision.source == "auto_execute"
    assert decision.snapshot["generation_id"] == str(fixture.generation.id)


@pytest.mark.parametrize(
    "status",
    [ExecutionRunStatus.FAILED, ExecutionRunStatus.CANCELLED],
)
def test_execute_exits_nonzero_for_unsuccessful_host_execution(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
    status: ExecutionRunStatus,
) -> None:
    _seed_executable_generation(
        committed_git_repo,
        planning_cli_repository,
        approved_task_generation,
        name="unsuccessful-execution",
    )
    _trust(cli_runner, committed_git_repo, monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "_invoke_host_execution",
        lambda *_args: _execution_result(status),
    )

    result = cli_runner.invoke(
        cli,
        ["execute", "unsuccessful-execution", "--auto-execute"],
    )

    assert result.exit_code == 1
    assert "Execution operation " in result.output
    assert f": {status.value}" in result.output


def test_task_digest_drift_blocks_before_decision_or_host_execution(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repository, paths, borg, _approval, _fixture, publication = (
        _seed_executable_generation(
            committed_git_repo,
            planning_cli_repository,
            approved_task_generation,
            name="digest-drift",
        )
    )
    _trust(cli_runner, committed_git_repo, monkeypatch)
    publication.files[0].path.write_text("drifted\n", encoding="utf-8")
    monkeypatch.setattr(
        cli_module,
        "_invoke_host_execution",
        lambda *_args: pytest.fail("host execution must not run after digest drift"),
    )

    result = cli_runner.invoke(
        cli, ["execute", "digest-drift", "--auto-execute"]
    )

    assert result.exit_code == 1
    assert "digest drifted" in result.output
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        assert store.get_current_execution_decision(borg.id) is None


def test_execute_assembly_invokes_the_concrete_host_execution_service(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, paths, borg, _approval, fixture, _publication = (
        _seed_executable_generation(
            committed_git_repo,
            planning_cli_repository,
            approved_task_generation,
            name="concrete-service",
        )
    )
    _trust(cli_runner, committed_git_repo, monkeypatch)
    config = cli_module.load_repository_config(paths)
    monkeypatch.setenv("EXECUTE_TOKEN", "owner-secret")
    adapters: list[MockAdapter] = []
    execution_trust: list[object] = []

    def select(*_args, **kwargs):
        adapter = MockAdapter(name="openai")
        adapters.append(adapter)
        execution_trust.append(kwargs["trust_requirement"])
        return adapter

    calls: list[tuple[object, ...]] = []

    def run(service, borg_id, generation_id, analyzer_plan, **kwargs):
        assert type(service) is cli_module.HostExecutionService
        assert type(service._runtime) is cli_module.HostTaskRuntime
        assert type(service._runtime._coding) is cli_module.HostCodingPhase
        assert type(service._runtime._review_fix) is cli_module.HostReviewFixPhase
        assert type(service._runtime._merge) is cli_module.HostMergePhase
        assert type(service._runtime._sanity) is cli_module.HostSanityPhase
        calls.append((borg_id, generation_id, analyzer_plan, kwargs))
        return HostExecutionResult(service._runtime.plan)

    monkeypatch.setattr(cli_module, "select_agent", select)
    monkeypatch.setattr(cli_module.HostExecutionService, "run", run)
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        analysis = store.get_prior_ready_analysis(repository.id)
        assert analysis is not None
        analyzer_plan = {
            **analysis.analysis_json,
            "required_secrets": [
                {
                    "name": "EXECUTE_TOKEN",
                    "used_by": ["coding"],
                    "scope": "agent",
                    "source": "test",
                }
            ],
        }
        monkeypatch.setattr(
            cli_module.SqliteStore,
            "get_prior_ready_analysis",
            lambda _store, _repository_id: SimpleNamespace(
                analysis_json=analyzer_plan
            ),
        )
        result = cli_module._invoke_host_execution(
            paths,
            store,
            config,
            repository.id,
            borg.id,
            fixture.generation.id,
        )

    assert isinstance(result, HostExecutionResult)
    assert len(adapters) == 3
    observed_trust_paths: list[Path] = []
    monkeypatch.setattr(
        cli_module,
        "require_workspace_trust",
        lambda observed, **_kwargs: observed_trust_paths.append(observed.root),
    )
    managed_worktree_paths = SimpleNamespace(root=paths.worktrees_dir / "task")
    for requirement in execution_trust:
        requirement(managed_worktree_paths)
    assert observed_trust_paths == [paths.root] * 3
    assert calls == [
        (
            borg.id,
            fixture.generation.id,
            analyzer_plan,
            {"secret_values": {"EXECUTE_TOKEN": "owner-secret"}},
        )
    ]
