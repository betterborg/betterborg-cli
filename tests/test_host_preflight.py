"""Analyzer-evidence contracts for trust-gated host preflight."""

from __future__ import annotations

import json
import shlex
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from betterborg_cli.agent_runtime import CancellationToken, run_captured
from betterborg_cli.host_execution import (
    HostCommand,
    HostExecutable,
    HostExecutionService,
    HostPreflight,
    HostPreflightBlock,
    HostPreflightPlan,
    HostSecret,
)
from betterborg_cli.progress import AgentActivity, AgentActivityKind
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.store import SqliteStore
from betterborg_cli.workspace_trust import TrustStore, require_workspace_trust


def _trust_store(repository: Path) -> TrustStore:
    return TrustStore(repository.parent / f"{repository.name}-trust" / "trust.json")


def _preflight(
    repository: Path, *, environment: dict[str, str] | None = None
) -> HostPreflight:
    store = _trust_store(repository)
    require_workspace_trust(RepoPaths.discover(repository), store=store, explicit=True)
    return HostPreflight(
        repository,
        trust_store=store,
        environment=environment or {},
    )


def _executable(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _base_plan() -> dict[str, object]:
    return {
        "command_catalog": {
            "source": "pyproject.toml",
            "commands": [
                {
                    "stage": "test",
                    "argv": ["example-runtime", "-m", "pytest"],
                    "cwd": "package",
                    "source": "pyproject.toml",
                    "uses_services": ["database", "search"],
                    "required_secrets": ["PACKAGE_TOKEN"],
                }
            ],
        },
        "environment": {
            "files": ["runtime.version"],
            "toolchains": [
                {
                    "name": "example-runtime",
                    "version": "3.11.9",
                    "source": "runtime.version",
                }
            ],
        },
        "required_secrets": [
            {
                "name": "PACKAGE_TOKEN",
                "used_by": ["test"],
                "scope": "build",
                "source": "pyproject.toml#test",
            }
        ],
        "compose": {
            "file": "compose.yml",
            "files": [
                {
                    "path": "compose.yml",
                    "source": "compose.yml",
                    "services": ["postgres"],
                }
            ],
            "source": "compose.yml",
        },
        "service_dependencies": [
            {
                "name": "database",
                "compose_service": "postgres",
                "url_env": "DATABASE_URL",
                "port": 5432,
                "source": "compose.yml#services.postgres",
            },
            {
                "name": "search",
                "url_env": "SEARCH_URL",
                "source": "pyproject.toml#search",
            },
            {
                "name": "unused-inferred-service",
                "source": "example.env",
            },
        ],
    }


def _complete_probe_plan(repository: Path) -> dict[str, object]:
    """Prepare one plan that reaches both direct preflight probes."""
    (repository / "package").mkdir(exist_ok=True)
    (repository / "runtime.version").write_text("3.11.9\n", encoding="utf-8")
    return _base_plan()


def _probe_name(command: list[str] | tuple[str, ...]) -> str | None:
    """Identify the two ordered direct probes covered by cancellation tests."""
    argv = tuple(command)
    if argv[-1:] == ("--show-toplevel",):
        return "root"
    if argv[-1:] == ("--git-common-dir",):
        return "identity"
    return None


def test_all_direct_probes_share_token_runner_and_command_activity(
    committed_git_repo: Path,
) -> None:
    binary_dir = committed_git_repo.parent / "all-probe-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-runtime", "echo 'example 3.11.9'")
    plan = _complete_probe_plan(committed_git_repo)
    store = _trust_store(committed_git_repo)
    require_workspace_trust(
        RepoPaths.discover(committed_git_repo), store=store, explicit=True
    )
    cancel = CancellationToken()
    calls: list[tuple[tuple[str, ...], CancellationToken | None]] = []
    activities: list[AgentActivity] = []

    def runner(command, **kwargs):
        calls.append((tuple(command), kwargs.get("cancel")))
        return run_captured(command, **kwargs)

    result = HostPreflight(
        committed_git_repo,
        trust_store=store,
        environment={"PATH": str(binary_dir)},
        cancel=cancel,
        command_runner=runner,
        activity=activities.append,
    ).validate(
        plan,
        available_secret_names={"PACKAGE_TOKEN"},
    )

    assert isinstance(result, HostPreflightPlan)
    assert [_probe_name(command) for command, _token in calls] == [
        "root",
        "identity",
    ]
    assert all(observed is cancel for _command, observed in calls)
    assert [activity.kind for activity in activities] == [
        AgentActivityKind.COMMAND
    ] * 2
    assert [activity.detail for activity in activities] == [
        shlex.join(command) for command, _token in calls
    ]


@pytest.mark.parametrize(
    "target",
    ["root", "identity"],
)
def test_cancellation_reaps_each_direct_probe_and_stops_later_probes(
    committed_git_repo: Path,
    real_process_harness,
    target: str,
) -> None:
    binary_dir = committed_git_repo.parent / f"{target}-cancel-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-runtime", "echo 'example 3.11.9'")
    _complete_probe_plan(committed_git_repo)
    store = _trust_store(committed_git_repo)
    require_workspace_trust(
        RepoPaths.discover(committed_git_repo), store=store, explicit=True
    )
    resistant = real_process_harness.resistant_argv(target)
    source = r'''
from __future__ import annotations

import json
import sys
from pathlib import Path

from betterborg_cli.agent_runtime import CancellationToken, run_captured
from betterborg_cli.host_execution import HostPreflight
from betterborg_cli.run_control import RunControl
from betterborg_cli.workspace_trust import TrustStore

tests_root = Path(sys.argv[1])
repository = Path(sys.argv[2])
trust_path = Path(sys.argv[3])
binary_dir = Path(sys.argv[4])
marker_root = Path(sys.argv[5])
target = sys.argv[6]
resistant = tuple(json.loads(sys.argv[7]))
sys.path.insert(0, str(tests_root))

from test_host_preflight import _base_plan, _probe_name


def append_marker(name: str, value: str) -> None:
    path = marker_root / name
    previous = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(previous + value + "\n", encoding="utf-8")


def runner(command, **kwargs):
    name = _probe_name(command)
    if name is not None:
        append_marker(f"{target}.probes", name)
    return run_captured(resistant if name == target else command, **kwargs)


def activity(current) -> None:
    append_marker(f"{target}.activities", current.detail or current.kind.value)


cancel = CancellationToken()
control = RunControl(cancel).install()
try:
    with control.protected():
        HostPreflight(
            repository,
            trust_store=TrustStore(trust_path),
            environment={"PATH": str(binary_dir)},
            cancel=cancel,
            command_runner=runner,
            activity=activity,
        ).validate(
            _base_plan(),
            available_secret_names={"PACKAGE_TOKEN"},
        )
except KeyboardInterrupt:
    (marker_root / f"{target}.interrupted").write_text("yes", encoding="utf-8")
    raise SystemExit(130) from None
finally:
    control.close()
'''
    process = real_process_harness.launch_python(
        source,
        str(Path(__file__).parent),
        str(committed_git_repo),
        str(store.path),
        str(binary_dir),
        str(real_process_harness.root),
        target,
        json.dumps(resistant),
        name=f"{target}-preflight",
    )
    real_process_harness.wait_for_marker(f"{target}.child.pid")
    real_process_harness.signal(process, signal.SIGINT)
    assert real_process_harness.wait_for_exit(process) == 130
    real_process_harness.assert_tree_absent(target)
    probes = real_process_harness.wait_for_marker(f"{target}.probes").splitlines()
    assert probes == ["root", "identity"][: probes.index(target) + 1]
    activities = real_process_harness.wait_for_marker(
        f"{target}.activities"
    ).splitlines()
    assert len(activities) == len(probes)


@pytest.mark.parametrize(
    "target",
    ["root", "identity"],
)
def test_cancelled_probe_result_propagates_interruption(
    committed_git_repo: Path,
    target: str,
) -> None:
    binary_dir = committed_git_repo.parent / f"{target}-interrupt-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-runtime", "echo 'example 3.11.9'")
    plan = _complete_probe_plan(committed_git_repo)
    store = _trust_store(committed_git_repo)
    require_workspace_trust(
        RepoPaths.discover(committed_git_repo), store=store, explicit=True
    )
    cancel = CancellationToken()
    probes: list[str] = []

    def runner(command, **kwargs):
        name = _probe_name(command)
        if name is not None:
            probes.append(name)
        if name == target:
            cancel.cancel()
            return subprocess.CompletedProcess(command, -1, "", "")
        return run_captured(command, **kwargs)

    with pytest.raises(KeyboardInterrupt):
        HostPreflight(
            committed_git_repo,
            trust_store=store,
            environment={"PATH": str(binary_dir)},
            cancel=cancel,
            command_runner=runner,
        ).validate(
            plan,
            available_secret_names={"PACKAGE_TOKEN"},
        )

    assert probes[-1] == target


@pytest.mark.parametrize(
    "target",
    ["root", "identity"],
)
def test_each_concrete_probe_cancels_before_run_or_claim_creation(
    committed_git_repo: Path,
    target: str,
) -> None:
    binary_dir = committed_git_repo.parent / f"{target}-boundary-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-runtime", "echo 'example 3.11.9'")
    plan = _complete_probe_plan(committed_git_repo)
    trust_store = _trust_store(committed_git_repo)
    require_workspace_trust(
        RepoPaths.discover(committed_git_repo),
        store=trust_store,
        explicit=True,
    )
    cancel = CancellationToken()

    def runner(command, **kwargs):
        if _probe_name(command) == target:
            cancel.cancel()
            return subprocess.CompletedProcess(command, -1, "", "")
        return run_captured(command, **kwargs)

    borg_id = uuid4()
    generation_id = uuid4()
    with SqliteStore.open(committed_git_repo / f"{target}-boundary.sqlite3") as store:
        with pytest.raises(KeyboardInterrupt):
            preflight = HostPreflight(
                committed_git_repo,
                trust_store=trust_store,
                environment={"PATH": str(binary_dir)},
                cancel=cancel,
                command_runner=runner,
            )
            HostExecutionService(
                store,
                preflight,
                SimpleNamespace(plan=None),
                worktree_manager=SimpleNamespace(),
            ).run(
                borg_id,
                generation_id,
                plan,
                secret_values={"PACKAGE_TOKEN": "available"},
                cancel=cancel,
            )

        assert store.list_execution_runs(borg_id) == []
        with store.locked_connection() as connection:
            claim_count = connection.execute(
                "SELECT COUNT(*) FROM task_claims"
            ).fetchone()[0]
        assert claim_count == 0


def test_workspace_trust_blocks_before_analyzer_plan_is_loaded(
    committed_git_repo: Path,
) -> None:
    loaded = False

    def load_plan() -> dict[str, object]:
        nonlocal loaded
        loaded = True
        (committed_git_repo / "repository-context-was-read").read_text()
        return {}

    result = HostPreflight(
        committed_git_repo,
        trust_store=_trust_store(committed_git_repo),
        environment={},
    ).validate(load_plan)

    assert isinstance(result, HostPreflightBlock)
    assert not loaded
    assert "workspace trust is required" in result.reason
    assert "betterborg trust --yes" in result.reason


def test_declared_services_and_compose_files_leave_the_plan(
    committed_git_repo: Path,
) -> None:
    """The analysis still names services; the validated plan no longer does.

    Nothing on this host can run Docker, and the analysis selects a Compose
    service and an external one. Both used to refuse the run before a task
    was claimed.
    """
    binary_dir = committed_git_repo.parent / "host-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-runtime", "echo 'example 3.11.9'")
    (committed_git_repo / "package").mkdir()
    (committed_git_repo / "runtime.version").write_text("3.11.9\n", encoding="utf-8")
    (committed_git_repo / "compose.yml").write_text(
        "services:\n  postgres:\n    image: postgres:16\n",
        encoding="utf-8",
    )

    result = _preflight(
        committed_git_repo,
        environment={"PATH": str(binary_dir)},
    ).validate(
        lambda: _base_plan(),
        available_secret_names={"PACKAGE_TOKEN"},
    )

    assert isinstance(result, HostPreflightPlan)
    assert result.commands[0].cwd == "package"
    assert result.environment_files == (committed_git_repo / "runtime.version",)
    assert {tool.name for tool in result.executables} == {"example-runtime"}
    assert result.required_secret_names == ("PACKAGE_TOKEN",)
    assert result.package_managers == ()
    assert [secret.scope for secret in result.secret_requirements] == ["build"]
    assert not hasattr(result, "services")
    assert not hasattr(result, "compose_files")
    assert not hasattr(result, "compose_profiles")
    assert not hasattr(result, "compose_networks")
    assert not hasattr(result, "compose_volumes")
    assert not hasattr(result, "compose_build_services")


def test_preserves_prepare_and_materialize_command_phases(
    committed_git_repo: Path,
) -> None:
    binary_dir = committed_git_repo.parent / "environment-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "prepare-environment", "exit 0")
    _executable(binary_dir, "materialize-environment", "exit 0")
    _executable(binary_dir, "example-test", "exit 0")
    plan = {
        # A check, so the subject here stays the phases rather than the
        # refusal a run with no check would take.
        "command_catalog": {
            "source": "pyproject.toml",
            "commands": [
                {"stage": "test", "argv": ["example-test"], "verifies": True}
            ],
        },
        "environment": {
            "prepare_commands": [{"argv": ["prepare-environment"]}],
            "materialize_commands": [{"argv": ["materialize-environment"]}],
        },
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan)

    assert isinstance(result, HostPreflightPlan)
    assert [command.argv for command in result.commands] == [("example-test",)]
    assert [command.argv for command in result.prepare_commands] == [
        ("prepare-environment",)
    ]
    assert [command.argv for command in result.materialize_commands] == [
        ("materialize-environment",)
    ]


def test_aggregates_cwd_runtime_and_secret_failures_with_evidence(
    committed_git_repo: Path,
) -> None:
    plan = _base_plan()
    plan["command_catalog"]["commands"][0]["cwd"] = "../outside"
    plan["command_catalog"]["commands"][0]["uses_services"] = []
    plan["environment"]["prepare_commands"] = [
        {"argv": ["example-runtime", "-m", "build"], "source": "pyproject.toml"}
    ]
    plan["required_secrets"][0]["used_by"] = ["environment"]

    result = _preflight(committed_git_repo).validate(plan)

    assert isinstance(result, HostPreflightBlock)
    assert len(result.failures) == 3
    assert "repo-relative directory" in result.reason
    # The catalogue named a check; it was refused for its own shape, and that
    # refusal is the reason. A second one saying none was named would be false.
    assert "declares no command that verifies" not in result.reason
    assert "referenced environment file must exist" not in result.reason
    assert "host executable is required: example-runtime" in result.reason
    assert "required secret is not configured: PACKAGE_TOKEN" in result.reason
    assert "Betterborg will not install runtimes" in result.reason
    assert "pyproject.toml" in result.reason


def test_a_declared_environment_file_that_is_not_there_runs_anyway(
    committed_git_repo: Path,
) -> None:
    """A lockfile the checkout gitignores is still a file the analysis names."""
    binary_dir = committed_git_repo.parent / "declared-file-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-runtime", "exit 0")
    plan = {
        "command_catalog": {
            "source": "pyproject.toml",
            "commands": [
                {
                    "stage": "test",
                    "argv": ["example-runtime", "-m", "pytest"],
                    "verifies": True,
                }
            ],
        },
        "environment": {"files": ["package-lock.json"]},
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan)

    assert isinstance(result, HostPreflightPlan)
    assert [command.argv for command in result.commands] == [
        ("example-runtime", "-m", "pytest")
    ]


def test_a_toolchain_version_the_host_does_not_satisfy_runs_anyway(
    committed_git_repo: Path,
) -> None:
    """Every guard the comparison sat behind is cleared and it still runs.

    The pinned file is here, it cites the pinned version, the toolchain's
    program resolves, and a validated command invokes it, so the host's own
    version is the only thing left that could refuse this run.
    """
    binary_dir = committed_git_repo.parent / "version-bin"
    binary_dir.mkdir()
    runtime = _executable(binary_dir, "example-runtime", "echo 'example 3.12.1'")
    cited = committed_git_repo / "runtime.version"
    cited.write_text("3.11.9\n", encoding="utf-8")
    plan = _base_plan()
    plan["command_catalog"]["commands"][0]["cwd"] = "."
    plan["command_catalog"]["commands"][0]["uses_services"] = []
    plan["required_secrets"] = []
    plan["command_catalog"]["commands"][0]["required_secrets"] = []

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan)

    assert isinstance(result, HostPreflightPlan)
    assert result.environment_files == (cited,)
    assert [command.argv for command in result.commands] == [
        ("example-runtime", "-m", "pytest")
    ]
    assert result.executables == (
        HostExecutable("example-runtime", runtime, "3.11.9"),
    )


def test_version_probe_preserves_shim_dispatch_path(
    committed_git_repo: Path,
) -> None:
    binary_dir = committed_git_repo.parent / "shim-bin"
    binary_dir.mkdir()
    dispatcher = _executable(
        binary_dir,
        "runtime-manager",
        (
            "test \"${0##*/} $1\" = 'python3 --version' "
            "&& echo 'Python 3.13.7'"
        ),
    )
    shim = binary_dir / "python3"
    shim.symlink_to(dispatcher.name)
    (committed_git_repo / ".python-version").write_text(
        "3.13.7\n", encoding="utf-8"
    )
    plan = {
        "command_catalog": {
            "source": "pyproject.toml",
            "commands": [
                {"stage": "test", "argv": ["python3", "-m", "pytest"], "verifies": True}
            ],
        },
        "environment": {
            "files": [".python-version"],
            "toolchains": [
                {
                    "name": "python3",
                    "version": "3.13.7",
                    "source": ".python-version",
                }
            ],
        },
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan)

    assert isinstance(result, HostPreflightPlan)
    assert result.executables[0].path == shim
    assert result.executables[0].path.is_symlink()


@pytest.mark.parametrize(
    "optional_version",
    [{}, {"version": None}],
    ids=["omitted", "null"],
)
def test_unpinned_toolchain_only_requires_available_executable(
    committed_git_repo: Path,
    optional_version: dict[str, None],
) -> None:
    binary_dir = committed_git_repo.parent / f"{committed_git_repo.name}-unpinned-bin"
    binary_dir.mkdir()
    executable = _executable(binary_dir, "python", "exit 7")
    plan = {
        "command_catalog": {
            "source": "pyproject.toml",
            "commands": [
                {"stage": "test", "argv": ["python", "-m", "pytest"], "verifies": True}
            ],
        },
        "environment": {
            "toolchains": [{"name": "python", **optional_version}],
        },
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan)

    assert isinstance(result, HostPreflightPlan)
    assert result.executables[0].name == "python"
    assert result.executables[0].path == executable
    assert result.executables[0].version is None


def test_rust_toolchain_resolves_and_probes_rustc(
    committed_git_repo: Path,
) -> None:
    binary_dir = committed_git_repo.parent / "rust-bin"
    binary_dir.mkdir()
    rustc = _executable(
        binary_dir,
        "rustc",
        "test \"$1\" = --version && echo 'rustc 1.88.0 (example)'",
    )
    (committed_git_repo / "rust-toolchain.toml").write_text(
        '[toolchain]\nchannel = "1.88.0"\n', encoding="utf-8"
    )
    plan = {
        "command_catalog": {
            "source": "pyproject.toml",
            "commands": [
                {"stage": "test", "argv": ["rustc", "--version"], "verifies": True}
            ],
        },
        "environment": {
            "toolchains": [
                {
                    "name": "rust",
                    "version": "1.88.0",
                    "source": "rust-toolchain.toml",
                }
            ],
        },
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan)

    assert isinstance(result, HostPreflightPlan)
    assert result.executables[0].name == "rustc"
    assert result.executables[0].path == rustc
    assert result.executables[0].version == "1.88.0"


def test_a_toolchain_citing_a_file_that_is_not_there_runs_anyway(
    committed_git_repo: Path,
) -> None:
    """A cited version source is evidence a person reads, not a requirement."""
    binary_dir = committed_git_repo.parent / "cited-version-bin"
    binary_dir.mkdir()
    runtime = _executable(binary_dir, "example-runtime", "echo 'example 3.11.9'")
    (committed_git_repo / "runtime.version").write_text("3.11.9\n", encoding="utf-8")
    plan = {
        "command_catalog": {
            "source": "pyproject.toml",
            "commands": [
                {
                    "stage": "test",
                    "argv": ["example-runtime", "-m", "pytest"],
                    "verifies": True,
                }
            ],
        },
        "environment": {
            "files": ["runtime.version"],
            "toolchains": [
                {
                    "name": "example-runtime",
                    "version": "3.11.9",
                    "source": "missing.version",
                }
            ],
        },
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan)

    assert isinstance(result, HostPreflightPlan)
    assert [command.argv for command in result.commands] == [
        ("example-runtime", "-m", "pytest")
    ]
    assert result.executables == (
        HostExecutable("example-runtime", runtime, "3.11.9"),
    )


def test_command_derived_failures_retain_exact_source_evidence(
    committed_git_repo: Path,
) -> None:
    source = "pyproject.toml#tool.pytest.ini_options"
    binary_dir = committed_git_repo.parent / "evidence-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-runtime", "exit 0")
    plan = {
        "command_catalog": {
            "commands": [
                {
                    "stage": "test",
                    "argv": ["example-runtime", "-m", "pytest"],
                    "source": source,
                    "required_secrets": ["PACKAGE_TOKEN"],
                    "uses_services": ["database"],
                }
            ]
        },
        "environment": {
            "prepare_commands": [{"argv": ["missing-runtime"], "source": source}]
        },
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan)

    assert isinstance(result, HostPreflightBlock)
    # The command names a service no dependency declares, which is no longer
    # anything to refuse.
    assert len(result.failures) == 2
    assert all(failure.evidence == source for failure in result.failures)
    assert "host executable is required: missing-runtime" in result.reason
    assert "undeclared required secret: PACKAGE_TOKEN" in result.reason
    assert "database" not in result.reason


def test_person_readable_toolchain_inventory_does_not_block_a_run(
    committed_git_repo: Path,
) -> None:
    binary_dir = committed_git_repo.parent / "inventory-bin"
    binary_dir.mkdir()
    runtime = _executable(binary_dir, "example-runtime", "exit 0")
    plan = {
        "command_catalog": {
            "commands": [{"stage": "test", "argv": ["example-runtime"]}]
        },
        "environment": {
            "package_managers": ["Go modules"],
            "toolchains": [{"name": "Node.js"}, {"name": "Go modules"}],
        },
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan)

    assert isinstance(result, HostPreflightPlan)
    assert result.dropped_commands == ()
    assert result.executables == (
        HostExecutable("example-runtime", runtime, None),
    )
    assert [command.argv for command in result.commands] == [("example-runtime",)]


def test_command_with_no_host_program_is_dropped_and_named(
    committed_git_repo: Path,
) -> None:
    binary_dir = committed_git_repo.parent / "partial-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-lint", "exit 0")
    plan = {
        "command_catalog": {
            "source": "pyproject.toml",
            "commands": [
                {"stage": "lint", "argv": ["example-lint", "--all"]},
                {
                    "stage": "test",
                    "argv": ["missing-runtime", "-m", "pytest"],
                    "source": "pyproject.toml#test",
                },
            ],
        }
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan)

    assert isinstance(result, HostPreflightPlan)
    assert [command.argv for command in result.commands] == [
        ("example-lint", "--all")
    ]
    assert [dropped.command.argv for dropped in result.dropped_commands] == [
        ("missing-runtime", "-m", "pytest")
    ]
    assert result.dropped_command_summary == (
        "1 sanity command dropped: missing-runtime -m pytest: host executable "
        "is not available: missing-runtime (evidence: pyproject.toml#test)"
    )


def test_environment_command_program_is_still_required(
    committed_git_repo: Path,
) -> None:
    plan = {
        "environment": {
            "prepare_commands": [{"argv": ["missing-runtime", "install"]}]
        }
    }

    result = _preflight(committed_git_repo).validate(plan)

    assert isinstance(result, HostPreflightBlock)
    assert "host executable is required: missing-runtime" in result.reason


def test_secret_named_only_by_a_dropped_command_does_not_block(
    committed_git_repo: Path,
) -> None:
    binary_dir = committed_git_repo.parent / "secret-scope-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-lint", "exit 0")
    plan = {
        "command_catalog": {
            "source": "pyproject.toml",
            "commands": [
                {"stage": "lint", "argv": ["example-lint"]},
                {
                    "stage": "test",
                    "argv": ["missing-runtime"],
                    "required_secrets": ["PACKAGE_TOKEN"],
                },
            ],
        },
        "required_secrets": [
            {
                "name": "PACKAGE_TOKEN",
                "used_by": ["test"],
                "scope": "build",
                "source": "pyproject.toml",
            },
            {
                "name": "DEPLOY_TOKEN",
                "used_by": ["deploy"],
                "scope": "build",
                "source": ".github/workflows/deploy.yml",
            },
        ],
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan, available_secret_names=set())

    assert isinstance(result, HostPreflightPlan)
    assert result.required_secret_names == ("DEPLOY_TOKEN", "PACKAGE_TOKEN")


def test_a_secret_a_surviving_command_consumes_still_blocks(
    committed_git_repo: Path,
) -> None:
    """Following the commands must not become excusing every secret.

    The surviving half of the rule is the load-bearing one: a command that
    will run and needs a secret still cannot run without it, and finding that
    out at preflight is the whole point of asking.
    """
    binary_dir = committed_git_repo.parent / "surviving-secret-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-test", "exit 0")
    plan = {
        "command_catalog": {
            "source": "pyproject.toml",
            "commands": [
                {
                    "stage": "test",
                    "argv": ["example-test"],
                    "required_secrets": ["PACKAGE_TOKEN"],
                }
            ],
        },
        "required_secrets": [
            {
                "name": "PACKAGE_TOKEN",
                "used_by": ["test"],
                "scope": "build",
                "source": "pyproject.toml",
            }
        ],
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan, available_secret_names=set())

    assert isinstance(result, HostPreflightBlock)
    assert "PACKAGE_TOKEN" in result.reason


def test_agent_scoped_secret_blocks_because_the_agents_always_run(
    committed_git_repo: Path,
) -> None:
    plan = {
        "required_secrets": [
            {
                "name": "AGENT_TOKEN",
                "used_by": ["coding"],
                "scope": "agent",
                "source": "AGENTS.md",
            }
        ]
    }

    result = _preflight(committed_git_repo).validate(
        plan, available_secret_names=set()
    )

    assert isinstance(result, HostPreflightBlock)
    assert "required secret is not configured: AGENT_TOKEN" in result.reason


def test_identically_repeated_secret_records_are_accepted(
    committed_git_repo: Path,
) -> None:
    binary_dir = committed_git_repo.parent / "repeated-secret-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-test", "exit 0")
    plan = {
        "command_catalog": {
            "source": "pyproject.toml",
            "commands": [
                {"stage": "test", "argv": ["example-test"], "verifies": True}
            ],
        },
        "required_secrets": [
            {
                "name": "PACKAGE_TOKEN",
                "used_by": ["build", "test"],
                "scope": "build",
                "source": ".github/workflows/ci.yml",
            },
            {
                "name": "PACKAGE_TOKEN",
                "used_by": ["test", "build"],
                "scope": "build",
                "source": ".github/workflows/release.yml",
            },
        ]
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(
        plan, available_secret_names={"PACKAGE_TOKEN"}
    )

    assert isinstance(result, HostPreflightPlan)
    assert result.secret_requirements == (
        HostSecret(
            "PACKAGE_TOKEN",
            "build",
            ("build", "test"),
            ".github/workflows/ci.yml",
        ),
    )


@pytest.mark.parametrize(
    ("repeated", "expected"),
    [
        (
            {"used_by": ["test"], "scope": "agent"},
            "disagree on scope 'build' and 'agent'",
        ),
    ],
    ids=["scope"],
)
def test_conflicting_repeated_secret_records_say_what_disagrees(
    committed_git_repo: Path,
    repeated: dict[str, object],
    expected: str,
) -> None:
    plan = {
        "required_secrets": [
            {
                "name": "PACKAGE_TOKEN",
                "used_by": ["test"],
                "scope": "build",
                "source": ".github/workflows/ci.yml",
            },
            {
                "name": "PACKAGE_TOKEN",
                "source": ".github/workflows/release.yml",
                **repeated,
            },
        ]
    }

    result = _preflight(committed_git_repo).validate(
        plan, available_secret_names={"PACKAGE_TOKEN"}
    )

    assert isinstance(result, HostPreflightBlock)
    assert "required secret name must be unambiguous: PACKAGE_TOKEN" in (
        result.reason
    )
    assert expected in result.reason
    assert ".github/workflows/ci.yml, .github/workflows/release.yml" in (
        result.reason
    )


def test_repeated_secret_records_accumulate_the_purposes_they_name(
    committed_git_repo: Path,
) -> None:
    """Two workflows needing one secret is not a disagreement about it.

    Scope decides how a value is supplied and cannot be two things at once.
    What each workflow wants the secret for accumulates, and refusing the run
    over that stops it for a secret nothing disagrees about, while keeping
    only the first record would leave it unrequired for the stage the second
    is the sole evidence for.
    """
    binary_dir = committed_git_repo.parent / "merged-secret-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-test", "exit 0")
    _executable(binary_dir, "example-release", "exit 0")
    plan = {
        "command_catalog": {
            "source": "pyproject.toml",
            "commands": [
                {"stage": "test", "argv": ["example-test"]},
                {"stage": "release", "argv": ["example-release"]},
            ],
        },
        "required_secrets": [
            {
                "name": "PACKAGE_TOKEN",
                "used_by": ["test"],
                "scope": "build",
                "source": ".github/workflows/ci.yml",
            },
            {
                "name": "PACKAGE_TOKEN",
                "used_by": ["release"],
                "scope": "build",
                "source": ".github/workflows/release.yml",
            },
        ],
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan, available_secret_names={"PACKAGE_TOKEN"})

    assert isinstance(result, HostPreflightPlan)
    assert result.required_secret_names == ("PACKAGE_TOKEN",)
    secret = next(
        item
        for item in result.secret_requirements
        if item.name == "PACKAGE_TOKEN"
    )
    assert set(secret.used_by) == {"test", "release"}


def test_host_that_satisfies_everything_produces_the_unchanged_plan(
    committed_git_repo: Path,
) -> None:
    binary_dir = committed_git_repo.parent / "complete-bin"
    binary_dir.mkdir()
    runtime = _executable(binary_dir, "example-runtime", "echo 'example 3.11.9'")
    npm = _executable(binary_dir, "npm", "exit 0")
    (committed_git_repo / "runtime.version").write_text(
        "3.11.9\n", encoding="utf-8"
    )
    plan = {
        "command_catalog": {
            "source": "pyproject.toml",
            "commands": [
                {
                    "stage": "test",
                    "argv": ["example-runtime", "-m", "pytest"],
                    "required_secrets": ["PACKAGE_TOKEN"],
                }
            ],
        },
        "environment": {
            "files": ["runtime.version"],
            "package_managers": ["npm"],
            "toolchains": [
                {
                    "name": "example-runtime",
                    "version": "3.11.9",
                    "source": "runtime.version",
                }
            ],
        },
        "required_secrets": [
            {
                "name": "PACKAGE_TOKEN",
                "used_by": ["test"],
                "scope": "build",
                "source": "pyproject.toml",
            }
        ],
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan, available_secret_names={"PACKAGE_TOKEN"})

    assert result == HostPreflightPlan(
        repository_root=committed_git_repo.resolve(),
        commands=(
            HostCommand(
                "test",
                ("example-runtime", "-m", "pytest"),
                ".",
                "pyproject.toml",
            ),
        ),
        prepare_commands=(),
        materialize_commands=(),
        environment_files=(committed_git_repo / "runtime.version",),
        executables=(
            HostExecutable("example-runtime", runtime, "3.11.9"),
            HostExecutable("npm", npm, None),
        ),
        required_secret_names=("PACKAGE_TOKEN",),
        package_managers=("npm",),
        secret_requirements=(
            HostSecret("PACKAGE_TOKEN", "build", ("test",), "pyproject.toml"),
        ),
    )


def test_the_gate_runs_only_the_commands_that_declare_they_verify(
    committed_git_repo: Path,
) -> None:
    """The catalogue lists what a repository can do; the gate proves a change.

    A docs watch server and a test target are one word apart in a package
    manifest, so the entry says which of them settles whether a change broke
    the repository, and only that one runs.
    """
    binary_dir = committed_git_repo.parent / "verifies-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-npm", "exit 0")
    plan = {
        "command_catalog": {
            "source": "package.json",
            "commands": [
                {
                    "stage": "test",
                    "argv": ["example-npm", "run", "test"],
                    "verifies": True,
                },
                {
                    "stage": "docs-development",
                    "argv": ["example-npm", "run", "dev"],
                    "verifies": False,
                },
                {
                    "stage": "release",
                    "argv": ["example-npm", "run", "release"],
                    "verifies": False,
                },
            ],
        }
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan)

    assert isinstance(result, HostPreflightPlan)
    assert [command.argv for command in result.commands] == [
        ("example-npm", "run", "test")
    ]
    assert result.dropped_commands == ()


def test_a_catalogue_that_declares_nothing_runs_every_command(
    committed_git_repo: Path,
) -> None:
    """An analysis recorded before the question was asked answers it by silence.

    Treating that silence as "not a check" would quietly stop running a
    repository's tests, and the run would look exactly like one that passed
    them.
    """
    binary_dir = committed_git_repo.parent / "silent-catalogue-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-npm", "exit 0")
    plan = {
        "command_catalog": {
            "source": "package.json",
            "commands": [
                {"stage": "test", "argv": ["example-npm", "run", "test"]},
                {"stage": "build", "argv": ["example-npm", "run", "build"]},
            ],
        }
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan)

    assert isinstance(result, HostPreflightPlan)
    assert [command.argv for command in result.commands] == [
        ("example-npm", "run", "test"),
        ("example-npm", "run", "build"),
    ]


def test_a_secret_only_a_non_verifying_command_names_does_not_block(
    committed_git_repo: Path,
) -> None:
    """A command the gate will not run states no requirement for the run."""
    binary_dir = committed_git_repo.parent / "publish-secret-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-npm", "exit 0")
    plan = {
        "command_catalog": {
            "source": "package.json",
            "commands": [
                {
                    "stage": "test",
                    "argv": ["example-npm", "run", "test"],
                    "verifies": True,
                },
                {
                    "stage": "release",
                    "argv": ["example-npm", "run", "release"],
                    "verifies": False,
                    "required_secrets": ["NPM_PUBLISH_TOKEN"],
                },
            ],
        },
        "required_secrets": [
            {
                "name": "NPM_PUBLISH_TOKEN",
                "used_by": ["release"],
                "scope": "build",
                "source": "package.json#release",
            }
        ],
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan, available_secret_names=set())

    assert isinstance(result, HostPreflightPlan)
    assert result.required_secret_names == ("NPM_PUBLISH_TOKEN",)


def test_a_surviving_command_makes_the_secret_it_names_this_runs_requirement(
    committed_git_repo: Path,
) -> None:
    """The command that names the secret says it needs it.

    Nothing in the analyzer contract makes a secret's used_by spell a catalog
    stage; a workflow-derived record plausibly names the job instead. Matching
    only on that string lets a secret a running command declared slip through
    preflight and fail the command at sanity, after coding, review and merge
    have been paid for.
    """
    binary_dir = committed_git_repo.parent / "named-secret-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-test", "exit 0")
    plan = {
        "command_catalog": {
            "source": "pyproject.toml",
            "commands": [
                {
                    "stage": "test",
                    "argv": ["example-test"],
                    "verifies": True,
                    "required_secrets": ["PACKAGE_TOKEN"],
                }
            ],
        },
        "required_secrets": [
            {
                "name": "PACKAGE_TOKEN",
                "used_by": ["ci"],
                "scope": "build",
                "source": ".github/workflows/ci.yml",
            }
        ],
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan, available_secret_names=set())

    assert isinstance(result, HostPreflightBlock)
    assert "required secret is not configured: PACKAGE_TOKEN" in result.reason


def test_a_host_that_can_run_no_catalogued_check_is_refused_before_the_spend(
    committed_git_repo: Path,
) -> None:
    """Dropping is a trade, and it stops paying when nothing survives.

    A run with no check left cannot publish anything: every task would be
    coded, reviewed and merged, and every one would then block. The refusal
    the stage removed was worth keeping for exactly this case, where the
    alternative is not a stricter run but the same no run, after the spend.
    """
    plan = {
        "command_catalog": {
            "source": "Cargo.toml",
            "commands": [
                {
                    "stage": "test",
                    "argv": ["absent-cargo", "test"],
                    "verifies": True,
                }
            ],
        }
    }

    result = _preflight(committed_git_repo).validate(plan)

    assert isinstance(result, HostPreflightBlock)
    assert "no catalogued check can run on this host" in result.reason
    # The program, not the command: a refusal reason is never redacted, and a
    # catalogued argv can carry a secret the repository spelled into a script.
    assert "absent-cargo" in result.reason
    assert "absent-cargo test" not in result.reason


def test_a_catalogue_declaring_no_check_is_refused_before_the_spend(
    committed_git_repo: Path,
) -> None:
    """A catalogue of servers and deploys leaves the same run as a dropped one.

    Nothing left could prove a change safe, so every task would be coded,
    reviewed and merged and every one would then block. The refusal belongs
    where the other empty gate is refused, and the reason has to name the
    declaration that emptied it.
    """
    binary_dir = committed_git_repo.parent / "no-check-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-npm", "exit 0")
    plan = {
        "command_catalog": {
            "source": "package.json",
            "commands": [
                {
                    "stage": "docs-development",
                    "argv": ["example-npm", "run", "dev"],
                    "verifies": False,
                },
                {
                    "stage": "release",
                    "argv": ["example-npm", "run", "release"],
                    "verifies": False,
                },
            ],
        }
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan)

    assert isinstance(result, HostPreflightBlock)
    assert "declares no command that verifies the repository" in result.reason
    assert "package.json" in result.reason


def test_a_non_verifying_command_requires_no_program_and_is_not_a_drop(
    committed_git_repo: Path,
) -> None:
    """A command the gate will not run states no requirement of the host.

    Nor is it a dropped check: nothing was given up, so naming it beside the
    checks this host could not run would report a loss the run did not take.
    """
    binary_dir = committed_git_repo.parent / "non-verifying-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-test", "exit 0")
    plan = {
        "command_catalog": {
            "source": "package.json",
            "commands": [
                {
                    "stage": "test",
                    "argv": ["example-test"],
                    "verifies": True,
                },
                {
                    "stage": "deploy",
                    "argv": ["absent-deployer", "ship"],
                    "verifies": False,
                },
            ],
        }
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan)

    assert isinstance(result, HostPreflightPlan)
    assert [command.argv for command in result.commands] == [("example-test",)]
    assert result.dropped_commands == ()
    assert result.dropped_command_summary == ""
    assert [tool.name for tool in result.executables] == ["example-test"]


def test_the_secret_a_command_named_reaches_the_command_that_named_it(
    committed_git_repo: Path,
) -> None:
    """Requiring a secret and delivering it must answer the same question.

    Preflight makes the operator configure a secret because a command said it
    needs one. If the phase that runs the command answers "which commands ask"
    differently, the command runs without it and fails after coding, review
    and merge have been paid for: the whole failure preflight exists to move
    forward in time, moved back again.
    """
    from betterborg_cli.host_execution.environment import (
        command_secret_environment,
    )

    binary_dir = committed_git_repo.parent / "delivered-secret-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-test", "exit 0")
    plan = {
        "command_catalog": {
            "source": "pyproject.toml",
            "commands": [
                {
                    "stage": "test",
                    "argv": ["example-test"],
                    "verifies": True,
                    "required_secrets": ["PACKAGE_TOKEN"],
                }
            ],
        },
        "required_secrets": [
            {
                "name": "PACKAGE_TOKEN",
                # The workflow job, not the catalog stage.
                "used_by": ["ci"],
                "scope": "build",
                "source": ".github/workflows/ci.yml",
            }
        ],
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan, available_secret_names={"PACKAGE_TOKEN"})

    assert isinstance(result, HostPreflightPlan)
    assert result.required_secret_names == ("PACKAGE_TOKEN",)
    environment = command_secret_environment(
        result, result.commands[0].stage, {}, {"PACKAGE_TOKEN": "s3cr3t"}
    )
    assert environment == {"PACKAGE_TOKEN": "s3cr3t"}


def test_a_materialize_command_the_host_cannot_run_still_refuses_the_run(
    committed_git_repo: Path,
) -> None:
    """Materialize builds the run itself, so it is never dropped."""
    plan = {
        "environment": {
            "materialize_commands": [{"argv": ["missing-runtime", "sync"]}]
        }
    }

    result = _preflight(committed_git_repo).validate(plan)

    assert isinstance(result, HostPreflightBlock)
    assert "host executable is required: missing-runtime" in result.reason


def test_a_scope_all_secret_blocks_whatever_the_commands_name(
    committed_git_repo: Path,
) -> None:
    """Scope 'all' reaches every phase, so no command has to ask for it."""
    binary_dir = committed_git_repo.parent / "scope-all-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-test", "exit 0")
    plan = {
        "command_catalog": {
            "source": "pyproject.toml",
            "commands": [
                {"stage": "test", "argv": ["example-test"], "verifies": True}
            ],
        },
        "required_secrets": [
            {
                "name": "SHARED_TOKEN",
                "used_by": ["deploy"],
                "scope": "all",
                "source": ".env.example",
            }
        ],
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan, available_secret_names=set())

    assert isinstance(result, HostPreflightBlock)
    assert "required secret is not configured: SHARED_TOKEN" in result.reason


def test_a_program_named_by_path_is_resolved_against_its_own_directory(
    committed_git_repo: Path,
) -> None:
    """A bare name is found on PATH; a path is found where the command runs."""
    tools = committed_git_repo / "tools"
    tools.mkdir()
    _executable(tools, "check.sh", "exit 0")
    plan = {
        "command_catalog": {
            "source": "pyproject.toml",
            "commands": [
                {
                    "stage": "test",
                    "argv": ["./check.sh"],
                    "cwd": "tools",
                    "verifies": True,
                }
            ],
        }
    }

    result = _preflight(committed_git_repo, environment={"PATH": ""}).validate(plan)

    assert isinstance(result, HostPreflightPlan)
    assert [command.argv for command in result.commands] == [("./check.sh",)]
    assert result.dropped_commands == ()


@pytest.mark.parametrize(
    "catalog",
    [None, {"source": "Makefile"}, {"source": "Makefile", "commands": []}],
    ids=["no-catalog", "no-commands-key", "empty-commands"],
)
def test_an_analysis_naming_no_check_at_all_is_refused(
    committed_git_repo: Path,
    catalog: dict[str, object] | None,
) -> None:
    """The commonest way to hold no check is to have catalogued nothing.

    The command catalog is optional, and the analyzer is told to omit a
    category it has no evidence for, so this is a state the producer is
    instructed to reach. It leaves the same run as a catalog of servers: every
    task coded, reviewed and merged, and every one blocked at the end.
    """
    plan: dict[str, object] = {}
    if catalog is not None:
        plan["command_catalog"] = catalog

    result = _preflight(committed_git_repo).validate(plan)

    assert isinstance(result, HostPreflightBlock)
    assert "declares no command that verifies the repository" in result.reason


def test_a_command_naming_an_agent_scoped_secret_is_refused(
    committed_git_repo: Path,
) -> None:
    """Requiring and delivering must be able to agree, or the run cannot work.

    An agent-scoped secret reaches the agent phases and no command. Accepting
    a command that names one makes the operator configure a value nothing will
    hand to that command, and the command fails at sanity after coding, review
    and merge have been paid for.
    """
    binary_dir = committed_git_repo.parent / "agent-scope-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-test", "exit 0")
    plan = {
        "command_catalog": {
            "source": "pyproject.toml",
            "commands": [
                {
                    "stage": "test",
                    "argv": ["example-test"],
                    "verifies": True,
                    "required_secrets": ["PACKAGE_TOKEN"],
                }
            ],
        },
        "required_secrets": [
            {
                "name": "PACKAGE_TOKEN",
                "used_by": ["ci"],
                "scope": "agent",
                "source": ".github/workflows/ci.yml",
            }
        ],
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan, available_secret_names={"PACKAGE_TOKEN"})

    assert isinstance(result, HostPreflightBlock)
    assert "scoped to the agents but named by a command that runs" in result.reason
    assert "test" in result.reason


def test_a_secret_only_a_dropped_command_names_is_not_required_to_be_declared(
    committed_git_repo: Path,
) -> None:
    """Following the commands covers the undeclared case as well as the named one.

    A dropped command's requirements are not this run's, so a secret only it
    names is not one the analysis had to declare either.
    """
    binary_dir = committed_git_repo.parent / "undeclared-drop-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-lint", "exit 0")
    plan = {
        "command_catalog": {
            "source": "pyproject.toml",
            "commands": [
                {"stage": "lint", "argv": ["example-lint"], "verifies": True},
                {
                    "stage": "test",
                    "argv": ["missing-runtime"],
                    "verifies": True,
                    "required_secrets": ["GHOST_TOKEN"],
                },
            ],
        }
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan, available_secret_names=set())

    assert isinstance(result, HostPreflightPlan)
    assert result.required_secret_names == ()
    assert [dropped.command.argv[0] for dropped in result.dropped_commands] == [
        "missing-runtime"
    ]


def test_a_non_verifying_command_needs_no_directory_of_this_host(
    committed_git_repo: Path,
) -> None:
    """Whether a directory is here is a fact about this checkout.

    A command the gate will never enter does not need it: an uninitialised
    docs submodule would otherwise refuse the whole run over a directory
    nothing opens. The shape of the path is still the analysis's to get right.
    """
    binary_dir = committed_git_repo.parent / "absent-dir-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-npm", "exit 0")
    plan = {
        "command_catalog": {
            "source": "package.json",
            "commands": [
                {
                    "stage": "test",
                    "argv": ["example-npm", "test"],
                    "verifies": True,
                },
                {
                    "stage": "docs",
                    "argv": ["example-npm", "run", "dev"],
                    "verifies": False,
                    "cwd": "website",
                },
            ],
        }
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan)

    assert isinstance(result, HostPreflightPlan)
    assert [command.argv for command in result.commands] == [
        ("example-npm", "test")
    ]


def test_a_verifying_command_still_needs_its_directory_to_exist(
    committed_git_repo: Path,
) -> None:
    """The gate will enter this one, so the directory has to be there."""
    binary_dir = committed_git_repo.parent / "present-dir-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-npm", "exit 0")
    plan = {
        "command_catalog": {
            "source": "package.json",
            "commands": [
                {
                    "stage": "test",
                    "argv": ["example-npm", "test"],
                    "verifies": True,
                    "cwd": "website",
                }
            ],
        }
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan)

    assert isinstance(result, HostPreflightBlock)
    assert "repo-relative directory" in result.reason


def test_the_refusal_does_not_quote_a_catalogued_command_verbatim(
    committed_git_repo: Path,
) -> None:
    """A block reason is never redacted, so it must quote nothing secret.

    It reaches the terminal and the headless payload, and a repository that
    spells a token into a script has it in the analysis.
    """
    plan = {
        "command_catalog": {
            "source": "Makefile",
            "commands": [
                {
                    "stage": "test",
                    "argv": ["absent-runner", "--token", "s3cr3t-value"],
                    "verifies": True,
                }
            ],
        }
    }

    result = _preflight(committed_git_repo).validate(plan)

    assert isinstance(result, HostPreflightBlock)
    assert "absent-runner" in result.reason
    assert "s3cr3t-value" not in result.reason


def test_a_catalogue_refused_for_its_own_shape_is_not_reported_as_empty(
    committed_git_repo: Path,
) -> None:
    """A refusal says one true thing about why the run stopped.

    The catalogue named a check. It was rejected for the directory it names,
    and that rejection is the reason. Telling the operator the analysis names
    no check as well sends them to the wrong file.
    """
    binary_dir = committed_git_repo.parent / "refused-shape-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-test", "exit 0")
    plan = {
        "command_catalog": {
            "source": "Makefile",
            "commands": [
                {
                    "stage": "test",
                    "argv": ["example-test"],
                    "verifies": True,
                    "cwd": "absent-directory",
                }
            ],
        }
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan)

    assert isinstance(result, HostPreflightBlock)
    assert "repo-relative directory" in result.reason
    assert "declares no command that verifies" not in result.reason
