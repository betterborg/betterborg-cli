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
from betterborg_cli.agent_runtime.mock import MockAdapter, MockResponse
from betterborg_cli.host_execution import (
    HostExecutionService,
    HostPreflight,
    HostPreflightBlock,
    HostPreflightPlan,
    service_url_environment,
)
from betterborg_cli.progress import AgentActivity, AgentActivityKind
from betterborg_cli.repo_analysis import DIMENSIONS, run_analyzer
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.store import Repository, SqliteStore
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


def _compose_executable(
    directory: Path,
    *,
    version: str = "2.30.0",
    model: dict[str, object] | None = None,
) -> Path:
    resolved_model = model or {
        "services": {"postgres": {}},
        "networks": {"default": {}},
    }
    return _executable(
        directory,
        "docker",
        (
            "if test \"$1 $2\" = 'compose version'; then "
            f"echo 'Docker Compose v{version}'; exit 0; fi\n"
            f"printf '%s\\n' {shlex.quote(json.dumps(resolved_model))}"
        ),
    )


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
    """Prepare one plan that reaches every direct preflight probe."""
    (repository / "package").mkdir(exist_ok=True)
    (repository / "runtime.version").write_text("3.11.9\n", encoding="utf-8")
    (repository / "compose.yml").write_text(
        "services:\n  postgres:\n    image: postgres:16\n",
        encoding="utf-8",
    )
    return _base_plan()


def _probe_name(command: list[str] | tuple[str, ...]) -> str | None:
    """Identify the five ordered direct probes covered by cancellation tests."""
    argv = tuple(command)
    if argv[-1:] == ("--show-toplevel",):
        return "root"
    if argv[-1:] == ("--git-common-dir",):
        return "identity"
    if argv[-2:] == ("compose", "version"):
        return "compose-version"
    if argv[-3:] == ("config", "--format", "json"):
        return "topology"
    if argv[-1:] == ("--version",):
        return "executable-version"
    return None


def test_all_direct_probes_share_token_runner_and_command_activity(
    committed_git_repo: Path,
) -> None:
    binary_dir = committed_git_repo.parent / "all-probe-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-runtime", "echo 'example 3.11.9'")
    _compose_executable(binary_dir)
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
        external_urls={"SEARCH_URL": "https://search.example.test/api"},
    )

    assert isinstance(result, HostPreflightPlan)
    assert [_probe_name(command) for command, _token in calls] == [
        "root",
        "identity",
        "executable-version",
        "compose-version",
        "topology",
    ]
    assert all(observed is cancel for _command, observed in calls)
    assert [activity.kind for activity in activities] == [
        AgentActivityKind.COMMAND
    ] * 5
    assert [activity.detail for activity in activities] == [
        shlex.join(command) for command, _token in calls
    ]


@pytest.mark.parametrize(
    "target",
    ["root", "identity", "executable-version", "compose-version", "topology"],
)
def test_cancellation_reaps_each_direct_probe_and_stops_later_probes(
    committed_git_repo: Path,
    real_process_harness,
    target: str,
) -> None:
    binary_dir = committed_git_repo.parent / f"{target}-cancel-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-runtime", "echo 'example 3.11.9'")
    _compose_executable(binary_dir)
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
            external_urls={"SEARCH_URL": "https://search.example.test/api"},
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
    assert probes == [
        "root",
        "identity",
        "executable-version",
        "compose-version",
        "topology",
    ][: probes.index(target) + 1]
    activities = real_process_harness.wait_for_marker(
        f"{target}.activities"
    ).splitlines()
    assert len(activities) == len(probes)


@pytest.mark.parametrize(
    "target",
    ["root", "identity", "executable-version", "compose-version", "topology"],
)
def test_cancelled_probe_result_propagates_interruption(
    committed_git_repo: Path,
    target: str,
) -> None:
    binary_dir = committed_git_repo.parent / f"{target}-interrupt-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-runtime", "echo 'example 3.11.9'")
    _compose_executable(binary_dir)
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
            external_urls={"SEARCH_URL": "https://search.example.test/api"},
        )

    assert probes[-1] == target


@pytest.mark.parametrize(
    "target",
    ["root", "identity", "executable-version", "compose-version", "topology"],
)
def test_each_concrete_probe_cancels_before_run_or_claim_creation(
    committed_git_repo: Path,
    target: str,
) -> None:
    binary_dir = committed_git_repo.parent / f"{target}-boundary-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-runtime", "echo 'example 3.11.9'")
    _compose_executable(binary_dir)
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
                compose_manager=SimpleNamespace(),
            ).run(
                borg_id,
                generation_id,
                plan,
                secret_values={"PACKAGE_TOKEN": "available"},
                external_urls={
                    "SEARCH_URL": "https://search.example.test/api"
                },
                cancel=cancel,
            )

        assert store.list_execution_runs(borg_id) == []
        with store.locked_connection() as connection:
            claim_count = connection.execute(
                "SELECT COUNT(*) FROM task_claims"
            ).fetchone()[0]
        assert claim_count == 0


@pytest.mark.parametrize("outcome", ["timeout", "nonzero"])
def test_ordinary_probe_failures_remain_host_blocks(
    committed_git_repo: Path,
    outcome: str,
) -> None:
    binary_dir = committed_git_repo.parent / f"ordinary-{outcome}-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-runtime", "echo 'example 3.11.9'")
    (committed_git_repo / "runtime.version").write_text(
        "3.11.9\n", encoding="utf-8"
    )
    store = _trust_store(committed_git_repo)
    require_workspace_trust(
        RepoPaths.discover(committed_git_repo), store=store, explicit=True
    )

    def runner(command, **kwargs):
        if _probe_name(command) == "executable-version":
            if outcome == "timeout":
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            return subprocess.CompletedProcess(command, 7, "", "probe failed")
        return run_captured(command, **kwargs)

    result = HostPreflight(
        committed_git_repo,
        trust_store=store,
        environment={"PATH": str(binary_dir)},
        command_runner=runner,
    ).validate(
        {
            "environment": {
                "files": ["runtime.version"],
                "toolchains": [
                    {
                        "name": "example-runtime",
                        "version": "3.11.9",
                        "source": "runtime.version",
                    }
                ],
            }
        }
    )

    assert isinstance(result, HostPreflightBlock)
    assert "host executable 'example-runtime' must satisfy" in result.reason


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


def test_validates_complete_plan_and_ignores_unselected_service(
    committed_git_repo: Path,
) -> None:
    binary_dir = committed_git_repo.parent / "host-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-runtime", "echo 'example 3.11.9'")
    docker = _compose_executable(
        binary_dir,
        model={
            "services": {
                "postgres": {
                    "build": {"context": "."},
                    "image": "betterborg/shared-postgres:dev",
                }
            },
            "networks": {"backend": {}},
            "volumes": {"database": {}},
        },
    )
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
        external_urls={"SEARCH_URL": "https://search.example.test/api"},
    )

    assert isinstance(result, HostPreflightPlan)
    assert result.commands[0].cwd == "package"
    assert result.environment_files == (committed_git_repo / "runtime.version",)
    assert {tool.name for tool in result.executables} == {
        "docker",
        "example-runtime",
    }
    assert next(tool.path for tool in result.executables if tool.name == "docker") == (
        docker
    )
    assert result.required_secret_names == ("PACKAGE_TOKEN",)
    assert result.package_managers == ()
    assert [secret.scope for secret in result.secret_requirements] == ["build"]
    assert result.compose_files == (committed_git_repo / "compose.yml",)
    assert result.compose_networks == ("backend",)
    assert result.compose_volumes == ("database",)
    assert result.compose_build_services == ("postgres",)
    assert [(service.name, service.kind) for service in result.services] == [
        ("database", "compose"),
        ("search", "external"),
    ]
    assert result.services[0].url_targets == (("DATABASE_URL", 5432, "tcp"),)
    assert result.services[1].url == "https://search.example.test/api"
    assert service_url_environment(result.services) == {
        "SEARCH_URL": "https://search.example.test/api",
    }
    assert service_url_environment(
        result.services, published_ports={("postgres", 5432, "tcp"): 49152}
    ) == {
        "DATABASE_URL": "postgres://127.0.0.1:49152/postgres",
        "SEARCH_URL": "https://search.example.test/api",
    }


def test_preserves_prepare_and_materialize_command_phases(
    committed_git_repo: Path,
) -> None:
    binary_dir = committed_git_repo.parent / "environment-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "prepare-environment", "exit 0")
    _executable(binary_dir, "materialize-environment", "exit 0")
    plan = {
        "environment": {
            "prepare_commands": [{"argv": ["prepare-environment"]}],
            "materialize_commands": [{"argv": ["materialize-environment"]}],
        }
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan)

    assert isinstance(result, HostPreflightPlan)
    assert result.commands == ()
    assert [command.argv for command in result.prepare_commands] == [
        ("prepare-environment",)
    ]
    assert [command.argv for command in result.materialize_commands] == [
        ("materialize-environment",)
    ]


def test_aggregates_missing_files_cwd_runtime_and_secret_with_evidence(
    committed_git_repo: Path,
) -> None:
    plan = _base_plan()
    plan["command_catalog"]["commands"][0]["cwd"] = "../outside"
    plan["command_catalog"]["commands"][0]["uses_services"] = []

    result = _preflight(committed_git_repo).validate(plan)

    assert isinstance(result, HostPreflightBlock)
    assert len(result.failures) == 4
    assert "repo-relative directory" in result.reason
    assert "runtime.version" in result.reason
    assert "host executable is required: example-runtime" in result.reason
    assert "required secret is not configured: PACKAGE_TOKEN" in result.reason
    assert "Betterborg will not install runtimes" in result.reason
    assert "pyproject.toml" in result.reason


def test_toolchain_version_must_match_repository_evidence_and_host(
    committed_git_repo: Path,
) -> None:
    binary_dir = committed_git_repo.parent / "version-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-runtime", "echo 'example 3.12.1'")
    (committed_git_repo / "runtime.version").write_text("3.11.9\n", encoding="utf-8")
    plan = _base_plan()
    plan["command_catalog"]["commands"][0]["cwd"] = "."
    plan["command_catalog"]["commands"][0]["uses_services"] = []
    plan["required_secrets"] = []
    plan["command_catalog"]["commands"][0]["required_secrets"] = []

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan)

    assert isinstance(result, HostPreflightBlock)
    assert "must satisfy analyzer version '3.11.9'" in result.reason
    assert "observed: example 3.12.1" in result.reason


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
        "environment": {
            "files": [".python-version"],
            "toolchains": [
                {
                    "name": "python3",
                    "version": "3.13.7",
                    "source": ".python-version",
                }
            ],
        }
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan)

    assert isinstance(result, HostPreflightPlan)
    assert result.executables[0].path == shim
    assert result.executables[0].path.is_symlink()


def test_go_version_probe_uses_supported_command_and_output(
    committed_git_repo: Path,
) -> None:
    binary_dir = committed_git_repo.parent / "go-bin"
    binary_dir.mkdir()
    _executable(
        binary_dir,
        "go",
        "test \"$1\" = version && echo 'go version go1.24.2 linux/amd64'",
    )
    (committed_git_repo / "go.mod").write_text(
        "module example.test/project\n\ngo 1.24.2\n", encoding="utf-8"
    )
    plan = {
        "environment": {
            "files": ["go.mod"],
            "toolchains": [
                {"name": "go", "version": "1.24.2", "source": "go.mod"}
            ],
        }
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan)

    assert isinstance(result, HostPreflightPlan)
    assert result.executables[0].version == "1.24.2"


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
        "environment": {
            "toolchains": [{"name": "python", **optional_version}],
        }
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
        "environment": {
            "toolchains": [
                {
                    "name": "rust",
                    "version": "1.88.0",
                    "source": "rust-toolchain.toml",
                }
            ],
        }
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan)

    assert isinstance(result, HostPreflightPlan)
    assert result.executables[0].name == "rustc"
    assert result.executables[0].path == rustc
    assert result.executables[0].version == "1.88.0"


def test_missing_cited_toolchain_file_is_not_masked_by_environment_files(
    committed_git_repo: Path,
) -> None:
    binary_dir = committed_git_repo.parent / "cited-version-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-runtime", "echo 'example 3.11.9'")
    (committed_git_repo / "runtime.version").write_text("3.11.9\n", encoding="utf-8")
    plan = {
        "environment": {
            "files": ["runtime.version"],
            "toolchains": [
                {
                    "name": "example-runtime",
                    "version": "3.11.9",
                    "source": "missing.version",
                }
            ],
        }
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan)

    assert isinstance(result, HostPreflightBlock)
    assert len(result.failures) == 1
    assert "version evidence file must exist" in result.reason
    assert "missing.version" in result.reason


def test_command_derived_failures_retain_exact_source_evidence(
    committed_git_repo: Path,
) -> None:
    source = "pyproject.toml#tool.pytest.ini_options"
    plan = {
        "command_catalog": {
            "commands": [
                {
                    "stage": "test",
                    "argv": ["missing-runtime", "-m", "pytest"],
                    "source": source,
                    "required_secrets": ["PACKAGE_TOKEN"],
                    "uses_services": ["database"],
                }
            ]
        }
    }

    result = _preflight(committed_git_repo).validate(plan)

    assert isinstance(result, HostPreflightBlock)
    assert len(result.failures) == 3
    assert all(failure.evidence == source for failure in result.failures)
    assert "host executable is required: missing-runtime" in result.reason
    assert "undeclared required secret: PACKAGE_TOKEN" in result.reason
    assert "exactly one analyzer dependency: database" in result.reason


@pytest.mark.parametrize(
    ("service", "external_urls", "expected"),
    [
        (
            {"name": "dependency", "source": "app.toml#dependency"},
            {},
            "ambiguous or inferred",
        ),
        (
            {
                "name": "dependency",
                "url_env": "DEPENDENCY_URL",
                "source": "app.toml#dependency",
            },
            {},
            "requires an absolute service URL in DEPENDENCY_URL",
        ),
        (
            {
                "name": "dependency",
                "url_env": "DEPENDENCY_URL",
                "source": "app.toml#dependency",
            },
            {"DEPENDENCY_URL": "https://exa mple.test/api"},
            "requires an absolute service URL in DEPENDENCY_URL",
        ),
        (
            {
                "name": "dependency",
                "url_env": "DEPENDENCY_URL",
                "source": "app.toml#dependency",
            },
            {"DEPENDENCY_URL": "https://example.test:not-a-port/api"},
            "requires an absolute service URL in DEPENDENCY_URL",
        ),
        (
            {
                "name": "dependency",
                "url_env": "DEPENDENCY_URL",
                "source": "app.toml#dependency",
            },
            {"DEPENDENCY_URL": "localhost:9000"},
            "requires an absolute service URL in DEPENDENCY_URL",
        ),
        (
            {
                "name": "dependency",
                "url_env": "DEPENDENCY_URL",
                "source": "app.toml#dependency",
            },
            {"DEPENDENCY_URL": "https://["},
            "requires an absolute service URL in DEPENDENCY_URL",
        ),
    ],
)
def test_selected_service_must_be_explicit_and_external_url_supplied(
    committed_git_repo: Path,
    service: dict[str, object],
    external_urls: dict[str, str],
    expected: str,
) -> None:
    binary_dir = committed_git_repo.parent / f"{committed_git_repo.name}-service-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "available-command", "exit 0")
    plan = {
        "command_catalog": {
            "commands": [
                {
                    "stage": "test",
                    "argv": ["available-command"],
                    "uses_services": ["dependency"],
                }
            ]
        },
        "service_dependencies": [service],
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan, external_urls=external_urls)

    assert isinstance(result, HostPreflightBlock)
    assert len(result.failures) == 1
    assert expected in result.reason
    assert "app.toml#dependency" in result.reason


def test_compose_url_environment_requires_an_exact_service_port(
    committed_git_repo: Path,
) -> None:
    binary_dir = committed_git_repo.parent / "compose-url-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-runtime", "echo '3.11.9'")
    _compose_executable(binary_dir)
    (committed_git_repo / "package").mkdir()
    (committed_git_repo / "runtime.version").write_text(
        "3.11.9\n", encoding="utf-8"
    )
    (committed_git_repo / "compose.yml").write_text(
        "services:\n  postgres:\n    image: postgres:16\n",
        encoding="utf-8",
    )
    plan = _base_plan()
    del plan["service_dependencies"][0]["port"]

    result = _preflight(
        committed_git_repo,
        environment={"PATH": str(binary_dir)},
    ).validate(
        plan,
        available_secret_names={"PACKAGE_TOKEN"},
        external_urls={"SEARCH_URL": "https://search.example.test/api"},
    )

    assert isinstance(result, HostPreflightBlock)
    assert "DATABASE_URL requires an exact port" in result.reason

def test_compose_url_environment_does_not_infer_a_multi_port_target(
    committed_git_repo: Path,
) -> None:
    binary_dir = committed_git_repo.parent / "compose-port-fallback-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-runtime", "echo '3.11.9'")
    _compose_executable(binary_dir)
    (committed_git_repo / "package").mkdir()
    (committed_git_repo / "runtime.version").write_text(
        "3.11.9\n", encoding="utf-8"
    )
    (committed_git_repo / "compose.yml").write_text(
        "services:\n  postgres:\n    image: postgres:16\n",
        encoding="utf-8",
    )
    plan = _base_plan()
    dependencies = plan["service_dependencies"]
    assert isinstance(dependencies, list)
    database = dependencies[0]
    assert isinstance(database, dict)
    del database["port"]
    database["ports"] = [
        {"port": 5432, "protocol": "udp", "env": "POSTGRES_PORT"},
        {"port": 9187, "protocol": "tcp", "env": "METRICS_URL"},
    ]

    result = _preflight(
        committed_git_repo,
        environment={"PATH": str(binary_dir)},
    ).validate(
        plan,
        available_secret_names={"PACKAGE_TOKEN"},
        external_urls={"SEARCH_URL": "https://search.example.test/api"},
    )

    assert isinstance(result, HostPreflightBlock)
    assert "DATABASE_URL requires an exact port" in result.reason

    database["port"] = 5432
    result = _preflight(
        committed_git_repo,
        environment={"PATH": str(binary_dir)},
    ).validate(
        plan,
        available_secret_names={"PACKAGE_TOKEN"},
        external_urls={"SEARCH_URL": "https://search.example.test/api"},
    )

    assert isinstance(result, HostPreflightPlan)
    assert result.services[0].url_targets == (("DATABASE_URL", 5432, "udp"),)
    assert result.services[0].port_targets == (
        ("POSTGRES_PORT", 5432, "udp"),
        ("METRICS_URL", 9187, "tcp"),
    )
    assert service_url_environment(
        result.services,
        published_ports={
            ("postgres", 5432, "udp"): 49152,
            ("postgres", 9187, "tcp"): 49153,
        },
    )["POSTGRES_PORT"] == "49152"


def test_missing_compose_metadata_and_plugin_block_before_claim(
    committed_git_repo: Path,
) -> None:
    binary_dir = committed_git_repo.parent / "compose-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "available-command", "exit 0")
    _executable(binary_dir, "docker", "exit 1")
    plan = {
        "command_catalog": {
            "commands": [
                {
                    "stage": "test",
                    "argv": ["available-command"],
                    "uses_services": ["database"],
                }
            ]
        },
        "service_dependencies": [
            {
                "name": "database",
                "compose_service": "postgres",
                "source": "compose.yml#postgres",
            }
        ],
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan)

    assert isinstance(result, HostPreflightBlock)
    assert "Compose metadata is required" in result.reason
    assert "Docker Compose plugin must be available" in result.reason
    assert "compose.yml#postgres" in result.reason


@pytest.mark.parametrize(
    ("compose_version", "supported"),
    [("2.24.3", False), ("2.24.4", True)],
)
def test_compose_plugin_must_support_runtime_overrides(
    committed_git_repo: Path,
    compose_version: str,
    supported: bool,
) -> None:
    binary_dir = (
        committed_git_repo.parent / f"{committed_git_repo.name}-compose-version-bin"
    )
    binary_dir.mkdir()
    _executable(binary_dir, "available-command", "exit 0")
    _compose_executable(binary_dir, version=compose_version)
    compose_file = committed_git_repo / "compose.yml"
    compose_file.write_text(
        "services:\n  postgres:\n    image: postgres:16\n", encoding="utf-8"
    )
    plan = {
        "command_catalog": {
            "commands": [
                {
                    "stage": "test",
                    "argv": ["available-command"],
                    "uses_services": ["database"],
                }
            ]
        },
        "compose": {"file": "compose.yml", "source": "compose.yml"},
        "service_dependencies": [
            {
                "name": "database",
                "compose_service": "postgres",
                "source": "compose.yml#services.postgres",
            }
        ],
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan)

    if supported:
        assert isinstance(result, HostPreflightPlan)
        return
    assert isinstance(result, HostPreflightBlock)
    assert len(result.failures) == 1
    assert "Docker Compose 2.24.4 or newer is required" in result.reason
    assert f"Docker Compose v{compose_version}" in result.reason
    assert "Upgrade the Docker Compose plugin" in result.reason


def test_compose_topology_is_validated_before_a_plan_is_returned(
    committed_git_repo: Path,
) -> None:
    binary_dir = committed_git_repo.parent / "compose-topology-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "available-command", "exit 0")
    _compose_executable(
        binary_dir,
        model={
            "services": {
                "postgres": {
                    "volumes": [
                        {
                            "type": "bind",
                            "source": "/var/lib/example",
                            "target": "/data",
                        }
                    ]
                }
            },
            "networks": {"backend": {}},
            "volumes": {"database": {}},
        },
    )
    (committed_git_repo / "compose.yml").write_text(
        "services:\n  postgres:\n    image: postgres:16\n", encoding="utf-8"
    )
    plan = {
        "command_catalog": {
            "commands": [
                {
                    "stage": "test",
                    "argv": ["available-command"],
                    "uses_services": ["database"],
                }
            ]
        },
        "compose": {"file": "compose.yml", "source": "compose.yml"},
        "service_dependencies": [
            {
                "name": "database",
                "compose_service": "postgres",
                "source": "compose.yml#services.postgres",
            }
        ],
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan)

    assert isinstance(result, HostPreflightBlock)
    assert "selected Compose service topology must be isolated" in result.reason
    assert "writable bind mounts cannot be isolated" in result.reason
    assert "postgres.volumes[0]" in result.reason


def test_preserves_ordered_compose_stack_and_active_profiles(
    committed_git_repo: Path,
) -> None:
    binary_dir = committed_git_repo.parent / "compose-stack-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "available-command", "exit 0")
    _compose_executable(binary_dir)
    for name in ("compose.yml", "compose.test.yml", "compose.fragment.yml"):
        (committed_git_repo / name).write_text(
            "services:\n  postgres:\n    image: postgres:16\n",
            encoding="utf-8",
        )
    analyzer_payload = {
        "summary": "A repository with a multi-file Compose service stack.",
        "primary_language": "python",
        "is_monorepo": False,
        "packages": [
            {
                "path": ".",
                "name": "root",
                "primary_language": "python",
                "rubric": {
                    dimension: {
                        "score": 3,
                        "evidence": f"README.md describes {dimension}",
                    }
                    for dimension in DIMENSIONS
                },
            }
        ],
        "recommendations": [],
        "themes": [],
        "command_catalog": {
            "commands": [
                {
                    "stage": "test",
                    "argv": ["available-command"],
                    "uses_services": ["database"],
                    "source": "README.md#test",
                }
            ],
            "source": "README.md",
        },
        "compose": {
            "file": "compose.yml",
            "files": [
                {
                    "path": "compose.yml",
                    "services": ["postgres"],
                    "source": "compose.yml",
                },
                {
                    "path": "compose.test.yml",
                    "profiles": ["test"],
                    "services": ["postgres"],
                    "source": "compose.test.yml",
                },
                {
                    "path": "compose.fragment.yml",
                    "source": "compose.fragment.yml",
                },
            ],
            "profiles": ["test", "integration"],
            "source": "compose.yml",
        },
        "service_dependencies": [
            {
                "name": "database",
                "compose_service": "postgres",
                "source": "compose.yml#services.postgres",
            }
        ],
    }

    repository = Repository(root=committed_git_repo)
    adapter = MockAdapter(name="openai").queue(
        MockResponse(payload=analyzer_payload)
    )
    with SqliteStore.open(committed_git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        analysis = run_analyzer(
            repository,
            store,
            adapter,
            artifact_dir=committed_git_repo / "artifacts",
        )

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(analysis.analysis_json)

    assert isinstance(result, HostPreflightPlan)
    assert result.compose_files == (
        committed_git_repo / "compose.yml",
        committed_git_repo / "compose.test.yml",
        committed_git_repo / "compose.fragment.yml",
    )
    assert result.compose_profiles == ("test", "integration")


def test_accepts_singular_compose_file_without_service_index(
    committed_git_repo: Path,
) -> None:
    binary_dir = committed_git_repo.parent / "single-compose-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "available-command", "exit 0")
    _compose_executable(binary_dir)
    compose_file = committed_git_repo / "compose.yml"
    compose_file.write_text(
        "services:\n  postgres:\n    image: postgres:16\n", encoding="utf-8"
    )
    plan = {
        "command_catalog": {
            "commands": [
                {
                    "stage": "test",
                    "argv": ["available-command"],
                    "uses_services": ["database"],
                }
            ]
        },
        "compose": {"file": "compose.yml", "source": "compose.yml"},
        "service_dependencies": [
            {
                "name": "database",
                "compose_service": "postgres",
                "source": "compose.yml#services.postgres",
            }
        ],
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan)

    assert isinstance(result, HostPreflightPlan)
    assert result.compose_files == (compose_file,)


def test_unused_service_does_not_require_docker_or_external_url(
    committed_git_repo: Path,
) -> None:
    binary_dir = committed_git_repo.parent / "unused-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "available-command", "exit 0")
    plan = {
        "command_catalog": {
            "commands": [{"stage": "build", "argv": ["available-command"]}]
        },
        "service_dependencies": [
            {"name": "ambiguous", "source": "example.env"},
            {
                "name": "external",
                "url_env": "UNSUPPLIED_URL",
                "source": "app.toml",
            },
        ],
    }

    result = _preflight(
        committed_git_repo, environment={"PATH": str(binary_dir)}
    ).validate(plan)

    assert isinstance(result, HostPreflightPlan)
    assert result.services == ()
    assert result.compose_files == ()
