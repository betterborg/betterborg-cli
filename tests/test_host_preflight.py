"""Analyzer-evidence contracts for trust-gated host preflight."""

from __future__ import annotations

from pathlib import Path

import pytest

from betterborg_cli.host_execution import (
    HostPreflight,
    HostPreflightBlock,
    HostPreflightPlan,
)
from betterborg_cli.repo_paths import RepoPaths
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
    assert "borg trust --yes" in result.reason


def test_validates_complete_plan_and_ignores_unselected_service(
    committed_git_repo: Path,
) -> None:
    binary_dir = committed_git_repo.parent / "host-bin"
    binary_dir.mkdir()
    _executable(binary_dir, "example-runtime", "echo 'example 3.11.9'")
    _executable(
        binary_dir,
        "docker",
        "test \"$1 $2\" = 'compose version' && echo 'Docker Compose v2.30.0'",
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
    assert result.required_secret_names == ("PACKAGE_TOKEN",)
    assert result.compose_files == (committed_git_repo / "compose.yml",)
    assert [(service.name, service.kind) for service in result.services] == [
        ("database", "compose"),
        ("search", "external"),
    ]
    assert result.services[1].url == "https://search.example.test/api"


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
    assert "BetterBorg will not install runtimes" in result.reason
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
            {"DEPENDENCY_URL": "localhost:9000"},
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
