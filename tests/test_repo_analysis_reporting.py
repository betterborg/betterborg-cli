"""Harness Performance report contract tests."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from betterborg_cli.agent_runtime.mock import MockAdapter, MockResponse
from betterborg_cli.agent_runtime.structured import (
    StructuredResultError,
    validate_structured_result,
)
from betterborg_cli.repo_analysis.analyzer import (
    ANALYZER_OUTPUT_SCHEMA,
    AnalyzerError,
    run_analyzer,
)
from betterborg_cli.repo_analysis.discovery import ANALYSIS_INPUT_FILENAME
from betterborg_cli.repo_analysis.reporting import (
    build_machine_report,
    render_json_report,
    render_markdown_report,
    render_terminal_report,
)
from betterborg_cli.repo_analysis.scoring import DIMENSIONS
from betterborg_cli.store import (
    Repository,
    RepositoryAnalysis,
    RepositoryPackage,
    SqliteStore,
)

_REPOSITORY_ID = UUID("10000000-0000-0000-0000-000000000000")
_ANALYSIS_ID = UUID("20000000-0000-0000-0000-000000000000")
_PRIOR_ID = UUID("30000000-0000-0000-0000-000000000000")


def _rubric(score: float) -> dict[str, dict[str, object]]:
    return {
        dimension: {"score": score, "evidence": f"evidence for {dimension}"}
        for dimension in DIMENSIONS
    }


def _commit_cited_evidence(git_repo: Path) -> None:
    """Commit the small evidence set the citation-shape tests analyze."""
    (git_repo / "README.md").write_text("# Example\n", encoding="utf-8")
    (git_repo / "Makefile").write_text(
        "test:\n\tpython -m pytest\n", encoding="utf-8"
    )
    (git_repo / "package.json").write_text(
        '{"scripts":{"test":"python -m pytest"}}\n', encoding="utf-8"
    )
    subprocess.run(["git", "-C", str(git_repo), "add", "--all"], check=True)
    subprocess.run(
        ["git", "-C", str(git_repo), "commit", "--quiet", "-m", "initial"],
        check=True,
    )


def _catalog_payload(catalog: dict[str, object]) -> dict[str, object]:
    """Wrap one command catalog in the smallest valid analyzer payload."""
    return {
        "summary": "A small Python command-line application.",
        "primary_language": "python",
        "is_monorepo": False,
        "packages": [
            {
                "path": ".",
                "name": "root",
                "primary_language": "python",
                "rubric": _rubric(3),
            }
        ],
        "recommendations": [],
        "themes": [],
        "command_catalog": catalog,
    }


@pytest.fixture
def analysis() -> RepositoryAnalysis:
    return RepositoryAnalysis(
        id=_ANALYSIS_ID,
        repository_id=_REPOSITORY_ID,
        head_sha="abc123",
        summary="A Python monorepo with two independently scored packages.",
        primary_language="python",
        is_monorepo=True,
        overall_score=3.0,
        prior_analysis_id=_PRIOR_ID,
        score_delta=0.5,
        created_at=datetime(2026, 8, 24, tzinfo=UTC),
        analysis_json={
            "packages": [
                {"path": "packages/api"},
                {"path": "packages/worker"},
            ],
            "command_catalog": {
                "source": "Makefile",
                "commands": [
                    {"stage": "test", "argv": ["make", "test"], "verifies": True}
                ],
            },
            "required_secrets": [
                {
                    "name": "PACKAGE_TOKEN",
                    "used_by": ["test"],
                    "scope": "build",
                    "source": ".env.example",
                }
            ],
            "service_dependencies": [],
            "themes": [
                {
                    "id": "theme-ci",
                    "title": "Strengthen CI feedback",
                    "recommendation_ids": ["rec-ci"],
                    "effort": "S",
                    "effort_rationale": "One workflow edit.",
                    "normalized_impact": 0.125,
                    "ranking_score": 0.125,
                    "recommendations": [
                        {
                            "id": "rec-ci",
                            "effective_delta": 1.0,
                            "delta_clamped": False,
                        }
                    ],
                }
            ],
        },
    )


@pytest.fixture
def packages() -> list[RepositoryPackage]:
    first_rubric = _rubric(4)
    second_rubric = _rubric(2)
    first_rubric["ci"]["score"] = 5
    second_rubric["ci"]["score"] = 1
    return [
        RepositoryPackage(
            repository_id=_REPOSITORY_ID,
            analysis_id=_ANALYSIS_ID,
            package_path="packages/worker",
            package_name="worker",
            primary_language="python",
            rubric=second_rubric,
            overall_score=2.0,
        ),
        RepositoryPackage(
            repository_id=_REPOSITORY_ID,
            analysis_id=_ANALYSIS_ID,
            package_path="packages/api",
            package_name="api",
            primary_language="python",
            rubric=first_rubric,
            overall_score=4.0,
        ),
    ]


def test_machine_report_preserves_arithmetic_history_and_ranked_theme_contract(
    analysis: RepositoryAnalysis, packages: list[RepositoryPackage]
) -> None:
    report = build_machine_report(analysis, packages)

    assert report["score"] == 3.0
    assert report["previous_score"] == 2.5
    assert report["delta"] == 0.5
    assert [package["path"] for package in report["packages"]] == [
        "packages/api",
        "packages/worker",
    ]
    dimension_scores = {
        dimension["id"]: dimension["score"] for dimension in report["dimensions"]
    }
    assert dimension_scores == {
        dimension: 3.0 for dimension in DIMENSIONS
    }
    assert report["themes"][0] == {
        "rank": 1,
        "id": "theme-ci",
        "title": "Strengthen CI feedback",
        "effort": "S",
        "effort_label": "S (estimated)",
        "effort_rationale": "One workflow edit.",
        "estimated_impact": 0.125,
        "ranking_score": 0.125,
        "recommendations": [
            {"id": "rec-ci", "effective_delta": 1.0, "delta_clamped": False}
        ],
    }
    assert report["estimated"] is True
    assert report["non_deterministic"] is True


def test_harness_impact_distinguishes_unknown_from_detected_and_not_detected(
    analysis: RepositoryAnalysis, packages: list[RepositoryPackage]
) -> None:
    impact = build_machine_report(analysis, packages)["harness_impact"]

    assert impact["commands"]["status"] == "detected"
    assert impact["environment"] == {
        "status": "unknown",
        "label": "Unknown",
        "summary": "No reliable environment inputs were persisted.",
        "files": [],
        "toolchains": [],
        "package_managers": [],
        "prepare_commands": [],
        "materialize_commands": [],
        "source": None,
    }
    assert impact["secrets"]["status"] == "detected"
    assert impact["services"]["status"] == "not_detected"


def test_analyzer_persists_harness_inputs_consumed_by_report(
    git_repo: Path,
) -> None:
    evidence = {
        "README.md": "# Example\n",
        "Makefile": "test:\n\tpython -m pytest\n",
        "package.json": '{"scripts":{"test":"python -m pytest"}}\n',
        "pyproject.toml": "[project]\nname = 'example'\nversion = '1.0.0'\n",
        ".python-version": "3.11.9\n",
        ".env.example": "PACKAGE_TOKEN=\n",
        "docker-compose.yml": "services:\n  postgres:\n    image: postgres:16\n",
    }
    for path, body in evidence.items():
        (git_repo / path).write_text(body, encoding="utf-8")
    subprocess.run(["git", "-C", str(git_repo), "add", "--all"], check=True)
    subprocess.run(
        ["git", "-C", str(git_repo), "commit", "--quiet", "-m", "initial"],
        check=True,
    )
    payload = {
        "summary": "A small Python command-line application.",
        "primary_language": "python",
        "is_monorepo": False,
        "packages": [
            {
                "path": ".",
                "name": "root",
                "primary_language": "python",
                "rubric": _rubric(3),
            }
        ],
        "recommendations": [],
        "themes": [],
        "command_catalog": {
            "source": "Makefile",
            "commands": [
                {
                    "stage": "test",
                    "verifies": True,
                    "argv": ["make", "test"],
                    "source": "package.json#scripts",
                    "uses_services": ["postgres"],
                    "required_secrets": ["PACKAGE_TOKEN"],
                }
            ]
        },
        "environment": {
            "version": 1,
            "files": ["pyproject.toml", ".python-version"],
            "toolchains": [
                {
                    "name": "python",
                    "version": "3.11",
                    "source": ".python-version",
                },
                {"name": "java", "version": "21", "source": "README.md"},
            ],
            "package_managers": ["pip"],
            "prepare_commands": [
                {
                    "argv": ["python", "-m", "pip", "install", "-e", "."],
                    "source": "pyproject.toml/project",
                }
            ],
            "materialize_commands": [
                {
                    "argv": ["git", "submodule", "update", "--init"],
                    "cwd": ".",
                    "source": "README.md",
                }
            ],
        },
        "required_secrets": [
            {
                "name": "PACKAGE_TOKEN",
                "used_by": ["test"],
                "scope": "build",
                "source": ".env.example",
            }
        ],
        "service_dependencies": [
            {
                "name": "postgres",
                "image": "postgres:16",
                "port": 5432,
                "source": "docker-compose.yml",
                "env": ["POSTGRES_PASSWORD"],
            }
        ],
    }
    repository = Repository(root=git_repo)
    adapter = MockAdapter(name="openai").queue(MockResponse(payload=payload))

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        persisted = run_analyzer(
            repository,
            store,
            adapter,
            artifact_dir=git_repo / "artifacts",
        )
        report = build_machine_report(
            persisted,
            store.list_packages(persisted.id),
        )

    assert persisted.analysis_json["command_catalog"] == payload["command_catalog"]
    assert persisted.analysis_json["environment"] == payload["environment"]
    assert persisted.analysis_json["required_secrets"] == payload["required_secrets"]
    assert persisted.analysis_json["service_dependencies"] == payload[
        "service_dependencies"
    ]
    assert persisted.analysis_json["command_catalog"]["commands"][0][
        "source"
    ] == "package.json#scripts"
    assert {
        key: impact["status"] for key, impact in report["harness_impact"].items()
    } == {
        "commands": "detected",
        "environment": "detected",
        "secrets": "detected",
        "services": "detected",
    }
    impact = report["harness_impact"]
    assert impact["commands"]["commands"] == payload["command_catalog"]["commands"]
    assert impact["environment"]["files"] == payload["environment"]["files"]
    assert impact["environment"]["toolchains"] == payload["environment"][
        "toolchains"
    ]
    assert impact["environment"]["prepare_commands"] == payload["environment"][
        "prepare_commands"
    ]
    assert impact["environment"]["materialize_commands"] == payload[
        "environment"
    ]["materialize_commands"]
    assert impact["environment"]["summary"] == (
        "7 environment inputs persisted for harness use."
    )
    assert [secret["name"] for secret in impact["secrets"]["secrets"]] == [
        "PACKAGE_TOKEN"
    ]
    assert [service["name"] for service in impact["services"]["services"]] == [
        "postgres"
    ]
    assert impact["services"]["services"][0]["env"] == ["POSTGRES_PASSWORD"]

    terminal = render_terminal_report(report)
    markdown = render_markdown_report(report)
    assert '["make", "test"]' in terminal
    assert '["python", "-m", "pip", "install", "-e", "."]' in terminal
    assert '["git", "submodule", "update", "--init"]' in terminal
    assert r'\["make", "test"\]' in markdown
    assert (
        r'\["python", "-m", "pip", "install", "-e", "."\]' in markdown
    )
    assert r'\["git", "submodule", "update", "--init"\]' in markdown
    assert "PACKAGE_TOKEN" in terminal
    assert r"PACKAGE\_TOKEN" in markdown
    for rendered in (terminal, markdown):
        assert "Environment file: pyproject.toml" in rendered
        assert "Environment file: .python-version" in rendered
        assert "python 3.11" in rendered
        assert "java 21" in rendered
        assert "postgres" in rendered


def test_analyzer_rejects_harness_evidence_outside_discovery_manifest(
    git_repo: Path,
) -> None:
    (git_repo / "README.md").write_text("# Example\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(git_repo), "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(git_repo), "commit", "--quiet", "-m", "initial"],
        check=True,
    )
    payload = {
        "summary": "A small Python command-line application.",
        "primary_language": "python",
        "is_monorepo": False,
        "packages": [
            {
                "path": ".",
                "name": "root",
                "primary_language": "python",
                "rubric": _rubric(3),
            }
        ],
        "recommendations": [],
        "themes": [],
        "command_catalog": {
            "source": "not-discovered.yml",
            "commands": [
                {
                    "stage": "test",
                    "verifies": True,
                    "argv": ["make", "test"],
                    "source": "command.missing.yml#jobs",
                }
            ],
        },
        "environment": {
            "files": ["missing.lock"],
            "toolchains": [
                {"name": "java", "version": "21", "source": "jdk.missing"}
            ],
        },
        "required_secrets": [
            {
                "name": "TOKEN",
                "used_by": ["test"],
                "scope": "build",
                "source": "secrets.example",
            }
        ],
        "service_dependencies": [
            {
                "name": "postgres",
                "source": "compose.missing.yml",
                "ports": [{"port": 5432, "source": "port.missing.yml"}],
            }
        ],
    }
    repository = Repository(root=git_repo)
    adapter = MockAdapter(name="openai").queue(MockResponse(payload=payload))

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)

        with pytest.raises(AnalyzerError, match="absent from manifest") as error:
            run_analyzer(
                repository,
                store,
                adapter,
                artifact_dir=git_repo / "artifacts",
            )

        assert store.list_analyses(repository.id) == []

    assert all(
        source in str(error.value)
        for source in (
            "not-discovered.yml",
            "command.missing.yml#jobs",
            "missing.lock",
            "jdk.missing",
            "secrets.example",
            "compose.missing.yml",
            "port.missing.yml",
        )
    )


def test_single_anchored_and_directory_relative_citations_stay_accepted(
    git_repo: Path,
) -> None:
    _commit_cited_evidence(git_repo)
    payload = _catalog_payload(
        {
            "source": "Makefile",
            "commands": [
                {
                    "stage": "test",
                    "verifies": True,
                    "argv": ["make", "test"],
                    "source": "package.json#scripts",
                },
                {
                    "stage": "lint",
                    "verifies": True,
                    "argv": ["make", "lint"],
                    "source": "package.json/scripts",
                },
            ],
        }
    )
    repository = Repository(root=git_repo)
    adapter = MockAdapter(name="openai").queue(MockResponse(payload=payload))

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        run_analyzer(
            repository,
            store,
            adapter,
            artifact_dir=git_repo / "artifacts",
        )

        assert len(store.list_analyses(repository.id)) == 1


def test_a_citation_naming_several_manifest_files_is_accepted(
    git_repo: Path,
) -> None:
    _commit_cited_evidence(git_repo)
    payload = _catalog_payload(
        {
            "source": "Makefile; package.json",
            "commands": [
                {
                    "stage": "test",
                    "verifies": True,
                    "argv": ["make", "test"],
                    "source": "package.json#scripts; Makefile",
                }
            ],
        }
    )
    repository = Repository(root=git_repo)
    adapter = MockAdapter(name="openai").queue(MockResponse(payload=payload))

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        run_analyzer(
            repository,
            store,
            adapter,
            artifact_dir=git_repo / "artifacts",
        )

        analyses = store.list_analyses(repository.id)
        assert len(analyses) == 1
        catalog = analyses[0].analysis_json["command_catalog"]
        assert catalog["source"] == "Makefile; package.json"


def test_a_citation_naming_several_files_reports_only_the_absent_one(
    git_repo: Path,
) -> None:
    _commit_cited_evidence(git_repo)
    payload = _catalog_payload(
        {
            "source": "Makefile; not-discovered.yml",
            "commands": [{"stage": "test", "argv": ["make", "test"], "verifies": True}],
        }
    )
    repository = Repository(root=git_repo)
    adapter = MockAdapter(name="openai").queue(MockResponse(payload=payload))

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)

        with pytest.raises(AnalyzerError, match="absent from manifest") as error:
            run_analyzer(
                repository,
                store,
                adapter,
                artifact_dir=git_repo / "artifacts",
            )

        assert store.list_analyses(repository.id) == []

    assert "not-discovered.yml" in str(error.value)
    assert "Makefile" not in str(error.value)


def test_materialize_command_alone_is_a_valid_environment_input(
    git_repo: Path,
) -> None:
    (git_repo / "README.md").write_text(
        "# Example\n\nInitialize submodules before building.\n", encoding="utf-8"
    )
    subprocess.run(["git", "-C", str(git_repo), "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(git_repo), "commit", "--quiet", "-m", "initial"],
        check=True,
    )
    materialize = {
        "argv": ["git", "submodule", "update", "--init"],
        "source": "README.md",
    }
    payload = {
        "summary": "A small repository with an offline materialization step.",
        "primary_language": "python",
        "is_monorepo": False,
        "packages": [
            {
                "path": ".",
                "name": "root",
                "primary_language": "python",
                "rubric": _rubric(3),
            }
        ],
        "recommendations": [],
        "themes": [],
        "environment": {"materialize_commands": [materialize]},
    }
    repository = Repository(root=git_repo)
    adapter = MockAdapter(name="openai").queue(MockResponse(payload=payload))

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        persisted = run_analyzer(
            repository,
            store,
            adapter,
            artifact_dir=git_repo / "artifacts",
        )
        report = build_machine_report(persisted, store.list_packages(persisted.id))

    assert report["harness_impact"]["environment"] == {
        "status": "detected",
        "label": "Detected",
        "summary": "1 environment input persisted for harness use.",
        "files": [],
        "toolchains": [],
        "package_managers": [],
        "prepare_commands": [],
        "materialize_commands": [materialize],
        "source": None,
    }


def test_a_package_manager_carries_the_command_that_installs_it(
    git_repo: Path,
) -> None:
    (git_repo / "pyproject.toml").write_text(
        "[project]\nname = 'example'\nversion = '1.0.0'\n", encoding="utf-8"
    )
    subprocess.run(["git", "-C", str(git_repo), "add", "pyproject.toml"], check=True)
    subprocess.run(
        ["git", "-C", str(git_repo), "commit", "--quiet", "-m", "initial"],
        check=True,
    )
    payload = {
        "summary": "A small Python application using the pip package manager.",
        "primary_language": "python",
        "is_monorepo": False,
        "packages": [
            {
                "path": ".",
                "name": "root",
                "primary_language": "python",
                "rubric": _rubric(3),
            }
        ],
        "recommendations": [],
        "themes": [],
        "environment": {
            "source": "pyproject.toml",
            "package_managers": ["pip"],
            "materialize_commands": [{"argv": ["pip", "install", "-e", "."]}],
        },
    }
    repository = Repository(root=git_repo)
    adapter = MockAdapter(name="openai").queue(MockResponse(payload=payload))

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        persisted = run_analyzer(
            repository,
            store,
            adapter,
            artifact_dir=git_repo / "artifacts",
        )
        environment = build_machine_report(
            persisted, store.list_packages(persisted.id)
        )["harness_impact"]["environment"]

    assert environment == {
        "status": "detected",
        "label": "Detected",
        "summary": "2 environment inputs persisted for harness use.",
        "files": [],
        "toolchains": [],
        "package_managers": ["pip"],
        "prepare_commands": [],
        "materialize_commands": [{"argv": ["pip", "install", "-e", "."]}],
        "source": "pyproject.toml",
    }


def test_a_named_package_manager_without_an_install_command_is_rejected() -> None:
    """Silence about how a repository installs is read as needing no install.

    A worktree holds only what its commit tracks, so an environment that names
    a package manager and no command hands every phase a checkout with nothing
    fetched and no record that anything is missing.
    """
    payload = {
        "summary": "A small Python application using the pip package manager.",
        "primary_language": "python",
        "is_monorepo": False,
        "packages": [
            {
                "path": ".",
                "name": "root",
                "primary_language": "python",
                "rubric": _rubric(3),
            }
        ],
        "recommendations": [],
        "themes": [],
        "environment": {
            "source": "pyproject.toml",
            "package_managers": ["pip"],
        },
    }

    with pytest.raises(StructuredResultError, match="package_managers"):
        validate_structured_result(payload, ANALYZER_OUTPUT_SCHEMA)

    payload["environment"]["materialize_commands"] = [
        {"argv": ["pip", "install", "-e", "."]}
    ]
    validate_structured_result(payload, ANALYZER_OUTPUT_SCHEMA)


def test_analyzer_rejects_uncited_harness_detections(git_repo: Path) -> None:
    (git_repo / "README.md").write_text("# Example\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(git_repo), "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(git_repo), "commit", "--quiet", "-m", "initial"],
        check=True,
    )
    payload = {
        "summary": "A small application with uncited Harness detections.",
        "primary_language": "python",
        "is_monorepo": False,
        "packages": [
            {
                "path": ".",
                "name": "root",
                "primary_language": "python",
                "rubric": _rubric(3),
            }
        ],
        "recommendations": [],
        "themes": [],
        "command_catalog": {
            "commands": [{"stage": "test", "argv": ["make", "test"], "verifies": True}]
        },
        "environment": {
            "toolchains": [{"name": "python"}],
            "package_managers": ["pip"],
            "prepare_commands": [{"argv": ["python", "-m", "pip", "install"]}],
        },
        "required_secrets": [
            {"name": "TOKEN", "used_by": ["test"], "scope": "build"}
        ],
        "service_dependencies": [
            {"name": "postgres", "ports": [{"port": 5432}]}
        ],
    }
    repository = Repository(root=git_repo)
    adapter = MockAdapter(name="openai").queue(MockResponse(payload=payload))

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        with pytest.raises(AnalyzerError, match="lacks bounded evidence") as error:
            run_analyzer(
                repository,
                store,
                adapter,
                artifact_dir=git_repo / "artifacts",
            )
        assert store.list_analyses(repository.id) == []

    assert all(
        claim in str(error.value)
        for claim in (
            "command",
            "environment package manager",
            "environment prepare_command",
            "environment toolchain",
            "required secret",
            "service",
            "service port",
        )
    )


def test_machine_report_never_exposes_service_environment_values(
    analysis: RepositoryAnalysis, packages: list[RepositoryPackage]
) -> None:
    payload = dict(analysis.analysis_json)
    payload["service_dependencies"] = [
        {
            "name": "postgres",
            "source": "docker-compose.yml",
            "url_env": "literal-url-value",
            "ports": [{"port": 5432, "env": "literal-port-value"}],
            "env": {
                "POSTGRES_PASSWORD": "actual-value",
                "POSTGRES_USER": "app-user",
            },
        }
    ]
    legacy_analysis = replace(analysis, analysis_json=payload)

    report = build_machine_report(legacy_analysis, packages)
    rendered = render_json_report(report)

    assert report["harness_impact"]["services"]["services"][0]["env"] == [
        "POSTGRES_PASSWORD",
        "POSTGRES_USER",
    ]
    assert "actual-value" not in rendered
    assert "app-user" not in rendered
    assert "literal-url-value" not in rendered
    assert "literal-port-value" not in rendered

    unsafe_analyzer_payload = {
        "summary": "A small application with an unsafe service environment.",
        "primary_language": "python",
        "is_monorepo": False,
        "packages": [
            {
                "path": ".",
                "name": "root",
                "primary_language": "python",
                "rubric": _rubric(3),
            }
        ],
        "recommendations": [],
        "themes": [],
        "service_dependencies": [
            {
                "name": "postgres",
                "source": "docker-compose.yml",
                "env": {"POSTGRES_PASSWORD": "actual-value"},
            }
        ],
    }
    with pytest.raises(
        StructuredResultError, match=r"service_dependencies\[0\]\.env"
    ):
        validate_structured_result(unsafe_analyzer_payload, ANALYZER_OUTPUT_SCHEMA)


def test_terminal_markdown_and_json_render_the_same_labeled_report(
    analysis: RepositoryAnalysis, packages: list[RepositoryPackage]
) -> None:
    report = build_machine_report(analysis, packages)

    terminal = render_terminal_report(report)
    markdown = render_markdown_report(report)
    machine = json.loads(render_json_report(report))

    for rendered in (terminal, markdown):
        assert "Harness Performance" in rendered
        assert "3.00/5" in rendered
        assert "Previous: 2.50" in rendered
        assert "Delta: +0.50" in rendered
        assert "██████░░░░" in rendered
        assert "packages/api" in rendered
        assert "Strengthen CI feedback" in rendered
        assert "S (estimated)" in rendered
        assert "Harness Impact" in rendered
        assert "Environment" in rendered
        assert "Unknown" in rendered
        assert "non-deterministic" in rendered
        lowered = rendered.lower()
        assert "readiness" not in lowered
        assert "reproducib" not in lowered

    assert machine == report


def test_machine_report_omits_analyzer_summary_claims(
    analysis: RepositoryAnalysis, packages: list[RepositoryPackage]
) -> None:
    claimed_analysis = replace(
        analysis,
        summary="This repository is fully reproducible and AI-ready.",
    )

    report = build_machine_report(claimed_analysis, packages)
    rendered = render_json_report(report).lower()

    assert "summary" not in report
    assert "readiness" not in rendered
    assert "reproducib" not in rendered


def test_human_reports_sanitize_control_characters_and_escape_markdown(
    analysis: RepositoryAnalysis, packages: list[RepositoryPackage]
) -> None:
    malicious_payload = dict(analysis.analysis_json)
    [theme] = malicious_payload["themes"]
    malicious_payload["themes"] = [
        {
            **theme,
            "title": "Break **bold**\n## Forged\x1b[31m",
            "effort_rationale": "Use | pipes\r\n- forged \u202e text.",
        }
    ]
    malicious_payload["command_catalog"] = {
        "commands": [
            {
                "stage": "test\n## Command\x1b[31m",
                "verifies": True,
                "argv": ["make", "bad\n## Arg"],
            }
        ]
    }
    malicious_payload["required_secrets"] = [
        {
            "name": "FORGED|TOKEN\n## Secret",
            "used_by": ["test"],
            "scope": "build",
        }
    ]
    malicious_payload["service_dependencies"] = [
        {"name": "postgres\n## Service", "image": "bad|image"}
    ]
    malicious_payload["packages"] = [
        {"path": "packages/evil|row\n## Forged"},
        {"path": "packages/api"},
    ]
    malicious_analysis = replace(analysis, analysis_json=malicious_payload)
    malicious_packages = [
        replace(
            packages[0],
            package_path="packages/evil|row\n## Forged",
            package_name="[link](javascript:alert(1))",
            primary_language="py\x00thon",
        ),
        packages[1],
    ]
    report = build_machine_report(malicious_analysis, malicious_packages)

    terminal = render_terminal_report(report)
    markdown = render_markdown_report(report)

    assert "\x1b" not in terminal
    assert "\x00" not in terminal
    assert "\r" not in terminal
    assert "\u202e" not in terminal
    assert "\n## Forged" not in terminal
    assert "packages/evil|row ## Forged" in terminal
    assert "Break **bold** ## Forged[31m" in terminal
    assert "test ## Command[31m" in terminal
    assert "FORGED|TOKEN ## Secret" in terminal
    assert "postgres ## Service" in terminal

    assert "\n## Forged" not in markdown
    assert r"packages/evil\|row \#\# Forged" in markdown
    assert r"\[link\](javascript:alert(1))" in markdown
    assert r"Break \*\*bold\*\* \#\# Forged\[31m" in markdown
    assert r"Use \| pipes - forged text." in markdown
    assert r"test \#\# Command\[31m" in markdown
    assert r"FORGED\|TOKEN \#\# Secret" in markdown
    assert r"postgres \#\# Service" in markdown


def test_report_rejects_packages_from_another_analysis(
    analysis: RepositoryAnalysis, packages: list[RepositoryPackage]
) -> None:
    wrong_package = RepositoryPackage(
        repository_id=_REPOSITORY_ID,
        analysis_id=UUID("40000000-0000-0000-0000-000000000000"),
        package_path=".",
        package_name="wrong",
        primary_language="python",
        rubric=_rubric(3),
        overall_score=3,
    )

    with pytest.raises(ValueError, match="belong to the supplied analysis"):
        build_machine_report(analysis, [*packages, wrong_package])


def test_report_rejects_incomplete_package_breakdown(
    analysis: RepositoryAnalysis, packages: list[RepositoryPackage]
) -> None:
    with pytest.raises(ValueError, match="complete persisted package list"):
        build_machine_report(analysis, packages[:1])


def test_report_rejects_duplicate_package_rows(
    analysis: RepositoryAnalysis, packages: list[RepositoryPackage]
) -> None:
    with pytest.raises(ValueError, match="each package exactly once"):
        build_machine_report(analysis, [packages[0], packages[0], packages[1]])


def test_analyzer_treats_the_workspace_index_as_no_harness_evidence(
    git_repo: Path,
) -> None:
    (git_repo / "README.md").write_text("# Example\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(git_repo), "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(git_repo), "commit", "--quiet", "-m", "initial"],
        check=True,
    )
    payload = {
        "summary": "A small application citing the index as Harness evidence.",
        "primary_language": "python",
        "is_monorepo": False,
        "packages": [
            {
                "path": ".",
                "name": "root",
                "primary_language": "python",
                "rubric": _rubric(3),
            }
        ],
        "recommendations": [],
        "themes": [],
        "command_catalog": {
            "source": ANALYSIS_INPUT_FILENAME,
            "commands": [{"stage": "test", "argv": ["make", "test"], "verifies": True}],
        },
    }
    repository = Repository(root=git_repo)
    adapter = MockAdapter(name="openai").queue(MockResponse(payload=payload))

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        with pytest.raises(AnalyzerError, match="lacks bounded evidence") as error:
            run_analyzer(
                repository,
                store,
                adapter,
                artifact_dir=git_repo / "artifacts",
            )
        assert store.list_analyses(repository.id) == []

    assert ANALYSIS_INPUT_FILENAME not in str(error.value)


def test_a_catalogued_command_must_say_whether_it_verifies(
    git_repo: Path,
) -> None:
    """The gate runs this list, so each entry answers what running it settles.

    Left to infer it, a gate cannot: a docs watch server and a docs build are
    one word apart in the same manifest.
    """
    (git_repo / "README.md").write_text("# Example\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(git_repo), "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(git_repo), "commit", "--quiet", "-m", "initial"],
        check=True,
    )
    payload = {
        "summary": "A small application whose catalogue declines the question.",
        "primary_language": "python",
        "is_monorepo": False,
        "packages": [
            {
                "path": ".",
                "name": "root",
                "primary_language": "python",
                "rubric": _rubric(3),
            }
        ],
        "recommendations": [],
        "themes": [],
        "command_catalog": {
            "source": "Makefile",
            "commands": [{"stage": "test", "argv": ["make", "test"]}],
        },
    }
    repository = Repository(root=git_repo)
    adapter = MockAdapter(name="openai").queue(MockResponse(payload=payload))

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        with pytest.raises(AnalyzerError, match="verifies"):
            run_analyzer(
                repository,
                store,
                adapter,
                artifact_dir=git_repo / "artifacts",
            )
        assert store.list_analyses(repository.id) == []


def test_a_command_directory_must_belong_to_the_repository(
    git_repo: Path,
) -> None:
    """A Dockerfile's WORKDIR reads like a directory and is one, elsewhere.

    Betterborg runs commands in the checkout, so the only directory it can act
    on is one relative to the repository root. An absolute path taken off an
    image build refuses at preflight, after analysis and planning have already
    been paid for.
    """
    (git_repo / "README.md").write_text("# Example\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(git_repo), "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(git_repo), "commit", "--quiet", "-m", "initial"],
        check=True,
    )
    payload = {
        "summary": "An application whose prepare step cites an image WORKDIR.",
        "primary_language": "go",
        "is_monorepo": False,
        "packages": [
            {
                "path": ".",
                "name": "root",
                "primary_language": "go",
                "rubric": _rubric(3),
            }
        ],
        "recommendations": [],
        "themes": [],
        "command_catalog": {
            "source": "Makefile",
            "commands": [
                {"stage": "test", "argv": ["make", "test"], "verifies": True}
            ],
        },
        "environment": {
            "files": ["Dockerfile"],
            "prepare_commands": [
                {
                    "argv": ["go", "mod", "vendor"],
                    "cwd": "/abs",
                    "source": "Dockerfile",
                }
            ],
        },
    }
    repository = Repository(root=git_repo)
    adapter = MockAdapter(name="openai").queue(MockResponse(payload=payload))

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        with pytest.raises(AnalyzerError, match="cwd"):
            run_analyzer(
                repository,
                store,
                adapter,
                artifact_dir=git_repo / "artifacts",
            )
        assert store.list_analyses(repository.id) == []


def test_a_catalogued_command_directory_must_belong_to_the_repository(
    git_repo: Path,
) -> None:
    """The rule holds for the catalog as well as for environment commands."""
    (git_repo / "README.md").write_text("# Example\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(git_repo), "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(git_repo), "commit", "--quiet", "-m", "initial"],
        check=True,
    )
    payload = {
        "summary": "An application whose catalogued check cites an image path.",
        "primary_language": "go",
        "is_monorepo": False,
        "packages": [
            {
                "path": ".",
                "name": "root",
                "primary_language": "go",
                "rubric": _rubric(3),
            }
        ],
        "recommendations": [],
        "themes": [],
        "command_catalog": {
            "source": "Makefile",
            "commands": [
                {
                    "stage": "test",
                    "argv": ["go", "test", "./..."],
                    "verifies": True,
                    "cwd": "/abs",
                }
            ],
        },
    }
    repository = Repository(root=git_repo)
    adapter = MockAdapter(name="openai").queue(MockResponse(payload=payload))

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        with pytest.raises(AnalyzerError, match="cwd"):
            run_analyzer(
                repository,
                store,
                adapter,
                artifact_dir=git_repo / "artifacts",
            )
        assert store.list_analyses(repository.id) == []


def test_the_analyzer_is_told_what_decides_whether_a_command_verifies() -> None:
    """The schema forces a boolean; only this sentence decides which one.

    Nothing downstream can recover the answer, so the rule living in one long
    prompt string is the whole contract. Trimmed or reformatted away, analysis
    still succeeds and the gate starts running watch servers again.
    """
    from betterborg_cli.repo_analysis.analyzer import _SYSTEM_PROMPT

    prompt = " ".join(_SYSTEM_PROMPT.split())
    assert "verifies is true only for a command that exits on its own" in prompt
    assert "leaves git status clean after it" in prompt
    # The admitted kinds, and the closing default that decides everything else.
    for admitted in ("a test run", "a linter", "a type checker", "a build"):
        assert admitted in prompt
    assert "It is false for everything else" in prompt
    for refused in ("serves", "watches", "publishes", "waits for input"):
        assert refused in prompt
    assert "where you cannot tell, verifies is false" in prompt
    # And the evidence the question depends on.
    assert ".gitignore" in prompt


@pytest.mark.parametrize(
    "cwd", [".", "package", "services/api", "Backend", "./package"]
)
def test_a_repository_relative_command_directory_is_accepted(
    git_repo: Path,
    cwd: str,
) -> None:
    """The rule refuses paths that leave the repository, and nothing else.

    Narrowing it further refuses a monorepo's whole analysis at the
    persistence edge, over a directory that was always fine.
    """
    (git_repo / "README.md").write_text("# Example\n", encoding="utf-8")
    (git_repo / "Makefile").write_text("test:\n\tgo test ./...\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(git_repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(git_repo), "commit", "--quiet", "-m", "initial"],
        check=True,
    )
    payload = {
        "summary": "An application whose check runs in a repository directory.",
        "primary_language": "go",
        "is_monorepo": False,
        "packages": [
            {
                "path": ".",
                "name": "root",
                "primary_language": "go",
                "rubric": _rubric(3),
            }
        ],
        "recommendations": [],
        "themes": [],
        "command_catalog": {
            "source": "Makefile",
            "commands": [
                {
                    "stage": "test",
                    "argv": ["go", "test", "./..."],
                    "verifies": True,
                    "cwd": cwd,
                    "source": "Makefile",
                }
            ],
        },
    }
    repository = Repository(root=git_repo)
    adapter = MockAdapter(name="openai").queue(MockResponse(payload=payload))

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        run_analyzer(
            repository, store, adapter, artifact_dir=git_repo / "artifacts"
        )
        stored = store.list_analyses(repository.id)

    assert len(stored) == 1
    assert stored[0].analysis_json["command_catalog"]["commands"][0]["cwd"] == cwd
