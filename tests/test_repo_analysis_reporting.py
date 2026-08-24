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
from betterborg_cli.repo_analysis.analyzer import AnalyzerError, run_analyzer
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
                "commands": [{"stage": "test", "argv": ["make", "test"]}]
            },
            "required_secrets": [
                {"name": "PACKAGE_TOKEN", "used_by": ["test"], "scope": "build"}
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
    }


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
