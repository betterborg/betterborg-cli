"""Tracked improvement PRDs generated from canonical analysis themes."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from betterborg_cli.repo_analysis.improvement_prds import (
    generate_improvement_prds,
    resolve_theme_key,
)
from betterborg_cli.repo_analysis.scoring import DIMENSIONS
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.store import Borg, Repository, RepositoryAnalysis, SqliteStore


def _rubric(score: float = 3.0) -> dict[str, dict[str, object]]:
    return {
        dimension: {"score": score, "evidence": f"evidence for {dimension}"}
        for dimension in DIMENSIONS
    }


@pytest.fixture
def analysis(git_repo: Path) -> RepositoryAnalysis:
    repository_id = uuid4()
    rubric = _rubric()
    rubric["ci"]["score"] = 4.0
    return RepositoryAnalysis(
        repository_id=repository_id,
        head_sha="a" * 40,
        summary="A complete bounded analysis.",
        primary_language="python",
        is_monorepo=False,
        overall_score=3.125,
        analysis_json={
            "packages": [
                {
                    "path": ".",
                    "name": "demo",
                    "primary_language": "python",
                    "rubric": rubric,
                    "overall_score": 3.125,
                }
            ],
            "recommendations": [
                {
                    "id": "rec-ci",
                    "title": "Require CI checks",
                    "package_path": ".",
                    "dimension": "ci",
                    "manifest_evidence": [
                        ".github/workflows/ci.yml",
                        "pyproject.toml",
                    ],
                    "estimated_delta": 2.0,
                    "effort": "L",
                    "overlap_group": None,
                },
                {
                    "id": "rec-docs",
                    "title": "Document local setup",
                    "package_path": ".",
                    "dimension": "documentation",
                    "manifest_evidence": ["README.md"],
                    "estimated_delta": 0.5,
                    "effort": "S",
                    "overlap_group": None,
                },
            ],
            "themes": [
                {
                    "id": "CI / Safety",
                    "title": "Reliable checks",
                    "recommendation_ids": ["rec-ci"],
                    "effort": "S",
                    "effort_rationale": "One shared workflow change.",
                    "normalized_impact": 0.125,
                    "ranking_score": 0.125,
                    "recommendations": [
                        {
                            "id": "rec-ci",
                            "effective_delta": 1.0,
                            "delta_clamped": True,
                        }
                    ],
                },
                {
                    "id": "docs",
                    "title": "Approachable setup",
                    "recommendation_ids": ["rec-docs"],
                    "effort": "L",
                    "effort_rationale": "Documentation spans several audiences.",
                    "normalized_impact": 0.0625,
                    "ranking_score": 0.020833333333333332,
                    "recommendations": [
                        {
                            "id": "rec-docs",
                            "effective_delta": 0.5,
                            "delta_clamped": False,
                        }
                    ],
                },
            ],
        },
    )


def test_generates_one_prd_per_theme_with_exact_canonical_values(
    git_repo: Path, analysis: RepositoryAnalysis
) -> None:
    paths = RepoPaths.discover(git_repo)

    documents = generate_improvement_prds(
        analysis,
        paths,
        {"ci-safety": "Sentinel", "docs": "Scribe"},
    )

    assert [document.theme_key for document in documents] == ["ci-safety", "docs"]
    assert [document.path.name for document in documents] == [
        "ci-safety.md",
        "docs.md",
    ]
    assert sorted(path.name for path in paths.improvement_prds_dir.iterdir()) == [
        "ci-safety.md",
        "docs.md",
    ]

    ci_prd = (paths.improvement_prds_dir / "ci-safety.md").read_text(
        encoding="utf-8"
    )
    assert "# Reliable checks" in ci_prd
    assert "Sentinel" in ci_prd
    assert "**Require CI checks** in `.`" in ci_prd
    assert "**S** — One shared workflow change." in ci_prd
    assert "| `.` | `ci` | +2.0 | +1.0 |" in ci_prd
    assert "**+0.125** repository score." in ci_prd
    assert "`.github/workflows/ci.yml`, `pyproject.toml`" in ci_prd
    assert "**L**" not in ci_prd

    docs_prd = (paths.improvement_prds_dir / "docs.md").read_text(encoding="utf-8")
    assert "**L** — Documentation spans several audiences." in docs_prd
    assert "| `.` | `documentation` | +0.5 | +0.5 |" in docs_prd
    assert "**+0.0625** repository score." in docs_prd
    assert "`README.md`" in docs_prd
    assert "**S**" not in docs_prd


@pytest.mark.parametrize(
    ("suggestions", "message"),
    [
        ({"ci-safety": "Sentinel"}, "match theme keys exactly"),
        (
            {"ci-safety": "Sentinel", "docs": "Scribe", "unknown": "Extra"},
            "match theme keys exactly",
        ),
        ({"ci-safety": "", "docs": "Scribe"}, "must not be empty"),
        ({"ci-safety": "Same", "docs": "Same"}, "must be unique"),
        ({"ci-safety": " Sentinel", "docs": "Scribe"}, "trimmed line"),
    ],
)
def test_validates_all_suggested_names_before_writing(
    git_repo: Path,
    analysis: RepositoryAnalysis,
    suggestions: dict[str, str],
    message: str,
) -> None:
    paths = RepoPaths.discover(git_repo)

    with pytest.raises(ValueError, match=message):
        generate_improvement_prds(analysis, paths, suggestions)

    assert not paths.improvement_prds_dir.exists()


def test_store_validation_does_not_create_a_borg_or_start_a_prd_session(
    git_repo: Path, analysis: RepositoryAnalysis
) -> None:
    repository = Repository(root=git_repo, id=analysis.repository_id)
    existing = Borg(repository_id=repository.id, name="Sentinel")
    database = git_repo / ".borg" / "state" / "borg.sqlite3"
    paths = RepoPaths.discover(git_repo)

    with SqliteStore.open(database) as store:
        store.add_repository(repository)
        store.add_borg(existing)
        with pytest.raises(ValueError, match="already exists"):
            generate_improvement_prds(
                analysis,
                paths,
                {"ci-safety": "Sentinel", "docs": "Scribe"},
                store=store,
            )
        with store.locked_connection() as connection:
            borg_count = connection.execute("SELECT COUNT(*) FROM borgs").fetchone()[0]
            session_count = connection.execute(
                "SELECT COUNT(*) FROM prd_sessions"
            ).fetchone()[0]

    assert borg_count == 1
    assert session_count == 0
    assert not paths.improvement_prds_dir.exists()


@pytest.mark.parametrize(
    ("theme_id", "expected"),
    [("CI / Safety", "ci-safety"), (" Déploiement ", "deploiement")],
)
def test_theme_key_resolution_is_portable(theme_id: str, expected: str) -> None:
    assert resolve_theme_key(theme_id) == expected


def test_theme_key_resolution_rejects_empty_and_colliding_keys(
    git_repo: Path, analysis: RepositoryAnalysis
) -> None:
    with pytest.raises(ValueError, match="does not resolve"):
        resolve_theme_key("🚀")

    analysis.analysis_json["themes"][1]["id"] = "CI---Safety"
    paths = RepoPaths.discover(git_repo)
    with pytest.raises(ValueError, match="same key"):
        generate_improvement_prds(
            analysis,
            paths,
            {"ci-safety": "Sentinel"},
        )

    assert not paths.improvement_prds_dir.exists()
