"""Detached planning worktree and context-materialization contracts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from betterborg_cli.planning import (
    PlanningWorktreeError,
    materialize_planning_worktree,
)
from betterborg_cli.repo_analysis import (
    DIMENSIONS,
    build_machine_report,
    render_markdown_report,
)
from betterborg_cli.store import (
    Borg,
    PlanChangeRequest,
    PlanningAttempt,
    PlanningFinding,
    PlanningQuestion,
    PrdSession,
    Repository,
    RepositoryAnalysis,
    RepositoryPackage,
    SqliteStore,
)


def test_materializes_detached_planning_context_without_touching_primary(
    committed_git_repo: Path,
) -> None:
    repository = Repository(root=committed_git_repo)
    borg = Borg(repository_id=repository.id, name="safe-planning")
    _write_config(committed_git_repo, repository)
    prd_path = Path(".borg/prds/safe-planning.md")
    (committed_git_repo / prd_path).parent.mkdir(parents=True)
    (committed_git_repo / prd_path).write_text(
        "# Confirmed PRD\n\nKeep the primary checkout unchanged.\n",
        encoding="utf-8",
    )
    deliberate_path = Path(".borg/notes/operator-context.md")
    (committed_git_repo / deliberate_path).parent.mkdir(parents=True)
    (committed_git_repo / deliberate_path).write_text(
        "# Operator context\n\nThis untracked Borg document is deliberate.\n",
        encoding="utf-8",
    )
    (committed_git_repo / ".borg/notes/not-supplied.md").write_text(
        "must stay out of the planning worktree\n", encoding="utf-8"
    )
    (committed_git_repo / "unrelated.py").write_text(
        "PRIMARY_DIRTY = True\n", encoding="utf-8"
    )

    database = committed_git_repo.parent / f"{committed_git_repo.name}-planning.db"
    with SqliteStore.open(database) as store:
        store.add_repository(repository)
        store.add_borg(borg)
        store.add_prd_session(
            PrdSession(
                repository_id=repository.id,
                borg_id=borg.id,
                prd_path=prd_path,
            )
        )
        analysis, packages = _persist_analysis(store, repository)
        prompts = {
            role: store.append_generated_prompt(
                repository_id=repository.id,
                analysis_id=analysis.id,
                role=role,
                body_md=f"# {role.title()} guidance\n\nUse the repository commands.\n",
            )
            for role in ("coding", "review", "merge")
        }
        attempt = PlanningAttempt(
            borg_id=borg.id,
            phase="tech_review",
            round=1,
            adapter="mock",
            model="planning-model",
        )
        store.append_planning_attempt(attempt)
        question = PlanningQuestion(
            borg_id=borg.id,
            attempt_id=attempt.id,
            round=1,
            questions=[{"id": "platforms", "question": "Which platforms?"}],
        )
        store.append_planning_question(question)
        answered = store.answer_planning_question(
            question.id,
            [{"q_id": "platforms", "answer": "Linux and macOS."}],
        )
        finding = PlanningFinding(
            borg_id=borg.id,
            attempt_id=attempt.id,
            round=1,
            severity="major",
            message="Rollback is not explicit.",
            suggestion="Add a recovery step.",
        )
        change_request = PlanChangeRequest(
            borg_id=borg.id,
            round=2,
            note="Keep migrations forward-only.",
            decided_by="operator",
        )
        store.append_planning_finding(finding)
        store.append_plan_change_request(change_request)

        status_before = _git(committed_git_repo, "status", "--short")
        worktrees_before = _git(committed_git_repo, "worktree", "list", "--porcelain")
        primary_prd_before = (committed_git_repo / prd_path).read_text(encoding="utf-8")
        current_plan = "# Current plan\n\n1. Preserve the checkout.\n"

        with materialize_planning_worktree(
            repository,
            borg,
            store,
            current_plan=current_plan,
            dirty_borg_documents=[deliberate_path],
        ) as worktree:
            assert not worktree.is_relative_to(committed_git_repo)
            assert worktree.is_dir()
            assert _git(worktree, "rev-parse", "--is-inside-work-tree") == "true\n"
            detached = subprocess.run(
                ["git", "-C", str(worktree), "symbolic-ref", "-q", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            )
            assert detached.returncode == 1
            assert (worktree / "README.md").is_file()
            assert not (worktree / "unrelated.py").exists()
            assert not (worktree / ".borg/notes/not-supplied.md").exists()
            assert (worktree / deliberate_path).read_text(encoding="utf-8") == (
                committed_git_repo / deliberate_path
            ).read_text(encoding="utf-8")
            assert (
                worktree / prd_path
            ).read_text(encoding="utf-8") == primary_prd_before
            assert (worktree / ".borg/config.toml").read_text(encoding="utf-8") == (
                committed_git_repo / ".borg/config.toml"
            ).read_text(encoding="utf-8")
            assert (worktree / ".borg/score.md").read_text(encoding="utf-8") == (
                render_markdown_report(build_machine_report(analysis, packages))
            )
            for role, prompt in prompts.items():
                assert (
                    worktree / f".borg/prompts/{role}.system.md"
                ).read_text(encoding="utf-8") == prompt.body_md
            assert (worktree / ".borg/plans/safe-planning.md").read_text(
                encoding="utf-8"
            ) == current_plan

            context = worktree / ".borg/state/planning/context"
            identity = _json(context / "repository.json")
            assert identity["repository"]["id"] == str(repository.id)
            assert identity["repository"]["head_sha"] == _git(
                committed_git_repo, "rev-parse", "HEAD"
            ).strip()
            persisted_analysis = _json(context / "analysis.json")
            assert persisted_analysis["id"] == str(analysis.id)
            assert persisted_analysis["analysis"] == analysis.analysis_json
            questions = _json(context / "questions.json")
            assert questions[0]["id"] == str(answered.id)
            assert questions[0]["answers"] == answered.answers
            changes = _json(context / "change-requests.json")
            assert changes[0]["note"] == change_request.note
            findings = _json(context / "findings.json")
            assert findings[0]["message"] == finding.message
            manifest = _json(context / "manifest.json")
            assert manifest["confirmed_prd"] == prd_path.as_posix()
            assert manifest["current_plan"] == ".borg/plans/safe-planning.md"
            assert manifest["dirty_borg_documents"] == [deliberate_path.as_posix()]
            assert set(manifest["prompts"]) == {"coding", "review", "merge"}

            assert _git(committed_git_repo, "status", "--short") == status_before
            assert (committed_git_repo / prd_path).read_text(
                encoding="utf-8"
            ) == primary_prd_before
            materialized_path = worktree

        assert not materialized_path.exists()
        assert _git(committed_git_repo, "status", "--short") == status_before
        assert (
            _git(committed_git_repo, "worktree", "list", "--porcelain")
            == worktrees_before
        )


def test_rejects_dirty_source_outside_borg_documents(
    committed_git_repo: Path,
) -> None:
    repository = Repository(root=committed_git_repo)
    borg = Borg(repository_id=repository.id, name="contained-planning")
    _write_config(committed_git_repo, repository)
    (committed_git_repo / "dirty-source.py").write_text(
        "DIRTY = True\n", encoding="utf-8"
    )
    database = committed_git_repo.parent / f"{committed_git_repo.name}-guard.db"

    with SqliteStore.open(database) as store:
        store.add_repository(repository)
        store.add_borg(borg)

        with pytest.raises(
            PlanningWorktreeError,
            match=r"repository-relative \.borg file",
        ):
            with materialize_planning_worktree(
                repository,
                borg,
                store,
                dirty_borg_documents=[Path("dirty-source.py")],
            ):
                pytest.fail("unsafe dirty source reached a planning worktree")


def _write_config(root: Path, repository: Repository) -> None:
    (root / ".borg").mkdir()
    (root / ".borg/config.toml").write_text(
        "version = 1\n\n"
        "[repository]\n"
        f'id = "{repository.id}"\n'
        'default_branch = "main"\n',
        encoding="utf-8",
    )


def _persist_analysis(
    store: SqliteStore, repository: Repository
) -> tuple[RepositoryAnalysis, list[RepositoryPackage]]:
    head_sha = _git(repository.root, "rev-parse", "HEAD").strip()
    analysis = RepositoryAnalysis(
        repository_id=repository.id,
        head_sha=head_sha,
        summary="A compact test repository.",
        primary_language="Python",
        is_monorepo=False,
        overall_score=4.0,
        analysis_json={
            "packages": [{"path": "."}],
            "themes": [],
            "command_catalog": {"commands": []},
            "environment": {"files": []},
            "required_secrets": [],
            "service_dependencies": [],
        },
    )
    package = RepositoryPackage(
        repository_id=repository.id,
        analysis_id=analysis.id,
        package_path=".",
        package_name="test-repository",
        primary_language="Python",
        rubric={dimension: {"score": 4} for dimension in DIMENSIONS},
        overall_score=4.0,
    )
    packages = [package]
    store.append_analysis(analysis, packages)
    return analysis, packages


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
