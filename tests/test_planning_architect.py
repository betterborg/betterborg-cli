"""Terminal Architect lifecycle, durability, and resume contracts."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from betterborg_cli.agent_runtime.mock import MockAdapter, MockResponse
from betterborg_cli.planning import ArchitectLoop
from betterborg_cli.prd_session import InteractiveIO
from betterborg_cli.repo_analysis import DIMENSIONS
from betterborg_cli.store import (
    Borg,
    BorgState,
    PlanningAttempt,
    PlanningAttemptStatus,
    PlanningQuestion,
    PrdSession,
    Repository,
    RepositoryAnalysis,
    RepositoryPackage,
    SqliteStore,
)


def test_answers_product_questions_inline_and_persists_plan(
    committed_git_repo: Path,
) -> None:
    answers = iter(["Linux and macOS in the first release."])
    output: list[str] = []
    adapter = MockAdapter(name="openai").queue(
        MockResponse(
            payload={
                "decision": "ask_more",
                "questions": [
                    {
                        "id": "q1",
                        "question": "Which platforms are required at launch?",
                        "why": "This determines the packaging test matrix.",
                        "hint": "Name the required operating systems.",
                    }
                ],
            }
        )
    )
    adapter.queue(MockResponse(payload={"decision": "ready_to_plan"}))
    def plan_after_answer(spec):
        questions = json.loads(
            (
                spec.cwd / ".borg/state/planning/context/questions.json"
            ).read_text(encoding="utf-8")
        )
        assert questions[0]["answers"] == [
            {"q_id": "q1", "answer": "Linux and macOS in the first release."}
        ]
        return _plan()

    adapter.queue(MockResponse(dynamic=plan_after_answer))

    database = committed_git_repo.parent / "architect-inline.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = _planning_context(committed_git_repo, store, "inline")
        result = ArchitectLoop(
            repository,
            borg,
            store,
            adapter,
            io=_io(answers, output),
        ).run()

        assert result.plan == _plan()
        assert result.borg.state is BorgState.TECH_REVIEW_WORKING
        assert len(adapter.calls) == 3
        assert all(call.cwd != committed_git_repo for call in adapter.calls)
        assert all(not call.cwd.exists() for call in adapter.calls)
        assert "Why this matters" in output[0]
        assert "Answer guidance" in output[1]

        questions = store.list_planning_questions(borg.id)
        assert len(questions) == 1
        assert questions[0].answers == [
            {"q_id": "q1", "answer": "Linux and macOS in the first release."}
        ]
        attempts = store.list_planning_attempts(borg.id)
        assert [attempt.phase for attempt in attempts] == [
            "architect_questions",
            "architect_questions",
            "architect_plan",
        ]
        assert all(
            attempt.status is PlanningAttemptStatus.COMPLETED for attempt in attempts
        )
        assert attempts[-1].result == _plan()


def test_resumes_a_stored_question_before_invoking_the_agent(
    committed_git_repo: Path,
) -> None:
    prompts: list[str] = []
    adapter = MockAdapter(name="openai").queue(
        MockResponse(
            dynamic=lambda spec: (
                prompts.append(spec.user_prompt) or {"decision": "ready_to_plan"}
            )
        )
    )
    adapter.queue(MockResponse(payload=_plan()))
    database = committed_git_repo.parent / "architect-question-resume.sqlite3"

    with SqliteStore.open(database) as store:
        repository, draft = _planning_context(committed_git_repo, store, "resume")
        working = store.compare_and_set_borg_state(
            draft.id,
            expected_state=BorgState.DRAFT,
            expected_version=0,
            new_state=BorgState.ARCHITECT_WORKING,
        )
        attempt = PlanningAttempt(
            borg_id=working.id,
            phase="architect_questions",
            round=1,
            adapter="mock",
            model="test-model",
            status=PlanningAttemptStatus.COMPLETED,
            result={
                "decision": "ask_more",
                "questions": [{"id": "q1", "question": "Which users are first?"}],
            },
            started_at=working.created_at,
            finished_at=working.created_at,
        )
        question = PlanningQuestion(
            borg_id=working.id,
            attempt_id=attempt.id,
            round=1,
            questions=[{"id": "q1", "question": "Which users are first?"}],
        )
        with store.transaction():
            store.append_planning_attempt(attempt)
            store.append_planning_question(question)
            awaiting = store.compare_and_set_borg_state(
                working.id,
                expected_state=working.state,
                expected_version=working.state_version,
                new_state=BorgState.ARCHITECT_AWAITING_ANSWERS,
            )

        result = ArchitectLoop(
            repository,
            awaiting,
            store,
            adapter,
            io=_io(iter(["Small internal teams."]), []),
        ).run()

        assert result.plan == _plan()
        assert len(adapter.calls) == 2
        assert "question round 2" in prompts[0]
        assert store.list_planning_questions(working.id)[0].answers == [
            {"q_id": "q1", "answer": "Small internal teams."}
        ]


def test_recovers_provider_completed_plan_without_duplicate_invocation_or_cost(
    committed_git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    adapter.queue(MockResponse(payload=_plan()))
    database = committed_git_repo.parent / "architect-interrupted.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = _planning_context(committed_git_repo, store, "interrupted")
        loop = ArchitectLoop(repository, borg, store, adapter, io=_io(iter(()), []))
        original_complete = store.complete_planning_attempt
        interrupted = False

        def interrupt_after_plan_result(attempt_id, **kwargs):
            nonlocal interrupted
            attempt = next(
                item
                for item in store.list_planning_attempts(borg.id)
                if item.id == attempt_id
            )
            if attempt.phase == "architect_plan" and not interrupted:
                interrupted = True
                raise RuntimeError("simulated terminal interruption")
            return original_complete(attempt_id, **kwargs)

        with monkeypatch.context() as interruption:
            interruption.setattr(
                store, "complete_planning_attempt", interrupt_after_plan_result
            )
            with pytest.raises(RuntimeError, match="terminal interruption"):
                loop.run()

        assert len(adapter.calls) == 2
        running_plan = store.list_planning_attempts(borg.id)[-1]
        assert running_plan.phase == "architect_plan"
        assert running_plan.status is PlanningAttemptStatus.RUNNING
        assert Path(running_plan.request["result_path"]).is_file()

        resumed = ArchitectLoop(
            repository,
            store.get_borg(borg.id),
            store,
            adapter,
            io=_io(iter(()), []),
        ).run()

        assert resumed.plan == _plan()
        assert resumed.borg.state is BorgState.TECH_REVIEW_WORKING
        assert len(adapter.calls) == 2
        assert len(store.list_planning_attempts(borg.id)) == 2


def _io(answers: Iterator[str], output: list[str]) -> InteractiveIO:
    return InteractiveIO(
        prompt=lambda _message: next(answers, None),
        confirm=lambda _message, _default: False,
        write=output.append,
    )


def _planning_context(
    root: Path, store: SqliteStore, name: str
) -> tuple[Repository, Borg]:
    repository = Repository(root=root)
    borg = Borg(repository_id=repository.id, name=name)
    (root / ".borg/prds").mkdir(parents=True)
    (root / ".borg/config.toml").write_text(
        "version = 1\n\n"
        "[repository]\n"
        f'id = "{repository.id}"\n'
        'default_branch = "main"\n',
        encoding="utf-8",
    )
    prd_path = Path(f".borg/prds/{name}.md")
    (root / prd_path).write_text(
        "# Confirmed PRD\n\nBuild a tested release workflow.\n", encoding="utf-8"
    )
    store.add_repository(repository)
    store.add_borg(borg)
    store.add_prd_session(
        PrdSession(
            repository_id=repository.id,
            borg_id=borg.id,
            prd_path=prd_path,
        )
    )
    head_sha = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    rubric = {
        dimension: {"score": 3, "evidence": "README.md"} for dimension in DIMENSIONS
    }
    analysis = RepositoryAnalysis(
        repository_id=repository.id,
        head_sha=head_sha,
        summary="A small test repository.",
        primary_language="python",
        is_monorepo=False,
        overall_score=3,
        analysis_json={
            "packages": [{"path": "."}],
            "themes": [],
            "command_catalog": {"commands": []},
            "environment": {"files": []},
            "required_secrets": [],
            "service_dependencies": [],
        },
    )
    store.append_analysis(
        analysis,
        [
            RepositoryPackage(
                repository_id=repository.id,
                analysis_id=analysis.id,
                package_path=".",
                package_name="root",
                primary_language="python",
                rubric=rubric,
                overall_score=3,
            )
        ],
    )
    return repository, borg


def _plan() -> dict[str, object]:
    return {
        "title": "Release workflow",
        "summary": "Add a small, tested release workflow.",
        "overall_approach": (
            "Extend the existing repository conventions and verify the public "
            "behavior."
        ),
        "phases": [
            {
                "name": "01-release-workflow",
                "title": "Add release workflow",
                "goal": "Ship a repeatable release path.",
                "technical_approach": (
                    "Add the workflow beside existing automation and exercise "
                    "its contract."
                ),
                "files_touched": [
                    {
                        "path": ".github/workflows/release.yml",
                        "role": "new",
                        "description": "Builds release artifacts.",
                    }
                ],
                "test_strategy": (
                    "Run the repository's declared checks with a release fixture."
                ),
                "acceptance_criteria": ["A tagged release builds the package."],
                "deliverables": ["Release workflow"],
            }
        ],
        "code_pointers": [{"path": "README.md", "why": "Repository overview."}],
        "risks": [],
        "open_questions": [],
    }
