"""Detached planning worktrees populated with durable BetterBorg context."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any
from uuid import uuid4

from betterborg_cli.repo_analysis import (
    build_machine_report,
    render_markdown_report,
)
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.repository_config import CONFIG_FILENAME, load_repository_config
from betterborg_cli.repository_files import publish_repository_text
from betterborg_cli.store import (
    Borg,
    PlanChangeRequest,
    PlanningFinding,
    PlanningQuestion,
    Repository,
    SqliteStore,
)

_CONTEXT_DIR = Path(".borg/state/planning/context")
_MANIFEST_PATH = _CONTEXT_DIR / "manifest.json"
_REPOSITORY_PATH = _CONTEXT_DIR / "repository.json"
_ANALYSIS_PATH = _CONTEXT_DIR / "analysis.json"
_QUESTIONS_PATH = _CONTEXT_DIR / "questions.json"
_CHANGE_REQUESTS_PATH = _CONTEXT_DIR / "change-requests.json"
_FINDINGS_PATH = _CONTEXT_DIR / "findings.json"


class PlanningWorktreeError(RuntimeError):
    """Raised when a planning worktree cannot be materialized safely."""


@contextmanager
def materialize_planning_worktree(
    repository: Repository,
    borg: Borg,
    store: SqliteStore,
    *,
    current_plan: str | None = None,
    dirty_borg_documents: Sequence[Path] = (),
    worktrees_root: Path | None = None,
) -> Iterator[Path]:
    """Yield an ephemeral detached worktree with current planning evidence.

    Git supplies only committed source. BetterBorg-owned context is rebuilt from
    the durable store and the confirmed PRD, while additional uncommitted files
    cross the checkout boundary only when the caller names a dirty ``.borg``
    document explicitly.
    """
    paths = _validate_inputs(repository, borg, store)
    supplied_documents = _read_deliberate_borg_documents(
        paths.root, dirty_borg_documents
    )
    root = Path(worktrees_root or paths.worktrees_dir / "planning").resolve()
    if root == paths.root or root.is_relative_to(paths.root):
        raise PlanningWorktreeError("planning worktrees must be outside the checkout")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{borg.id}-{uuid4().hex}"
    head_sha = _git_output(paths.root, "rev-parse", "--verify", "HEAD^{commit}")

    created = False
    active_error: BaseException | None = None
    try:
        try:
            _run_git(
                paths.root,
                "worktree",
                "add",
                "--detach",
                str(destination),
                head_sha,
            )
            created = True
            _materialize_context(
                destination=destination,
                paths=paths,
                repository=repository,
                borg=borg,
                store=store,
                head_sha=head_sha,
                current_plan=current_plan,
                supplied_documents=supplied_documents,
            )
        except (
            OSError,
            subprocess.CalledProcessError,
            UnicodeError,
            ValueError,
        ) as error:
            raise PlanningWorktreeError(
                f"unable to materialize planning worktree: {error}"
            ) from error
        yield destination
    except BaseException as error:
        active_error = error
        raise
    finally:
        if created:
            try:
                _run_git(
                    paths.root,
                    "worktree",
                    "remove",
                    "--force",
                    str(destination),
                )
            except (
                OSError,
                subprocess.CalledProcessError,
                UnicodeError,
                ValueError,
            ) as error:
                cleanup_error = PlanningWorktreeError(
                    f"unable to remove planning worktree {destination}: {error}"
                )
                if active_error is not None:
                    active_error.add_note(str(cleanup_error))
                else:
                    raise cleanup_error from error


def _validate_inputs(
    repository: Repository, borg: Borg, store: SqliteStore
) -> RepoPaths:
    if store.get_repository(repository.id) != repository:
        raise PlanningWorktreeError(
            "repository must already be present in the supplied store"
        )
    if borg.repository_id != repository.id or store.get_borg(borg.id) != borg:
        raise PlanningWorktreeError(
            "Borg must already belong to the supplied repository and store"
        )
    paths = RepoPaths.discover(repository.root)
    if paths.root != repository.root:
        raise PlanningWorktreeError(
            "repository root does not match its discovered Git root"
        )
    config = load_repository_config(paths)
    if config.repository_id != repository.id:
        raise PlanningWorktreeError(
            "tracked repository identity does not match the supplied repository"
        )
    return paths


def _materialize_context(
    *,
    destination: Path,
    paths: RepoPaths,
    repository: Repository,
    borg: Borg,
    store: SqliteStore,
    head_sha: str,
    current_plan: str | None,
    supplied_documents: Mapping[Path, str],
) -> None:
    session = store.get_prd_session_for_borg(borg.id)
    if session is None:
        raise PlanningWorktreeError("Borg has no persisted PRD session")
    prd_body = _read_owned_text(paths.root, session.prd_path, "confirmed PRD")

    analysis = store.get_prior_ready_analysis(repository.id)
    if analysis is None:
        raise PlanningWorktreeError("repository has no persisted analysis")
    packages = store.list_packages(analysis.id)
    prompts = store.get_latest_generated_prompts(repository.id)

    plan_path = (
        Path(".borg/plans") / f"{session.prd_path.stem}.md"
        if current_plan is not None
        else None
    )
    reserved = {
        Path(".borg") / CONFIG_FILENAME,
        paths.score_report.relative_to(paths.root),
        session.prd_path,
        _MANIFEST_PATH,
        _REPOSITORY_PATH,
        _ANALYSIS_PATH,
        _QUESTIONS_PATH,
        _CHANGE_REQUESTS_PATH,
        _FINDINGS_PATH,
        *(
            paths.prompts_dir.relative_to(paths.root) / f"{role}.system.md"
            for role in prompts
        ),
    }
    if plan_path is not None:
        reserved.add(plan_path)
    overlap = sorted(set(supplied_documents).intersection(reserved))
    if overlap:
        raise PlanningWorktreeError(
            "dirty Borg documents overlap generated planning context: "
            + ", ".join(path.as_posix() for path in overlap)
        )

    _publish(
        destination,
        Path(".borg") / CONFIG_FILENAME,
        _read_owned_text(
            paths.root,
            Path(".borg") / CONFIG_FILENAME,
            "repository identity",
        ),
    )
    _publish(destination, session.prd_path, prd_body)
    report = render_markdown_report(build_machine_report(analysis, packages))
    _publish(destination, paths.score_report.relative_to(paths.root), report)

    prompt_manifest: dict[str, dict[str, Any]] = {}
    for role, prompt in prompts.items():
        prompt_path = paths.prompts_dir.relative_to(paths.root) / f"{role}.system.md"
        _publish(destination, prompt_path, prompt.body_md)
        prompt_manifest[role] = {
            "analysis_id": str(prompt.analysis_id),
            "path": prompt_path.as_posix(),
            "version": prompt.version,
        }

    if plan_path is not None:
        _publish(destination, plan_path, current_plan or "")
    for relative_path, body in supplied_documents.items():
        _publish(destination, relative_path, body)

    _publish_json(
        destination,
        _REPOSITORY_PATH,
        {
            "borg": {
                "id": str(borg.id),
                "name": borg.name,
                "state": borg.state.value,
                "state_version": borg.state_version,
            },
            "repository": {
                "head_sha": head_sha,
                "id": str(repository.id),
                "name": repository.root.name,
            },
        },
    )
    _publish_json(
        destination,
        _ANALYSIS_PATH,
        {
            "analysis": analysis.analysis_json,
            "created_at": analysis.created_at.isoformat(),
            "head_sha": analysis.head_sha,
            "id": str(analysis.id),
            "overall_score": analysis.overall_score,
            "primary_language": analysis.primary_language,
        },
    )
    questions = store.list_planning_questions(borg.id)
    change_requests = store.list_plan_change_requests(borg.id)
    findings = store.list_planning_findings(borg.id)
    _publish_json(
        destination,
        _QUESTIONS_PATH,
        [_question_json(item) for item in questions],
    )
    _publish_json(
        destination,
        _CHANGE_REQUESTS_PATH,
        [_change_request_json(item) for item in change_requests],
    )
    _publish_json(
        destination,
        _FINDINGS_PATH,
        [_finding_json(item) for item in findings],
    )
    _publish_json(
        destination,
        _MANIFEST_PATH,
        {
            "analysis": _ANALYSIS_PATH.as_posix(),
            "change_requests": _CHANGE_REQUESTS_PATH.as_posix(),
            "confirmed_prd": session.prd_path.as_posix(),
            "current_plan": plan_path.as_posix() if plan_path is not None else None,
            "dirty_borg_documents": [
                path.as_posix() for path in sorted(supplied_documents)
            ],
            "findings": _FINDINGS_PATH.as_posix(),
            "prompts": prompt_manifest,
            "questions": _QUESTIONS_PATH.as_posix(),
            "repository": _REPOSITORY_PATH.as_posix(),
            "schema_version": 1,
            "score_report": paths.score_report.relative_to(paths.root).as_posix(),
        },
    )


def _read_deliberate_borg_documents(
    repository_root: Path, documents: Sequence[Path]
) -> dict[Path, str]:
    supplied: dict[Path, str] = {}
    for document in documents:
        relative = _borg_relative_path(document, repository_root)
        if relative in supplied:
            raise PlanningWorktreeError(
                f"dirty Borg document was supplied more than once: {relative}"
            )
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                relative.as_posix(),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if not status.stdout:
            raise PlanningWorktreeError(
                f"supplied Borg document is not dirty: {relative.as_posix()}"
            )
        supplied[relative] = _read_owned_text(
            repository_root, relative, "dirty Borg document"
        )
    return supplied


def _borg_relative_path(path: Path, repository_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            candidate = candidate.relative_to(repository_root)
        except ValueError as error:
            raise PlanningWorktreeError(
                f"dirty Borg document is outside the repository: {path}"
            ) from error
    if (
        candidate == Path(".")
        or ".." in candidate.parts
        or PureWindowsPath(str(candidate)).is_absolute()
        or not candidate.parts
        or candidate.parts[0] != ".borg"
    ):
        raise PlanningWorktreeError(
            f"dirty document must be a repository-relative .borg file: {path}"
        )
    return candidate


def _read_owned_text(repository_root: Path, relative: Path, label: str) -> str:
    source = repository_root / relative
    if source.is_symlink() or not source.is_file():
        raise PlanningWorktreeError(f"{label} is not a regular file: {relative}")
    if not source.resolve(strict=True).is_relative_to(repository_root):
        raise PlanningWorktreeError(f"{label} escapes the repository: {relative}")
    return source.read_text(encoding="utf-8")


def _publish(root: Path, relative: Path, body: str) -> None:
    publish_repository_text(root / relative, body, root=root, overwrite=True)


def _publish_json(root: Path, relative: Path, value: Any) -> None:
    _publish(root, relative, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _question_json(question: PlanningQuestion) -> dict[str, Any]:
    return {
        "answered_at": _timestamp(question.answered_at),
        "answers": question.answers,
        "asked_at": question.asked_at.isoformat(),
        "attempt_id": str(question.attempt_id) if question.attempt_id else None,
        "id": str(question.id),
        "questions": question.questions,
        "round": question.round,
    }


def _change_request_json(request: PlanChangeRequest) -> dict[str, Any]:
    return {
        "created_at": request.created_at.isoformat(),
        "decided_by": request.decided_by,
        "id": str(request.id),
        "note": request.note,
        "round": request.round,
    }


def _finding_json(finding: PlanningFinding) -> dict[str, Any]:
    return {
        "attempt_id": str(finding.attempt_id),
        "created_at": finding.created_at.isoformat(),
        "id": str(finding.id),
        "message": finding.message,
        "round": finding.round,
        "severity": finding.severity,
        "suggestion": finding.suggestion,
    }


def _timestamp(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _run_git(repository_root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def _git_output(repository_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
