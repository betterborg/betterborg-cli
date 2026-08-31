"""Crash-safe publication of immutable task-generation Markdown trees."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from betterborg_cli.agent_runtime.base import CancellationToken
from betterborg_cli.agent_runtime.process import run_captured
from betterborg_cli.planning.pm import approved_plan_digest
from betterborg_cli.planning.task_render import (
    render_task_markdown,
    task_markdown_digest,
)
from betterborg_cli.prd_session import validate_borg_name
from betterborg_cli.repo_paths import RepoPaths, ensure_managed_gitignore
from betterborg_cli.repository_files import (
    RepositoryGitVisibilityError,
    RepositoryPathError,
    require_git_trackable,
)
from betterborg_cli.store import (
    Borg,
    BorgState,
    PlanningAttemptStatus,
    Repository,
    SqliteStore,
    TaskGeneration,
    TaskGenerationStatus,
    TaskRecord,
)

FailureInjector = Callable[[str], None]


class TaskPublicationError(RuntimeError):
    """Raised when a generation cannot safely become executable."""


class TaskPublicationCancelled(TaskPublicationError):
    """Raised when publication stops before its durable database commit."""


class TaskDigestDriftError(TaskPublicationError):
    """Raised when durable task bytes no longer match SQLite metadata."""


@dataclass(frozen=True, slots=True)
class PublishedTaskFile:
    """One verified task record and its tracked Markdown path."""

    task: TaskRecord
    path: Path


@dataclass(frozen=True, slots=True)
class TaskPublication:
    """The sole SQLite-current generation and all verified files it exposes."""

    generation: TaskGeneration
    files: tuple[PublishedTaskFile, ...]


class TaskPublisher:
    """Publish an approved preparing generation across the filesystem/DB seam."""

    def __init__(
        self,
        repository: Repository,
        store: SqliteStore,
        *,
        failure_injector: FailureInjector | None = None,
        cancel: CancellationToken | None = None,
        command_runner: Callable[..., subprocess.CompletedProcess[str]] = run_captured,
    ) -> None:
        self.cancel = cancel
        self.command_runner = command_runner
        self._raise_if_cancelled()
        try:
            paths = RepoPaths.discover(
                repository.root,
                cancel=cancel,
                command_runner=command_runner,
            )
        except ValueError as error:
            if cancel is not None and cancel.is_set():
                raise TaskPublicationCancelled(
                    "task publication cancelled during repository discovery"
                ) from error
            raise
        self._raise_if_cancelled()
        if paths.root != repository.root:
            raise ValueError("repository root does not match its discovered Git root")
        self.repository = repository
        self.store = store
        self.paths = paths
        self.failure_injector = failure_injector

    def publish(self, generation_id: UUID) -> TaskPublication:
        """Durably publish one approved generation, resuming any prior attempt."""
        self._raise_if_cancelled()
        generation = self.store.get_task_generation(generation_id)
        if generation is None:
            raise TaskPublicationError(f"task generation {generation_id} not found")
        borg = self._borg(generation)
        expected = self._expected_files(borg, generation)
        destination = self._generation_dir(borg, generation)

        current = self.store.get_current_task_generation(borg.id)
        if current is not None:
            self._verify_current(borg, current)
            self._raise_if_cancelled()
        if generation.status is TaskGenerationStatus.CURRENT:
            publication = self._verify_tree(borg, generation, destination, expected)
            self._raise_if_cancelled()
            self._cleanup_noncurrent(borg, generation.id)
            return publication
        if generation.status is not TaskGenerationStatus.PREPARING:
            raise TaskPublicationError(
                f"task generation {generation.id} is {generation.status.value}, "
                "not preparing"
            )
        self._require_approved_handoff(borg, generation)
        self._checkpoint("after_preparing")
        self._raise_if_cancelled()

        ensure_managed_gitignore(self.paths)
        staging_parent = self.paths.task_staging_dir / borg.name
        destination_parent = destination.parent
        self._mkdir_durable(staging_parent)
        self._mkdir_durable(destination_parent)
        if staging_parent.stat().st_dev != destination_parent.stat().st_dev:
            raise TaskPublicationError(
                "task staging and destination must be on the same filesystem"
            )

        if not os.path.lexists(destination):
            staging = staging_parent / str(generation.id)
            self._remove_tree(staging)
            self._raise_if_cancelled()
            self._stage(staging, expected)
            self._raise_if_cancelled()
            try:
                os.rename(staging, destination)
            except OSError as error:
                raise TaskPublicationError(
                    f"could not rename staged task generation: {error}"
                ) from error
            self._checkpoint("after_rename")
            _fsync_directory(staging_parent)
        else:
            self._verify_tree(borg, generation, destination, expected)

        self._checkpoint("during_parent_fsync")
        _fsync_directory(destination_parent)
        self._checkpoint("after_parent_fsync")
        self._raise_if_cancelled()
        publication = self._verify_tree(borg, generation, destination, expected)
        self._raise_if_cancelled()
        self._require_git_trackable(publication.files)
        self._checkpoint("before_db_commit")
        self._raise_if_cancelled()
        try:
            generation = self.store._promote_published_task_generation(
                generation.id, durable_root=destination
            )
        except (KeyError, ValueError) as error:
            raise TaskPublicationError(str(error)) from error
        self._checkpoint("after_db_commit")

        publication = TaskPublication(generation=generation, files=publication.files)
        self._verify_current(borg, generation)
        self._cleanup_noncurrent(borg, generation.id)
        return publication

    def current_task_files(self, borg_id: UUID) -> TaskPublication:
        """Return executable task files only after rechecking every digest."""
        publication = self.inspect_current_task_files(borg_id)
        self._cleanup_noncurrent(
            self._borg(publication.generation), publication.generation.id
        )
        return publication

    def inspect_current_task_files(self, borg_id: UUID) -> TaskPublication:
        """Read the verified SQLite-current task files without reconciling state."""
        borg = self.store.get_borg(borg_id)
        if borg is None:
            raise TaskPublicationError(f"Borg {borg_id} not found")
        current = self.store.get_current_task_generation(borg_id)
        if current is None:
            raise TaskPublicationError(f"Borg {borg.name!r} has no current tasks")
        return self._verify_current(borg, current)

    def reconcile(self, borg_id: UUID) -> TaskPublication | None:
        """Resume approved publication before cleaning noncurrent managed trees."""
        borg = self.store.get_borg(borg_id)
        if borg is None:
            raise TaskPublicationError(f"Borg {borg_id} not found")
        approved_preparing = [
            generation
            for generation in self.store.list_task_generations(borg_id)
            if generation.status is TaskGenerationStatus.PREPARING
            and self._has_approved_handoff(borg, generation)
        ]
        if len(approved_preparing) > 1:
            raise TaskPublicationError(
                f"Borg {borg.name!r} has multiple approved preparing generations"
            )
        if approved_preparing:
            return self.publish(approved_preparing[0].id)
        current = self.store.get_current_task_generation(borg_id)
        if current is None:
            return None
        publication = self._verify_current(borg, current)
        self._cleanup_noncurrent(borg, current.id)
        return publication

    def _borg(self, generation: TaskGeneration) -> Borg:
        borg = self.store.get_borg(generation.borg_id)
        if borg is None or borg.repository_id != self.repository.id:
            raise TaskPublicationError("task generation repository does not match")
        try:
            validate_borg_name(borg.name)
        except ValueError as error:
            raise TaskPublicationError(str(error)) from error
        return borg

    def _require_approved_handoff(self, borg: Borg, generation: TaskGeneration) -> None:
        if borg.state not in {
            BorgState.SUPERVISOR_WORKING,
            BorgState.READY_TO_EXECUTE,
        }:
            raise TaskPublicationError(
                "only a Supervisor-approved task generation can be published"
            )
        if not self._has_approved_handoff(borg, generation):
            raise TaskPublicationError(
                "task generation has no completed Supervisor approval"
            )

    def _has_approved_handoff(
        self, borg: Borg, generation: TaskGeneration
    ) -> bool:
        if borg.state not in {
            BorgState.SUPERVISOR_WORKING,
            BorgState.READY_TO_EXECUTE,
        }:
            return False
        return any(
            attempt.phase == "supervisor_review"
            and attempt.status is PlanningAttemptStatus.COMPLETED
            and (attempt.result or {}).get("decision") == "approve"
            and attempt.request.get("generation_id") == str(generation.id)
            and attempt.request.get("batch_id") == str(generation.batch_id)
            for attempt in self.store.list_planning_attempts(borg.id)
        )

    def _expected_files(
        self, borg: Borg, generation: TaskGeneration
    ) -> dict[Path, tuple[TaskRecord, bytes]]:
        if approved_plan_digest(generation.manifest) != generation.digest:
            raise TaskDigestDriftError(
                f"task generation {generation.id} manifest digest drifted"
            )
        records = self.store.list_task_records(generation.id)
        if not records:
            raise TaskPublicationError("a task generation must contain tasks")
        expected: dict[Path, tuple[TaskRecord, bytes]] = {}
        manifest_tasks: list[dict[str, object]] = []
        for record in records:
            body = render_task_markdown(record.task).encode("utf-8")
            digest = task_markdown_digest(body)
            if record.digest != digest or record.manifest.get("task.md") != digest:
                raise TaskDigestDriftError(
                    f"task record {record.task_ref} does not match rendered Markdown"
                )
            relative = Path(record.stage) / f"{record.stem}.md"
            expected[relative] = (record, body)
            manifest_tasks.append(
                {
                    "digest": digest,
                    "path": (
                        f".betterborg/tasks/{borg.name}/{generation.id}/"
                        f"{record.stage}/{record.stem}.md"
                    ),
                    "position": record.position,
                    "task_ref": record.task_ref,
                }
            )
        if generation.manifest.get("tasks") != manifest_tasks:
            raise TaskDigestDriftError(
                f"task generation {generation.id} file manifest drifted"
            )
        return expected

    def _generation_dir(self, borg: Borg, generation: TaskGeneration) -> Path:
        return self.paths.tasks_dir / borg.name / str(generation.id)

    def _stage(
        self,
        staging: Path,
        expected: dict[Path, tuple[TaskRecord, bytes]],
    ) -> None:
        self._mkdir_durable(staging)
        for index, (relative, (_record, body)) in enumerate(expected.items()):
            self._raise_if_cancelled()
            parent = staging / relative.parent
            self._mkdir_durable(parent)
            path = staging / relative
            with path.open("xb") as output:
                output.write(body)
                output.flush()
                os.fsync(output.fileno())
            if index == 0:
                self._checkpoint("during_staging")
            self._raise_if_cancelled()
        self._checkpoint("after_file_fsync")
        self._raise_if_cancelled()
        directories = {staging}
        directories.update(
            path.parent for path in (staging / item for item in expected)
        )
        for directory in sorted(
            directories, key=lambda item: len(item.parts), reverse=True
        ):
            _fsync_directory(directory)
        _fsync_directory(staging.parent)

    def _verify_current(
        self, borg: Borg, generation: TaskGeneration
    ) -> TaskPublication:
        if generation.status is not TaskGenerationStatus.CURRENT:
            raise TaskPublicationError("executable task generation is not current")
        expected = self._expected_files(borg, generation)
        return self._verify_tree(
            borg, generation, self._generation_dir(borg, generation), expected
        )

    def _verify_tree(
        self,
        borg: Borg,
        generation: TaskGeneration,
        root: Path,
        expected: dict[Path, tuple[TaskRecord, bytes]],
    ) -> TaskPublication:
        self._require_safe_directory_lineage(root.parent)
        if not root.is_dir() or root.is_symlink():
            raise TaskDigestDriftError(
                f"task generation {generation.id} durable tree is missing"
            )
        actual_files: set[Path] = set()
        actual_directories: set[Path] = set()
        for path in root.rglob("*"):
            if path.is_symlink():
                raise TaskDigestDriftError(
                    f"task generation {generation.id} contains a symlink"
                )
            relative = path.relative_to(root)
            if path.is_file():
                actual_files.add(relative)
            elif path.is_dir():
                actual_directories.add(relative)
            else:
                raise TaskDigestDriftError(
                    f"task generation {generation.id} contains a non-file entry"
                )
        expected_directories = {
            parent
            for relative in expected
            for parent in relative.parents
            if parent != Path(".")
        }
        if actual_files != set(expected) or actual_directories != expected_directories:
            raise TaskDigestDriftError(
                f"task generation {generation.id} durable layout drifted"
            )
        files: list[PublishedTaskFile] = []
        for relative, (record, body) in expected.items():
            path = root / relative
            try:
                actual = path.read_bytes()
            except OSError as error:
                raise TaskDigestDriftError(
                    f"could not read task file {relative.as_posix()}: {error}"
                ) from error
            if actual != body or task_markdown_digest(actual) != record.digest:
                raise TaskDigestDriftError(
                    f"task file {relative.as_posix()} digest drifted"
                )
            files.append(PublishedTaskFile(task=record, path=path))
        return TaskPublication(generation=generation, files=tuple(files))

    def _cleanup_noncurrent(self, borg: Borg, current_id: UUID) -> None:
        destination_parent = self.paths.tasks_dir / borg.name
        if destination_parent.is_dir():
            for path in destination_parent.iterdir():
                if path.name != str(current_id):
                    self._remove_tree(path)
        staging_parent = self.paths.task_staging_dir / borg.name
        if staging_parent.is_symlink():
            raise TaskPublicationError(
                f"task publication directory is a symlink: {staging_parent}"
            )
        if staging_parent.is_dir():
            self._require_safe_directory_lineage(staging_parent)
            for path in staging_parent.iterdir():
                self._remove_tree(path)
        if destination_parent.is_dir():
            _fsync_directory(destination_parent)

    def _require_git_trackable(
        self, files: tuple[PublishedTaskFile, ...]
    ) -> None:
        for published in files:
            try:
                require_git_trackable(
                    published.path,
                    root=self.paths.root,
                    cancel=self.cancel,
                    command_runner=self.command_runner,
                )
            except (RepositoryGitVisibilityError, RepositoryPathError) as error:
                if self.cancel is not None and self.cancel.is_set():
                    raise TaskPublicationCancelled(
                        "task publication cancelled while checking Git visibility"
                    ) from error
                raise TaskPublicationError(str(error)) from error
            self._raise_if_cancelled()

    def _mkdir_durable(self, path: Path) -> None:
        missing: list[Path] = []
        candidate = path
        while not candidate.exists():
            missing.append(candidate)
            candidate = candidate.parent
        self._require_safe_directory_lineage(candidate)
        path.mkdir(parents=True, exist_ok=True)
        self._require_safe_directory_lineage(path)
        for directory in reversed(missing):
            _fsync_directory(directory)
            _fsync_directory(directory.parent)

    def _require_safe_directory_lineage(self, path: Path) -> None:
        try:
            relative = path.relative_to(self.paths.root)
        except ValueError as error:
            raise TaskPublicationError(
                "task publication directory escapes repository"
            ) from error
        candidate = self.paths.root
        for component in relative.parts:
            candidate /= component
            if candidate.is_symlink():
                raise TaskPublicationError(
                    f"task publication directory is a symlink: {candidate}"
                )
        if not path.resolve().is_relative_to(self.paths.root):
            raise TaskPublicationError("task publication directory escapes repository")

    def _remove_tree(self, path: Path) -> None:
        if not os.path.lexists(path):
            return
        if path.is_symlink() or not path.is_dir():
            path.unlink()
        else:
            shutil.rmtree(path)
        if path.parent.is_dir():
            _fsync_directory(path.parent)

    def _checkpoint(self, name: str) -> None:
        if self.failure_injector is not None:
            self.failure_injector(name)

    def _raise_if_cancelled(self) -> None:
        if self.cancel is not None and self.cancel.is_set():
            raise TaskPublicationCancelled("task publication cancelled")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "PublishedTaskFile",
    "TaskDigestDriftError",
    "TaskPublication",
    "TaskPublicationCancelled",
    "TaskPublicationError",
    "TaskPublisher",
]
