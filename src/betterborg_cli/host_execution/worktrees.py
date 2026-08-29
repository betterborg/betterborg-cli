"""Persisted task-branch and sibling-worktree management."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

from betterborg_cli.host_execution.git import SafeGit
from betterborg_cli.host_execution.guard import PrimaryCheckoutGuard
from betterborg_cli.store import SqliteStore, TaskRuntime, TaskRuntimeStatus

_TASK_BRANCH = re.compile(
    r"^betterborg-tasks/(?P<stage>[^/]+)/(?P<stem>[^/]+)-(?P<identity>[0-9a-f]{16})$"
)


class WorktreeError(RuntimeError):
    """Raised when persisted task work cannot be materialized safely."""


@dataclass(frozen=True, slots=True)
class WorktreeSpec:
    """The durable Git identity and sibling path for one task."""

    path: Path
    branch: str


class HostWorktreeManager:
    """Create run-owned task worktrees without switching the primary checkout."""

    def __init__(
        self,
        repo_root: Path,
        worktree_root: Path,
        *,
        source_branch: str,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.worktree_root = Path(worktree_root).resolve()
        if self.worktree_root == self.repo_root or self.worktree_root.is_relative_to(
            self.repo_root
        ):
            raise WorktreeError("task worktrees must be siblings of the checkout")
        self.source_branch = source_branch
        self._git = SafeGit(self.repo_root)
        self._guard = PrimaryCheckoutGuard(self.repo_root)

    def ensure_project_base(self, project_name: str) -> str:
        """Create or fast-forward ``project/<name>`` from the configured source."""
        branch = f"project/{project_name}"
        self._validate_ref(branch)
        self._validate_ref(self.source_branch)
        self._guard.assert_clean("project-base preparation")
        if not self._git.branch_exists(branch):
            self._git.create_branch(branch, self.source_branch)
            return branch

        current = self._resolve(branch)
        target = self._resolve(self.source_branch)
        if current == target:
            return branch
        if not self._git.is_ancestor(current, target):
            raise WorktreeError(
                f"project base {branch!r} diverged from {self.source_branch!r}"
            )
        if not self._git.fast_forward_branch(branch, target):
            raise WorktreeError(
                f"project base {branch!r} could not be fast-forwarded safely"
            )
        return branch

    def prepare_current_task_worktrees(
        self,
        store: SqliteStore,
        *,
        run_id: UUID,
        owner_token: str,
        generation_id: UUID,
        project_name: str,
        now: datetime | None = None,
    ) -> list[WorktreeSpec]:
        """Persist and materialize every task in the run's current generation.

        Identity is committed before filesystem work. A crash, block, or Git
        error therefore leaves an exact branch/path pair for the next run to
        reuse instead of minting a replacement and losing access to task work.
        """
        records = store.list_task_records(generation_id)
        if not records:
            raise WorktreeError("current task generation has no tasks")

        prepared: list[WorktreeSpec] = []
        for task in records:
            runtime = store.get_task_runtime(task.id)
            if runtime is None:
                raise WorktreeError(f"task runtime is missing for {task.task_ref}")
            spec = self._persisted_or_new_spec(task.stage, task.stem, runtime)
            runtime = store.assign_task_worktree(
                run_id,
                owner_token,
                task.id,
                branch=spec.branch,
                worktree_path=str(spec.path),
                now=now,
            )
            persisted = self._spec_from_runtime(task.stage, task.stem, runtime)
            prepared.append(persisted)

        # Do not mutate Git until every identity has passed live-run ownership
        # checks and is durable. This also makes a partial Git failure wholly
        # resumable: all later tasks already have the exact identity to reuse.
        project_branch = self.ensure_project_base(project_name)
        for spec in prepared:
            self._ensure_worktree(spec, base_branch=project_branch)
        return prepared

    def cleanup_task_worktree(self, runtime: TaskRuntime) -> bool:
        """Remove a clean completed worktree while always retaining its branch.

        Blocked, failed, active, or dirty worktrees are deliberately retained.
        """
        if runtime.status is not TaskRuntimeStatus.DONE:
            return False
        return self._cleanup_task_worktree(runtime)

    def cleanup_published_task_worktree(self, runtime: TaskRuntime) -> bool:
        """Remove a sanity-published worktree before its terminal transition.

        The sanity phase calls this while holding the repository lock and while
        the runtime is still ``merging``. Keeping the runtime nonterminal until
        removal succeeds prevents dependents from racing worktree cleanup.
        """
        if runtime.status is not TaskRuntimeStatus.MERGING:
            return False
        return self._cleanup_task_worktree(runtime)

    def _cleanup_task_worktree(self, runtime: TaskRuntime) -> bool:
        """Remove one eligible task worktree without deleting its branch."""
        if runtime.branch is None or runtime.worktree_path is None:
            return False
        path = self._managed_path(Path(runtime.worktree_path))
        if not path.exists():
            return True
        if not self._is_worktree_for_branch(path, runtime.branch):
            raise WorktreeError(
                f"refusing to clean foreign path or branch at {path}"
            )
        try:
            self._git.remove_worktree(path)
        except subprocess.CalledProcessError as error:
            raise WorktreeError(
                f"completed task worktree is dirty; preserving {path}"
            ) from error
        return True

    def _persisted_or_new_spec(
        self, stage: str, stem: str, runtime: TaskRuntime
    ) -> WorktreeSpec:
        if runtime.branch is None and runtime.worktree_path is None:
            identity = uuid4().hex[:16]
            return WorktreeSpec(
                path=self.worktree_root / stage / f"{stem}-{identity}",
                branch=f"betterborg-tasks/{stage}/{stem}-{identity}",
            )
        return self._spec_from_runtime(stage, stem, runtime)

    def _spec_from_runtime(
        self, stage: str, stem: str, runtime: TaskRuntime
    ) -> WorktreeSpec:
        if runtime.branch is None or runtime.worktree_path is None:
            raise WorktreeError("persisted task identity is incomplete")
        match = _TASK_BRANCH.fullmatch(runtime.branch)
        if (
            match is None
            or match["stage"] != stage
            or match["stem"] != stem
        ):
            raise WorktreeError(
                f"persisted task branch does not match {stage}/{stem}: "
                f"{runtime.branch}"
            )
        path = self._managed_path(Path(runtime.worktree_path))
        expected = self.worktree_root / stage / f"{stem}-{match['identity']}"
        if path != expected:
            raise WorktreeError(
                f"persisted task worktree path does not match its branch: {path}"
            )
        return WorktreeSpec(path=path, branch=runtime.branch)

    def _ensure_worktree(self, spec: WorktreeSpec, *, base_branch: str) -> None:
        if self._is_worktree_for_branch(spec.path, spec.branch):
            return
        if spec.path.exists():
            raise WorktreeError(
                f"path exists but is not {spec.branch!r}: {spec.path}"
            )
        spec.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._git.add_worktree(spec.path, spec.branch, base=base_branch)
        except (OSError, subprocess.CalledProcessError) as error:
            raise WorktreeError(
                f"unable to create task worktree {spec.path}: {error}"
            ) from error

    def _is_worktree_for_branch(self, path: Path, branch: str) -> bool:
        if not path.exists() or not (path / ".git").exists():
            return False
        return any(
            Path(entry.get("path", "")).resolve() == path
            and entry.get("branch") == f"refs/heads/{branch}"
            for entry in self._git.worktree_list()
        )

    def _managed_path(self, path: Path) -> Path:
        resolved = path.resolve()
        if resolved == self.worktree_root or not resolved.is_relative_to(
            self.worktree_root
        ):
            raise WorktreeError(f"worktree path is outside managed root: {resolved}")
        return resolved

    def _validate_ref(self, reference: str) -> None:
        if not reference or reference.startswith("-"):
            raise WorktreeError(f"invalid Git reference: {reference!r}")
        if not self._git.is_valid_branch_name(reference):
            raise WorktreeError(f"invalid Git reference: {reference!r}")

    def _resolve(self, reference: str) -> str:
        result = self._git.run(
            ["rev-parse", "--verify", f"{reference}^{{commit}}"], check=False
        )
        value = result.stdout.strip()
        if result.returncode != 0 or not value:
            raise WorktreeError(f"Git reference does not resolve: {reference!r}")
        return value
