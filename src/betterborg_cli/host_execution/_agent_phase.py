"""Shared, internal contracts for host agent phases."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from betterborg_cli.agent_runtime import (
    AgentResult,
    AgentStatus,
    CancellationToken,
)
from betterborg_cli.host_execution.git import SafeGit
from betterborg_cli.host_execution.scheduler import ScheduledTaskContext
from betterborg_cli.planning import TaskDigestDriftError, TaskPublisher
from betterborg_cli.store import (
    ExecutionAttemptStatus,
    TaskRecord,
    TaskRuntime,
    TaskRuntimeStatus,
)


class HostAgentPhaseError(RuntimeError):
    """Raised when a claimed worktree is not safe or ready for an agent."""


@dataclass(frozen=True, slots=True)
class VerifiedTaskInputs:
    """Digest-verified task inputs and one generated role prompt."""

    task: TaskRecord
    task_path: Path
    task_markdown: str
    dependencies: tuple[tuple[TaskRecord, Path, str], ...]
    system_prompt: str


def cancelled_agent_reason(
    result: AgentResult,
    cancel: CancellationToken,
    *,
    phase: str,
) -> str | None:
    """Turn an adapter-level cancellation into a resumable run stop."""
    if result.status is not AgentStatus.CANCELLED:
        return None
    if cancel.is_set():
        return f"{phase} agent was interrupted"

    # API adapters also use CANCELLED when bounded transient retries are
    # exhausted. Propagate that resumable stop to the scheduler so it releases
    # the claim instead of treating the unchanged phase as a task failure (or
    # immediately invoking another review/fix attempt).
    cancel.cancel()
    return result.error or f"{phase} agent requested a resumable stop"


def require_ready_worktree(
    repository_root: Path,
    primary_git: SafeGit,
    context: ScheduledTaskContext,
    *,
    expected_statuses: Collection[TaskRuntimeStatus],
) -> tuple[TaskRuntime, Path]:
    """Validate ownership, registration, materialization, and branch identity."""
    runtime = context.runtime
    if runtime.status not in expected_statuses:
        expected = ", ".join(sorted(status.value for status in expected_statuses))
        raise HostAgentPhaseError(
            f"task is {runtime.status.value}, expected one of: {expected}"
        )
    if runtime.last_run_id != context.claim.run_id:
        raise HostAgentPhaseError("task runtime is not owned by the claimed run")
    if runtime.worktree_path is None or runtime.branch is None:
        raise HostAgentPhaseError("claimed task has no persisted worktree")
    worktree = Path(runtime.worktree_path).resolve()
    if not any(
        Path(entry.get("path", "")).resolve() == worktree
        and entry.get("branch") == f"refs/heads/{runtime.branch}"
        for entry in primary_git.worktree_list()
    ):
        raise HostAgentPhaseError(
            "claimed task path is not its registered Betterborg worktree"
        )
    materializations = [
        attempt
        for attempt in context.store.list_environment_attempts(
            context.claim.task_id
        )
        if attempt.kind == "materialize"
        and attempt.status is ExecutionAttemptStatus.COMPLETED
    ]
    if not materializations:
        raise HostAgentPhaseError("claimed task environment is not materialized")
    marker = worktree / ".borg/state/environment-materialization"
    try:
        fingerprint = marker.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise HostAgentPhaseError(
            "claimed task environment marker is missing"
        ) from error
    if fingerprint != materializations[-1].fingerprint:
        raise HostAgentPhaseError("claimed task environment marker has drifted")
    if current_branch(primary_git.for_worktree(worktree)) != runtime.branch:
        raise HostAgentPhaseError("claimed task worktree is on the wrong branch")
    return runtime, worktree


def verified_task_inputs(
    repository_root: Path,
    context: ScheduledTaskContext,
    worktree: Path,
    *,
    prompt_role: str,
) -> VerifiedTaskInputs:
    """Load current-generation task files and a persisted role prompt."""
    generation = context.store.get_task_generation(context.runtime.generation_id)
    if generation is None:
        raise HostAgentPhaseError("task generation is missing")
    borg = context.store.get_borg(generation.borg_id)
    if borg is None:
        raise HostAgentPhaseError("task Borg is missing")
    repository = context.store.get_repository(borg.repository_id)
    if repository is None or repository.root != repository_root:
        raise HostAgentPhaseError("task repository does not match agent checkout")
    publication = TaskPublisher(
        repository, context.store
    ).inspect_current_task_files(borg.id)
    by_id = {published.task.id: published for published in publication.files}
    published = by_id.get(context.claim.task_id)
    if published is None:
        raise HostAgentPhaseError("claimed task is not in the current generation")

    relative = published.path.relative_to(repository_root)
    task_markdown = read_digest_valid(worktree / relative, published.task.digest)
    dependency_ids = {
        edge.depends_on_task_id
        for edge in context.store.list_task_dependencies(generation.id)
        if edge.task_id == context.claim.task_id
    }
    dependencies: list[tuple[TaskRecord, Path, str]] = []
    for dependency_id in sorted(
        dependency_ids, key=lambda item: by_id[item].task.position
    ):
        dependency = by_id.get(dependency_id)
        if dependency is None:
            raise HostAgentPhaseError("task dependency is not in the generation")
        dependency_relative = dependency.path.relative_to(repository_root)
        dependencies.append(
            (
                dependency.task,
                dependency_relative,
                read_digest_valid(
                    worktree / dependency_relative, dependency.task.digest
                ),
            )
        )

    prompt = context.store.get_latest_generated_prompts(repository.id).get(
        prompt_role
    )
    if prompt is None or not prompt.body_md.strip():
        raise HostAgentPhaseError(
            f"repository has no generated {prompt_role} prompt"
        )
    return VerifiedTaskInputs(
        task=published.task,
        task_path=relative,
        task_markdown=task_markdown,
        dependencies=tuple(dependencies),
        system_prompt=prompt.body_md,
    )


class AgentAttemptArtifacts:
    """Create and seal one immutable host-agent artifact directory."""

    def __init__(
        self,
        repository_root: Path,
        attempt_dir: Path,
        worktree: Path,
        phase: str,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        self.attempt_dir = Path(attempt_dir).resolve()
        self.worktree = Path(worktree).resolve()
        self.phase = phase

    def write_text(self, name: str, content: str) -> Path:
        path = self.attempt_dir / name
        write_new(path, content)
        return path

    def finish(
        self, result: AgentResult, durable_result: dict[str, Any]
    ) -> None:
        canonical_result = self.attempt_dir / f"{self.phase}.result.json"
        if result.payload is not None and not canonical_result.exists():
            write_new(
                canonical_result,
                json.dumps(result.payload, indent=2, sort_keys=True) + "\n",
            )
        durable_result["_betterborg"]["adapter_artifacts"] = (
            self._snapshot_adapter_artifacts(result)
        )
        self.write_text(
            f"{self.phase}.outcome.json",
            json.dumps(
                {
                    "status": result.status.value,
                    "exit_code": result.exit_code,
                    "error": result.error,
                    "result": durable_result,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        files = sorted(
            path for path in self.attempt_dir.rglob("*") if path.is_file()
        )
        manifest = {
            path.relative_to(self.attempt_dir).as_posix(): (
                f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
            )
            for path in files
        }
        self.write_text(
            "artifact-manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        for path in self.attempt_dir.rglob("*"):
            if path.is_file():
                path.chmod(0o444)

    def reference(self, path: Path) -> str:
        resolved = Path(path).resolve()
        if resolved.is_relative_to(self.repository_root):
            return resolved.relative_to(self.repository_root).as_posix()
        return str(resolved)

    def _snapshot_adapter_artifacts(
        self, result: AgentResult
    ) -> list[dict[str, str]]:
        snapshots: list[dict[str, str]] = []
        destination_root = self.attempt_dir / "adapter-artifacts"
        for index, artifact in enumerate(result.artifacts, start=1):
            reference = str(artifact.path)
            if "://" in reference:
                snapshots.append(
                    {"kind": artifact.kind, "reference": reference}
                )
                continue
            source = Path(artifact.path)
            if not source.is_absolute():
                source = self.worktree / source
            source = source.resolve()
            if not source.is_file():
                raise OSError(f"agent artifact is not a file: {source}")
            if source.is_relative_to(self.attempt_dir):
                snapshot = source
            else:
                destination_root.mkdir(exist_ok=True)
                snapshot = destination_root / f"{index:03d}-{source.name}"
                write_new_bytes(snapshot, source.read_bytes())
            snapshots.append(
                {"kind": artifact.kind, "path": self.reference(snapshot)}
            )
        return snapshots


def current_branch(git: SafeGit) -> str:
    result = git.run(["rev-parse", "--abbrev-ref", "HEAD"], check=False)
    branch = result.stdout.strip()
    if result.returncode != 0 or not branch or branch == "HEAD":
        raise HostAgentPhaseError("task worktree is not on an attached branch")
    return branch


def read_digest_valid(path: Path, digest: str) -> str:
    try:
        body = path.read_bytes()
    except OSError as error:
        raise TaskDigestDriftError(f"task input is missing: {path}") from error
    actual = f"sha256:{hashlib.sha256(body).hexdigest()}"
    if actual != digest:
        raise TaskDigestDriftError(f"task input digest drifted: {path}")
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TaskDigestDriftError(f"task input is not UTF-8: {path}") from error


def result_summary(result: AgentResult) -> str:
    summary = (result.payload or {}).get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary[:1000]
    return (result.error or result.status.value)[:1000]


def write_new(path: Path, content: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(content)


def write_new_bytes(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
