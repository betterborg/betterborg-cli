"""Claim-owned Docker Compose lifecycle for validated host services."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from uuid import UUID

from betterborg_cli.agent_runtime import CancellationToken
from betterborg_cli.agent_runtime.process import run_captured
from betterborg_cli.host_execution._locking import path_lock
from betterborg_cli.host_execution.preflight import HostPreflightPlan, HostService
from betterborg_cli.host_execution.scheduler import TaskActivitySink
from betterborg_cli.progress import AgentActivity, AgentActivityKind
from betterborg_cli.store import (
    ComposeResource,
    ExecutionEvent,
    ExecutionOwnershipError,
    SqliteStore,
    TaskClaim,
    TaskRuntimeStatus,
)
from betterborg_cli.store.models import utcnow

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
Clock = Callable[[], datetime]
ActivitySink = Callable[[AgentActivity], None]

_COMPOSE_ENVIRONMENT_NAMES = ("PATH", "PATHEXT", "SYSTEMROOT", "LANG", "LC_ALL")
_COMPOSE_HEALTH_TIMEOUT_SECONDS = 120
_COMPOSE_COMMAND_TIMEOUT_SECONDS = 150
_TIMED_OUT_RETURN_CODE = 124


class ComposeStackError(RuntimeError):
    """A Compose project could not become ready or stop exactly."""

    def __init__(
        self,
        message: str,
        *,
        project_name: str,
        command: Sequence[str],
    ) -> None:
        self.project_name = project_name
        self.command = tuple(command)
        super().__init__(
            f"{message}; project {project_name!r}; command: "
            f"{shlex.join(self.command)}"
        )


@dataclass(frozen=True, slots=True)
class ComposeStack:
    """The exact persisted Compose project owned by one task claim."""

    run_id: UUID
    claim_id: UUID
    task_id: UUID
    project_name: str
    docker_executable: Path
    network_name: str
    network_names: tuple[str, ...]
    image_names: tuple[str, ...]
    worktree: Path
    runtime_directory: Path
    compose_files: tuple[Path, ...]
    profiles: tuple[str, ...]
    services: tuple[str, ...]
    environment: dict[str, str]
    resources: tuple[ComposeResource, ...]


@dataclass(frozen=True, slots=True)
class ComposeCleanupResult:
    """Outcome of stopping one project supplied by reconciliation."""

    run_id: UUID
    task_id: UUID
    project_name: str
    command: tuple[str, ...]
    stopped: bool
    error: str | None = None


class HostComposeManager:
    """Start and stop only task/claim-owned validated Compose projects."""

    def __init__(
        self,
        repository_root: Path,
        *,
        environment: Mapping[str, str] | None = None,
        command_runner: CommandRunner | None = None,
        clock: Clock = utcnow,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        source_environment = os.environ if environment is None else environment
        self._environment = {
            name: source_environment[name]
            for name in _COMPOSE_ENVIRONMENT_NAMES
            if name in source_environment
        }
        self._run = command_runner or run_captured
        self._clock = clock

    def start_claimed_stack(
        self,
        store: SqliteStore,
        plan: HostPreflightPlan,
        claim: TaskClaim,
        owner_token: str,
        *,
        cancel: CancellationToken | None = None,
        activity: ActivitySink | None = None,
    ) -> ComposeStack | None:
        """Start selected Compose services and return their consumer contract."""
        return self._start_claimed_stack(
            store,
            plan,
            claim,
            owner_token,
            project_name=compose_project_name(claim),
            cancel=cancel,
            activity=activity,
        )

    def start_claimed_sanity_stack(
        self,
        store: SqliteStore,
        plan: HostPreflightPlan,
        claim: TaskClaim,
        owner_token: str,
        *,
        cancel: CancellationToken | None = None,
        activity: ActivitySink | None = None,
    ) -> ComposeStack | None:
        """Start a fresh sanity-only project for the merged task tip."""
        return self._start_claimed_stack(
            store,
            plan,
            claim,
            owner_token,
            project_name=f"{compose_project_name(claim)}-sanity",
            cancel=cancel,
            activity=activity,
        )

    def _start_claimed_stack(
        self,
        store: SqliteStore,
        plan: HostPreflightPlan,
        claim: TaskClaim,
        owner_token: str,
        *,
        project_name: str,
        cancel: CancellationToken | None,
        activity: ActivitySink | None,
    ) -> ComposeStack | None:
        compose_services = tuple(
            service for service in plan.services if service.kind == "compose"
        )
        if not compose_services:
            return None
        if not plan.compose_files:
            raise ValueError("validated Compose services require Compose files")
        docker_executable = _validated_docker_executable(plan)

        worktree = self._claimed_worktree(store, claim)
        source_files = self._worktree_compose_files(plan, worktree)
        selected = tuple(
            dict.fromkeys(
                service.compose_service
                for service in compose_services
                if service.compose_service is not None
            )
        )
        if any(service.compose_service is None for service in compose_services):
            raise ValueError("validated Compose services require exact service names")

        try:
            startup_files, files, network_names, image_names = (
                self._write_runtime_configuration(
                    docker_executable,
                    project_name,
                    source_files,
                    plan.compose_profiles,
                    plan.compose_networks,
                    plan.compose_volumes,
                    plan.compose_build_services,
                    compose_services,
                    selected,
                )
            )
        except ComposeStackError as error:
            self._block_task(store, claim, owner_token, str(error))
            raise
        runtime_directory = files[0].parent
        network_name = network_names[0]
        labels = {
            "betterborg.run_id": str(claim.run_id),
            "betterborg.claim_id": str(claim.id),
            "betterborg.task_id": str(claim.task_id),
            "betterborg.compose.executable": str(docker_executable),
            "betterborg.compose.files": json.dumps([str(path) for path in files]),
            "betterborg.compose.profiles": json.dumps(list(plan.compose_profiles)),
            "betterborg.compose.cwd": str(runtime_directory),
            "betterborg.compose.worktree": str(worktree),
            "com.docker.compose.project": project_name,
        }
        created_at = self._clock()
        resources = (
            ComposeResource(
                run_id=claim.run_id,
                claim_id=claim.id,
                task_id=claim.task_id,
                project_name=project_name,
                resource_type="project",
                resource_name=project_name,
                labels=labels,
                created_at=created_at,
            ),
            *(
                ComposeResource(
                    run_id=claim.run_id,
                    claim_id=claim.id,
                    task_id=claim.task_id,
                    project_name=project_name,
                    resource_type="network",
                    resource_name=name,
                    labels=labels,
                    created_at=created_at,
                )
                for name in network_names
            ),
            *(
                ComposeResource(
                    run_id=claim.run_id,
                    claim_id=claim.id,
                    task_id=claim.task_id,
                    project_name=project_name,
                    resource_type="image",
                    resource_name=name,
                    labels=labels,
                    created_at=created_at,
                )
                for name in image_names
            ),
        )
        command = _compose_base(
            docker_executable,
            project_name,
            startup_files,
            plan.compose_profiles,
        ) + (
            "up",
            "--detach",
            "--wait",
            "--wait-timeout",
            str(_COMPOSE_HEALTH_TIMEOUT_SECONDS),
            "--remove-orphans",
            "--no-deps",
            *selected,
        )
        stack = ComposeStack(
            run_id=claim.run_id,
            claim_id=claim.id,
            task_id=claim.task_id,
            project_name=project_name,
            docker_executable=docker_executable,
            network_name=network_name,
            network_names=network_names,
            image_names=image_names,
            worktree=worktree,
            runtime_directory=runtime_directory,
            compose_files=files,
            profiles=plan.compose_profiles,
            services=selected,
            environment={},
            resources=resources,
        )
        persisted = False
        with path_lock(self._lifecycle_lock(stack)):
            try:
                for resource in resources:
                    store.add_compose_resource(
                        resource,
                        owner_token,
                        claim.claim_token,
                        now=created_at,
                    )
                    persisted = True
                self._append_owned_event(
                    store,
                    claim,
                    owner_token,
                    "compose.starting",
                    project_name,
                    command,
                    now=created_at,
                )

                result = self._invoke(
                    command,
                    cwd=worktree,
                    cancel=cancel,
                    activity=activity,
                )
                self._raise_if_cancelled(cancel)
                if result.returncode != 0:
                    raise ComposeStackError(
                        _command_error(
                            "Compose services did not become healthy", result
                        ),
                        project_name=project_name,
                        command=command,
                    )
                self._assert_services_healthy(
                    docker_executable,
                    project_name,
                    files,
                    plan.compose_profiles,
                    selected,
                    runtime_directory,
                    cancel=cancel,
                    activity=activity,
                )
                published_ports = self._published_ports(
                    docker_executable,
                    project_name,
                    files,
                    plan.compose_profiles,
                    compose_services,
                    runtime_directory,
                    cancel=cancel,
                    activity=activity,
                )
                environment = service_url_environment(
                    plan.services,
                    published_ports=published_ports,
                )
                stack = replace(stack, environment=environment)

                # This guarded write is the final ownership fence.  Reconciliation
                # cannot run project teardown concurrently because it takes the
                # same claim-owned lifecycle lock before invoking Compose.
                self._append_owned_event(
                    store,
                    claim,
                    owner_token,
                    "compose.ready",
                    project_name,
                    command,
                    now=self._clock(),
                    extra={
                        "network_name": network_name,
                        "network_names": list(network_names),
                        "services": list(selected),
                        "url_env_names": sorted(
                            service.url_env
                            for service in plan.services
                            if service.url_env in environment
                        ),
                        "port_env_names": sorted(
                            env_name
                            for service in plan.services
                            for env_name, _port, _protocol in service.port_targets
                            if env_name in environment
                        ),
                    },
                )
            except ExecutionOwnershipError as error:
                cleanup = (
                    self._stop_project_locked(
                        store,
                        stack,
                        command_owner=None,
                        cancel=cancel,
                        activity=activity,
                    )
                    if persisted
                    else None
                )
                message = "Compose ownership expired during startup"
                if cleanup is not None and cleanup.error is not None:
                    message = f"{message}; cleanup also failed: {cleanup.error}"
                raise ComposeStackError(
                    message,
                    project_name=project_name,
                    command=command,
                ) from error
            except ComposeStackError as error:
                if cancel is not None and cancel.is_set():
                    cleanup = self._stop_project_locked(
                        store,
                        stack,
                        command_owner=None,
                        cancel=cancel,
                        activity=activity,
                    )
                    interrupted = KeyboardInterrupt()
                    if cleanup.error is not None:
                        interrupted.add_note(
                            f"Compose cleanup failed: {cleanup.error}"
                        )
                    raise interrupted from error
                try:
                    self._block_task(store, claim, owner_token, str(error))
                except ExecutionOwnershipError:
                    pass
                cleanup = self._stop_project_locked(
                    store,
                    stack,
                    command_owner=None,
                    cancel=cancel,
                    activity=activity,
                )
                message = str(error)
                if cleanup.error is not None:
                    message = f"{message}; cleanup also failed: {cleanup.error}"
                raise ComposeStackError(
                    message,
                    project_name=error.project_name,
                    command=error.command,
                ) from error
            except KeyboardInterrupt as error:
                cleanup = self._stop_project_locked(
                    store,
                    stack,
                    command_owner=None,
                    cancel=cancel,
                    activity=activity,
                )
                if cleanup.error is not None:
                    error.add_note(f"Compose cleanup failed: {cleanup.error}")
                raise
        return stack

    def _write_runtime_configuration(
        self,
        docker_executable: Path,
        project_name: str,
        files: Sequence[Path],
        profiles: Sequence[str],
        network_keys: Sequence[str],
        volume_keys: Sequence[str],
        build_services: Sequence[str],
        compose_services: Sequence[HostService],
        selected: Sequence[str],
    ) -> tuple[
        tuple[Path, ...],
        tuple[Path, ...],
        tuple[str, ...],
        tuple[str, ...],
    ]:
        """Create claim-owned startup isolation and secret-free cleanup files."""
        network_keys = tuple(network_keys) or ("default",)
        network_names = tuple(
            _owned_resource_name(project_name, key) for key in network_keys
        )
        owned_networks = dict(zip(network_keys, network_names, strict=True))
        owned_volumes = {
            key: _owned_resource_name(project_name, f"volume-{key}")
            for key in volume_keys
        }
        owned_images = {
            service: _owned_image_name(project_name, service)
            for service in build_services
            if service in selected
        }
        ports = _service_target_ports(compose_services)
        override_text = _render_runtime_override(
            selected,
            ports,
            owned_networks,
            owned_volumes,
            owned_images,
        )
        override_directory = self.repository_root / ".borg/state/compose" / project_name
        cleanup = override_directory / "compose.cleanup.json"
        override = override_directory / "compose.override.yml"
        cleanup_model = _cleanup_compose_model(
            selected,
            owned_networks,
            owned_volumes,
            owned_images,
        )
        try:
            override_directory.mkdir(parents=True, exist_ok=True)
            cleanup.write_text(
                json.dumps(cleanup_model, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            override.write_text(override_text, encoding="utf-8")
        except OSError as error:
            raise ComposeStackError(
                f"Compose isolation override could not be written: {error}",
                project_name=project_name,
                command=_compose_base(
                    docker_executable, project_name, files, profiles
                ),
            ) from error
        return (
            (*files, override),
            (cleanup, override),
            network_names,
            tuple(owned_images.values()),
        )

    def _assert_services_healthy(
        self,
        docker_executable: Path,
        project_name: str,
        files: Sequence[Path],
        profiles: Sequence[str],
        selected: Sequence[str],
        runtime_directory: Path,
        *,
        cancel: CancellationToken | None,
        activity: ActivitySink | None,
    ) -> None:
        command = _compose_base(docker_executable, project_name, files, profiles) + (
            "ps",
            "--format",
            "json",
            *selected,
        )
        result = self._invoke(
            command,
            cwd=runtime_directory,
            cancel=cancel,
            activity=activity,
        )
        self._raise_if_cancelled(cancel)
        healthy = _healthy_compose_services(result.stdout)
        if result.returncode != 0 or any(
            not healthy.get(service) or not all(healthy[service])
            for service in selected
        ):
            raise ComposeStackError(
                _command_error(
                    "Every selected Compose service must report healthy", result
                ),
                project_name=project_name,
                command=command,
            )

    def _published_ports(
        self,
        docker_executable: Path,
        project_name: str,
        files: Sequence[Path],
        profiles: Sequence[str],
        compose_services: Sequence[HostService],
        worktree: Path,
        *,
        cancel: CancellationToken | None,
        activity: ActivitySink | None,
    ) -> dict[tuple[str, int, str], int]:
        published: dict[tuple[str, int, str], int] = {}
        for service_name, target, protocol in _service_target_ports(compose_services):
            command = _compose_base(
                docker_executable, project_name, files, profiles
            ) + (
                "port",
                "--protocol",
                protocol,
                service_name,
                str(target),
            )
            result = self._invoke(
                command,
                cwd=worktree,
                cancel=cancel,
                activity=activity,
            )
            self._raise_if_cancelled(cancel)
            host_port = _parse_published_port(result.stdout)
            if result.returncode != 0 or host_port is None:
                raise ComposeStackError(
                    _command_error(
                        f"Compose published port for {service_name}:{target} "
                        "could not be resolved",
                        result,
                    ),
                    project_name=project_name,
                    command=command,
                )
            published[(service_name, target, protocol)] = host_port
        return published

    def stop_claimed_stack(
        self,
        store: SqliteStore,
        stack: ComposeStack,
        claim: TaskClaim,
        owner_token: str,
        *,
        cancel: CancellationToken | None = None,
        activity: ActivitySink | None = None,
    ) -> None:
        """Tear down exactly one live claim's persisted project."""
        if (stack.run_id, stack.claim_id, stack.task_id) != (
            claim.run_id,
            claim.id,
            claim.task_id,
        ):
            raise ValueError("Compose stack does not belong to the supplied claim")
        result = self._stop_project(
            store,
            stack,
            command_owner=(claim, owner_token),
            cancel=cancel,
            activity=activity,
        )
        if not result.stopped:
            raise ComposeStackError(
                result.error or "Compose teardown failed",
                project_name=result.project_name,
                command=result.command,
            )

    def cleanup_stale_projects(
        self,
        store: SqliteStore,
        resources: Sequence[ComposeResource],
        *,
        cancel: CancellationToken | None = None,
        activity: TaskActivitySink | None = None,
    ) -> tuple[ComposeCleanupResult, ...]:
        """Stop exactly the project identities returned by reconciliation."""
        groups: dict[tuple[UUID, UUID, str], list[ComposeResource]] = {}
        for resource in resources:
            key = (resource.run_id, resource.task_id, resource.project_name)
            groups.setdefault(key, []).append(resource)

        outcomes: list[ComposeCleanupResult] = []
        for project_resources in groups.values():
            stack = _stack_from_resources(
                project_resources,
                fallback_worktree=self.repository_root,
            )
            outcomes.append(
                self._stop_project(
                    store,
                    stack,
                    command_owner=None,
                    cancel=cancel,
                    activity=(
                        (lambda item, task_id=stack.task_id: activity(task_id, item))
                        if activity is not None
                        else None
                    ),
                )
            )
        return tuple(outcomes)

    def _stop_project(
        self,
        store: SqliteStore,
        stack: ComposeStack,
        *,
        command_owner: tuple[TaskClaim, str] | None,
        cancel: CancellationToken | None,
        activity: ActivitySink | None,
    ) -> ComposeCleanupResult:
        with path_lock(self._lifecycle_lock(stack)):
            return self._stop_project_locked(
                store,
                stack,
                command_owner=command_owner,
                cancel=cancel,
                activity=activity,
            )

    def _stop_project_locked(
        self,
        store: SqliteStore,
        stack: ComposeStack,
        *,
        command_owner: tuple[TaskClaim, str] | None,
        cancel: CancellationToken | None,
        activity: ActivitySink | None,
    ) -> ComposeCleanupResult:
        command = _compose_base(
            stack.docker_executable,
            stack.project_name,
            stack.compose_files,
            stack.profiles,
        ) + ("down", "--volumes", "--remove-orphans", "--rmi", "all")
        now = self._clock()
        if command_owner is None:
            store.append_execution_event(
                ExecutionEvent(
                    run_id=stack.run_id,
                    task_id=stack.task_id,
                    kind="compose.stopping",
                    payload={
                        "project_name": stack.project_name,
                        "command": list(command),
                    },
                    created_at=now,
                )
            )
        else:
            claim, owner_token = command_owner
            self._append_owned_event(
                store,
                claim,
                owner_token,
                "compose.stopping",
                stack.project_name,
                command,
                now=now,
            )

        result = self._invoke(
            command,
            cwd=stack.runtime_directory,
            cancel=cancel,
            activity=activity,
            cleanup=True,
        )
        if result.returncode != 0:
            error = _command_error("Compose teardown failed", result)
            failure_recorded = store.record_compose_cleanup_failure(
                stack.run_id,
                stack.task_id,
                stack.project_name,
                command=command,
                error=error,
                now=self._clock(),
            )
            if not failure_recorded:
                return ComposeCleanupResult(
                    stack.run_id,
                    stack.task_id,
                    stack.project_name,
                    command,
                    True,
                )
            return ComposeCleanupResult(
                stack.run_id,
                stack.task_id,
                stack.project_name,
                command,
                False,
                error,
            )

        store.confirm_compose_project_cleanup(
            stack.run_id,
            stack.task_id,
            stack.project_name,
            command=command,
            now=self._clock(),
        )
        return ComposeCleanupResult(
            stack.run_id,
            stack.task_id,
            stack.project_name,
            command,
            True,
        )

    @staticmethod
    def _lifecycle_lock(stack: ComposeStack) -> Path:
        return stack.runtime_directory / ".betterborg-lifecycle.lock"

    def _invoke(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path,
        cancel: CancellationToken | None,
        activity: ActivitySink | None,
        cleanup: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        self._report_command(command, activity)
        deadline = (
            cancel.force_deadline
            or time.monotonic() + _COMPOSE_COMMAND_TIMEOUT_SECONDS
            if cleanup and cancel is not None
            else None
        )
        try:
            return self._run(
                list(command),
                cwd=cwd,
                env=self._environment,
                check=False,
                timeout=_COMPOSE_COMMAND_TIMEOUT_SECONDS,
                cancel=cancel,
                terminate_on_cancel=not cleanup or cancel is None,
                deadline=deadline,
            )
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(
                command,
                _TIMED_OUT_RETURN_CODE,
                "",
                "Compose command timed out",
            )
        except OSError as error:
            return subprocess.CompletedProcess(command, 127, "", str(error))

    @staticmethod
    def _report_command(
        command: Sequence[str], activity: ActivitySink | None
    ) -> None:
        if activity is None:
            return
        try:
            activity(AgentActivity(AgentActivityKind.COMMAND, shlex.join(command)))
        except Exception:
            return

    @staticmethod
    def _raise_if_cancelled(cancel: CancellationToken | None) -> None:
        if cancel is not None and cancel.is_set():
            raise KeyboardInterrupt

    def _claimed_worktree(self, store: SqliteStore, claim: TaskClaim) -> Path:
        runtime = store.get_task_runtime(claim.task_id)
        if runtime is None or runtime.worktree_path is None:
            raise ValueError("claimed task does not have an assigned worktree")
        worktree = Path(runtime.worktree_path).resolve()
        if not worktree.is_dir():
            raise ValueError(f"claimed task worktree does not exist: {worktree}")
        return worktree

    def _worktree_compose_files(
        self, plan: HostPreflightPlan, worktree: Path
    ) -> tuple[Path, ...]:
        if plan.repository_root.resolve() != self.repository_root:
            raise ValueError("Compose manager and preflight plan roots differ")
        files: list[Path] = []
        for source in plan.compose_files:
            try:
                relative = source.relative_to(plan.repository_root)
            except ValueError as error:
                raise ValueError(
                    "Compose file is outside the validated repository"
                ) from error
            candidate = (worktree / relative).resolve()
            if not candidate.is_relative_to(worktree) or not candidate.is_file():
                raise ValueError(
                    f"validated Compose file is missing from task worktree: {relative}"
                )
            files.append(candidate)
        return tuple(files)

    def _append_owned_event(
        self,
        store: SqliteStore,
        claim: TaskClaim,
        owner_token: str,
        kind: str,
        project_name: str,
        command: Sequence[str],
        *,
        now: datetime,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "claim_id": str(claim.id),
            "project_name": project_name,
            "command": list(command),
        }
        payload.update(extra or {})
        store.append_claim_execution_event(
            ExecutionEvent(
                run_id=claim.run_id,
                task_id=claim.task_id,
                kind=kind,
                payload=payload,
                created_at=now,
            ),
            owner_token,
            claim.claim_token,
            now=now,
        )

    def _block_task(
        self,
        store: SqliteStore,
        claim: TaskClaim,
        owner_token: str,
        reason: str,
    ) -> None:
        runtime = store.get_task_runtime(claim.task_id)
        if runtime is None or runtime.status in {
            TaskRuntimeStatus.DONE,
            TaskRuntimeStatus.BLOCKED,
            TaskRuntimeStatus.FAILED,
        }:
            return
        store.transition_task_runtime(
            claim.run_id,
            owner_token,
            claim.id,
            claim.claim_token,
            expected_status=runtime.status,
            new_status=TaskRuntimeStatus.BLOCKED,
            resume_phase=runtime.resume_phase,
            state_reason=reason,
            now=self._clock(),
        )


def compose_project_name(claim: TaskClaim) -> str:
    """Return a stable run/task/claim-unique Docker Compose project name."""
    return (
        f"borg-{claim.run_id.hex[:6]}-{claim.task_id.hex[:6]}-"
        f"{claim.id.hex}"
    )


def service_url_environment(
    services: Sequence[HostService],
    *,
    published_ports: Mapping[tuple[str, int] | tuple[str, int, str], int]
    | None = None,
) -> dict[str, str]:
    """Return endpoint variables declared by the validated service plan."""
    environment: dict[str, str] = {}
    for service in services:
        if (
            service.kind == "external"
            and service.url_env is not None
            and service.url is not None
        ):
            environment[service.url_env] = service.url
        elif service.kind == "compose" and service.compose_service is not None:
            for env_name, target_port, protocol in service.url_targets:
                if service.url is not None and env_name == service.url_env:
                    environment[env_name] = service.url
                    continue
                published = _published_port(
                    published_ports,
                    service.compose_service,
                    target_port,
                    protocol,
                )
                if published is not None:
                    environment[env_name] = _service_url(
                        env_name,
                        service.compose_service,
                        "127.0.0.1",
                        published,
                        target_port=target_port,
                        protocol=protocol,
                    )
            for env_name, target_port, protocol in service.port_targets:
                published = _published_port(
                    published_ports,
                    service.compose_service,
                    target_port,
                    protocol,
                )
                if published is not None:
                    environment[env_name] = str(published)
    return environment


def _published_port(
    published_ports: Mapping[tuple[str, int] | tuple[str, int, str], int] | None,
    service: str,
    port: int,
    protocol: str,
) -> int | None:
    published = (published_ports or {}).get((service, port, protocol))
    if published is None and protocol == "tcp":
        published = (published_ports or {}).get((service, port))
    return published


def _service_url(
    env_name: str,
    service_identity: str,
    host: str,
    port: int,
    *,
    target_port: int | None = None,
    protocol: str = "tcp",
) -> str:
    if protocol == "udp":
        return f"udp://{host}:{port}"
    key = f"{env_name} {service_identity}".casefold()
    if "redis" in key:
        return f"redis://{host}:{port}/0"
    if "postgres" in key or "database" in key:
        return f"postgres://{host}:{port}/postgres"
    if "mysql" in key or "mariadb" in key:
        return f"mysql://{host}:{port}/mysql"
    if "mongo" in key:
        return f"mongodb://{host}:{port}"
    if "http" in key or target_port in {80, 3000, 8000, 8080}:
        return f"http://{host}:{port}"
    return f"tcp://{host}:{port}"


def _service_target_ports(
    services: Sequence[HostService],
) -> tuple[tuple[str, int, str], ...]:
    targets: list[tuple[str, int, str]] = []
    for service in services:
        if service.compose_service is None:
            continue
        targets.extend(
            (service.compose_service, port, protocol)
            for _env_name, port, protocol in service.url_targets
        )
        targets.extend(
            (service.compose_service, port, protocol)
            for _env_name, port, protocol in service.port_targets
        )
        if service.port is not None:
            protocol = next(
                (
                    target_protocol
                    for _env_name, target_port, target_protocol in service.url_targets
                    if target_port == service.port
                ),
                "tcp",
            )
            targets.append((service.compose_service, service.port, protocol))
    return tuple(dict.fromkeys(targets))


def _owned_resource_name(project_name: str, key: str) -> str:
    return f"{project_name}_{key}"


def _owned_image_name(project_name: str, service: str) -> str:
    service_identity = hashlib.sha256(service.encode()).hexdigest()[:12]
    return f"betterborg/{project_name}-{service_identity}:claim"


def _render_runtime_override(
    services: Sequence[str],
    ports: Sequence[tuple[str, int, str]],
    networks: Mapping[str, str],
    volumes: Mapping[str, str],
    images: Mapping[str, str],
) -> str:
    by_service: dict[str, list[tuple[int, str]]] = {name: [] for name in services}
    for service, target, protocol in ports:
        by_service[service].append((target, protocol))
    lines = ["services:"]
    for service in services:
        lines.extend(
            (
                f"  {json.dumps(service)}:",
                "    container_name: !reset null",
            )
        )
        if service in images:
            lines.append(f"    image: {json.dumps(images[service])}")
            lines.append("    pull_policy: build")
        targets = tuple(dict.fromkeys(by_service[service]))
        if not targets:
            lines.append("    ports: !override []")
            continue
        lines.append("    ports: !override")
        for target, protocol in targets:
            lines.extend(
                (
                    f"      - target: {target}",
                    '        published: "0"',
                    '        host_ip: "127.0.0.1"',
                    f"        protocol: {json.dumps(protocol)}",
                )
            )
    if networks:
        lines.append("networks:")
        for key, name in networks.items():
            lines.extend(
                (
                    f"  {json.dumps(key)}:",
                    f"    name: {json.dumps(name)}",
                    "    external: false",
                )
            )
    if volumes:
        lines.append("volumes:")
        for key, name in volumes.items():
            lines.extend(
                (
                    f"  {json.dumps(key)}:",
                    f"    name: {json.dumps(name)}",
                    "    external: false",
                )
            )
    return "\n".join(lines) + "\n"


def _cleanup_compose_model(
    services: Sequence[str],
    networks: Mapping[str, str],
    volumes: Mapping[str, str],
    images: Mapping[str, str],
) -> dict[str, object]:
    """Return only immutable project metadata needed by ``compose down``.

    The resolved Compose model can contain literals from service env files,
    interpolation, build arguments, and other repository-controlled fields.
    Cleanup deliberately retains none of those values.  A harmless build
    marker preserves Compose's default project/service image identity.  A
    claim-owned image name is retained only for validated build services so
    ``down --rmi all`` removes those exact tags without touching shared images.
    """
    service_models: dict[str, dict[str, object]] = {
        service: {"build": {"context": "."}} for service in services
    }
    for service, image in images.items():
        service_models[service]["image"] = image
    model: dict[str, object] = {
        "services": service_models,
        "networks": {
            key: {"name": name, "external": False}
            for key, name in networks.items()
        },
    }
    if volumes:
        model["volumes"] = {
            key: {"name": name, "external": False}
            for key, name in volumes.items()
        }
    return model


def _parse_published_port(output: str) -> int | None:
    address = output.strip().splitlines()[0] if output.strip() else ""
    _separator, _colon, port_text = address.rpartition(":")
    try:
        port = int(port_text)
    except ValueError:
        return None
    return port if 0 < port <= 65535 else None


def _healthy_compose_services(output: str) -> dict[str, list[bool]]:
    records: list[object]
    try:
        decoded = json.loads(output)
    except json.JSONDecodeError:
        records = []
        for line in output.splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                return {}
    else:
        if isinstance(decoded, list):
            records = decoded
        else:
            records = [decoded]

    services: dict[str, list[bool]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            return {}
        service = record.get("Service")
        state = record.get("State")
        health = record.get("Health")
        if not all(isinstance(value, str) for value in (service, state, health)):
            return {}
        services.setdefault(service, []).append(
            state.casefold() == "running" and health.casefold() == "healthy"
        )
    return services


def _compose_base(
    docker_executable: Path,
    project_name: str,
    compose_files: Sequence[Path],
    profiles: Sequence[str],
) -> tuple[str, ...]:
    command = [str(docker_executable), "compose", "--project-name", project_name]
    for path in compose_files:
        command.extend(("--file", str(path)))
    for profile in profiles:
        command.extend(("--profile", profile))
    return tuple(command)


def _stack_from_resources(
    resources: Sequence[ComposeResource], *, fallback_worktree: Path
) -> ComposeStack:
    if not resources:
        raise ValueError("stale Compose cleanup requires recorded resources")
    first = resources[0]
    if any(
        (resource.run_id, resource.task_id, resource.project_name)
        != (first.run_id, first.task_id, first.project_name)
        for resource in resources
    ):
        raise ValueError("stale Compose resources identify different projects")
    labels = first.labels
    try:
        files = tuple(
            Path(value)
            for value in json.loads(labels["betterborg.compose.files"])
        )
        profiles = tuple(json.loads(labels["betterborg.compose.profiles"]))
        runtime_directory = Path(labels["betterborg.compose.cwd"])
        worktree = Path(labels.get("betterborg.compose.worktree", runtime_directory))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        # Schema-v6 resources predate the runtime metadata labels.  Their exact
        # project identity remains sufficient for a project-scoped teardown;
        # Compose performs its normal file discovery from the trusted root.
        files = ()
        profiles = ()
        worktree = fallback_worktree
        runtime_directory = fallback_worktree
    docker_executable = Path(labels.get("betterborg.compose.executable", "docker"))
    network = next(
        (
            resource.resource_name
            for resource in resources
            if resource.resource_type == "network"
        ),
        f"{first.project_name}_default",
    )
    return ComposeStack(
        run_id=first.run_id,
        claim_id=first.claim_id,
        task_id=first.task_id,
        project_name=first.project_name,
        docker_executable=docker_executable,
        network_name=network,
        network_names=tuple(
            resource.resource_name
            for resource in resources
            if resource.resource_type == "network"
        )
        or (network,),
        image_names=tuple(
            resource.resource_name
            for resource in resources
            if resource.resource_type == "image"
        ),
        worktree=worktree,
        runtime_directory=runtime_directory,
        compose_files=files,
        profiles=profiles,
        services=(),
        environment={},
        resources=tuple(resources),
    )


def _validated_docker_executable(plan: HostPreflightPlan) -> Path:
    docker = next(
        (
            executable.path
            for executable in plan.executables
            if executable.name == "docker"
        ),
        None,
    )
    if docker is None or not docker.is_absolute():
        raise ValueError(
            "validated Compose services require an absolute Docker executable"
        )
    return docker


def _command_error(prefix: str, result: subprocess.CompletedProcess[str]) -> str:
    # Compose output may echo repository-controlled environment values.  The
    # durable error carries the actionable exit status, project, and argv but
    # deliberately does not persist raw stdout/stderr.
    if result.returncode == _TIMED_OUT_RETURN_CODE:
        return (
            f"{prefix} (timed out after "
            f"{_COMPOSE_COMMAND_TIMEOUT_SECONDS} seconds)"
        )
    return f"{prefix} (exit code {result.returncode})"
