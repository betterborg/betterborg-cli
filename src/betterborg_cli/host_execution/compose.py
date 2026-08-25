"""Claim-owned Docker Compose lifecycle for validated host services."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from uuid import UUID

from betterborg_cli.host_execution.preflight import HostPreflightPlan, HostService
from betterborg_cli.store import (
    ComposeResource,
    ExecutionEvent,
    SqliteStore,
    TaskClaim,
    TaskRuntimeStatus,
)
from betterborg_cli.store.models import utcnow

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
Clock = Callable[[], datetime]


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
    network_name: str
    network_names: tuple[str, ...]
    worktree: Path
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
        self._environment = dict(os.environ if environment is None else environment)
        self._run = command_runner or subprocess.run
        self._clock = clock

    def start_claimed_stack(
        self,
        store: SqliteStore,
        plan: HostPreflightPlan,
        claim: TaskClaim,
        owner_token: str,
    ) -> ComposeStack | None:
        """Start selected Compose services and return their consumer contract."""
        compose_services = tuple(
            service for service in plan.services if service.kind == "compose"
        )
        if not compose_services:
            return None
        if not plan.compose_files:
            raise ValueError("validated Compose services require Compose files")

        worktree = self._claimed_worktree(store, claim)
        files = self._worktree_compose_files(plan, worktree)
        project_name = compose_project_name(claim)
        selected = tuple(
            dict.fromkeys(
                service.compose_service
                for service in compose_services
                if service.compose_service is not None
            )
        )
        if len(selected) != len(compose_services):
            raise ValueError("validated Compose services require exact service names")

        try:
            override, network_names = self._write_runtime_override(
                project_name,
                files,
                plan.compose_profiles,
                compose_services,
                selected,
                worktree,
            )
        except ComposeStackError as error:
            self._block_task(store, claim, owner_token, str(error))
            raise
        files = (*files, override)
        network_name = network_names[0]
        labels = {
            "betterborg.run_id": str(claim.run_id),
            "betterborg.claim_id": str(claim.id),
            "betterborg.task_id": str(claim.task_id),
            "betterborg.compose.files": json.dumps([str(path) for path in files]),
            "betterborg.compose.profiles": json.dumps(list(plan.compose_profiles)),
            "betterborg.compose.cwd": str(worktree),
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
        )
        command = _compose_base(project_name, files, plan.compose_profiles) + (
            "up",
            "--detach",
            "--wait",
            "--remove-orphans",
            "--no-deps",
            *selected,
        )
        for resource in resources:
            store.add_compose_resource(
                resource,
                owner_token,
                claim.claim_token,
                now=created_at,
            )
        self._append_owned_event(
            store,
            claim,
            owner_token,
            "compose.starting",
            project_name,
            command,
            now=created_at,
        )

        stack = ComposeStack(
            run_id=claim.run_id,
            claim_id=claim.id,
            task_id=claim.task_id,
            project_name=project_name,
            network_name=network_name,
            network_names=network_names,
            worktree=worktree,
            compose_files=files,
            profiles=plan.compose_profiles,
            services=selected,
            environment={},
            resources=resources,
        )
        result = self._invoke(command, cwd=worktree)
        if result.returncode != 0:
            error = _command_error("Compose services did not become healthy", result)
            self._block_task(store, claim, owner_token, error)
            cleanup = self._stop_project(store, stack, command_owner=None)
            if cleanup.error is not None:
                error = f"{error}; cleanup also failed: {cleanup.error}"
            raise ComposeStackError(
                error,
                project_name=project_name,
                command=command,
            )

        try:
            published_ports = self._published_ports(
                project_name,
                files,
                plan.compose_profiles,
                compose_services,
                worktree,
            )
        except ComposeStackError as error:
            self._block_task(store, claim, owner_token, str(error))
            cleanup = self._stop_project(store, stack, command_owner=None)
            message = str(error)
            if cleanup.error is not None:
                message = f"{message}; cleanup also failed: {cleanup.error}"
            raise ComposeStackError(
                message,
                project_name=error.project_name,
                command=error.command,
            ) from error
        environment = service_url_environment(
            plan.services,
            published_ports=published_ports,
        )
        stack = replace(stack, environment=environment)

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
                "url_env_names": sorted(environment),
            },
        )
        return stack

    def _write_runtime_override(
        self,
        project_name: str,
        files: Sequence[Path],
        profiles: Sequence[str],
        compose_services: Sequence[HostService],
        selected: Sequence[str],
        worktree: Path,
    ) -> tuple[Path, tuple[str, ...]]:
        """Resolve topology and write a claim-local isolation override."""
        config_command = _compose_base(project_name, files, profiles) + (
            "config",
            "--format",
            "json",
        )
        result = self._invoke(config_command, cwd=worktree)
        if result.returncode != 0:
            raise ComposeStackError(
                _command_error("Compose configuration could not be resolved", result),
                project_name=project_name,
                command=config_command,
            )
        try:
            model = json.loads(result.stdout)
            service_records = model["services"]
            network_records = model.get("networks", {})
            volume_records = model.get("volumes", {})
            if not isinstance(service_records, Mapping):
                raise TypeError
            selected_records = {
                name: service_records[name]
                for name in selected
                if isinstance(service_records.get(name), Mapping)
            }
            if len(selected_records) != len(selected):
                raise KeyError
            if any(record.get("network_mode") for record in selected_records.values()):
                raise ValueError("host or service network_mode cannot be isolated")
            writable_binds = _writable_bind_mounts(selected_records)
            if writable_binds:
                raise ValueError(
                    "writable bind mounts cannot be isolated: "
                    + ", ".join(writable_binds)
                )
            if not isinstance(network_records, Mapping) or not isinstance(
                volume_records, Mapping
            ):
                raise TypeError
            # Compose creates every declared network even when --no-deps limits
            # container startup, so every effective network must be claim-owned.
            network_keys = tuple(str(key) for key in network_records) or ("default",)
            volume_keys = tuple(str(key) for key in volume_records)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            detail = str(error) or "selected service topology is incomplete"
            raise ComposeStackError(
                f"Compose configuration cannot be isolated: {detail}",
                project_name=project_name,
                command=config_command,
            ) from error

        network_names = tuple(
            _owned_resource_name(project_name, key) for key in network_keys
        )
        owned_networks = dict(zip(network_keys, network_names, strict=True))
        owned_volumes = {
            key: _owned_resource_name(project_name, f"volume-{key}")
            for key in volume_keys
        }
        ports = _service_target_ports(compose_services)
        override_text = _render_runtime_override(
            selected,
            ports,
            owned_networks,
            owned_volumes,
        )
        override_directory = self.repository_root / ".borg/state/compose" / project_name
        override = override_directory / "compose.override.yml"
        try:
            override_directory.mkdir(parents=True, exist_ok=True)
            override.write_text(override_text, encoding="utf-8")
        except OSError as error:
            raise ComposeStackError(
                f"Compose isolation override could not be written: {error}",
                project_name=project_name,
                command=config_command,
            ) from error
        return override, network_names

    def _published_ports(
        self,
        project_name: str,
        files: Sequence[Path],
        profiles: Sequence[str],
        compose_services: Sequence[HostService],
        worktree: Path,
    ) -> dict[tuple[str, int], int]:
        published: dict[tuple[str, int], int] = {}
        for service_name, target in _service_target_ports(compose_services):
            command = _compose_base(project_name, files, profiles) + (
                "port",
                service_name,
                str(target),
            )
            result = self._invoke(command, cwd=worktree)
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
            published[(service_name, target)] = host_port
        return published

    def stop_claimed_stack(
        self,
        store: SqliteStore,
        stack: ComposeStack,
        claim: TaskClaim,
        owner_token: str,
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
            outcomes.append(self._stop_project(store, stack, command_owner=None))
        return tuple(outcomes)

    def _stop_project(
        self,
        store: SqliteStore,
        stack: ComposeStack,
        *,
        command_owner: tuple[TaskClaim, str] | None,
    ) -> ComposeCleanupResult:
        command = _compose_base(
            stack.project_name, stack.compose_files, stack.profiles
        ) + ("down", "--volumes", "--remove-orphans", "--rmi", "local")
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

        result = self._invoke(command, cwd=stack.worktree)
        if result.returncode != 0:
            error = _command_error("Compose teardown failed", result)
            store.record_compose_cleanup_failure(
                stack.run_id,
                stack.task_id,
                stack.project_name,
                command=command,
                error=error,
                now=self._clock(),
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

    def _invoke(
        self, command: tuple[str, ...], *, cwd: Path
    ) -> subprocess.CompletedProcess[str]:
        try:
            return self._run(
                list(command),
                cwd=cwd,
                env=self._environment,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as error:
            return subprocess.CompletedProcess(command, 127, "", str(error))

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
    published_ports: Mapping[tuple[str, int], int] | None = None,
) -> dict[str, str]:
    """Return only URL variables declared by the validated service plan."""
    environment: dict[str, str] = {}
    for service in services:
        if service.kind == "compose" and service.compose_service is not None:
            for env_name, port in service.url_targets:
                published = (published_ports or {}).get(
                    (service.compose_service, port)
                )
                if published is None:
                    continue
                environment[env_name] = _service_url(
                    env_name,
                    "127.0.0.1",
                    published,
                    target_port=port,
                )
        if service.url_env is None:
            continue
        if service.kind == "external" and service.url is not None:
            environment[service.url_env] = service.url
        elif service.kind == "compose" and service.compose_service is not None:
            if service.url is not None:
                environment[service.url_env] = service.url
            elif service.port is not None and 0 < service.port <= 65535:
                published = (published_ports or {}).get(
                    (service.compose_service, service.port)
                )
                if published is None:
                    continue
                environment[service.url_env] = _service_url(
                    service.url_env,
                    "127.0.0.1",
                    published,
                    target_port=service.port,
                )
    return environment


def _service_url(
    env_name: str,
    service_name: str,
    port: int,
    *,
    target_port: int | None = None,
) -> str:
    key = f"{env_name} {service_name}".casefold()
    if "redis" in key:
        return f"redis://{service_name}:{port}/0"
    if "postgres" in key or "database" in key:
        return f"postgres://{service_name}:{port}/postgres"
    if "mysql" in key or "mariadb" in key:
        return f"mysql://{service_name}:{port}/mysql"
    if "mongo" in key:
        return f"mongodb://{service_name}:{port}"
    if "http" in key or target_port in {80, 3000, 8000, 8080}:
        return f"http://{service_name}:{port}"
    return f"tcp://{service_name}:{port}"


def _service_target_ports(
    services: Sequence[HostService],
) -> tuple[tuple[str, int], ...]:
    targets: list[tuple[str, int]] = []
    for service in services:
        if service.compose_service is None:
            continue
        targets.extend(
            (service.compose_service, port) for _env_name, port in service.url_targets
        )
        if service.port is not None:
            targets.append((service.compose_service, service.port))
    return tuple(dict.fromkeys(targets))


def _owned_resource_name(project_name: str, key: str) -> str:
    return f"{project_name}_{key}"


def _writable_bind_mounts(
    services: Mapping[str, Mapping[str, object]],
) -> tuple[str, ...]:
    violations: list[str] = []
    for service_name, service in services.items():
        volumes = service.get("volumes")
        if not isinstance(volumes, Sequence) or isinstance(volumes, str | bytes):
            continue
        for index, volume in enumerate(volumes):
            if not isinstance(volume, Mapping):
                continue
            volume_type = volume.get("type")
            if (
                isinstance(volume_type, str)
                and volume_type.casefold() == "bind"
                and not _volume_read_only(volume)
            ):
                violations.append(f"{service_name}.volumes[{index}]")
    return tuple(violations)


def _volume_read_only(volume: Mapping[object, object]) -> bool:
    if volume.get("read_only") is True or volume.get("readonly") is True:
        return True
    mode = volume.get("mode")
    if not isinstance(mode, str):
        return False
    modes = {part.strip().casefold() for part in mode.split(",")}
    return bool({"ro", "readonly"} & modes)


def _render_runtime_override(
    services: Sequence[str],
    ports: Sequence[tuple[str, int]],
    networks: Mapping[str, str],
    volumes: Mapping[str, str],
) -> str:
    by_service: dict[str, list[int]] = {name: [] for name in services}
    for service, target in ports:
        by_service[service].append(target)
    lines = ["services:"]
    for service in services:
        lines.extend(
            (
                f"  {json.dumps(service)}:",
                "    container_name: !reset null",
            )
        )
        targets = tuple(dict.fromkeys(by_service[service]))
        if not targets:
            lines.append("    ports: !override []")
            continue
        lines.append("    ports: !override")
        for target in targets:
            lines.extend(
                (
                    f"      - target: {target}",
                    '        published: "0"',
                    '        host_ip: "0.0.0.0"',
                    '        protocol: "tcp"',
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


def _parse_published_port(output: str) -> int | None:
    address = output.strip().splitlines()[0] if output.strip() else ""
    _separator, _colon, port_text = address.rpartition(":")
    try:
        port = int(port_text)
    except ValueError:
        return None
    return port if 0 < port <= 65535 else None


def _compose_base(
    project_name: str,
    compose_files: Sequence[Path],
    profiles: Sequence[str],
) -> tuple[str, ...]:
    command = ["docker", "compose", "--project-name", project_name]
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
        worktree = Path(labels["betterborg.compose.cwd"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        # Schema-v6 resources predate the runtime metadata labels.  Their exact
        # project identity remains sufficient for a project-scoped teardown;
        # Compose performs its normal file discovery from the trusted root.
        files = ()
        profiles = ()
        worktree = fallback_worktree
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
        network_name=network,
        network_names=tuple(
            resource.resource_name
            for resource in resources
            if resource.resource_type == "network"
        )
        or (network,),
        worktree=worktree,
        compose_files=files,
        profiles=profiles,
        services=(),
        environment={},
        resources=tuple(resources),
    )


def _command_error(prefix: str, result: subprocess.CompletedProcess[str]) -> str:
    # Compose output may echo repository-controlled environment values.  The
    # durable error carries the actionable exit status, project, and argv but
    # deliberately does not persist raw stdout/stderr.
    return f"{prefix} (exit code {result.returncode})"
