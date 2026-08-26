"""Trust-gated validation of analyzer plans before host execution starts."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import urlsplit

from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.workspace_trust import (
    TrustStore,
    UntrustedWorkspaceError,
    require_workspace_trust,
)

_MINIMUM_COMPOSE_VERSION = (2, 24, 4)


@dataclass(frozen=True, slots=True)
class HostPreflightFailure:
    """One exact, evidence-backed requirement that blocks host execution."""

    requirement: str
    evidence: str
    guidance: str


@dataclass(frozen=True, slots=True)
class HostPreflightBlock:
    """An actionable collection of preflight requirements that were not met."""

    failures: tuple[HostPreflightFailure, ...]

    @property
    def reason(self) -> str:
        return "\n".join(
            f"{failure.requirement} (evidence: {failure.evidence}). {failure.guidance}"
            for failure in self.failures
        )


@dataclass(frozen=True, slots=True)
class HostCommand:
    """One shell-free analyzer command with a validated working directory."""

    stage: str
    argv: tuple[str, ...]
    cwd: str
    evidence: str = "analyzer command catalog"


@dataclass(frozen=True, slots=True)
class HostExecutable:
    """One resolved host executable and any validated version requirement."""

    name: str
    path: Path
    version: str | None = None


@dataclass(frozen=True, slots=True)
class HostService:
    """One selected service whose runtime source is unambiguous."""

    name: str
    kind: Literal["compose", "external"]
    evidence: str
    compose_service: str | None = None
    url_env: str | None = None
    port: int | None = None
    url_targets: tuple[tuple[str, int, Literal["tcp", "udp"]], ...] = ()
    port_targets: tuple[tuple[str, int, Literal["tcp", "udp"]], ...] = ()
    url: str | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class HostSecret:
    """One analyzer-declared secret and its permitted execution scope."""

    name: str
    scope: Literal["all", "build", "agent"]
    used_by: tuple[str, ...]
    evidence: str


@dataclass(frozen=True, slots=True)
class HostPreflightPlan:
    """All repository-controlled host inputs validated before task claiming."""

    repository_root: Path
    commands: tuple[HostCommand, ...]
    prepare_commands: tuple[HostCommand, ...]
    materialize_commands: tuple[HostCommand, ...]
    environment_files: tuple[Path, ...]
    executables: tuple[HostExecutable, ...]
    required_secret_names: tuple[str, ...]
    compose_files: tuple[Path, ...]
    services: tuple[HostService, ...]
    compose_profiles: tuple[str, ...] = ()
    compose_networks: tuple[str, ...] = ()
    compose_volumes: tuple[str, ...] = ()
    package_managers: tuple[str, ...] = ()
    secret_requirements: tuple[HostSecret, ...] = ()


HostPreflightResult = HostPreflightPlan | HostPreflightBlock
AnalyzerPlanLoader = Callable[[], Mapping[str, Any]]


class HostPreflight:
    """Validate trusted analyzer evidence without preparing the host.

    Passing a callable to :meth:`validate` keeps repository context lazy: the
    callable is not invoked until the machine-local workspace trust gate has
    succeeded.
    """

    def __init__(
        self,
        repository_root: Path,
        *,
        trust_store: TrustStore | None = None,
        environment: Mapping[str, str] | None = None,
        executable_finder: Callable[[str, str | None], str | None] | None = None,
        command_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        self._paths = RepoPaths.discover(self.repository_root)
        self._trust_store = trust_store
        self._environment = dict(os.environ if environment is None else environment)
        self._find_executable = executable_finder or _which
        self._run = command_runner or subprocess.run

    def validate(
        self,
        analyzer_plan: Mapping[str, Any] | AnalyzerPlanLoader,
        *,
        available_secret_names: Collection[str] = (),
        external_urls: Mapping[str, str] | None = None,
    ) -> HostPreflightResult:
        """Return a complete plan or every actionable reason it is blocked."""
        try:
            require_workspace_trust(self._paths, store=self._trust_store)
        except (UntrustedWorkspaceError, ValueError, RuntimeError) as error:
            return HostPreflightBlock(
                (
                    HostPreflightFailure(
                        requirement="workspace trust is required before host preflight",
                        evidence=str(error),
                        guidance=(
                            "Run 'borg trust --yes' for this exact workspace, "
                            "then retry."
                        ),
                    ),
                )
            )

        plan = analyzer_plan() if callable(analyzer_plan) else analyzer_plan
        failures: list[HostPreflightFailure] = []
        commands, prepare_commands, materialize_commands = self._commands(
            plan, failures
        )
        environment_files = self._environment_files(plan, failures)
        executables = self._executables(
            plan,
            (*commands, *prepare_commands, *materialize_commands),
            failures,
        )
        secret_requirements = self._required_secrets(
            plan, available_secret_names, failures
        )
        (
            compose_files,
            compose_profiles,
            compose_networks,
            compose_volumes,
            services,
        ) = self._services(
            plan,
            external_urls or {},
            executables,
            failures,
        )

        if failures:
            return HostPreflightBlock(tuple(failures))
        return HostPreflightPlan(
            repository_root=self.repository_root,
            commands=tuple(commands),
            prepare_commands=tuple(prepare_commands),
            materialize_commands=tuple(materialize_commands),
            environment_files=tuple(environment_files),
            executables=tuple(executables),
            required_secret_names=tuple(
                secret.name for secret in secret_requirements
            ),
            compose_files=tuple(compose_files),
            services=tuple(services),
            compose_profiles=tuple(compose_profiles),
            compose_networks=tuple(compose_networks),
            compose_volumes=tuple(compose_volumes),
            package_managers=tuple(_package_managers(plan)),
            secret_requirements=tuple(secret_requirements),
        )

    def _commands(
        self,
        plan: Mapping[str, Any],
        failures: list[HostPreflightFailure],
    ) -> tuple[list[HostCommand], list[HostCommand], list[HostCommand]]:
        catalog = plan.get("command_catalog")
        environment = plan.get("environment")
        groups = (
            (
                "catalog",
                _mappings(catalog.get("commands"))
                if isinstance(catalog, Mapping)
                else [],
                "command",
                _evidence(catalog, "analyzer command catalog")
                if isinstance(catalog, Mapping)
                else "analyzer command catalog",
            ),
            (
                "prepare",
                _mappings(environment.get("prepare_commands"))
                if isinstance(environment, Mapping)
                else [],
                "environment",
                _evidence(environment, "analyzer environment")
                if isinstance(environment, Mapping)
                else "analyzer environment",
            ),
            (
                "materialize",
                _mappings(environment.get("materialize_commands"))
                if isinstance(environment, Mapping)
                else [],
                "environment",
                _evidence(environment, "analyzer environment")
                if isinstance(environment, Mapping)
                else "analyzer environment",
            ),
        )

        validated_groups: list[list[HostCommand]] = []
        for group, records, default_stage, group_evidence in groups:
            commands: list[HostCommand] = []
            for index, record in enumerate(records):
                argv = record.get("argv")
                if not _string_sequence(argv):
                    failures.append(
                        HostPreflightFailure(
                            requirement=(
                                f"{group} command {index + 1} must have a non-empty "
                                "argv"
                            ),
                            evidence=_evidence(record, "analyzer command catalog"),
                            guidance=(
                                "Correct the analyzer command metadata and rerun "
                                "analysis."
                            ),
                        )
                    )
                    continue
                cwd = record.get("cwd", ".")
                resolved = self._repository_path(cwd, require_directory=True)
                if resolved is None:
                    failures.append(
                        HostPreflightFailure(
                            requirement=(
                                f"{group} command cwd must be an existing "
                                f"repo-relative directory: {cwd!r}"
                            ),
                            evidence=_evidence(record, "analyzer command catalog"),
                            guidance=(
                                "Create the directory or correct the command cwd in "
                                "repository metadata."
                            ),
                        )
                    )
                    continue
                commands.append(
                    HostCommand(
                        stage=str(record.get("stage", default_stage)),
                        argv=tuple(argv),
                        cwd=(
                            resolved.relative_to(self.repository_root).as_posix()
                            or "."
                        ),
                        evidence=_evidence(record, group_evidence),
                    )
                )
            validated_groups.append(commands)
        catalog_commands, prepare_commands, materialize_commands = validated_groups
        return catalog_commands, prepare_commands, materialize_commands

    def _environment_files(
        self,
        plan: Mapping[str, Any],
        failures: list[HostPreflightFailure],
    ) -> list[Path]:
        environment = plan.get("environment")
        if not isinstance(environment, Mapping):
            return []
        paths: list[Path] = []
        for value in environment.get("files") or ():
            resolved = self._repository_path(value)
            if resolved is None or not resolved.is_file():
                failures.append(
                    HostPreflightFailure(
                        requirement=(
                            "referenced environment file must exist inside the "
                            f"repository: {value!r}"
                        ),
                        evidence=_evidence(environment, str(value)),
                        guidance=(
                            "Restore the referenced file or rerun analysis so its "
                            "environment evidence is current."
                        ),
                    )
                )
                continue
            paths.append(resolved)
        return _unique_paths(paths)

    def _executables(
        self,
        plan: Mapping[str, Any],
        commands: Sequence[HostCommand],
        failures: list[HostPreflightFailure],
    ) -> list[HostExecutable]:
        requested: dict[tuple[str, str], tuple[str | None, list[str]]] = {}

        def add_request(
            name: str, cwd: str, version: str | None, evidence: str
        ) -> None:
            current_version, evidence_values = requested.get(
                (name, cwd), (None, [])
            )
            requested[(name, cwd)] = (
                version if version is not None else current_version,
                _unique_strings((*evidence_values, evidence)),
            )

        for command in commands:
            executable_cwd = command.cwd if "/" in command.argv[0] else "."
            add_request(
                command.argv[0], executable_cwd, None, command.evidence
            )

        environment = plan.get("environment")
        toolchains: list[Mapping[str, Any]] = []
        if isinstance(environment, Mapping):
            for manager in environment.get("package_managers") or ():
                add_request(
                    str(manager),
                    ".",
                    None,
                    _evidence(environment, "environment"),
                )
            toolchains = _mappings(environment.get("toolchains"))
            for toolchain in toolchains:
                name = toolchain.get("name")
                if isinstance(name, str) and name:
                    add_request(
                        _toolchain_executable_name(name),
                        ".",
                        toolchain.get("version")
                        if isinstance(toolchain.get("version"), str)
                        else None,
                        _evidence(
                            toolchain, _evidence(environment, "analyzer toolchain")
                        ),
                    )

        resolved_tools: list[HostExecutable] = []
        for (name, cwd), (version, evidence_values) in requested.items():
            evidence = _join_evidence(evidence_values)
            path = self._resolve_executable(name, cwd=cwd)
            if path is None:
                failures.append(
                    HostPreflightFailure(
                        requirement=f"host executable is required: {name}",
                        evidence=evidence,
                        guidance=(
                            f"Install {name!r} on the host or update the analyzer "
                            "command/toolchain evidence; BetterBorg will not install "
                            "runtimes during preflight."
                        ),
                    )
                )
                continue
            resolved_tools.append(HostExecutable(name=name, path=path, version=version))

        by_name = {tool.name: tool for tool in resolved_tools}
        for toolchain in toolchains:
            name = toolchain.get("name")
            if not isinstance(name, str):
                continue
            executable_name = _toolchain_executable_name(name)
            if executable_name not in by_name:
                continue
            version = toolchain.get("version")
            evidence = _evidence(toolchain, "analyzer toolchain")
            if not isinstance(version, str) or not version.strip():
                continue
            cited_source = toolchain.get("source")
            if isinstance(cited_source, str) and cited_source:
                source_values = [cited_source]
            elif isinstance(environment, Mapping):
                source_values = [environment.get("source")]
                source_values.extend(environment.get("files") or ())
            else:
                source_values = []
            source_paths = [
                path
                for value in source_values
                if (path := self._source_path(value)) is not None and path.is_file()
            ]
            if not source_paths:
                failures.append(
                    HostPreflightFailure(
                        requirement=(
                            f"toolchain {name!r} version evidence file must exist "
                            "inside the repository"
                        ),
                        evidence=evidence,
                        guidance=(
                            "Restore the version manifest or correct the analyzer "
                            "source reference."
                        ),
                    )
                )
                continue
            version_is_cited = False
            for source_path in source_paths:
                try:
                    source_text = source_path.read_text(
                        encoding="utf-8", errors="replace"
                    )
                except OSError as error:
                    evidence = f"{evidence}: {error}"
                    continue
                version_is_cited = version_is_cited or version in source_text
            if not version_is_cited:
                failures.append(
                    HostPreflightFailure(
                        requirement=(
                            f"toolchain {name!r} version {version!r} must appear "
                            "in its evidence file"
                        ),
                        evidence=evidence,
                        guidance=(
                            "Update the repository version pin or rerun analysis "
                            "with current evidence."
                        ),
                    )
                )
                continue
            output = self._version_output(name, by_name[executable_name].path)
            if output is None or not _contains_version(output, version):
                observed = (
                    output.strip().splitlines()[0] if output else "no version output"
                )
                failures.append(
                    HostPreflightFailure(
                        requirement=(
                            f"host executable {name!r} must satisfy analyzer "
                            f"version {version!r}"
                        ),
                        evidence=f"{evidence}; observed: {observed}",
                        guidance=(
                            "Install the repository-declared "
                            f"{name} {version} runtime on the host, then retry."
                        ),
                    )
                )
        return resolved_tools

    def _required_secrets(
        self,
        plan: Mapping[str, Any],
        available: Collection[str],
        failures: list[HostPreflightFailure],
    ) -> list[HostSecret]:
        records = _mappings(plan.get("required_secrets"))
        by_name: dict[str, Mapping[str, Any]] = {}
        for record in records:
            name = record.get("name")
            if isinstance(name, str) and name:
                if name in by_name:
                    failures.append(
                        HostPreflightFailure(
                            requirement=(
                                f"required secret name must be unambiguous: {name}"
                            ),
                            evidence=_evidence(record, "analyzer required_secrets"),
                            guidance=(
                                "Deduplicate the analyzer secret requirements and "
                                "rerun analysis."
                            ),
                        )
                    )
                by_name[name] = record

        command_records = _mappings(
            plan.get("command_catalog", {}).get("commands")
            if isinstance(plan.get("command_catalog"), Mapping)
            else None
        )
        catalog = plan.get("command_catalog")
        catalog_evidence = (
            _evidence(catalog, "analyzer command catalog")
            if isinstance(catalog, Mapping)
            else "analyzer command catalog"
        )
        referenced: dict[str, list[str]] = {}
        for record in command_records:
            for name in record.get("required_secrets") or ():
                if isinstance(name, str):
                    referenced.setdefault(name, []).append(
                        _evidence(record, catalog_evidence)
                    )
        for name in sorted(referenced.keys() - by_name.keys()):
            failures.append(
                HostPreflightFailure(
                    requirement=(
                        f"command references undeclared required secret: {name}"
                    ),
                    evidence=_join_evidence(referenced[name]),
                    guidance=(
                        "Declare the secret by name in required_secrets and "
                        "rerun analysis."
                    ),
                )
            )

        available_names = set(available)
        validated: list[HostSecret] = []
        for name, record in by_name.items():
            scope = record.get("scope")
            used_by = record.get("used_by")
            valid_used_by = (
                isinstance(used_by, Sequence)
                and not isinstance(used_by, str | bytes)
                and bool(used_by)
                and all(isinstance(value, str) and value for value in used_by)
            )
            if scope not in {"all", "build", "agent"} or not valid_used_by:
                failures.append(
                    HostPreflightFailure(
                        requirement=(
                            f"required secret {name!r} must declare a valid scope "
                            "and non-empty used_by stages"
                        ),
                        evidence=_evidence(record, "analyzer required_secrets"),
                        guidance=(
                            "Set scope to all, build, or agent and identify the "
                            "commands that use this secret, then rerun analysis."
                        ),
                    )
                )
                continue
            if name not in available_names:
                failures.append(
                    HostPreflightFailure(
                        requirement=f"required secret is not configured: {name}",
                        evidence=_evidence(record, "analyzer required_secrets"),
                        guidance=(
                            f"Configure secret {name!r} in BetterBorg repository "
                            "secret storage, then retry preflight."
                        ),
                    )
                )
            assert isinstance(scope, str)
            assert isinstance(used_by, Sequence)
            validated.append(
                HostSecret(
                    name=name,
                    scope=scope,
                    used_by=tuple(used_by),
                    evidence=_evidence(record, "analyzer required_secrets"),
                )
            )
        return sorted(validated, key=lambda secret: secret.name)

    def _services(
        self,
        plan: Mapping[str, Any],
        external_urls: Mapping[str, str],
        executables: list[HostExecutable],
        failures: list[HostPreflightFailure],
    ) -> tuple[list[Path], list[str], list[str], list[str], list[HostService]]:
        catalog = plan.get("command_catalog")
        command_records = (
            _mappings(catalog.get("commands")) if isinstance(catalog, Mapping) else []
        )
        catalog_evidence = (
            _evidence(catalog, "analyzer command catalog")
            if isinstance(catalog, Mapping)
            else "analyzer command catalog"
        )
        selected: dict[str, list[str]] = {}
        for record in command_records:
            for name in record.get("uses_services") or ():
                if isinstance(name, str):
                    selected.setdefault(name, []).append(
                        _evidence(record, catalog_evidence)
                    )
        if not selected:
            return [], [], [], [], []

        by_name: dict[str, list[Mapping[str, Any]]] = {}
        for record in _mappings(plan.get("service_dependencies")):
            name = record.get("name")
            if isinstance(name, str):
                by_name.setdefault(name, []).append(record)

        compose_selected: list[tuple[str, Mapping[str, Any]]] = []
        services: list[HostService] = []
        for name in sorted(selected):
            matches = by_name.get(name, [])
            if len(matches) != 1:
                failures.append(
                    HostPreflightFailure(
                        requirement=(
                            "selected service must resolve to exactly one "
                            f"analyzer dependency: {name}"
                        ),
                        evidence=_join_evidence(
                            (
                                *selected[name],
                                *(
                                    _evidence(item, "analyzer service dependency")
                                    for item in matches
                                ),
                            )
                        ),
                        guidance=(
                            "Declare one evidence-backed service dependency with "
                            "this exact name and rerun analysis."
                        ),
                    )
                )
                continue
            service = matches[0]
            evidence = _evidence(service, "analyzer service dependency")
            compose_service = service.get("compose_service")
            url_env = service.get("url_env")
            has_compose = isinstance(compose_service, str) and bool(compose_service)
            has_url_env = isinstance(url_env, str) and bool(url_env)
            if has_compose:
                assert isinstance(compose_service, str)
                url_targets = _compose_service_url_targets(service)
                port_targets = _compose_service_port_targets(service)
                if has_url_env and not any(
                    target_env == url_env
                    for target_env, _port, _protocol in url_targets
                ):
                    failures.append(
                        HostPreflightFailure(
                            requirement=(
                                f"Compose service {name!r} URL environment "
                                f"{url_env} requires an exact port"
                            ),
                            evidence=evidence,
                            guidance=(
                                "Record the service port alongside url_env in "
                                "analyzer evidence and rerun analysis."
                            ),
                        )
                    )
                compose_selected.append((compose_service, service))
                services.append(
                    HostService(
                        name=name,
                        kind="compose",
                        evidence=evidence,
                        compose_service=compose_service,
                        url_env=url_env if has_url_env else None,
                        port=(
                            next(
                                (
                                    port
                                    for target_env, port, _protocol in url_targets
                                    if target_env == url_env
                                ),
                                None,
                            )
                            if has_url_env
                            else None
                        ),
                        url_targets=url_targets,
                        port_targets=port_targets,
                    )
                )
            elif has_url_env:
                assert isinstance(url_env, str)
                url = external_urls.get(url_env) or self._environment.get(url_env)
                if not _valid_external_url(url):
                    failures.append(
                        HostPreflightFailure(
                            requirement=(
                                f"external service {name!r} requires an absolute "
                                f"service URL in {url_env}"
                            ),
                            evidence=evidence,
                            guidance=(
                                f"Supply {url_env} for this run or configure an "
                                "explicit Compose service instead."
                            ),
                        )
                    )
                    continue
                services.append(
                    HostService(
                        name=name,
                        kind="external",
                        evidence=evidence,
                        url_env=url_env,
                        url=url,
                    )
                )
            else:
                failures.append(
                    HostPreflightFailure(
                        requirement=(
                            f"selected service is ambiguous or inferred: {name}"
                        ),
                        evidence=evidence,
                        guidance=(
                            "Set an exact compose_service or url_env in analyzer "
                            "evidence; preflight will not infer a runtime service."
                        ),
                    )
                )

        compose_files: list[Path] = []
        compose_profiles: list[str] = []
        compose_networks: list[str] = []
        compose_volumes: list[str] = []
        if compose_selected:
            compose_files, compose_profiles = self._validate_compose(
                plan, compose_selected, failures
            )
            docker = next((tool for tool in executables if tool.name == "docker"), None)
            if docker is None:
                docker_path = self._resolve_executable("docker")
                if docker_path is None:
                    failures.append(
                        HostPreflightFailure(
                            requirement=(
                                "Docker with the Compose plugin is required for "
                                "selected services"
                            ),
                            evidence=", ".join(
                                _evidence(item, name) for name, item in compose_selected
                            ),
                            guidance=(
                                "Install Docker and its Compose plugin on the host, "
                                "then retry; BetterBorg will not install them."
                            ),
                        )
                    )
                else:
                    docker = HostExecutable("docker", docker_path)
                    executables.append(docker)
            if docker is not None:
                compose_version = self._compose_version_output(docker.path)
                if compose_version is None:
                    failures.append(
                        HostPreflightFailure(
                            requirement=(
                                "the Docker Compose plugin must be available on the "
                                "host"
                            ),
                            evidence=", ".join(
                                _evidence(item, name)
                                for name, item in compose_selected
                            ),
                            guidance=(
                                "Install or enable 'docker compose' and verify "
                                "'docker compose version' succeeds."
                            ),
                        )
                    )
                elif not _supported_compose_version(compose_version):
                    minimum = ".".join(map(str, _MINIMUM_COMPOSE_VERSION))
                    failures.append(
                        HostPreflightFailure(
                            requirement=(
                                f"Docker Compose {minimum} or newer is required for "
                                "selected services"
                            ),
                            evidence=(
                                ", ".join(
                                    _evidence(item, name)
                                    for name, item in compose_selected
                                )
                                + f"; docker compose version reported: "
                                f"{compose_version}"
                            ),
                            guidance=(
                                "Upgrade the Docker Compose plugin, verify "
                                f"'docker compose version' reports {minimum} or "
                                "newer, then retry."
                            ),
                        )
                    )
                elif compose_files:
                    compose_networks, compose_volumes = self._compose_topology(
                        docker.path,
                        compose_files,
                        compose_profiles,
                        compose_selected,
                        failures,
                    )
        return (
            compose_files,
            compose_profiles,
            compose_networks,
            compose_volumes,
            services,
        )

    def _compose_topology(
        self,
        docker: Path,
        files: Sequence[Path],
        profiles: Sequence[str],
        selected: Sequence[tuple[str, Mapping[str, Any]]],
        failures: list[HostPreflightFailure],
    ) -> tuple[list[str], list[str]]:
        """Validate isolation once and retain only secret-free topology keys."""
        command = [
            str(docker),
            "compose",
            "--project-name",
            "betterborg-preflight",
        ]
        for path in files:
            command.extend(("--file", str(path)))
        for profile in profiles:
            command.extend(("--profile", profile))
        command.extend(("config", "--format", "json"))
        evidence = _join_evidence(
            _evidence(service, name) for name, service in selected
        )
        try:
            result = self._run(
                command,
                cwd=self.repository_root,
                env=self._probe_environment(),
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            result = None
        if result is None or result.returncode != 0:
            failures.append(
                HostPreflightFailure(
                    requirement=(
                        "Compose configuration must resolve before task claims"
                    ),
                    evidence=evidence,
                    guidance=(
                        "Run the validated Compose file stack with the selected "
                        "profiles, correct its configuration, and rerun preflight."
                    ),
                )
            )
            return [], []

        try:
            model = json.loads(result.stdout)
            service_records = model["services"]
            network_records = model.get("networks", {})
            volume_records = model.get("volumes", {})
            if not isinstance(service_records, Mapping):
                raise TypeError
            selected_records = {
                name: service_records[name]
                for name, _service in selected
                if isinstance(service_records.get(name), Mapping)
            }
            if len(selected_records) != len({name for name, _service in selected}):
                raise KeyError("selected service topology is incomplete")
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
            networks = [str(key) for key in network_records] or ["default"]
            volumes = [str(key) for key in volume_records]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            detail = str(error) or "selected service topology is incomplete"
            failures.append(
                HostPreflightFailure(
                    requirement=(
                        "selected Compose service topology must be isolated: "
                        f"{detail}"
                    ),
                    evidence=evidence,
                    guidance=(
                        "Remove host/service network_mode and writable bind mounts, "
                        "ensure every selected service resolves, then rerun preflight."
                    ),
                )
            )
            return [], []
        return networks, volumes

    def _validate_compose(
        self,
        plan: Mapping[str, Any],
        selected: Sequence[tuple[str, Mapping[str, Any]]],
        failures: list[HostPreflightFailure],
    ) -> tuple[list[Path], list[str]]:
        compose = plan.get("compose")
        if not isinstance(compose, Mapping):
            failures.append(
                HostPreflightFailure(
                    requirement=(
                        "Compose metadata is required for selected Compose services"
                    ),
                    evidence=", ".join(
                        _evidence(item, name) for name, item in selected
                    ),
                    guidance=(
                        "Rerun analysis with the exact Compose file and service "
                        "metadata."
                    ),
                )
            )
            return [], []
        file_records = _mappings(compose.get("files"))
        primary_file = compose.get("file")
        declared_paths = [record.get("path") for record in file_records]
        if (
            file_records
            and isinstance(primary_file, str)
            and primary_file
            and primary_file not in declared_paths
        ):
            failures.append(
                HostPreflightFailure(
                    requirement=(
                        "primary Compose file must match one exact compose.files "
                        f"entry: {primary_file!r}"
                    ),
                    evidence=_evidence(compose, "analyzer Compose metadata"),
                    guidance=(
                        "Remove conflicting Compose metadata or record the exact "
                        "primary file and rerun analysis."
                    ),
                )
            )
        if not file_records and isinstance(primary_file, str) and primary_file:
            file_records = [
                {
                    "path": primary_file,
                    "source": _evidence(compose, primary_file),
                }
            ]
        if not file_records:
            failures.append(
                HostPreflightFailure(
                    requirement=(
                        "Compose metadata must declare at least one exact file for "
                        "selected Compose services"
                    ),
                    evidence=_evidence(compose, "analyzer Compose metadata"),
                    guidance=(
                        "Record compose.file or the ordered compose.files stack and "
                        "rerun analysis."
                    ),
                )
            )
            return [], _compose_profiles(compose, failures)

        paths: list[Path] = []
        for record in file_records:
            value = record.get("path")
            resolved = self._repository_path(value)
            if resolved is None or not resolved.is_file():
                failures.append(
                    HostPreflightFailure(
                        requirement=(
                            "Compose file must exist inside the repository: "
                            f"{value!r}"
                        ),
                        evidence=_evidence(
                            record, _evidence(compose, "analyzer Compose metadata")
                        ),
                        guidance=(
                            "Restore the Compose file or rerun analysis with "
                            "current metadata."
                        ),
                    )
                )
                continue
            paths.append(resolved)

        service_lists = [
            record.get("services") for record in file_records if "services" in record
        ]
        for compose_service, service in selected:
            if len(service_lists) == len(file_records) and not any(
                compose_service in services
                for services in service_lists
                if isinstance(services, Sequence)
                and not isinstance(services, str | bytes)
            ):
                failures.append(
                    HostPreflightFailure(
                        requirement=(
                            f"Compose service {compose_service!r} must appear in at "
                            "least one declared compose.files service list"
                        ),
                        evidence=_evidence(
                            service, _evidence(compose, "analyzer Compose metadata")
                        ),
                        guidance=(
                            "Correct the selected service or its Compose service "
                            "metadata, then rerun analysis."
                        ),
                    )
                )
        return paths, _compose_profiles(compose, failures)

    def _repository_path(
        self, value: object, *, require_directory: bool = False
    ) -> Path | None:
        if not isinstance(value, str) or not value:
            return None
        portable = PurePosixPath(value)
        if portable.is_absolute() or ".." in portable.parts or "\\" in value:
            return None
        resolved = (self.repository_root / portable).resolve()
        if not resolved.is_relative_to(self.repository_root):
            return None
        if require_directory and not resolved.is_dir():
            return None
        return resolved

    def _source_path(self, value: object) -> Path | None:
        if not isinstance(value, str):
            return None
        return self._repository_path(value.partition("#")[0])

    def _resolve_executable(self, name: str, *, cwd: str = ".") -> Path | None:
        if "/" in name or "\\" in name:
            candidate = self.repository_root / PurePosixPath(cwd) / PurePosixPath(name)
            resolved = candidate.resolve()
            if (
                resolved.is_relative_to(self.repository_root)
                and resolved.is_file()
                and os.access(resolved, os.X_OK)
            ):
                return Path(os.path.abspath(candidate))
            return None
        found = self._find_executable(name, self._environment.get("PATH"))
        return Path(os.path.abspath(found)) if found else None

    def _version_output(self, name: str, path: Path) -> str | None:
        if name == "go":
            arguments = ("version",)
        elif name == "java":
            arguments = ("-version",)
        else:
            arguments = ("--version",)
        try:
            result = self._run(
                [str(path), *arguments],
                cwd=self.repository_root,
                env=self._probe_environment(),
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        output = f"{result.stdout}\n{result.stderr}".strip()
        return output if result.returncode == 0 and output else None

    def _compose_version_output(self, docker: Path) -> str | None:
        try:
            result = self._run(
                [str(docker), "compose", "version"],
                cwd=self.repository_root,
                env=self._probe_environment(),
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        output = f"{result.stdout}\n{result.stderr}".strip()
        return output if result.returncode == 0 and output else None

    def _probe_environment(self) -> dict[str, str]:
        """Keep arbitrary host secrets out of repository-controlled probes."""
        allowed = ("PATH", "PATHEXT", "SYSTEMROOT", "LANG", "LC_ALL")
        return {
            name: self._environment[name]
            for name in allowed
            if name in self._environment
        }


def _which(name: str, path: str | None) -> str | None:
    return shutil.which(name, path=path)


def _toolchain_executable_name(name: str) -> str:
    return {"rust": "rustc"}.get(name, name)


def _package_managers(plan: Mapping[str, Any]) -> list[str]:
    environment = plan.get("environment")
    if not isinstance(environment, Mapping):
        return []
    values = environment.get("package_managers")
    if not isinstance(values, Sequence) or isinstance(values, str | bytes):
        return []
    return _unique_strings(
        [value for value in values if isinstance(value, str) and value]
    )


def _mappings(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _string_sequence(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str | bytes)
        and bool(value)
        and all(isinstance(item, str) and item for item in value)
    )


def _compose_service_url_targets(
    service: Mapping[str, Any],
) -> tuple[tuple[str, int, Literal["tcp", "udp"]], ...]:
    url_env = service.get("url_env")
    port = service.get("port")
    protocol: Literal["tcp", "udp"] = "tcp"
    port_records = _mappings(service.get("ports"))
    matching_port = next(
        (
            record
            for record in port_records
            if _service_port(port) and record.get("port") == port
        ),
        None,
    )
    if matching_port is not None:
        protocol = _service_protocol(matching_port.get("protocol"))
    if isinstance(url_env, str) and _service_port(port):
        return ((url_env, port, protocol),)
    return ()


def _compose_service_port_targets(
    service: Mapping[str, Any],
) -> tuple[tuple[str, int, Literal["tcp", "udp"]], ...]:
    targets: list[tuple[str, int, Literal["tcp", "udp"]]] = []
    url_env = service.get("url_env")
    port_records = _mappings(service.get("ports"))
    for record in port_records:
        port_env = record.get("env")
        port_value = record.get("port")
        if (
            isinstance(port_env, str)
            and port_env
            and port_env != url_env
            and _service_port(port_value)
        ):
            targets.append(
                (
                    port_env,
                    port_value,
                    _service_protocol(record.get("protocol")),
                )
            )

    seen: set[str] = set()
    deduped: list[tuple[str, int, Literal["tcp", "udp"]]] = []
    for target in targets:
        if target[0] in seen:
            continue
        seen.add(target[0])
        deduped.append(target)
    return tuple(deduped)


def _service_port(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 < value <= 65535


def _service_protocol(value: object) -> Literal["tcp", "udp"]:
    return "udp" if value == "udp" else "tcp"


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


def _evidence(record: Mapping[str, Any], fallback: str) -> str:
    source = record.get("source")
    return source if isinstance(source, str) and source else fallback


def _unique_paths(paths: Sequence[Path]) -> list[Path]:
    return list(dict.fromkeys(paths))


def _unique_strings(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _join_evidence(values: Sequence[str]) -> str:
    return ", ".join(_unique_strings(values))


def _compose_profiles(
    compose: Mapping[str, Any], failures: list[HostPreflightFailure]
) -> list[str]:
    value = compose.get("profiles")
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or not all(
        isinstance(item, str) and item for item in value
    ):
        failures.append(
            HostPreflightFailure(
                requirement="Compose profiles must be non-empty names",
                evidence=_evidence(compose, "analyzer Compose metadata"),
                guidance="Correct the Compose profiles and rerun analysis.",
            )
        )
        return []
    return list(value)


def _contains_version(output: str, version: str) -> bool:
    normalized = version.removeprefix("v")
    return (
        re.search(
            rf"(?<![0-9A-Za-z])(?:v|go)?{re.escape(normalized)}"
            r"(?![0-9A-Za-z])",
            output,
        )
        is not None
    )


def _supported_compose_version(output: str) -> bool:
    match = re.search(
        r"(?<![0-9A-Za-z])v?(\d+)\.(\d+)\.(\d+)(?!\d)",
        output,
    )
    if match is None:
        return False
    version = tuple(int(component) for component in match.groups())
    return version >= _MINIMUM_COMPOSE_VERSION


def _valid_external_url(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    if any(character.isspace() or not character.isprintable() for character in value):
        return False
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return False
    authority = parsed.netloc.rsplit("@", 1)[-1]
    return (
        bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9+.-]*", parsed.scheme))
        and bool(parsed.netloc)
        and bool(hostname)
        and "\\" not in parsed.netloc
        and (port is not None or not authority.endswith(":"))
    )
