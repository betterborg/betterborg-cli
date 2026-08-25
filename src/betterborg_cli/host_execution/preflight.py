"""Trust-gated validation of analyzer plans before host execution starts."""

from __future__ import annotations

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
    url: str | None = field(default=None, repr=False)


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
        required_secrets = self._required_secrets(
            plan, available_secret_names, failures
        )
        compose_files, services = self._services(
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
            required_secret_names=tuple(required_secrets),
            compose_files=tuple(compose_files),
            services=tuple(services),
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
            ),
            (
                "prepare",
                _mappings(environment.get("prepare_commands"))
                if isinstance(environment, Mapping)
                else [],
                "environment",
            ),
            (
                "materialize",
                _mappings(environment.get("materialize_commands"))
                if isinstance(environment, Mapping)
                else [],
                "environment",
            ),
        )

        validated_groups: list[list[HostCommand]] = []
        for group, records, default_stage in groups:
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
        requested: dict[tuple[str, str], tuple[str | None, str]] = {}
        for command in commands:
            executable_cwd = command.cwd if "/" in command.argv[0] else "."
            requested.setdefault(
                (command.argv[0], executable_cwd),
                (None, f"command stage {command.stage}"),
            )

        environment = plan.get("environment")
        toolchains: list[Mapping[str, Any]] = []
        if isinstance(environment, Mapping):
            for manager in environment.get("package_managers") or ():
                requested.setdefault(
                    (str(manager), "."),
                    (None, _evidence(environment, "environment")),
                )
            toolchains = _mappings(environment.get("toolchains"))
            for toolchain in toolchains:
                name = toolchain.get("name")
                if isinstance(name, str) and name:
                    requested[(name, ".")] = (
                        toolchain.get("version")
                        if isinstance(toolchain.get("version"), str)
                        else None,
                        _evidence(
                            toolchain, _evidence(environment, "analyzer toolchain")
                        ),
                    )

        resolved_tools: list[HostExecutable] = []
        for (name, cwd), (version, evidence) in requested.items():
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
            if not isinstance(name, str) or name not in by_name:
                continue
            version = toolchain.get("version")
            evidence = _evidence(toolchain, "analyzer toolchain")
            if not isinstance(version, str) or not version.strip():
                failures.append(
                    HostPreflightFailure(
                        requirement=(
                            f"toolchain {name!r} requires exact version evidence"
                        ),
                        evidence=evidence,
                        guidance=(
                            "Pin the runtime version in a repository file and "
                            "rerun analysis."
                        ),
                    )
                )
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
            output = self._version_output(by_name[name].path)
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
    ) -> list[str]:
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
        referenced = {
            name
            for record in command_records
            for name in record.get("required_secrets") or ()
            if isinstance(name, str)
        }
        for name in sorted(referenced - by_name.keys()):
            failures.append(
                HostPreflightFailure(
                    requirement=(
                        f"command references undeclared required secret: {name}"
                    ),
                    evidence="analyzer command catalog",
                    guidance=(
                        "Declare the secret by name in required_secrets and "
                        "rerun analysis."
                    ),
                )
            )

        available_names = set(available)
        for name, record in by_name.items():
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
        return sorted(by_name)

    def _services(
        self,
        plan: Mapping[str, Any],
        external_urls: Mapping[str, str],
        executables: list[HostExecutable],
        failures: list[HostPreflightFailure],
    ) -> tuple[list[Path], list[HostService]]:
        catalog = plan.get("command_catalog")
        command_records = (
            _mappings(catalog.get("commands")) if isinstance(catalog, Mapping) else []
        )
        selected = {
            name
            for record in command_records
            for name in record.get("uses_services") or ()
            if isinstance(name, str)
        }
        if not selected:
            return [], []

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
                        evidence="analyzer command uses_services",
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
            has_external = isinstance(url_env, str) and bool(url_env)
            if has_compose and has_external:
                failures.append(
                    HostPreflightFailure(
                        requirement=(
                            f"selected service is ambiguous or inferred: {name}"
                        ),
                        evidence=evidence,
                        guidance=(
                            "Set exactly one of compose_service or url_env in "
                            "analyzer evidence; preflight will not choose between "
                            "runtime sources."
                        ),
                    )
                )
            elif has_compose:
                assert isinstance(compose_service, str)
                compose_selected.append((compose_service, service))
                services.append(
                    HostService(
                        name=name,
                        kind="compose",
                        evidence=evidence,
                        compose_service=compose_service,
                    )
                )
            elif has_external:
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
        if compose_selected:
            compose_files = self._validate_compose(plan, compose_selected, failures)
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
            if docker is not None and self._compose_version_output(docker.path) is None:
                failures.append(
                    HostPreflightFailure(
                        requirement=(
                            "the Docker Compose plugin must be available on the host"
                        ),
                        evidence=", ".join(
                            _evidence(item, name) for name, item in compose_selected
                        ),
                        guidance=(
                            "Install or enable 'docker compose' and verify "
                            "'docker compose version' succeeds."
                        ),
                    )
                )
        return compose_files, services

    def _validate_compose(
        self,
        plan: Mapping[str, Any],
        selected: Sequence[tuple[str, Mapping[str, Any]]],
        failures: list[HostPreflightFailure],
    ) -> list[Path]:
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
            return []
        file_records = _mappings(compose.get("files"))
        primary_file = compose.get("file")
        declared_paths = [record.get("path") for record in file_records]
        if (
            isinstance(primary_file, str)
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
        paths: list[Path] = []
        for compose_service, service in selected:
            matches = [
                record
                for record in file_records
                if compose_service in (record.get("services") or ())
            ]
            if len(matches) != 1:
                failures.append(
                    HostPreflightFailure(
                        requirement=(
                            f"Compose service {compose_service!r} must appear in "
                            "exactly one compose.files service list"
                        ),
                        evidence=_evidence(
                            service, _evidence(compose, "analyzer Compose metadata")
                        ),
                        guidance=(
                            "Record the exact Compose file and its services in "
                            "analyzer metadata, then retry."
                        ),
                    )
                )
                continue
            value = matches[0].get("path")
            resolved = self._repository_path(value)
            if resolved is None or not resolved.is_file():
                failures.append(
                    HostPreflightFailure(
                        requirement=(
                            f"Compose file must exist inside the repository: {value!r}"
                        ),
                        evidence=_evidence(
                            matches[0], _evidence(compose, "analyzer Compose metadata")
                        ),
                        guidance=(
                            "Restore the Compose file or rerun analysis with "
                            "current metadata."
                        ),
                    )
                )
                continue
            paths.append(resolved)
        return _unique_paths(paths)

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
            candidate = (
                self.repository_root / PurePosixPath(cwd) / PurePosixPath(name)
            ).resolve()
            if (
                candidate.is_relative_to(self.repository_root)
                and candidate.is_file()
                and os.access(candidate, os.X_OK)
            ):
                return candidate
            return None
        found = self._find_executable(name, self._environment.get("PATH"))
        return Path(found).resolve() if found else None

    def _version_output(self, path: Path) -> str | None:
        try:
            result = self._run(
                [str(path), "--version"],
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


def _evidence(record: Mapping[str, Any], fallback: str) -> str:
    source = record.get("source")
    return source if isinstance(source, str) and source else fallback


def _unique_paths(paths: Sequence[Path]) -> list[Path]:
    return list(dict.fromkeys(paths))


def _contains_version(output: str, version: str) -> bool:
    normalized = version.removeprefix("v")
    return (
        re.search(rf"(?<![0-9A-Za-z])v?{re.escape(normalized)}(?![0-9A-Za-z])", output)
        is not None
    )


def _valid_external_url(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9+.-]*", parsed.scheme)) and bool(
        parsed.netloc
    )
