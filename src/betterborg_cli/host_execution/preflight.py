"""Trust-gated validation of analyzer plans before host execution starts."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import urlsplit

from betterborg_cli.agent_runtime.base import CancellationToken
from betterborg_cli.agent_runtime.process import run_captured
from betterborg_cli.progress import AgentActivity, AgentActivityKind
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
class HostDroppedCommand:
    """One catalog command excluded because the host cannot invoke it."""

    command: HostCommand
    reason: str


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
    compose_build_services: tuple[str, ...] = ()
    package_managers: tuple[str, ...] = ()
    secret_requirements: tuple[HostSecret, ...] = ()
    dropped_commands: tuple[HostDroppedCommand, ...] = ()

    @property
    def dropped_command_summary(self) -> str:
        """Name every catalog command this host could not be given.

        A check that does not run cannot fail, so a run that skipped one has
        to be tellable apart from a run that passed it. Every surface that
        reports a validated plan says this, and says nothing when the host
        could run the whole catalog.
        """
        if not self.dropped_commands:
            return ""
        count = len(self.dropped_commands)
        return (
            f"{count} sanity command{'' if count == 1 else 's'} dropped: "
            + "; ".join(
                f"{shlex.join(dropped.command.argv)}: {dropped.reason}"
                for dropped in self.dropped_commands
            )
        )


HostPreflightResult = HostPreflightPlan | HostPreflightBlock
AnalyzerPlanLoader = Callable[[], Mapping[str, Any]]
ActivitySink = Callable[[AgentActivity], None]


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
        cancel: CancellationToken | None = None,
        command_runner: Callable[
            ..., subprocess.CompletedProcess[str]
        ] = run_captured,
        activity: ActivitySink | None = None,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        self._cancel = cancel
        self._run = command_runner
        self._activity = activity
        self._validated_result: HostPreflightResult | None = None
        self._report_command(
            [
                "git",
                "-C",
                str(self.repository_root),
                "rev-parse",
                "--show-toplevel",
            ]
        )
        try:
            self._paths = RepoPaths.discover(
                self.repository_root,
                cancel=cancel,
                command_runner=command_runner,
            )
        except BaseException as error:
            self._raise_if_cancelled(error)
            raise
        self._trust_store = trust_store
        self._environment = dict(os.environ if environment is None else environment)
        self._find_executable = executable_finder or _which

    def validate(
        self,
        analyzer_plan: Mapping[str, Any] | AnalyzerPlanLoader,
        *,
        available_secret_names: Collection[str] = (),
        external_urls: Mapping[str, str] | None = None,
    ) -> HostPreflightResult:
        """Return a complete plan or every actionable reason it is blocked."""
        self._report_command(
            [
                "git",
                "-C",
                str(self._paths.root),
                "rev-parse",
                "--git-common-dir",
            ]
        )
        try:
            require_workspace_trust(
                self._paths,
                store=self._trust_store,
                cancel=self._cancel,
                command_runner=self._run,
            )
        except (UntrustedWorkspaceError, ValueError, RuntimeError) as error:
            self._raise_if_cancelled(error)
            return self._record_validated_result(
                HostPreflightBlock(
                    (
                        HostPreflightFailure(
                            requirement=(
                                "workspace trust is required before host preflight"
                            ),
                            evidence=str(error),
                            guidance=(
                                "Run 'betterborg trust --yes' for this "
                                "exact workspace, then retry."
                            ),
                        ),
                    )
                )
            )

        plan = analyzer_plan() if callable(analyzer_plan) else analyzer_plan
        failures: list[HostPreflightFailure] = []
        (
            commands,
            prepare_commands,
            materialize_commands,
            catalog_records,
        ) = self._commands(plan, failures)
        environment_files = self._environment_files(plan, failures)
        executables, unresolved = self._executables(
            plan,
            commands,
            (*prepare_commands, *materialize_commands),
            failures,
        )
        dropped_commands: list[HostDroppedCommand] = []
        running_commands: list[HostCommand] = []
        running_records: list[Mapping[str, Any]] = []
        for command, record in zip(commands, catalog_records, strict=True):
            if _command_executable_key(command) in unresolved:
                dropped_commands.append(
                    HostDroppedCommand(
                        command,
                        f"host executable is not available: {command.argv[0]} "
                        f"(evidence: {command.evidence})",
                    )
                )
                continue
            running_commands.append(command)
            running_records.append(record)
        # A run holding no check cannot publish anything: every task would be
        # coded, reviewed and merged, and every one would then block. Whether
        # the checks were dropped here or the analysis declared none, the
        # answer is the same run and the refusal belongs before the spend.
        # A catalogue whose records were refused already said why, in the
        # failures those refusals raised. Adding that the analysis names no
        # check would be a second reason, and a false one: it named several.
        if not running_commands and not failures:
            if dropped_commands:
                failures.append(
                    HostPreflightFailure(
                        # The programs, not the commands. A refusal reason is
                        # never redacted, and it reaches the terminal and the
                        # headless payload; a catalogued argv can carry a
                        # secret the repository spelled into a script, and a
                        # program name is what the operator needs anyway.
                        requirement=(
                            "no catalogued check can run on this host: "
                            + ", ".join(
                                sorted(
                                    {
                                        dropped.command.argv[0]
                                        for dropped in dropped_commands
                                    }
                                )
                            )
                        ),
                        evidence=_join_evidence(
                            tuple(
                                dropped.command.evidence
                                for dropped in dropped_commands
                            )
                        ),
                        guidance=(
                            "Install one of the repository's checks on this "
                            "host, or run where one is available."
                        ),
                    )
                )
            else:
                failures.append(
                    HostPreflightFailure(
                        requirement=(
                            "the analysis declares no command that verifies "
                            "the repository, so nothing could prove a change "
                            "safe to publish"
                        ),
                        evidence=_evidence(
                            plan.get("command_catalog"),
                            "analyzer command catalog",
                        )
                        if isinstance(plan.get("command_catalog"), Mapping)
                        else "analyzer command catalog",
                        guidance=(
                            "Mark the command that shows a change did not "
                            "break the repository as verifying, then rerun "
                            "analysis."
                        ),
                    )
                )
        commands = running_commands
        # Which command asks for a secret is answered by the command, and the
        # answer has to reach the phase that runs it: a stage named here is
        # the stage the environment is built for.
        named_by: dict[str, list[str]] = {}
        for command, record in zip(commands, running_records, strict=True):
            for secret_name in record.get("required_secrets") or ():
                if isinstance(secret_name, str) and secret_name:
                    named_by.setdefault(secret_name, []).append(command.stage)
        secret_requirements = self._required_secrets(
            plan,
            running_records,
            {
                command.stage
                for command in (
                    *commands,
                    *prepare_commands,
                    *materialize_commands,
                )
            },
            available_secret_names,
            failures,
            named_by,
        )
        (
            compose_files,
            compose_profiles,
            compose_networks,
            compose_volumes,
            compose_build_services,
            services,
        ) = self._services(
            plan,
            running_records,
            external_urls or {},
            executables,
            failures,
        )

        if failures:
            return self._record_validated_result(
                HostPreflightBlock(tuple(failures))
            )
        return self._record_validated_result(
            HostPreflightPlan(
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
                compose_build_services=tuple(compose_build_services),
                package_managers=tuple(_package_managers(plan)),
                secret_requirements=tuple(secret_requirements),
                dropped_commands=tuple(dropped_commands),
            )
        )

    @property
    def validated_result(self) -> HostPreflightResult | None:
        """Return the exact result most recently produced by this validator."""
        return self._validated_result

    def _record_validated_result(
        self, result: HostPreflightResult
    ) -> HostPreflightResult:
        self._validated_result = result
        return result

    def _commands(
        self,
        plan: Mapping[str, Any],
        failures: list[HostPreflightFailure],
    ) -> tuple[
        list[HostCommand],
        list[HostCommand],
        list[HostCommand],
        list[Mapping[str, Any]],
    ]:
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
        # Only a catalog record carries the secrets its command asks for, and
        # a command is only kept once its argv and cwd survive validation, so
        # the surviving records are collected alongside the commands they
        # produced rather than re-derived from the plan later.
        validated_records: list[list[Mapping[str, Any]]] = []
        for group, records, default_stage, group_evidence in groups:
            commands: list[HostCommand] = []
            accepted: list[Mapping[str, Any]] = []
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
                # Whether the directory is here is a fact about this host and
                # this checkout, and a command the gate will not run does not
                # need it: an uninitialised docs submodule would refuse the
                # whole run over a directory nothing enters. The shape of the
                # path is still the analysis's to get right, so it is still
                # resolved, just not required to exist.
                resolved = self._repository_path(
                    cwd, require_directory=_verifies(record) or group != "catalog"
                )
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
                accepted.append(record)
            validated_groups.append(commands)
            validated_records.append(accepted)
        catalog_commands, prepare_commands, materialize_commands = validated_groups
        checks: list[HostCommand] = []
        check_records: list[Mapping[str, Any]] = []
        for command, record in zip(
            catalog_commands, validated_records[0], strict=True
        ):
            if not _verifies(record):
                continue
            checks.append(command)
            check_records.append(record)
        return (checks, prepare_commands, materialize_commands, check_records)

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
        catalog_commands: Sequence[HostCommand],
        environment_commands: Sequence[HostCommand],
        failures: list[HostPreflightFailure],
    ) -> tuple[list[HostExecutable], set[tuple[str, str]]]:
        requested: dict[tuple[str, str], tuple[str | None, list[str]]] = {}
        required: set[tuple[str, str]] = set()

        def add_request(
            name: str,
            cwd: str,
            version: str | None,
            evidence: str,
            *,
            blocking: bool,
        ) -> None:
            current_version, evidence_values = requested.get(
                (name, cwd), (None, [])
            )
            requested[(name, cwd)] = (
                version if version is not None else current_version,
                _unique_strings((*evidence_values, evidence)),
            )
            if blocking:
                required.add((name, cwd))

        # Environment commands build the run itself, so a program one of them
        # invokes has to be here.  A catalog command is a check, and a check
        # this host cannot invoke is dropped from the run rather than being
        # allowed to refuse it.
        invoked: set[str] = set()
        for command, blocking in (
            *((command, False) for command in catalog_commands),
            *((command, True) for command in environment_commands),
        ):
            name, cwd = _command_executable_key(command)
            invoked.add(name)
            add_request(name, cwd, None, command.evidence, blocking=blocking)

        # The toolchain and package-manager inventory is prose written for a
        # person: "Go modules" and "Node.js" name no program.  It is resolved
        # when the name happens to be one, because that is what carries a
        # version pin onto the executable, but it requires nothing on its own
        # and adds no coverage, since every command already requires what it
        # invokes.
        environment = plan.get("environment")
        toolchains: list[Mapping[str, Any]] = []
        if isinstance(environment, Mapping):
            for manager in environment.get("package_managers") or ():
                add_request(
                    str(manager),
                    ".",
                    None,
                    _evidence(environment, "environment"),
                    blocking=False,
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
                        blocking=False,
                    )

        resolved_tools: list[HostExecutable] = []
        unresolved: set[tuple[str, str]] = set()
        for (name, cwd), (version, evidence_values) in requested.items():
            path = self._resolve_executable(name, cwd=cwd)
            if path is not None:
                resolved_tools.append(
                    HostExecutable(name=name, path=path, version=version)
                )
                continue
            unresolved.add((name, cwd))
            if (name, cwd) not in required:
                continue
            failures.append(
                HostPreflightFailure(
                    requirement=f"host executable is required: {name}",
                    evidence=_join_evidence(evidence_values),
                    guidance=(
                        f"Install {name!r} on the host or update the analyzer "
                        "command/toolchain evidence; Betterborg will not install "
                        "runtimes during preflight."
                    ),
                )
            )

        # A version pin can only be checked against a program that is here,
        # and it is only this run's requirement when this run invokes it. The
        # inventory names what the repository uses somewhere; refusing over a
        # pin on a program nothing in the run calls is the refusal over tools
        # the run would never invoke that this stage exists to remove.
        by_name = {tool.name: tool for tool in resolved_tools}
        for toolchain in toolchains:
            name = toolchain.get("name")
            if not isinstance(name, str):
                continue
            executable_name = _toolchain_executable_name(name)
            if executable_name not in by_name or executable_name not in invoked:
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
        return resolved_tools, unresolved

    def _required_secrets(
        self,
        plan: Mapping[str, Any],
        command_records: Sequence[Mapping[str, Any]],
        running_stages: Collection[str],
        available: Collection[str],
        failures: list[HostPreflightFailure],
        named_by: Mapping[str, Sequence[str]] | None = None,
    ) -> list[HostSecret]:
        records = _mappings(plan.get("required_secrets"))
        by_name: dict[str, Mapping[str, Any]] = {}
        for record in records:
            name = record.get("name")
            if not isinstance(name, str) or not name:
                continue
            declared = by_name.get(name)
            # Two workflows naming one secret the same way are two sightings
            # of one requirement.  Only records that contradict each other
            # leave preflight without a single answer to act on.
            disagreements = (
                _secret_disagreements(declared, record)
                if declared is not None
                else ()
            )
            if disagreements:
                failures.append(
                    HostPreflightFailure(
                        requirement=(
                            f"required secret name must be unambiguous: {name}"
                        ),
                        evidence=_join_evidence(
                            (
                                _evidence(declared, "analyzer required_secrets"),
                                _evidence(record, "analyzer required_secrets"),
                            )
                        ),
                        guidance=(
                            "Reconcile the analyzer secret requirements that "
                            f"disagree on {'; '.join(disagreements)}, then "
                            "rerun analysis."
                        ),
                    )
                )
            if declared is None:
                by_name[name] = record
            elif not disagreements:
                # The purposes accumulate. Keeping only the first record's
                # would leave the secret unrequired for a stage the second
                # record is the sole evidence for.
                by_name[name] = {
                    **declared,
                    "used_by": _merged_used_by(declared, record),
                }

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
        stages = set(running_stages)
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
            # A secret is this run's requirement when something the run will
            # execute consumes it. A surviving command that names the secret
            # says so directly; used_by says so for the rest, and nothing in
            # the analyzer contract makes it spell a catalog stage.
            asking_stages = tuple((named_by or {}).get(name, ()))
            # An agent-scoped secret reaches the agent phases and no command.
            # A command that names one describes a requirement the run cannot
            # satisfy: preflight would make the operator configure it, and the
            # command would still run without it and fail at sanity. Refuse the
            # contradiction here rather than after the whole spend.
            if scope == "agent" and asking_stages:
                failures.append(
                    HostPreflightFailure(
                        requirement=(
                            f"required secret {name!r} is scoped to the agents "
                            "but named by a command that runs: "
                            + ", ".join(sorted(set(asking_stages)))
                        ),
                        evidence=_evidence(record, "analyzer required_secrets"),
                        guidance=(
                            "Scope the secret to build if the command needs it, "
                            "or drop it from the command's required_secrets, "
                            "then rerun analysis."
                        ),
                    )
                )
            reaches_run = (
                scope in {"all", "agent"}
                or bool(asking_stages)
                or bool(stages.intersection(used_by))
            )
            if reaches_run and name not in available_names:
                failures.append(
                    HostPreflightFailure(
                        requirement=f"required secret is not configured: {name}",
                        evidence=_evidence(record, "analyzer required_secrets"),
                        guidance=(
                            f"Configure secret {name!r} in Betterborg repository "
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
                    # The stages that use it, whatever the record called them.
                    # A command that named the secret is one of them, and the
                    # phase building its environment reads this tuple.
                    used_by=tuple(dict.fromkeys((*used_by, *asking_stages))),
                    evidence=_evidence(record, "analyzer required_secrets"),
                )
            )
        return sorted(validated, key=lambda secret: secret.name)

    def _services(
        self,
        plan: Mapping[str, Any],
        command_records: Sequence[Mapping[str, Any]],
        external_urls: Mapping[str, str],
        executables: list[HostExecutable],
        failures: list[HostPreflightFailure],
    ) -> tuple[
        list[Path],
        list[str],
        list[str],
        list[str],
        list[str],
        list[HostService],
    ]:
        catalog = plan.get("command_catalog")
        catalog_evidence = (
            _evidence(catalog, "analyzer command catalog")
            if isinstance(catalog, Mapping)
            else "analyzer command catalog"
        )
        selected: dict[str, list[str]] = {}
        # Only the commands that survived the drop. A service is selected
        # because something is going to talk to it, so one reachable only from
        # a command this host cannot run is a refusal over a dependency the
        # run does not have, which is the refusal this stage exists to remove.
        for record in command_records:
            for name in record.get("uses_services") or ():
                if isinstance(name, str):
                    selected.setdefault(name, []).append(
                        _evidence(record, catalog_evidence)
                    )
        if not selected:
            return [], [], [], [], [], []

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
        compose_build_services: list[str] = []
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
                                "then retry; Betterborg will not install them."
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
                    (
                        compose_networks,
                        compose_volumes,
                        compose_build_services,
                    ) = self._compose_topology(
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
            compose_build_services,
            services,
        )

    def _compose_topology(
        self,
        docker: Path,
        files: Sequence[Path],
        profiles: Sequence[str],
        selected: Sequence[tuple[str, Mapping[str, Any]]],
        failures: list[HostPreflightFailure],
    ) -> tuple[list[str], list[str], list[str]]:
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
            self._report_command(command)
            result = self._run(
                command,
                cwd=self.repository_root,
                env=self._probe_environment(),
                check=False,
                timeout=30,
                cancel=self._cancel,
            )
        except (OSError, subprocess.SubprocessError) as error:
            self._raise_if_cancelled(error)
            result = None
        self._raise_if_cancelled()
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
            return [], [], []

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
            build_services = [
                name
                for name, record in selected_records.items()
                if record.get("build") is not None
            ]
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
            return [], [], []
        return networks, volumes, build_services

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
            command = [str(path), *arguments]
            self._report_command(command)
            result = self._run(
                command,
                cwd=self.repository_root,
                env=self._probe_environment(),
                check=False,
                timeout=10,
                cancel=self._cancel,
            )
        except (OSError, subprocess.SubprocessError) as error:
            self._raise_if_cancelled(error)
            return None
        self._raise_if_cancelled()
        output = f"{result.stdout}\n{result.stderr}".strip()
        return output if result.returncode == 0 and output else None

    def _compose_version_output(self, docker: Path) -> str | None:
        command = [str(docker), "compose", "version"]
        try:
            self._report_command(command)
            result = self._run(
                command,
                cwd=self.repository_root,
                env=self._probe_environment(),
                check=False,
                timeout=10,
                cancel=self._cancel,
            )
        except (OSError, subprocess.SubprocessError) as error:
            self._raise_if_cancelled(error)
            return None
        self._raise_if_cancelled()
        output = f"{result.stdout}\n{result.stderr}".strip()
        return output if result.returncode == 0 and output else None

    def _report_command(self, command: Sequence[str]) -> None:
        """Publish the current secret-free probe without affecting validation."""
        if self._activity is None:
            return
        try:
            self._activity(
                AgentActivity(AgentActivityKind.COMMAND, shlex.join(command))
            )
        except Exception:
            return

    def _raise_if_cancelled(self, cause: BaseException | None = None) -> None:
        """Keep cancellation distinct from ordinary host validation failures."""
        if self._cancel is None or not self._cancel.is_set():
            return
        if cause is None:
            raise KeyboardInterrupt
        raise KeyboardInterrupt from cause

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


def _command_executable_key(command: HostCommand) -> tuple[str, str]:
    """Return how one command's program is resolved.

    A program named by path is resolved against the command's own working
    directory; a bare name is resolved on PATH, where the working directory
    makes no difference.
    """
    return (command.argv[0], command.cwd if "/" in command.argv[0] else ".")


def _merged_used_by(
    declared: Mapping[str, Any], repeated: Mapping[str, Any]
) -> list[Any]:
    """Return every purpose either record gives for one secret, once each."""
    merged: list[Any] = []
    for record in (declared, repeated):
        used_by = record.get("used_by")
        if not isinstance(used_by, Sequence) or isinstance(used_by, str | bytes):
            continue
        for value in used_by:
            if value not in merged:
                merged.append(value)
    return merged


def _verifies(record: Mapping[str, Any]) -> bool:
    """Return whether running this catalogued command proves the repository.

    The catalogue lists what a repository can do; the gate runs what shows a
    change did not break it. A record written before the analyzer was asked to
    tell the two apart declares nothing, and is run: a gate that quietly
    stopped running a repository's tests would be the worse failure, and it
    would look exactly like a gate that passed them.
    """

    return record.get("verifies") is not False


def _secret_disagreements(
    declared: Mapping[str, Any], repeated: Mapping[str, Any]
) -> tuple[str, ...]:
    """Name what two records for one secret say differently.

    Only a contradiction counts. Scope decides how the value is supplied, so
    two records cannot both be right about it. Everything else accumulates:
    two workflows naming one secret for different purposes are two things it
    is needed for, and where the evidence was found differs whenever a secret
    is named twice. Reading either as a conflict refuses a run over a secret
    nothing disagrees about.
    """
    disagreements: list[str] = []
    if declared.get("scope") != repeated.get("scope"):
        disagreements.append(
            f"scope {declared.get('scope')!r} and {repeated.get('scope')!r}"
        )
    return tuple(disagreements)


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
