"""Trust-gated validation of analyzer plans before host execution starts."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from betterborg_cli.agent_runtime.base import CancellationToken
from betterborg_cli.agent_runtime.process import run_captured
from betterborg_cli.progress import AgentActivity, AgentActivityKind
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
    evidence: str = "analyzer command catalog"


@dataclass(frozen=True, slots=True)
class HostDroppedCommand:
    """One catalog command excluded because the host cannot invoke it."""

    command: HostCommand
    reason: str


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
    required_secret_names: tuple[str, ...]
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


def selected_preparation_commands(
    *,
    prepare_commands: Sequence[HostCommand],
    materialize_commands: Sequence[HostCommand],
) -> tuple[HostCommand, ...]:
    """Return the one declared list a task worktree is prepared by.

    Two lists are declared and exactly one runs. Everything that needs that
    answer asks here, so no site restates the rule: the materialization that
    executes the commands, the key that decides whether a worktree has already
    run them, and the check of which programs this host must be able to run.
    """
    return tuple(materialize_commands or prepare_commands)


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
        unresolved = self._unrunnable_programs(
            commands,
            selected_preparation_commands(
                prepare_commands=prepare_commands,
                materialize_commands=materialize_commands,
            ),
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
                required_secret_names=tuple(
                    secret.name for secret in secret_requirements
                ),
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

    def _unrunnable_programs(
        self,
        catalog_commands: Sequence[HostCommand],
        preparation_commands: Sequence[HostCommand],
        failures: list[HostPreflightFailure],
    ) -> set[tuple[str, str]]:
        """Name the programs this host cannot run, refusing the required ones."""
        requested: dict[tuple[str, str], list[str]] = {}
        required: set[tuple[str, str]] = set()

        # A preparation command builds the run itself, so a program it invokes
        # has to be here.  A catalog command is a check, and a check this host
        # cannot invoke is dropped from the run rather than being allowed to
        # refuse it.
        for command, blocking in (
            *((command, False) for command in catalog_commands),
            *((command, True) for command in preparation_commands),
        ):
            key = _command_executable_key(command)
            requested[key] = _unique_strings(
                (*requested.get(key, ()), command.evidence)
            )
            if blocking:
                required.add(key)

        unresolved: set[tuple[str, str]] = set()
        for key, evidence_values in requested.items():
            name, cwd = key
            if self._can_run(name, cwd=cwd):
                continue
            unresolved.add(key)
            if key not in required:
                continue
            failures.append(
                HostPreflightFailure(
                    requirement=f"host executable is required: {name}",
                    evidence=_join_evidence(evidence_values),
                    guidance=(
                        f"Install {name!r} on the host or update the analyzer "
                        "command evidence; Betterborg will not install "
                        "runtimes during preflight."
                    ),
                )
            )

        return unresolved

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

    def _repository_path(
        self, value: object, *, require_directory: bool
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

    def _can_run(self, name: str, *, cwd: str = ".") -> bool:
        """Answer whether this host can invoke one command's program.

        Nothing records where the program was found, so nothing resolves it
        beyond the question a refusal is made of.
        """
        if "/" in name or "\\" in name:
            candidate = self.repository_root / PurePosixPath(cwd) / PurePosixPath(name)
            resolved = candidate.resolve()
            return (
                resolved.is_relative_to(self.repository_root)
                and resolved.is_file()
                and os.access(resolved, os.X_OK)
            )
        return self._find_executable(name, self._environment.get("PATH")) is not None

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


def _which(name: str, path: str | None) -> str | None:
    return shutil.which(name, path=path)


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


def _unique_strings(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _join_evidence(values: Sequence[str]) -> str:
    return ", ".join(_unique_strings(values))
