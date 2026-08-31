"""Typed, non-secret configuration tracked with a Betterborg repository."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any
from uuid import UUID

from betterborg_cli.repo_paths import RepoPaths

CONFIG_FILENAME = "config.toml"
CONFIG_VERSION = 1

_AGENT_PHASES = ("coding", "review", "merge")
_SECRET_KEY_PARTS = {
    "credential",
    "credentials",
    "password",
    "secret",
    "secrets",
    "token",
}


class RepositoryConfigError(ValueError):
    """Raised when tracked repository configuration is invalid or unsafe."""


@dataclass(frozen=True)
class AgentChoice:
    """Optional adapter selection for one agent phase."""

    adapter: str | None = None
    model: str | None = None
    effort: str | None = None


@dataclass(frozen=True)
class AgentChoices:
    """Agent selections used during coding, review, and merge phases."""

    coding: AgentChoice = field(default_factory=AgentChoice)
    review: AgentChoice = field(default_factory=AgentChoice)
    merge: AgentChoice = field(default_factory=AgentChoice)


@dataclass(frozen=True)
class ExecutionLimits:
    """Repository defaults that bound concurrent and repeated execution."""

    jobs: int = 1
    review_passes: int = 3


@dataclass(frozen=True)
class RepositoryConfig:
    """Validated contents of tracked ``.betterborg/config.toml``."""

    version: int
    repository_id: UUID
    default_branch: str
    agents: AgentChoices = field(default_factory=AgentChoices)
    execution: ExecutionLimits = field(default_factory=ExecutionLimits)


def load_repository_config(paths: RepoPaths) -> RepositoryConfig:
    """Load and validate tracked configuration for ``paths``."""
    config_path = paths.tracked_dir / CONFIG_FILENAME
    try:
        with config_path.open("rb") as config_file:
            document = tomllib.load(config_file)
    except FileNotFoundError as error:
        raise RepositoryConfigError(
            f"repository configuration does not exist: {config_path}"
        ) from error
    except tomllib.TOMLDecodeError as error:
        raise RepositoryConfigError(
            f"invalid repository configuration {config_path}: {error}"
        ) from error

    _reject_unsafe_tracked_values(document)
    return _parse_document(document)


def _parse_document(document: Mapping[str, Any]) -> RepositoryConfig:
    _require_only_keys(document, {"version", "repository", "agents", "execution"})

    version = _require_int(document, "version")
    if version != CONFIG_VERSION:
        raise RepositoryConfigError(
            f"unsupported configuration version {version}; expected {CONFIG_VERSION}"
        )

    repository = _require_table(document, "repository")
    _require_only_keys(repository, {"id", "default_branch"}, section="repository")
    repository_id_text = _require_nonempty_string(repository, "id", "repository")
    try:
        repository_id = UUID(repository_id_text)
    except ValueError as error:
        raise RepositoryConfigError("repository.id must be a valid UUID") from error
    default_branch = _require_nonempty_string(
        repository, "default_branch", "repository"
    )

    agents_document = _optional_table(document, "agents")
    _require_only_keys(agents_document, set(_AGENT_PHASES), section="agents")
    agent_choices: dict[str, AgentChoice] = {}
    for phase in _AGENT_PHASES:
        choice_document = _optional_table(agents_document, phase, section="agents")
        section = f"agents.{phase}"
        _require_only_keys(
            choice_document, {"adapter", "model", "effort"}, section=section
        )
        agent_choices[phase] = AgentChoice(
            adapter=_optional_nonempty_string(choice_document, "adapter", section),
            model=_optional_nonempty_string(choice_document, "model", section),
            effort=_optional_nonempty_string(choice_document, "effort", section),
        )

    execution_document = _optional_table(document, "execution")
    _require_only_keys(
        execution_document, {"jobs", "review_passes"}, section="execution"
    )
    jobs = _optional_int(execution_document, "jobs", default=1, section="execution")
    review_passes = _optional_int(
        execution_document, "review_passes", default=3, section="execution"
    )
    if not 1 <= jobs <= 10:
        raise RepositoryConfigError("execution.jobs must be in 1..10")
    if review_passes < 1:
        raise RepositoryConfigError("execution.review_passes must be at least 1")

    return RepositoryConfig(
        version=version,
        repository_id=repository_id,
        default_branch=default_branch,
        agents=AgentChoices(**agent_choices),
        execution=ExecutionLimits(jobs=jobs, review_passes=review_passes),
    )


def _reject_unsafe_tracked_values(value: Any, location: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_parts = set(key.casefold().replace("-", "_").split("_"))
            if key_parts & _SECRET_KEY_PARTS or key.casefold() in {
                "api_key",
                "private_key",
            }:
                dotted = ".".join((*location, key))
                raise RepositoryConfigError(
                    f"secret setting {dotted!r} is forbidden in tracked configuration"
                )
            _reject_unsafe_tracked_values(child, (*location, key))
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _reject_unsafe_tracked_values(child, (*location, str(index)))
        return
    if isinstance(value, str) and _is_absolute_machine_path(value):
        dotted = ".".join(location)
        raise RepositoryConfigError(
            f"absolute machine path at {dotted!r} is forbidden in tracked configuration"
        )


def _is_absolute_machine_path(value: str) -> bool:
    return (
        Path(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or value.startswith("~/")
        or value.startswith("~\\")
    )


def _require_only_keys(
    document: Mapping[str, Any], allowed: set[str], *, section: str = "root"
) -> None:
    unexpected = sorted(set(document) - allowed)
    if unexpected:
        raise RepositoryConfigError(
            f"unknown {section} configuration key: {unexpected[0]}"
        )


def _require_table(
    document: Mapping[str, Any], key: str, *, section: str = "root"
) -> Mapping[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise RepositoryConfigError(f"{section}.{key} must be a table")
    return value


def _optional_table(
    document: Mapping[str, Any], key: str, *, section: str = "root"
) -> Mapping[str, Any]:
    value = document.get(key, {})
    if not isinstance(value, dict):
        raise RepositoryConfigError(f"{section}.{key} must be a table")
    return value


def _require_int(document: Mapping[str, Any], key: str) -> int:
    value = document.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RepositoryConfigError(f"{key} must be an integer")
    return value


def _optional_int(
    document: Mapping[str, Any], key: str, *, default: int, section: str
) -> int:
    value = document.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RepositoryConfigError(f"{section}.{key} must be an integer")
    return value


def _require_nonempty_string(
    document: Mapping[str, Any], key: str, section: str
) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RepositoryConfigError(f"{section}.{key} must be a non-empty string")
    return value


def _optional_nonempty_string(
    document: Mapping[str, Any], key: str, section: str
) -> str | None:
    value = document.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RepositoryConfigError(f"{section}.{key} must be a non-empty string")
    return value
