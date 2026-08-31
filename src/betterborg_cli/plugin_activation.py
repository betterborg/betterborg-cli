"""Shared safety gate for installing host MCP plugins."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Generic, TypeVar

PERSISTENT_INSTALL_COMMAND = "uv tool install betterborg"

_SETUP_GUIDANCE = (
    "Install a persistent Betterborg CLI with "
    f"`{PERSISTENT_INSTALL_COMMAND}`, ensure its bin directory is on the host "
    "launch PATH, and confirm `betterborg version` succeeds. A `betterborg` "
    "supplied by `uvx` is disposable and cannot back a persistent MCP plugin."
)


class PluginActivationStatus(StrEnum):
    """Outcome of checking the shared plugin activation prerequisite."""

    READY = "ready"
    SETUP_REQUIRED = "setup_required"


@dataclass(frozen=True, slots=True)
class PluginActivationPreflight:
    """Persistent CLI identity, or actionable guidance when it is unavailable."""

    status: PluginActivationStatus
    executable: Path | None = None
    version: str | None = None
    reason: str | None = None
    guidance: str | None = None

    @property
    def ready(self) -> bool:
        return self.status is PluginActivationStatus.READY


Bundle = TypeVar("Bundle")


@dataclass(frozen=True, slots=True)
class PluginActivationPreparation(Generic[Bundle]):
    """Preflight result and the bundle created after a successful check."""

    preflight: PluginActivationPreflight
    bundle: Bundle | None = None


ExecutableLookup = Callable[..., str | None]
VersionRunner = Callable[..., subprocess.CompletedProcess[str]]


class PluginActivationVerificationError(RuntimeError):
    """The persistent CLI could not serve the bundled MCP integration."""


def preflight_plugin_activation(
    *,
    launch_environment: Mapping[str, str] | None = None,
    transient_roots: Iterable[Path] = (),
    executable_lookup: ExecutableLookup = shutil.which,
    version_runner: VersionRunner = subprocess.run,
) -> PluginActivationPreflight:
    """Resolve and verify the persistent ``betterborg`` seen by a plugin host.

    Resolution uses the launch environment's PATH rather than an ambient or
    interactive-shell lookup. Explicit transient roots let host installers
    reject their extraction directories before publishing any owned files.
    Known uvx and frozen-application extraction layouts are rejected too.
    """

    environment = dict(os.environ if launch_environment is None else launch_environment)
    path = environment.get("PATH", os.defpath)
    candidate = executable_lookup("betterborg", path=path)
    if candidate is None:
        return _setup_required(
            "No `betterborg` executable was found on the host launch PATH."
        )

    try:
        executable = Path(candidate).resolve(strict=True)
    except OSError as error:
        return _setup_required(
            "The `betterborg` executable from the host launch PATH is "
            f"unavailable: {error}."
        )

    if _is_transient(executable, transient_roots):
        return _setup_required(
            f"The resolved `betterborg` executable is transient: {executable}."
        )

    try:
        completed = version_runner(
            [str(executable), "version"],
            capture_output=True,
            check=False,
            env=environment,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return _setup_required(f"Unable to run `betterborg version`: {error}.")

    version = completed.stdout.strip()
    if completed.returncode != 0:
        detail = completed.stderr.strip() or version or "no diagnostic output"
        return _setup_required(
            f"`betterborg version` failed with exit code "
            f"{completed.returncode}: {detail}."
        )
    if not version.startswith("betterborg "):
        return _setup_required(
            "`betterborg version` did not identify the Betterborg CLI."
        )

    return PluginActivationPreflight(
        status=PluginActivationStatus.READY,
        executable=executable,
        version=version,
    )


def prepare_plugin_activation(
    materialize_owned_bundle: Callable[[PluginActivationPreflight], Bundle],
    *,
    launch_environment: Mapping[str, str] | None = None,
    transient_roots: Iterable[Path] = (),
    executable_lookup: ExecutableLookup = shutil.which,
    version_runner: VersionRunner = subprocess.run,
) -> PluginActivationPreparation[Bundle]:
    """Run preflight and materialize an owned bundle only when it succeeds."""

    preflight = preflight_plugin_activation(
        launch_environment=launch_environment,
        transient_roots=transient_roots,
        executable_lookup=executable_lookup,
        version_runner=version_runner,
    )
    if not preflight.ready:
        return PluginActivationPreparation(preflight=preflight)
    return PluginActivationPreparation(
        preflight=preflight,
        bundle=materialize_owned_bundle(preflight),
    )


def verify_borg_mcp(
    preflight: PluginActivationPreflight,
    environment: Mapping[str, str],
) -> None:
    """Spawn the persistent MCP command and require an initialize response."""

    if preflight.executable is None:
        raise PluginActivationVerificationError(
            "persistent borg executable is unavailable"
        )
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "betterborg-plugin-installer", "version": "1"},
        },
    }
    try:
        completed = subprocess.run(
            [str(preflight.executable), "mcp"],
            input=json.dumps(request) + "\n",
            capture_output=True,
            check=False,
            env=dict(environment),
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PluginActivationVerificationError(
            f"unable to start `betterborg mcp`: {error}"
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "no diagnostic output"
        raise PluginActivationVerificationError(f"`betterborg mcp` failed: {detail}")
    for line in completed.stdout.splitlines():
        try:
            response = json.loads(line)
        except json.JSONDecodeError:
            continue
        if response.get("id") == 1 and isinstance(response.get("result"), dict):
            return
    raise PluginActivationVerificationError(
        "`betterborg mcp` did not answer the initialize request"
    )


def _setup_required(reason: str) -> PluginActivationPreflight:
    return PluginActivationPreflight(
        status=PluginActivationStatus.SETUP_REQUIRED,
        reason=reason,
        guidance=_SETUP_GUIDANCE,
    )


def _is_transient(executable: Path, transient_roots: Iterable[Path]) -> bool:
    resolved_roots = (Path(root).resolve(strict=False) for root in transient_roots)
    if any(executable.is_relative_to(root) for root in resolved_roots):
        return True

    parts = tuple(part.casefold() for part in executable.parts)
    return any(
        part.startswith("archive-v") or part.startswith("_mei")
        for part in parts
    )
