"""Typed orchestration for installing Betterborg host plugins."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from betterborg_cli.claude_plugin import install_claude_plugin
from betterborg_cli.codex_plugin import install_codex_plugin
from betterborg_cli.plugin_activation import (
    PluginActivationPreflight,
    preflight_plugin_activation,
)

SUPPORTED_PLUGIN_HOSTS = ("claude", "codex")


class HostPluginInstallation(Protocol):
    """Host-specific result fields consumed by the shared orchestrator."""

    status: StrEnum
    reason: str | None
    guidance: str | None
    reload_guidance: str | None
    new_thread_guidance: str | None


HostInstaller = Callable[..., HostPluginInstallation]
Preflight = Callable[[], PluginActivationPreflight]


class PluginInstallStatus(StrEnum):
    """Consumer-visible outcome for one selected plugin host."""

    COMPLETED = "completed"
    SETUP_REQUIRED = "setup_required"
    DEFERRED = "deferred"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PluginHostInstallation:
    """Normalized installation outcome for one host."""

    host: str
    status: PluginInstallStatus
    detail: str
    guidance: str | None = None


@dataclass(frozen=True, slots=True)
class PluginInstallation:
    """Aggregate result that retains every selected host outcome."""

    hosts: tuple[PluginHostInstallation, ...]

    @property
    def ready(self) -> bool:
        return all(
            result.status
            in {PluginInstallStatus.COMPLETED, PluginInstallStatus.DEFERRED}
            for result in self.hosts
        )


class PluginInstaller:
    """Verify the persistent CLI, then invoke only selected host installers."""

    def __init__(
        self,
        *,
        preflight: Preflight = preflight_plugin_activation,
        host_installers: Mapping[str, HostInstaller] | None = None,
    ) -> None:
        self._preflight = preflight
        self._host_installers = dict(
            host_installers
            or {
                "claude": install_claude_plugin,
                "codex": install_codex_plugin,
            }
        )

    def install(
        self, hosts: Sequence[str] = SUPPORTED_PLUGIN_HOSTS
    ) -> PluginInstallation:
        """Install selected hosts in order after one shared activation preflight."""

        selected = tuple(hosts)
        unsupported = [host for host in selected if host not in SUPPORTED_PLUGIN_HOSTS]
        if unsupported:
            raise ValueError(f"unsupported plugin host: {unsupported[0]}")

        preflight = self._preflight()
        if not preflight.ready:
            detail = preflight.reason or "Persistent Betterborg CLI setup is required."
            return PluginInstallation(
                tuple(
                    PluginHostInstallation(
                        host=host,
                        status=PluginInstallStatus.SETUP_REQUIRED,
                        detail=detail,
                        guidance=preflight.guidance,
                    )
                    for host in selected
                )
            )

        return PluginInstallation(
            tuple(
                self._normalize(host, self._host_installers[host](preflight=preflight))
                for host in selected
            )
        )

    @staticmethod
    def _normalize(
        host: str, installation: HostPluginInstallation
    ) -> PluginHostInstallation:
        status = installation.status.value
        if status in {"installed", "unchanged"}:
            normalized = PluginInstallStatus.COMPLETED
            action = "Installed" if status == "installed" else "Already installed"
            detail = f"{action} the Betterborg plugin."
        elif status == "deferred":
            normalized = PluginInstallStatus.DEFERRED
            detail = (
                installation.reason
                or "Host is not installed; activation deferred."
            )
        elif status == "setup_required":
            normalized = PluginInstallStatus.SETUP_REQUIRED
            detail = installation.reason or "Host setup is required."
        else:
            normalized = PluginInstallStatus.FAILED
            detail = installation.reason or "Plugin activation failed."
        guidance = (
            getattr(installation, "guidance", None)
            or getattr(installation, "reload_guidance", None)
            or getattr(installation, "new_thread_guidance", None)
        )
        return PluginHostInstallation(
            host=host,
            status=normalized,
            detail=detail,
            guidance=guidance,
        )
