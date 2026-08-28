"""TTY-aware adapter selection and trust-gated execution policy."""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from betterborg_cli.agent_runtime.anthropic import AnthropicAdapter
from betterborg_cli.agent_runtime.api_tools import ApiAgentRole
from betterborg_cli.agent_runtime.base import (
    AgentAdapter,
    AgentCapabilities,
    AgentResult,
    AgentRunSpec,
    BillingMode,
    CancellationToken,
)
from betterborg_cli.agent_runtime.claude import ClaudeAdapter
from betterborg_cli.agent_runtime.codex import CodexAdapter
from betterborg_cli.agent_runtime.openai import OpenAIAdapter
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.repository_config import AgentChoice, RepositoryConfig
from betterborg_cli.workspace_trust import (
    TrustStore,
    WorkspaceIdentity,
    require_workspace_trust,
)

_NATIVE_ADAPTERS = ("claude", "codex")
_API_ADAPTERS = ("anthropic", "openai")
_KNOWN_ADAPTERS = frozenset((*_NATIVE_ADAPTERS, *_API_ADAPTERS))
_EXECUTION_ROLES = frozenset(
    {ApiAgentRole.CODING, ApiAgentRole.REVIEW, ApiAgentRole.MERGE}
)
_API_CREDENTIALS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}
_SETUP_GUIDANCE = (
    "Install and log in to the 'claude' or 'codex' CLI for interactive use, "
    "or set ANTHROPIC_API_KEY or OPENAI_API_KEY for API use."
)


class AgentSelectionError(RuntimeError):
    """Raised with operator-facing guidance when no adapter can be selected."""


@dataclass(slots=True)
class SelectedAgent:
    """A resolved adapter plus the role policy applied to each invocation."""

    role: ApiAgentRole
    adapter: AgentAdapter = field(repr=False)
    paths: RepoPaths = field(repr=False)
    model: str | None = None
    effort: str | None = None
    interactive: bool = False
    trust_store: TrustStore | None = field(default=None, repr=False)
    trust_explicit: bool = False
    trust_confirm: Callable[[str], bool] | None = field(default=None, repr=False)
    trust_requirement: Callable[..., Any] = field(
        default=require_workspace_trust,
        repr=False,
    )

    @property
    def name(self) -> str:
        """Return the selected transport's stable configuration name."""
        return self.adapter.name

    @property
    def capabilities(self) -> AgentCapabilities:
        """Disclose effective capabilities after role policy is applied."""
        return self.adapter.capabilities

    def run(
        self,
        spec: AgentRunSpec,
        *,
        cancel: CancellationToken | None = None,
    ) -> AgentResult:
        """Apply overrides, require trust when needed, then invoke the adapter."""
        resolved_spec = replace(
            spec,
            model=self.model or spec.model,
            effort=self.effort if self.effort is not None else spec.effort,
            billing_mode=(
                BillingMode.SUBSCRIPTION
                if self.name in _NATIVE_ADAPTERS
                else BillingMode.API
            ),
        )
        if cancel is not None and cancel.is_set():
            return self.adapter.run(resolved_spec, cancel=cancel)

        run_paths = self._bound_run_paths(resolved_spec.cwd)
        if self.capabilities.host_capable:
            self.trust_requirement(
                run_paths,
                store=self.trust_store,
                explicit=self.trust_explicit,
                interactive=self.interactive,
                confirm=self.trust_confirm,
            )
            if isinstance(self.adapter, AnthropicAdapter | OpenAIAdapter):
                self.adapter.workspace_trusted = True

        return self.adapter.run(resolved_spec, cancel=cancel)

    def _bound_run_paths(self, cwd: Path) -> RepoPaths:
        """Resolve a run cwd that belongs to the selected repository."""
        try:
            run_paths = RepoPaths.discover(cwd)
            selected_identity = WorkspaceIdentity.discover(self.paths)
            run_identity = WorkspaceIdentity.discover(run_paths)
        except ValueError as error:
            raise AgentSelectionError(
                f"Agent run cwd is not a usable Git workspace: {cwd}"
            ) from error

        same_workspace = run_paths.root == self.paths.root
        managed_worktree = (
            run_paths.root.is_relative_to(self.paths.worktrees_dir.resolve())
            and run_identity.git_common_dir == selected_identity.git_common_dir
        )
        if not same_workspace and not managed_worktree:
            raise AgentSelectionError(
                "Agent run cwd belongs to a different repository: "
                f"{run_paths.root} (selected {self.paths.root})"
            )
        return run_paths


def select_agent(
    config: RepositoryConfig,
    role: ApiAgentRole | str,
    paths: RepoPaths,
    *,
    interactive: bool | None = None,
    credentials: Mapping[str, str] | None = None,
    executable_lookup: Callable[[str], str | None] | None = None,
    trust_store: TrustStore | None = None,
    trust_explicit: bool = False,
    trust_confirm: Callable[[str], bool] | None = None,
    trust_requirement: Callable[..., Any] | None = None,
) -> SelectedAgent:
    """Resolve one configured role across native and provider API transports.

    Native transports are eligible only for an interactive invocation. Provider
    credentials are read here, by the selection-policy owner, and only the
    credential for the selected API adapter is retained.
    """
    resolved_role = ApiAgentRole(role)
    tty = sys.stdin.isatty() if interactive is None else interactive
    environment = os.environ if credentials is None else credentials
    find_executable = executable_lookup or shutil.which
    choice = _choice_for_role(config, resolved_role)

    if choice.adapter is not None:
        adapter_name = choice.adapter
        if adapter_name not in _KNOWN_ADAPTERS:
            raise AgentSelectionError(
                _with_setup(
                    f"Configured adapter {adapter_name!r} for role "
                    f"{resolved_role.value!r} is unknown; choose one of "
                    f"{', '.join(sorted(_KNOWN_ADAPTERS))}."
                )
            )
        unusable = _unusable_reason(
            adapter_name,
            tty=tty,
            effort=choice.effort,
            credentials=environment,
            executable_lookup=find_executable,
        )
        if unusable is not None:
            raise AgentSelectionError(
                _with_setup(
                    f"Configured adapter {adapter_name!r} for role "
                    f"{resolved_role.value!r} is not usable: {unusable}."
                )
            )
    else:
        candidates = (*_NATIVE_ADAPTERS, *_API_ADAPTERS) if tty else _API_ADAPTERS
        adapter_name = next(
            (
                candidate
                for candidate in candidates
                if _unusable_reason(
                    candidate,
                    tty=tty,
                    effort=choice.effort,
                    credentials=environment,
                    executable_lookup=find_executable,
                )
                is None
            ),
            None,
        )
        if adapter_name is None:
            context = (
                "No usable agent adapter is configured."
                if tty
                else "No provider API credential is configured for non-interactive use."
            )
            raise AgentSelectionError(_with_setup(context))

    adapter = _build_adapter(
        adapter_name,
        resolved_role,
        credentials=environment,
    )
    if adapter_name in _API_ADAPTERS and resolved_role in _EXECUTION_ROLES:
        adapter.capabilities = replace(adapter.capabilities, host_capable=True)

    return SelectedAgent(
        role=resolved_role,
        adapter=adapter,
        model=choice.model,
        effort=choice.effort,
        paths=paths,
        interactive=tty,
        trust_store=trust_store,
        trust_explicit=trust_explicit,
        trust_confirm=trust_confirm,
        trust_requirement=trust_requirement or require_workspace_trust,
    )


def _choice_for_role(config: RepositoryConfig, role: ApiAgentRole) -> AgentChoice:
    if role in _EXECUTION_ROLES:
        return getattr(config.agents, role.value)
    return AgentChoice()


def _unusable_reason(
    name: str,
    *,
    tty: bool,
    effort: str | None,
    credentials: Mapping[str, str],
    executable_lookup: Callable[[str], str | None],
) -> str | None:
    if name in _NATIVE_ADAPTERS:
        if not tty:
            return "native CLI adapters require an interactive TTY"
        if executable_lookup(name) is None:
            return f"the {name!r} executable was not found on PATH"
        return None
    if effort is not None and name == "anthropic":
        return "Anthropic does not support an effort override"
    variable = _API_CREDENTIALS[name]
    if not credentials.get(variable):
        return f"{variable} is not set"
    return None


def _build_adapter(
    name: str,
    role: ApiAgentRole,
    *,
    credentials: Mapping[str, str],
) -> AgentAdapter:
    if name == "claude":
        return ClaudeAdapter(role)
    if name == "codex":
        return CodexAdapter(role)
    variable = _API_CREDENTIALS[name]
    credential = credentials[variable]
    if name == "anthropic":
        return AnthropicAdapter(role, api_key=credential)
    return OpenAIAdapter(role, api_key=credential)


def _with_setup(message: str) -> str:
    return f"{message} {_SETUP_GUIDANCE}"
