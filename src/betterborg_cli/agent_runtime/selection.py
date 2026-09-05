"""Native-first adapter selection and trust-gated execution policy."""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from inspect import Parameter, signature
from pathlib import Path
from typing import TYPE_CHECKING, Any

from betterborg_cli.agent_runtime.anthropic import AnthropicAdapter
from betterborg_cli.agent_runtime.api_tools import ApiAgentRole, is_read_only_tool_set
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
from betterborg_cli.agent_runtime.process import run_captured
from betterborg_cli.repository_config import AgentStage, RepositoryConfig

if TYPE_CHECKING:
    from betterborg_cli.repo_paths import RepoPaths
    from betterborg_cli.workspace_trust import TrustStore

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
_DEFAULT_MODELS = {
    "anthropic": "claude-opus-5",
    "claude": "claude-opus-5",
    "codex": "gpt-5.6-sol",
    "openai": "gpt-5.6-sol",
}
_DEFAULT_EFFORT = "high"
_STAGE_ROLES = {
    AgentStage.ANALYSIS: ApiAgentRole.ANALYSIS,
    AgentStage.REQUIREMENTS: ApiAgentRole.PLANNING,
    AgentStage.ARCHITECT: ApiAgentRole.PLANNING,
    AgentStage.TECH_LEAD: ApiAgentRole.PLANNING,
    AgentStage.PM: ApiAgentRole.PLANNING,
    AgentStage.SUPERVISOR: ApiAgentRole.PLANNING,
    AgentStage.CODING: ApiAgentRole.CODING,
    AgentStage.REVIEW: ApiAgentRole.REVIEW,
    AgentStage.MERGE: ApiAgentRole.MERGE,
}
_SETUP_GUIDANCE = (
    "Install and log in to the 'claude' or 'codex' CLI, or set "
    "ANTHROPIC_API_KEY or OPENAI_API_KEY for API use."
)


def _default_trust_requirement() -> Callable[..., Any]:
    from betterborg_cli.workspace_trust import require_workspace_trust

    return require_workspace_trust


class AgentSelectionError(RuntimeError):
    """Raised with operator-facing guidance when no adapter can be selected."""


def require_read_only_agent(
    agent: AgentAdapter | SelectedAgent,
    *,
    role: str,
    error_factory: Callable[[str], Exception],
) -> None:
    """Reject an adapter that cannot enforce a read-only execution boundary.

    An adapter qualifies by confining tool access to an allowlist, or by
    running its provider CLI in a read-only sandbox. An operator who has
    declared the environment already isolated satisfies the second without a
    sandbox actually in force; see ``codex._sandbox_setting``. A host-capable adapter
    additionally has to arrive wrapped, because ``SelectedAgent`` is what
    holds it to a read-only tool set and to workspace trust.
    """
    if not (
        agent.capabilities.tool_allowlist or agent.capabilities.read_only_sandbox
    ):
        raise error_factory(
            f"adapter {agent.name!r} cannot enforce the {role} read-only "
            "execution boundary"
        )
    if agent.capabilities.host_capable and not isinstance(agent, SelectedAgent):
        raise error_factory(
            f"host-capable adapter {agent.name!r} must be wrapped by SelectedAgent"
        )


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
        default_factory=_default_trust_requirement,
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
        resolved_spec = self._resolve_spec(spec)
        if cancel is not None and cancel.is_set():
            return self.adapter.run(resolved_spec, cancel=cancel)

        run_paths = self._bound_run_paths(resolved_spec.cwd, cancel)
        if self.capabilities.host_capable:
            self._require_trust(
                run_paths,
                cancel,
            )
            if isinstance(self.adapter, AnthropicAdapter | OpenAIAdapter):
                self.adapter.workspace_trusted = True

        return self.adapter.run(resolved_spec, cancel=cancel)

    def run_contained(
        self,
        spec: AgentRunSpec,
        *,
        cancel: CancellationToken | None = None,
    ) -> AgentResult:
        """Invoke a read-only adapter in a caller-built contained workspace.

        The caller owns the workspace, so this path deliberately does not
        interpret the sanitized directory as a raw Git checkout. How tightly
        the run is held to that directory depends on the transport:

        A provider API adapter is path-contained, because its file tools
        resolve every path beneath the run directory. It reaches nothing else
        on the host and therefore needs no repository trust.

        A native CLI is held to reading by whatever its provider offers:
        Claude by a read-only tool set, Codex by a sandbox an operator can
        decline, after which Codex can write. Neither provider confines reads
        to the working directory, so either can still reach the repository the
        workspace was sampled from. For a native CLI the workspace is
        therefore a bound on cost and on the evidence the prompt cites, not a
        containment boundary.

        Trust is required for that host access, but every current caller
        already trusts the workspace before selecting an agent, so this is
        defence in depth for future callers rather than a gate an operator
        will meet here.
        """
        if cancel is not None and cancel.is_set():
            return self.adapter.run(self._resolve_spec(spec), cancel=cancel)

        require_read_only_agent(
            self,
            role=self.role.value,
            error_factory=AgentSelectionError,
        )
        if self.capabilities.host_capable:
            if not is_read_only_tool_set(spec.allowed_tools):
                raise AgentSelectionError(
                    f"Host-capable adapter {self.name!r} may only run in a "
                    "bounded workspace under a read-only tool set"
                )
            self._require_trust(
                self.paths,
                cancel,
            )
        return self.adapter.run(self._resolve_spec(spec), cancel=cancel)

    def _resolve_spec(self, spec: AgentRunSpec) -> AgentRunSpec:
        return replace(
            spec,
            model=self.model or spec.model,
            effort=self.effort if self.effort is not None else spec.effort,
            billing_mode=(
                BillingMode.SUBSCRIPTION
                if self.name in _NATIVE_ADAPTERS
                else BillingMode.API
            ),
        )

    def _bound_run_paths(
        self,
        cwd: Path,
        cancel: CancellationToken | None,
    ) -> RepoPaths:
        """Resolve a run cwd that belongs to the selected repository."""
        from betterborg_cli.repo_paths import RepoPaths
        from betterborg_cli.workspace_trust import WorkspaceIdentity

        try:
            run_paths = RepoPaths.discover(cwd, cancel=cancel)
            selected_identity = WorkspaceIdentity.discover(
                self.paths,
                cancel=cancel,
            )
            run_identity = WorkspaceIdentity.discover(
                run_paths,
                cancel=cancel,
            )
        except ValueError as error:
            raise AgentSelectionError(
                f"Agent run cwd is not a usable Git workspace: {cwd}"
            ) from error

        same_workspace = run_paths.root == self.paths.root
        managed_worktree = (
            self.paths.manages(run_paths.root)
            and run_identity.git_common_dir == selected_identity.git_common_dir
        )
        if not same_workspace and not managed_worktree:
            raise AgentSelectionError(
                "Agent run cwd belongs to a different repository: "
                f"{run_paths.root} (selected {self.paths.root})"
            )
        return run_paths

    def _require_trust(
        self,
        paths: RepoPaths,
        cancel: CancellationToken | None,
    ) -> None:
        """Call injected trust checks with supported cancellation keywords."""
        kwargs: dict[str, Any] = {
            "store": self.trust_store,
            "explicit": self.trust_explicit,
            "interactive": self.interactive,
            "confirm": self.trust_confirm,
        }
        parameters = signature(self.trust_requirement).parameters.values()
        accepts_arbitrary_keywords = any(
            parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters
        )
        supported_keywords = {
            parameter.name
            for parameter in parameters
            if parameter.kind
            in {Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY}
        }
        if accepts_arbitrary_keywords or "cancel" in supported_keywords:
            kwargs["cancel"] = cancel
        if accepts_arbitrary_keywords or "command_runner" in supported_keywords:
            kwargs["command_runner"] = run_captured
        self.trust_requirement(paths, **kwargs)


def select_agent(
    config: RepositoryConfig,
    stage: AgentStage,
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
    """Resolve one configured stage across native and API transports.

    The stage uses its exact configuration and maps to an internal security role.
    Every stage prefers a logged-in native CLI and falls back to a provider
    API credential, so subscription billing is chosen ahead of metered billing.
    ``interactive`` governs only whether a trust prompt may be shown, never
    which transports are eligible. Provider credentials are read here, by the
    selection-policy owner, and only the credential for the selected API
    adapter is retained.
    """
    resolved_role = _STAGE_ROLES[stage]
    choice = config.agents.resolve(stage)
    identity_description = f"stage {stage.value!r}"
    tty = sys.stdin.isatty() if interactive is None else interactive
    environment = os.environ if credentials is None else credentials
    find_executable = executable_lookup or shutil.which

    if choice.adapter is not None:
        adapter_name = choice.adapter
        if adapter_name not in _KNOWN_ADAPTERS:
            raise AgentSelectionError(
                _with_setup(
                    f"Configured adapter {adapter_name!r} for "
                    f"{identity_description} is unknown; choose one of "
                    f"{', '.join(sorted(_KNOWN_ADAPTERS))}."
                )
            )
        unusable = _unusable_reason(
            adapter_name,
            credentials=environment,
            executable_lookup=find_executable,
        )
        if unusable is not None:
            raise AgentSelectionError(
                _with_setup(
                    f"Configured adapter {adapter_name!r} for "
                    f"{identity_description} is not usable: {unusable}."
                )
            )
    else:
        adapter_name = next(
            (
                candidate
                for candidate in (*_NATIVE_ADAPTERS, *_API_ADAPTERS)
                if _unusable_reason(
                    candidate,
                    credentials=environment,
                    executable_lookup=find_executable,
                )
                is None
            ),
            None,
        )
        if adapter_name is None:
            raise AgentSelectionError(
                _with_setup(
                    f"No usable agent adapter is configured for "
                    f"{identity_description}."
                )
            )

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
        model=resolve_adapter_model(adapter_name, choice.model),
        effort=choice.effort or _DEFAULT_EFFORT,
        paths=paths,
        interactive=tty,
        trust_store=trust_store,
        trust_explicit=trust_explicit,
        trust_confirm=trust_confirm,
        trust_requirement=trust_requirement or _default_trust_requirement(),
    )


def resolve_agent_model(
    agent: AgentAdapter | SelectedAgent,
    configured_model: str | None,
) -> str:
    """Resolve an explicit, selected, or provider-default agent model."""
    if configured_model is not None:
        return configured_model
    if isinstance(agent, SelectedAgent) and agent.model is not None:
        return agent.model
    return resolve_adapter_model(agent.name, None)


def resolve_adapter_model(adapter: str, configured_model: str | None) -> str:
    """Resolve an explicit model or the runtime default for an adapter name."""
    if configured_model is not None:
        return configured_model
    try:
        return _DEFAULT_MODELS[adapter]
    except KeyError as error:
        raise AgentSelectionError(
            f"Agent model must be configured for adapter {adapter!r}"
        ) from error


def _unusable_reason(
    name: str,
    *,
    credentials: Mapping[str, str],
    executable_lookup: Callable[[str], str | None],
) -> str | None:
    if name in _NATIVE_ADAPTERS:
        if executable_lookup(name) is None:
            return f"the {name!r} executable was not found on PATH"
        return None
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
