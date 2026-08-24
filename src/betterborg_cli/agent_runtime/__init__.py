"""Provider-neutral agent runtime contracts and helpers."""

from betterborg_cli.agent_runtime.anthropic import (
    ANTHROPIC_API_URL,
    ANTHROPIC_API_VERSION,
    AnthropicAdapter,
    AnthropicApiError,
    AnthropicTransport,
    UrllibAnthropicTransport,
)
from betterborg_cli.agent_runtime.api_tools import (
    ApiAgentRole,
    ApiToolDefinition,
    ApiToolError,
    CommandResult,
    ContainedApiTools,
    PathContainmentError,
    SearchMatch,
    ToolGrantError,
    api_tool_definition,
    select_api_tool_names,
)
from betterborg_cli.agent_runtime.base import (
    AgentAdapter,
    AgentArtifact,
    AgentCapabilities,
    AgentResult,
    AgentRunSpec,
    AgentStatus,
    AgentUsage,
    BillingMode,
    CancellationToken,
    combine_agent_usage,
)
from betterborg_cli.agent_runtime.claude import ClaudeAdapter
from betterborg_cli.agent_runtime.codex import CodexAdapter
from betterborg_cli.agent_runtime.mock import MockAdapter, MockResponse
from betterborg_cli.agent_runtime.openai import (
    OPENAI_API_URL,
    OpenAIAdapter,
    OpenAIApiError,
    OpenAITransport,
    UrllibOpenAITransport,
)
from betterborg_cli.agent_runtime.process import ProcessRunner, run_streamed
from betterborg_cli.agent_runtime.retry import (
    DEFAULT_TRANSIENT_BACKOFF_SECONDS,
    DEFAULT_TRANSIENT_MAX_ATTEMPTS,
    RetryOutcome,
    retry_outcome_to_result,
    run_with_transient_retry,
)
from betterborg_cli.agent_runtime.selection import (
    AgentSelectionError,
    SelectedAgent,
    select_agent,
)
from betterborg_cli.agent_runtime.structured import (
    StructuredResultError,
    extract_json,
    extract_structured_result,
    extract_structured_result_file,
    validate_structured_result,
)

__all__ = [
    "DEFAULT_TRANSIENT_BACKOFF_SECONDS",
    "DEFAULT_TRANSIENT_MAX_ATTEMPTS",
    "ANTHROPIC_API_URL",
    "ANTHROPIC_API_VERSION",
    "OPENAI_API_URL",
    "AgentAdapter",
    "AgentArtifact",
    "AgentCapabilities",
    "AgentResult",
    "AgentRunSpec",
    "AgentSelectionError",
    "AgentStatus",
    "AgentUsage",
    "AnthropicAdapter",
    "AnthropicApiError",
    "AnthropicTransport",
    "ApiAgentRole",
    "ApiToolDefinition",
    "ApiToolError",
    "BillingMode",
    "CancellationToken",
    "ClaudeAdapter",
    "CodexAdapter",
    "CommandResult",
    "ContainedApiTools",
    "MockAdapter",
    "MockResponse",
    "OpenAIAdapter",
    "OpenAIApiError",
    "OpenAITransport",
    "ProcessRunner",
    "PathContainmentError",
    "RetryOutcome",
    "StructuredResultError",
    "SearchMatch",
    "SelectedAgent",
    "ToolGrantError",
    "UrllibAnthropicTransport",
    "UrllibOpenAITransport",
    "api_tool_definition",
    "combine_agent_usage",
    "extract_json",
    "extract_structured_result",
    "extract_structured_result_file",
    "retry_outcome_to_result",
    "run_streamed",
    "run_with_transient_retry",
    "select_api_tool_names",
    "select_agent",
    "validate_structured_result",
]
