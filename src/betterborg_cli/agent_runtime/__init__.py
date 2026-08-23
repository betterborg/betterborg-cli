"""Provider-neutral agent runtime contracts and helpers."""

from betterborg_cli.agent_runtime.api_tools import (
    ApiAgentRole,
    ApiToolError,
    CommandResult,
    ContainedApiTools,
    PathContainmentError,
    SearchMatch,
    ToolGrantError,
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
from betterborg_cli.agent_runtime.mock import MockAdapter, MockResponse
from betterborg_cli.agent_runtime.process import ProcessRunner, run_streamed
from betterborg_cli.agent_runtime.retry import (
    DEFAULT_TRANSIENT_BACKOFF_SECONDS,
    DEFAULT_TRANSIENT_MAX_ATTEMPTS,
    RetryOutcome,
    retry_outcome_to_result,
    run_with_transient_retry,
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
    "AgentAdapter",
    "AgentArtifact",
    "AgentCapabilities",
    "AgentResult",
    "AgentRunSpec",
    "AgentStatus",
    "AgentUsage",
    "ApiAgentRole",
    "ApiToolError",
    "BillingMode",
    "CancellationToken",
    "CommandResult",
    "ContainedApiTools",
    "MockAdapter",
    "MockResponse",
    "ProcessRunner",
    "PathContainmentError",
    "RetryOutcome",
    "StructuredResultError",
    "SearchMatch",
    "ToolGrantError",
    "combine_agent_usage",
    "extract_json",
    "extract_structured_result",
    "extract_structured_result_file",
    "retry_outcome_to_result",
    "run_streamed",
    "run_with_transient_retry",
    "validate_structured_result",
]
