"""Logged-in Codex CLI adapter with native host tool access."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from betterborg_cli.agent_runtime.api_tools import ApiAgentRole, is_read_only_tool_set
from betterborg_cli.agent_runtime.base import (
    AgentArtifact,
    AgentCapabilities,
    AgentRunSpec,
    AgentUsage,
    BillingMode,
)
from betterborg_cli.agent_runtime.native_cli import (
    NativeCliAdapter,
    NativeInvocation,
    NativePayload,
)
from betterborg_cli.agent_runtime.process import ProcessRunner, run_streamed
from betterborg_cli.agent_runtime.retry import (
    DEFAULT_TRANSIENT_BACKOFF_SECONDS,
    DEFAULT_TRANSIENT_MAX_ATTEMPTS,
)
from betterborg_cli.agent_runtime.structured import (
    StructuredResultError,
    extract_json,
)

_PROVIDER = "codex"
_PROMPT_SCHEMA_INSTRUCTIONS = """
## Output format requirement

Your response MUST be a single JSON object matching this JSON Schema:

```json
{schema}
```

Do not wrap the JSON in prose. Do not include any text before or after the
JSON object.
""".strip()
_OPENAI_UNSUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {
        "$schema",
        "title",
        "default",
        "minLength",
        "maxLength",
        "pattern",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
)


class _PromptSchemaFallback(ValueError):
    """The caller schema needs prompt-constrained output and local validation."""


@dataclass(slots=True)
class CodexAdapter(NativeCliAdapter):
    """Run Codex non-interactively for a Betterborg role."""

    role: ApiAgentRole | str
    binary: str = "codex"
    proc_runner: ProcessRunner = run_streamed
    artifacts: tuple[AgentArtifact, ...] = ()
    transient_backoff_seconds: float = DEFAULT_TRANSIENT_BACKOFF_SECONDS
    transient_max_attempts: int = DEFAULT_TRANSIENT_MAX_ATTEMPTS
    name: str = field(default=_PROVIDER, init=False)
    provider_label: str = field(default="Codex", init=False)
    capabilities: AgentCapabilities = field(
        default_factory=lambda: AgentCapabilities(
            billing_modes=frozenset({BillingMode.SUBSCRIPTION}),
            structured_output=True,
            streaming=True,
            resumable=True,
            host_capable=True,
            read_only_sandbox=True,
        ),
        init=False,
    )

    def __post_init__(self) -> None:
        self._validate_native_configuration()

    @contextmanager
    def _prepare_invocation(
        self, spec: AgentRunSpec
    ) -> Iterator[NativeInvocation]:
        with tempfile.TemporaryDirectory(prefix="betterborg-codex-") as directory:
            temp_directory = Path(directory)
            schema_path = temp_directory / "schema.json"
            invocation_result_path = temp_directory / "result.json"
            try:
                transport_schema = _normalize_for_openai_strict(spec.schema)
            except _PromptSchemaFallback:
                transport_schema = None
            if transport_schema is not None:
                schema_path.write_text(
                    json.dumps(transport_schema, sort_keys=True), encoding="utf-8"
                )
            yield NativeInvocation(
                command=self._command(
                    spec,
                    schema_path if transport_schema is not None else None,
                    invocation_result_path,
                ),
                stdin_text=_stdin_prompt(
                    spec, include_output_schema=transport_schema is None
                ),
                load_payload=lambda: self._load_payload(
                    invocation_result_path,
                    spec.log_path,
                    spec.schema,
                    strip_optional_nulls=transport_schema is not None,
                ),
                before_attempt=lambda: invocation_result_path.unlink(
                    missing_ok=True
                ),
                accept_payload_on_nonzero_exit=True,
            )

    def _command(
        self,
        spec: AgentRunSpec,
        schema_path: Path | None,
        invocation_result_path: Path,
    ) -> list[str]:
        command = [
            self.binary,
            "exec",
            "--json",
            "-C",
            str(spec.cwd),
            "-m",
            spec.model,
            "-s",
            _sandbox_for(spec),
            "--skip-git-repo-check",
            "--ignore-user-config",
            "--ephemeral",
        ]
        if schema_path is not None:
            command.extend(("--output-schema", str(schema_path)))
        command.extend(("-o", str(invocation_result_path)))
        if spec.effort:
            command.extend(("-c", f"model_reasoning_effort={spec.effort}"))
        command.append("-")
        return command

    def _load_payload(
        self,
        invocation_result_path: Path,
        log_path: Path,
        schema: Mapping[str, Any],
        *,
        strip_optional_nulls: bool,
    ) -> NativePayload:
        payload = _load_payload(invocation_result_path)
        if payload is None:
            return NativePayload(error=_terminal_error(log_path, 0))
        if strip_optional_nulls:
            payload = _strip_optional_nulls(payload, schema)
        return NativePayload(payload=payload)

    def _extract_usage(self, log_path: Path) -> AgentUsage | None:
        return _extract_usage(log_path)

    def _classify_transient_error(
        self, log_path: Path, exit_code: int
    ) -> str | None:
        if exit_code == 0:
            return None
        return _classify_transient_error(log_path)

    def _terminal_error(self, log_path: Path, exit_code: int) -> str:
        return _terminal_error(log_path, exit_code)


def _sandbox_for(spec: AgentRunSpec) -> str:
    if is_read_only_tool_set(spec.allowed_tools):
        return "read-only"
    return "danger-full-access"


def _stdin_prompt(
    spec: AgentRunSpec, *, include_output_schema: bool = False
) -> str:
    system_prompt = spec.system_prompt.rstrip()
    if include_output_schema:
        system_prompt += "\n\n" + _PROMPT_SCHEMA_INSTRUCTIONS.format(
            schema=json.dumps(spec.schema, indent=2, sort_keys=True)
        )
    user_prompt = spec.user_prompt.rstrip()
    return f"{system_prompt}\n\n<stdin>\n{user_prompt}\n</stdin>\n"


def _load_payload(result_path: Path) -> dict[str, Any] | None:
    if not result_path.exists() or result_path.stat().st_size == 0:
        return None
    try:
        payload = extract_json(
            result_path.read_text(encoding="utf-8", errors="replace")
        )
    except (OSError, StructuredResultError):
        return None
    return payload if isinstance(payload, dict) else None


def _classify_transient_error(log_path: Path) -> str | None:
    if not log_path.exists() or log_path.stat().st_size == 0:
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    patterns = (
        ("usage limit", "Codex usage limit hit"),
        ("status 429", "Codex rate limit"),
        ("rate limit", "Codex rate limit"),
        ("at capacity", "Codex model at capacity"),
        ("is overloaded", "Codex model overloaded"),
        ("model is overloaded", "Codex model overloaded"),
        ("try a different model", "Codex model at capacity"),
        ("status 503", "Codex service unavailable"),
        ("failed to refresh token", "Codex authentication refresh failed"),
        ("could not be refreshed", "Codex authentication refresh failed"),
        ("401 unauthorized", "Codex authentication failed"),
        ("stream disconnected", "Codex network transport failed"),
        ("error sending request", "Codex network transport failed"),
        ("connection reset by peer", "Codex network transport failed"),
    )
    for marker, description in patterns:
        if marker in lowered:
            return f"{description}: {_matching_log_line(text, marker)}"
    return None


def _matching_log_line(text: str, marker: str) -> str:
    for line in reversed(text.splitlines()):
        if marker in line.lower():
            return line.strip()[:300]
    return "see log"


def _terminal_error(log_path: Path, exit_code: int) -> str:
    message = (
        "Codex exited 0 and produced no result"
        if exit_code == 0
        else f"Codex exited {exit_code} without completing successfully"
    )
    if not log_path.exists():
        return message
    lines = [
        line.strip()
        for line in log_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if line.strip()
    ]
    if not lines:
        return message
    for line in reversed(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, Mapping):
            error = event.get("error")
            if isinstance(error, Mapping):
                error = error.get("message")
            detail = event.get("message") or error
            if isinstance(detail, str) and detail.strip():
                return f"{message}: {detail.strip()[:1000]}"
    return f"{message}: {lines[-1][-1000:]}"


def _extract_usage(log_path: Path) -> AgentUsage | None:
    """Sum Codex usage while preserving mutually exclusive token buckets."""
    if not log_path.exists() or log_path.stat().st_size == 0:
        return None
    total_input = 0
    cached_input = 0
    cache_write_input = 0
    total_output = 0
    turns = 0
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping) or event.get("type") != "turn.completed":
            continue
        usage = event.get("usage")
        if not isinstance(usage, Mapping):
            continue
        turn_input = _token(usage.get("input_tokens"))
        turn_cached = _token(usage.get("cached_input_tokens"))
        turn_cache_write = _token(usage.get("cache_write_input_tokens"))
        total_input += max(turn_input - turn_cached - turn_cache_write, 0)
        cached_input += turn_cached
        cache_write_input += turn_cache_write
        total_output += _token(usage.get("output_tokens"))
        turns += 1
    if turns == 0:
        return None
    return AgentUsage(
        tokens_input=total_input,
        tokens_output=total_output,
        tokens_cache_read=cached_input,
        tokens_cache_write=cache_write_input,
        num_turns=turns,
    )


def _token(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def _normalize_for_openai_strict(
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate locally supported schemas to Codex's strict transport subset."""

    def walk(value: Any, *, path: str, schema_context: bool = True) -> Any:
        if isinstance(value, Mapping):
            if not schema_context:
                return {
                    name: walk(child, path=f"{path}.{name}")
                    for name, child in value.items()
                }
            unsupported_composition = {"allOf", "anyOf", "not", "oneOf"}.intersection(
                value
            )
            if unsupported_composition:
                names = ", ".join(sorted(unsupported_composition))
                raise _PromptSchemaFallback(f"{path} uses {names}")

            normalized: dict[str, Any] = {}
            for name, child in value.items():
                if name in _OPENAI_UNSUPPORTED_SCHEMA_KEYWORDS:
                    continue
                if name in {"const", "enum"}:
                    normalized[name] = deepcopy(child)
                    continue
                normalized[name] = walk(
                    child,
                    path=f"{path}.{name}",
                    schema_context=name not in {"properties", "$defs"},
                )

            declared_type = normalized.get("type")
            is_object = declared_type == "object" or (
                isinstance(declared_type, list) and "object" in declared_type
            )
            if any(
                name in normalized
                for name in ("properties", "required", "additionalProperties")
            ) and not is_object:
                raise _PromptSchemaFallback(
                    f"{path} uses object keywords without type object"
                )
            if is_object:
                properties = normalized.get("properties")
                if (
                    not isinstance(properties, dict)
                    or normalized.get("additionalProperties") is not False
                ):
                    raise _PromptSchemaFallback(
                        f"{path} cannot be represented as a closed strict object"
                    )
                originally_required = set(normalized.get("required", ()))
                for name, property_schema in properties.items():
                    if name not in originally_required:
                        _make_nullable(property_schema)
                normalized["required"] = list(properties)
                normalized["additionalProperties"] = False
            return normalized
        if isinstance(value, list):
            return [
                walk(child, path=f"{path}[{index}]")
                for index, child in enumerate(value)
            ]
        return deepcopy(value)

    normalized = walk(schema, path="$")
    if normalized.get("type") != "object" or "anyOf" in normalized:
        raise _PromptSchemaFallback("root must be a non-union object")
    return normalized


def _make_nullable(schema: Any) -> None:
    if not isinstance(schema, dict):
        raise _PromptSchemaFallback("property schema must be an object")

    if "const" in schema:
        original = dict(schema)
        schema.clear()
        schema["anyOf"] = [original, {"type": "null"}]
        return

    enum = schema.get("enum")
    if isinstance(enum, list) and None not in enum:
        schema["enum"] = [*enum, None]

    expected_type = schema.get("type")
    if isinstance(expected_type, str):
        if expected_type != "null":
            schema["type"] = [expected_type, "null"]
        return
    if isinstance(expected_type, list):
        if "null" not in expected_type:
            schema["type"] = [*expected_type, "null"]
        return
    if isinstance(enum, list):
        return
    original = dict(schema)
    schema.clear()
    schema["anyOf"] = [original, {"type": "null"}]


def _strip_optional_nulls(
    payload: dict[str, Any], schema: Mapping[str, Any]
) -> dict[str, Any]:
    """Remove null placeholders introduced only for strict transport."""

    def resolve(node: Mapping[str, Any]) -> Mapping[str, Any]:
        reference = node.get("$ref")
        if not isinstance(reference, str) or not reference.startswith("#/"):
            return node
        target: Any = schema
        for raw_part in reference[2:].split("/"):
            part = raw_part.replace("~1", "/").replace("~0", "~")
            if not isinstance(target, Mapping) or part not in target:
                return node
            target = target[part]
        return target if isinstance(target, Mapping) else node

    def walk(value: Any, node: Mapping[str, Any]) -> Any:
        node = resolve(node)
        if isinstance(value, dict):
            properties = node.get("properties", {})
            properties = properties if isinstance(properties, Mapping) else {}
            required = set(node.get("required", ()))
            cleaned: dict[str, Any] = {}
            for name, child in value.items():
                child_schema = properties.get(name)
                if (
                    child is None
                    and isinstance(child_schema, Mapping)
                    and name not in required
                    and not accepts_null(child_schema)
                ):
                    continue
                cleaned[name] = (
                    walk(child, child_schema)
                    if isinstance(child_schema, Mapping)
                    else child
                )
            return cleaned
        items = node.get("items")
        if isinstance(value, list) and isinstance(items, Mapping):
            return [walk(child, items) for child in value]
        return value

    def accepts_null(node: Mapping[str, Any]) -> bool:
        node = resolve(node)
        expected_type = node.get("type")
        if expected_type == "null" or (
            isinstance(expected_type, list) and "null" in expected_type
        ):
            return True
        if node.get("const", object()) is None:
            return True
        enum = node.get("enum")
        if isinstance(enum, list):
            return None in enum
        return expected_type is None and "const" not in node

    return walk(payload, schema)
