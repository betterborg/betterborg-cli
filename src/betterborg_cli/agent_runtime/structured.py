"""Extract and validate structured agent results without provider policy."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


class StructuredResultError(ValueError):
    """Raised when agent output cannot produce a schema-valid JSON object."""


_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> dict[str, Any] | list[Any]:
    """Extract the first JSON object or array from plain or prose output."""
    candidates = list(_json_candidates(text))
    if not candidates:
        raise StructuredResultError("no parseable JSON payload found")
    return candidates[0]


def extract_structured_result(
    text: str,
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the first JSON object in ``text`` that satisfies ``schema``."""
    found_json = False
    errors: list[str] = []
    for candidate in _json_candidates(text):
        found_json = True
        if not isinstance(candidate, dict):
            errors.append("result must be a JSON object")
            continue
        try:
            validate_structured_result(candidate, schema)
        except StructuredResultError as error:
            errors.append(str(error))
            continue
        return candidate
    if not found_json:
        raise StructuredResultError("no parseable JSON payload found")
    detail = errors[-1] if errors else "no JSON object found"
    raise StructuredResultError(f"structured result validation failed: {detail}")


def extract_structured_result_file(
    source: Path,
    schema: Mapping[str, Any],
    *,
    destination: Path | None = None,
) -> dict[str, Any]:
    """Extract a result from ``source`` and optionally write canonical JSON."""
    payload = extract_structured_result(
        source.read_text(encoding="utf-8", errors="replace"), schema
    )
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return payload


def validate_structured_result(
    payload: Mapping[str, Any], schema: Mapping[str, Any]
) -> None:
    """Validate the JSON Schema features used by Betterborg result contracts.

    The runtime intentionally supports the deterministic validation keywords its
    own schemas use, avoiding a large runtime dependency for the CLI core.
    Unsupported schema keywords fail closed instead of silently weakening a
    result contract.
    """
    _validate_schema_shape(schema, path="$schema")
    _validate_value(payload, schema, path="$", root=schema)


def _json_candidates(text: str) -> Iterable[dict[str, Any] | list[Any]]:
    stripped = text.strip()
    if stripped:
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(value, dict | list):
                yield value
                return

    seen: set[tuple[int, int]] = set()
    for match in _FENCE.finditer(text):
        try:
            value = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict | list):
            yield value

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, length = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        marker = (index, index + length)
        if marker in seen or not isinstance(value, dict | list):
            continue
        seen.add(marker)
        yield value


_SUPPORTED_KEYWORDS = {
    "$schema",
    "$defs",
    "$ref",
    "additionalProperties",
    "allOf",
    "anyOf",
    "const",
    "default",
    "description",
    "enum",
    "items",
    "maxItems",
    "maxLength",
    "maximum",
    "minItems",
    "minLength",
    "minimum",
    "not",
    "oneOf",
    "pattern",
    "properties",
    "required",
    "title",
    "type",
    "uniqueItems",
}


def _validate_schema_shape(schema: Mapping[str, Any], *, path: str) -> None:
    unsupported = set(schema) - _SUPPORTED_KEYWORDS
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise StructuredResultError(f"{path}: unsupported schema keyword(s): {names}")
    expected_type = schema.get("type")
    if expected_type is not None and not (
        isinstance(expected_type, str)
        or (
            isinstance(expected_type, Sequence)
            and not isinstance(expected_type, str | bytes)
            and all(isinstance(item, str) for item in expected_type)
        )
    ):
        raise StructuredResultError(f"{path}.type must be a string or array")
    required = schema.get("required")
    if required is not None and not (
        isinstance(required, Sequence)
        and not isinstance(required, str | bytes)
        and all(isinstance(item, str) for item in required)
    ):
        raise StructuredResultError(f"{path}.required must be an array of strings")
    enum = schema.get("enum")
    if enum is not None and (
        not isinstance(enum, Sequence) or isinstance(enum, str | bytes)
    ):
        raise StructuredResultError(f"{path}.enum must be an array")
    pattern = schema.get("pattern")
    if pattern is not None and not isinstance(pattern, str):
        raise StructuredResultError(f"{path}.pattern must be a string")
    if pattern is not None:
        try:
            re.compile(pattern)
        except re.error as error:
            raise StructuredResultError(
                f"{path}.pattern is invalid: {error}"
            ) from error
    for keyword in ("properties", "$defs"):
        children = schema.get(keyword, {})
        if not isinstance(children, Mapping):
            raise StructuredResultError(f"{path}.{keyword} must be an object")
        for name, child in children.items():
            if not isinstance(child, Mapping):
                raise StructuredResultError(
                    f"{path}.{keyword}.{name} must be an object"
                )
            _validate_schema_shape(child, path=f"{path}.{keyword}.{name}")
    items = schema.get("items")
    if items is not None:
        if not isinstance(items, Mapping):
            raise StructuredResultError(f"{path}.items must be an object")
        _validate_schema_shape(items, path=f"{path}.items")
    additional = schema.get("additionalProperties")
    if additional is not None and not isinstance(additional, bool | Mapping):
        raise StructuredResultError(
            f"{path}.additionalProperties must be a boolean or object"
        )
    if isinstance(additional, Mapping):
        _validate_schema_shape(additional, path=f"{path}.additionalProperties")
    for keyword in ("allOf", "anyOf", "oneOf"):
        choices = schema.get(keyword)
        if choices is None:
            continue
        if not isinstance(choices, Sequence) or isinstance(choices, str | bytes):
            raise StructuredResultError(f"{path}.{keyword} must be an array")
        for index, child in enumerate(choices):
            if not isinstance(child, Mapping):
                raise StructuredResultError(
                    f"{path}.{keyword}[{index}] must be an object"
                )
            _validate_schema_shape(child, path=f"{path}.{keyword}[{index}]")
    negated = schema.get("not")
    if negated is not None:
        if not isinstance(negated, Mapping):
            raise StructuredResultError(f"{path}.not must be an object")
        _validate_schema_shape(negated, path=f"{path}.not")


def _validate_value(
    value: Any,
    schema: Mapping[str, Any],
    *,
    path: str,
    root: Mapping[str, Any],
) -> None:
    reference = schema.get("$ref")
    if reference is not None:
        _validate_value(
            value, _resolve_reference(reference, root), path=path, root=root
        )
        return

    for choice in schema.get("allOf", ()):
        _validate_value(value, choice, path=path, root=root)
    _validate_choices(value, schema, "anyOf", path=path, root=root, exactly_one=False)
    _validate_choices(value, schema, "oneOf", path=path, root=root, exactly_one=True)
    if "not" in schema and _matches(value, schema["not"], path=path, root=root):
        raise StructuredResultError(f"{path}: value matches forbidden schema")

    if "const" in schema and not _json_equal(value, schema["const"]):
        raise StructuredResultError(f"{path}: expected constant {schema['const']!r}")
    if "enum" in schema and not any(
        _json_equal(value, candidate) for candidate in schema["enum"]
    ):
        raise StructuredResultError(f"{path}: value is not in enum")

    expected = schema.get("type")
    if expected is not None:
        types = [expected] if isinstance(expected, str) else list(expected)
        if not any(_is_json_type(value, candidate) for candidate in types):
            raise StructuredResultError(
                f"{path}: expected {' or '.join(types)}, got {_type_name(value)}"
            )

    if isinstance(value, dict):
        _validate_object(value, schema, path=path, root=root)
    elif isinstance(value, list):
        _validate_array(value, schema, path=path, root=root)
    elif isinstance(value, str):
        _validate_string(value, schema, path=path)
    elif isinstance(value, int | float) and not isinstance(value, bool):
        _validate_number(value, schema, path=path)


def _validate_object(value, schema, *, path, root) -> None:
    properties = schema.get("properties", {})
    for required in schema.get("required", ()):
        if required not in value:
            raise StructuredResultError(
                f"{path}: missing required property {required!r}"
            )
    additional = schema.get("additionalProperties", True)
    for name, child in value.items():
        if name in properties:
            _validate_value(child, properties[name], path=f"{path}.{name}", root=root)
        elif additional is False:
            raise StructuredResultError(f"{path}: unexpected property {name!r}")
        elif isinstance(additional, Mapping):
            _validate_value(child, additional, path=f"{path}.{name}", root=root)


def _validate_array(value, schema, *, path, root) -> None:
    if len(value) < schema.get("minItems", 0):
        raise StructuredResultError(f"{path}: array has too few items")
    if "maxItems" in schema and len(value) > schema["maxItems"]:
        raise StructuredResultError(f"{path}: array has too many items")
    if schema.get("uniqueItems") and any(
        _json_equal(left, right)
        for index, left in enumerate(value)
        for right in value[index + 1 :]
    ):
        raise StructuredResultError(f"{path}: array items must be unique")
    if "items" in schema:
        for index, item in enumerate(value):
            _validate_value(item, schema["items"], path=f"{path}[{index}]", root=root)


def _validate_string(value, schema, *, path) -> None:
    if len(value) < schema.get("minLength", 0):
        raise StructuredResultError(f"{path}: string is too short")
    if "maxLength" in schema and len(value) > schema["maxLength"]:
        raise StructuredResultError(f"{path}: string is too long")
    if "pattern" in schema and re.search(schema["pattern"], value) is None:
        raise StructuredResultError(f"{path}: string does not match pattern")


def _validate_number(value, schema, *, path) -> None:
    if not math.isfinite(value):
        raise StructuredResultError(f"{path}: number must be finite")
    if "minimum" in schema and value < schema["minimum"]:
        raise StructuredResultError(f"{path}: number is below minimum")
    if "maximum" in schema and value > schema["maximum"]:
        raise StructuredResultError(f"{path}: number is above maximum")


def _validate_choices(value, schema, keyword, *, path, root, exactly_one) -> None:
    choices = schema.get(keyword)
    if choices is None:
        return
    matches = sum(_matches(value, choice, path=path, root=root) for choice in choices)
    if matches == 0 or (exactly_one and matches != 1):
        raise StructuredResultError(f"{path}: value does not satisfy {keyword}")


def _matches(value, schema, *, path, root) -> bool:
    try:
        _validate_value(value, schema, path=path, root=root)
    except StructuredResultError:
        return False
    return True


def _resolve_reference(reference: Any, root: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(reference, str) or not reference.startswith("#/"):
        raise StructuredResultError("only local JSON Schema references are supported")
    current: Any = root
    for raw_part in reference[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or part not in current:
            raise StructuredResultError(
                f"unresolved JSON Schema reference: {reference}"
            )
        current = current[part]
    if not isinstance(current, Mapping):
        raise StructuredResultError(
            f"JSON Schema reference is not an object: {reference}"
        )
    return current


def _is_json_type(value: Any, expected: str) -> bool:
    return {
        "array": lambda: isinstance(value, list),
        "boolean": lambda: isinstance(value, bool),
        "integer": lambda: (
            isinstance(value, int) or (isinstance(value, float) and value.is_integer())
        )
        and not isinstance(value, bool),
        "null": lambda: value is None,
        "number": lambda: isinstance(value, int | float)
        and not isinstance(value, bool),
        "object": lambda: isinstance(value, dict),
        "string": lambda: isinstance(value, str),
    }.get(expected, lambda: False)()


def _type_name(value: Any) -> str:
    for name in ("null", "boolean", "integer", "number", "string", "array", "object"):
        if _is_json_type(value, name):
            return name
    return type(value).__name__


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, int | float) and isinstance(right, int | float):
        return left == right
    return type(left) is type(right) and left == right
