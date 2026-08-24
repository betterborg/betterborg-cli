"""Render persisted repository analysis without changing scoring policy."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from betterborg_cli.repo_analysis.scoring import DIMENSIONS
from betterborg_cli.store import RepositoryAnalysis, RepositoryPackage

_MAX_SCORE = 5.0
_BAR_WIDTH = 10
_DIMENSION_LABELS = {
    "agent_guidance": "Agent guidance",
    "documentation": "Documentation",
    "testing": "Testing",
    "ci": "CI",
    "coding_standards": "Coding standards",
    "build_ergonomics": "Build ergonomics",
    "type_discipline": "Type discipline",
    "deployment": "Deployment",
}
_ESTIMATE_LABEL = "Estimated from bounded repository evidence."
_EFFORT_LABEL = "S/M/L are estimated effort labels."
_NON_DETERMINISM_LABEL = (
    "Analyzer output is non-deterministic and may vary between runs."
)
_MARKDOWN_SPECIALS = frozenset(r"\`*_{}[]<>#+!|")


def build_machine_report(
    analysis: RepositoryAnalysis,
    packages: Sequence[RepositoryPackage],
) -> dict[str, Any]:
    """Build the shared, JSON-compatible Harness Performance report shape."""
    ordered_packages = sorted(packages, key=lambda package: package.package_path)
    _validate_packages(analysis, ordered_packages)

    dimensions = [
        {
            "id": dimension,
            "label": _DIMENSION_LABELS[dimension],
            "score": _mean_dimension_score(ordered_packages, dimension),
            "max_score": _MAX_SCORE,
        }
        for dimension in DIMENSIONS
    ]
    previous_score = (
        analysis.overall_score - analysis.score_delta
        if analysis.score_delta is not None
        else None
    )

    return {
        "schema_version": 1,
        "analysis_id": str(analysis.id),
        "repository_id": str(analysis.repository_id),
        "head_sha": analysis.head_sha,
        "primary_language": analysis.primary_language,
        "is_monorepo": analysis.is_monorepo,
        "score": analysis.overall_score,
        "previous_score": previous_score,
        "delta": analysis.score_delta,
        "max_score": _MAX_SCORE,
        "dimensions": dimensions,
        "packages": [
            {
                "path": package.package_path,
                "name": package.package_name,
                "primary_language": package.primary_language,
                "score": package.overall_score,
                "max_score": _MAX_SCORE,
            }
            for package in ordered_packages
        ],
        "themes": _themes(analysis.analysis_json),
        "harness_impact": _harness_impact(analysis.analysis_json),
        "estimated": True,
        "non_deterministic": True,
        "labels": {
            "score": _ESTIMATE_LABEL,
            "effort": _EFFORT_LABEL,
            "non_determinism": _NON_DETERMINISM_LABEL,
        },
    }


def render_terminal_report(report: Mapping[str, Any]) -> str:
    """Render a compact plain-text report suitable for a terminal."""
    lines = ["Harness Performance", "===================", ""]
    lines.append(_score_line(report))
    lines.extend(
        [
            _terminal_text(report["labels"]["score"]),
            _terminal_text(report["labels"]["non_determinism"]),
            "",
            "Dimensions",
        ]
    )
    for dimension in report["dimensions"]:
        lines.append(
            f"  {_terminal_text(dimension['label']):<18} "
            f"[{_score_bar(float(dimension['score']))}] "
            f"{float(dimension['score']):.2f}/{int(dimension['max_score'])}"
        )

    lines.extend(["", "Packages"])
    for package in report["packages"]:
        lines.append(
            f"  {_terminal_text(package['path'])} "
            f"({_terminal_text(package['name'])}, "
            f"{_terminal_text(package['primary_language'])}): "
            f"{float(package['score']):.2f}/{int(package['max_score'])}"
        )

    lines.extend(["", "Ranked themes"])
    if report["themes"]:
        for theme in report["themes"]:
            lines.append(
                f"  {theme['rank']}. {_terminal_text(theme['title'])} — "
                f"effort {_terminal_text(theme['effort_label'])}; "
                "estimated impact "
                f"+{float(theme['estimated_impact']):.2f}"
            )
            lines.append(f"     {_terminal_text(theme['effort_rationale'])}")
    else:
        lines.append("  No recommendation themes were reported.")

    lines.extend(["", "Harness Impact"])
    for key in ("commands", "environment", "secrets", "services"):
        impact = report["harness_impact"][key]
        lines.append(
            f"  {key.title()}: {_terminal_text(impact['label'])} — "
            f"{_terminal_text(impact['summary'])}"
        )
        lines.extend(
            f"    - {_terminal_text(detail)}"
            for detail in _impact_details(key, impact)
        )
    lines.extend(["", _terminal_text(report["labels"]["effort"])])
    return "\n".join(lines) + "\n"


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render the shared report shape as Markdown."""
    lines = [
        "# Harness Performance",
        "",
        _score_line(report),
        "",
        f"> {_markdown_text(report['labels']['score'])} "
        f"{_markdown_text(report['labels']['non_determinism'])}",
        "",
        "## Dimensions",
        "",
        "| Dimension | Score |",
        "| --- | ---: |",
    ]
    for dimension in report["dimensions"]:
        lines.append(
            f"| {_markdown_text(dimension['label'])} | "
            f"`{_score_bar(float(dimension['score']))}` "
            f"{float(dimension['score']):.2f}/{int(dimension['max_score'])} |"
        )

    lines.extend(
        [
            "",
            "## Packages",
            "",
            "| Path | Package | Language | Score |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for package in report["packages"]:
        lines.append(
            f"| {_markdown_text(package['path'])} | "
            f"{_markdown_text(package['name'])} | "
            f"{_markdown_text(package['primary_language'])} | "
            f"{float(package['score']):.2f}/{int(package['max_score'])} |"
        )

    lines.extend(["", "## Ranked themes", ""])
    if report["themes"]:
        for theme in report["themes"]:
            lines.extend(
                [
                    f"{theme['rank']}. **{_markdown_text(theme['title'])}** — "
                    f"effort **{_markdown_text(theme['effort_label'])}**, "
                    "estimated impact "
                    f"**+{float(theme['estimated_impact']):.2f}**",
                    f"   - {_markdown_text(theme['effort_rationale'])}",
                ]
            )
    else:
        lines.append("No recommendation themes were reported.")

    lines.extend(["", "## Harness Impact", ""])
    for key in ("commands", "environment", "secrets", "services"):
        impact = report["harness_impact"][key]
        lines.append(
            f"- **{key.title()}:** {_markdown_text(impact['label'])} — "
            f"{_markdown_text(impact['summary'])}"
        )
        lines.extend(
            f"  - {_markdown_text(detail)}"
            for detail in _impact_details(key, impact)
        )
    lines.extend(["", f"_{_markdown_text(report['labels']['effort'])}_"])
    return "\n".join(lines) + "\n"


def render_json_report(report: Mapping[str, Any]) -> str:
    """Serialize the machine report deterministically."""
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def _validate_packages(
    analysis: RepositoryAnalysis, packages: Sequence[RepositoryPackage]
) -> None:
    if not packages:
        raise ValueError("a report requires at least one package")
    if any(
        package.repository_id != analysis.repository_id
        or package.analysis_id != analysis.id
        for package in packages
    ):
        raise ValueError("report packages must belong to the supplied analysis")

    persisted_packages = analysis.analysis_json.get("packages")
    if not isinstance(persisted_packages, list) or any(
        not isinstance(package, Mapping)
        or not isinstance(package.get("path"), str)
        for package in persisted_packages
    ):
        raise ValueError("analysis does not contain a valid persisted package list")

    expected_paths = [package["path"] for package in persisted_packages]
    if len(expected_paths) != len(set(expected_paths)):
        raise ValueError("analysis contains duplicate persisted package paths")

    supplied_paths = [package.package_path for package in packages]
    if len(supplied_paths) != len(set(supplied_paths)):
        raise ValueError("report packages must contain each package exactly once")
    if set(supplied_paths) != set(expected_paths):
        raise ValueError(
            "report packages must match the complete persisted package list"
        )


def _mean_dimension_score(
    packages: Sequence[RepositoryPackage], dimension: str
) -> float:
    values: list[float] = []
    for package in packages:
        cell = package.rubric.get(dimension)
        raw = cell.get("score", 0) if isinstance(cell, Mapping) else 0
        try:
            score = float(raw)
        except (TypeError, ValueError):
            score = 0.0
        values.append(max(0.0, min(_MAX_SCORE, score)))
    return sum(values) / len(values)


def _themes(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_themes = payload.get("themes")
    if not isinstance(raw_themes, list):
        return []
    themes: list[dict[str, Any]] = []
    for rank, raw in enumerate(raw_themes, start=1):
        if not isinstance(raw, Mapping):
            continue
        effort = str(raw.get("effort", "unknown"))
        themes.append(
            {
                "rank": rank,
                "id": str(raw.get("id", "")),
                "title": str(raw.get("title", "Untitled theme")),
                "effort": effort,
                "effort_label": f"{effort} (estimated)",
                "effort_rationale": str(
                    raw.get("effort_rationale", "No rationale was reported.")
                ),
                "estimated_impact": _number(raw.get("normalized_impact")),
                "ranking_score": _number(raw.get("ranking_score")),
                "recommendations": _recommendation_results(raw),
            }
        )
    return themes


def _recommendation_results(theme: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_results = theme.get("recommendations")
    if not isinstance(raw_results, list):
        return []
    return [dict(result) for result in raw_results if isinstance(result, Mapping)]


def _harness_impact(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "commands": _catalog_impact(payload),
        "environment": _environment_impact(payload),
        "secrets": _list_impact(
            payload,
            "required_secrets",
            detail_key="secrets",
            singular="required secret name",
            plural="required secret names",
        ),
        "services": _list_impact(
            payload,
            "service_dependencies",
            detail_key="services",
            singular="service dependency",
            plural="service dependencies",
        ),
    }


def _catalog_impact(payload: Mapping[str, Any]) -> dict[str, Any]:
    catalog = payload.get("command_catalog")
    if catalog is None:
        return {
            **_unknown_impact("No reliable command catalog was persisted."),
            "commands": [],
            "source": None,
            "notes": None,
        }
    items = catalog.get("commands") if isinstance(catalog, Mapping) else None
    if not isinstance(items, list):
        return {
            **_unknown_impact("The persisted command catalog was incomplete."),
            "commands": [],
            "source": None,
            "notes": None,
        }
    commands = [dict(item) for item in items if isinstance(item, Mapping)]
    return {
        **_count_impact(
            len(commands), "cataloged command", "cataloged commands"
        ),
        "commands": commands,
        "source": catalog.get("source"),
        "notes": catalog.get("notes"),
    }


def _environment_impact(payload: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "files",
        "toolchains",
        "package_managers",
        "prepare_commands",
        "materialize_commands",
    )
    if "environment" not in payload:
        return {
            **_unknown_impact("No reliable environment inputs were persisted."),
            **dict.fromkeys(fields, []),
        }
    environment = payload["environment"]
    if not isinstance(environment, Mapping):
        return {
            **_unknown_impact("The persisted environment inputs were incomplete."),
            **dict.fromkeys(fields, []),
        }
    details = {
        key: [dict(item) if isinstance(item, Mapping) else item for item in value]
        if isinstance((value := environment.get(key)), list)
        else []
        for key in fields
    }
    count = sum(len(details[key]) for key in fields)
    return {
        **_count_impact(count, "environment input", "environment inputs"),
        **details,
    }


def _list_impact(
    payload: Mapping[str, Any],
    key: str,
    *,
    detail_key: str,
    singular: str,
    plural: str,
) -> dict[str, Any]:
    if key not in payload:
        return {
            **_unknown_impact(f"No reliable {plural} were persisted."),
            detail_key: [],
        }
    values = payload[key]
    if not isinstance(values, list):
        return {
            **_unknown_impact(f"The persisted {plural} were incomplete."),
            detail_key: [],
        }
    records = [dict(value) for value in values if isinstance(value, Mapping)]
    return {
        **_count_impact(len(records), singular, plural),
        detail_key: records,
    }


def _count_impact(count: int, singular: str, plural: str) -> dict[str, str]:
    if count == 0:
        return {
            "status": "not_detected",
            "label": "Not detected",
            "summary": f"No {plural} were found in bounded evidence.",
        }
    noun = singular if count == 1 else plural
    return {
        "status": "detected",
        "label": "Detected",
        "summary": f"{count} {noun} persisted for harness use.",
    }


def _unknown_impact(summary: str) -> dict[str, str]:
    return {"status": "unknown", "label": "Unknown", "summary": summary}


def _impact_details(key: str, impact: Mapping[str, Any]) -> list[str]:
    if key == "commands":
        return [
            f"Command {_describe_command(command)}"
            for command in impact.get("commands", [])
            if isinstance(command, Mapping)
        ]
    if key == "environment":
        details = [
            f"Environment file: {path}" for path in impact.get("files", [])
        ]
        details.extend(
            f"Toolchain: {_describe_toolchain(toolchain)}"
            for toolchain in impact.get("toolchains", [])
            if isinstance(toolchain, Mapping)
        )
        details.extend(
            f"Package manager: {manager}"
            for manager in impact.get("package_managers", [])
        )
        details.extend(
            f"Prepare command {_describe_command(command, include_stage=False)}"
            for command in impact.get("prepare_commands", [])
            if isinstance(command, Mapping)
        )
        details.extend(
            "Materialize command "
            f"{_describe_command(command, include_stage=False)}"
            for command in impact.get("materialize_commands", [])
            if isinstance(command, Mapping)
        )
        return details
    if key == "secrets":
        return [
            f"Secret: {_describe_secret(secret)}"
            for secret in impact.get("secrets", [])
            if isinstance(secret, Mapping)
        ]
    if key == "services":
        return [
            f"Service: {_describe_service(service)}"
            for service in impact.get("services", [])
            if isinstance(service, Mapping)
        ]
    return []


def _describe_command(
    command: Mapping[str, Any], *, include_stage: bool = True
) -> str:
    stage = f"{command.get('stage', 'unnamed')}: " if include_stage else ""
    argv = command.get("argv")
    rendered_argv = (
        json.dumps(argv, ensure_ascii=False) if isinstance(argv, list) else "[]"
    )
    qualifiers: list[str] = []
    _append_qualifier(qualifiers, "cwd", command.get("cwd"))
    _append_qualifier(qualifiers, "source", command.get("source"))
    _append_values(qualifiers, "services", command.get("uses_services"))
    _append_values(qualifiers, "secrets", command.get("required_secrets"))
    suffix = f" ({'; '.join(qualifiers)})" if qualifiers else ""
    return f"{stage}{rendered_argv}{suffix}"


def _describe_toolchain(toolchain: Mapping[str, Any]) -> str:
    version = toolchain.get("version")
    return (
        f"{toolchain.get('name', 'unnamed')} {version}"
        if version is not None
        else str(toolchain.get("name", "unnamed"))
    )


def _describe_secret(secret: Mapping[str, Any]) -> str:
    qualifiers: list[str] = []
    _append_qualifier(qualifiers, "scope", secret.get("scope"))
    _append_values(qualifiers, "used by", secret.get("used_by"))
    _append_qualifier(qualifiers, "source", secret.get("source"))
    suffix = f" ({'; '.join(qualifiers)})" if qualifiers else ""
    return f"{secret.get('name', 'unnamed')}{suffix}"


def _describe_service(service: Mapping[str, Any]) -> str:
    qualifiers: list[str] = []
    _append_qualifier(qualifiers, "image", service.get("image"))
    _append_qualifier(qualifiers, "port", service.get("port"))
    ports = service.get("ports")
    if isinstance(ports, list):
        rendered_ports = [
            f"{port.get('port')}/{port.get('protocol', 'tcp')}"
            for port in ports
            if isinstance(port, Mapping)
        ]
        if rendered_ports:
            qualifiers.append(f"ports: {', '.join(rendered_ports)}")
    _append_qualifier(qualifiers, "compose service", service.get("compose_service"))
    _append_qualifier(qualifiers, "URL environment", service.get("url_env"))
    _append_qualifier(qualifiers, "source", service.get("source"))
    suffix = f" ({'; '.join(qualifiers)})" if qualifiers else ""
    return f"{service.get('name', 'unnamed')}{suffix}"


def _append_qualifier(values: list[str], label: str, value: object) -> None:
    if value is not None:
        values.append(f"{label}: {value}")


def _append_values(values: list[str], label: str, raw: object) -> None:
    if isinstance(raw, list) and raw:
        values.append(f"{label}: {', '.join(str(value) for value in raw)}")


def _score_line(report: Mapping[str, Any]) -> str:
    current = float(report["score"])
    previous = report["previous_score"]
    delta = report["delta"]
    line = f"Score: {current:.2f}/{int(report['max_score'])} (estimated)"
    if previous is not None and delta is not None:
        line += f" | Previous: {float(previous):.2f} | Delta: {float(delta):+.2f}"
    else:
        line += " | Previous: unavailable | Delta: unavailable"
    return line


def _score_bar(score: float) -> str:
    filled = round(max(0.0, min(_MAX_SCORE, score)) / _MAX_SCORE * _BAR_WIDTH)
    return "█" * filled + "░" * (_BAR_WIDTH - filled)


def _number(value: object) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


def _terminal_text(value: object) -> str:
    """Flatten text and remove characters that can control a terminal."""
    cleaned: list[str] = []
    for character in str(value):
        if character.isspace():
            cleaned.append(" ")
        elif not unicodedata.category(character).startswith("C"):
            cleaned.append(character)
    return " ".join("".join(cleaned).split())


def _markdown_text(value: object) -> str:
    """Render analyzer-controlled text as one escaped Markdown fragment."""
    return "".join(
        f"\\{character}" if character in _MARKDOWN_SPECIALS else character
        for character in _terminal_text(value)
    )
