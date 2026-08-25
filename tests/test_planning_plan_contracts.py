"""Deterministic Architect plan validation and Markdown rendering."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from betterborg_cli.planning import (
    PlanValidationError,
    render_plan_markdown,
    validate_plan,
    validate_plan_json,
)


@pytest.mark.parametrize(
    "name",
    [
        "1-setup",
        "01-Setup",
        "01-bad_name",
        "01--bad",
        "01-bad-",
        "01-" + "a" * 30,
    ],
)
def test_rejects_invalid_phase_names(repository: Path, name: str) -> None:
    plan = _plan()
    plan["phases"][0]["name"] = name

    with pytest.raises(PlanValidationError, match="phase|schema"):
        validate_plan(plan, repository)


@pytest.mark.parametrize(
    ("names", "message"),
    [
        (["01-foundation", "03-finish"], "expected number 02"),
        (["02-foundation", "03-finish"], "expected number 01"),
    ],
)
def test_requires_consecutive_phase_numbers(
    repository: Path, names: list[str], message: str
) -> None:
    plan = _two_phase_plan()
    for phase, name in zip(plan["phases"], names, strict=True):
        phase["name"] = name

    with pytest.raises(PlanValidationError, match=message):
        validate_plan(plan, repository)


@pytest.mark.parametrize(
    "dependency",
    ["02-finish", "03-future", "01-missing"],
)
def test_rejects_dependencies_that_are_not_earlier(
    repository: Path, dependency: str
) -> None:
    plan = _two_phase_plan()
    plan["phases"][0]["dependencies_on"] = [dependency]

    with pytest.raises(PlanValidationError, match="earlier phase"):
        validate_plan(plan, repository)


def test_accepts_an_earlier_phase_dependency(repository: Path) -> None:
    validate_plan(_two_phase_plan(), repository)


@pytest.mark.parametrize(
    ("path", "role", "message"),
    [
        ("missing.py", "modified", "not a repository file"),
        ("missing.py", "read", "not a repository file"),
        ("README.md", "new", "existing path"),
        ("../outside.py", "new", "repository-relative"),
        ("/tmp/outside.py", "new", "repository-relative"),
        ("src//module.py", "new", "repository-relative"),
        (r"src\\module.py", "new", "POSIX separators"),
    ],
)
def test_rejects_ungrounded_planned_paths(
    repository: Path, path: str, role: str, message: str
) -> None:
    plan = _plan()
    plan["phases"][0]["files_touched"] = [{"path": path, "role": role}]

    with pytest.raises(PlanValidationError, match=message):
        validate_plan(plan, repository)


def test_accepts_grounded_existing_and_new_paths(repository: Path) -> None:
    plan = _plan()
    plan["phases"][0]["files_touched"] = [
        {"path": "README.md", "role": "modified"},
        {"path": "src/new_module.py", "role": "new"},
    ]

    validate_plan(plan, repository)


def test_repository_qualified_paths_use_their_owning_roots(
    repository: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    secondary = tmp_path_factory.mktemp("secondary-repository")
    (secondary / "README.md").write_text("# Secondary\n", encoding="utf-8")
    plan = _multi_repository_plan()
    plan["phases"][0]["repositories"] = ["primary", "secondary"]
    plan["phases"][0]["files_touched"] = [
        {"path": "README.md", "role": "modified", "repo": "primary"},
        {"path": "README.md", "role": "modified", "repo": "secondary"},
    ]

    validate_plan(
        plan,
        repository,
        repository_roots={"primary": repository, "secondary": secondary},
    )


def test_does_not_ground_secondary_path_against_primary_repository(
    repository: Path,
) -> None:
    plan = _multi_repository_plan()

    with pytest.raises(PlanValidationError, match="no repository root was provided"):
        validate_plan(plan, repository)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda plan: plan["phases"][0]["files_touched"][0].pop("repo"),
            "must name its repository",
        ),
        (
            lambda plan: plan["phases"][0]["files_touched"][0].update(
                {"repo": "unknown"}
            ),
            "not declared by the plan",
        ),
        (
            lambda plan: plan["phases"][0]["contracts"][0].pop("repo"),
            "must name its repository",
        ),
        (
            lambda plan: plan["phases"][0]["contracts"][0].update(
                {"repo": "unknown"}
            ),
            "not declared by the plan",
        ),
        (
            lambda plan: plan["phases"][0].update({"repositories": ["unknown"]}),
            "not declared by the plan",
        ),
    ],
)
def test_rejects_ambiguous_or_unknown_repository_ownership(
    repository: Path, mutate, message: str
) -> None:
    plan = _multi_repository_plan()
    mutate(plan)
    secondary = repository / "secondary"
    secondary.mkdir()
    (secondary / "README.md").write_text("# Secondary\n", encoding="utf-8")

    with pytest.raises(PlanValidationError, match=message):
        validate_plan(
            plan,
            repository,
            repository_roots={"primary": repository, "secondary": secondary},
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda plan: plan.pop("summary"), "schema"),
        (
            lambda plan: plan["phases"][0].update({"acceptance_criteria": []}),
            "schema",
        ),
        (
            lambda plan: plan.update({"open_questions": ["Which platform?"]}),
            "open_questions",
        ),
        (lambda plan: plan.update({"title": " \t"}), "title"),
        (lambda plan: plan.update({"summary": "\n"}), "summary"),
        (
            lambda plan: plan.update({"overall_approach": " \n"}),
            "overall_approach",
        ),
        (
            lambda plan: plan["phases"][0].update({"title": " "}),
            r"phases\[0\]\.title",
        ),
        (
            lambda plan: plan["phases"][0].update({"goal": "\t"}),
            r"phases\[0\]\.goal",
        ),
        (
            lambda plan: plan["phases"][0].update({"technical_approach": "\n"}),
            r"phases\[0\]\.technical_approach",
        ),
        (
            lambda plan: plan["phases"][0].update({"test_strategy": " "}),
            r"phases\[0\]\.test_strategy",
        ),
        (
            lambda plan: plan["phases"][0].update(
                {"acceptance_criteria": [" "]}
            ),
            r"acceptance_criteria\[0\]",
        ),
        (
            lambda plan: plan["phases"][0].update({"deliverables": ["\t"]}),
            r"deliverables\[0\]",
        ),
        (
            lambda plan: plan.update({"open_questions": [" \n"]}),
            r"open_questions\[0\]",
        ),
    ],
)
def test_rejects_incomplete_plans(repository: Path, mutate, message: str) -> None:
    plan = _plan()
    mutate(plan)

    with pytest.raises(PlanValidationError, match=message):
        validate_plan(plan, repository)


@pytest.mark.parametrize("rendered", ["{not json", "[]", '"plan"'])
def test_rejects_invalid_rendered_json(repository: Path, rendered: str) -> None:
    with pytest.raises(PlanValidationError, match="JSON"):
        validate_plan_json(rendered, repository)


def test_validates_rendered_json(repository: Path) -> None:
    plan = _plan()
    assert validate_plan_json(json.dumps(plan), repository) == plan


def test_renders_portable_plan_markdown() -> None:
    plan = _two_phase_plan()
    plan["phases"][0].update(
        {
            "contracts": [{"kind": "config", "spec": "release.enabled: bool"}],
            "constraints": ["Keep the default disabled."],
            "risks": ["Configuration drift."],
        }
    )
    plan["risks"] = ["Release failures."]

    markdown = render_plan_markdown(plan)

    assert markdown.startswith("# Release workflow\n\n")
    assert "## Overall approach" in markdown
    assert "### 01-foundation \u2014 Establish foundation" in markdown
    assert "- `CHANGELOG.md` (new) \u2014 Documents releases." in markdown
    assert "- _config_ \u2014 release.enabled: bool" in markdown
    assert "**Dependencies:**\n- 01-foundation" in markdown
    assert "## Code pointers\n\n- `README.md` \u2014 Repository overview." in markdown
    assert markdown.endswith("\n")
    assert not markdown.endswith("\n\n")


def test_renders_repository_ownership_in_portable_markdown() -> None:
    plan = _multi_repository_plan()

    markdown = render_plan_markdown(plan)

    assert (
        "## Repositories\n\n- `primary` (primary)\n- `secondary` (secondary)"
        in markdown
    )
    assert "**Repositories:**\n- `secondary`" in markdown
    assert "- `README.md` (modified; repo: `secondary`)" in markdown
    assert "- _config_ — secondary.enabled: bool (repo: `secondary`)" in markdown


def test_renderer_flattens_and_escapes_model_controlled_markdown() -> None:
    plan = _plan()
    plan["title"] = "Release\n## Forged heading"
    plan["summary"] = "Summary\n- forged item\x1b"
    plan["phases"][0].update(
        {
            "title": "Use *safe* output",
            "goal": "Goal\n# forged goal",
            "files_touched": [
                {
                    "path": "docs/`release`.md",
                    "role": "new",
                    "description": "Docs\n- forged file",
                }
            ],
            "acceptance_criteria": ["Works\n## forged criterion"],
        }
    )
    plan["code_pointers"] = [
        {"path": "docs/`guide`.md", "why": "Guide\n# forged pointer"}
    ]

    markdown = render_plan_markdown(plan)

    assert markdown.startswith("# Release \\#\\# Forged heading\n\n")
    assert "Summary - forged item" in markdown
    assert "Use \\*safe\\* output" in markdown
    assert "Goal \\# forged goal" in markdown
    assert "- ``docs/`release`.md`` (new) — Docs - forged file" in markdown
    assert "- ``docs/`guide`.md`` — Guide \\# forged pointer" in markdown
    assert "Works \\#\\# forged criterion" in markdown
    assert "\x1b" not in markdown
    assert "\n## Forged heading" not in markdown
    assert "\n# forged" not in markdown


def test_renderer_tolerates_partial_legacy_plans() -> None:
    assert render_plan_markdown(None) == ""
    assert render_plan_markdown({}) == ""
    assert render_plan_markdown(
        {"title": "Legacy", "phases": ["bad", {"name": "old"}]}
    ) == "# Legacy\n\n## Phases\n\n### old\n"


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    (tmp_path / "README.md").write_text("# Repository\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    return tmp_path


def _plan() -> dict:
    return {
        "title": "Release workflow",
        "summary": "Add a tested release workflow.",
        "overall_approach": "Build on the repository's existing conventions.",
        "phases": [
            {
                "name": "01-foundation",
                "title": "Establish foundation",
                "goal": "Prepare the release path.",
                "technical_approach": "Add the release documentation first.",
                "files_touched": [
                    {
                        "path": "CHANGELOG.md",
                        "role": "new",
                        "description": "Documents releases.",
                    }
                ],
                "test_strategy": "Run the repository checks.",
                "acceptance_criteria": ["The release path is documented."],
                "deliverables": ["Release foundation"],
            }
        ],
        "code_pointers": [
            {"path": "README.md", "why": "Repository overview."}
        ],
        "risks": [],
        "open_questions": [],
    }


def _two_phase_plan() -> dict:
    plan = copy.deepcopy(_plan())
    plan["phases"].append(
        {
            "name": "02-finish",
            "title": "Finish workflow",
            "goal": "Ship the release path.",
            "technical_approach": "Wire the documented release behavior.",
            "files_touched": [{"path": "release.py", "role": "new"}],
            "test_strategy": "Exercise the release behavior.",
            "acceptance_criteria": ["The release path works."],
            "dependencies_on": ["01-foundation"],
            "deliverables": ["Release workflow"],
        }
    )
    return plan


def _multi_repository_plan() -> dict:
    plan = _plan()
    plan["repositories"] = [
        {"id": "primary", "role": "primary"},
        {"id": "secondary", "role": "secondary"},
    ]
    plan["phases"][0].update(
        {
            "repositories": ["secondary"],
            "files_touched": [
                {
                    "path": "README.md",
                    "role": "modified",
                    "repo": "secondary",
                }
            ],
            "contracts": [
                {
                    "kind": "config",
                    "spec": "secondary.enabled: bool",
                    "repo": "secondary",
                }
            ],
        }
    )
    return plan
