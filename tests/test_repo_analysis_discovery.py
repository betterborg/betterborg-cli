"""Security and resource-boundary tests for repository discovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from betterborg_cli.repo_analysis.discovery import (
    DiscoveryLimits,
    build_discovery_workspace,
    discovery_limits_from_mapping,
)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


def _workspace_files(workspace: Path) -> list[str]:
    return sorted(
        path.relative_to(workspace).as_posix()
        for path in workspace.rglob("*")
        if path.is_file()
    )


def test_discovery_limits_parse_known_fields_and_ignore_unknown_fields() -> None:
    assert discovery_limits_from_mapping(None) == DiscoveryLimits()
    assert discovery_limits_from_mapping(
        {
            "per_file_bytes": 10,
            "total_bytes": 20,
            "deadline_seconds": 1.5,
            "untrusted_extra": "ignored",
        }
    ) == DiscoveryLimits(
        per_file_bytes=10,
        total_bytes=20,
        deadline_seconds=1.5,
    )


def test_discovery_rejects_file_and_directory_symlinks(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    workspace = tmp_path / "analysis-workspace"
    (repo / "README.md").write_text("# Real docs\n", encoding="utf-8")
    secret = tmp_path / "secret.md"
    secret.write_text("do not copy\n", encoding="utf-8")
    (repo / "LINK.md").symlink_to(secret)
    linked_docs = tmp_path / "linked-docs"
    linked_docs.mkdir()
    (linked_docs / "guide.md").write_text("external\n", encoding="utf-8")
    (repo / "docs-link").symlink_to(linked_docs, target_is_directory=True)

    manifest = build_discovery_workspace(repo, workspace)

    assert [file.path for file in manifest.files] == ["README.md"]
    symlink_omissions = {
        omission.path
        for omission in manifest.omitted
        if omission.reason == "symlink"
    }
    assert {"LINK.md", "docs-link"}.issubset(symlink_omissions)
    assert _workspace_files(workspace) == [
        "analysis_input.json",
        "files/README.md",
    ]


def test_discovery_enforces_per_file_cap_and_records_truncation(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    workspace = tmp_path / "analysis-workspace"
    (repo / "README.md").write_text("abcdefghij", encoding="utf-8")

    manifest = build_discovery_workspace(
        repo,
        workspace,
        limits=DiscoveryLimits(per_file_bytes=4, total_bytes=100),
    )

    assert len(manifest.files) == 1
    file = manifest.files[0]
    assert file.path == "README.md"
    assert file.copied_bytes == 4
    assert file.size_bytes == 10
    assert file.truncated is True
    assert file.truncation_reason == "per_file_byte_cap"
    assert (workspace / file.workspace_path).read_text(encoding="utf-8") == "abcd"


def test_discovery_enforces_total_byte_cap_and_records_omissions(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    workspace = tmp_path / "analysis-workspace"
    (repo / "go.mod").write_text("12345", encoding="utf-8")
    (repo / "package.json").write_text("67890", encoding="utf-8")
    (repo / "pyproject.toml").write_text("abcde", encoding="utf-8")

    manifest = build_discovery_workspace(
        repo,
        workspace,
        limits=DiscoveryLimits(per_file_bytes=10, total_bytes=8),
    )

    assert manifest.total_copied_bytes == 8
    assert sum(file.copied_bytes for file in manifest.files) == 8
    assert any(file.truncation_reason == "total_byte_cap" for file in manifest.files)
    assert any(omission.reason == "total_byte_cap" for omission in manifest.omitted)


def test_discovery_copies_only_allowlisted_evidence(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    workspace = tmp_path / "analysis-workspace"
    (repo / ".devcontainer").mkdir()
    (repo / ".github" / "workflows").mkdir(parents=True)
    (repo / "docs").mkdir()
    (repo / "scripts").mkdir()
    (repo / "service").mkdir()
    (repo / "src").mkdir()
    (repo / ".devcontainer" / "devcontainer.json").write_text(
        '{"image":"python:3.12"}', encoding="utf-8"
    )
    (repo / ".devcontainer.json").write_text(
        '{"image":"python:3.12"}', encoding="utf-8"
    )
    (repo / ".mise.toml").write_text('[tools]\npython = "3.12"\n')
    (repo / ".node-version").write_text("24\n")
    (repo / ".npmrc").write_text("registry=https://registry.npmjs.org/\n")
    (repo / ".nvmrc").write_text("24.14.0\n")
    (repo / ".yarnrc.yml").write_text("nodeLinker: node-modules\n")
    (repo / "SECURITY.md").write_text("policy")
    (repo / "README.py").write_text("print('not docs')\n")
    (repo / "package.json").write_text('{"scripts":{"build":"vite build"}}')
    (repo / "gradlew").write_text("#!/usr/bin/env sh\n")
    (repo / "mvnw").write_text("#!/usr/bin/env sh\n")
    (repo / "gradle" / "wrapper").mkdir(parents=True)
    (repo / "gradle" / "wrapper" / "gradle-wrapper.properties").write_text(
        "distributionUrl=https://services.gradle.org/distributions/gradle.zip\n"
    )
    (repo / ".mvn" / "wrapper").mkdir(parents=True)
    (repo / ".mvn" / "wrapper" / "maven-wrapper.properties").write_text(
        "distributionUrl=https://repo.maven.apache.org/maven.zip\n"
    )
    (repo / "service" / "gradlew").write_text("#!/usr/bin/env sh\n")
    (repo / "docs" / "guide.md").write_text("docs")
    (repo / "scripts" / "build.sh").write_text("#!/usr/bin/env bash\n")
    (repo / ".github" / "workflows" / "ci.yml").write_text("name: ci\n")
    (repo / "src" / "app.py").write_text("print('raw source')\n")
    (repo / "src" / "README.md").write_text("not root docs\n")
    (repo / "src" / ".devcontainer.json").write_text('{"image":"source"}\n')
    (repo / ".env").write_text("SECRET=value\n")

    manifest = build_discovery_workspace(repo, workspace)

    assert {file.path for file in manifest.files} == {
        ".devcontainer/devcontainer.json",
        ".devcontainer.json",
        ".github/workflows/ci.yml",
        ".mise.toml",
        ".node-version",
        ".npmrc",
        ".nvmrc",
        ".yarnrc.yml",
        ".mvn/wrapper/maven-wrapper.properties",
        "SECURITY.md",
        "docs/guide.md",
        "gradle/wrapper/gradle-wrapper.properties",
        "gradlew",
        "mvnw",
        "package.json",
        "scripts/build.sh",
        "service/gradlew",
    }
    workspace_files = set(_workspace_files(workspace))
    assert "files/README.py" not in workspace_files
    assert "files/src/app.py" not in workspace_files
    assert "files/src/README.md" not in workspace_files
    assert "files/src/.devcontainer.json" not in workspace_files
    assert "files/.env" not in workspace_files


def test_discovery_stops_on_monotonic_deadline(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    workspace = tmp_path / "analysis-workspace"
    (repo / "README.md").write_text("# docs\n", encoding="utf-8")
    calls = 0

    def clock() -> float:
        nonlocal calls
        calls += 1
        return 0.0 if calls == 1 else 2.0

    manifest = build_discovery_workspace(
        repo,
        workspace,
        limits=DiscoveryLimits(deadline_seconds=1.0),
        clock=clock,
    )

    assert manifest.deadline_exceeded is True
    assert manifest.files == []
    assert any(omission.reason == "deadline_exceeded" for omission in manifest.omitted)
    assert (workspace / "analysis_input.json").is_file()


def test_discovery_workspace_is_sanitized_and_manifest_matches_files(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    workspace = tmp_path / "analysis-workspace"
    workspace.mkdir()
    (workspace / "stale.txt").write_text("stale", encoding="utf-8")
    (repo / "README.md").write_text("# docs\n", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("print('nope')\n", encoding="utf-8")

    manifest = build_discovery_workspace(repo, workspace)

    assert _workspace_files(workspace) == [
        "analysis_input.json",
        "files/README.md",
    ]
    manifest_json = json.loads((workspace / "analysis_input.json").read_text())
    assert manifest_json["files"] == [file.__dict__ for file in manifest.files]
    assert manifest_json["files"][0]["workspace_path"] == "files/README.md"


def test_discovery_refuses_raw_repository_as_workspace(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    marker = repo / "README.md"
    marker.write_text("# Keep me\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must not be the repository root"):
        build_discovery_workspace(repo, repo)

    assert marker.read_text(encoding="utf-8") == "# Keep me\n"


def test_discovery_refuses_workspace_containing_repository(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    marker = repo / "README.md"
    marker.write_text("# Keep me\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must not contain the repository root"):
        build_discovery_workspace(repo, tmp_path)

    assert marker.read_text(encoding="utf-8") == "# Keep me\n"
