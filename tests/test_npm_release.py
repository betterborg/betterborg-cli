"""Immutable npm package reconciliation contracts."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest
from release_test_support import load_script

reconcile_npm_release = load_script("reconcile_npm_release")


def _tarball(tmp_path: Path) -> Path:
    tarball = tmp_path / "betterborg-cli-1.2.3.tgz"
    tarball.write_bytes(b"reviewed npm fixture")
    return tarball


def _integrity(body: bytes) -> str:
    digest = base64.b64encode(hashlib.sha512(body).digest()).decode("ascii")
    return f"sha512-{digest}"


def _fixture(tmp_path: Path, integrity: str | None) -> Path:
    fixture = tmp_path / "registry.json"
    payload = None
    if integrity is not None:
        payload = {
            "name": "@betterborg/cli",
            "version": "1.2.3",
            "dist": {"integrity": integrity},
        }
    fixture.write_text(json.dumps(payload), encoding="utf-8")
    return fixture


def test_missing_npm_version_is_the_only_publish_plan(tmp_path: Path) -> None:
    tarball = _tarball(tmp_path)

    action = reconcile_npm_release.publication_action(
        "1.2.3", tarball, fixture=_fixture(tmp_path, None)
    )

    assert action == "publish"


def test_matching_npm_integrity_is_a_safe_resume_skip(tmp_path: Path) -> None:
    tarball = _tarball(tmp_path)
    fixture = _fixture(tmp_path, _integrity(tarball.read_bytes()))

    action = reconcile_npm_release.publication_action(
        "1.2.3", tarball, fixture=fixture
    )

    assert action == "skip"


def test_mismatching_npm_integrity_requires_a_new_version(tmp_path: Path) -> None:
    tarball = _tarball(tmp_path)
    fixture = _fixture(tmp_path, _integrity(b"different public bytes"))

    with pytest.raises(
        reconcile_npm_release.NpmReconciliationError,
        match="digest mismatch.*immutable.*new version",
    ):
        reconcile_npm_release.publication_action(
            "1.2.3", tarball, fixture=fixture
        )


def test_npm_fixture_cli_reports_action_without_registry_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tarball = _tarball(tmp_path)
    fixture = _fixture(tmp_path, _integrity(tarball.read_bytes()))
    output = tmp_path / "github-output"
    monkeypatch.setattr(
        reconcile_npm_release.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("fixture must not access npm"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "reconcile_npm_release.py",
            "--version",
            "1.2.3",
            "--tarball",
            str(tarball),
            "--fixture",
            str(fixture),
            "--github-output",
            str(output),
        ],
    )

    assert reconcile_npm_release.main() == 0
    assert output.read_text(encoding="utf-8") == "action=skip\n"
