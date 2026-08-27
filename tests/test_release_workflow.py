"""Nonpublishing contracts for the protected PyPI release path."""

from __future__ import annotations

import base64
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[1]
WORKFLOW = REPOSITORY_ROOT / ".github/workflows/release.yml"
RUNBOOK = REPOSITORY_ROOT / "docs/releasing.md"

_VERIFY_SPEC = importlib.util.spec_from_file_location(
    "verify_pypi_release",
    REPOSITORY_ROOT / "scripts/verify_pypi_release.py",
)
assert _VERIFY_SPEC is not None and _VERIFY_SPEC.loader is not None
verify_pypi_release = importlib.util.module_from_spec(_VERIFY_SPEC)
sys.modules[_VERIFY_SPEC.name] = verify_pypi_release
_VERIFY_SPEC.loader.exec_module(verify_pypi_release)


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_release_is_manual_and_nonpublishing_by_default() -> None:
    workflow = _workflow_text()

    assert "workflow_dispatch:" in workflow
    assert re.search(r"publish:\n(?: {8}.*\n)*? {8}default: false", workflow)
    assert "pull_request:" not in workflow
    assert re.search(r"^  push:", workflow, re.MULTILINE) is None
    assert "Validate release without publishing" in workflow
    assert "scripts/check_versions.py --expected" in workflow


def test_protected_job_depends_on_exact_reviewed_artifacts() -> None:
    workflow = _workflow_text()

    assert "needs: [validate-release]" in workflow
    assert "inputs.publish && github.ref == 'refs/heads/main'" in workflow
    assert "name: pypi" in workflow
    assert workflow.count("id-token: write") == 1
    assert workflow.index("id-token: write") > workflow.index("publish-pypi:")
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow
    assert "packages-dir: dist/" in workflow
    assert "skip-existing: false" in workflow
    assert "password:" not in workflow
    assert "PYPI_API_TOKEN" not in workflow
    for artifact in (
        "dist/betterborg-${{ inputs.version }}-py3-none-any.whl",
        "dist/betterborg-${{ inputs.version }}.tar.gz",
    ):
        assert workflow.count(artifact) == 1


def test_protected_smoke_has_one_provider_and_explicit_trust() -> None:
    workflow = _workflow_text()
    script = (REPOSITORY_ROOT / "scripts/verify_pypi_release.py").read_text(
        encoding="utf-8"
    )

    assert workflow.count("OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}") == 1
    assert "ANTHROPIC_API_KEY: ${{ secrets." not in workflow
    assert "f\"betterborg=={version}\"" in script
    assert '_uvx_command(version, "version")' in script
    assert '_uvx_command(version, "init", "--yes", "--json")' in script


def test_public_smoke_uses_exact_commands_and_isolated_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential = "release-secret-value-12345"
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    def fake_run(command, **kwargs):
        child_environment = kwargs.get("env")
        calls.append((command, child_environment))
        if command[-1:] == ["version"]:
            stdout = b"borg 1.2.3\n"
        elif command[-2:] == ["--yes", "--json"]:
            stdout = b'{"initialized": true}\n'
        else:
            stdout = b""
        return subprocess.CompletedProcess(command, 0, stdout, b"")

    monkeypatch.setenv("OPENAI_API_KEY", credential)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-be-forwarded")
    monkeypatch.setattr(verify_pypi_release.subprocess, "run", fake_run)

    verify_pypi_release.verify_release(
        "1.2.3", tmp_path / "fixture", attempts=1, retry_delay=0
    )

    uvx_calls = [call for call in calls if call[0][0] == "uvx"]
    assert [command for command, _environment in uvx_calls] == [
        ["uvx", "--refresh", "--from", "betterborg==1.2.3", "borg", "version"],
        [
            "uvx",
            "--refresh",
            "--from",
            "betterborg==1.2.3",
            "borg",
            "init",
            "--yes",
            "--json",
        ],
    ]
    for _command, environment in uvx_calls:
        assert environment is not None
        assert environment["OPENAI_API_KEY"] == credential
        assert "ANTHROPIC_API_KEY" not in environment
        assert Path(environment["XDG_STATE_HOME"]).is_relative_to(
            tmp_path / "fixture"
        )


@pytest.mark.parametrize("location", ["stdout", "stderr", "fixture"])
def test_release_smoke_rejects_credential_leaks(
    tmp_path: Path,
    location: str,
) -> None:
    credential = "release-secret-value-12345"
    encoded = base64.urlsafe_b64encode(credential.encode())
    stdout = encoded if location == "stdout" else b""
    stderr = encoded if location == "stderr" else b""
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    if location == "fixture":
        (fixture / "state.bin").write_bytes(encoded)

    with pytest.raises(verify_pypi_release.ReleaseVerificationError) as raised:
        verify_pypi_release._assert_no_credential_leak(
            credential,
            [verify_pypi_release.CommandCapture("test command", stdout, stderr)],
            (fixture,),
        )

    assert location in str(raised.value)
    assert credential not in str(raised.value)


def test_runbook_pins_identity_authorization_redaction_and_recovery() -> None:
    runbook = " ".join(RUNBOOK.read_text(encoding="utf-8").split())

    for required in (
        "required reviewer",
        "PyPI project: `betterborg`",
        "GitHub owner: `betterborg`",
        "GitHub repository: `betterborg-cli`",
        "Workflow filename: `release.yml`",
        "GitHub environment: `pypi`",
        "OPENAI_API_KEY",
        "do not delete, replace, or retry with the same version",
        "new version",
    ):
        assert required in runbook
