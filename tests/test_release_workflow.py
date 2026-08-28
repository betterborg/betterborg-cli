"""Nonpublishing contracts for synchronized protected publication."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import urllib.parse
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

LEAK_TEST_CREDENTIAL = "release/secret+value?12345"
LEAK_TEST_REPRESENTATIONS = (
    ("raw", LEAK_TEST_CREDENTIAL.encode()),
    (
        "percent-encoded",
        urllib.parse.quote(LEAK_TEST_CREDENTIAL, safe="").encode(),
    ),
    (
        "standard-base64-padded",
        base64.b64encode(LEAK_TEST_CREDENTIAL.encode()),
    ),
    (
        "standard-base64-unpadded",
        base64.b64encode(LEAK_TEST_CREDENTIAL.encode()).rstrip(b"="),
    ),
    (
        "urlsafe-base64-padded",
        base64.urlsafe_b64encode(LEAK_TEST_CREDENTIAL.encode()),
    ),
    (
        "urlsafe-base64-unpadded",
        base64.urlsafe_b64encode(LEAK_TEST_CREDENTIAL.encode()).rstrip(b"="),
    ),
)


class _RegistryResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._body = json.dumps(payload).encode()

    def __enter__(self) -> _RegistryResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _release_artifacts(directory: Path, version: str) -> dict[str, bytes]:
    distributions = {
        f"betterborg-{version}-py3-none-any.whl": b"fixture wheel bytes",
        f"betterborg-{version}.tar.gz": b"fixture source distribution bytes",
    }
    directory.mkdir()
    for filename, body in distributions.items():
        (directory / filename).write_bytes(body)
    return distributions


def _registry_payload(
    distributions: dict[str, bytes],
    *,
    digest_overrides: dict[str, str] | None = None,
) -> dict[str, object]:
    overrides = digest_overrides or {}
    return {
        "urls": [
            {
                "filename": filename,
                "digests": {
                    "sha256": overrides.get(filename, hashlib.sha256(body).hexdigest())
                },
            }
            for filename, body in distributions.items()
        ]
    }


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_release_is_manual_and_nonpublishing_by_default() -> None:
    workflow = _workflow_text()

    assert "workflow_dispatch:" in workflow
    assert re.search(r"publish:\n(?: {8}.*\n)*? {8}default: false", workflow)
    assert "pull_request:" not in workflow
    assert re.search(r"^  push:", workflow, re.MULTILINE) is None
    assert "Validate final tag and build release inputs once" in workflow
    assert '--tag "v$REVIEWED_VERSION"' in workflow
    assert "--greater-than 0.1.0" in workflow
    assert 'git rev-list -n 1 "v$REVIEWED_VERSION"' in workflow


def test_build_once_artifacts_feed_digest_gated_registry_order() -> None:
    workflow = _workflow_text()

    assert "needs: [validate-release]" in workflow
    assert "inputs.publish && github.ref == 'refs/heads/main'" in workflow
    assert "name: pypi" in workflow
    assert workflow.count("Build reviewed Python distributions once") == 1
    assert workflow.count("Build reviewed npm package once") == 1
    assert "betterborg-registry-inputs-${{ inputs.version }}" in workflow
    pypi = workflow.index("publish-pypi:")
    github = workflow.index("reconcile-github-release:")
    npm = workflow.index("publish-npm:")
    smokes = workflow.index("smoke-public-release:")
    assert pypi < github < npm < smokes
    assert "needs: [publish-pypi]" in workflow
    assert "needs: [reconcile-github-release]" in workflow
    assert "needs: [publish-npm]" in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow
    assert "--plan-publication" in workflow
    assert "steps.pypi-plan.outputs.action == 'publish'" in workflow
    assert "scripts/reconcile_npm_release.py" in workflow
    assert "steps.npm-plan.outputs.action == 'publish'" in workflow
    assert "packages-dir: dist/" in workflow
    assert "skip-existing: false" in workflow
    assert "password:" not in workflow
    assert "PYPI_API_TOKEN" not in workflow
    assert "NPM_TOKEN" not in workflow


def test_protected_smokes_follow_all_registries_with_one_provider() -> None:
    workflow = _workflow_text()
    script = (REPOSITORY_ROOT / "scripts/verify_public_installations.py").read_text(
        encoding="utf-8"
    )

    assert workflow.count("OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}") == 1
    assert "ANTHROPIC_API_KEY: ${{ secrets." not in workflow
    assert "f\"betterborg=={version}\"" in script
    assert 'f"@betterborg/cli@{version}"' in script
    assert "releases/download/v{version}/install.sh" in script
    assert '[*prefix, "init", "--yes", "--json"]' in script
    assert workflow.index("publish-npm:") < workflow.index("smoke-public-release:")


def test_public_smoke_uses_exact_commands_and_isolated_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential = "release-secret-value-12345"
    version = "1.2.3"
    artifacts = tmp_path / "artifacts"
    distributions = _release_artifacts(artifacts, version)
    calls: list[tuple[list[str], dict[str, str] | None]] = []
    requests = []

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

    def fake_urlopen(request, *, timeout):
        requests.append((request, timeout))
        return _RegistryResponse(_registry_payload(distributions))

    monkeypatch.setenv("OPENAI_API_KEY", credential)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-be-forwarded")
    monkeypatch.setattr(verify_pypi_release.subprocess, "run", fake_run)
    monkeypatch.setattr(verify_pypi_release.urllib.request, "urlopen", fake_urlopen)

    verify_pypi_release.verify_release(
        version,
        tmp_path / "fixture",
        artifacts,
        attempts=1,
        retry_delay=0,
    )

    assert len(requests) == 1
    request, timeout = requests[0]
    assert request.full_url == "https://pypi.org/pypi/betterborg/1.2.3/json"
    assert request.get_method() == "GET"
    assert timeout == 30
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
    version_environment = uvx_calls[0][1]
    init_environment = uvx_calls[1][1]
    assert version_environment is not None
    assert init_environment is not None
    assert "OPENAI_API_KEY" not in version_environment
    assert init_environment["OPENAI_API_KEY"] == credential
    for environment in (version_environment, init_environment):
        assert "ANTHROPIC_API_KEY" not in environment
        assert Path(environment["XDG_STATE_HOME"]).is_relative_to(
            tmp_path / "fixture"
        )
    assert all(
        not ({"publish", "upload", "twine"} & set(command))
        for command, _environment in calls
    )


def test_release_smoke_rejects_a_public_digest_mismatch_before_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    version = "1.2.3"
    artifacts = tmp_path / "artifacts"
    distributions = _release_artifacts(artifacts, version)
    wheel = f"betterborg-{version}-py3-none-any.whl"
    payload = _registry_payload(distributions, digest_overrides={wheel: "0" * 64})

    monkeypatch.setattr(
        verify_pypi_release.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _RegistryResponse(payload),
    )
    monkeypatch.setattr(
        verify_pypi_release.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "no command may run after an immutable digest mismatch"
        ),
    )

    with pytest.raises(
        verify_pypi_release.ReleaseVerificationError,
        match="digest mismatch.*immutable.*new version",
    ):
        verify_pypi_release.verify_release(
            version,
            tmp_path / "fixture",
            artifacts,
            attempts=1,
            retry_delay=0,
        )


def test_pypi_publication_plan_publishes_only_a_missing_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    _release_artifacts(artifacts, "1.2.3")
    monkeypatch.setattr(
        verify_pypi_release, "_public_distribution_digests", lambda _version: None
    )

    assert verify_pypi_release.publication_action("1.2.3", artifacts) == "publish"


def test_pypi_publication_plan_skips_only_matching_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    distributions = _release_artifacts(artifacts, "1.2.3")
    digests = {
        filename: hashlib.sha256(body).hexdigest()
        for filename, body in distributions.items()
    }
    monkeypatch.setattr(
        verify_pypi_release,
        "_public_distribution_digests",
        lambda _version: digests,
    )

    assert verify_pypi_release.publication_action("1.2.3", artifacts) == "skip"


def test_release_smoke_rejects_wrong_version_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    version = "1.2.3"
    artifacts = tmp_path / "artifacts"
    distributions = _release_artifacts(artifacts, version)

    def fake_run(command, **_kwargs):
        stdout = b"borg 1.2.2\n" if command[-1:] == ["version"] else b""
        return subprocess.CompletedProcess(command, 0, stdout, b"")

    monkeypatch.setenv("OPENAI_API_KEY", "release-secret-value-12345")
    monkeypatch.setattr(verify_pypi_release.subprocess, "run", fake_run)
    monkeypatch.setattr(
        verify_pypi_release.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _RegistryResponse(
            _registry_payload(distributions)
        ),
    )

    with pytest.raises(
        verify_pypi_release.ReleaseVerificationError,
        match="exact-version uvx check",
    ):
        verify_pypi_release.verify_release(
            version,
            tmp_path / "fixture",
            artifacts,
            attempts=1,
            retry_delay=0,
        )


def test_credential_markers_cover_every_distinct_representation() -> None:
    expected = {encoded for _name, encoded in LEAK_TEST_REPRESENTATIONS}

    assert len(expected) == 6
    assert (
        set(verify_pypi_release._credential_markers(LEAK_TEST_CREDENTIAL))
        == expected
    )


@pytest.mark.parametrize("location", ["stdout", "stderr", "fixture"])
@pytest.mark.parametrize(
    "encoded",
    [encoded for _name, encoded in LEAK_TEST_REPRESENTATIONS],
    ids=[name for name, _encoded in LEAK_TEST_REPRESENTATIONS],
)
def test_release_smoke_rejects_credential_leaks(
    tmp_path: Path,
    location: str,
    encoded: bytes,
) -> None:
    stdout = encoded if location == "stdout" else b""
    stderr = encoded if location == "stderr" else b""
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    if location == "fixture":
        (fixture / "state.bin").write_bytes(encoded)

    with pytest.raises(verify_pypi_release.ReleaseVerificationError) as raised:
        verify_pypi_release._assert_no_credential_leak(
            LEAK_TEST_CREDENTIAL,
            [verify_pypi_release.CommandCapture("test command", stdout, stderr)],
            (fixture,),
        )

    assert location in str(raised.value)
    assert LEAK_TEST_CREDENTIAL not in str(raised.value)


def test_runbook_pins_identity_authorization_redaction_and_recovery() -> None:
    runbook = " ".join(RUNBOOK.read_text(encoding="utf-8").split())

    for required in (
        "required reviewer",
        "dispatching operator must not approve their own deployment",
        "different required reviewer",
        "maintainer with push access",
        "token only read access",
        "--reviewed-sha REVIEWED_COMMIT_SHA",
        "remote tag no longer resolves to `REVIEWED_COMMIT_SHA`",
        "API presence alone cannot prove the attestation's signature",
        "`refs/heads/main`",
        "do not publish the draft while any listed attestation-verification",
        "PyPI project: `betterborg`",
        "GitHub owner: `betterborg`",
        "GitHub repository: `betterborg-cli`",
        "Workflow filename: `release.yml`",
        "GitHub environment: `pypi`",
        "OPENAI_API_KEY",
        "reviewed `vVERSION` tag",
        "Re-run failed jobs",
        "compares their SHA-256 digests",
        "existing publication is the reviewed release",
        "do not delete, replace, or retry with the same version",
        "new version",
    ):
        assert required in runbook
