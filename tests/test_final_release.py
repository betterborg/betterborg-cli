"""Fixture-driven contracts for synchronized post-publication verification."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from release_test_support import (
    REPOSITORY_ROOT,
    load_script,
    write_binary_artifact_set,
)

verify_final_release = load_script("verify_final_release")

VERSION = "1.2.3"
CREDENTIAL = "fixture-provider-credential-12345"


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    registry = root / "reviewed-registry"
    registry.mkdir(parents=True)
    distributions = {
        f"betterborg-{VERSION}-py3-none-any.whl": b"reviewed wheel",
        f"betterborg-{VERSION}.tar.gz": b"reviewed source distribution",
    }
    for name, body in distributions.items():
        (registry / name).write_bytes(body)
    tarball = registry / f"betterborg-cli-{VERSION}.tgz"
    tarball.write_bytes(b"reviewed npm tarball")

    reviewed_github = root / "reviewed-github"
    write_binary_artifact_set(reviewed_github, VERSION)

    fixture = root / "public"
    fixture.mkdir()
    (fixture / "pypi.json").write_text(
        json.dumps(
            {
                "urls": [
                    {
                        "filename": name,
                        "digests": {"sha256": hashlib.sha256(body).hexdigest()},
                    }
                    for name, body in distributions.items()
                ]
            }
        ),
        encoding="utf-8",
    )
    public_github = fixture / "github"
    shutil.copytree(reviewed_github, public_github / "assets")
    names = sorted(path.name for path in reviewed_github.iterdir())
    (public_github / "release.json").write_text(
        json.dumps(
            {
                "tag_name": f"v{VERSION}",
                "draft": False,
                "attestations": names,
            }
        ),
        encoding="utf-8",
    )
    integrity = verify_final_release.reconcile_npm_release.package_integrity(tarball)
    (fixture / "npm.json").write_text(
        json.dumps(
            {
                "name": "@betterborg/cli",
                "version": VERSION,
                "dist": {"integrity": integrity},
            }
        ),
        encoding="utf-8",
    )
    (fixture / "smoke.json").write_text(
        json.dumps({"credential": CREDENTIAL, "leak": None}), encoding="utf-8"
    )
    return registry, reviewed_github, fixture


def _deny_public_access(monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(*_args: object, **_kwargs: object) -> None:
        pytest.fail("a local fixture must not access a registry or GitHub")

    monkeypatch.setattr(
        verify_final_release.verify_pypi_release.urllib.request, "urlopen", denied
    )
    monkeypatch.setattr(
        verify_final_release.reconcile_npm_release.urllib.request, "urlopen", denied
    )
    monkeypatch.setattr(
        verify_final_release.verify_github_release.subprocess, "run", denied
    )


def test_complete_fixture_verifies_all_surfaces_and_three_init_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, github, fixture = _write_fixture(tmp_path)
    _deny_public_access(monkeypatch)

    result = verify_final_release.verify_final_release(
        VERSION,
        registry,
        github,
        fixture=fixture,
        attempts=1,
        retry_delay=0,
    )

    assert result == verify_final_release.VerificationResult(True, ())


@pytest.mark.parametrize("partial", ("pypi", "github", "npm"))
def test_partial_fixture_stops_at_the_first_ordered_publication_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    partial: str,
) -> None:
    registry, github, fixture = _write_fixture(tmp_path)
    if partial == "pypi":
        (fixture / "pypi.json").write_text("null\n", encoding="utf-8")
        shutil.rmtree(fixture / "github" / "assets")
        (fixture / "github" / "release.json").write_text("null\n", encoding="utf-8")
        (fixture / "npm.json").write_text("null\n", encoding="utf-8")
    elif partial == "github":
        missing = "borg-linux-x86_64"
        (fixture / "github" / "assets" / missing).unlink()
        metadata_path = fixture / "github" / "release.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["draft"] = True
        metadata["attestations"].remove(missing)
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        (fixture / "npm.json").write_text("null\n", encoding="utf-8")
    else:
        (fixture / "npm.json").write_text("null\n", encoding="utf-8")
    _deny_public_access(monkeypatch)

    result = verify_final_release.verify_final_release(
        VERSION, registry, github, fixture=fixture
    )

    assert result.complete is False
    joined = " ".join(result.remaining)
    assert partial in joined.casefold()


@pytest.mark.parametrize("surface", ("pypi", "github", "npm"))
def test_public_digest_mismatch_is_terminal_for_every_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
) -> None:
    registry, github, fixture = _write_fixture(tmp_path)
    if surface == "pypi":
        payload = json.loads((fixture / "pypi.json").read_text(encoding="utf-8"))
        payload["urls"][0]["digests"]["sha256"] = "0" * 64
        (fixture / "pypi.json").write_text(json.dumps(payload), encoding="utf-8")
    elif surface == "github":
        (fixture / "github" / "assets" / "borg-linux-arm64").write_bytes(
            b"different public bytes"
        )
    else:
        payload = json.loads((fixture / "npm.json").read_text(encoding="utf-8"))
        payload["dist"]["integrity"] = (
            verify_final_release.reconcile_npm_release.package_integrity(
                fixture / "github" / "assets" / "install.sh"
            )
        )
        (fixture / "npm.json").write_text(json.dumps(payload), encoding="utf-8")
    _deny_public_access(monkeypatch)

    with pytest.raises(
        verify_final_release.FinalReleaseVerificationError,
        match="mismatch.*immutable.*new version",
    ) as raised:
        verify_final_release.verify_final_release(
            VERSION, registry, github, fixture=fixture
        )

    assert surface in str(raised.value).casefold()


def test_out_of_order_publication_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, github, fixture = _write_fixture(tmp_path)
    (fixture / "pypi.json").write_text("null\n", encoding="utf-8")
    _deny_public_access(monkeypatch)

    with pytest.raises(
        verify_final_release.FinalReleaseVerificationError,
        match="publication order violation.*GitHub or npm.*before.*PyPI",
    ):
        verify_final_release.verify_final_release(
            VERSION, registry, github, fixture=fixture
        )


def test_integrated_fixture_rejects_a_provider_credential_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, github, fixture = _write_fixture(tmp_path)
    (fixture / "smoke.json").write_text(
        json.dumps({"credential": CREDENTIAL, "leak": "file"}), encoding="utf-8"
    )
    _deny_public_access(monkeypatch)

    with pytest.raises(
        verify_final_release.FinalReleaseVerificationError,
        match="credential leaked in fixture file",
    ) as raised:
        verify_final_release.verify_final_release(
            VERSION, registry, github, fixture=fixture
        )

    assert CREDENTIAL not in str(raised.value)


def test_public_install_and_command_docs_match_release_artifacts() -> None:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    installation = (REPOSITORY_ROOT / "docs/installation.md").read_text(
        encoding="utf-8"
    )
    commands = (REPOSITORY_ROOT / "docs/commands.md").read_text(encoding="utf-8")
    documentation = "\n".join((readme, installation, commands))

    for required in (
        "releases/latest/download/install.sh",
        "uvx --from betterborg",
        "npx --yes @betterborg/cli",
        "borg version",
        "borg trust",
        "borg init",
        "Darwin",
        "Linux",
        "ARM64",
        "x86_64",
    ):
        assert required in documentation
    assert "install.betterborg.ai" in readme
    assert "remains pending" in readme
    assert "install.betterborg.ai" in installation
    assert "not active yet" in installation
