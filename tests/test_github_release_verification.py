"""Read-only post-publication verification for standalone GitHub releases."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[1]


def _load_script(name: str):
    specification = importlib.util.spec_from_file_location(
        name, REPOSITORY_ROOT / "scripts" / f"{name}.py"
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


release_artifacts = _load_script("release_artifacts")
verify_github_release = _load_script("verify_github_release")


def _release_fixture(
    directory: Path,
    *,
    draft: bool = False,
    attested: set[str] | None = None,
) -> tuple[Path, tuple[str, ...]]:
    assets = directory / "assets"
    assets.mkdir(parents=True)
    for index, target in enumerate(release_artifacts.TARGETS, start=1):
        binary = assets / target.filename
        binary.write_bytes(f"binary fixture {index}\n".encode())
        release_artifacts.write_checksum(binary)
    release_artifacts.write_manifest(
        "1.2.3", assets, assets / "release-manifest.json"
    )
    names = verify_github_release._expected_names()
    metadata = {
        "tag_name": "v1.2.3",
        "draft": draft,
        "attestations": sorted(set(names) if attested is None else attested),
    }
    (directory / "release.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return directory, names


def _run_fixture(
    monkeypatch: pytest.MonkeyPatch,
    fixture: Path,
) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_github_release.py",
            "--version",
            "1.2.3",
            "--fixture",
            str(fixture),
        ],
    )
    monkeypatch.setattr(
        verify_github_release.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "fixture verification must not call GitHub or execute a mutation"
        ),
    )
    return verify_github_release.main()


def test_complete_release_fixture_verifies_every_asset_and_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture, _names = _release_fixture(tmp_path / "complete")

    result = _run_fixture(monkeypatch, fixture)

    captured = capsys.readouterr()
    assert result == 0
    assert captured.err == ""
    assert "9 assets and attestations" in captured.out


def test_partial_draft_fixture_reports_each_remaining_publication_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture, names = _release_fixture(tmp_path / "partial", draft=True)
    missing = "borg-linux-x86_64"
    (fixture / "assets" / missing).unlink()
    metadata_path = fixture / "release.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["attestations"] = [name for name in names if name != missing]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    result = _run_fixture(monkeypatch, fixture)

    captured = capsys.readouterr()
    assert result == 2
    assert captured.err == ""
    assert f"upload release asset {missing}" in captured.out
    assert f"publish GitHub artifact attestation for {missing}" in captured.out
    assert "publish the draft GitHub Release" in captured.out


@pytest.mark.parametrize("mismatch", ["checksum", "manifest"])
def test_digest_mismatch_fixture_is_terminal_and_requires_a_new_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mismatch: str,
) -> None:
    fixture, _names = _release_fixture(tmp_path / mismatch, draft=True)
    (fixture / "assets" / "borg-darwin-x86_64.sha256").unlink()
    if mismatch == "checksum":
        (fixture / "assets" / "release-manifest.json").unlink()
        (fixture / "assets" / "borg-linux-arm64").unlink()
        (fixture / "assets" / "borg-linux-arm64.sha256").write_text(
            "not a canonical checksum\n", encoding="utf-8"
        )
    else:
        manifest_path = fixture / "assets" / "release-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"][0]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = _run_fixture(monkeypatch, fixture)

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert "mismatch" in captured.err
    assert "immutable" in captured.err
    assert "new version" in captured.err


def test_published_partial_release_is_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture, _names = _release_fixture(tmp_path / "published-partial")
    (fixture / "assets" / "borg-darwin-arm64.sha256").unlink()

    result = _run_fixture(monkeypatch, fixture)

    captured = capsys.readouterr()
    assert result == 1
    assert "published GitHub Release is partial and immutable" in captured.err
    assert "prepare a new version" in captured.err


@pytest.mark.parametrize(
    ("missing", "draft"),
    [(None, False), ("borg-linux-x86_64", True)],
)
def test_live_verification_credits_only_cryptographically_verified_attestations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str | None,
    draft: bool,
) -> None:
    fixture, names = _release_fixture(tmp_path / "source")
    bodies = {
        name: (fixture / "assets" / name).read_bytes()
        for name in names
    }
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        release_api = "repos/betterborg/betterborg-cli/releases/tags/v1.2.3"
        if command[1:3] == ["api", release_api]:
            payload = {
                "tag_name": "v1.2.3",
                "draft": draft,
                "assets": [
                    {"name": name, "url": f"asset-api/{name}"}
                    for name in names
                    if name != missing
                ],
            }
            return verify_github_release.subprocess.CompletedProcess(
                command, 0, json.dumps(payload), ""
            )
        if command[1:2] == ["api"] and command[2].startswith("asset-api/"):
            name = command[2].removeprefix("asset-api/")
            return verify_github_release.subprocess.CompletedProcess(
                command, 0, bodies[name], b""
            )
        if command[1:2] == ["api"] and "/attestations/sha256:" in command[2]:
            return verify_github_release.subprocess.CompletedProcess(
                command, 0, json.dumps({"attestations": [{}]}), ""
            )
        if command[1:3] == ["attestation", "verify"]:
            return verify_github_release.subprocess.CompletedProcess(
                command, 0, "verified", ""
            )
        pytest.fail(f"unexpected command: {command}")

    monkeypatch.setattr(verify_github_release.subprocess, "run", fake_run)

    result = verify_github_release.verify_release(
        "1.2.3", "betterborg/betterborg-cli"
    )

    if missing is None:
        assert result.complete is True
        assert result.remaining == ()
    else:
        assert result.complete is False
        assert result.remaining == (
            f"upload release asset {missing}",
            f"publish GitHub artifact attestation for {missing}",
            "publish the draft GitHub Release",
        )
    attestation_queries = [
        command
        for command in commands
        if command[1:2] == ["api"]
        and "/attestations/sha256:" in command[2]
    ]
    assert len(attestation_queries) == 9
    verification_commands = [
        command
        for command in commands
        if command[1:3] == ["attestation", "verify"]
    ]
    expected_verifications = 8 if missing is not None else 9
    assert len(verification_commands) == expected_verifications
    assert all(
        command[1] in {"api", "attestation"}
        and not ({"create", "upload", "edit", "delete"} & set(command))
        for command in commands
    )
    assert all(
        command[-2:]
        == [
            "--signer-workflow",
            "betterborg/betterborg-cli/.github/workflows/binary-release.yml",
        ]
        for command in commands
        if command[1:3] == ["attestation", "verify"]
    )


def test_partial_live_release_rejects_wrong_attestation_signer_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, names = _release_fixture(tmp_path / "source")
    missing = "borg-linux-x86_64"
    rejected = "borg-darwin-arm64"
    bodies = {
        name: (fixture / "assets" / name).read_bytes()
        for name in names
    }

    def fake_run(command, **kwargs):
        release_api = "repos/betterborg/betterborg-cli/releases/tags/v1.2.3"
        if command[1:3] == ["api", release_api]:
            payload = {
                "tag_name": "v1.2.3",
                "draft": True,
                "assets": [
                    {"name": name, "url": f"asset-api/{name}"}
                    for name in names
                    if name != missing
                ],
            }
            return verify_github_release.subprocess.CompletedProcess(
                command, 0, json.dumps(payload), ""
            )
        if command[1:2] == ["api"] and command[2].startswith("asset-api/"):
            name = command[2].removeprefix("asset-api/")
            return verify_github_release.subprocess.CompletedProcess(
                command, 0, bodies[name], b""
            )
        if command[1:2] == ["api"] and "/attestations/sha256:" in command[2]:
            return verify_github_release.subprocess.CompletedProcess(
                command, 0, json.dumps({"attestations": [{}]}), ""
            )
        if command[1:3] == ["attestation", "verify"]:
            path = Path(command[3])
            return verify_github_release.subprocess.CompletedProcess(
                command,
                1 if path.name == rejected else 0,
                "" if path.name == rejected else "verified",
                "signer workflow mismatch" if path.name == rejected else "",
            )
        pytest.fail(f"unexpected command: {command}")

    monkeypatch.setattr(verify_github_release.subprocess, "run", fake_run)

    with pytest.raises(
        verify_github_release.GitHubReleaseVerificationError,
        match=f"attestation digest or provenance mismatch for {rejected}",
    ):
        verify_github_release.verify_release(
            "1.2.3", "betterborg/betterborg-cli"
        )
