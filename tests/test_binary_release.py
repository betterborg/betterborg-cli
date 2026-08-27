"""Four-platform binary manifest and protected reconciliation contracts."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[1]
BINARY_WORKFLOW = REPOSITORY_ROOT / ".github/workflows/binary-release.yml"
RELEASE_WORKFLOW = REPOSITORY_ROOT / ".github/workflows/release.yml"
CI_WORKFLOW = REPOSITORY_ROOT / ".github/workflows/ci.yml"
BINARY_LOCK = REPOSITORY_ROOT / "requirements-binary.lock"


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
reconcile_github_release = _load_script("reconcile_github_release")


def _artifact_set(directory: Path, version: str = "1.2.3") -> dict[str, str]:
    directory.mkdir()
    for index, target in enumerate(release_artifacts.TARGETS, start=1):
        path = directory / target.filename
        path.write_bytes(f"binary fixture {index}\n".encode())
        release_artifacts.write_checksum(path)
    release_artifacts.write_manifest(
        version, directory, directory / "release-manifest.json"
    )
    return reconcile_github_release.expected_assets(version, directory)


def test_release_manifest_has_exact_stable_shape(tmp_path: Path) -> None:
    directory = tmp_path / "release"
    expected_assets = _artifact_set(directory)

    manifest = json.loads(
        (directory / "release-manifest.json").read_text(encoding="utf-8")
    )

    assert manifest == {
        "schema_version": 1,
        "version": "1.2.3",
        "artifacts": [
            {
                "filename": target.filename,
                "os": target.operating_system,
                "arch": target.architecture,
                "sha256": expected_assets[target.filename],
                "size": len(f"binary fixture {index}\n".encode()),
            }
            for index, target in enumerate(release_artifacts.TARGETS, start=1)
        ],
    }
    assert set(expected_assets) == {
        "borg-darwin-arm64",
        "borg-darwin-arm64.sha256",
        "borg-darwin-x86_64",
        "borg-darwin-x86_64.sha256",
        "borg-linux-arm64",
        "borg-linux-arm64.sha256",
        "borg-linux-x86_64",
        "borg-linux-x86_64.sha256",
        "release-manifest.json",
    }


def test_manifest_rejects_a_stale_checksum(tmp_path: Path) -> None:
    directory = tmp_path / "release"
    _artifact_set(directory)
    artifact = directory / "borg-linux-x86_64"
    artifact.write_bytes(b"changed after checksum")

    with pytest.raises(
        release_artifacts.ReleaseArtifactError,
        match="checksum sidecar does not match",
    ):
        release_artifacts.build_manifest("1.2.3", directory)


def test_matching_partial_draft_uploads_only_missing_assets(tmp_path: Path) -> None:
    local = _artifact_set(tmp_path / "release")
    existing_names = list(local)[:3]
    remote = reconcile_github_release.RemoteRelease(
        draft=True,
        assets={name: local[name] for name in existing_names},
    )

    plan = reconcile_github_release.plan_reconciliation(
        local, remote, publish=True
    )

    assert plan.create_draft is False
    assert plan.upload == tuple(name for name in local if name not in existing_names)
    assert plan.publish_draft is True


def test_mismatching_partial_release_blocks_before_mutation(tmp_path: Path) -> None:
    local = _artifact_set(tmp_path / "release")
    first_name = next(iter(local))
    remote = reconcile_github_release.RemoteRelease(
        draft=True, assets={first_name: "0" * 64}
    )

    with pytest.raises(
        reconcile_github_release.ReleaseReconciliationError,
        match="digest mismatch.*immutable.*new version",
    ):
        reconcile_github_release.plan_reconciliation(local, remote, publish=True)


def test_published_partial_release_is_never_modified(tmp_path: Path) -> None:
    local = _artifact_set(tmp_path / "release")
    first_name = next(iter(local))
    remote = reconcile_github_release.RemoteRelease(
        draft=False, assets={first_name: local[first_name]}
    )

    with pytest.raises(
        reconcile_github_release.ReleaseReconciliationError,
        match="published GitHub Release is partial.*new version",
    ):
        reconcile_github_release.plan_reconciliation(local, remote, publish=True)


def test_fixture_dry_run_never_calls_gh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "release"
    local = _artifact_set(directory)
    fixture = tmp_path / "remote.json"
    fixture.write_text(
        json.dumps({"draft": True, "assets": local}), encoding="utf-8"
    )
    monkeypatch.setattr(
        reconcile_github_release.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("fixture dry-run must not invoke gh"),
    )

    plan = reconcile_github_release.reconcile(
        "1.2.3", directory, "", publish=False, fixture=fixture
    )

    assert plan.upload == ()
    assert plan.publish_draft is False


def test_workflows_encode_four_native_targets_old_glibc_and_attestations() -> None:
    workflow = BINARY_WORKFLOW.read_text(encoding="utf-8")

    for artifact in (
        "borg-darwin-arm64",
        "borg-darwin-x86_64",
        "borg-linux-arm64",
        "borg-linux-x86_64",
    ):
        assert workflow.count(f"artifact: {artifact}") == 1
    for runner in (
        "macos-14",
        "macos-15-intel",
        "ubuntu-24.04-arm",
        "ubuntu-24.04",
    ):
        assert f"runner: {runner}" in workflow
    assert "manylinux2014_aarch64" in workflow
    assert "manylinux2014_x86_64" in workflow
    assert "--only-binary=:all:" in workflow
    assert workflow.count("--requirement requirements-binary.lock") == 2
    assert "--no-index" in workflow
    assert "--find-links /tmp/betterborg-wheels" in workflow
    assert "--requirement requirements-dev.lock" not in workflow
    assert 'test "$(getconf GNU_LIBC_VERSION)" = "glibc 2.17"' in workflow
    assert 'version)" = "borg $REVIEWED_VERSION"' in workflow
    assert "scripts/release_artifacts.py checksum" in workflow
    assert "scripts/release_artifacts.py manifest" in workflow
    assert workflow.count("uses: actions/attest@v4") == 2
    assert workflow.count("id-token: write") == 2
    assert workflow.count("attestations: write") == 2


def test_linux_binary_lock_selects_old_glibc_cryptography_wheel() -> None:
    lock = BINARY_LOCK.read_text(encoding="utf-8")

    assert "cryptography==50.0.0" in lock
    assert "cryptography==50.0.1" not in lock


def test_protected_ordering_and_nonpublishing_ci_path() -> None:
    release = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    ci = CI_WORKFLOW.read_text(encoding="utf-8")

    assert release.index("publish-pypi:") < release.index("pypi-verification-gate:")
    assert release.index("pypi-verification-gate:") < release.index("build-binaries:")
    assert release.index("build-binaries:") < release.index(
        "reconcile-github-release:"
    )
    assert "needs: [pypi-verification-gate]" in release
    assert "needs: [build-binaries]" in release
    assert "contents: write" in release
    assert re.search(
        r"reconcile-github-release:\n(?:.*\n)*?    if: \$\{\{ inputs.publish \}\}",
        release,
    )
    assert "uses: ./.github/workflows/binary-release.yml" in ci
    assert "attest: false" in ci
    assert "contents: write" not in ci
