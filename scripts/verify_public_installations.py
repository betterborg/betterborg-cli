"""Smoke exact curl, uvx, and npx releases in isolated trusted repositories."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from scripts import protected_smoke
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root.
    import protected_smoke

INSTALL_URL = (
    "https://github.com/betterborg/betterborg-cli/"
    "releases/download/v{version}/install.sh"
)


PublicInstallationError = protected_smoke.ProtectedSmokeError
Runner = protected_smoke.Runner


def command_shapes(version: str) -> dict[str, tuple[str, ...]]:
    """Return the exact-version public commands exercised by the smoke."""
    return {
        "curl": (
            "curl",
            "--proto",
            "=https",
            "--tlsv1.2",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            INSTALL_URL.format(version=version),
        ),
        "uvx": (
            "uvx",
            "--refresh",
            "--from",
            f"betterborg=={version}",
            "borg",
        ),
        "npx": ("npx", "--yes", f"@betterborg/cli@{version}"),
    }


def _commands_for_installation(
    method: str,
    version: str,
    fixture: Path,
    environment: dict[str, str],
    captures: list[protected_smoke.CommandCapture],
    credential: str,
    runner: Runner,
) -> tuple[list[str], dict[str, str]]:
    shapes = command_shapes(version)
    if method == "uvx":
        return list(shapes[method]), environment
    if method == "npx":
        return list(shapes[method]), environment

    installer = fixture / "install.sh"
    download = [*shapes["curl"], "--output", str(installer)]
    completed = protected_smoke.run_command(
        runner,
        download,
        label="exact-version curl installer download",
        captures=captures,
        credential=credential,
        roots=(fixture,),
        cwd=fixture,
        env=environment,
    )
    if completed.returncode != 0:
        protected_smoke.fail("exact-version curl installer download failed")
    install_environment = dict(environment)
    install_environment["BETTERBORG_VERSION"] = version
    completed = protected_smoke.run_command(
        runner,
        ["sh", str(installer)],
        label="checksum-verifying curl installation",
        captures=captures,
        credential=credential,
        roots=(fixture,),
        cwd=fixture,
        env=install_environment,
    )
    if completed.returncode != 0:
        protected_smoke.fail("checksum-verifying curl installation failed")
    return [str(fixture / "home/.local/bin/borg")], environment


def verify_installations(
    version: str,
    root: Path,
    *,
    attempts: int = 6,
    retry_delay: float = 10.0,
    runner: Runner = subprocess.run,
) -> None:
    """Run each public source in a fresh repository with isolated machine state."""
    if attempts < 1:
        protected_smoke.fail("attempts must be at least one")
    credential, base_environment = protected_smoke.protected_environment()
    root.mkdir(parents=True, exist_ok=False)

    for method in ("curl", "uvx", "npx"):
        fixture = root / method
        captures: list[protected_smoke.CommandCapture] = []
        environment = dict(base_environment)
        environment.update(
            {
                "HOME": str(fixture / "home"),
                "NO_COLOR": "1",
                "XDG_CACHE_HOME": str(fixture / "cache"),
                "XDG_DATA_HOME": str(fixture / "data"),
                "XDG_STATE_HOME": str(fixture / "state"),
            }
        )
        protected_smoke.initialize_git_fixture(
            fixture, environment, captures, credential, runner
        )
        prefix, command_environment = _commands_for_installation(
            method, version, fixture, environment, captures, credential, runner
        )
        protected_smoke.verify_cli_initialization(
            prefix,
            method=method,
            version=version,
            fixture=fixture,
            environment=command_environment,
            captures=captures,
            credential=credential,
            runner=runner,
            attempts=attempts,
            retry_delay=retry_delay,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--retry-delay", type=float, default=10.0)
    arguments = parser.parse_args()
    try:
        with tempfile.TemporaryDirectory(
            prefix="betterborg-public-installations-"
        ) as temporary:
            verify_installations(
                arguments.version,
                Path(temporary) / "fixtures",
                attempts=arguments.attempts,
                retry_delay=arguments.retry_delay,
            )
    except PublicInstallationError as error:
        print(f"public installation verification failed: {error}", file=sys.stderr)
        return 1
    print(f"verified curl, uvx, and npx for BetterBorg {arguments.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
