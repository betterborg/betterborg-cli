"""Exact-version curl, uvx, and npx protected smoke contracts."""

from __future__ import annotations

import base64
import subprocess
import urllib.parse
from pathlib import Path

import pytest

from release_test_support import load_script

verify_public_installations = load_script("verify_public_installations")

CREDENTIAL = "release/secret+value?12345"


def test_public_command_shapes_pin_all_three_sources() -> None:
    commands = verify_public_installations.command_shapes("1.2.3")

    assert commands["curl"][-1] == (
        "https://github.com/betterborg/betterborg-cli/"
        "releases/download/v1.2.3/install.sh"
    )
    assert commands["uvx"] == (
        "uvx",
        "--refresh",
        "--from",
        "betterborg==1.2.3",
        "borg",
    )
    assert commands["npx"] == (
        "npx",
        "--yes",
        "@betterborg/cli@1.2.3",
    )


def test_three_fresh_fixtures_isolate_trust_provider_and_machine_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], Path, dict[str, str]]] = []

    def fake_run(command, **kwargs):
        environment = kwargs["env"]
        cwd = kwargs["cwd"]
        calls.append((command, cwd, environment))
        if command[-1:] == ["version"]:
            stdout = b"borg 1.2.3\n"
        elif command[-3:] == ["init", "--yes", "--json"]:
            stdout = b'{"initialized":true}\n'
        else:
            stdout = b""
        return subprocess.CompletedProcess(command, 0, stdout, b"")

    monkeypatch.setenv("OPENAI_API_KEY", CREDENTIAL)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-be-forwarded")

    verify_public_installations.verify_installations(
        "1.2.3",
        tmp_path / "fixtures",
        attempts=1,
        retry_delay=0,
        runner=fake_run,
    )

    init_calls = [call for call in calls if call[0][-3:] == ["init", "--yes", "--json"]]
    assert len(init_calls) == 3
    assert {cwd.name for _command, cwd, _environment in init_calls} == {
        "curl",
        "uvx",
        "npx",
    }
    for command, cwd, environment in calls:
        if command[-3:] == ["init", "--yes", "--json"]:
            assert environment["OPENAI_API_KEY"] == CREDENTIAL
        else:
            assert "OPENAI_API_KEY" not in environment
        assert "ANTHROPIC_API_KEY" not in environment
        assert Path(environment["XDG_STATE_HOME"]).is_relative_to(cwd)
    assert not any(
        {"publish", "upload"} & set(command)
        for command, _cwd, _environment in calls
    )


@pytest.mark.parametrize(
    "encoded",
    (
        CREDENTIAL.encode(),
        urllib.parse.quote(CREDENTIAL, safe="").encode(),
        base64.b64encode(CREDENTIAL.encode()),
        base64.b64encode(CREDENTIAL.encode()).rstrip(b"="),
        base64.urlsafe_b64encode(CREDENTIAL.encode()),
        base64.urlsafe_b64encode(CREDENTIAL.encode()).rstrip(b"="),
    ),
)
def test_public_smoke_rejects_every_credential_representation(
    tmp_path: Path, encoded: bytes
) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "state").write_bytes(encoded)

    with pytest.raises(
        verify_public_installations.PublicInstallationError,
        match="provider credential leaked",
    ):
        verify_public_installations._assert_no_credential_leak(
            CREDENTIAL, [], fixture
        )
