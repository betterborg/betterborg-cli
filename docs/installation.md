# Install BetterBorg CLI

## Persistent GitHub installer

Darwin and Linux users on ARM64 or x86_64 can install the current release and
activate available Claude Code and Codex integrations with:

```console
curl --proto '=https' --tlsv1.2 -fsSL \
  https://github.com/betterborg/betterborg-cli/releases/latest/download/install.sh \
  | sh
```

The installer reads the GitHub-hosted release manifest, downloads the matching
binary, verifies its SHA-256 digest and exact version, atomically installs it at
`~/.local/bin/borg`, verifies the persistent executable, and only then runs
`borg plugins install --all`. Follow its PATH guidance and restart open shells
and plugin hosts. The vanity URL `install.betterborg.ai` is not active yet; do
not substitute it for the GitHub URL.

To verify or install one immutable release instead of following `latest`, pin
both the release URL and installer version:

```console
curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error \
  https://github.com/betterborg/betterborg-cli/releases/download/vVERSION/install.sh \
  --output install.sh
BETTERBORG_VERSION=VERSION sh ./install.sh
~/.local/bin/borg version
```

The final command must print exactly `borg VERSION`.

Native Windows and WSL1 are unsupported. On Windows, open a WSL2 shell and run
the Linux installer there. The CLI, machine state, and plugin hosts must all be
used from that WSL2 environment.

## Ephemeral uvx and npx commands

Use either public package wrapper without a persistent install:

```console
uvx --from betterborg borg version
npx --yes @betterborg/cli version
```

For an immutable invocation, pin the synchronized version:

```console
uvx --from betterborg==VERSION borg version
npx --yes @betterborg/cli@VERSION version
```

The npm package is a launcher, not a second CLI implementation. It resolves an
exact compatible `borg`, downloads and checksum-verifies the matching GitHub
binary when supported, or uses exact-version `uvx`. Native Windows has no
standalone binary, so the npm launcher requires `uvx` there; WSL2 remains the
supported persistent Windows installation path.

## Trust and provider setup

Run commands from a Git worktree. Trust is machine-local and bound to the
resolved worktree and Git common directory:

```console
borg trust
```

Use `borg trust --yes` only for a deliberate noninteractive decision. A fresh
noninteractive initialization can combine explicit trust with one provider:

```console
export OPENAI_API_KEY='your-provider-key'
borg init --yes
```

Set `ANTHROPIC_API_KEY` instead to use Anthropic. Interactive users may also
install and log in to the Claude Code or Codex CLI. Never write a provider key
to `.borg/config.toml`, a PRD, shell output, or any tracked file; use the shell
environment or a secret manager.

The same initialization is available through pinned wrappers:

```console
OPENAI_API_KEY='your-provider-key' \
  uvx --from betterborg==VERSION borg init --yes
OPENAI_API_KEY='your-provider-key' \
  npx --yes @betterborg/cli@VERSION init --yes
```

## Recovery

If checksum or staged-version verification fails, the installer does not
replace an existing `~/.local/bin/borg` and does not activate plugins. Retry
only after confirming the requested GitHub Release is complete. A plugin
activation failure happens after the verified CLI is installed; rerun
`~/.local/bin/borg plugins install --all` after fixing the host setup.

If the install directory is absent from PATH, apply the printed export and
restart terminals. On an unsupported target, install
[uv](https://docs.astral.sh/uv/) and use the `uvx` command above. For a bad or
digest-mismatched public release, do not bypass verification or reuse that
version; use the last known-good version while maintainers fix forward with a
new release.
