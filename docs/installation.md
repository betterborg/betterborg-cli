# Install Betterborg CLI

## Persistent install

Install the published PyPI package to get a persistent `borg` command:

```console
pip install betterborg
borg version
```

The distribution is named `betterborg`; it installs the `borg` console command.
Requires Python 3.11 or newer. Any tool that installs a Python console script
works: `pipx install betterborg` and `uv tool install betterborg` both produce
the same persistent command.

A persistent install is required for host plugin activation. Activate the
Claude Code and Codex integrations once `borg version` works from the same
launch environment your host uses:

```console
borg plugins install --all
```

## Standalone binary

Install the latest release binary for Darwin or Linux without a Python
toolchain:

```console
curl --proto '=https' --tlsv1.2 -fsSL https://install.betterborg.ai | sh
```

`install.betterborg.ai` serves the installer from the latest GitHub Release.
The canonical release URL works the same way:

```console
curl --proto '=https' --tlsv1.2 -fsSL \
  https://github.com/betterborg/betterborg-cli/releases/latest/download/install.sh \
  | sh
```

The installer selects ARM64 or x86_64 from the release manifest, verifies the
binary's SHA-256 digest and exact version before atomically replacing
`~/.local/bin/borg`, and only then runs `borg plugins install --all`. It prints
PATH guidance when `~/.local/bin` is not already visible.

## Ephemeral wrappers

Use either public package wrapper without a persistent install:

```console
npx --yes @betterborg/cli version
uvx --from betterborg borg version
```

The npm package is a launcher, not a second CLI implementation. It resolves an
exact compatible `borg`, or uses exact-version `uvx`. Plugin activation
deliberately refuses to run from these wrappers: it resolves `borg` on the host
launch PATH and rejects transient extraction directories, so a plugin can never
point at a path that disappears.

## Version pinning

Pin the synchronized version whenever reproducibility matters:

```console
pip install betterborg==VERSION
uvx --from betterborg==VERSION borg version
npx --yes @betterborg/cli@VERSION version
```

`borg version` must print exactly `borg VERSION`.

## Platforms

macOS and Linux are supported. Native Windows and WSL1 are not; on Windows,
open a WSL2 shell and install there. The CLI, machine state, and plugin hosts
must all be used from that WSL2 environment.

## Trust and provider setup

Run commands from a Git worktree. Trust is machine-local and bound to the
resolved worktree and Git common directory:

```console
borg trust
```

Use `borg trust --yes` only for a deliberate noninteractive decision. A fresh
initialization needs one agent transport. Every command prefers the `claude`
or `codex` CLI whenever one is on `PATH`:

```console
borg init --yes
```

Selection tests only for the executable, so a CLI on `PATH` must already be
logged in. An installed but unauthenticated CLI is still chosen, and the run
fails rather than falling back to a provider key. With neither CLI on `PATH`,
supply a key instead:

```console
export OPENAI_API_KEY='your-provider-key'
borg init --yes
```

`ANTHROPIC_API_KEY` is the supported alternative. Never write a provider key
to `.borg/config.toml`, a PRD, shell output, or any tracked file; use the shell
environment or a secret manager.

## Recovery

If `borg` is installed but not found, confirm that the console-script directory
your installer used is on `PATH`, then restart open terminals and plugin hosts.

A plugin activation failure leaves the CLI installed and working. Fix the host
setup, then rerun `borg plugins install --all`. Run `/reload-plugins` in open
Claude Code sessions after installation or an upgrade, and start a new Codex
session when the installer requests it.

For a bad public release, do not work around it: install the last known-good
version with an explicit pin while maintainers fix forward with a new release.

## Not yet available

A Homebrew formula is planned and not published.
