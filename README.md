# BetterBorg CLI

BetterBorg CLI is the open source, local-first command-line interface for
BetterBorg. This repository currently contains the public Python package
foundation; repository and service features will be added incrementally.

## Requirements

- Python 3.11 or newer
- Python's `venv` module
- Node.js 18 or newer (for npm launcher checks)

## Install and run

Install the latest release binary for Darwin or Linux, verify it, and activate
integrations for installed supported hosts:

```console
curl --proto '=https' --tlsv1.2 -fsSL \
  https://github.com/betterborg/betterborg-cli/releases/latest/download/install.sh \
  | sh
```

The installer selects ARM64 or x86_64 from the release manifest, verifies the
binary's SHA-256 digest and exact version before atomically replacing
`~/.local/bin/borg`, and only then runs `borg plugins install --all`. It prints
PATH guidance when `~/.local/bin` is not already visible. Native Windows and
WSL1 are unsupported; use a WSL2 shell. For another unsupported target, install
uv and run `uvx --from betterborg borg version` as a non-persistent fallback;
plugin activation still requires the persistent installer.

To install the CLI from a checkout and print its version instead:

```console
python -m pip install .
borg version
```

The Python distribution is named `betterborg`; it installs the `borg` console
command. Release versions are sourced from `betterborg_cli.__version__` and
checked against npm metadata plus the bundled Claude and Codex plugin and
marketplace manifests.

For development, create the locked environment and run the standard checks:

```console
make sync
make lint
make test
make build
make binary
```

`make build` creates the wheel and source distribution. `make binary` creates a
local one-file `dist/borg` executable with the same package assets; neither
target publishes an artifact.

Release maintainers use the protected, manual process in
[the release runbook](docs/releasing.md). Its default validation path does not
publish. The protected path verifies PyPI before building attested one-file
binaries for Darwin and Linux on ARM64 and x86_64.

`borg version` and `borg --help` are bootstrap commands. They do not create or
initialize a repository.

Install the user-scoped integrations for every available supported host after
`borg version` works from the host launch environment:

```console
borg plugins install
```

Use `--host claude` or `--host codex` to select one host; `--all` makes the
default explicit. The installer uses each host's supported marketplace and
plugin commands and does not edit host settings directly. Run
`/reload-plugins` in open Claude Code sessions after installation or an upgrade.

Before a command can load repository configuration or prompts for a
host-capable agent, trust the current Git workspace interactively:

```console
borg trust
```

For a deliberate noninteractive decision, use `borg trust --yes`. Trust is
bound to the resolved repository and Git common-directory paths and is stored
in the user's machine-local state directory, not in the repository.

Provider API agents use contained file tools that reject absolute paths,
traversal, and symlinks escaping the run directory. Analysis and planning
agents receive only those file tools. Coding, review, and merge agents may
also receive an argv-only command runner after workspace trust. Avoiding a
shell keeps metacharacters literal, but it is not a sandbox: programs invoked
by the command runner remain host-capable. Native Claude and Codex tools are
outside this API file-tool boundary.

After plan approval publishes a current task generation, inspect its execution
commitment with `borg task estimate <name>`. The estimate reports P50/P80 total
agent work, local sample sizes, and API versus subscription billing separately.
The bootstrap prior is prominently marked as dummy data and is gradually
replaced by repository-local completions; subscription work is never assigned
a fabricated USD value. Add `--json` for the machine-readable shape.

## Development safety

Follow [AGENTS.md](AGENTS.md) when using the private BetterBorg repository as a
reference. In particular, reference checkouts are read-only: all changes for
this package belong in this public repository.

## License

Copyright 2026 BetterBorg. Licensed under the
[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution.
