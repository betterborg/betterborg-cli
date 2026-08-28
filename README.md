# BetterBorg CLI

BetterBorg CLI is the open source, local-first command-line interface for
BetterBorg. This repository currently contains the public Python package
foundation; repository and service features will be added incrementally.

## Requirements

- Python 3.11 or newer
- Python's `venv` module

## Install and run

Install the CLI from a checkout and print its version:

```console
python -m pip install .
borg version
```

For development, create the locked environment and run the standard checks:

```console
make sync
make lint
make test
make build
```

`borg version` and `borg --help` are bootstrap commands. They do not create or
initialize a repository.

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

## Development safety

Follow [AGENTS.md](AGENTS.md) when using the private BetterBorg repository as a
reference. In particular, reference checkouts are read-only: all changes for
this package belong in this public repository.

## License

Copyright 2026 BetterBorg. Licensed under the
[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution.
