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

## Development safety

Follow [AGENTS.md](AGENTS.md) when using the private BetterBorg repository as a
reference. In particular, reference checkouts are read-only: all changes for
this package belong in this public repository.

## License

Copyright 2026 BetterBorg. Licensed under the
[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution.
