# Repository guidance

This is the public BetterBorg CLI repository. Keep implementation, tests,
documentation, generated metadata, and commits for the public CLI in this
checkout.

The private BetterBorg repository may be consulted only as read-only reference
material. Never edit files, create commits or branches, run formatters that
write files, or generate artifacts in a reference checkout. Do not copy
credentials, private paths, cloud-only imports, proprietary code, or other
unauthorized material into this repository.

Use the locked environment and the repository's standard commands:

- `make sync` installs the locked development environment.
- `make lint` runs static checks.
- `make test` runs the test suite and public-tree extraction scans.
- `make build` creates distribution artifacts.

New public files must remain compatible with the extraction scans in
`tests/test_public_tree.py`.
