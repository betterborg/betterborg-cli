# PyPI release runbook

The `.github/workflows/release.yml` workflow is manual and defaults to a
nonpublishing validation run. Public upload requires the `publish` input, a
run from `main`, and approval of the protected `pypi` GitHub environment.
Ordinary pushes and pull requests never invoke this workflow.

## One-time protected setup

Only a BetterBorg release maintainer who is authorized as both a PyPI project
owner and a required reviewer of the GitHub `pypi` environment may approve a
release. Configure that environment with required reviewers, prevent
self-review and administrator bypass, and limit deployment branches to
`main`. Store one environment secret named `OPENAI_API_KEY`; do not add
`ANTHROPIC_API_KEY` or a PyPI password/token. The provider credential is used
only by the post-publication fixture smoke. PyPI authentication uses OIDC and
has no long-lived registry secret.

For the first release, a PyPI owner must create the `betterborg` project with a
pending trusted publisher (or add the publisher to the existing project) with
this exact identity:

- PyPI project: `betterborg`
- GitHub owner: `betterborg`
- GitHub repository: `betterborg-cli`
- Workflow filename: `release.yml`
- GitHub environment: `pypi`

The reviewer approving the first run must confirm those five fields on PyPI
and confirm that the workflow run's commit is the reviewed `main` commit.

## Release and verify

1. Update `betterborg_cli.__version__` and both bundled plugin manifests in one
   reviewed change. Run `make lint`, `make test`, and `make build`.
2. Dispatch **Release to PyPI** from `main`, enter that exact version, leave
   `publish` disabled, and require the nonpublishing validation job to pass.
3. Dispatch the same commit and version with `publish` enabled. An authorized
   reviewer checks the commit, version, artifact names, and trusted-publisher
   identity before approving the `pypi` environment deployment.
4. Require the protected job to finish. It publishes only the reviewed wheel
   and sdist, then runs `uvx --refresh --from 'betterborg==VERSION' borg
   version` and `borg init --yes --json` from that same exact distribution in
   a disposable, committed Git fixture.

The smoke child process receives exactly one provider variable,
`OPENAI_API_KEY`, and puts machine state inside the disposable fixture. The
verification captures stdout and stderr and recursively scans the fixture,
including Git and BetterBorg state, for raw, URL-encoded, standard-base64, and
URL-safe-base64 forms of the credential. GitHub log masking is defense in
depth; it is not the redaction assertion.

## Immutable-version recovery

PyPI versions and their artifact bytes are immutable. If an uploaded digest
does not match the reviewed artifact, a publish is partial, or verification
fails after any file reached PyPI, do not delete, replace, or retry with the
same version. Preserve the failed run and artifact digests for investigation,
increment the Python source and both plugin manifests to a new version, review
that commit, rerun the nonpublishing validation, and publish the new version.
