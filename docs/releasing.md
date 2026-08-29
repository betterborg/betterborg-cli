# PyPI and standalone binary release runbook

The `.github/workflows/release.yml` workflow is manual and defaults to a
nonpublishing validation run. Public upload requires the `publish` input, a
run from `main`, and approval of the protected `pypi` GitHub environment.
Ordinary pushes and pull requests never invoke this workflow; CI calls the
same reusable four-platform binary workflow with publication and attestations
disabled.

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

## Select the reviewed release

Update `betterborg_cli.__version__` and both bundled plugin manifests in one
reviewed change. Run `make lint`, `make test`, and `make build`, then create the
reviewed `vVERSION` tag on that exact commit. Verify that the tag is annotated
or signed according to the project's release policy, that its version matches
the source and artifact names, and that the tagged commit is the current tip of
`main`.

The workflow's publication guard accepts only `main`, so select `main` in the
Actions form, not the tag. Before each dispatch, compare the workflow run's SHA
with the peeled `vVERSION` tag SHA. Stop if `main` has moved; never publish a
different commit under the reviewed tag's version.

## Run or resume the merged workflow

1. After `.github/workflows/release.yml` is merged, open **Release BetterBorg** in
   GitHub Actions, choose **Run workflow**, select `main`, enter the exact
   reviewed version, leave `publish` disabled, and require **Validate release
   without publishing** to pass.
2. Dispatch the same tagged `main` commit and version with `publish` enabled.
   An authorized reviewer checks the run SHA, version, artifact names, and the
   five trusted-publisher identity fields before approving the protected
   `pypi` environment deployment.
3. Require the protected job to finish. It publishes only the reviewed wheel
   and sdist, compares their SHA-256 digests with the public PyPI metadata, then
   runs `uvx --refresh --from 'betterborg==VERSION' borg version` and `borg
   init --yes --json` from that exact distribution in a disposable, committed
   Git fixture.
4. The PyPI verification gate must succeed before any standalone build starts.
   The reusable build workflow then produces and version-smokes
   `borg-darwin-arm64`, `borg-darwin-x86_64`, `borg-linux-arm64`, and
   `borg-linux-x86_64`, writes one `.sha256` sidecar per binary, and generates
   `release-manifest.json`. Each protected binary/checksum pair and the
   manifest receives a GitHub artifact attestation.
5. The final job reads the existing `vVERSION` GitHub Release before changing
   it. It compares every existing asset byte-for-byte by SHA-256, uploads only
   missing assets to a matching draft, and publishes the complete draft. It
   never overwrites an asset. A complete matching published release is a
   successful resume and needs no mutation.

Linux binaries are built natively on `ubuntu-24.04` and
`ubuntu-24.04-arm` inside the architecture-matched PyPA `manylinux2014`
container. That documented old-glibc runner has glibc 2.17; do not replace it
with the host Python or a newer manylinux image without a compatibility review.
The build resolves `requirements-binary.lock` using wheels only before
installation, so it fails rather than compiling a dependency against an
incompatible glibc.
Darwin ARM64 uses `macos-14`, and Darwin x86_64 uses `macos-15-intel`; none of
the four deliverables is cross-compiled.

The reusable workflow grants `id-token: write` and `attestations: write` only
to its build and manifest jobs. The PyPI job is the only other OIDC consumer.
Only the final, publish-enabled reconciliation job receives `contents: write`;
the fixture validation path remains read-only and cannot create a GitHub
Release.

If a run is interrupted before the upload step starts, use **Re-run failed
jobs** on the same workflow run after confirming its SHA and inputs. If any
file may have reached PyPI, do not rerun the publish job. Download the preserved
`betterborg-VERSION` workflow artifact, place its wheel and sdist together in a
directory, and run the read-only verification from the reviewed checkout:

```console
python scripts/verify_pypi_release.py --version VERSION --artifacts PATH
```

Supply only `OPENAI_API_KEY` from the protected `pypi` environment. GitHub must
mask that secret, and the operator must confirm no unmasked value occurs in any
step output before accepting the run. Do not supply `ANTHROPIC_API_KEY`, a PyPI
password, or a PyPI token. The verification script only performs a GET of the
exact-version PyPI metadata and runs the public CLI; it has no upload path.

The smoke child process receives exactly one provider variable,
`OPENAI_API_KEY`, and puts machine state inside the disposable fixture. The
verification captures stdout and stderr and recursively scans the fixture,
including Git and BetterBorg state, for raw, URL-encoded, standard-base64, and
URL-safe-base64 forms of the credential. GitHub log masking is defense in
depth; it is not the redaction assertion.

## Immutable-version decision

PyPI versions and their artifact bytes are immutable. If the version already
exists, compare both public SHA-256 digests with the preserved reviewed wheel
and sdist by running the verification command above. Exact filename and digest
matches mean the existing publication is the reviewed release: finish the
version and init checks, record the successful verification, and do not upload
again.

If any public filename or digest differs, a publish is partial, or a verified
artifact contains a release defect, do not delete, replace, or retry with the
same version. Preserve the failed run and all local and public digests for
investigation, increment the Python source and both plugin manifests to a new
version, review and tag that commit, rerun the nonpublishing validation, and
publish the new version. A transient version or init check may be rerun only as
read-only verification after the artifact digests have matched; it never
authorizes another upload for the existing version.

## Binary recovery and rollback

For an interrupted binary build, use **Re-run failed jobs** only after the PyPI
digests have been verified again. Builds are replaceable workflow artifacts;
the GitHub Release assets are not. Before approving a resume, download the
`binary-release-VERSION` workflow artifact and confirm that it contains exactly
the four binaries, their four checksum sidecars, and `release-manifest.json`.
The manifest records `schema_version`, `version`, and, for each target, its
`filename`, `os`, `arch`, `sha256`, and byte `size`.

If reconciliation stopped after creating or partially filling a draft, rerun
the final job. Matching remote digests are retained and only absent assets are
uploaded. If any draft digest differs, delete nothing and overwrite nothing:
preserve the draft for investigation and prepare a new reviewed version. If a
published release is partial, it is also immutable and requires a new version.

After publication, verify a downloaded binary before use:

```console
sha256sum --check borg-OPERATING_SYSTEM-ARCHITECTURE.sha256
gh attestation verify borg-OPERATING_SYSTEM-ARCHITECTURE \
  --repo betterborg/betterborg-cli
./borg-OPERATING_SYSTEM-ARCHITECTURE version
```

On Darwin, use `shasum -a 256 -c` in place of `sha256sum --check`. The version
output must be exactly `borg VERSION`.

There is no destructive rollback for PyPI or GitHub Release bytes. For a
release defect, stop promoting the affected version, direct existing binary
users to the last known-good attested version, and fix forward with a new
reviewed version and tag. Do not delete assets, replace a tag, upload with
`--clobber`, or reuse the affected version. Record both versions and all
digests in the incident notes so the rollback decision remains auditable.
