# Synchronized PyPI, GitHub, and npm release runbook

The `.github/workflows/release.yml` workflow is manual and defaults to a
nonpublishing validation run. Public upload requires the `publish` input, a
run from `main`, and approval of the protected `pypi` and `npm` GitHub
environments at their registry gates.
Ordinary pushes and pull requests never invoke this workflow; CI calls the
same reusable four-platform binary workflow with publication and attestations
disabled.

## One-time protected setup

Only a Betterborg release maintainer who is authorized as both a PyPI project
owner and a required reviewer of the GitHub `pypi` environment may approve a
release. Configure that environment with required reviewers, prevent
self-review and administrator bypass, and limit deployment branches to
`main`. It holds no secrets: PyPI authentication uses OIDC and has no
long-lived registry secret.

Configure a third environment named `smoke`, restricted to `main` with
administrator bypass prevented and no required reviewers. It holds one secret
named `OPENAI_API_KEY`; do not add `ANTHROPIC_API_KEY` or a PyPI
password/token. That credential is used only by the post-publication fixture
smoke, which runs after every registry is already immutable, so it gates
nothing that could be rolled back and does not need an approval.

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

Configure a second GitHub environment named `npm` with required reviewers,
prevent self-review and administrator bypass, and restrict deployment branches
to `main`. It has no secrets. An npm owner configures trusted publishing for
the public `@betterborg/cli` package with organization `betterborg`, repository
`betterborg-cli`, workflow filename `release.yml`, environment `npm`, and the
`npm publish` action allowed. npm trusted publishing requires Node 22.14 or
newer and npm 11.5.1 or newer; the workflow uses Node 24 and npm 11. Do not add
an `NPM_TOKEN`.

If `@betterborg/cli` does not exist yet, an npm owner must complete a separate,
reviewed lower-version package-claim checkpoint before configuring its trusted
publisher. Do not claim it with the reviewed release version, and do not
dispatch this workflow until the trusted publisher is active. Restrict
token-based publishing after OIDC succeeds.

## Select the reviewed release

Update `betterborg_cli.__version__`, `npm/package.json`, both bundled plugin
manifests, and both marketplace entries in one reviewed change. Run
`python scripts/check_versions.py --tag vVERSION`, `make lint`, `make test`,
and `make build`, then create the reviewed `vVERSION` tag on that exact commit.
A release version becomes immutable as soon as any registry accepts it: never
overwrite, reuse, move, or retag one.
Verify that the tag is annotated or signed according to the project's release
policy, that its version matches the source and artifact names, and that the
tagged commit is the current tip of `main`.

The workflow's publication guard accepts only `main`, so select `main` in the
Actions form, not the tag. Before each dispatch, compare the workflow run's SHA
with the peeled `vVERSION` tag SHA. Stop if `main` has moved; never publish a
different commit under the reviewed tag's version.

## Run or resume the merged workflow

1. After `.github/workflows/release.yml` is merged, open **Release Betterborg** in
   GitHub Actions, choose **Run workflow**, select `main`, enter the exact
   reviewed version, leave `publish` disabled, and require every dry-run job to
   pass. The run builds the wheel, sdist, npm tarball, four binaries, manifest,
   and installer once, then exercises PyPI, GitHub, npm, curl, uvx, npx, and
   credential fixtures without contacting a write API or mutating a registry.
2. Dispatch the same tagged `main` commit and version with `publish` enabled.
   An authorized reviewer checks the run SHA, version, artifact names, and the
   five trusted-publisher identity fields before approving the protected
   `pypi` environment deployment.
3. Require the PyPI gate to finish. Before uploading, it reads the exact public
   version metadata. A missing version is published from the preserved wheel
   and sdist; a complete matching version is skipped only after both SHA-256
   digests match. It compares their SHA-256 digests again after any upload.
4. The PyPI verification gate must succeed before any standalone build starts.
   The reusable build workflow then produces and version-smokes
   `betterborg-darwin-arm64`, `betterborg-darwin-x86_64`, `betterborg-linux-arm64`, and
   `betterborg-linux-x86_64`, writes one `.sha256` sidecar per binary, and generates
   `release-manifest.json`, and packages `install.sh` beside them. Each
   protected binary/checksum pair, the manifest, and the installer receives a
   GitHub artifact attestation.
5. The GitHub reconciliation job reads the existing `vVERSION` GitHub Release before changing
   it. It compares every existing asset byte-for-byte by SHA-256, uploads only
   missing assets to a matching draft, and publishes the complete draft. It
   never overwrites an asset. A complete matching published release is a
   successful resume and needs no mutation.
6. After GitHub is complete, approve the protected `npm` environment. The job
   compares the preserved npm tarball's SHA-512 integrity with
   `@betterborg/cli@VERSION`. It publishes only a missing version with npm OIDC,
   skips an exact match, and rejects a mismatch as immutable.
7. After all three public sources match, the final smoke runs unattended in
   the `smoke` environment. One provider credential runs fresh,
   isolated trusted initialization fixtures through the versioned GitHub curl
   installer, `uvx --from betterborg==VERSION`, and
   `npx @betterborg/cli@VERSION`. Each fixture scans stdout, stderr, paths,
   symlinks, and files for raw and encoded credential forms.

## Verify the synchronized public release

Retain the publish-enabled workflow run ID. The registry inputs exist as soon
as validation completes, so download them even when the protected run stops at
PyPI. Do not rebuild an input for verification:

```console
mkdir -p verified-registry-inputs
gh run download RUN_ID \
  --name betterborg-registry-inputs-VERSION \
  --dir verified-registry-inputs
git rev-parse HEAD
```

From the reviewed tagged checkout, provide the one masked provider credential
and run the read-only cross-surface verifier. `REVIEWED_COMMIT_SHA` is the
recorded full SHA from the final command above:

```console
export OPENAI_API_KEY='value supplied by the protected secret manager'
python scripts/verify_final_release.py \
  --version VERSION \
  --reviewed-sha REVIEWED_COMMIT_SHA \
  --registry-artifacts verified-registry-inputs
```

This first check can report a missing PyPI version or a not-yet-started GitHub
Release without the downstream binary artifact. Once the workflow has produced
`binary-release-VERSION`, download that artifact from the same run and include
it in every subsequent check:

```console
mkdir -p verified-github-assets
gh run download RUN_ID \
  --name binary-release-VERSION \
  --dir verified-github-assets
python scripts/verify_final_release.py \
  --version VERSION \
  --reviewed-sha REVIEWED_COMMIT_SHA \
  --registry-artifacts verified-registry-inputs \
  --github-artifacts verified-github-assets
```

The verifier first requires the preserved wheel and sdist SHA-256 digests to
match PyPI. It then downloads the GitHub Release assets read-only, verifies
their attestations and SHA-256 digests against the preserved build-once set,
and finally compares the preserved npm tarball's SHA-512 integrity with the
exact public npm version. This enforced PyPI → GitHub → npm order rejects a
later publication when an earlier gate is absent or partial. Only after all
three surfaces match does it run these exact final-version command shapes in
three fresh trusted repositories:

```console
curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error \
  https://github.com/betterborg/betterborg-cli/releases/download/vVERSION/install.sh \
  --output install.sh
BETTERBORG_VERSION=VERSION sh ./install.sh
~/.local/bin/betterborg version
uvx --refresh --from betterborg==VERSION betterborg version
npx --yes @betterborg/cli@VERSION version
```

Each path also runs `init --yes --json` with only `OPENAI_API_KEY`, never
`ANTHROPIC_API_KEY`, and scans captured output plus disposable HOME, cache,
data, state, paths, symlinks, and files for raw and encoded credential forms.
The version output from curl, uvx, and npx must be exactly `betterborg VERSION`.

Exit status `0` means every public digest and smoke matches. Exit status `2`
means publication is safely partial and lists only the next digest-gated step;
resume the same reviewed workflow run with **Re-run failed jobs**, then rerun
the verifier. Exit status `1` means verification could not safely determine a
resumable state. Inspect its error: only an error that explicitly identifies
immutable public state and says to prepare a new version is terminal. Retry
transient registry, GitHub, credential, tooling, and local-input failures after
correcting the reported cause. The verifier has no registry or GitHub
publication command and must never be given write-scoped credentials.

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
to its build and manifest jobs. The PyPI and npm publishing jobs are the only
other OIDC consumers. Only the final, publish-enabled GitHub reconciliation
job receives `contents: write`; the fixture validation path remains read-only
and cannot create a GitHub Release.

## Authorize and verify the binary publication

The binary publication is authorized only through the reviewed workflow run.
Two authorized Betterborg release maintainers must participate: a dispatching
operator who is allowed to run Actions on `betterborg/betterborg-cli`, and a
different required reviewer for the protected `pypi` environment. Prevent
self-review remains enabled, so the dispatching operator must not approve
their own deployment. Before the reviewer approves, both maintainers confirm
that `vVERSION` is the reviewed tag, that its peeled commit is the run SHA on
`main`, that the nonpublishing run passed, and that the requested version
matches the tag and source. Authenticate `gh` as a maintainer with push access
to the public repository so the post-publish verifier can see an interrupted
draft, but give its token only read access. If a fine-grained token is used,
grant repository contents read and attestations read, but no write permission.
Do not use an administrator bypass or a personal token with release-write
scope for verification.

To start the binary path, the dispatching operator runs **Release Betterborg**
for that exact reviewed version with `publish` enabled. The different required
reviewer independently checks the run SHA, tag, version, inputs, and trusted
publisher identity, then approves the protected environment. To resume an
interrupted run, the dispatching operator uses **Re-run failed jobs** on the
same run only after both maintainers reconfirm its inputs, SHA, tag, and any
assets already visible on the draft; the different required reviewer approves
the protected deployment if GitHub requests approval again. Do not dispatch a
newer `main` commit for the old version. The PyPI gate must remain successful
before the four-platform build and the final GitHub Release reconciliation
run.

After the protected workflow succeeds, run the read-only GitHub verification from the
reviewed checkout. First record that checkout's full commit SHA; this trusted
value lets the verifier reject a tag that has moved since review:

```console
git rev-parse HEAD
python scripts/verify_github_release.py \
  --version VERSION \
  --repository betterborg/betterborg-cli \
  --reviewed-sha REVIEWED_COMMIT_SHA
```

The command downloads the ten expected assets through `gh api`: the four
binaries, their four `.sha256` sidecars, `release-manifest.json`, and
`install.sh`. It checks
the manifest's recorded version, target metadata, sizes, and binary digests;
checks every checksum sidecar; rejects the release if the remote tag no longer
resolves to `REVIEWED_COMMIT_SHA`; and verifies the provenance of every asset
with `gh attestation verify`, pinned to
`betterborg/betterborg-cli/.github/workflows/binary-release.yml`, the peeled
`vVERSION` commit (which must equal the supplied reviewed SHA), and
`refs/heads/main`. It performs no create, upload, edit, delete, or overwrite
operation. Exit status `0` means
the published release is complete, `2` means a draft or its attestations are
partial and the output lists the publication steps that remain, and `1` means
verification is terminal or could not establish trust.

For exit status `2`, perform only the listed steps, in order, through the same
reviewed workflow run. Rerun verification after satisfying an upload or
attestation-publication prerequisite; do not publish the draft while any
listed attestation-verification step remains. For a missing asset, API presence
alone cannot prove the attestation's signature or provenance without the
subject bytes, so the verifier conservatively keeps both its provenance
publication and verification steps open. The protected workflow may satisfy
the publication step by retaining provenance that it establishes is already
the expected record, but the verifier must then cryptographically verify that
record against the uploaded asset bytes and reviewed source before draft
publication. If the release is already public but an attestation is missing,
stop promotion and have a release maintainer complete the attestation for the
unchanged digest through the protected release process. Never replace an asset
to repair an attestation.

Release assets are immutable. Matching names and SHA-256 digests are verified
and retained; a digest, checksum, manifest, tag, or attestation-provenance
mismatch is terminal for that version. Do not delete an asset, use `--clobber`,
move the tag, or rerun publication in an attempt to replace bytes. Preserve the
run and observed digests for investigation, increment to a new reviewed
version, and repeat the validation and protected publication path. A published
release missing any of the ten assets is likewise terminal and requires a new
version.

If a run is interrupted, use **Re-run failed jobs** on the same workflow run
after confirming its SHA and inputs. Each registry job reads public metadata
before its mutation step: missing bytes may be published, exact matching bytes
are skipped, and mismatches stop the release. Download the preserved
`betterborg-registry-inputs-VERSION` workflow artifact and run the read-only
PyPI verification from the reviewed checkout:

```console
python scripts/verify_pypi_release.py \
  --version VERSION \
  --artifacts PATH \
  --artifacts-only
```

This verification needs no credential and has no upload path. For the final
three-source smoke, supply only `OPENAI_API_KEY` from the protected `pypi`
environment. GitHub must mask that secret, and the operator must confirm no
unmasked value occurs in any step output before accepting the run. Do not
supply `ANTHROPIC_API_KEY`, a PyPI password, a PyPI token, or an npm token.

Each curl, uvx, and npx init child receives exactly one provider variable,
`OPENAI_API_KEY`, and puts HOME, cache, data, and state inside its own
disposable fixture. The verification captures stdout and stderr and recursively
scans each fixture, including Git and Betterborg state, for raw, URL-encoded,
standard-base64, and URL-safe-base64 forms of the credential. GitHub log
masking is defense in depth; it is not the redaction assertion.

## Immutable-version decision

PyPI and npm versions and their artifact bytes are immutable. If the PyPI version already
exists, compare both public SHA-256 digests with the preserved reviewed wheel
and sdist by running the verification command above. Exact filename and digest
matches mean the existing publication is the reviewed release: finish the
version and init checks, record the successful verification, and do not upload
again.

If any public filename or digest differs, a publish is partial, or a verified
artifact contains a release defect, do not delete, replace, or retry with the
same version. Preserve the failed run and all local and public digests for
investigation, increment all six version-bearing sources to a new version,
review and tag that commit, rerun the nonpublishing validation, and publish the
new version. A transient version or init check may be rerun only as read-only
verification after the artifact digests have matched; it never authorizes
another upload for the existing version.

Apply the same decision to `@betterborg/cli@VERSION`: compare the preserved
tarball's SHA-512 integrity with the exact public npm metadata. A match is a
successful skip. A missing version may be published only after GitHub assets
are complete. A different integrity value, partial npm state, or package defect
requires a new synchronized version; never unpublish and reuse the version.

## Binary recovery and rollback

For an interrupted binary build, use **Re-run failed jobs** only after the PyPI
digests have been verified again. Builds are replaceable workflow artifacts;
the GitHub Release assets are not. Before approving a resume, download the
`binary-release-VERSION` workflow artifact and confirm that it contains exactly
the four binaries, their four checksum sidecars, `release-manifest.json`, and
`install.sh`.
The manifest records `schema_version`, `version`, and, for each target, its
`filename`, `os`, `arch`, `sha256`, and byte `size`.

If reconciliation stopped after creating or partially filling a draft, rerun
the GitHub reconciliation job. Matching remote digests are retained and only absent assets are
uploaded. If any draft digest differs, delete nothing and overwrite nothing:
preserve the draft for investigation and prepare a new reviewed version. If a
published release is partial, it is also immutable and requires a new version.

After publication, verify a downloaded binary before use:

```console
sha256sum --check betterborg-OPERATING_SYSTEM-ARCHITECTURE.sha256
gh attestation verify betterborg-OPERATING_SYSTEM-ARCHITECTURE \
  --repo betterborg/betterborg-cli
./betterborg-OPERATING_SYSTEM-ARCHITECTURE version
```

On Darwin, use `shasum -a 256 -c` in place of `sha256sum --check`. The version
output must be exactly `betterborg VERSION`.

There is no destructive rollback for PyPI or GitHub Release bytes. For a
release defect, stop promoting the affected version, direct existing binary
users to the last known-good attested version, and fix forward with a new
reviewed version and tag. Do not delete assets, replace a tag, upload with
`--clobber`, or reuse the affected version. Record both versions and all
digests in the incident notes so the rollback decision remains auditable.
