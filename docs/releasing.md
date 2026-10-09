# Release validation and publishing

The publication workflow builds distributions once, checks their metadata, tests
their clean installation, and runs the source suite. It publishes those same
artifacts only for a published GitHub release. Manual dispatch performs validation
without uploading to PyPI.

## Configure PyPI trusted publishing

The maintainer must configure an existing project's publisher in PyPI before the
next release. Use these values for `cellseg-models-pytorch`:

| Field | Value |
| --- | --- |
| Repository owner | `okunator` |
| Repository name | `cellseg_models.pytorch` |
| Workflow filename | `publish.yml` |
| GitHub environment | `pypi` |

Create the matching GitHub environment and choose its tag/release restrictions
and any desired maintainer review rules. The publishing job alone receives
`id-token: write`; source tests and builds do not. Do not add `PYPI_TOKEN` or a
password fallback. Remove the obsolete publishing secret after the OIDC setup
is verified. Coverage upload configuration is separate from PyPI authentication.

See [PyPI's publisher setup](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
and [the publishing action](https://github.com/pypa/gh-action-pypi-publish).
Workflow syntax checks and ordinary CI cannot verify the PyPI-side configuration.

## Prepare and validate a release

1. Complete the relevant checks in [the maintenance plan](maintenance-plan.md),
   including real checkpoint and optional/device checks affected by the release.
2. Update the version in `pyproject.toml` and
   `cellseg_models_pytorch/__init__.py` together, with release notes and any
   migration instructions. Run the release metadata guard.
3. Run the ordinary source suite, build both distributions, and verify clean
   wheel/source installation outside the checkout. Review required hosted checks.
4. Use the release candidate process for dependency or compatibility changes.
   A prerelease must have its own matching prerelease package version and tag.
5. Tag the reviewed commit with the package version, optionally prefixed by `v`.
   Publish its GitHub release only when the artifacts and release notes are ready.
6. Check publication and install the released package in a clean environment.
   If publication fails, inspect the validation or publisher error before retrying.

Do not replace an existing release artifact or reuse a published PyPI version.
Resolve a regression with a reviewed fix and new version, or an explicitly
authorized yank when appropriate; keep older checkpoints and baselines available.
