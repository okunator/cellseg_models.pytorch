---
orphan: true
---

# Release validation and publishing

The publication workflow builds distributions once, checks their metadata, tests
their clean installation, and runs the source suite. It publishes those same
artifacts only for a published GitHub release. The existing source job also builds
the docs once and preserves the HTML for deployment. After a stable release passes
validation and PyPI publication, the workflow deploys those pages to GitHub Pages.
Prereleases do not replace the stable docs. Manual dispatch validates and preserves
artifacts without publishing the package. Docs deployment is off by default.
Metrics releases use a separate workflow and never publish the main package or
deploy documentation.

To update documentation between releases, first push the reviewed changes to
`main` and check CI. Run **Publish to PyPI** manually on that same commit with
**Deploy documentation after validation** enabled. The existing validation jobs
build and check the pages, then the docs job deploys them. The package publication
job stays skipped; no new release or package version is needed.

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
and any desired maintainer review rules. The publishing job receives
`id-token: write`; source tests and builds do not. Do not add `PYPI_TOKEN` or a
password fallback. Remove the obsolete publishing secret after the OIDC setup
is verified. Coverage upload configuration is separate from PyPI authentication.

See [PyPI's publisher setup](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
and [the publishing action](https://github.com/pypa/gh-action-pypi-publish).
Workflow syntax checks and ordinary CI cannot verify the PyPI-side configuration.

## Standalone metrics distribution

`packages/metrics/` builds the independent `cellseg-metrics` distribution. The
existing build/install jobs retain its verified wheel and source archive in the
`metrics-distributions` artifact and test both without Torch or imaging packages.
The main distribution retains only its compatibility imports and requires
`cellseg-metrics>=0.1.0`.

`publish-metrics.yml` handles only releases tagged `metrics-v<version>`, such as
`metrics-v0.1.0`. It builds metrics once, checks metadata, and tests clean wheel and
source installations on Python 3.10–3.14 before publishing those same artifacts.
Manual runs perform validation only. This workflow adds no jobs to routine CI.
The main `publish.yml` skips all `metrics-` releases and still requires an
available metrics wheel on PyPI before publishing the main library.

For the first metrics release, add a pending publisher under
[PyPI account publishing](https://pypi.org/manage/account/publishing/):

| Field | Value |
| --- | --- |
| PyPI project name | `cellseg-metrics` |
| Repository owner | `okunator` |
| Repository name | `cellseg_models.pytorch` |
| Workflow filename | `publish-metrics.yml` |
| GitHub environment | `pypi` |

The first successful upload creates the PyPI project. A pending publisher does
not reserve its name. No API token or preliminary manual upload is required.
See [PyPI's first-publication instructions](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

1. Add `metrics-v*` to the allowed **tags** in GitHub's existing `pypi` environment,
   preserving its main-release rules and required approval.
2. Keep the version in `packages/metrics/pyproject.toml` and
   `packages/metrics/cellseg_metrics/__init__.py` consistent. Check it with
   `uv run --no-project python tools/check_release_metadata.py --metrics`.
3. Push the reviewed workflow and package changes. Run **Publish metrics to PyPI**
   manually on the reviewed ref and check that both artifacts pass validation.
   A manual run cannot upload packages.
4. Only after publication is authorized, create a GitHub release at the reviewed
   commit with the matching tag, initially `metrics-v0.1.0`. Leave **Set as the
   latest release** unchecked so the main library remains the repository's latest
   release. Approve the `pypi` deployment after the release checks pass.
5. Confirm a fresh `pip install cellseg-metrics==0.1.0` works outside the checkout,
   then update the README and API page to replace the temporary source-install
   instructions. Do not reuse a published version.

Metrics publication does not require a main-library version bump. Later main
releases may reuse an already published compatible metrics release. Publication
always needs explicit authorization; do not add a stored-token fallback.

## Configure documentation hosting

Enable GitHub Pages in the repository settings with **GitHub Actions** as its
publishing source. The deployment job uses the `github-pages` environment and
alone receives `pages: write`; both deployment jobs receive `id-token: write`.
Allow release tags in that environment's deployment rules. The stable site is
<https://okunator.github.io/cellseg_models.pytorch/>.

## Prepare and validate a release

1. Complete the [relevant checks](../CONTRIBUTING.md#changes-and-checks),
   including real checkpoint and optional/device checks affected by the release.
2. Update `[project].version` in `pyproject.toml` and
   `cellseg_models_pytorch/__init__.py` together, with release notes and any
   migration instructions. Run `uv lock` and include the updated lockfile in the
   release commit, then run `uv run --locked python tools/check_release_metadata.py`.
3. Run the ordinary source suite, build both distributions with `uv build`, and
   verify clean wheel/source installation outside the checkout. Review required
   hosted checks.
4. Use the release candidate process for dependency or compatibility changes.
   A prerelease must have its own matching prerelease package version and tag.
5. Tag the reviewed commit with the package version, optionally prefixed by `v`.
   Ensure that tag is allowed in both deployment environments.
   Publish its GitHub release only when the artifacts and release notes are ready.
6. Check publication and install the released package in a clean environment.
   If publication fails, inspect the validation or publisher error before retrying.
   If only docs deployment fails, rerun failed jobs to avoid repeating the PyPI upload.

Do not replace an existing release artifact or reuse a published PyPI version.
Resolve a regression with a reviewed fix and new version, or an explicitly
authorized yank when appropriate; keep older checkpoints and baselines available.
