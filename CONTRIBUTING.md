# Contributing

Bug reports, reproducible examples, documentation, tests, and code contributions
are welcome. Discuss substantial API or architecture changes in an issue first;
small fixes can go directly to a pull request.

## Development setup

CI targets Python 3.10, 3.11, and 3.12 on Linux for source tests and clean wheel
and source-distribution installation checks. The minimum remains Python 3.10.
Use a Python version in this matrix and Poetry 2.2.1. From the repository
root, install the locked development dependencies and hooks:

```sh
poetry sync --all-extras
poetry run pre-commit install --hook-type pre-commit --hook-type pre-push
```

The package is currently at `cellseg_models_pytorch/`, with tests beside their
modules. See [AGENTS.md](AGENTS.md) for the layout and development conventions,
and [the maintenance plan](docs/maintenance-plan.md) for upcoming typing and uv
work. Poetry remains the project manager until that migration lands.

## Changes and checks

Keep the change focused and update affected callers, exports, tests, and examples.
Preserve public APIs, checkpoint loading, preprocessing, and segmentation output
semantics unless the change explicitly addresses them. Report compatibility
changes and provide a migration example when needed.

Use small, seeded tests without downloading pretrained weights. Run a targeted
test first, then the ordinary suite when relevant:

```sh
poetry run pytest path/to/test_file.py -x
HF_HUB_OFFLINE=1 NUMBA_DISABLE_JIT=1 poetry run pytest
poetry run pre-commit run --files path/to/changed_file.py
```

The `--slow`, `--optional`, and `--cuda` flags select separate test categories.
Describe which category and device you tested, and list any relevant skips.
Optional integrations require their packages to be installed explicitly.
For numerical changes, compare values and final masks, not only tensor shapes.
For performance changes, include hardware, precision, workload, memory, and
measurement details. Keep data/checkpoint licenses and attribution intact.

Ruff 0.17.0 is the single linter/formatter and matches the hook revision. CI checks
`tools/` with Ruff and runs strict mypy on `tools/check_release_metadata.py`,
targeting Python 3.10. Run `poetry run ruff check tools`,
`poetry run ruff format --check tools`, and `poetry run mypy`. Library typing is
not gated yet; expand its scope gradually without broad suppressions or casts.
Annotate changed public APIs accurately and document tensor contracts.

## Pull requests and releases

Explain the problem, resulting behavior, validation, and any compatibility impact.
Use small thematic commits and keep direct regression tests with their fixes.
Document release-relevant public changes in `CHANGELOG.md`; do not invent a
release date or version.

CI checks the source suite and clean installations of both built distributions.
Manual publication-workflow runs only validate. Release publication waits for
these checks and uploads the verified artifacts through PyPI trusted publishing.
Maintainers should follow [the release guide](docs/releasing.md).
