# Contributing

Bug reports, reproducible examples, documentation, tests, and code contributions
are welcome. Discuss substantial API or architecture changes in an issue first;
small fixes can go directly to a pull request.

## Development setup

CI targets Python 3.10, 3.11, and 3.12 on Linux for source tests and clean wheel
and source-distribution installation checks. The minimum remains Python 3.10.
Use a Python version in this matrix and uv 0.11.4 (pinned in `pyproject.toml`). From the repository
root, install the locked development dependencies and hooks:

```sh
python -m pip install uv==0.11.4
uv sync --locked --all-extras
uv run --locked pre-commit install --hook-type pre-commit --hook-type pre-push
```

The package is currently at `cellseg_models_pytorch/`, with tests beside their
modules. See [AGENTS.md](AGENTS.md) for the layout and development conventions,
and [the maintenance plan](docs/maintenance-plan.md) for upcoming typing work.
`uv.lock` is the single project lockfile; poetry-core remains the build backend.
Use `uv lock` after intentional dependency edits, then `uv sync --locked`.

## Changes and checks

Keep the change focused and update affected callers, exports, tests, and examples.
Preserve public APIs, checkpoint loading, preprocessing, and segmentation output
semantics unless the change explicitly addresses them. Report compatibility
changes and provide a migration example when needed.

Use small, seeded tests without downloading pretrained weights. Run a targeted
test first, then the ordinary suite when relevant:

```sh
uv run --locked pytest path/to/test_file.py -x
HF_HUB_OFFLINE=1 NUMBA_DISABLE_JIT=1 uv run --locked pytest
uv run --locked pre-commit run --files path/to/changed_file.py
```

The `--slow`, `--optional`, and `--cuda` flags select separate test categories.
Describe which category and device you tested, and list any relevant skips.
Optional integrations require their packages to be installed explicitly. Run
those checks with `uv run --no-sync pytest --optional <test-path>` after installing
the needed packages into the project environment with `uv pip install`; ordinary
locked synchronization removes undeclared packages. The
[pinned CPU checkpoint baseline](tests/fixtures/checkpoint_baseline/README.md)
has separate offline integrity and explicit public-download checks.
For numerical changes, compare values and final masks, not only tensor shapes.
For performance changes, include hardware, precision, workload, memory, and
measurement details. Keep data/checkpoint licenses and attribution intact.

Ruff 0.17.0 is the single linter/formatter and matches the hook revision. CI checks
`tools/` with Ruff and runs strict mypy on `tools/check_release_metadata.py` and
`cellseg_models_pytorch/utils/convolve.py`,
targeting Python 3.10. Run `uv run --locked ruff check tools`,
`uv run --locked ruff format --check tools`, and `uv run --locked mypy`. Other library
modules remain outside this gate; expand it gradually without broad suppressions or casts.
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

## Documentation

Use Sphinx with PyData and MyST-NB. Docs tools live in the separate `docs`
dependency group. Synchronize it and build the pages without executing notebooks:

```sh
uv sync --locked --group docs
uv run --locked --group docs sphinx-build -W --keep-going -b html docs docs/_build/html
```

To validate the short offline notebooks during the build:

```sh
HF_HUB_OFFLINE=1 NUMBA_DISABLE_JIT=1 uv run --locked --group docs sphinx-build -W --keep-going -b html -D nb_execution_mode=force docs docs/_build/executed
```

The real-data training notebook is excluded from forced offline builds. Prepare
its PanopTILs subset as described on the page, then execute it explicitly and
review its saved outputs. Offline tests use temporary folder fixtures to validate
the same training code, without downloading data. They require Albumentations
from the docs group; environments without it report the training cases as skipped.

Open `docs/_build/html/index.html` to read the site. Build outputs stay out of Git.
Open notebooks under `docs/tutorials/` in your Jupyter client with a kernel from
the synchronized environment. They are the source for the corresponding pages.

### Docstrings

Use Google-style docstrings for public APIs, including relevant arguments,
output semantics, exceptions, and a working example. Describe tensor shapes and
preprocessing; omit empty boilerplate sections and types already in signatures.
Preserve existing docstring content when converting style. Keep unrelated
documentation rewrites out of functional changes.
Use generic example paths and no machine-specific identifiers or patient data.

### User-facing pages

- Write for a first-time user who knows basic Python but has no segmentation
  background. Start with what the task achieves, what input they need and the
  result they will get; introduce concepts as the workflow needs them.
- Use short, direct sentences and familiar words. Define essential terms on first
  use with a concrete example: an instance label identifies one object; a class
  label describes its category. Explain abbreviations before using them.
- Keep each page focused on one task and a clear sequence of steps. Explain why
  a choice matters to the user. Put detailed tensor contracts, architecture internals
  and advanced options in the relevant reference section rather than the introduction.
- Show how to recognize the expected result and what to do next. Use a small useful
  image, output or example when it clarifies the task; avoid diagnostic dumps.
- Describe supported behavior accurately, with essential prerequisites and relevant
  limitations stated once where they matter. Keep maintenance history, CI details
  and developer instructions in contributor guidance, not user tutorials.
- Review prose alongside code: remove repetition, unnecessary technical explanation
  and jargon that does not help the reader complete the task.

### Code and notebooks

- Apply the development, Python and PyTorch rules to example code and notebook cells
  as well as library code. Review every code block before presenting it to users.
- Teach one complete workflow with clear names, explicit imports grouped at the
  beginning, normal formatting and short cells organized by purpose. Explain
  shapes, preprocessing and decisions in prose rather than adding diagnostic code.
  Extract a helper only when it removes real repetition without hiding the lesson.
- Separate semantic code blocks with blank lines. Add short comments that clarify
  unfamiliar steps for first-time users. Prefer a straight-line workflow over
  optional branches; keep control flow when device support, validation or safe
  loading requires it. Show relevant outputs with clearly labelled plots.
- Show the actual task: inference uses meaningful model outputs; fine-tuning loads
  pretrained weights. Synthetic data or random weights are appropriate for a
  clearly labelled demonstration or test, not a substitute for the promised workflow.
- Keep assertions minimal in rendered examples. Include a concise assertion only
  when it teaches an essential contract. Put parameter snapshots, gradient checks,
  numerical comparisons and other regression checks in maintained tests outside
  the tutorial pages; do not turn a tutorial into a test suite.
- Preserve necessary input validation, error handling and protection against data
  loss. Use explicit exceptions at external-input boundaries, not assertions.
- Use existing public APIs and idiomatic PyTorch loops. Avoid fallback branches,
  private implementation access, configuration machinery and wrappers added just
  to make an example execute. Fix a broken public contract at its owning layer.
- Notebooks under `docs/tutorials/` are the source for Sphinx pages. Keep setup and
  dependencies reproducible and outputs relevant to the lesson. Retain reviewed
  execution outputs when they demonstrate a verified run; exclude machine-specific
  paths and warning/diagnostic clutter, and regenerate stale outputs after code edits.
  Execute examples separately to validate them; successful
  execution alone does not establish numerical correctness or model quality.
