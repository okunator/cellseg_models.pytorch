# Agent instructions

## Project orientation

Read this file and [the maintenance plan](docs/maintenance-plan.md) before making
changes. This is an open-source PyTorch library for 2D cell and nuclei instance
segmentation, with pretrained checkpoints, training components, tiled inference,
post-processing, and metrics. Preserve its public APIs and checkpoint compatibility.

The code and configuration define current behavior. The maintenance plan defines
future work; it does not make planned tools or features available today.

| Location | Purpose |
| --- | --- |
| `cellseg_models_pytorch/models/` | Architectures, high-level model APIs, checkpoint loading |
| `cellseg_models_pytorch/encoders/`, `decoders/`, `modules/` | PyTorch building blocks |
| `cellseg_models_pytorch/inference/`, `wsi/`, `postproc/` | Prediction, whole-slide workflows, instance reconstruction |
| `cellseg_models_pytorch/torch_datasets/`, `transforms/` | Data loading and preprocessing |
| `cellseg_models_pytorch/losses/`, `metrics/` | Training objectives and evaluation |
| `cellseg_models_pytorch/**/tests/`, `conftest.py` | Library tests and shared fixtures |
| `tests/`, `tests/fixtures/` | Repository-only checkpoint comparisons and reference data |
| `tools/`, `tools/tests/` | Build and release checks and their tests |
| `docs/`, `docs/tutorials/`, `.github/` | Sphinx/PyData pages, notebook examples, maintenance guidance and CI |

Keep this layout. A migration to `src/`, a new training framework, or a new output
representation is a separate change, not incidental cleanup.

## Before you write

Read the task and affected code. List the callers, tests, fixtures, config and
exports the change must reach, and risks to users: data loss/exposure or broken
callers. That is scope; extra features are not. For a bug fix, grep every caller
of the affected function before editing.

Fix the root cause once in the layer that owns it. Avoid hot fixes and creative
workarounds that bypass interfaces or hide broken contracts. If the proper fix
needs broader scope, state that scope instead of adding a workaround.

## The smallest complete change

Take the first option that fully works:

1. Skip unrequested features, options and flexibility; name omissions in one line.
   For a vague request ("build me X"), build the smallest version that does the core job.
2. Reuse an existing helper, component, service or pattern as surrounding code does.
3. Use the standard library or platform feature unless the project has its own.
   A house component beats a native widget.
4. Use an installed dependency. Never add one for a few lines.
5. Use one line when a reader can understand it at a glance.
6. Otherwise, use the minimum code that works.

- Finish the whole scoped change, including affected callers, tests and fixtures.
- Add no unrequested abstraction, wrapper, conversion, option, config, boilerplate
  or "for later" code. Keep platform-native values and existing layers, interfaces
  and conventions. Deletion beats addition.
- Prefer the shortest complete working diff. A cryptic one-liner is not short.
- Comment only the why the code cannot show, in one line.
- Code you move or merge keeps its error handling and validation.
- Between options of equal size, take the one that is correct on edge cases.
- New non-trivial logic (branch, loop, parser, money/security logic, or a new script
  or app) needs one small test or assert-based self-check. Trivial changes need none.
- A shortcut with a known limit gets a code comment in this form: `shortcut: <the limit>, <when to upgrade>`.

Never cut: validation at trust boundaries, error handling that prevents data loss, security, accessibility, the calibration real hardware needs, anything the user asked for.

## Python and typing

- Add accurate annotations to new or changed public APIs. Prefer `list[T]`,
  `dict[K, V]`, and `X | None` where they work on the supported Python versions.
  Avoid unrelated annotation sweeps.
- Type checking is being introduced gradually. Mypy checks
  `tools/check_release_metadata.py` and
  `cellseg_models_pytorch/utils/convolve.py` in strict mode, targeting Python 3.10.
  `uv run --locked mypy` runs both scopes; CI checks the library helper in the
  existing Python 3.11 source job and the release guard in the developer-only job.
  Expand the checked scope explicitly before requiring broader runs.
- Preserve existing tensor, array, dictionary, and dataclass contracts. Do not
  add wrappers, conversions, or `cast()` calls just to silence a checker.
- Narrow optional values and validate external inputs. Keep `Any` at genuinely
  untyped integration boundaries; do not spread it through the internal API.
- Do not hide errors with blanket `ignore_errors`, `# type: ignore` or lint
  suppressions. Necessary third-party suppressions must name the error and reason.
- Prefer functions for stateless work, `pathlib` for paths, explicit imports,
  and the existing `__all__` exports. Keep I/O separate from numerical logic.
- Avoid mutable defaults and silent failures. Use contextual built-in exceptions
  unless the codebase already has an appropriate custom exception.
- Library code uses `logging.getLogger(__name__)`; do not configure global logging.
  Short diagnostic output is appropriate in command-line tools.

## PyTorch and segmentation contracts

- Document named axes, dtype, device, channel order, value range, and whether an
  output is logits, probabilities, classes, or instance labels. Distinguish BCHW
  tensors from HWC images and binary masks from labelled instances.
- Keep image/mask transforms aligned. Preserve instance identifiers, background
  conventions, class mappings, pixel coordinates, and spatial calibration where
  relevant. Do not change normalization or metric matching semantics silently.
- Keep device and dtype selection explicit at API boundaries. Avoid implicit
  CUDA use, host/device transfers, or dtype changes in numerical kernels and
  `forward` paths. Use the existing inference API to handle mode and precision.
- Respect autograd, train/eval mode, and mixed precision. Exports and temporary
  inference operations must restore state after failure as well as success.
  Use AMP APIs supported by the declared minimum PyTorch version.
- CPU tests must work without a GPU. Exercise supported GPU paths separately;
  CPU correctness or speed does not establish GPU correctness or speed.
- Before GPU benchmarks or long runs, check active jobs, available memory, and
  the authorized compute budget. Preserve other runs; keep configs, logs, and
  resumable checkpoints with optimizer, scheduler, scaler, and RNG state as needed.
- Keep whole-slide and dataset processing bounded in memory. Stream or reduce
  batches instead of collecting every prediction when only aggregates are needed.
  Avoid unnecessary copies, synchronization, and checkpoint/data downloads.
- Measure optimizations on representative workloads. Record hardware, precision,
  input/batch sizes, warmup, synchronization, latency, and peak memory; compare
  quality as well as throughput. Do not promote defaults from one noisy run.
- Keep patient/slide splits independent before making patches. Prevent leakage
  across training, validation, and test data. Record seeds and nondeterminism;
  confirm scientific claims across suitable seeds and datasets.
- Pin integration-test checkpoint identity and preprocessing. Check dense outputs
  and reconstructed masks with explicit tolerances. Never loosen tolerances,
  replace expected results, or skip a regression solely to make CI green.
- Do not overwrite checkpoints, raw images, annotations, or benchmark results
  implicitly. Treat external data and checkpoint files as untrusted inputs and
  preserve attribution, dataset permissions, and dependency/checkpoint licenses.

## Tests and validation

Use small pytest functions and fixtures, fixed seeds for randomized tests, and
the smallest valid model/input that exercises the behavior. Reuse parametrization
when cases share a contract. Keep implementation and its regression tests together.
Existing release-tool unittest checks can remain; do not rewrite them for style.

For each test, ask what bug it catches and whether plausibly wrong code could
pass it. Assert meaningful shapes, values, state, or errors, not just execution.
Keep ordinary tests offline and disable pretrained encoders in unit tests.
Real checkpoint, optional-package, slow, and GPU tests belong in explicit checks.

`--slow`, `--optional`, and `--cuda` each select only their marked category and
otherwise skip it. Run categories separately. Do not assume `--all-extras`
installs undeclared optional packages or that a skipped integration test validated
anything. After explicitly installing optional packages, use `uv run --no-sync`
for those checks so automatic synchronization does not remove them. Check collection, skips, and execution when adding a CI gate.

Run the narrowest meaningful checks, then required CI checks. Avoid unrelated
formatting and repeated full suites without a new reason. Report exact scope,
results and unverified devices, versions or optional paths.

## Current development commands

uv 0.11.4 and `uv.lock` are authoritative. Use the required version in
`pyproject.toml`, locked synchronization and one project lockfile.
The build backend remains poetry-core 2.2.1; Poetry CLI is not needed.
Use [CONTRIBUTING.md](CONTRIBUTING.md) for setup and
[the release guide](docs/releasing.md) before changing publication.

| Task | Command from repository root |
| --- | --- |
| Synchronize development environment | `uv sync --locked --all-extras` |
| Targeted test | `uv run --locked pytest path/to/test_file.py -x` |
| Ordinary offline suite | `HF_HUB_OFFLINE=1 NUMBA_DISABLE_JIT=1 uv run --locked pytest` |
| Coverage | `HF_HUB_OFFLINE=1 NUMBA_DISABLE_JIT=1 uv run --locked pytest --cov=cellseg_models_pytorch --cov-report=xml` |
| Release guard tests | `uv run --locked python -m unittest discover -s tools/tests -v` |
| Hooks for changed files | `uv run --locked pre-commit run --files <changed-files>` |
| Install commit and push hooks | `uv run --locked pre-commit install --hook-type pre-commit --hook-type pre-push` |
| Build distributions | `uv build` |
| Build docs | `uv run --locked --group docs sphinx-build -W --keep-going -b html docs docs/_build/html` |

Ruff 0.17.0 is the development linter/formatter and matches the pre-commit hook
revision. Black, isort, and Flake8 are no longer parallel toolchains. Preserve the
current lint rule scope unless a separate cleanup establishes broader coverage.
CI checks Ruff on `tools/` and mypy on the limited scope above; this is not a
whole-library lint or typing claim. Use `uv run --locked ruff check tools`,
`uv run --locked ruff format --check tools`, and `uv run --locked mypy` locally. Never bypass
hooks or hide failures with broad ignores.

## Documentation and delivery

Before editing public docstrings, pages or notebooks, read and follow
[the documentation standards](CONTRIBUTING.md#documentation). They cover
concise Google-style docstrings, prose for new library users, readable examples
and validation outside tutorial pages. Development and PyTorch rules apply to
example code too.

Keep commits thematic and use Conventional Commits. Leave unrelated user edits
untouched and preserve contributor attribution. Do not rewrite shared history.
Update `CHANGELOG.md` for release-relevant public changes without inventing a
release/version; internal docs and test-only changes need no release entry.

The optional repository skills are
[commit](.agents/skills/commit/SKILL.md) and
[proper-code-review](.agents/skills/proper-code-review/SKILL.md). Use them for their
named workflows; reading guidance does not authorize a merge, release, data
deletion, long compute run, or external message. Follow the user's existing scope.

For temporary maintenance documents, diagnostics, fixtures, and CI checks, record
their purpose and removal trigger in the plan or PR. Retire them when superseded;
review CI cost and distribution contents after each maintenance batch.

Keep runtime upgrades, typing rollout, and the uv migration independently
reviewable. Validate built wheels and source distributions outside the checkout.
Publication must use the verified artifacts and PyPI trusted publishing once the
maintainer has configured it; do not introduce a stored-token fallback.
