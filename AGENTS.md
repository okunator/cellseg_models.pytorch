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
| `tools/`, `tools/tests/` | Build and release checks and their tests |
| `examples/`, `docs/`, `.github/` | User examples, maintenance guidance, CI |

Keep this layout. A migration to `src/`, a new training framework, or a new output
representation is a separate change, not incidental cleanup.

The following two sections are the Ponytail development instructions.

## Before you write

Read the task and the code it touches. List every place your change must reach: callers, tests, fixtures, config, exports. Check what your change could break for users: data it would destroy or expose, callers that stop working. That is scope. Extra features are not.

## The smallest complete change

Take the first option that fully works:

1. Does it need to exist? Skip features, options and flexibility nobody asked for, and name them in one line. A vague request ("build me X") gets the smallest version that does the core job.
2. Already in this codebase (a helper, component, service, pattern)? Use it the way the surrounding code does.
3. Standard library or a platform feature? Use it, unless the project has its own. A house component beats a native widget.
4. An installed dependency? Use it. Never add a dependency for a few lines.
5. Can it be one line a reader gets at a glance? One line.
6. Otherwise: the minimum code that works.

- Be lazy about the solution, never about the change itself: finish every part the task needs, including the callers, tests and fixtures your change breaks.
- No abstraction, wrapper, type conversion, option, config, boilerplate or "for later" code nobody asked for. Keep values in the form the platform already gives you. Deletion beats addition. Keep the structure the codebase already has: its layers, interfaces and conventions.
- The shortest working diff wins, once you know everything it must touch. A one-liner that needs decoding is not short.
- Comment only the why the code cannot show, in one line.
- Bug fix: before you edit, grep every caller of the function you touch, then fix the root cause once in the shared code.
- Code you move or merge keeps its error handling and validation.
- Between options of equal size, take the one that is correct on edge cases.
- Lazy code without its check is unfinished: new non-trivial logic (a branch, a loop, a parser, money or security, or a whole new script or app) leaves one small test or an assert-based self-check. Trivial changes need none.
- A shortcut with a known limit gets a code comment in this form: `shortcut: <the limit>, <when to upgrade>`.

Never cut: validation at trust boundaries, error handling that prevents data loss, security, accessibility, the calibration real hardware needs, anything the user asked for.

## Python and typing

- Add accurate annotations to new or changed public APIs. Prefer `list[T]`,
  `dict[K, V]`, and `X | None` where they work on the supported Python versions.
  Avoid unrelated annotation sweeps.
- Type checking is being introduced gradually. Mypy is currently an old dev
  dependency without a configured CI gate. Establish an explicit checked scope
  before treating a broad mypy run as a required passing check.
- Preserve existing tensor, array, dictionary, and dataclass contracts. Do not
  add wrappers, conversions, or `cast()` calls just to silence a checker.
- Narrow optional values and validate external inputs. Keep `Any` at genuinely
  untyped integration boundaries; do not spread it through the internal API.
- Do not hide errors with blanket `ignore_errors`, `# type: ignore`, or lint
  suppressions. A necessary third-party suppression must name the error and why.
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
anything. Check collection, skips, and execution when adding a CI gate.

Run the narrowest meaningful checks, then the required CI checks. Avoid unrelated
formatting or repeated full suites without a new reason. Report the exact scope,
results, and any unverified device, version, or optional path.

## Current development commands

Poetry 2.2.1 and `poetry.lock` remain authoritative until the planned uv migration
lands. Do not create a second project lockfile or silently switch tools mid-task.
Use [CONTRIBUTING.md](CONTRIBUTING.md) for setup and
[the release guide](docs/releasing.md) before changing publication.

| Task | Command from repository root |
| --- | --- |
| Synchronize development environment | `poetry sync --all-extras` |
| Targeted test | `poetry run pytest path/to/test_file.py -x` |
| Ordinary offline suite | `HF_HUB_OFFLINE=1 NUMBA_DISABLE_JIT=1 poetry run pytest` |
| Coverage | `HF_HUB_OFFLINE=1 NUMBA_DISABLE_JIT=1 poetry run pytest --cov=cellseg_models_pytorch --cov-report=xml` |
| Release guard tests | `poetry run python -m unittest discover -s tools/tests -v` |
| Hooks for changed files | `poetry run pre-commit run --files <changed-files>` |
| Install commit and push hooks | `poetry run pre-commit install --hook-type pre-commit --hook-type pre-push` |
| Build distributions | `poetry build` |

The Ruff hooks are pinned in `.pre-commit-config.yaml`; Ruff is not yet a project
dev dependency. Use the configured hooks on touched files. Modernizing Ruff,
its configuration, and the competing older format/lint tools is planned work.
The future mypy command and checked modules must be documented when its gate lands.
Do not bypass hooks to hide a failure; repair it or report the actual blocker.

## Documentation and delivery

Write Google-style docstrings for new public APIs, including relevant arguments,
output semantics, exceptions, and a working example. Describe tensor shapes and
preprocessing; omit empty boilerplate sections and types already in signatures.
Preserve clear existing docstrings instead of reformatting unrelated APIs.
Use generic example paths and no machine-specific identifiers or patient data.

Keep commits thematic and use Conventional Commits. Leave unrelated user edits
untouched and preserve contributor attribution. Do not rewrite shared history.
Update `CHANGELOG.md` for release-relevant public changes without inventing a
release/version; internal docs and test-only changes need no release entry.

The optional repository skills are
[commit](.agents/skills/commit/SKILL.md) and
[proper-code-review](.agents/skills/proper-code-review/SKILL.md). Use them for their
named workflows; reading guidance does not authorize a merge, release, data
deletion, long compute run, or external message. Follow the user's existing scope.

Keep runtime upgrades, typing rollout, and the uv migration independently
reviewable. Validate built wheels and source distributions outside the checkout.
Publication must use the verified artifacts and PyPI trusted publishing once the
maintainer has configured it; do not introduce a stored-token fallback.
