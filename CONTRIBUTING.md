# Contributing

Bug reports, reproducible examples, documentation, tests, and code contributions
are welcome. Discuss substantial API or architecture changes in an issue first;
small fixes can go directly to a pull request.

## Development setup

Linux source and Numba JIT checks cover Python 3.10–3.14. Routine wheel and
source-distribution install checks use Python 3.10 and 3.14; manual and release
runs cover all five versions. The minimum remains Python 3.10.
Use a supported Python version and uv 0.11.4 (pinned in `pyproject.toml`).
From the repository root, install the locked development dependencies and hooks:

```sh
python -m pip install uv==0.11.4
uv sync --locked --all-extras
uv run --locked pre-commit install --hook-type pre-commit --hook-type pre-push
```

The package is currently at `cellseg_models_pytorch/`, with tests beside their
modules. See [AGENTS.md](AGENTS.md) for the layout and development conventions.
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
`tools/` with Ruff and runs strict mypy on `tools/check_release_metadata.py`,
`cellseg_models_pytorch/utils/convolve.py` and `cellseg_models_pytorch/wsi/inst_merger.py`,
targeting Python 3.10. Run `uv run --locked ruff check tools`,
`uv run --locked ruff format --check tools`, and `uv run --locked mypy`. Other library
modules remain outside this gate; expand it gradually without broad suppressions or casts.
Annotate new or changed library functions, including private helpers, and document
tensor contracts.

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

Validate notebook code with offline fixtures separately from the site build:

```sh
HF_HUB_OFFLINE=1 NUMBA_DISABLE_JIT=1 uv run --locked --group docs pytest tests/test_prepare_panoptils.py tests/test_notebook_tutorials.py tests/test_wsi_tutorial.py tests/test_evaluation_tutorial.py
```

API example tests execute the page code in order and the documented docstring
examples, using temporary images, labels and local weights:

```sh
HF_HUB_OFFLINE=1 NUMBA_DISABLE_JIT=1 uv run --no-sync pytest tests/test_api_examples.py
HF_HUB_OFFLINE=1 NUMBA_DISABLE_JIT=1 uv run --no-sync pytest --optional tests/test_api_examples.py
```

The optional cases need PyTables and the packages in `tools/onnx-requirements.txt`.
Install these explicitly in the test environment and use `--no-sync` afterward.
The ONNX example is exported and executed on CPU, then compared with PyTorch.

### Maintaining the API reference

Pages under `docs/api/` own the list of documented objects and members. Use explicit
`autoclass`, `autofunction` and `automethod` entries with selected members; avoid
recursive module listings and inherited implementation details. An omitted object
remains importable; reference scope does not change the public API.

Cover APIs used by the quick start and maintained guides, plus model-family classes
and factory functions. Keep internal building blocks out of the reference. Database
creation, alternative dataset preparation and custom transform authoring need a
maintained user workflow before expanding their reference scope.

When a guide uses another package API, add or link its reference entry in the same
change. Cover the methods, attributes and output fields the example uses, including
inherited methods. The API tests check guide imports against the reference, and
execute page code and docstring examples. Add new page examples to those checks.

Review both the docstrings and rendered page for accuracy and readability. Keep
introductions brief, explain required inputs and returned values, and put necessary
limits beside the affected API. Link complete workflows to the notebooks. Keep
deferred features and functional fixes in the maintenance plan, not a second API list.

The notebooks use real data or weights and are excluded from forced offline
builds. Execute them explicitly and review saved outputs; CI renders those outputs
without downloading data or checkpoints. Execute inference after training so
its image, labels, weights and class taxonomy match. The quick start covers
independent use of public pretrained checkpoints on an appropriate input.
The WSI notebook uses the same training checkpoint on held-out TCGA-AC-A6IW.
The notebook defaults to cuCIM GPU reads; its setup installs the NVIDIA decoding
packages and enables the upstream GPU reader before imports. OpenSlide remains
an alternative, with dependencies in the docs group. Offline tests skip the
cuCIM initialization and substitute a CPU reader and model, but run the real tiled
writer and instance merger; execute the full-slide workflow on the DGX and inspect
all twelve seam regions separately.
Overlap regressions check source-tile identity, connected contours from complete
predictions, and preservation of separate nuclei and prediction attributes.
The separate checkpoint baseline tests retain their original 256-pixel inputs and
numerical references; do not replace them with tutorial predictions.
Run the data-preparation notebook once and reuse its patient-separated PanopTILs
patches for training. The recipe follows the nuclei-only part of
the Histolytics example: augmented image/mask pairs, MSE flows, CE with spectral
decoupling plus Dice cell-type loss, AdamW and a cosine schedule. Validation uses
no random augmentation. Evaluate reconstructed instances and per-class predictions
on predetermined held-out regions; loss reduction alone is insufficient.
Offline training tests use temporary folder fixtures. Inference tests load local
safetensors fixtures. They validate preprocessing, training, optional encoder
freezing, checkpoint reload and plotting, not model quality.
Training checks require Albumentations from the docs group; environments
without it report those cases as skipped. Run the existing `--cuda` category separately
to check device state restoration. Training saves safetensors model weights beside
a separate resume checkpoint. Resume state includes optimizer, scheduler,
Torch/Python/NumPy random states and data split. Checkpoint directories and files
are created exclusively; choose a new directory before repeating a real run.
The examples save at epoch boundaries and recreate workers each epoch. Resume with
the same augmentation seed and worker count. Stateful or persistent workers and
mid-epoch resumption require additional loader and transform state.

Use one OpenCV provider: `opencv-python-headless`. If an existing environment
had both OpenCV packages, repair their overlapping files once with
`uv sync --locked --group docs --reinstall-package opencv-python-headless`.

Open `docs/_build/html/index.html` to read the site. Build outputs stay out of Git.
Open notebooks under `docs/tutorials/` in your Jupyter client with a kernel from
the synchronized environment. They are the source for the corresponding pages.

### Docstrings

Use Google-style docstrings for public APIs, including relevant arguments,
output semantics, exceptions, and a working example. Describe tensor shapes and
preprocessing; omit empty boilerplate sections and types already in signatures.
Start with a short statement of the operation's purpose. Describe the caller's
contract, not a walkthrough of the algorithm. Keep implementation details beside
the relevant code, with short comments in plain language.
Preserve existing docstring content when converting style. Keep unrelated
documentation rewrites out of functional changes.
Use generic example paths and no machine-specific identifiers or patient data.

### User-facing pages

- Write for a first-time user of this library who knows Python and basic deep
  learning but may be new to instance segmentation. Do not define batches, epochs,
  losses or encoders. Explain library-specific behavior and segmentation contracts.
  Start with what the task achieves, what input they need and the result they get.
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
- Keep API introductions brief. Put detailed requirements and limitations beside
  the relevant object or method, and link to guides for full workflows.

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
