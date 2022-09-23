<a id='changelog-0.1.3'></a>

# 0.1.3 — 2022-09-23

## Test

    - Update tests for Inferes and mask utils.

    - Add tests for the benchmarkers.

## Fixes

    - init and typing fixes

## Docs

    - Typo fies in docs

## Features

    - Add numba parallellized median filter and majority voting for post-processing
    - Add support for own semantic and type seg post-proc funcs in Inferers

    - Add segmentation performance benchmarking helper class.
    - Add segmentation latency benchmarking helper class.

<a id='changelog-0.1.2'></a>

# 0.1.2 — 2022-09-09

## Fixes

- **datasets.writers**: Update `save2db` & `save2folder` for optional type_map and sem_map args.
- **datasets.writers**: Pre-processing (`pre-proc`) callable arg for `_get_tiles` method. This enables the Lizard datamodule.
- **inference**: Fix- padding bug with sliding window inference.

## Features

- **datamodules**: Lizard datamodule (https://arxiv.org/abs/2108.11195)

- **models**: Add a universal multi-task U-net model builder (experimental)

## Test

- **dataset**: Update dataset tests.

- **models**: Update tests for multi-task U-Net

## Type Hints

- **models**: Fix incorrect type hints.

## Examples

- Add cellpose training with Lizard dataset notebook.
