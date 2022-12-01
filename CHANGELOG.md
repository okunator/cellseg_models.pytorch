
<a id='changelog-0.1.14'></a>
# 0.1.14 — 2022-12-01

## Performance

- Throw away some unnecessary parts of the cellpose post-proc pipeline that just brought overhead and did nothing.

## Refactor

- Refactor the whole cellpose post-processing pipeline for readability.

- Refactored multiprocessing code to be reusable and moved it under `utils`.

## Features

- Add exact euler integration (on CPU) for cellpose post-processing.

- added more pathos.Pool options for parallel processing. Added `ThreadPool`, `ProcessPool` & `SerialPool`
- add all the mapping methods for each Pool obj. I.e. `amap`, `imap`, `uimap` and `map`

## Tests

- added tests for the multiprocessing tools.
<a id='changelog-0.1.13'></a>

# 0.1.13 — 2022-11-25

## Features

- Add option to return encoder features, and decoder features along the outputs in the forward pass of any model.

## Fixes

- Turn the `cellpose` and `stardist` postproc dirs into modules.
  <a id='changelog-0.1.13'></a>

# 0.1.12 — 2022-11-03

## Performance

- Reverse engineered the `stardist` post-processing pipeline to python. Accelerated it with Numba and optimized it even further. Now it runs almost 2x faster than the original C++ verion.

## Fixes

- Fix bug with padding in `SlidingWindowInferer`
  <a id='changelog-0.1.11'></a>

# 0.1.11 — 2022-10-21

## Removed

- unnecessary torchvision dependency

<a id='changelog-0.1.10'></a>

# 0.1.10 — 2022-10-21

## Removed

- torch-optimizer removed from the optional dependency list. Started to cause headache.

<a id='changelog-0.1.9'></a>

# 0.1.9 — 2022-10-21

## Refactor

- Moved saving utilities to `FileHandler` and updated tests.

## Features

- Added geojson saving support for inference

<a id='changelog-0.1.8'></a>

# 0.1.8 — 2022-10-18

## Features

- Support to return all of the feature maps from each decoder stage.

- Add multi-gpu inference via DataParallel

<a id='changelog-0.1.7'></a>

# 0.1.7 — 2022-10-15

## Fixes

- Fix SCE loss bug.
  <a id='changelog-0.1.6'></a>

# 0.1.6 — 2022-10-14

## Features

- Add a Wandb artifact table callback for loading a table of test data metrics and insights to wandb.

## Fixes

- Symmetric CE loss fixed.

- Add option to return binary and instance labelled mask from the dataloader. Previously binary was returned with `return_inst` flag which was confusing.
- Fix the `SegmentationExperiment` to return preds and masks at test time.

<a id='changelog-0.1.5'></a>

# 0.1.5 — 2022-10-07

## Fixes

- Wandb Callback bugs fixed.
  <a id='changelog-0.1.4'></a>

# 0.1.4 — 2022-10-06

## Test

- Update loss tests

## Fixes

- Add a conv block `BasicConvOld` to enable `Dippa` to cellseg conversion of models.
- Fix `inst_key`, `aux_key` bug in `MultiTaskUnet`
- Add a type_map > 0 masking for the `inst_map`s in post-processing

- Modify the optimizer adjustment utility function to adjust any optim/weight params.

- Modify lit `SegmentationExperiment` according to new changes.

## Features

- Add optional spectral decoupliing to all losses
- Add optional Label smoothing to all losses
- Add optional Spatially varying label smoothing to all losses

- Add mse, ssim and iqi torchmetrics for metric logging.
- Add wandb per class metric callback for logging.
- Add `from_yaml` init classmethod to initialize from yaml files.

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

- Update `save2db` & `save2folder` for optional type_map and sem_map args.
- Pre-processing (`pre-proc`) callable arg for `_get_tiles` method. This enables the Lizard datamodule.
- Fix- padding bug with sliding window inference.

## Features

- Lizard datamodule (https://arxiv.org/abs/2108.11195)

- Add a universal multi-task U-net model builder (experimental)

## Test

- Update dataset tests.

- Update tests for multi-task U-Net

## Type Hints

- Fix incorrect type hints.

## Examples

- Add cellpose training with Lizard dataset notebook.
