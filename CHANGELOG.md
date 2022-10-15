
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
