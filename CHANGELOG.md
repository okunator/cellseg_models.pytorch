<a id='changelog-0.1.28'></a>
# 0.1.28 — 2025-08-17

## Fixed

- `gdf2inst` fixed for empty masks.

<a id='changelog-0.1.27'></a>
# 0.1.27 — 2025-07-25

## Fixed

- Top-level key error bug in Predictor class [#75](https://github.com/okunator/cellseg_models.pytorch/issues/75)

<a id='changelog-0.1.26'></a>
# 0.1.26 — 2025-05-07

## Removed

- Removed `datamodules` module
- Removed `datasets` module

## Refactor

- Refactored the whole model interface to be more user-friendly.

## Features

- Added a new `wsi` module, including:
    - A `SlideReader` class to read patches from a WSI slide.
        - Backends: Openslide, CUCIM
        - Adapted the reader class from HistoPrep library. Props to Jopo666
    - `get_sub_grids` function to get subgrids from a WSI slide. Can be used to filter the patches. Based on connected components.

- Added a new torch_datasets module, including:
    - `WSIDatasetInfer` class to run inference directly from WSIs.
        - Adapted the class from HistoPrep library. Props to Jopo666
    - `TrainDatasetH5` class to handle training data for the models from a h5 file.
    - `TrainDatasetFolder` class to handle training data for the models from img and label folders.

- Added a new `inference.WsiSegmenter`-class to handle the segmentation of WSIs.

- Added a new `wsi.inst_merger.InstMerger`-class to handle the merging of instance masks at image boundaries.

- Added `inst2gdf` and `sem2gdf` functions to `utils.vectorize` module. These functions convert efficiently instance and semantic masks to GeoDataFrame objects.

- Added `FileHandler.to_mat` and `FileHandler.to_gson` save functions that take in a dictionary of model output masks (output from the `Inferer`-classes) and save it to a .mat or '.feather', '.geojson', '.parquet' files.

## Added Dependencies

- Added `libpysal` dependency
- Added `networkx` dependency

## Removed Dependencies

- Removed `lightning` dependency
- Removed `albumentations` dependency

## Chore

- Move `FolderDatasetInfer`to `torch_datasets` module

<a id='changelog-0.1.25'></a>
# 0.1.25 — 2024-07-05

## Features

- Image encoders are imported now only from timm models.
- Add `enc_out_indices` to model classes, to enable selecting which layers to use as the encoder outputs.

# Removed
- Removed SAM and DINOv2 original implementation image-encoders from this repo. These can be found from timm models these days.
- Removed `cellseg_models_pytorch.training` module which was left unused after example notebooks were updated.

## Examples

- Updated example notebooks.
- Added new example notebooks utilizing UNI foundation model from the MahmoodLab.
- Added new example notebooks utilizing the Prov-GigaPath foundation model from the Microsoft Research.
- **NOTE:** These examples use the huggingface model hub to load the weights. Permission to use the model weights is required to run these examples.

## Chore

- Update timm version to above 1.0.0.

## Breaking changes

- Lose support for python 3.9
- The `self.encoder` in each model is new, thus, models with trained weights from previous versions of the package will not work with this version.

<a id='changelog-0.1.24'></a>
# 0.1.24 — 2023-10-13

## Style

- Update the `Ìnferer.infer()` -method api to accept arguments related to saving the model outputs.

## Features

- Add `CPP-Net`. https://arxiv.org/abs/2102.06867

- Add option for mixed precision inference

- Add option to interpolate model outputs to a given size to all of the segmentation models.

- Add DINOv2 Backbone

- Add support for `.geojson`, `.feather`, `.parquet` file formats when running inference.

## Docs

- Add `CPP-Net` example trainng with Pannuke dataset.

## Fixes

- Fix resize transformation bug.

<a id='changelog-0.1.23'></a>
# 0.1.23 — 2023-08-28

## Features

- add a stem-skip module. (Long skip for the input image resolution feature map)

- add UnetTR transformer encoder wrapper class
- add a new Encoder wrapper for timm and unetTR based encoders

- Add stem skip support and upsampling block options to all current model architectures

- Add masking option to all the criterions
- Add `MAELoss`
- Add `BCELoss`

- Add base class for transformer based backbones
- Add SAM-VitDet image encoder with support to load pre-trained SAM weights

- Add `CellVIT-SAM` model.

## Docs

- Add notebook example on training Hover-Net with lightning from scratch.

- Add notebook example on training StarDist with lightning from scratch.
- Add notebook example on training CellPose with accelerate from scratch.
- Add notebook example on training OmniPose with accelerate from scratch.

- Add notebook example on finetuning CellVIT-SAM with accelerate.

## Fixes

- Fix current TimmEncoder to store feature info

- Fix Up block to support transconv and bilinear upsampling and fix data flow issues.

- Fix StardistUnet class to output all the decoder features.

- Fix Decoder, DecoderStage and long-skip modules to work with up scale factors  instead of output dimensions.

<a id='changelog-0.1.22'></a>
# 0.1.22 — 2023-07-10

## Features

- Add mps (Mac) support for inference
- Add cell class probabilities to saved geojson files

<a id='changelog-0.1.21'></a>
# 0.1.21 — 2023-06-12

## Features

- Add StrongAugment data augmentation to data-loading pipeline: https://arxiv.org/abs/2206.15274

## Fixes

- Minor bug fixes
<a id='changelog-0.1.20'></a>

# 0.1.20 — 2023-01-13

## Fixes

- Enable writing folder & hdf5 datasets with only images (previously needed image-mask pairs)
- Enable writing datasets without patching.

- Add long missing h5 reading utility function to `FileHandler`

## Features

- Add hdf5 input file reading to `Inferer` classes.

- Add option to write pannuke dataset to h5 db in `PannukeDataModule` and `LizardDataModule`.

- Add a generic model builder function `get_model` to `models.__init__.py`

- Rewrite segmentation benchmarker. Now it can take in hdf5 datasets.

<a id='changelog-0.1.19'></a>

# 0.1.19 — 2023-01-04

## Features

- Add pytorch lightning in-built `auto_lr_finder` option to `SegmentationExperiment`

<a id='changelog-0.1.18'></a>

# 0.1.18 — 2023-01-03

## Features

- Add Multi-scale-convolutional-attention (MSCA) module (SegNexT).
- Add TokenMixer & MetaFormer modules.

<a id='changelog-0.1.17'></a>

# 0.1.17 — 2022-12-29

## Features

- Add transformer modules
- Add exact, slice, and memory efficient (xformers) self attention computations
- Add transformers modules to `Decoder` modules
- Add common transformer mlp activation functions: star-relu, geglu, approximate-gelu.
- Add Linformer self-attention mechanism.
- Add support for model intialization from yaml-file in `MultiTaskUnet`.
- Add a new cross-attention long-skip module. Works with `long_skip='cross-attn'`

## Refactor

- Added more verbose error messages for the abstract wrapper-modules in `modules.base_modules`
- Added more verbose error catching for xformers.ops.memory_efficient_attention.

<a id='changelog-0.1.16'></a>

# 0.1.16 — 2022-12-14

## Dependencies

- Bump old versions of numpy & scipy

<a id='changelog-0.1.15'></a>

# 0.1.15 — 2022-12-13

## Features

- Use the inferer class as input to segmentation benchmarker class

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
