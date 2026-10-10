# Draft module boundaries and dependency groups

Status: deferred structural work. Keep this proposal separate from dependency
version bumps; review it after maintenance and before changing import contracts.

Aim for one-way dependencies so a model can run without slide-reading or export
packages, and a slide reader can run without model or training packages. Shared
NumPy/Torch representations are legitimate dependencies. Avoid cycles and eager
imports of unrelated optional features rather than promising complete isolation.

| Area | Responsibility | Proposed upgrade group |
| --- | --- | --- |
| Model core | Architectures, tensor prediction, checkpoint loading | Torch, torchvision, timm, safetensors; checkpoint download dependencies reviewed separately |
| Numerical operations | Shared array/tensor helpers, metrics, transforms | NumPy, SciPy, Numba, scikit-image, OpenCV; coordinate changes with affected consumers |
| Reconstruction | Dense predictions to instance labels/classes | Existing model-specific algorithms and numerical group; no WSI or file-writing imports |
| WSI reading | Pixels, tile coordinates, calibration and slide backends | Individual optional OpenSlide, cuCIM, BioIO backends; minimal shared image/array dependencies |
| Geometry and export | Vectorization, instance merging, serialization | GeoPandas, PyArrow, Shapely, Rasterio, libpysal, pyogrio; pandas is shared and needs its own compatibility decision |
| Training | Training datasets and augmentation | Optional augmentation and HDF5 packages, independently validated |
| Orchestration | Connect readers, models, reconstruction and exports | May depend on the selected features; no lower-level module imports orchestration |
| Developer tools | Lint, types, tests, docs and builds | Separate development upgrades |

These are upgrade/review groups, not new Poetry groups or published extras yet.
Poetry development groups control developer installs; runtime extras must be
backed by real import boundaries and missing-dependency behavior.

## Current couplings to audit

High-level models import PostProcessor, which imports geospatial vectorization
and file writing. Importing a shared utility package loads its broad initializer,
including file handling. WSI data/tiling helpers and merging also use geometry
packages. Inspect these transitive chains before calling either area independent.
The repaired training dataset modules demonstrate dependency checks in the layer
that actually needs the optional package; keep ordinary public exports.

## Small staged plan

1. Inventory imports, including package initializers and annotation-only imports.
   Write the allowed dependency directions and identify actual cycles/couplings.
2. Separate numerical reconstruction from geometry/file export at their owning
   modules. Retain existing public exports and representations; move shared
   output contracts only when necessary to remove a proven dependency cycle.
3. Establish a minimal model install and independent reader/backend installs.
   Define runtime extras only after each required package and import path is known.
4. Check dependency availability at feature use, with actionable errors. Do not
   hide the coupling through package-level attribute interception, broad exception
   handling, or a new generic plugin/registry framework.
5. Test isolated installs: model prediction without WSI/export/training packages;
   reading without models/training; export without reader backends; training with
   its declared requirements. Include genuine values/contracts, not imports alone.
6. Land each boundary separately from version updates. Retest public APIs,
   checkpoint compatibility and outputs before removing mandatory requirements.

Success means independent supported feature installs with explicit dependencies,
ordinary exports, preserved error handling, and no accidental reverse imports.
GPU reconstruction remains a separate deferred proposal. Do not reorganize the
repository or invent service layers just to match this table.
