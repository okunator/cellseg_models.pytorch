"""Exercise the installed distribution outside the source checkout."""

import argparse
import importlib
import sys
from importlib.metadata import (
    Distribution,
    distribution,
    packages_distributions,
    version,
)
from pathlib import Path
from tempfile import TemporaryDirectory
from importlib.util import find_spec

import geopandas as gpd
import torch
from shapely.geometry import box

import cellseg_models_pytorch
from cellseg_models_pytorch.models.cellpose.cellpose_unet import cellpose_nuclei
from cellseg_models_pytorch.utils import FileHandler


EXTRA_IMPORTS = {
    "training": ("albumentations", "tables"),
    "onnx": ("onnx", "onnxscript"),
    "openslide": ("openslide", "openslide_bin"),
}


def check_optional_dependencies(extra: str | None) -> None:
    """Reject missing feature packages and dependencies from unrelated extras."""
    for feature, modules in EXTRA_IMPORTS.items():
        for module in modules:
            installed = find_spec(module) is not None
            if installed != (feature == extra):
                state = "Missing" if feature == extra else "Unexpected"
                raise RuntimeError(f"{state} optional package {module} for {extra!r}")


def check_training_extra(directory: Path) -> None:
    """Read actual HDF5 and folder samples with aligned image/mask augmentation."""
    import albumentations as A
    import numpy as np
    import tables
    from PIL import Image
    from scipy.io import savemat

    from cellseg_models_pytorch.torch_datasets import TrainDatasetFolder, TrainDatasetH5
    from cellseg_models_pytorch.transforms.albu_transforms import (
        ApplyEach,
        CellposeTransform,
    )

    image = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    labels = np.zeros((12, 16), dtype=np.int32)
    labels[2:8, 3:7] = 7
    path = directory / "samples.h5"
    with tables.open_file(path, "w") as h5:
        h5.create_array("/", "image", image[None])
        h5.create_array("/", "inst", labels[None])
    images, masks = directory / "images", directory / "masks"
    images.mkdir()
    masks.mkdir()
    Image.fromarray(image).save(images / "sample.png")
    savemat(masks / "sample.mat", {"inst": labels})
    transforms = A.Compose([A.HorizontalFlip(p=1)])
    targets = ApplyEach([CellposeTransform()], as_list=True)
    datasets = (
        TrainDatasetH5(str(path), "image", ("inst",), ("inst",), transforms, targets),
        TrainDatasetFolder(str(images), str(masks), ("inst",), transforms, targets),
    )
    for dataset, target in zip(datasets, ("inst_cellpose", "cellpose_inst")):
        assert len(dataset) == 1
        sample = dataset[0]
        np.testing.assert_array_equal(
            sample["image"], image[:, ::-1].transpose(2, 0, 1)
        )
        np.testing.assert_array_equal(sample["inst"], labels[:, ::-1])
        assert sample["image"].dtype == torch.float32
        if isinstance(dataset, TrainDatasetH5):
            assert sample["inst"].dtype == torch.int64
        assert sample[target].shape == (2, 12, 16)
        assert torch.isfinite(sample[target]).all()


def check_onnx_extra(model: torch.nn.Module, directory: Path) -> None:
    """Export a real network and check the saved graph without ONNX Runtime."""
    import onnx

    from cellseg_models_pytorch.models.cellpose import export_cellpose_onnx

    path = export_cellpose_onnx(
        model, directory / "model.onnx", (1, 3, 64, 64), dynamic_batch=False
    )
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    assert [output.name for output in graph.graph.output] == ["flow_map", "type_map"]
    for output, channels in zip(graph.graph.output, (2, 3)):
        assert output.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        assert [dim.dim_value for dim in output.type.tensor_type.shape.dim] == [
            1,
            channels,
            64,
            64,
        ]


def check_openslide_extra(directory: Path) -> None:
    """Read exact deflate TIFF pixels through CSMP and native OpenSlide."""
    import numpy as np
    import tifffile

    from cellseg_models_pytorch.wsi import SlideReader

    image = np.arange(64 * 96 * 3, dtype=np.uint8).reshape(64, 96, 3)
    path = directory / "slide.tiff"
    tifffile.imwrite(
        path,
        image,
        tile=(16, 16),
        photometric="rgb",
        metadata=None,
        compression="deflate",
    )
    reader = SlideReader(path, backend="OPENSLIDE")
    assert reader.level_dimensions[0] == (64, 96)
    pixels = reader.read_region((7, 11, 23, 19), level=0)
    np.testing.assert_array_equal(pixels, image[11:30, 7:30])
    assert pixels.dtype == np.uint8
    del reader


def check_license_files(package: Distribution) -> None:
    """Require the shipped notices and full texts, including nonempty contents."""
    for name in (
        "LICENSE",
        "NOTICE.md",
        "LICENSES/Apache-2.0.txt",
        "LICENSES/Cellpose.txt",
        "LICENSES/HistoPrep.txt",
        "LICENSES/HoVer-Net.txt",
        "LICENSES/StarDist.txt",
        "LICENSES/scikit-image.txt",
        "LICENSES/WS-BCN.txt",
        "LICENSES/UvA-DL.txt",
        "LICENSES/LovaszSoftmax.txt",
        "LICENSES/InstanSeg.txt",
        "LICENSES/StrongAugment.txt",
        "LICENSES/SSIM.txt",
    ):
        paths = [
            package.locate_file(entry)
            for entry in package.files or ()
            if str(entry).endswith(f".dist-info/licenses/{name}")
        ]
        if (
            len(paths) != 1
            or not paths[0].is_file()
            or not paths[0].read_text().strip()
        ):
            raise RuntimeError(f"Missing or empty installed license file: {name}")


def main(extra: str | None = None) -> None:
    """Check package identity, public imports, and a CPU forward/backward pass."""
    installed_path = Path(cellseg_models_pytorch.__file__).resolve()
    if not installed_path.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"Expected an installed package, got {installed_path}")
    if version("cellseg_models_pytorch") != cellseg_models_pytorch.__version__:
        raise RuntimeError("Installed metadata and module versions disagree")
    providers = packages_distributions().get("cv2", [])
    if len(providers) != 1:
        raise RuntimeError(f"Expected one OpenCV provider, got {providers}")

    check_optional_dependencies(extra)
    check_license_files(distribution("cellseg_models_pytorch"))

    for module in (
        "models.cellpose",
        "models.stardist",
        "models.hovernet",
        "models.cellvit",
        "models.instanseg",
        "metrics",
        "losses",
        "inference.predictor",
        "inference.wsi_segmenter",
        "torch_datasets",
        "utils",
        "wsi",
    ):
        importlib.import_module(f"cellseg_models_pytorch.{module}")

    geometry = gpd.GeoDataFrame(
        {"uid": [101], "class_name": ["nucleus"]},
        geometry=[box(10, 20, 14, 25)],
        crs="EPSG:4326",
    )
    readers = {
        "parquet": gpd.read_parquet,
        "feather": gpd.read_feather,
        "geojson": gpd.read_file,
    }
    with TemporaryDirectory() as directory:
        for suffix, reader in readers.items():
            path = Path(directory) / f"instances.{suffix}"
            FileHandler.gdf_to_file(geometry, path)
            result = reader(path)
            if (
                result["uid"].tolist() != [101]
                or result["class_name"].tolist() != ["nucleus"]
                or result.crs != geometry.crs
                or not result.geometry.iloc[0].equals(geometry.geometry.iloc[0])
            ):
                raise RuntimeError(
                    f"Installed {suffix} geometry round trip changed data"
                )

    torch.manual_seed(0)
    torch.set_num_threads(2)
    model = cellpose_nuclei(3, enc_name="resnet18", enc_pretrain=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    output = model(torch.rand(2, 3, 64, 64))["nuc"]
    if output.type_map.shape != (2, 3, 64, 64):
        raise RuntimeError(f"Unexpected type output: {output.type_map.shape}")
    if output.aux_map.shape != (2, 2, 64, 64):
        raise RuntimeError(f"Unexpected flow output: {output.aux_map.shape}")
    loss = output.type_map.square().mean() + output.aux_map.square().mean()
    if not torch.isfinite(loss):
        raise RuntimeError("CPU training loss is not finite")
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(grad).all() for grad in gradients):
        raise RuntimeError("CPU model gradients are missing or nonfinite")
    optimizer.step()
    with TemporaryDirectory() as directory:
        path = Path(directory)
        if extra == "training":
            check_training_extra(path)
        elif extra == "onnx":
            check_onnx_extra(model, path)
        elif extra == "openslide":
            check_openslide_extra(path)
    print(f"Validated {installed_path} with a CPU training step")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra", choices=tuple(EXTRA_IMPORTS))
    main(parser.parse_args().extra)
