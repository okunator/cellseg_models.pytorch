from typing import Dict, List

from ._base_datamodule import BaseDataModule

__all__ = ["CustomDataModule"]


class CustomDataModule(BaseDataModule):
    def __init__(
        self,
        train_data_path: str,
        test_data_path: str,
        valid_data_path: str,
        img_transforms: List[str],
        inst_transforms: List[str],
        type_classes: Dict[str, int] = None,
        sem_classes: Dict[str, int] = None,
        class_weights: List[float] = None,
        normalization: str = None,
        return_weight: bool = False,
        return_inst: bool = True,
        return_type: bool = True,
        return_sem: bool = False,
        batch_size: int = 8,
        num_workers: int = 8,
        **kwargs,
    ) -> None:
        """Set up a custom datamodule.

        Parameters
        ----------
            train_data_path : str
                Path to train data folder or database.
            valid_data_path : str
                Path to validation data folder or database.
            test_data_path : str
                Path to the test data folder or database.
            img_transforms : List[str]
                A list containing all the transformations that are applied to the input
                images and corresponding masks. Allowed ones: "blur", "non_spatial",
                "non_rigid", "rigid", "hue_sat", "random_crop", "center_crop", "resize"
            inst_transforms : List[str]
                A list containg all the transformations that are applied to only the
                instance labelled masks. Allowed ones: "cellpose", "contour", "dist",
                "edgeweight", "hovernet", "omnipose", "smooth_dist", "binarize"
            type_classes : Dict[str, int], optional
                A dict containing classname-classnumber pairs. E.g. {"cl1": 0, "cl2": 1}
            sem_classes : Dict[str, int], optional
                A dict containing classname-classnumber pairs. E.g. {"cl1": 0, "cl2": 1}
                If None, no semantic map will be returned during dataloading.
            class_weights : List[float], optional:
                A list of contianing class-weights for each class.
            normalization : str, optional
                Apply img normalization after all the transformations. One of "minmax",
                "norm", "percentile", None.
            return_inst : bool, default=True
                If True, returns a binary gt mask.
            return_weight : bool, default=False
                Include a nuclear border weight map in the output.
            batch_size (int, default=8):
                Batch size for the dataloader
            num_workers (int, default=8):
                number of cpu cores/threads used in the dataloading
                process.
        """
        if class_weights is not None:
            if not all([0.0 <= c <= 1.0 for c in class_weights]):
                raise ValueError(
                    f"Class weights need to be b/w [0, 1]. Got {class_weights}"
                )
            if type_classes is not None:
                if len(class_weights) != len(type_classes.keys()):
                    raise ValueError(
                        "Same number of class weights and type classe is needed.",
                        f"Got {class_weights} and {type_classes}.",
                    )
        return_type = True if type_classes is not None else False
        return_sem = True if sem_classes is not None else False

        super().__init__(
            train_data_path=train_data_path,
            valid_data_path=valid_data_path,
            test_data_path=test_data_path,
            img_transforms=img_transforms,
            inst_transforms=inst_transforms,
            normalization=normalization,
            return_weight=return_weight,
            return_inst=return_inst,
            return_type=return_type,
            return_sem=return_sem,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.type_classes = type_classes
        self.sem_classes = sem_classes
