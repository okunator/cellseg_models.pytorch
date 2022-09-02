from typing import Callable, Dict, List

import numpy as np
from torch.utils.data import Dataset

try:
    from ..transforms.albu_transforms import (
        IMG_TRANSFORMS,
        INST_TRANSFORMS,
        NORM_TRANSFORMS,
        apply_each,
        compose,
        to_tensorv3,
    )
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "To use the `csmp.dataset` module, the albumentations lib is needed. "
        "Install with `pip install albumentations`"
    )

__all__ = ["TrainDatasetBase"]


class TrainDatasetBase(Dataset):
    def __init__(
        self,
        img_transforms: List[str],
        inst_transforms: List[str],
        normalization: str = None,
        return_inst: bool = True,
        return_type: bool = True,
        return_sem: bool = False,
        return_weight: bool = False,
        **kwargs,
    ) -> None:
        """Training dataset baseclass.

        Parameters
        ----------
            img_transforms : List[str]
                A list containing all the transformations that are applied to the input
                images and corresponding masks. Allowed ones: "blur", "non_spatial",
                "non_rigid", "rigid", "hue_sat", "random_crop", "center_crop", "resize"
            inst_transforms : List[str]
                A list containg all the transformations that are applied to only the
                instance labelled masks. Allowed ones: "cellpose", "contour", "dist",
                "edgeweight", "hovernet", "omnipose", "smooth_dist", "binarize"
            normalization : str, optional
                Apply img normalization after all the transformations. One of "minmax",
                "norm", "percentile", None.
            return_inst : bool, default=True
                If True, returns a binarized instance mask. (If the db contains these.)
            return_type : bool, default=True
                If True, returns a type mask. (If the db contains these.)
            return_sem : bool, default=False
                If True, returns a semantic mask, (If the db contains these.)
            return_weight : bool, default=False
                Include a nuclear border weight map in the output.
            **kwargs:
                Arbitrary key-word arguments for the transformations.
        """
        super().__init__()

        allowed = list(IMG_TRANSFORMS.keys())
        if not all([tr in allowed for tr in img_transforms]):
            raise ValueError(
                f"Wrong img transformation. Got: {img_transforms}. Allowed: {allowed}."
            )
        allowed = list(NORM_TRANSFORMS.keys()) + [None]
        if normalization not in allowed:
            raise ValueError(
                f"Wrong norm transformation. Got: {normalization}. Allowed: {allowed}."
            )

        allowed = list(INST_TRANSFORMS.keys())
        if not all([tr in allowed for tr in inst_transforms]):
            raise ValueError(
                f"Wrong inst transformation. Got: {inst_transforms}. Allowed: {allowed}"
            )

        # Return masks
        self.return_inst = return_inst
        self.return_type = return_type
        self.return_sem = return_sem

        # Set transformations
        img_transforms = [IMG_TRANSFORMS[tr](**kwargs) for tr in img_transforms]
        if normalization is not None:
            img_transforms.append(NORM_TRANSFORMS[normalization]())

        inst_transforms = [INST_TRANSFORMS[tr](**kwargs) for tr in inst_transforms]
        if return_inst:
            inst_transforms.append(INST_TRANSFORMS["binarize"]())

        if return_weight:
            inst_transforms.append(INST_TRANSFORMS["edgeweight"]())

        self.img_transforms = compose(img_transforms)
        self.inst_transforms = apply_each(inst_transforms)
        self.to_tensor = to_tensorv3()

    def _getitem(
        self, ix: int, read_input_func: Callable, **kwargs
    ) -> Dict[str, np.ndarray]:
        """Get item.

        Parameters
        ----------
            ix : int
                An index for the iterable dataset.
            read_input_func : Callable
                A function that reads the images and corresponding masks and returns
                the inputs in a dictionary format. Return example:
                {"image": np.ndarray, "inst_map": np.ndarray, "type_map": np.ndarray}

        Returns
        -------
            Dict[str, np.ndarray]:
                A dictionary containing all the augmented data patches.
                Keys are: "im", "inst", "type", "sem". Image shape: (B, 3, H, W).
                Mask shapes: (B, C_mask, H, W).

        """
        inputs = read_input_func(ix, self.return_type, self.return_sem)

        # wrangle inputs to albumentations format
        mask_names = [key for key in inputs.keys() if key != "image"]
        masks = [arr for key, arr in inputs.items() if key != "image"]
        data = dict(image=inputs["image"], masks=masks)

        # transform + convert to tensor
        aug = self.img_transforms(**data)
        aux = self.inst_transforms(image=aug["image"], inst_map=aug["masks"][0])
        data = self.to_tensor(image=aug["image"], masks=aug["masks"], aux=aux)

        # wrangle data to return format
        out = dict(image=data["image"])
        for m, n in zip(data["masks"], mask_names):
            out[n] = m

        for n, aux_map in aux.items():
            out[n] = aux_map

        # remove redundant target (not needed in downstream).
        if self.return_inst:
            out["inst"] = out["binary"]
            del out["binary"]
        else:
            del out["inst"]

        return out
