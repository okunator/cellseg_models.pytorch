from typing import List

import albumentations as A
import cv2
from albumentations.core.transforms_interface import BasicTransform

__all__ = [
    "rigid_transforms",
    "non_rigid_transforms",
    "non_spatial_transforms",
    "hue_saturation_transforms",
    "blur_transforms",
    "random_crop",
    "center_crop",
    "resize",
]


def rigid_transforms(**kwargs) -> List[BasicTransform]:
    """Return rigid albumentations augmentations.

    For every patch, either:
    - Rotate
    - random rotate 90 degrees
    - flip (rotate 180 degrees)
    - transpose (flip x and y axis)
    is applied with a probability of 0.5*(0.5/(0.5+0.5+0.5+0.5))=0.125

    Returns
    -------
        List[BasicTransform]:
            A List of possible albumentations data augmentations.
    """
    return [
        A.OneOf(
            [
                A.Rotate(p=0.8),
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.Transpose(p=0.5),
            ],
            p=0.5,
        )
    ]


def non_rigid_transforms(**kwargs) -> List[BasicTransform]:
    """Return non rigid albumentations augmentations.

    For every patch, either:
    - elastic transformation
    - grid distortion
    - optical distortion
    is applied with a probability of 0.3*(0.5/(0.5+0.5+0.5))=0.1

    Returns
    -------
        List[BasicTransform]:
            A List of possible albumentations data augmentations.
    """
    return [
        A.OneOf(
            [
                A.ElasticTransform(
                    alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
                ),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5),
            ],
            p=0.3,
        )
    ]


def hue_saturation_transforms(**kwargs) -> List[BasicTransform]:
    """Return hue saturation albumentations augmentations.

    For every patch, either:
    - hue saturation value shift is applied with a probability of 0.5*0.25*0.125

    Returns
    -------
        List[BasicTransform]:
            A List of possible albumentations data augmentations.
    """
    return [
        A.OneOf(
            [
                A.HueSaturationValue(
                    hue_shift_limit=(0, 15),
                    sat_shift_limit=(0, 30),
                    val_shift_limit=(0, 20),
                    p=0.25,
                )
            ],
            p=0.5,
        )
    ]


def blur_transforms(**kwargs) -> List[BasicTransform]:
    """Return blur albumentations augmentations.

    For every patch, either:
    - motion blur
    - median blur
    - gaussian blur
    is applied with a probability of 0.3*(0.5/(0.5+0.5+0.5))=0.1

    Returns
    -------
        List[BasicTransform]:
            A List of possible albumentations data augmentations.
    """
    return [
        A.OneOf(
            [
                A.MotionBlur(blur_limit=7, p=0.5),
                A.MedianBlur(blur_limit=7, p=0.5),
                A.Blur(blur_limit=7, p=0.5),
            ],
            p=0.3,
        )
    ]


def non_spatial_transforms(**kwargs) -> List[BasicTransform]:
    """Return non spatial albumentations augmentations.

    For every patch, either:
    - CLAHE
    - brightness contrast
    - random gamma
    is applied with a probability of 0.7*(0.5/(0.5+0.5+0.5))=0.233

    Returns
    -------
        List[BasicTransform]:
            A List of possible albumentations data augmentations.
    """
    return [
        A.OneOf(
            [A.CLAHE(p=0.5), A.RandomBrightnessContrast(p=0.5), A.RandomGamma(p=0.5)],
            p=0.5,
        )
    ]


def center_crop(height: int, width: int, **kwargs) -> List[BasicTransform]:
    """Return albumentations center crop.

    For every patch a crop a is extracted from the center, p=1.0.

    Parameters
    ----------
        height : int
            Height of the output image.
        width : int
            Width of the input image.

    Returns
    -------
        List[BasicTransform]:
            A List containing the center crop transform.
    """
    return [
        A.OneOf(
            [A.CenterCrop(height=height, width=width, always_apply=True, p=1)], p=1.0
        )
    ]


def random_crop(height: int, width: int, **kwargs) -> List[BasicTransform]:
    """Return albumentations random crop.

    For every patch a crop a is extracted randomly, p=1.0.

    Parameters
    ----------
        height : int
            Height of the output image.
        width : int
            Width of the input image.

    Returns
    -------
        List[BasicTransform]:
            A List containing the random crop transform.
    """
    return [
        A.OneOf(
            [A.RandomCrop(height=height, width=width, always_apply=True, p=1)], p=1.0
        )
    ]


def resize(height: int, width: int, **kwargs) -> List[BasicTransform]:
    """Return albumentations resize transform.

    Parameters
    ----------
        height : int
            Height of the output image
        width : int
            Width of the input image

    Returns
    -------
        List[BasicTransform]:
            A List containing the resize transform.
    """
    return [
        A.OneOf(
            [A.Resize(height=height, width=width, interpolation=cv2.INTER_CUBIC, p=1)]
        )
    ]
