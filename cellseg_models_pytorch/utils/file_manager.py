from pathlib import Path
from typing import Union

import cv2
import numpy as np


class FileHandler:
    """Class for handling file I/O."""

    @staticmethod
    def read_img(path: Union[str, Path]) -> np.ndarray:
        """Read an image & convert from bgr to rgb. (cv2 reads imgs in bgr).

        Parameters
        ----------
            path : str or Path
                Path to the image file.

        Returns
        -------
            np.ndarray:
                The image. Shape (H, W, 3).
        """
        path = Path(path)
        return cv2.cvtColor(cv2.imread(path.as_posix()), cv2.COLOR_BGR2RGB)

    @staticmethod
    def write_img(path: Union[str, Path], img: np.ndarray) -> None:
        """Write an image.

        Parameters
        ----------
            path : str or Path
                Path to the image file.
            img : np.ndarray
                The image to be written.

        """
        path = Path(path)
        cv2.imwrite(path.as_posix(), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
