"""IO image operations."""

import os

import cv2
import numpy as np
import numpy.typing as npt

from gouda import constants
from gouda.goudapath import GPathLike
from gouda.typing import ImageArrayType

__all__ = ["imread", "imwrite"]


def imread(path: GPathLike, flag: int = constants.RGB) -> npt.NDArray:
    """Shortcut method: Load an image from a path using OpenCV modified for RGB.

    Parameters
    ----------
    path: GPathLike
        Path to image file
    flag: int
        The way to read the image (the default is :data:`gouda.constants.RGB`)

    Note
    ----
    * Valid flags are :data:`gouda.constants.UNCHANGED`, :data:`gouda.constants.RGB`, and :data:`gouda.constants.GRAYSCALE`. Any other flags will perform the default opencv.imread without additional arguments
    * Grayscale transforms input image based on perceived color. See [https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor]
    """
    path = str(path)
    if not os.path.exists(path):
        raise ValueError(f"No file found at path '{path}'")
    elif not os.access(path, os.R_OK):
        raise ValueError(f"File at path '{path}' is not readable")
    image: npt.NDArray | None
    if flag == constants.GRAYSCALE:
        image = cv2.imread(path, 0)
    elif flag == constants.RGB:
        image = cv2.imread(path)
        if image is not None:
            image = image[:, :, ::-1]
    elif flag == constants.UNCHANGED:
        image = cv2.imread(path, -1)
    else:
        image = cv2.imread(path, 1)
    if image is None:
        raise ValueError(f"OpenCV could not read the image at path '{path}'")
    return image


def imwrite(path: GPathLike, image: ImageArrayType, as_rgb: bool = True) -> None:
    """Shortcut method: Write an image to a path using OpenCV modified for RGB.

    Parameters
    ----------
    path: GPathLike
        Path to save image file to
    image: FloatArrayType
        image data to save - must be uint8/uint16 and with shape [x, y], [x, y, 1], or [x, y, 3]
    as_RGB: bool
        If true, flips the channels before saving (OpenCV assumes BGR image by default)
    """
    path = str(path)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
        as_rgb = False
    elif image.ndim == 3 and image.shape[2] == 1:
        as_rgb = False
    elif image.ndim == 3 and image.shape[2] == 3:
        pass
    else:
        raise ValueError(f"Image must be of shape [x, y, 1], [x, y, 3], or [x, y], not {image.shape}")
    if as_rgb:
        cv2.imwrite(path, image[:, :, ::-1])
    else:
        cv2.imwrite(path, image)
