"""Methods to aid with visualizing images/masks."""

import numpy as np
import numpy.typing as npt

from gouda.data_methods import to_uint8
from gouda.image.masks import add_mask
from gouda.plotting import parse_color
from gouda.typing import ColorType

__all__ = ["masked_lineup", "split_signs", "stack_label"]


def stack_label(label: npt.NDArray, label_channel: int = 0, as_uint8: bool = True) -> npt.NDArray:
    """Convert 2d label to 3d.

    Parameters
    ----------
    label : np.ndarray
        2D label to stack
    label_channel : int
        The color channel for the label, or -1 to set all channels to the label (the default is 0)
    as_uint8 : bool
        Whether to convert the label to a uint8
    """
    label = np.squeeze(label)
    if as_uint8:
        label = to_uint8(label)
    if label_channel < 0:
        return np.dstack([label, label, label])
    elif label_channel < 3:
        to_stack = [np.zeros_like(label), np.zeros_like(label), np.zeros_like(label)]
        to_stack[label_channel] = label
        return np.dstack(to_stack)
    else:
        raise ValueError(f"Not a valid color channel index: {label_channel}")


def split_signs(
    mask: npt.NDArray,
    positive_color: ColorType = (0.0, 1.0, 0.0),
    negative_color: ColorType = (1.0, 0.0, 0.0),
) -> npt.NDArray:
    """Split a single channel image mask with +/- values into color channels.

    Parameters
    ----------
    mask : np.ndarray
        The mask to split into colors
    positive_color : see gouda.parse_color
        The color to use for the positive part of the mask
    negative_color : see gouda.parse_color
        The color to use for the negative part of the mask

    NOTE
    ----
    positive_color and negative_color can be any color format recognized by gouda.parse_color (which includes all formats recognized by matplotlib)
    """
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError("Single channel mask required for split_signs")
    negative = -np.clip(mask, -np.inf, 0)
    positive = np.clip(mask, 0, np.inf)
    negative_image = np.dstack([negative * neg_channel for neg_channel in parse_color(negative_color)])
    positive_image = np.dstack([positive * pos_channel for pos_channel in parse_color(positive_color)])
    new_mask: npt.NDArray = negative_image + positive_image
    return new_mask


def masked_lineup(image: npt.NDArray, label: npt.NDArray) -> list[npt.NDArray]:
    """Return a list of image, masked_image, mask.

    Note
    ----
        Image will not be changed, but the masked_image and mask will be uint8 arrays with range [0, 255].

    Parameters
    ----------
        image: np array
            Image array with shape (x, y, channels)
        label: np array
            np array with shape (x, y) and range [0, 1]
        norm_type: int | None
            Type of normalization to undo when masking the image. If None, it will
            be inferred based on the value range of the array.
    """
    return [
        image,
        add_mask(image, label),
        np.dstack([label * 255, np.zeros_like(label), np.zeros_like(label)]).astype(np.uint8),
    ]
