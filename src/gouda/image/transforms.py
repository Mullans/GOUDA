"""Image transformations."""

from collections.abc import Sequence

import cv2
import numpy as np
import numpy.typing as npt

from gouda.typing import ImageArrayType

__all__ = ["adjust_gamma", "padded_resize", "polar_to_cartesian", "rotate"]


def rotate(img: ImageArrayType, degrees: int = 90, allow_resize: bool = True) -> npt.NDArray:
    """Rotate image clock-wise using OpenCV.

    Parameters
    ----------
    img: image (as required by OpenCV getRotationMatrix2D)
        image/path to rotate
    degrees: int
        Degrees to rotate the image clockwise
    allow_resize: bool
        If yes, the image boundaries will change to fit the rotated image. Otherwise the output rotated image is cropped to fit the original boundaries.
    """
    (h, w) = img.shape[:2]
    (center_x, center_y) = (w / 2, h / 2)
    mat = cv2.getRotationMatrix2D((center_x, center_y), -degrees, 1.0)
    cos = np.abs(mat[0, 0])
    sin = np.abs(mat[0, 1])
    if allow_resize:
        new_width = int((h * sin) + (w * cos))
        new_height = int((h * cos) + (w * sin))
    else:
        new_width, new_height = w, h
    mat[0, 2] += (new_width / 2) - center_x
    mat[1, 2] += (new_height / 2) - center_y
    # Fixes a 1-pixel offset courtesy of: https://github.com/opencv/opencv/issues/4585#issuecomment-397895187
    mat[0, 2] += (mat[0, 0] + mat[0, 1] - 1) / 2
    mat[1, 2] += (mat[1, 0] + mat[1, 1] - 1) / 2
    result: npt.NDArray = cv2.warpAffine(img, mat, (new_width, new_height)).astype(img.dtype)
    return result


def padded_resize(
    image: ImageArrayType,
    size: tuple[int, int] = (960, 540),
    allow_rotate: bool = True,
    interpolation: int = cv2.INTER_LINEAR,
) -> ImageArrayType:
    """Resize input image to given size, only padding as needed for aspect ratio.

    Parameters
    ----------
    image: image
        image to image to resize
    size: [int, int]
        output size
    allow_rotate: bool
        Whether the image can be rotated to minimize required padding or if orientation should be preserved
    interpolation: int
        The interpolation function to use when resizing the image (the default is cv2.INTER_LINEAR)

    Notes
    -----
    Input image number of channels does not matter as long as the first two dimensions are x and y.

    """
    data_type = image.dtype
    if image.ndim == 2:
        x, y = image.shape
        c = 0
        image = image[:, :, np.newaxis]
    else:
        x, y, c = image.shape
    if ((size[0] < size[1] and x > y) or (size[0] > size[1] and x < y)) and allow_rotate:
        image = rotate(image, 90)
        x, y = image.shape[:2]
    padx = int(y * (float(size[0]) / size[1])) - x
    pady = int(x * (float(size[1]) / size[0])) - y
    padded_image: npt.NDArray
    if abs(padx) > 10 or abs(pady) > 10:
        padx = 10000 if padx < 0 else padx
        pady = 10000 if pady < 0 else pady
        if padx < pady:
            new_shape = [x + padx, y, max(c, 1)]
            padded_image = np.zeros(new_shape, dtype=data_type)
            padded_image[int(padx // 2) : -int(padx - padx // 2), :] = image
        else:
            new_shape = [x, y + pady, max(c, 1)]
            padded_image = np.zeros(new_shape, dtype=data_type)
            padded_image[:, int(pady // 2) : -int(pady - pady // 2)] = image
    else:
        padded_image = image
    padded_image = cv2.resize(padded_image, dsize=(size[1], size[0]), interpolation=interpolation).astype(data_type)
    if c == 1:
        # cv2.resize auto-squeezes images
        return padded_image[:, :, np.newaxis]
    return padded_image


def adjust_gamma(image: ImageArrayType, gamma: float = 1.0) -> npt.NDArray[np.floating | np.integer]:
    """Adjust the gamma of the image."""
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    result: npt.NDArray[np.floating | np.integer] = cv2.LUT(image, table)
    return result


def polar_to_cartesian(
    image: ImageArrayType, output_shape: Sequence[int] | None = None
) -> npt.NDArray[np.floating | np.integer]:
    """Convert a square image with a polar object (circle/tube) to cartesian (unroll it).

    NOTE: output_shape uses numpy shape: [rows, columns]
    """
    if output_shape is None:
        output_shape = image.shape

    degrees = np.linspace(0, 360, output_shape[1], endpoint=False, dtype=np.float32)
    degrees = np.radians(degrees)
    center = image.shape[0] // 2
    radius = np.linspace(0, center, output_shape[0], endpoint=False, dtype=np.float32)
    d, r = np.meshgrid(degrees, radius)
    newx = r * np.cos(d) + center
    newy = r * np.sin(d) + center

    output: npt.NDArray[np.floating | np.integer] = cv2.remap(image, newx, newy, cv2.INTER_LINEAR)
    return output
