"""Methods to work with image masks."""

from typing import Literal, overload

import cv2
import numpy as np
import numpy.typing as npt

from gouda.color_lists import find_color_rgb
from gouda.typing import ColorType, ImageArrayType

__all__ = [
    "add_mask",
    "clean_grabCut_mask",
    "crop_to_content",
    "crop_to_mask",
    "fast_label",
    "get_bounds",
    "get_mask_border",
    "grabCut",
    "mask_by_triplet",
]


def add_mask(
    image: npt.NDArray,
    mask: npt.NDArray,
    color: str | ColorType = "red",
    opacity: float = 0.5,
    mask_threshold: float = 0.5,
) -> npt.NDArray:
    """Add a binary outline/mask over a given image.

    Parameters
    ----------
    image: numpy.ndarray | list
        The image(s) to add the outline to
    mask: numpy.ndarray
        A binary mask to overlay on the image(s)
    color: str
        A matplotlib color to use for the overlay (the default is 'red')
    opacity: float
        The opacity to use when overlaying the mask (the default is 0.5)
    mask_threshold: float
        The threshold to use if a non-boolean mask is used

    NOTE
    ----
    The colors use a maximum value of 1.0 for floating type images, the maximum for the given integer type for integer type images, and the maximum present value for any other types.
    Boolean images will be converted to float32 images with values of 0.0 and 1.0
    """
    # TODO - allow for colors with alpha channel
    if isinstance(image, list):
        return [add_mask(item, mask, color=color, opacity=opacity) for item in image]
    if opacity < 0.0 or opacity > 1.0:
        raise ValueError("opacity must be between 0.0 and 1.0")
    image = np.squeeze(image).copy()
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError("Only 2-dimensional binary masks can be used")
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("image and outline must have the same height and width")
    if image.ndim == 2:
        image = np.dstack([image] * 3)
    scaler: float
    if isinstance(image.flat[0], np.integer):
        scaler = np.iinfo(image.dtype).max
    elif isinstance(image.flat[0], np.floating):
        scaler = 1.0
    elif isinstance(image.flat[0], (bool, np.bool_)):
        image = image.astype(np.float32)
        scaler = 1.0
    else:
        scaler = np.max(image)  # pragma: no cover
    color = find_color_rgb(color)
    if mask.dtype != bool:
        mask = mask > mask_threshold
    scaler = scaler * opacity
    bias = image[mask] * (1 - opacity)
    image[:, :, 0][mask] = color[0] * scaler + bias[:, 0]
    image[:, :, 1][mask] = color[1] * scaler + bias[:, 1]
    image[:, :, 2][mask] = color[2] * scaler + bias[:, 2]
    return image


def crop_to_mask(
    image: ImageArrayType, mask: npt.NDArray, with_label: bool = False, smoothing: bool = True
) -> npt.NDArray[np.uint8] | tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Crop input image to only be size of input mask.

    Parameters
    ----------
        image: numpy uint8 array
            Image to mask/crop
        mask: numpy bool array
            Masking array with same x and y shape as image
        with_label: bool
            Whether or not to return a version of the mask that is cropped to the new boundaries
        smoothing: bool
            Whether or not to smooth the border of the mask

    Returns
    -------
        numpy uint8 array
            Masked image with dimensions that are only as large
            as the positive region of the mask
    """
    vert = np.any(mask, axis=0)
    y_range = np.where(vert == True)  # noqa: E712
    horiz = np.any(mask, axis=1)
    x_range = np.where(horiz == True)  # noqa: E712
    x0, x1 = x_range[0][0], x_range[0][-1]
    y0, y1 = y_range[0][0], y_range[0][-1]
    if smoothing:
        smooth_mask: npt.NDArray[np.uint8] = cv2.GaussianBlur((mask * 255.0).astype(np.uint8), (5, 5), 0).astype(
            np.uint8
        )
        # mask[mask<0.8] = 0
        # mask[mask>=0.8] = 1
    else:
        smooth_mask = (mask * 255.0).astype(np.uint8)
    masked_image: npt.NDArray[np.uint8] = cv2.bitwise_and(image, image, mask=smooth_mask).astype(np.uint8)
    if with_label:
        return masked_image[x0:x1, y0:y1], mask[x0:x1, y0:y1]
    return masked_image[x0:x1, y0:y1]


@overload
def get_bounds(mask: np.ndarray, bg_val: float = ..., as_slice: Literal[False] = ...) -> list[tuple[int, int]]: ...


@overload
def get_bounds(mask: np.ndarray, bg_val: float = ..., as_slice: Literal[True] = ...) -> tuple[slice, ...]: ...


def get_bounds(
    mask: np.ndarray, bg_val: float = 0, as_slice: bool = False
) -> list[tuple[int, int]] | tuple[slice, ...]:
    """Get the corners of the bounding box/cube for the given binary label.

    Returns
    -------
    List[Tuple[int, int]]
        A list of the [start, stop) indices for each axis - NOTE: inclusive start and exclusive stop
    """
    bounds = []
    if bg_val != 0:
        mask = mask != bg_val
    for i in range(mask.ndim):
        axis_check = np.any(mask, axis=tuple([j for j in range(mask.ndim) if j != i]))
        axis_range: npt.NDArray[np.integer] = np.where(axis_check == True)[0]  # noqa
        bounds.append((int(axis_range[0]), int(axis_range[-1]) + 1))
    if as_slice:
        return tuple([slice(b[0], b[1]) for b in bounds])
    return bounds


def crop_to_content(
    image: npt.NDArray, return_bounds: bool = False
) -> npt.NDArray | tuple[npt.NDArray, list[tuple[int, int]]]:
    """Crop image to only be as large as the contained image excluding black space."""
    if return_bounds:
        bounds: list[tuple[int, int]] = get_bounds(image, bg_val=0, as_slice=False)
        bounds_slice = tuple([slice(b[0], b[1]) for b in bounds])
        return image[bounds_slice], bounds
    else:
        return image[get_bounds(image, bg_val=0, as_slice=True)]


def get_mask_border(
    mask: npt.NDArray, inside_border: bool = True, border_thickness: int = 2, kernel: str | int = "ellipse"
) -> npt.NDArray:
    """Get the border of a boolean mask.

    mask: np.ndarray
        The mask to get the border from
    inside_border: bool
        If true, uses pixels inside the mask as the border, otherwise uses pixels outside the mask (the default is true)
    border_thickness: int
        The thickness of the border in pixels (the default is 2)
    kernel: str | cv2.MorphShapes enum
        The kernel shape to use for the morphological operation (the default is 'elipse')

    NOTE
    ----
    kernel options are ['rect', 'cross', 'ellipse'] or any cv2.MorphShapes enum
    """
    if isinstance(kernel, str):
        kernel = {"rect": cv2.MORPH_RECT, "cross": cv2.MORPH_CROSS, "ellipse": cv2.MORPH_ELLIPSE}[kernel]
    mask_type = mask.dtype
    mask = mask.astype(np.float32)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_thickness * 2 + 1, border_thickness * 2 + 1))
    if inside_border:
        border: npt.NDArray = (mask - cv2.erode(mask, element)).astype(mask_type)
    else:
        border = (cv2.dilate(mask, element) - mask).astype(mask_type)
    return border


def grabCut(
    image: ImageArrayType,
    labels: npt.NDArray[np.integer],
    thresholds: tuple[float, float, float] = (0.2, 0.6, 0.8),
    iterations: int = 2,
    clean: bool = False,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Use the predicted mask and OpenCV's GrabCut algorithm to mask an image.

    Parameters
    ----------
    image : numpy.ndarray
        input image to mask
    labels : numpy.ndarray
        labels for background/foreground.
    thresholds : (int, int, int)
        Thesholds to divide [BG, PR_BG, PR_FG, FG] in the input labels
    iterations : int
        Number of GrabCut iterations to perform (the default is 2).
    clean : bool
        Should the GrabCut results be smoothed using :func:`~gouda.image.clean_grabCut_mask` (the default is False).

    Returns
    -------
    The GrabCut masked image and the generated mask : numpy.ndarray, numpy.ndarray

    """
    if labels.max() <= 1:
        mask = np.zeros_like(labels, dtype=np.uint8)
        mask += cv2.GC_BGD
        mask[labels > thresholds[0]] = cv2.GC_PR_BGD
        mask[labels > thresholds[1]] = cv2.GC_PR_FGD
        mask[labels > thresholds[2]] = cv2.GC_FGD
    else:
        mask = labels.astype(np.uint8)

    if cv2.GC_FGD not in mask and cv2.GC_PR_FGD not in mask:
        raise ValueError("Labels cannot all be background")
    elif cv2.GC_BGD not in mask and cv2.GC_PR_BGD not in mask:
        raise ValueError("Labels cannot all be foreground")

    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(
        image, mask=mask, rect=[0, 0, 0, 0], bgdModel=bg, fgdModel=fg, iterCount=iterations, mode=cv2.GC_INIT_WITH_MASK
    )

    if clean:
        mask = clean_grabCut_mask(mask)
    mask_out = np.where((mask == 1) + (mask == 3), 1, 0).astype(np.uint8)
    masked_img = cv2.bitwise_and(image, image, mask=mask_out)
    return masked_img, mask_out


def clean_grabCut_mask(mask: npt.NDArray) -> npt.NDArray:
    """Apply light smoothing to grabCut labels.

    #TODO - Make this generic for any mask values
    """
    temp = np.copy(mask)
    change = np.where((temp == 2) + (temp == 3), 1, 0)
    remapped = np.zeros_like(temp)
    remapped[temp == 0] = 0
    remapped[temp == 1] = 3
    remapped[temp == 2] = 1
    remapped[temp == 3] = 2
    up = cv2.pyrUp(remapped)
    for _ in range(15):
        up = cv2.medianBlur(up, 7)
    down: npt.NDArray = cv2.pyrDown(up)
    down = np.round(down)
    if temp.ndim == 3:
        down = down[:, :, np.newaxis]
    remapped = np.where(change == 1, down, remapped)
    output = np.zeros_like(temp)
    output[remapped == 0] = 0
    output[remapped == 3] = 1
    output[remapped == 1] = 2
    output[remapped == 2] = 3
    return output.reshape(mask.shape).astype(mask.dtype)


def fast_label(item: npt.NDArray) -> npt.NDArray:
    """Run a stripped-down, faster version of skimage.measure.label .

    Note
    ----
    requires scipy, which is not a main requirement of the rest of GOUDA
    """
    # mypy can't find this for some reason
    from scipy.ndimage._measurements import _ni_label  # type: ignore[attr-defined]  # noqa: PLC0415

    label_dest = np.empty(item.shape, dtype=np.uint16)
    structure = np.ones([3] * len(item.shape), dtype=bool)
    _ni_label._label(item, structure, label_dest)
    return label_dest


def mask_by_triplet(
    pred: npt.NDArray[np.floating],
    lower_thresh: float = 0.3,
    upper_thresh: float = 0.75,
    area_thresh: int | float = 2000,
    fast: bool = True,
) -> npt.NDArray[np.bool_]:
    """Convert a probability mask into a binary mask using multiple thresholds.

    Parameters
    ----------
    pred : numpy.ndarray
        1-3 dimensional array with continuous values
    lower_thresh : float
        The minimum threshold for potential foreground values (the default is 0.3)
    upper_thresh : float
        The minimum threshold of peak foreground values (the default is 0.75)
    area_thresh : float
        The minimum size a peak object needs to be for its base to be considered foreground (the default is 2000)
    fast : bool
        Whether to use the faster version that has less type-checking involved

    NOTE
    ----
    Individual peaks and bases are identified by the given thresholds. If a peak object has the minimum area, then the base object that it is a part of is considered to be foreground in the final mask.

    NOTE
    ----
    Using an area_thresh of 0 is equivalent to hysterisis thresholding

    Note
    ----
    requires scikit-image which is not a requirement of the rest of GOUDA
    """
    # TODO - Add tests
    import skimage.measure  # noqa: PLC0415

    if 0.0 <= area_thresh < 1.0:
        area_thresh = pred.size * area_thresh

    flat_pred = np.squeeze(pred)
    mask_shape = np.shape(flat_pred)

    if fast:
        peaks = fast_label(flat_pred > upper_thresh)
        bases = fast_label(flat_pred > lower_thresh)
        keep_idx = [idx for idx, count in enumerate(np.bincount(peaks.ravel())) if count > area_thresh][1:]
        obj_mask = np.isin(peaks, keep_idx)
        valid = bases * obj_mask
        valid_idx = np.nonzero(np.bincount(valid.ravel()))[0][1:]
        final_mask = np.isin(bases, valid_idx)
    else:
        peaks = skimage.measure.label(flat_pred > upper_thresh)
        bases = skimage.measure.label(flat_pred > lower_thresh)
        indices, counts = np.unique(peaks, return_counts=True)
        obj_mask = np.zeros(mask_shape, dtype=bool)
        for idx, count in zip(indices[1:], counts[1:], strict=True):
            if count > area_thresh:
                obj_mask[peaks == idx] = True
        final_mask = np.zeros(mask_shape, dtype=bool)
        for idx in np.unique(bases):
            if obj_mask[bases == idx].any():
                final_mask[bases == idx] = 1
    return final_mask.reshape(pred.shape)
