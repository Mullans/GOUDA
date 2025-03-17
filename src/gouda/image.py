"""Methods/Shortcuts for modifying and handling image data."""

import os
import warnings

import matplotlib
import numpy as np
import numpy.typing as npt

from gouda import constants, data_methods, plotting
from gouda.goudapath import GoudaPath, GPathLike
from gouda.typing import ColorType, ImageArrayType

__author__ = "Sean Mullan"
__copyright__ = "Sean Mullan"
__license__ = "mit"

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover
    warnings.warn("OpenCV module not found - some image methods will raise exceptions")


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
    if isinstance(path, GoudaPath):
        path = path.path
    else:
        path = str(path)
    if not os.path.exists(path):
        raise ValueError(f"No file found at path '{path}'")
    if flag == constants.GRAYSCALE:
        return cv2.imread(path, 0)
    elif flag == constants.RGB:
        return cv2.imread(path)[:, :, ::-1]
    elif flag == constants.UNCHANGED:
        return cv2.imread(path, -1)
    else:
        return cv2.imread(path, 1)


def imwrite(path: GPathLike, image: ImageArrayType, as_RGB: bool = True):
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
    if isinstance(path, GoudaPath):
        path = path.path
    else:
        path = str(path)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
        as_RGB = False
    elif image.ndim == 3 and image.shape[2] == 1:
        as_RGB = False
    elif image.ndim == 3 and image.shape[2] == 3:
        pass
    else:
        raise ValueError(f"Image must be of shape [x, y, 1], [x, y, 3], or [x, y], not {image.shape}")
    if as_RGB:
        cv2.imwrite(path, image[:, :, ::-1])
    else:
        cv2.imwrite(path, image)


imsave = imwrite


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
        label = data_methods.to_uint8(label)
    if label_channel < 0:
        return np.dstack([label, label, label])
    elif label_channel < 3:
        to_stack = [np.zeros_like(label), np.zeros_like(label), np.zeros_like(label)]
        to_stack[label_channel] = label
        return np.dstack(to_stack)
    else:
        raise ValueError(f"Not a valid color channel index: {label_channel}")


def laplacian_var(image: npt.NDArray) -> npt.NDArray:
    """Return the laplacian variance of an image"""
    # Laplacian is the rate of change of pixel intensity (2nd order derivative)
    blur = cv2.GaussianBlur(image, (3, 3), 0, 0)
    grey = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(grey, cv2.CV_16S).var()


def sobel_var(image: npt.NDArray) -> npt.NDArray:
    """Return the sobal variance of an image"""
    # Sobel is the gradient of pixel intensity (1st order derivative)
    blur = cv2.GaussianBlur(image, (3, 3), 0, 0)
    grey = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(grey, cv2.CV_16S, 1, 0, 3)
    grad_x = cv2.convertScaleAbs(grad_x)
    grad_y = cv2.Sobel(grey, cv2.CV_16S, 0, 1, 3)
    grad_y = cv2.convertScaleAbs(grad_y)
    return cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0).var()


def split_signs(
    mask: npt.NDArray, positive_color: ColorType = (0.0, 1.0, 0.0), negative_color: ColorType = (1.0, 0.0, 0.0)
) -> npt.NDArray:
    """Split a single channel image mask with +/- values into color channels

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
    negative_image = np.dstack([negative * neg_channel for neg_channel in plotting.parse_color(negative_color)])
    positive_image = np.dstack([positive * pos_channel for pos_channel in plotting.parse_color(positive_color)])
    new_mask = negative_image + positive_image
    return new_mask


def masked_lineup(image, label):
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


def grabCut(image, labels, use_tresholds=True, thresholds=(0.2, 0.6, 0.8), iterations=2, clean=False):
    """Use the predicted mask and OpenCV's GrabCut algorithm to mask an image.

    Parameters
    ----------
    image : numpy.ndarray
        input image to mask
    labels : numpy.ndarray
        labels for background/foreground.
    use_thresholds: bool
        If true, thresholds the labels to find FG/BG components. Otherwise assumes that the values are [0, 2, 3, 1] defined by OpenCV for [BG, PR_BG, PR_FG, FG]
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
    cv2.grabCut(image, mask, None, bg, fg, iterations, cv2.GC_INIT_WITH_MASK)

    if clean:
        mask = clean_grabCut_mask(mask)
    mask_out = np.where((mask == 1) + (mask == 3), 1, 0).astype(np.uint8)
    masked_img = cv2.bitwise_and(image, image, mask=mask_out)
    return masked_img, mask_out


def clean_grabCut_mask(mask):
    """Apply light smoothing to grabCut labels"""
    temp = np.copy(mask)
    change = np.where((temp == 2) + (temp == 3), 1, 0)
    remapped = np.zeros_like(temp)
    remapped[temp == 0] = 0
    remapped[temp == 1] = 3
    remapped[temp == 2] = 1
    remapped[temp == 3] = 2
    up = cv2.pyrUp(remapped)
    for i in range(15):
        up = cv2.medianBlur(up, 7)
    down = cv2.pyrDown(up)
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


def crop_to_mask(image, mask, with_label=False, smoothing=True):
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
        smooth_mask = cv2.GaussianBlur(mask.astype(np.uint8) * 255.0, (5, 5), 0)
        # mask[mask<0.8] = 0
        # mask[mask>=0.8] = 1
    else:
        smooth_mask = mask.astype(np.uint8) * 255.0
    masked_image = cv2.bitwise_and(image, image, mask=smooth_mask.astype(np.uint8))
    if with_label:
        return masked_image[x0:x1, y0:y1], mask[x0:x1, y0:y1]
    return masked_image[x0:x1, y0:y1]


def get_bounds(mask: np.ndarray, bg_val: float = 0, as_slice: bool = False) -> list[tuple[int, int]]:
    """Get the corners of the bounding box/cube for the given binary label

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
        axis_range = np.where(axis_check == True)  # noqa
        bounds.append([axis_range[0][0], axis_range[0][-1] + 1])
    if as_slice:
        bounds = tuple([slice(b[0], b[1]) for b in bounds])
    return bounds


def crop_to_content(image, return_bounds=False):
    """Crop image to only be as large as the contained image excluding black space."""
    if return_bounds:
        bounds = get_bounds(image, bg_val=0, as_slice=False)
        bounds_slice = tuple([slice(*b) for b in bounds])
        return image[bounds_slice], bounds
    else:
        return image[get_bounds(image, bg_val=0, as_slice=True)]


def rotate(img, degrees=90, allow_resize=True):
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
    (cX, cY) = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D((cX, cY), -degrees, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    if allow_resize:
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
    else:
        nW, nH = w, h
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # Fixes a 1-pixel offset courtesy of: https://github.com/opencv/opencv/issues/4585#issuecomment-397895187
    M[0, 2] += (M[0, 0] + M[0, 1] - 1) / 2
    M[1, 2] += (M[1, 0] + M[1, 1] - 1) / 2
    return cv2.warpAffine(img, M, (nW, nH)).astype(img.dtype)


def padded_resize(image, size=[960, 540], allow_rotate=True, interpolation=cv2.INTER_LINEAR):
    """Resize input image to given size, only padding as needed for aspect ratio.

    Parameters
    ----------
    image: image/path
        image/path to image to resize
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
    if isinstance(image, (str, GoudaPath)):
        image = imread(image, flag=constants.RGB)
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
    padded_image = cv2.resize(padded_image, (size[1], size[0]), interpolation).astype(data_type)
    if c == 1:
        # cv2.resize auto-squeezes images
        return padded_image[:, :, np.newaxis]
    return padded_image


def horizontal_flip(image):
    """This is for convenience, but it is ~2e-7s faster to just copy the source and do this in-line"""
    return image[:, ::-1, :]


def vertical_flip(image):
    """This is for convenience, but it is ~1e-7s faster to just copy the source and do this in-line"""
    return image[::-1, ...]


def adjust_gamma(image, gamma=1.0):
    """Adjust the gamma of the image."""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    return cv2.LUT(image, table)


def polar_to_cartesian(image, output_shape=None):
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

    output = cv2.remap(image, newx, newy, cv2.INTER_LINEAR)
    return output


def get_mask_border(mask, inside_border=True, border_thickness=2, kernel="ellipse"):
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
        border = (mask - cv2.erode(mask, element)).astype(mask_type)
    else:
        border = (cv2.dilate(mask, element) - mask).astype(mask_type)
    return border


def add_mask(image, mask, color="red", opacity=0.5, mask_threshold=0.5):
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
    if isinstance(image.flat[0], np.integer):
        scaler = np.iinfo(image.dtype).max
    elif isinstance(image.flat[0], np.floating):
        scaler = 1
    elif isinstance(image.flat[0], (np.bool_, bool)):
        image = image.astype(np.float32)
        scaler = 1
    else:
        scaler = np.max(image)  # pragma: no cover
    color = matplotlib.colors.to_rgb(color)
    if mask.dtype != bool:
        mask = mask > mask_threshold
    scaler = scaler * opacity
    bias = image[mask] * (1 - opacity)
    image[:, :, 0][mask] = color[0] * scaler + bias[:, 0]
    image[:, :, 1][mask] = color[1] * scaler + bias[:, 1]
    image[:, :, 2][mask] = color[2] * scaler + bias[:, 2]
    return image


# def add_overlay(image, mask, label_channel=0, separate_signs=False, opacity=0.5):
#     """Return image with a mask overlay. DEPRECATED - use add_mask instead.

#     Parameters
#     ----------
#         image: numpy.ndarray
#             Image array with shape (x, y, channels)
#         mask: numpy.ndarray
#             np array with shape (x, y) and range [0, 1]
#         label_channel : int
#             The color channel to use for the overlay if separate_signs is False (the default is 0)
#         separate_signs : bool
#             Whether to separate +/- values of the mask into green/red channels
#     """
#     warnings.warn('add_overlay has been deprecated, use add_mask instead', DeprecationWarning)
#     output = np.squeeze(image)
#     mask = np.squeeze(mask)
#     if mask.dtype == 'bool':
#         mask = mask.astype(np.float32)
#     if mask.shape[:2] != image.shape[:2]:
#         raise ValueError('Mask width/height does not match image width/height: {} != {}'.format(mask.shape[:2], image.shape[:2]))
#     if mask.ndim == 2:
#         if separate_signs:
#             mask = split_signs(mask)
#         else:
#             mask = stack_label(mask, label_channel, as_uint8=False)
#     elif mask.ndim == 3 and mask.shape[2] == 3:
#         pass
#     else:
#         raise ValueError("Mask must have shape [x, y] or [x, y, z], not {}".format(mask.shape))
#     if image.dtype == np.uint8:
#         mask = data_methods.to_uint8(mask)

#     if output.ndim == 2:
#         output = np.dstack([output, output, output])

#     overlay = cv2.addWeighted(output, 1 - opacity, mask, opacity, 0)
#     output = np.where(mask.sum(axis=2, keepdims=True) != 0, overlay, output)
#     return output


def fast_label(item):
    """A stripped-down, faster version of skimage.measure.label

    Note
    ----
    requires scipy, which is not a main requirement of the rest of GOUDA
    """
    from scipy.ndimage import _ni_label

    label_dest = np.empty(item.shape, dtype=np.uint16)
    structure = np.ones([3] * len(item.shape), dtype=bool)
    _ni_label._label(item, structure, label_dest)
    return label_dest


def mask_by_triplet(pred, lower_thresh=0.3, upper_thresh=0.75, area_thresh=2000, fast=True):
    """Convert a probability mask into a binary mask using multiple thresholds

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
    import skimage.measure

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
        for idx, count in zip(indices[1:], counts[1:]):
            if count > area_thresh:
                obj_mask[peaks == idx] = True
        final_mask = np.zeros(mask_shape, dtype=bool)
        for idx in np.unique(bases):
            if obj_mask[bases == idx].any():
                final_mask[bases == idx] = 1
    return final_mask.reshape(pred.shape)
