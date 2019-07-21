"""Methods/Shortcuts for modifying and handling image data."""
import cv2
import numpy as np

# from gouda import __version__

__author__ = "Sean Mullan"
__copyright__ = "Sean Mullan"
__license__ = "mit"

NO_NORM = 0
TANH_NORM = 1
SIGMOID_NORM = 2


def imread(path, as_RGB=True):
    """SHORTCUT: Load an image from a path using OpenCV modified for RGB."""
    if as_RGB:
        return cv2.imread(path)[:, :, ::-1]
    else:
        return cv2.imread(path)


def imwrite(path, image, as_RGB=True):
    """SHORTCUT: Write an image to a path using OpenCV modified for RGB."""
    if image.ndim == 2:
        image = np.dstack([image, image, image])
        as_RGB = False
    if as_RGB:
        cv2.imwrite(path, image[:, :, ::-1])
    else:
        cv2.imwrite(path, image)


def val_type(x):
    """Get the normalization type used for a numpy array."""
    if x.max() > 1 and x.max() <= 255 and x.min() >= 0:
        return NO_NORM  # Values are in range 0-255, no norm
    elif x.min() < 0 and x.min() >= -1 and x.max() <= 1:
        return TANH_NORM  # Values are in range [-1, 1], tanh norm
    elif x.min() >= 0 and x.max() <= 1:
        return SIGMOID_NORM  # Values are in range [0, 1], sigmoid norm
    else:
        raise ValueError(
            "Given values are not in one of the valid image normalization ranges."
        )


def denorm(x, norm_type=None):
    """Denormalize an image based on specified or inferred normalization type."""
    if norm_type is None:
        norm_type = val_type(x)
    if norm_type == 1:
        x = ((x * 127.5) + 127.5)
    elif norm_type == 2:
        x = (x * 255.0)
    return x.astype(np.uint8)


def rescale(data, column_wise=False):
    """Scale either an image or rows of data to [0, 1]"""
    if column_wise:
        range = data.max(axis=0) - data.min(axis=0)
        range[range == 0] = 1.0 / data.shape[0]
        return (data - data.min(axis=0)) / range
    else:
        range = data.max() - data.min() + 0.0
        if range == 0:
            return np.zeros_like(data)
        else:
            return (data - data.min()) / range


def stack_label(label, denorm=True):
    """Denormalize and change a 1d label to a 3d image.

    Note
    ----
        If denorm is True, return image is in range [0, 255].
    """
    x = np.dstack([label, label, label])
    if denorm:
        x = denorm(x)
    return x


def laplacian_var(image):
    """Return the laplacian variance of an image"""
    # Laplacian is the rate of change of pixel intensity (2nd order derivative)
    image = denorm(image)
    blur = cv2.GaussianBlur(image, (3, 3), 0, 0)
    grey = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(grey, cv2.CV_16S).var()


def sobel_var(image):
    """Return the sobal variance of an image"""
    # Sobel is the gradient of pixel intensity (1st order derivative)
    image = denorm(image)
    blur = cv2.GaussianBlur(image, (3, 3), 0, 0)
    grey = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(grey, cv2.CV_16S, 1, 0, 3)
    grad_x = cv2.convertScaleAbs(grad_x)
    grad_y = cv2.Sobel(grey, cv2.CV_16S, 0, 1, 3)
    grad_y = cv2.convertScaleAbs(grad_y)
    return cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0).var()


def mask(img, mask, overlay=True, norm_type=None, renorm=True):
    """Return img with the red channel replaced by mask.

    Parameters
    ----------
        img: np array
            Image array with shape (x, y, channels)
        mask: np array
            np array with shape (x, y) and range [0, 1]
        overlay: bool
            If true, the mask is overlaid on the image rather than replacing the red channel
        norm_type: int | None
            Type of normalization to undo on the input img. If None, it will be
            inferred based on the value range of the array.
        renorm: bool
            Should the output image be normalized to the same range as the input
    """
    output = np.copy(img)
    if mask.ndim == 2:
        mask = mask[:, :, np.newaxis]
    elif mask.shape[2] == 3:
        mask = np.mean(mask, axis=2, keepdims=True)
    if mask.max() > 1.0 or mask.min() < 0:
        raise ValueError('Mask values must be in range [0, 1].')
    mask = (mask * 255.0).astype(np.uint8)

    if output.ndim == 2 or (output.ndim == 3 and output.shape[2] == 1):
        # Image has to be RGB for output
        output = np.dstack([output, output, output])

    if norm_type is None:
        norm_type = val_type(output)
    if norm_type == 1:
        output = ((output * 127.5) + 127.5).astype(np.uint8)
    elif norm_type == 2:
        output = (output * 255.0).astype(np.uint8)

    if overlay:
        overlay = cv2.addWeighted(
            output,
            0.8,
            np.dstack([mask, np.zeros_like(mask), np.zeros_like(mask)]),
            0.2,
            0)
        output = np.where(mask >= 127.5, overlay, output)
    else:
        output[:, :, :1] = mask

    if renorm:
        if norm_type == 1:
            output = (output.astype(img.dtype) - 127.5) / 127.0
        if norm_type == 2:
            output = output.astype(img.dtype) / 255.0
    return output


def masked_lineup(image, label, norm_type=None):
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
        mask(image, label, norm_type),
        np.dstack([label * 255,
                   np.zeros_like(label),
                   np.zeros_like(label)]).astype(np.uint8)
    ]


def grabCut(image, labels, thresholds=(0.2, 0.6, 0.8), iterations=2):
    """Use the predicted mask and grabCut algorithm to mask an image."""
    image = denorm(image)
    if labels.max() <= 1:
        mask = np.zeros_like(image).astype(np.uint8)
        mask[labels > thresholds[0]] = cv2.GC_PR_BGD
        mask[labels > thresholds[1]] = cv2.GC_PR_FGD
        mask[labels > thresholds[2]] = cv2.GC_FGD
    else:
        mask = labels

    if labels.sum() == 0:
        raise ValueError("Labels cannot all be background")
    elif labels.sum() == labels.size():
        raise ValueError("Labels cannot all be foreground")

    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, None, bg, fg, iterations, cv2.GC_INIT_WITH_MASK)
    mask2 = np.where((mask == 1) + (mask == 3), 1, 0).astype(np.uint8)
    masked_img = cv2.bitwise_and(image, image, mask=mask2)
    return masked_img, mask2


def label_grabCut(image,
                  labels,
                  thresholds=(0, 0.6, 0.8),
                  iterations=2,
                  clean=False):
    """Use the predicted mask and grabCut to threshold a probability label to true/false"""
    image = denorm(image)

    mask = np.zeros_like(labels).astype(np.uint8)
    mask[labels > thresholds[0]] = cv2.GC_PR_BGD
    mask[labels > thresholds[1]] = cv2.GC_PR_FGD
    mask[labels > thresholds[2]] = cv2.GC_FGD

    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)

    cv2.grabCut(image, mask, None, bg, fg, iterations, cv2.GC_INIT_WITH_MASK)

    if clean:
        mask = clean_grabCut_mask(mask)

    mask2 = np.where((mask == 1) + (mask == 3), True, False)

    return mask2


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
        smooth_mask = cv2.GaussianBlur(
            mask.astype(np.uint8) * 255.0, (5, 5), 0)
        # mask[mask<0.8] = 0
        # mask[mask>=0.8] = 1
    else:
        smooth_mask = mask.astype(np.uint8) * 255.0
    masked_image = cv2.bitwise_and(image,
                                   image,
                                   mask=smooth_mask.astype(np.uint8))
    if with_label:
        return masked_image[x0:x1, y0:y1], mask[x0:x1, y0:y1]
    return masked_image[x0:x1, y0:y1]


def get_bounds(mask):
    """Return the bounding box corners of the positive content for the mask."""
    vert = np.any(mask, axis=0)
    horiz = np.any(mask, axis=1)
    y_range = np.where(vert == True)  # noqa: E712
    x_range = np.where(horiz == True)  # noqa: E712
    x0, x1 = x_range[0][0], x_range[0][-1]
    y0, y1 = y_range[0][0], y_range[0][-1]
    return (x0, y0), (x1, y1)


def crop_to_content(image):
    """Crop image to only be as large as the contained image."""
    vert = np.mean(image, axis=(-1, 0))
    y_range = np.where(vert > 0)
    horiz = np.mean(image, axis=(-1, 1))
    x_range = np.where(horiz > 0)
    x0, x1 = x_range[0][0], x_range[0][-1]
    y0, y1 = y_range[0][0], y_range[0][-1]
    return image[x0:x1, y0:y1]


def rotate(img, degrees=90):
    """Rotate image using OpenCV methods."""
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), degrees, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(img, M, (nW, nH))


def padded_resize(image, size=[960, 540], allow_rotate=True):
    """Resize input image to given size, only padding as needed for aspect ratio.

    Parameters
    ----------
    image: image
        image to resize
    size: [int, int]
        output size
    allow_rotate: bool
        Whether the image can be rotated to minimize required padding or if orientation should be preserved
    Input image number of channels does not matter as long as the first two dimensions are x and y.
    """
    if type(image) == str:
        image = imread(image, as_RGB=True)
    data_type = image.dtype
    x, y = image.shape[:2]
    if x < y and allow_rotate:
        image = rotate(image)
        x, y = image.shape[:2]
    padx = int(y * (float(size[0]) / size[1])) - x
    pady = int(x * (float(size[1]) / size[0])) - y
    if abs(padx) > 10 or abs(pady) > 10:
        padx = 10000 if padx < 0 else padx
        pady = 10000 if pady < 0 else pady
        new_shape = list(image.shape)
        if padx < pady:
            new_shape[0] += padx
            padded_image = np.zeros(new_shape, dtype=data_type)
            padded_image[int(padx // 2):-int(padx - padx // 2), :] = image
            image = padded_image
        else:
            new_shape[1] += pady
            padded_image = np.zeros(new_shape, dtype=data_type)
            padded_image[:, int(pady // 2):-int(pady - pady // 2)] = image
            image = padded_image
    return cv2.resize(image, (size[1], size[0]))


def padded_resize_label(label, size=[960, 540]):
    """Resize input label to given size, only padding as needed for aspect ratio."""
    x, y = label.shape[:2]
    if label.ndim == 2:
        label = label[:, :, np.newaxis]
    if x < y:
        label = rotate(label, 90).astype(np.bool)
        x, y = label.shape[:2]
    padx = int(y * (float(size[0]) / size[1])) - x
    pady = int(x * (float(size[1]) / size[0])) - y
    if abs(padx) > 10 or abs(pady) > 10:
        padx = 10000 if padx < 0 else padx
        pady = 10000 if pady < 0 else pady
        if padx < pady:
            padded_image = np.zeros([x + padx, y, 1], dtype=np.bool)
            padded_image[int(padx // 2):-int(padx - padx // 2), :, :] = label
            label = padded_image
        else:
            padded_image = np.zeros([x, y + pady, 1], dtype=np.bool)
            padded_image[:, int(pady // 2):-int(pady - pady // 2), :] = label
            label = padded_image
    return cv2.resize(label, (size[1], size[0])).astype(np.bool)


def horizontal_flip(image):
    """This is for convenience, but it is ~2e-7s faster to just do this in-line"""
    return image[:, ::-1, :]


def vertical_flip(image):
    """This is for convenience, but it is ~1e-7s faster to just do this in-line"""
    return image[::-1, ...]


def adjust_gamma(image, gamma=1.0):
    """Adjust the gamma of the image."""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0)**invGamma) * 255
                      for i in np.arange(0, 256)]).astype(np.uint8)
    return cv2.LUT(image, table)
