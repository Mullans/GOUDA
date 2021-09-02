"""Convenience methods to display images using matplotlib.pyplot."""
import inspect
import matplotlib.pyplot as plt
import numpy as np

__author__ = "Sean Mullan"
__copyright__ = "Sean Mullan"
__license__ = "mit"


def _extract_method_kwargs(kwargs, method, remove=True):
    """Internal method to extract keyword arguments related to a given method.

    Parameters
    ----------
    kwargs : dict
        A dictionary of keyword arguments
    method : func
        A function with keyword arguments
    remove : bool
        Whether to remove the extracted keyword arguments from the original dict (the default is True)
    """
    method_params = inspect.signature(method).parameters
    method_kwargs = {}
    to_remove = []
    for key in kwargs:
        if key in method_params:
            method_kwargs[key] = kwargs[key]
            to_remove.append(key)
    if remove:
        for key in to_remove:
            del kwargs[key]
    return method_kwargs


def print_grid(*images, figsize=(8, 8), toFile=None, show=True, return_grid_shape=False, **kwargs):
    """Print out images as a grid.

    Parameters
    ----------
    *images : list or numpy.ndarray
        Image(s) to print as a grid
    figsize : (int, int)
        Figure size to pass to pyplot
    toFile : str
        File to save image to
    show : bool
        Whether to show the grid or not (the default is True)
    return_grid_shape : bool
        Whether to return the (height, width) of the grid or not
    image_kwargs : dict
        Keyword arguments to be used for each matplotlib.pyplot.imshow call.
    **kwargs : dict
        Any parameters for :meth:`matplotlib.pyplot.subplots_adjust` can be passed for use in the grid. Parameters for :meth:`matplotlib.pyplot.imshow` will be used as defaults for all images in the grid, but will be replaced by any image-specific arguments (pass image as dict).

    Note
    ----
    Images can be in shape [x, y] or [x, y, c], but only 1 or 3 channels will work (assumes row/col structure otherwise)

    General accepted formats
    ------------------------
    * List of lists
    * List of dicts with key 'image' with image value
    * List of numpy arrays
    * 2, 3, 4, or 5 dimensional numpy arrays (leading rows will be used as row/column)
    """
    defaults = ['hspace', 'wspace', 'left', 'bottom', 'right', 'top']
    for item in defaults:
        if item not in kwargs:
            kwargs[item] = None
    image_kwargs = _extract_method_kwargs(kwargs, plt.imshow)

    if len(images) == 1:
        images = images[0]
    if hasattr(images, '__array__'):
        images = np.array(images)
    if isinstance(images, (list, tuple)):
        if isinstance(images[0], (list, tuple)):
            # list of lists of images
            rows = len(images)
            cols = max([len(item) for item in images])
            to_show = images
        else:
            # list of images
            rows = 1
            cols = len(images)
            to_show = [images]
    elif isinstance(images, np.ndarray):
        # input as array
        while images.shape[0] == 1:
            images = np.squeeze(images, axis=0)
        if images.shape[-1] == 1:
            images = np.squeeze(images, axis=-1)
        ndim = images.ndim
        if images.shape[-1] == 3:
            ndim -= 1
        if ndim == 2:
            # single image
            rows = 1
            cols = 1
            to_show = [[images]]
        elif ndim == 3:
            rows = 1
            cols = images.shape[0]
            to_show = [[images[i] for i in range(images.shape[0])]]
        elif ndim == 4:
            rows = images.shape[0]
            cols = images.shape[1]
            to_show = [[images[i, j] for j in range(images.shape[1])] for i in range(images.shape[0])]
        else:
            raise ValueError('Invalid array shape: {}'.format(images.shape))
    elif isinstance(images, dict):
        rows = 1
        cols = 1
        to_show = [[images]]
    else:
        raise ValueError("Invalid input type: {}".format(type(images)))

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(int(rows), int(cols), hspace=kwargs['hspace'], wspace=kwargs['wspace'])

    for row in range(rows):
        image_row = to_show[row]
        for col in range(len(image_row)):
            image = image_row[col]
            if image is None:
                continue
            ax = fig.add_subplot(gs[row, col])

            if isinstance(image, dict):
                image_dict = image
                image = np.squeeze(image_dict['image'])
                for key in image_kwargs:
                    if key not in image_dict:
                        image_dict[key] = image_kwargs[key]
                imshow_kwargs = _extract_method_kwargs(image_dict, plt.imshow)
                plt.imshow(image, **imshow_kwargs)
                if 'title' in image_dict:
                    ax.set_title(image_dict['title'])
                if 'xlabel' in image_dict:
                    ax.set_xlabel(image_dict['xlabel'])
                if 'ylabel' in image_dict:
                    ax.set_ylabel(image_dict['ylabel'])
            else:
                image = np.squeeze(image)
                plt.imshow(image, **image_kwargs)
            ax.set_axis_off()
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(left=kwargs['left'], bottom=kwargs['bottom'], right=kwargs['right'], top=kwargs['top'])
    if all([kwargs[key] is None for key in defaults]):
        fig.tight_layout()
    if 'suptitle' in kwargs:
        plt.suptitle(kwargs['suptitle'])
    if toFile is not None:
        plt.savefig(toFile, dpi=fig.dpi)
    if show:  # pragma: no cover
        # Check manually
        plt.show()
    else:
        plt.close(fig)
    if return_grid_shape:
        return rows, cols


def print_image(image, figsize=(8, 6.5), toFile=None, show=True, allow_interpolation=False, imshow_args={}, **kwargs):
    """Similar to pyplot.imshow, but with axes and margins for a single image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image to display
    figsize : (float, float)
        The size in inches of the image to display (the default is (8, 6.5))
    toFile : str | None
        The path to save the displayed figure to, or None to not save (the default is None)
    show : bool
        Whether to call pyplot.show to display the image (the default is True)
    allow_interpolation : bool
        Whether to allow automatic interpolation (nearest neighbor and automatic aspect ratio) when either height or width is 10x larger than the other (the default is False)
    imshow_args : dict
        Extra args to pass directly to the pyplot.imshow call (the default is {})
    """
    defaults = {
        'hspace': 0,
        'wspace': 0,
        'left': 0,
        'bottom': 0,
        'right': 1,
        'top': 1,
        'cmap': 'bone'
    }

    for item in defaults:
        if item not in kwargs:
            kwargs[item] = defaults[item]
    image = np.squeeze(image)
    fig = plt.figure(figsize=figsize)
    if max(image.shape[:2]) / min(image.shape[:2]) > 10 and allow_interpolation:
        imshow_args['interpolation'] = 'nearest'
        imshow_args['aspect'] = 'auto'
    if image.ndim == 2:
        imshow_args['cmap'] = kwargs['cmap']

    plt.imshow(image, **imshow_args)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.subplots_adjust(top=kwargs['top'],
                        bottom=kwargs['bottom'],
                        right=kwargs['right'],
                        left=kwargs['left'],
                        hspace=kwargs['hspace'],
                        wspace=kwargs['wspace'])
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if toFile is not None:
        plt.savefig(toFile, dpi=fig.dpi)
    if show:  # pragma: no cover
        # Check manually
        plt.show()


def squarify(image, axis=0, as_array=False):
    """Reshape a list/array of images into nested elements with the same numbers of rows and columns.

    Parameters
    ----------
    image: list | numpy.ndarray
        The list/array of images to reshape
    axis: int
        If the image is an array, the axis to split it along (the default is 0)
    as_array: bool
        Whether to convert the result into an array with rows and columns as the first two axes (the default is False)

    NOTE
    ----
    If there are not a square number of images, then the last row will have None values as placeholders. If as_array is True, these will be zeros instead.
    If as_array is True, this assumes that all images have the same shape.
    """
    if isinstance(image, list):
        num_images = len(image)
        images = [item for item in image]
    else:
        # images = np.split(image, image.shape[axis], axis=axis)
        # images = [item for item in images]
        images = []
        axis_slice = [slice(None) for _ in range(image.ndim)]
        num_images = image.shape[axis]
        for idx in range(num_images):
            axis_slice[axis] = idx
            images.append(image[tuple(axis_slice)])
    num_rows = int(np.ceil(np.sqrt(num_images)))
    outer_list = []
    for i in range(0, num_images, num_rows):
        inner_list = []
        for j in range(0, num_rows):
            if i + j >= num_images:
                if as_array:
                    inner_list.append(np.zeros_like(images[0]))
                else:
                    inner_list.append(None)
            else:
                inner_list.append(images[i + j])
        outer_list.append(inner_list)
    if as_array:
        rows = [np.stack(inner_list, axis=0) for inner_list in outer_list]
        outer_list = np.stack(rows, axis=0)
    return outer_list
