"""Convenience methods to display images using pyplot."""
import matplotlib.pyplot as plt
import numpy as np

__author__ = "Sean Mullan"
__copyright__ = "Sean Mullan"
__license__ = "mit"


FULL_RANGE = 0
TANH = 1
SIGMOID = 2


def _denorm(x, norm_type=None):
    """Denormalize an image to [0, 255] based on specified or inferred normalization type."""
    if norm_type is None:
        if x.max() > 1 and x.max() <= 255 and x.min() >= 0:
            norm_type = 0  # Values are in range [0, 255], no norm
        elif x.min() < 0 and x.min() >= -1 and x.max() <= 1:
            norm_type = 1  # Values are in range [-1, 1], tanh norm
        elif x.min() >= 0 and x.max() <= 1:
            norm_type = 2  # Values are in range [0, 1], sigmoid norm
        else:
            raise ValueError(
                "Given values are not in one of the valid image normalization ranges."
            )
    if norm_type == TANH:
        x = ((x * 127.5) + 127.5)
    elif norm_type == SIGMOID:
        x = (x * 255.0)
    return x.astype(np.uint8)


def print_grid(image, *images, figsize=(8, 8), toFile=None, show=True, **kwargs):
    """Print out images as a grid.

    Parameters
    ----------
    image : list | np.ndarray
        Image(s) to print as a grid
    *images : list | np.ndarray
        More images to print
    figsize : (int, int)
        Figure size to pass to pyplot
    toFile : str
        File to save image to
    show : bool
        Whether to show the grid or not (the default is True)

    Optional Parameters
    -------------------
    Any parameters for matplotlib.pyplot.subplots_adjust can be passed for use in the grid

    Note
    ----
    General accepted formats:
        * List of lists
        * List of dicts with key 'image' with image value
        * List of numpy arrays
        * 2, 3, 4, or 5 dimensional numpy arrays (leading rows will be used as row/column)
    Images can be in shape [x, y] or [x, y, c], but only 1 or 3 channels will work (assumes row/col structure otherwise)
    """
    if len(images) > 0:
        images = [image, *images]
    else:
        images = image
    defaults = ['hspace', 'wspace', 'left', 'bottom', 'right', 'top']
    for item in defaults:
        if item not in kwargs:
            kwargs[item] = None
    unwrapped = []
    if isinstance(images, list):
        print('list of stuff')
        if isinstance(images[0], list):
            # List of lists of images
            rows = len(images)
            cols = max([len(item) for item in images if item is not None])
            unwrapped = images
        elif isinstance(images[0], dict):
            rows = 1
            cols = len(images)
            unwrapped = [images]
        elif isinstance(images[0], np.ndarray) and images[0].ndim in [3, 4] and images[0].shape[-1] not in [1, 3]:
            # Batches of images - assumes that if image.shape[-1] > 3 that it isn't color channels
            rows = len(images)
            cols = max([item.shape[0] for item in images if item is not None])
            for item in images:
                unwrapped.append([item[i] if item is not None else None for i in range(item.shape[0])])
        elif isinstance(images[0], np.ndarray):
            # Single row of images
            rows = 1
            cols = len(images)
            unwrapped = [images]
        else:
            raise ValueError("Error printing image grid")
    elif isinstance(images, np.ndarray):
        if (images.ndim == 3 or images.ndim == 4) and (images.shape[-1] != 3 and images.shape[-1] != 1):
            if images.ndim == 3:
                # [batch, x,  y]
                rows = 1
                cols = images.shape[0]
                unwrapped = [images]
            else:
                # [row, col, x, y]
                rows = images.shape[0]
                cols = images.shape[1]
                unwrapped = images
        elif (images.ndim == 3 and images.shape[-1] not in [1, 3]) or images.ndim == 2:
            # [x, y, c] or [x, y]
            rows = 1
            cols = 1
            unwrapped = [[images]]
        elif images.ndim == 4 and (images.shape[-1] == 3 or images.shape[-1] == 1):
            # [b, x, y, c]
            rows = 1
            cols = images.shape[0]
            unwrapped = [images]
        elif images.ndim == 5:
            # [row, col, x, y, c]
            rows = images.shape[0]
            cols = images.shape[1]
            unwrapped = images
        else:
            raise ValueError("Invalid array shape {}".format(images.shape))
    elif isinstance(images, dict):
        unwrapped = [[images]]
        rows = 1
        cols = 1
    else:
        raise TypeError("Print grid requires list/np.ndarray/dict type inputs, not {}".format(type(images)))

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(int(rows), int(cols), hspace=kwargs['hspace'], wspace=kwargs['wspace'])

    for row in range(rows):
        image_row = unwrapped[row]
        for col in range(len(image_row)):
            image = image_row[col]
            if image is None:
                continue
            ax = fig.add_subplot(gs[row, col])

            if isinstance(image, dict):
                image_dict = image
                image = np.squeeze(image_dict['image'])
                if image.ndim == 2:
                    if 'cmap' in image_dict:
                        cmap = image_dict['cmap']
                    else:
                        cmap = 'bone'
                    plt.imshow(image, cmap=cmap)
                else:
                    plt.imshow(image)
                if 'title' in image_dict:
                    ax.set_title(image_dict['title'])
                if 'xlabel' in image_dict:
                    ax.set_xlabel(image_dict['xlabel'])
                if 'ylabel' in image_dict:
                    ax.set_ylabel(image_dict['ylabel'])
            else:
                image = np.squeeze(image)
                plt.imshow(image)
            ax.set_axis_off()
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(left=kwargs['left'], bottom=kwargs['bottom'], right=kwargs['right'], top=kwargs['top'])
    if all([kwargs[key] is None for key in defaults]):
        fig.tight_layout()
    if 'suptitle' in kwargs:
        plt.suptitle(kwargs['suptitle'])
    if toFile is not None:
        plt.toFile(toFile, dpi=fig.dpi)
    if show:
        plt.show()


def print_image(image, figsize=(8, 6.5), **kwargs):
    """Similar to pyplot.imshow, but with axes and margins for a single image."""
    image = image.flatten()
    if 'cmap' not in kwargs and image.ndim == 2:
        kwargs['cmap'] = 'bone'
    fig = plt.figure(figsize=figsize)
    plt.imshow(image.flatten(), cmap=kwargs['cmap'])
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if 'toFile' in kwargs:
        plt.savefig(kwargs['toFile'], dpi=fig.dpi)
    if 'hide' not in kwargs:
        plt.show()
