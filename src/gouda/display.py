"""Convenience methods to display images using pyplot."""
import matplotlib.pyplot as plt
import numpy as np

__author__ = "Sean Mullan"
__copyright__ = "Sean Mullan"
__license__ = "mit"


def print_grid(*images, figsize=(8, 8), toFile=None, show=True, return_grid_shape=False, **kwargs):
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
    defaults = ['hspace', 'wspace', 'left', 'bottom', 'right', 'top']
    for item in defaults:
        if item not in kwargs:
            kwargs[item] = None

    if len(images) == 1:
        images = images[0]
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
        plt.savefig(toFile, dpi=fig.dpi)
    if show:  # pragma: no cover
        # Check manually
        plt.show()
    if return_grid_shape:
        return rows, cols


def print_image(image, figsize=(8, 6.5), toFile=None, show=True, **kwargs):
    """Similar to pyplot.imshow, but with axes and margins for a single image."""
    defaults = {'hspace': 0, 'wspace': 0, 'left': 0, 'bottom': 0, 'right': 1, 'top': 1}
    for item in defaults:
        if item not in kwargs:
            kwargs[item] = defaults[item]
    image = np.squeeze(image)
    fig = plt.figure(figsize=figsize)
    if image.ndim == 2:
        cmap = kwargs['cmap'] if 'cmap' in kwargs else 'bone'
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)
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
        plt.savefig(kwargs['toFile'], dpi=fig.dpi)
    if show:  # pragma: no cover
        # Check manually
        plt.show()
