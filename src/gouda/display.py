"""Convenience methods to display images using pyplot."""
import matplotlib.pyplot as plt
import numpy as np

# from gouda import __version__

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


def print_grid(image_grid,
               show_axis=False,
               figsize=(8, 8),
               norm_type=None,
               toFile=None,
               show=True):
    """Print the input list of images (or list of lists) as a grid with pyplot.

    Parameters
    ----------
        image_grid: np array, list of np arrays, or list of lists of np arrays
            The image, row of images, or rows of images to display
        figsize: (int, int)
            The size of the total figure to display.
        norm_type: int
            The global normalization type for the images (0: none, 1: tanh, 2: sigmoid)
            If no type is given, a type will be interpreted for each image separately.
        toFile: string
            An optional path to save the generated figure to.
    """
    fig = plt.figure(figsize=figsize)
    if type(image_grid) is np.ndarray:
        if image_grid.ndim == 4:
            # Input is a numpy batch of images
            image_grid = [image_grid[i] for i in range(image_grid.shape[0])]
            rows = 1
            cols = len(image_grid)
            image_grid = [image_grid]
        else:
            # Input is a single image
            rows = 1
            cols = 1
            image_grid = [[image_grid]]
    elif type(image_grid[0]) is list:
        # Input is a list of lists of images
        rows = len(image_grid)
        cols = max([len(image_grid[i]) for i in range(rows)])
    elif type(image_grid[0] is np.ndarray):
        # Input is a list of images
        rows = 1
        cols = len(image_grid)
        image_grid = [image_grid]
    else:
        raise ValueError(
            "input must be 2d list of images, 1d list of images, or single image"
        )

    for row in range(rows):
        for col in range(len(image_grid[row])):
            i = row * cols + col
            if image_grid[row][col] is None:
                continue
            fig.add_subplot(rows, cols, i + 1)
            img = np.copy(image_grid[row][col])
            if img.ndim == 2 or img.shape[2] == 1:
                img = np.dstack([img, img, img])
            img = _denorm(img, norm_type=norm_type)
            if not show_axis:
                plt.axis('off')
            plt.imshow(img)
    if toFile:
        plt.savefig(toFile)
    if show:
        plt.show()
    fig.close()


def print_grid_v2(charts, figsize=(8, 8), toFile=None):
    """Similar to print_grid, but each item is a dictionary with an image and arguments rather than just an image.

    Parameters
    ----------
    charts: list of objects
        The list, or list of lists, of dictionaries holding the images.
    figsize: (int, int)
        Size of the output chart.
    toFile: string
        An optional path to save the generated figure to.

    Notes
    -----
    Each dictionary object should have a `image` key with the image to show as the value.
    Possible optional keys are: `title` and `hide_axis`
    """
    fig = plt.figure(figsize=figsize)
    if type(charts[0]) is not list:
        charts = [charts]
    rows = len(charts)
    cols = len(charts[0])
    for row in range(rows):
        for col in range(len(charts[row])):
            i = row * cols + col
            if charts[row][col] is None:
                continue
            item = charts[row][col]
            ax = fig.add_subplot(rows, cols, i + 1)
            if 'title' in item:
                ax.set_title(item['title'])
            if 'hide_axis' in item:
                if item['hide_axis']:
                    ax.axis('off')
            plt.imshow(item['image'])
    if toFile:
        plt.savefig(toFile)
    plt.show()
