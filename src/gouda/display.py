"""Convenience methods to display images using matplotlib.pyplot."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

from gouda.general import extract_method_kwargs
from gouda.typing import ImageLikeType


def print_grid(
    *input_images: Any,  # noqa: ANN401 # Using Any for input_images to avoid complex nested types
    figsize: tuple[int, int] = (8, 8),
    file_name: Union[str, os.PathLike, None] = None,
    do_squarify: bool = False,
    show: bool = True,
    return_grid_shape: bool = False,
    return_fig: bool = False,
    cmap: str = "gray",
    **kwargs: Any,  # noqa: ANN401
) -> Union[Figure, tuple[int, int], tuple[Figure, tuple[int, int]], None]:
    """Print out images as a grid.

    Parameters
    ----------
    *input_images : ImageLikeType | Sequence[ImageLikeType] | Sequence[Sequence[ImageLikeType]]
        Image(s) to print as a grid
    figsize : (int, int)
        Figure size to pass to pyplot
    file_name : str
        File to save image to
    do_squarify : bool
        If True, runs :py:meth:`gouda.display.squarify` on the images before displaying
    show : bool
        Whether to show the grid or not (the default is True)
    return_grid_shape : bool
        Whether to return the (height, width) of the grid or not
    return_fig : bool
        Whether to return the figure object or not
    cmap : str
        The color map to use for the images (the default is 'gray')
    image_kwargs : dict
        Keyword arguments to be used for each matplotlib.pyplot.imshow call.
    **kwargs
        Any parameters for :meth:`matplotlib.pyplot.subplots_adjust`, :meth:`matplotlib.pyplot.figure`, or :meth:`gouda.display.squarify` can be passed for use in the grid. Parameters for :meth:`matpl[...]

    Note
    ----
    * Images can be in shape [x, y] or [x, y, c], but only 1 or 3 channels will work (assumes row/col structure otherwise)
    * If no image kwargs are passed (ex: top, right, etc.), fig.tight_layout is applied
    * A "cmap" argument passed with an item dictionary will override the function-level "cmap" argument

    General accepted formats
    ------------------------
    * List of lists
    * List of dicts with key 'image' with image value
    * List of numpy arrays
    * 2, 3, 4, or 5 dimensional numpy arrays (leading rows will be used as row/column)

    #TODO - allow passing ncols or nrows as arguments... move this to squarify?
    #TODO - allow passing row/col height/width instead of figsize - https://stackoverflow.com/a/4306340/2348288
    #TODO - allow passing row/col labels
    """
    defaults = ["hspace", "wspace", "left", "bottom", "right", "top"]
    for item in defaults:
        if item not in kwargs:
            kwargs[item] = None
    kwargs["cmap"] = cmap
    image_kwargs = extract_method_kwargs(kwargs, plt.imshow)
    fig_kwargs = extract_method_kwargs(kwargs, plt.figure)
    squarify_kwargs = extract_method_kwargs(kwargs, squarify)

    # Use type: ignore to tell mypy to skip checking this line
    images: Any = input_images[0] if len(input_images) == 1 else input_images

    if do_squarify:
        # Use type: ignore to bypass the type check for the squarify function call
        images = squarify(images, **squarify_kwargs)  # type: ignore
    if not isinstance(images, (list, tuple, np.ndarray, str, dict)):
        try:
            images = np.asarray(images)
        except ValueError:
            pass
    if hasattr(images, "__array__"):
        images = np.asarray(images)
    if isinstance(images, (list, tuple)):
        if len(images) == 0:
            raise ValueError("No images to display")
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
            raise ValueError(f"Invalid array shape: {images.shape}")
    elif isinstance(images, dict):
        rows = 1
        cols = 1
        to_show = [[images]]
    else:
        raise ValueError(f"Invalid input type: {type(images)}")
    if rows == 0 or cols == 0:
        raise ValueError("No images to display")

    fig = plt.figure(figsize=figsize, **fig_kwargs)
    gs = fig.add_gridspec(int(rows), int(cols), hspace=kwargs["hspace"], wspace=kwargs["wspace"])

    for row in range(rows):
        image_row = to_show[row]
        for col in range(len(image_row)):
            image = image_row[col]
            if image is None or (isinstance(image, dict) and image["image"] is None):
                continue
            ax = fig.add_subplot(gs[row, col])

            if isinstance(image, dict):
                image_dict = image
                image = np.squeeze(image_dict["image"])
                for key in image_kwargs:
                    if key not in image_dict:
                        image_dict[key] = image_kwargs[key]
                imshow_kwargs = extract_method_kwargs(image_dict, plt.imshow)
                if image.dtype == bool:
                    image = image.astype(np.uint8)
                plt.imshow(image, **imshow_kwargs)
                if "title" in image_dict:
                    ax.set_title(image_dict["title"])
                if "xlabel" in image_dict:
                    ax.set_xlabel(image_dict["xlabel"])
                if "ylabel" in image_dict:
                    ax.set_ylabel(image_dict["ylabel"])
            else:
                image = np.squeeze(image)
                plt.imshow(image, **image_kwargs)
            # ax.set_axis_off()
            plt.setp(ax.spines.values(), visible=False)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(left=kwargs["left"], bottom=kwargs["bottom"], right=kwargs["right"], top=kwargs["top"])
    if "suptitle" in kwargs:
        plt.suptitle(kwargs["suptitle"])
    if all(kwargs[key] is None for key in defaults):
        fig.tight_layout()
    if file_name is not None:
        plt.savefig(file_name, dpi=fig.dpi)
    if show:  # pragma: no cover
        # Check manually
        plt.show()
    else:
        plt.close(fig)
    if return_fig and return_grid_shape:
        return (fig, (rows, cols))
    elif return_fig:
        return fig
    elif return_grid_shape:
        return (rows, cols)
    else:
        return None


def print_image(
    image: npt.NDArray,
    figsize: tuple[float, float] = (8.0, 6.5),
    file_name: Union[str, os.PathLike, None] = None,
    show: bool = True,
    allow_interpolation: bool = False,
    imshow_args: Union[dict[str, Any], None] = None,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """Similar to pyplot.imshow, but with axes and margins for a single image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image to display
    figsize : (float, float)
        The size in inches of the image to display (the default is (8, 6.5))
    file_name : str | None
        The path to save the displayed figure to, or None to not save (the default is None)
    show : bool
        Whether to call pyplot.show to display the image (the default is True)
    allow_interpolation : bool
        Whether to allow automatic interpolation (nearest neighbor and automatic aspect ratio) when either height or width is 10x larger than the other (the default is False)
    imshow_args : dict
        Extra args to pass directly to the pyplot.imshow call (the default is {})
    """
    defaults = {"hspace": 0, "wspace": 0, "left": 0, "bottom": 0, "right": 1, "top": 1, "cmap": "bone"}
    imshow_args = imshow_args if imshow_args is not None else {}
    for item, val in defaults.items():
        if item not in kwargs:
            kwargs[item] = val
    image = np.squeeze(image)
    fig = plt.figure(figsize=figsize)
    if max(image.shape[:2]) / min(image.shape[:2]) > 10 and allow_interpolation:
        imshow_args["interpolation"] = "nearest"
        imshow_args["aspect"] = "auto"
    if image.ndim == 2:
        imshow_args["cmap"] = kwargs["cmap"]

    plt.imshow(image, **imshow_args)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.subplots_adjust(
        top=kwargs["top"],
        bottom=kwargs["bottom"],
        right=kwargs["right"],
        left=kwargs["left"],
        hspace=kwargs["hspace"],
        wspace=kwargs["wspace"],
    )
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if file_name is not None:
        plt.savefig(file_name, dpi=fig.dpi)
    if show:  # pragma: no cover
        # Check manually
        plt.show()


def squarify(
    image: Union[npt.NDArray, Sequence[npt.NDArray]],
    primary_axis: int = 0,
    num_cols: Optional[int] = None,
    as_array: bool = False,
) -> Union[npt.NDArray, list[list[ImageLikeType]]]:
    """Reshape a list/array of images into nested elements with the same numbers of rows and columns.

    Parameters
    ----------
    image: list | numpy.ndarray
        The list/array of images to reshape
    axis: int
        If the image is an array, the axis to split it along (the default is 0)
    num_cols: int | None, optional
        If provided, the number of columns to use in the reshaped array (the default is None)
    as_array: bool
        Whether to convert the result into an array with rows and columns as the first two axes (the default is False)

    NOTE
    ----
    If there are not a square number of images, then the last row will have None values as placeholders. If as_array is True, these will be zeros instead.
    If as_array is True, this assumes that all images have the same shape.
    """
    if isinstance(image, (tuple, list)):
        num_images = len(image)
        images = list(image)
    elif isinstance(image, np.ndarray):
        # images = np.split(image, image.shape[axis], axis=axis)
        # images = [item for item in images]
        images = []
        axis_slice: list[Union[slice, int]] = [slice(None) for _ in range(image.ndim)]
        num_images = image.shape[primary_axis]
        for idx in range(num_images):
            axis_slice[primary_axis] = idx
            images.append(image[tuple(axis_slice)])
    else:
        raise ValueError("Unknown image type: {type(image)}")
    if num_cols is None:
        num_cols = int(np.ceil(np.sqrt(num_images)))
    outer_list: list[list[ImageLikeType]] = []
    for i in range(0, num_images, num_cols):
        inner_list = []
        for j in range(num_cols):
            if i + j >= num_images:
                if as_array:
                    inner_list.append(np.zeros_like(images[0]))
                else:
                    inner_list.append(None)
            else:
                inner_list.append(images[i + j])
        outer_list.append(inner_list)
    if as_array and isinstance(outer_list[0][0], npt.ArrayLike):
        rows: list[npt.ArrayLike] = [
            np.stack([np.asarray(item) for item in inner_list], axis=0) for inner_list in outer_list
        ]
        result_array: npt.NDArray = np.stack(rows, axis=0)
        return result_array
    return outer_list
