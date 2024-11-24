"""Convenience methods to display images using matplotlib.pyplot."""
import inspect
import matplotlib.pyplot as plt
import numpy as np

from gouda import data_methods

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


def print_grid(*images, figsize=(8, 8), toFile=None, do_squarify=False, show=True, return_grid_shape=False, return_fig=False, cmap='gray', **kwargs):
    """Print out images as a grid.

    Parameters
    ----------
    *images : list or numpy.ndarray
        Image(s) to print as a grid
    figsize : (int, int)
        Figure size to pass to pyplot
    toFile : str
        File to save image to
    do_squarify : bool
        If True, runs :py:meth:`gouda.display.squarify` on the images before displaying
    show : bool
        Whether to show the grid or not (the default is True)
    return_grid_shape : bool
        Whether to return the (height, width) of the grid or not
    image_kwargs : dict
        Keyword arguments to be used for each matplotlib.pyplot.imshow call.
    **kwargs : dict
        Any parameters for :meth:`matplotlib.pyplot.subplots_adjust`, :meth:`matplotlib.pyplot.figure`, or :meth:`gouda.display.squarify` can be passed for use in the grid. Parameters for :meth:`matplotlib.pyplot.imshow` will be used as defaults for all images in the grid, but will be replaced by any image-specific arguments (pass image as dict).

    Note
    ----
    * Images can be in shape [x, y] or [x, y, c], but only 1 or 3 channels will work (assumes row/col structure otherwise)
    * If no image kwargs are passed (ex: top, right, etc.), fig.tight_layout is applied

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
    defaults = ['hspace', 'wspace', 'left', 'bottom', 'right', 'top']
    for item in defaults:
        if item not in kwargs:
            kwargs[item] = None
    kwargs['cmap'] = cmap
    image_kwargs = _extract_method_kwargs(kwargs, plt.imshow)
    fig_kwargs = _extract_method_kwargs(kwargs, plt.figure)
    squarify_kwargs = _extract_method_kwargs(kwargs, squarify)

    if do_squarify:
        images = squarify(images, **squarify_kwargs)

    if len(images) == 1:
        images = images[0]
    if hasattr(images, '__array__'):
        images = np.array(images)
    if isinstance(images, (list, tuple)):
        if len(images) == 0:
            raise ValueError('No images to display')
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
    if rows == 0 or cols == 0:
        raise ValueError('No images to display')

    fig = plt.figure(figsize=figsize, **fig_kwargs)
    gs = fig.add_gridspec(int(rows), int(cols), hspace=kwargs['hspace'], wspace=kwargs['wspace'])

    for row in range(rows):
        image_row = to_show[row]
        for col in range(len(image_row)):
            image = image_row[col]
            if image is None or (isinstance(image, dict) and image['image'] is None):
                continue
            ax = fig.add_subplot(gs[row, col])

            if isinstance(image, dict):
                image_dict = image
                image = np.squeeze(image_dict['image'])
                for key in image_kwargs:
                    if key not in image_dict:
                        image_dict[key] = image_kwargs[key]
                imshow_kwargs = _extract_method_kwargs(image_dict, plt.imshow)
                if image.dtype == bool:
                    image = image.astype(np.uint8)
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
            # ax.set_axis_off()
            plt.setp(ax.spines.values(), visible=False)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(left=kwargs['left'], bottom=kwargs['bottom'], right=kwargs['right'], top=kwargs['top'])
    if 'suptitle' in kwargs:
        plt.suptitle(kwargs['suptitle'])
    if all([kwargs[key] is None for key in defaults]):
        fig.tight_layout()
    if toFile is not None:
        plt.savefig(toFile, dpi=fig.dpi)
    if show:  # pragma: no cover
        # Check manually
        plt.show()
    else:
        plt.close(fig)
    to_return = []
    if return_fig:
        to_return.append(fig)
    if return_grid_shape:
        to_return.append((rows, cols))
    if len(to_return) == 1:
        return to_return[0]
    elif len(to_return) > 1:
        return tuple(to_return)


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


def squarify(image, axis=0, num_cols=None, as_array=False):
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
    if num_cols is None:
        num_cols = int(np.ceil(np.sqrt(num_images)))
    outer_list = []
    for i in range(0, num_images, num_cols):
        inner_list = []
        for j in range(0, num_cols):
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


class VideoWriter:
    import cv2

    def __init__(self, out_path, fps=10, codec='MJPG', output_shape=None, interpolator=cv2.INTER_LINEAR):
        """A convenience wrapper for OpenCV video writing"""
        self.out_path = out_path
        self.output_shape = output_shape  # assumes (height, width)
        self.writer = None
        self.fps = fps
        self.codec = codec
        self.interpolator = interpolator

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer is not None:
            self.writer.release()

    def start_writer(self):
        codec = self.cv2.VideoWriter_fourcc(*self.codec)
        self.writer = self.cv2.VideoWriter(self.out_path, codec, self.fps, (self.output_shape[1], self.output_shape[0]))

    def write(self, data):
        if self.writer is None:
            if self.output_shape is None:
                self.output_shape = np.squeeze(data).shape[:2]
            self.start_writer()
        data = data_methods.to_uint8(data)
        if data.ndim == 2:
            # TODO - allow for color maps
            data = np.dstack([data] * 3)
        elif data.ndim == 3 and data.shape[-1] == 4:
            data = self.cv2.cvtColor(data, self.cv2.RGBA2RGB)
        if data.shape[:2] != self.output_shape:
            data = self.cv2.resize(data, (self.output_shape[1], self.output_shape[0]), self.interpolator)
        self.writer.write(data)

    def __call__(self, data):
        self.write(data)


def show_video(data, player_width=500, player_height=300, frame_height=None, frame_width=None, toFile='temp.mp4', show='ipython', **kwargs):
    """Convert a series of frames to a video and display it.

    Parameters
    ----------
    data : list or numpy.ndarray
        The frames to join into a video
    player_width : int
        The width in pixels of the video player
    player_height : int
        The height in pixels of the video player
    frame_height : int
        The height in pixels of the result video (if None, it will be determined based on the first frame)
    frame_width : int
        The width in pixels of the result video (if None, it will be determined based on the first frame)
    toFile : str or os.PathLike
        The path to save the output video to
    show : str or None
        The method to show the video or None to not display the result (options are 'ipython', 'opencv', None)
    **kwargs : dict
        Other parameters for VideoWriter such as fps, codec, interpolator


    Note
    ----
    Data can be in shape [frames, x, y], [frames, x, y, c], but only 1, 3, or 4 channels will work
    """
    defaults = {
        'fps': 10,
        'codec': 'H264',
        'frame_height': frame_height,
        'frame_width': frame_width,
        'interpolator': 1  # 1 = cv2.InterLinear
    }

    for item in defaults:
        if item not in kwargs:
            kwargs[item] = defaults[item]
    if isinstance(data, (list, tuple)):
        # A list/tuple of arrays
        if hasattr(data[0], '__array__'):
            nframes = len(data)
            temp = np.array(data[0])
            data_shape = temp.shape
            ndim = temp.ndim + 1
        else:
            raise ValueError('Frames must be array-like, not {}'.format(type(data[0])))
    elif hasattr(data, '__array__'):
        # Array
        data = np.array(data)
        nframes = data.shape[0]
        data_shape = data.shape[1:]
        ndim = data.ndim
    else:
        raise ValueError('Unknown data type: {}'.format(type(data)))

    if ndim < 2:
        raise ValueError('Video data must have at frames, height, and width')
    if not (ndim == 3 or (ndim == 4 and data_shape[-1] in [1, 3, 4])):
        raise ValueError('Unknown video shape: {}'.format(data_shape))

    if kwargs['frame_height'] is not None and kwargs['frame_width'] is not None:
        kwargs['output_shape'] = (int(kwargs['frame_height']), int(kwargs['frame_width']))
    elif kwargs['frame_height'] is not None:
        width = data_shape[1] * (kwargs['frame_height'] / data_shape[0])
        kwargs['output_shape'] = (int(kwargs['frame_height']), int(width))
    elif kwargs['frame_width'] is not None:
        height = data_shape[0] * (kwargs['frame_width'] / data_shape[1])
        kwargs['output_shape'] = (int(height), int(kwargs['frame_width']))
    else:
        kwargs['output_shape'] = (int(data_shape[0]), int(data_shape[1]))

    kwargs['output_shape'] = (int(kwargs['output_shape'][1]), int(kwargs['output_shape'][0]))
    writer_kwargs = _extract_method_kwargs(kwargs, VideoWriter.__init__)
    print(writer_kwargs)
    with VideoWriter(str(toFile), **writer_kwargs) as writer:
        for i in range(nframes):
            writer.write(data[i])

    if show == 'ipython':
        from IPython.display import Video
        return Video(str(toFile), height=player_height, width=player_width)
    elif show == 'opencv':
        raise NotImplementedError('Still working on this - use ipython for now')
