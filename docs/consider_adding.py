# isort: skip_file
# flake8: noqa
# pylint: skip-file

# These two are useful, but don't fit any current file name. Something new?
import numpy as np
import scipy.ndimage
from collections.abc import Iterable
from typing import Any, Optional, Union


# Maybe? Has a dependence on scipy.ndimage, but is very useful
# Can we replace opencv with scipy.ndimage stuff? Are both needed?
def largest_connected_component(image, as_bool=True):
    input_shape = image.shape
    image = np.squeeze(image)
    if as_bool:
        image = image != 0
    labels, num_features = scipy.ndimage.label(image)
    if num_features == 1:
        biggest_cc = labels
    else:
        idx, counts = np.unique(labels, return_counts=True)
        biggest = idx[np.argmax(counts[1:]) + 1]
        biggest_cc = np.where(labels == biggest, 1, 0).reshape(input_shape)
    return biggest_cc


import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path


def dice_curve(label, pred, num_decimals=2, return_thresh=False):
    label = np.ravel(label)
    label = gouda.rescale(label, 0, 1)
    pred = np.ravel(pred)
    if num_decimals is not None:
        pred = np.round(pred, decimals=num_decimals)
    desc_score_indices = np.argsort(pred, kind='mergesort')[::-1]
    y_score = pred[desc_score_indices]
    y_true = label[desc_score_indices]
    distinct_idx = np.where(np.diff(y_score))[0]
    thresh_idx = np.concatenate([distinct_idx, [y_true.size - 1]])

    tps = np.cumsum(y_true)
    tps = tps[thresh_idx]
    fps = 1 + thresh_idx - tps
    thresh = y_score[thresh_idx]

    tps = np.concatenate(([0], tps))
    fps = np.concatenate(([0], fps))
    thresh = np.concatenate(([1], thresh))
    fns = tps[-1] - tps

    dice = (2 * tps) / (2 * tps + fps + fns)
    dice = dice[::-1]
    if return_thresh:
        thresh = thresh[::-1]
        return dice, thresh
    return dice

def correlation_matrix(data,
                       label=None,
                       columns=None,
                       ignore_columns=[],
                       matrix_kwargs={'cmap': 'viridis'},
                       rescale_features=True,
                       text_kwargs={}):
    # TODO: update this for generic arrays
    from sklearn.linear_model import LinearRegression
    available_columns = []
    for column in data.columns:
        dtype = str(data[column].dtype)
        if not ('int' in dtype or 'float' in dtype):
            continue
        if len(np.unique(data[column])) == 1:
            continue
        if column in ignore_columns:
            continue
        available_columns.append(column)
    if columns is None:
        columns = available_columns#[:-1]
    if label is None:
        label = data[available_columns[-1]]
    elif isinstance(label, str):
        label = data[label]
    results = np.zeros([len(columns), len(columns)])
    results[:] = np.nan
    for i in range(len(columns)):
        for j in range(i + 1):
            if i == j:
                features = data[columns[i]]
            else:
                features = data[[columns[i], columns[j]]]
            features = features.values.astype(np.float)
            if rescale_features:
                features = gouda.rescale(features, axis=0)
            if features.ndim == 1:
                features = features[:, None]
            model = LinearRegression()
            model.fit(features, label)
            pred = model.predict(features)
            score = r2_score(label, pred)
            results[i, j] = score
    im, cbar = heatmap(results, columns, columns, **matrix_kwargs)
    texts = annotate_heatmap(im, **text_kwargs)


from mpl_toolkits.axes_grid1 import make_axes_locatable


def heatmap(data, row_labels, col_labels, ax=None, show_cbar=True,
            cbar_kw={}, cbarlabel="", tick_rot=-30, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    show_cbar : bool
        Whether to show the colorbar, by default True
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    if show_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = ax.figure.colorbar(im, cax=cax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    else:
        cbar = None

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=tick_rot, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("white", "black"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
        mask = data.mask
    else:
        mask = np.isnan(data)

    has_nans = isinstance(mask, np.ndarray)

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(np.nanmax(data)) / 2.
    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if has_nans and mask[i, j]:
                kw.update(color=textcolors[1])
                text = im.axes.text(j, i, 'nan', **kw)
                texts.append(text)
            else:
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)
    return texts





def overlay_cmap(image, mask, opacity=0.5, rescale_mask=True, cmap=cv2.COLORMAP_VIRIDIS):
    import cv2

    from gouda.plotting import parse_color
    image = gimage.to_uint8(np.squeeze(image))
    image = np.dstack([image] * 3)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    mask = gouda.rescale(np.squeeze(mask), 0, 255).astype(np.uint8)
    print(mask.shape)
    if isinstance(cmap, int):
        mapped = cv2.applyColorMap(mask, cmap)[:, :, ::-1]
    else:
        color = parse_color(cmap)
        mapped = np.dstack([mask * color[i] for i in range(3)]).astype(np.uint8)
    overlay = cv2.addWeighted(image, 1 - opacity, mapped, opacity, 0)
    # overlay = np.where((mask == 0)[:, :, None], image, overlay)
    return overlay


def overlay_cmaps(image, masks, opacity=0.5, rescale_mask=True, cmaps=[]):
    """Somehow merge this with the above"""
    image = gimage.to_uint8(np.squeeze(image))
    image = np.dstack([image] * 3)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    mapped = np.zeros_like(image)
    for mask, cmap in zip(masks, cmaps):
        mapped += overlay_cmap(image, mask, opacity=opacity, rescale_mask=rescale_mask, cmap=cmap, return_mask=True)
    overlay = cv2.addWeighted(image, 1 - opacity, mapped, opacity, 0)
    return overlay


def overlay_divergent_cmaps(image, mask, threshold=0, opacity=0.5, rescale_mask=True, cmaps=['green', 'red']):
    """Maybe this one too?"""
    image = gimage.to_uint8(np.squeeze(image))
    image = np.dstack([image] * 3)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    shifted_mask = mask - threshold
    max_val = np.abs(shifted_mask).max() if rescale_mask else 1
    masks = [np.clip(shifted_mask, 0, max_val) / max_val, (-1 * np.clip(shifted_mask, -max_val, 0)) / max_val]

    mapped = np.zeros_like(image)
    for mask, cmap in zip(masks, cmaps):
        mapped += overlay_cmap(image, mask, opacity=opacity, rescale_mask=False, cmap=cmap, return_mask=True)
    overlay = cv2.addWeighted(image, 1 - opacity, mapped, opacity, 0)
    return overlay


import fnmatch
import os


# TODO - Add this to gouda.path_methods and gouda.GoudaPath
def multiglob(path, patterns, ignore_case=False):
    def inner_compile(x):
        x = os.path.normcase(x)
        if ignore_case:  # already done on windows, but not on posix
            x = x.lower()
        return fnmatch._compile_pattern(x)
    results = [[] for _ in patterns]
    patterns = [inner_compile(pattern) for pattern in patterns]
    for path_entry in os.scandir(label_dir):
        item = os.path.normcase(path_entry)
        if ignore_case:
            item = item.lower()
        for idx, pattern in enumerate(patterns):
            if pattern(item):
                results[idx].append(path_entry.path)
    return results


def search_for_dicom(base_dir):
    possible = []
    for root, dirs, files in os.walk(base_dir):
        has_dicom = False
        for item in files:
            if item.endswith('.dcm'):
                has_dicom = True
                break
        if has_dicom:
            possible.append(root)
    return possible


def clip_iqr(data, iqr_range=1.5, axis=1):
    clipped = []
    for group in np.split(data, data.shape[axis], axis=axis):
        q1, q3 = np.percentile(group, [25, 75])
        iqr = np.abs(q3 - q1)
        clipped.append(np.clip(group, q1 - iqr * iqr_range, q3 + iqr * iqr_range))
    return np.concatenate(clipped, axis=axis)


###############################
import multiprocessing as mp
import numpy as np
import tqdm.auto as tqdm
from timeit import default_timer as tic


def check_time(method, iterations=10, args=[], kwargs={}):
    times = []
    for _ in range(iterations):
        start = tic()
        method(*args, **kwargs)
        times.append(tic() - start)
    print("{:.4f}s +- {:.4f} per iteration".format(np.mean(times), np.std(times)))
    print("{:.4f}s total".format(np.sum(times)))


class Timer(object):
    def __init__(self, start=False, title='', max_message_length=10):
        self.start_time = None
        self.prev_time = None
        self.title = title
        self.out_string = "{:" + str(max_message_length) + "s}: {:.4f}"
        if start:
            self.start()

    def start(self):
        self.start_time = tic()
        self.prev_time = self.start_time
        if self.title is None:
            print("Start")
        else:
            print("Start: {}".format(self.title))

    def stop(self):
        print('Stop {}- Total: {:.4f}'.format(self.title + ' ', tic() - self.start_time))

    def __call__(self, message=None, last=False):
        if message is None:
            message = 'Tic'
        step = tic()
        print(self.out_string.format(message, step - self.prev_time))
        if last:
            print('Stop {} - Total: {:.4f}'.format(self.title, step - self.start_time))
        self.prev_time = tic()


def pbar_pool(function, list_of_args):
    pool = mp.Pool()
    procs = []
    for item in list_of_args:
        procs.append(pool.apply_async(function, args=item))
    pbar = tqdm.tqdm(total=len(procs))
    for p in procs:
        p.get()
        pbar.update(1)
    pool.close()
    pool.terminate()
    pool.join()
    pbar.close()


class PrintOnce:
    def __init__(self, message=None):
        if message is None:
            self.messages = []
        elif isinstance(message, str):
            self.messages = [message]
        elif isinstance(message, Iterable):
            self.messages = list(message)
        else:
            self.messages = [str(message)]

    def add(self, message):
        self.messages.append(str(message))
        return self

    def copy(self):
        new_self = PrintOnce()
        new_self.messages = self.messages[:]
        return new_self

    def print(self):
        for item in self.messages:
            print(item)
        self.messages = []

    def clear(self, num=-1):
        if num == -1:
            self.messages = []
        else:
            num = min(num, len(self))
            self.messages = self.messages[:len(self) - num]

    @property
    def empty(self):
        return len(self.messages) == 0

    def __len__(self):
        return len(self.messages)



def create_ellipsoid(shape, center, radius, ndim=None):
    # Really just `create_sphere` - see Code.ChestSeg.notebooks.Explainability
    (shape, center), ndim = gouda.match_len(shape, center, count=ndim)
    coords = np.ogrid[tuple([slice(0, shape[i]) for i in range(ndim)])]
    distance = np.sqrt(sum([(coords[i] - center[i]) ** 2 for i in range(ndim)]))
    return distance <= radius


from contextlib import contextmanager


# https://stackoverflow.com/a/52743526/2348288
@contextmanager
def preserve_limits(ax=None, save_x=True, save_y=True):
    """ Plot without modifying axis limits """
    if ax is None:
        ax = plt.gca()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    try:
        yield ax
    finally:
        if save_x:
            ax.set_xlim(xlim)
        if save_y:
            ax.set_ylim(ylim)


def set_dict_defaults(kwargs: Union[None, dict], defaults: dict):
    """Set default values for dictionary

    Parameters
    ----------
    kwargs : Union[None, dict]
        Dictionary to update with defaults
    defaults : dict
        Dictionary of default key/values
    """
    if kwargs is None:
        kwargs = {}
    for key in defaults:
        kwargs.setdefault(key, defaults[key])
    return kwargs


# NOTE - Depends on pandas... maybe put in plotting?
def group_melt(data, id_var, group_var, group_vals, merge_names, group_vars):
    grouped_results = []
    assert len(group_vals) == len(group_vars), 'There should be 1 set of group variables for each split value'
    for row in data.itertuples():
        for idx, vars in enumerate(group_vars):
            group_data = {id_var: getattr(row, id_var), group_var: group_vals[idx]}
            for sub_idx, key in enumerate(vars):
                group_data[merge_names[sub_idx]] = getattr(row, key)
            grouped_results.append(group_data)
    grouped_results = pd.DataFrame(grouped_results)
    return grouped_results


import io
import sys


class OutputMuffle:
    def __init__(self, to_muffle='stdout'):
        self.stream = [to_muffle] if isinstance(to_muffle, str) else to_muffle
        self._old_targets = {key: [] for key in self.stream}
        self.new_targets = {key: io.StringIO() for key in self.stream}

    def __enter__(self):
        for key in self.stream:
            self._old_targets[key].append(getattr(sys, key))
            setattr(sys, key, self.new_targets[key])
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        for key in self.stream:
            setattr(sys, key, self._old_targets[key].pop())

    def __getitem__(self, key):
        return self.new_targets[key].getvalue()


def nested_indexing(*max_idx):
    indices = [np.arange(val) for val in max_idx]
    return np.stack(np.meshgrid(*indices, indexing='ij'), -1).reshape(-1, len(max_idx))
