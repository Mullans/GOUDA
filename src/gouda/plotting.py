"""matplotlib plotting and helper methods."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from gouda.color_lists import find_color_hex
from gouda.data_methods import line_dist, rescale, segment_line
from gouda.general import is_iter
from gouda.typing import ColorType, Unpack


def parse_color(color: ColorType, float_cmap: str = "viridis", int_cmap: str = "Set1") -> tuple[float, float, float]:
    """Convert the input to a rgb color.

    NOTE
    ----
    Recognizes all formats that can be used with matplotlib in addition to rgb/rgba tuples as strings [ie. '(0.1, 0.2, 0.5)'] and single values.
    Single values will be mapped to the given matplotlib colormap. Ints will wrap-around and floats will be clipped to [0, 1].

    TODO - add lookup to get hexcodes from colors.py
    """
    if isinstance(color, str):  # If the color is a string, try to find the named color as a hexcode
        check = find_color_hex(color)
        if check is not None:
            color = check
    try:
        return mpl.colors.to_rgb(color)  # type: ignore  # NOTE - same as to_rgba[:3]
    except ValueError:
        if isinstance(color, str):
            # Format is comma- and/or space-separated values
            bad_chars = str.maketrans(dict.fromkeys("()[]", ""))
            color.translate(bad_chars)
            if ", " in color:
                divided = color.split(", ")
            elif "," in color:
                divided = color.split(",")
            else:
                divided = color.split(" ")
            rgb = np.array(divided).astype(np.float32)
            return mpl.colors.to_rgb(rgb / 255 if rgb.max() > 1.0 else rgb)  # type: ignore
        elif isinstance(color, float):
            return mpl.colormaps.get_cmap(float_cmap)
        elif isinstance(color, int):
            return mpl.colormaps.get_cmap(int_cmap)(color % 9)
        else:
            # Format is any array-like set of values
            rgb = np.array(color).astype(np.float32)
            return mpl.colors.to_rgb(rgb / 255 if rgb.max() > 1.0 else rgb)  # type: ignore


def plot_accuracy_curve(
    acc: npt.NDArray[np.floating],
    thresh: npt.NDArray[np.floating],
    label_height: float = 0.5,
    line_args: dict | None = None,
    thresh_args: dict | None = None,
) -> None:
    """Plot the accuracy for the given accuracy and threshold values.

    Parameters
    ----------
    acc : numpy.ndarray
        An array of accuracy values for each given threshold
    thresh : numpy.ndarray
        An array of threshold values matching each accuracy value
    label_height : float
        The value between 0 and 1 for the y-position of the threshold label on the graph. Set to None for no label (default is 0.5)
    line_args : dict
        A dictionary of keyword arguments to pass when plotting the accuracy line
    thresh_args : dict
        A dictionary of keyword arguments to pass when plotting the threshold vline
    """
    line_args = {} if line_args is None else line_args
    thresh_args = {} if thresh_args is None else thresh_args
    line_defaults = {"color": "black", "label": "accuracy"}
    for key, val in line_defaults.items():
        if key not in line_args:
            line_args[key] = val
    thresh_defaults = {"color": "r", "label": "best threshold"}
    for key, val in thresh_defaults.items():
        if key not in thresh_args:
            thresh_args[key] = val

    best_idx = np.argmax(acc)
    best_thresh = thresh[best_idx]
    best_acc = acc[best_idx]

    plt.plot(thresh, acc, **line_args)
    plt.vlines(best_thresh, 0, 1, **thresh_args)
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    if label_height is not None:
        plt.text(best_thresh + 0.01, label_height, f"{best_acc * 100:.2f}% @ {best_thresh:.2f}")
    plt.xlabel("Prediction Threshold")
    plt.ylabel("Accuracy")


def quick_line(x1: float, x2: float, y1: float, ytop: float, y2: float, lw: int = 1) -> PathPatch:
    """Generate a path in matplotlib."""
    verts = [(x1, y1), (x1, ytop), (x2, ytop), (x2, y2)]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
    path = Path(verts, codes)
    patch = PathPatch(path, fill=False, lw=1)
    return patch


def annotate_arrows(
    ax: mpl.axes.Axes,
    text: str,
    start_point: tuple[float, float],
    end_points: list[tuple[float, float]],
    y_top: float | None = None,
    line_width: int = 1,
    direction: Literal["left", "right"] = "left",
    x_spacing: float = 0.05,
    y_spacing: float = 0.07,
    y_top_spacing: float = 0.02,
    repeat_text: bool = True,
    text_spacing: float = 0.02,
) -> None:
    """Add arrows that start at a point, go upwards, go over, and then descend to another point - mostly useful for bar graphs.

    Parameters
    ----------
    ax : mpl.axes.Axes
        Axis to plot on
    text : str
        text to annotate over lines
    start_point : (float, float)
        x, y coordinates of the base location
    end_points : list of (float, float)
        x, y end-points for the lines
    y_top : float
        The height of the top of the lines. If none, automatically selected above the highest point
    line_width : float
        line width
    direction : 'left' | 'right'
        The direction of the end-points from the base
    x_spacing : float
        horizontal distance between starting points
    y_spacing : float
        vertical distance from point to line start
    y_top_spacing : float
        vertical distance between top lines
    repeat_text : bool
        if true, text added over top line. If false, text added over each line and vertical spacing increased
    text_spacing : bool
        extra spacing between lines if text repeated
    """
    end_points = sorted(end_points, key=lambda x: x[0], reverse=direction == "left")
    if not repeat_text:
        text_spacing = 0.0
    if y_top is None:
        y_top = max([point[1] for point in end_points] + [start_point[1]]) + y_top_spacing + y_spacing

    x_jitter = np.arange(len(end_points)) * x_spacing
    x_jitter -= x_jitter[-1] / 2.0

    # best_mid = [0, None]
    x_start, y_start = start_point
    y_start = y_start + y_spacing
    max_y = 0
    for i in range(len(end_points)):
        x_end, y_end = end_points[i]
        y_end = y_end + y_spacing
        max_y = y_top + (y_top_spacing + text_spacing) * i
        line = quick_line(x_start + x_jitter[i], x_end, y_start, max_y, y_end, lw=line_width)
        if repeat_text:
            ax.annotate(text, xy=((x_start + x_end) / 2.0, max_y), ha="center", va="bottom", zorder=10)
        ax.add_patch(line)

    x_vals = [point[0] for point in end_points] + [x_start]
    mid_x = (min(x_vals) + max(x_vals)) / 2.0
    if not repeat_text:
        ax.annotate(
            text, xy=(mid_x, y_top + y_top_spacing * (len(end_points) - 1)), ha="center", va="bottom", zorder=10
        )


def colorplot(
    x: Sequence[float],
    y: Sequence[float],
    ax: mpl.axes.Axes | None = None,
    cmap: mpl.colors.Colormap | str = "jet",
    step_size: float = 0.01,
    step_as_percent: bool = True,
    start_val: float = 0.0,
    end_val: float = 1.0,
    **kwargs: Unpack[tuple[str, Any]],
) -> mpl.axes.Axes:
    """Plot a line with a color gradient.

    Parameters
    ----------
    x : Iterable
        x values to plot
    y : Iterable
        y values to plot
    ax : Optional[matplotlib.axes.Axes], optional
        Pre-existing axes for the plot. If None, uses :meth:`matplotlib.pyplot.gca`. By default None
    cmap : Union[matplotlib.colors.Colormap, str], optional
        Colormap to use for coloring the line, by default 'jet'
    step_size : float, optional
        Maximum length of each color segment (segments between points will always be at least 1 step), by default 0.01
    step_as_percent: bool, optional
        If True, `step_size` will be interpreted as a percent of the total line length. If False, `step_size` will be interpreted as an absolute distance, by default True
    start_val : float
        Starting value for the color gradient as a percent of the colormap. Values must be between 0 and 1, the start and end of the colormap respectively. By default 0.0
    end_val : float
        Ending value for the color gradient as a percent of the colormap. Values must be between 0 and 1, the start and end of the colormap respectively. By default 1.0
    **kwargs : dict
        Optional key-word arguments to be passed to :class:`matplotlib.collections.LineCollection`

    Returns
    -------
    matplotlib.pyplot.Axes
        The axes used for the plot

    Notes
    -----
    The full line will iterate from 0 to 1 along the given colormap.

    Each line segment will always be at least 1 step along the color gradient. If the line segment is longer than 1 step, the segment will be subdivided into smaller segments of approximately length `step_size` (see :meth:`gouda.segment_line` for details).
    """
    if not (is_iter(x) and is_iter(y)):
        raise ValueError("x and y must be iterables of values to plot")
    if len(x) != len(y):
        raise ValueError(f"x and y must be the same length but got {len(x)} and {len(y)}")

    if ax is None:
        ax = plt.gca()
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    if step_as_percent:
        step_size = line_dist(x, y) * step_size

    all_segments = []
    for idx in range(len(x) - 1):
        x1, x2 = x[idx : idx + 2]
        y1, y2 = y[idx : idx + 2]
        segments = segment_line(x1, x2, y1, y2, step_size=step_size)
        all_segments.append(segments)
    all_segments = np.concatenate(all_segments, axis=0)
    segment_dists = np.sqrt(
        (all_segments[:, 1, 0] - all_segments[:, 0, 0]) ** 2 + (all_segments[:, 1, 1] - all_segments[:, 0, 1]) ** 2
    )
    segment_dists = rescale(np.cumsum(segment_dists), start_val, end_val)
    lc = mpl.collections.LineCollection(
        all_segments, array=segment_dists, cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=1), **kwargs
    )
    ax.add_collection(lc)
    if ax.get_autoscale_on():
        ax.autoscale()
    return ax, lc


def plot_joint_arrow(
    x: Sequence[float],
    y: Sequence[float],
    linewidth: float = 5.0,
    headwidth: float = 2.0,
    headlength: float | None = None,
    ax: mpl.axes.Axes | None = None,
    color: ColorType | None = None,
    label: str | None = None,
) -> mpl.axes.Axes:
    """Plot a jointed line ending in an arrowhead.

    Parameters
    ----------
    x : Iterable
        x values to plot
    y : Iterable
        y values to plot
    linewidth : float, optional
        Width of the line to draw, by default 5.0
    headwidth : float, optional
        Width of the arrowhead relative to `linewidth`, by default 2.0
    headlength : Optional[float], optional
        Length of the arrowhead relative to `linewidth`. If None, defaults to `headwidth`. By default None
    ax : Optional[matplotlib.axes.Axes], optional
        Pre-existing axes for the plot. If None, uses :meth:`matplotlib.pyplot.gca`. By default None
    color : Optional[gouda.ColorType], optional
        Color for the line and arrowhead, by default None
    label : str, optional
        Label that will be displayed in the legend, by default None

    Returns
    -------
    matplotlib.pyplot.Axes
        The axes used for the plot
    """
    if not (is_iter(x) and is_iter(y)):
        raise ValueError("x and y must be iterables of values to plot")
    if len(x) != len(y):
        raise ValueError(f"x and y must be the same length but got {len(x)} and {len(y)}")

    if ax is None:
        ax = plt.gca()

    linewidth = mpl.rcParams["lines.linewidth"] if linewidth is None else linewidth
    mutation_scale = mpl.rcParams["font.size"]
    line_kwargs = {}
    arrow_kwargs = {"mutation_scale": mutation_scale}
    arrowstyle = "simple"
    style_scale = linewidth / mutation_scale
    arrowstyle = mpl.patches.ArrowStyle(
        arrowstyle, tail_width=style_scale, head_width=style_scale * headwidth, head_length=style_scale * headlength
    )

    points = np.column_stack([x, y])
    points[-1] = (points[-2] + points[-1]) / 2  # only go halfway so it doesn't overlap arrowhead
    path = mpl.path.Path(points, [mpl.path.Path.MOVETO] + [mpl.path.Path.LINETO] * (len(points) - 1))
    patch = mpl.patches.PathPatch(path, facecolor="none", edgecolor=color, lw=linewidth, label=None, **line_kwargs)
    ax.add_patch(patch)

    arrow = mpl.patches.FancyArrowPatch(
        [x[-2], y[-2]],
        [x[-1], y[-1]],
        color=color,
        label=label,
        lw=0,
        shrinkA=0,
        shrinkB=0,
        arrowstyle=arrowstyle,
        **arrow_kwargs,
    )
    ax.add_patch(arrow)
    plt.plot([min(x), max(x)], [min(y), max(y)], alpha=0, label=None)
    return ax
