"""These are all matplotlib helper methods, and are still being developed."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path


def parse_color(color, float_cmap='viridis', int_cmap='Set1'):
    """Convert the input to a rgb color

    NOTE
    ----
    Recognizes all formats that can be used with matplotlib in addition to rgb/rgba tuples as strings [ie. '(0.1, 0.2, 0.5)'] and single values.
    Single values will be mapped to the given matplotlib colormap. Ints will wrap-around and floats will be clipped to [0, 1].
    """
    try:
        return matplotlib.colors.to_rgb(color)
    except ValueError:
        if isinstance(color, str):
            # Format is comma- and/or space-separated values
            color.translate(None, '()[]')
            if ', ' in color:
                divided = color.split(', ')
            elif ',' in color:
                divided = color.split(',')
            else:
                divided = color.split(' ')
            rgb = np.array(divided).astype(np.float32)
            return matplotlib.colors.to_rgb(rgb / 255 if rgb.max() > 1.0 else rgb)
        elif isinstance(color, float):
            return matplotlib.cm.get_cmap(float_cmap)
        elif isinstance(color, int):
            return matplotlib.cm.get_cmap(int_cmap)(color % 9)
        else:
            # Format is any array-like set of values
            rgb = np.array(color).astype(np.float32)
            return matplotlib.colors.to_rgb(rgb / 255 if rgb.max() > 1.0 else rgb)


def plot_accuracy_curve(acc, thresh, label_height=0.5, line_args={}, thresh_args={}):
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
    line_defaults = {'color': 'black', 'label': 'accuracy'}
    for key in line_defaults:
        if key not in line_args:
            line_args[key] = line_defaults[key]
    thresh_defaults = {'color': 'r', 'label': 'best threshold'}
    for key in thresh_defaults:
        if key not in thresh_args:
            thresh_args[key] = thresh_defaults[key]

    best_idx = np.argmax(acc)
    best_thresh = thresh[best_idx]
    best_acc = acc[best_idx]

    plt.plot(thresh, acc, **line_args)
    plt.vlines(best_thresh, 0, 1, **thresh_args)
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    if label_height is not None:
        plt.text(best_thresh + 0.01, label_height, "{:.2f}% @ {:.2f}".format(best_acc * 100, best_thresh))
    plt.xlabel('Prediction Threshold')
    plt.ylabel('Accuracy')


def quick_line(x1, x2, y1, ytop, y2, lw=1):
    """Generate a path in matplotlib"""
    verts = [
        (x1, y1),
        (x1, ytop),
        (x2, ytop),
        (x2, y2)
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO
    ]
    path = Path(verts, codes)
    patch = PathPatch(path, fill=False, lw=1)
    return patch


def annotate_arrows(ax,
                    text,
                    start_point,
                    end_points,
                    y_top=None,
                    line_width=1,
                    direction='left',
                    x_spacing=0.05,
                    y_spacing=0.07,
                    y_top_spacing=0.02,
                    repeat_text=True,
                    text_spacing=0.02):
    """Add arrows that start at a point, go upwards, go over, and then descend to another point - mostly useful for bar graphs

    Parameters
    ----------
    ax : plt.Axis (?)
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
    end_points = sorted(end_points, key=lambda x: x[0], reverse=direction == 'left')
    if not repeat_text:
        text_spacing = 0.
    if y_top is None:
        y_top = max([point[1] for point in end_points] + [start_point[1]]) + y_top_spacing + y_spacing

    x_jitter = np.arange(len(end_points)) * x_spacing
    x_jitter -= x_jitter[-1] / 2.

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
            ax.annotate(text, xy=((x_start + x_end) / 2., max_y), ha='center', va='bottom', zorder=10)
        ax.add_patch(line)

    x_vals = [point[0] for point in end_points] + [x_start]
    mid_x = (min(x_vals) + max(x_vals)) / 2.
    if not repeat_text:
        ax.annotate(text, xy=(mid_x, y_top + y_top_spacing * (len(end_points) - 1)), ha='center', va='bottom', zorder=10)
