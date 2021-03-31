"""These are all matplotlib helper methods, and are still being developed."""
import matplotlib.pyplot as plt
import numpy as np


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
