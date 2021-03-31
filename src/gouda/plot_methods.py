"""These are all matplotlib helper methods, and are still being developed."""
import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_curve(acc, thresh):
    """Plot the accuracy curve for the given accuracy and threshold values.

    Parameters
    ----------
    acc: np.ndarray
        An array of the accuracy values for each threshold.
    thresh: np.ndarray
        An array of the threshold values to plot
    """
    plt.plot(thresh, acc)
    best_idx = np.argmax(acc)
    plt.vlines(thresh[best_idx], 0, 1, color='r')
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.text(thresh[best_idx] + 0.01, 0.5, "{:.2f}%".format(acc[best_idx] * 100))
    plt.xlabel("Prediction Threshold")
    plt.ylabel("Accuracy")
