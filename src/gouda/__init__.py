# -*- coding: utf-8 -*-
"""Miscellaneous data utilities."""
import copy
import json
import os
from pkg_resources import DistributionNotFound, get_distribution

import colorama
import numpy as np

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = 'GOUDA'
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound


def arr_sample(arr, rate):
    """Return an array linearly sampled from the input array at the given rate.

    Examples
    --------
    [1, 2, 3, 4] and rate 2   -> [1, 3]
    [1, 2, 3, 4] and rate 0.5 -> [1, 1, 2, 2, 3, 3, 4, 4]
    """
    if arr.ndim != 1:
        raise ValueError("Only 1d arrays can be sampled from.")
    i = 0
    out = []
    while i < arr.shape[0]:
        out.append(arr[np.floor(i).astype(np.int)])
        i += rate
    return np.array(out)


def ensure_dir(path):
    """Check if a given directory exists, and create it if it doesn't."""
    import os
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def next_filename(filename):
    """Check if a given file exists, and return a new filename for a numbered copy if it does."""
    if os.path.isfile(filename):
        base, extension = filename.rsplit('.', 1)
        i = 2
        while True:
            next_check = '{}_{}.{}'.format(base, i, extension)
            if os.path.isfile(next_check):
                i += 1
            else:
                return next_check
    else:
        return filename


def sigmoid(x):
    """Return the sigmoid of the given value/array."""
    return 1.0 / (1.0 + np.exp(-x) + 1e-7)


def _unnumpy(data):
    """Convert numpy arrays to lists for JSON"""
    if type(data) == list:
        new_data = []
        for i in range(len(data)):
            new_data.append(_unnumpy(data[i]))
    elif type(data) == dict:
        new_data = {}
        for key in data.keys():
            new_data[key] = _unnumpy(data[key])
    elif type(data) == np.ndarray:
        new_data = {"numpy_array": data.tolist(), "dtype": str(data.dtype), "shape": data.shape}
    else:
        new_data = copy.copy(data)
    return new_data


def _renumpy(data):
    """Convert JSON back to numpy arrays"""
    if type(data) == list:
        for i in range(len(data)):
            data[i] = _renumpy(data[i])
    elif type(data) == dict:
        if "numpy_array" in data:
            data = np.array(data['numpy_array']).astype(data['dtype']).reshape(data['shape'])
        else:
            for key in data.keys():
                data[key] = _renumpy(data[key])
    return data


def save_json(data, filename, numpy=False):
    """Save a list/dict as a json object."""
    if numpy:
        data = _unnumpy(data)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)


def load_json(filename, numpy=False):
    """Load a json file as a list/dict."""
    with open(filename, 'r') as f:
        data = json.load(f)
        if numpy:
            data = _renumpy(data)
        return data


# Confusion Matrix utils
def underline(string):
    """Underline a string with ANSI escape characters"""
    return '\033[4m' + string + '\033[0m'


def get_specificities(confusion_matrix):
    """Return the specificity for each column in a confusion matrix"""
    return [confusion_matrix[i, i] / confusion_matrix[:, i].sum() for i in range(confusion_matrix.shape[0])]


def get_accuracy(confusion_matrix):
    """Return the accuracy from a confusion matrix"""
    return np.sum([confusion_matrix[i, i] for i in range(confusion_matrix.shape[0])]) / np.sum(confusion_matrix)


def get_binary_confusion_matrix(predictions, labels, threshold=0.5):
    """Get confusion matrix for 2-class predictions.

    Note
    ----
    Predictions can be either boolean values or continuous probabilities
    """
    if predictions.dtype != np.bool:
        rounded_pred = predictions > threshold
    else:
        rounded_pred = predictions
    output = np.empty((2, 2))
    output[1, 1] = np.where(rounded_pred == labels, labels, 0).sum()  # true positive
    output[0, 0] = (1 - np.where(rounded_pred == labels, labels, 1)).sum()  # true negative
    output[0, 1] = rounded_pred.sum() - output[1, 1]  # false positive
    output[1, 0] = (1 - rounded_pred.sum()) - output[0, 0]  # false negative
    return output


def get_confusion_matrix(predictions, labels, num_classes=None):
    """Get confusion matrix for multi-class predictions.

    Note
    ----
    Predictions and labels must both be integer class labels.
    """
    if num_classes is None:
        num_classes = labels.max()
    confusion = np.zeros((num_classes, num_classes))
    for i in range(predictions.shape[0]):
        confusion[labels[i], predictions[i]] += 1
    return confusion


def print_confusion_matrix(confusion_matrix):
    """Format and print a confusion matrix"""
    expected_string = u"\u2193" + " Expected"
    predicted_string = u"\u2192" + "  Predicted"
    leading_space = "            "
    confusion_string = "         "
    header_string = "".join(['|   {:1d}   '.format(i) for i in range(confusion_matrix.shape[1])])
    confusion_string += predicted_string + "\n" + expected_string + "  " + underline("        " + header_string + "| Sensitivity") + "\n"
    for i in range(confusion_matrix.shape[0]):
        correct = confusion_matrix[i, i]
        line_string = "    {:1d}   |"
        for j in range(confusion_matrix.shape[1]):
            if i == j:
                line_string += colorama.Fore.GREEN + "{:5d}  " + colorama.Style.RESET_ALL + "|"
            else:
                line_string += "{:5d}  |"
        line_string += "{:.5f}"
        line_string = line_string.format(i, *confusion_matrix[i], correct / confusion_matrix[i].sum())
        if i == confusion_matrix.shape[0] - 1:
            line_string = underline(line_string)
        confusion_string += leading_space + line_string + '\n'

    specificities = get_specificities(confusion_matrix)
    specificity_string = '        Specificity'
    for _ in range(confusion_matrix.shape[1]):
        specificity_string += ' |{:.4f}'
    confusion_string += (specificity_string + '\n').format(*specificities)
    confusion_string += "\nAccuracy: {:.4f}".format(get_accuracy(confusion_matrix))

    print(confusion_string)


class ConfusionMatrix(object):
    """
    2D array to represent and evaluate a confusion matrix.

    Parameters
    ----------
    predictions : type
        Optional initial predictions for the matrix. (the default is None)
    labels : type
        Optional initial lables for the matrix. (the default is None)
    threshold : float
        Threshold to use for binary labels with continuous predictions
    num_classes : int
        Number of classes to use for the matrix
    dtype : numpy.dtype
        Numpy variable type to use for the matrix

    Attributes
    ----------
    shape : [int, int]
        Shape of the matrix
    size : [int, int]
        Size of the matrix (not to be confused with :count: `gouda.ConfusionMatrix.count`)
    dtype : numpy.dtype
        Variable type of matrix items
    num_classes : int
        Number of possible classes


    Note
    ----
    Threshold only used for binary class probabilities

    Matrix is 0-indexed

    Dtype may be set to change memory usage, but will always be
    treated as an int. No checking is done to prevent overflow if
    dtype is manually set.
    """
    def __init__(self, predictions=None, labels=None, threshold=None, num_classes=None, dtype=np.int):

        self.matrix = None
        self._num_classes = 0
        self.threshold = threshold
        self.add_warned = False
        if num_classes is not None:
            self.reset(num_classes, dtype=dtype)
        if predictions is not None and labels is not None:
            self.add(predictions, labels, threshold=threshold)

    @property
    def shape(self):
        return [self._num_classes, self._num_classes]

    @property
    def size(self):
        return self._num_classes * self._num_classes

    @property
    def dtype(self):
        return self.matrix.dtype

    @property
    def num_classes(self):
        return self._num_classes

    @dtype.setter
    def dtype(self, dtype):
        self.matrix = self.matrix.astype(dtype)

    def reset(self, num_classes=None, dtype=np.int):
        """Reset all matrix entries

        Parameters
        ----------
        num_classes : int
            The number of classes in the new matrix. (defaults to the current matrix size).
        dtype : type
            Numpy type of the new matrix
        """
        if num_classes is None:
            num_classes = self._num_classes
        if num_classes <= 0:
            raise ValueError("Matrix must have at least 1 class")
        self._num_classes = num_classes
        self.matrix = np.zeros((self._num_classes, self._num_classes), dtype=dtype)

    def __iadd__(self, data):
        """Add single datapoint (predicted, expected)"""
        self.add(data[0], data[1])
        return self

    def __add__(self, matrix):
        """Add two matrices together.

        NOTE: Output dtype defaults to first matrix type."""
        output_size = max(self._num_classes, matrix.num_classes)
        output = np.zeros((output_size, output_size), dtype=self.dtype)
        output[:self._num_classes, :self._num_classes] += self.matrix
        output[:matrix.num_classes, :matrix.num_classes] += matrix.matrix

    def __str__(self):
        return str(self.matrix)

    def count(self):
        """Count the number of items in the matrix"""
        return self.matrix.sum()

    def accuracy(self):
        """Get the total accuracy in the matrix"""
        return np.sum([self.matrix[i, i] for i in range(self._num_classes)]) / np.sum(self.matrix)

    def specificity(self, class_label=None):
        """Return the specificity of all classes or a single class.

        NOTE: specificity = (true negative) / (true negative + false positive) for each class.
        """
        if class_label is None:
            tn = np.array([sum([self.matrix[j, :i].sum() + self.matrix[j, i + 1:].sum() for j in range(self._num_classes) if j != i]) for i in range(self._num_classes)])
            fp = np.array([self.matrix[i, :].sum() - self.matrix[i, i].sum() for i in range(self._num_classes)])
            return tn / (tn + fp)
        else:
            tn = sum([self.matrix[j, :class_label].sum() + self.matrix[j, class_label + 1:].sum() for j in range(self._num_classes) if j != class_label])
            fp = self.matrix[class_label, :].sum() - self.matrix[class_label, class_label].sum()
            return tn / (tn + fp)

    def sensitivity(self, class_label=None):
        """Return the sensitivity of all classes or a single class.

        NOTE: sensitivity = (true positive) / (true positive + false negative) for each class.
        """
        if class_label is None:
            return [self.matrix[i, i] / self.matrix[i, :].sum() if self.matrix[i, :].sum() > 0 else 0 for i in range(self._num_classes)]
        else:
            return self.matrix[class_label, class_label] / self.matrix[class_label, :].sum()

    def add(self, predicted, expected, threshold=None):
        """Add data to the confusion Matrix

        Parameters
        ----------
        predicted : [bool, float, int, list]
            Predicted value(s) to add to the matrix
        expected : [bool, float, int, list]
            Expected value(s) to add to matrix
        threshold : type
            Threshold used for predicted probabilities. Defaults to self.threshold

        NOTE
        ----
        Accepted formats:
            * predicted class     vs expected class
            * probability of true vs expected boolean
            * class probabilities vs expected class
            * list of predictions vs list of class labels (equal length)
        """
        if threshold is None:
            threshold = self.threshold
        if isinstance(predicted, (float, np.float_)):
            # Single value: prediction of True (class 1)
            if self.threshold is not None:
                predicted_class = 1 if predicted > threshold else 0
            else:
                if not self.add_warned:
                    self.add_warned = True
                    print("Warning: Float predicted classes without a threshold are rounded to the nearest integer.")
                predicted_class = np.round(predicted).astype(int)
        elif isinstance(predicted, (int, np.int_)):
            # Single value: class label
            predicted_class = predicted
        elif isinstance(predicted, (bool, np.bool_)):
            predicted_class = 1 if predicted else 0
        elif isinstance(expected, (float, int, bool, np.bool_)):
            # Class probabilities with single expected label
            predicted_class = np.argmax(predicted).astype(int)
        elif len(predicted) == len(expected):
            # Paired lists
            for x, y in zip(predicted, expected):
                self.add(x, y, threshold=threshold)
            return
        else:
            raise ValueError("Unsupported input format")
        if not isinstance(expected, (float, np.float_, np.int_, int, bool, np.bool_)):
            print(type(expected))
            raise ValueError("Only 1 expected value per prediction is supported")
        if isinstance(expected, (bool, np.bool_)):
            expected_class = 1 if expected else 0
        elif expected % 1 == 0:
            expected_class = np.round(expected).astype(int)
        else:
            raise ValueError("Expected values must be class label integers or boolean")

        max_in = max(predicted_class, expected_class) + 1
        if self.matrix is None:
            self.reset(max_in)
        if self._num_classes < max_in:
            new_matrix = np.zeros((max_in, max_in), dtype=self.dtype)
            new_matrix[:self._num_classes, :self._num_classes] += self.matrix
            self.matrix = new_matrix
            self._num_classes = max_in
        self.matrix[expected_class, predicted_class] += 1

    def print(self, return_string=False):
        """Format and print the confusion matrix

        Parameters
        ----------
        return_string : bool
            Whether to return a plain-text version of the matrix (the default is False).

        Returns
        -------
        str
            Confusion matrix formatted for plain-text printing if return_string is True.

        """
        specificities = self.specificity()
        sensitivities = self.sensitivity()
        expected_string = u"\u2193" + " Expected"
        predicted_string = u"\u2192" + "  Predicted"
        leading_space = "            "
        confusion_string = "         "
        header_string = "".join(['|   {:1d}   '.format(i) for i in range(self._num_classes)])
        confusion_string += predicted_string + "\n" + expected_string + "  " + underline("        " + header_string + "| Sensitivity") + "\n"
        for i in range(self._num_classes):
            line_string = "    {:1d}   |"
            for j in range(self._num_classes):
                if i == j:
                    line_string += colorama.Fore.GREEN + "{:5d}  " + colorama.Style.RESET_ALL + "|"
                else:
                    line_string += "{:5d}  |"
            line_string += "{:.5f}"
            line_string = line_string.format(i, *self.matrix[i], sensitivities[i])
            if i == self.num_classes - 1:
                line_string = underline(line_string)
            confusion_string += leading_space + line_string + '\n'

        specificity_string = '        Specificity'
        for _ in range(self._num_classes):
            specificity_string += ' |{:.4f}'
        confusion_string += (specificity_string + '\n').format(*specificities)
        confusion_string += "\nAccuracy: {:.4f}".format(self.accuracy())

        print(confusion_string)
        if return_string:
            for item in [colorama.Fore.GREEN, colorama.Style.RESET_ALL, '\033[4m', '\033[0m']:
                confusion_string = confusion_string.replace(item, '')
            return confusion_string


# Moving statistics classes
class MMean(object):
    """Class to hold a moving mean with constant-time update and memory."""

    def __init__(self):
        self.mean = 0.0
        self.count = 0

    def __iadd__(self, value):
        """Update the mean, including the given value."""
        self.count += 1
        self.mean += (1.0 / self.count) * (value - self.mean)
        return self

    def __add__(self, value):
        """Add a new value to the mean, does not update class values."""
        return self.mean + value

    def __sub__(self, value):
        """Subtract a new value from the mean, does not update class values."""
        return self.mean - value

    def __str__(self):
        """Return the mean as a string."""
        return str(self.mean)

    def val(self):
        """Return the mean."""
        return self.mean


class MStddev(object):
    """Class to hold a moving standard deviation with constant-time update and memory."""

    def __init__(self):
        self.count = 0.0
        self.mean = 0.0
        self.stddev = 0.0

    def __iadd__(self, value):
        """Update the mean and stddev, including the new value."""
        self.count += 1
        prev_mean = self.mean
        self.mean += (1.0 / self.count) * (value - self.mean)
        self.stddev += (value - self.mean) * (value - prev_mean)
        return self

    def __add__(self, value):
        """Add a value to the stddev, does not update class values."""
        return self.stddev + value

    def __sub__(self, value):
        """Subtract a value from the stddev, does not update class values."""
        return self.stddev - value

    def __str__(self):
        """Return the stddev as a string."""
        return str(self.stddev)

    def val(self):
        """Return the current stddev."""
        return self.stddev


class MMeanArray(object):
    """Class to hold an array of independent means that update in constant-time and memory.

    Note
    ----
        Value shape must be the same or broadcastable to the shape of the
        mean array for all operations.
    """

    def __init__(self, shape, dtype=np.float32):
        self.mean = np.zeros(shape, dtype=dtype)
        self.count = 0
        self.shape = shape

    def __iadd__(self, value):
        """Update the mean, including the given value."""
        self.count += 1
        self.mean += (value - self.mean) * (1.0 / self.count)
        return self

    def __add__(self, value):
        """Add a new value to the mean, does not update class values."""
        return self.mean + value

    def __sub__(self, value):
        """Subtract a new value from the mean, does not update class values."""
        return self.mean - value

    def __str__(self):
        """Return the mean as a string."""
        return str(self.mean)

    def val(self):
        """Return the mean."""
        return self.mean


class MStddevArray(object):
    """Class to hold an array of independent standard deviations that update in constant-time and memory.

    Note
    ----
        Value shape must be the same or broadcastable to the shape of the
        mean array for all operations.
    """

    def __init__(self, shape, dtype=np.float32):
        self.mean = np.zeros(shape, dtype=dtype)
        self.stddev = np.zeros(shape, dtype=dtype)
        self.count = 0
        self.shape = shape

    def __iadd__(self, value):
        """Update the mean and stddev, including the new value."""
        self.count += 1
        prev_mean = np.copy(self.mean)
        self.mean += (value - self.mean) * (1.0 / self.count)
        self.stddev += (value - self.mean) * (value - prev_mean)
        return self

    def __add__(self, value):
        """Add a value to the stddev, does not update class values."""
        return self.stddev + value

    def __sub__(self, value):
        """Subtract a value from the stddev, does not update class values."""
        return self.stddev - value

    def __str__(self):
        """Return the stddev as a string."""
        return str(self.stddev)

    def val(self):
        """Return the current stddev."""
        return self.stddev
