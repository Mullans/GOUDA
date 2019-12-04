# -*- coding: utf-8 -*-
"""Miscellaneous data utilities."""
import copy
import json
import os
import warnings
from pkg_resources import DistributionNotFound, get_distribution

import colorama
import numpy as np

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = 'GOUDA'
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:  # pragma: no cover
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound


def num_digits(x):
    if x == 0:
        return 1
    return int(np.ceil(np.log10(np.abs(x) + 1)))


def arr_sample(arr, rate):
    """Return an array linearly sampled from the input array at the given rate.

    Examples
    --------
    * [1, 2, 3, 4] and rate 2   -> [1, 3]
    * [1, 2, 3, 4] and rate 0.5 -> [1, 1, 2, 2, 3, 3, 4, 4]
    """
    if arr.ndim != 1:
        raise ValueError("Only 1d arrays can be sampled from.")
    i = 0
    out = []
    while i < arr.shape[0]:
        out.append(arr[np.floor(i).astype(np.int)])
        i += rate
    return np.array(out)


def ensure_dir(*paths):
    """Check if a given directory exists, and create it if it doesn't. Multiple directories can be passed as a top-to-bottom path structure."""
    full_path = ''
    for path in paths:
        full_path = os.path.join(full_path, path)
        if os.path.exists(full_path) and os.path.isdir(full_path):
            continue
        elif os.path.exists(full_path):
            raise ValueError("A file without an extension is blocking directory creation at {}".format(full_path))
        else:
            os.mkdir(full_path)
    return full_path


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


def rescale(data, new_min=0, new_max=1, axis=None):
    """Rescales data to have range [new_min, new_max] along axis or axes indicated."""
    data_range = data.max(axis=axis, keepdims=True) - data.min(axis=axis, keepdims=True)
    x = np.divide(data - data.min(axis=axis, keepdims=True), data_range, where=data_range > 0)
    new_range = new_max - new_min
    return (x * new_range) + new_min


def normalize(data, axis=None):
    """Return data normalized to [0, 1] along axis or axes indicated."""
    data_range = data.max(axis=axis, keepdims=True) - data.min(axis=axis, keepdims=True)
    return np.divide(data - data.min(axis=axis, keepdims=True), data_range, where=data_range > 0)


def standardize(data, axis=None):
    """Standardize data to have zero mean and unit variance along axis or axes indicated."""
    stds = data.std(axis=axis, keepdims=True)
    return np.divide(data - data.mean(axis=axis, keepdims=True), stds, where=stds > 0)


# def _unnumpy(data):
#     """Convert numpy arrays to lists for JSON"""
#     if type(data) == list:
#         new_data = []
#         for i in range(len(data)):
#             new_data.append(_unnumpy(data[i]))
#     elif type(data) == dict:
#         new_data = {}
#         for key in data.keys():
#             new_data[key] = _unnumpy(data[key])
#     elif type(data) == np.ndarray:
#         new_data = {"numpy_array": data.tolist(), "dtype": str(data.dtype), "shape": data.shape}
#     else:
#         new_data = copy.copy(data)
#     return new_data
#
#
# def _renumpy(data):
#     """Convert JSON back to numpy arrays"""
#     if type(data) == list:
#         for i in range(len(data)):
#             data[i] = _renumpy(data[i])
#     elif type(data) == dict:
#         if "numpy_array" in data:
#             data = np.array(data['numpy_array']).astype(data['dtype']).reshape(data['shape'])
#         else:
#             for key in data.keys():
#                 data[key] = _renumpy(data[key])
#     return data
#
#
# def save_json(data, filename, numpy=False):
#     """Save a list/dict as a json object.
#
#     NOTE
#     ----
#     This is intended for structures with small numpy arrays (if any). Large arrays such as images may take a while to process and make it hard to read the JSON.
#     """
#     if numpy:
#         data = _unnumpy(data)
#     with open(filename, 'w') as f:
#         json.dump(data, f, indent=2, sort_keys=True)
#
#
# def load_json(filename, numpy=False):
#     """Load a json file as a list/dict."""
#     with open(filename, 'r') as f:
#         data = json.load(f)
#         if numpy:
#             data = _renumpy(data)
#         return data

def save_json(data, filename, embed_arrays=True, compressed=False):
    """Save a list/dict/numpy.ndarray as a JSON file.

    Parameters
    ----------
    data : [list, dict, numpy.ndarray]
        Data to save as a JSON file
    filename : string
        Path to write the JSON to
    embed_arrays : bool [defaults to True]
        Whether to embed any numpy arrays into the JSON as lists with metadata. If false saves them to a separate file with placeholders in the JSON.
    compressed : bool [defaults to False]
        If saving numpy arrays in a separate file, this determines if they are zipped or not.

    NOTE
    ----
    JSON files saved this way can be read with any JSON reader, but will have an extra numpy tag at the end that is used to tell :func:`~gouda.load_json` how to read the arrays back in.
    """
    out_arrays = {}
    used_numpy = [False]
    if embed_arrays and compressed:
        warnings.warn('Cannot compress an array that is embedded in a JSON', UserWarning)
        compressed = False

    def unnumpy(_data):
        if isinstance(_data, list):
            new_data = []
            for i in range(len(_data)):
                new_data.append(unnumpy(_data[i]))
        elif isinstance(_data, dict):
            new_data = {}
            for key in _data.keys():
                new_data[key] = unnumpy(_data[key])
        elif isinstance(_data, np.ndarray):
            used_numpy[0] = True
            if embed_arrays:
                new_data = {"numpy_array": _data.tolist(), "dtype": str(_data.dtype), "shape": _data.shape}
            else:
                new_data = {"numpy_array": 'array_{}'.format(len(out_arrays)), 'dtype': str(_data.dtype), 'shape': _data.shape}
                out_arrays['array_{}'.format(len(out_arrays))] = _data
        else:
            new_data = copy.copy(_data)
        return new_data

    data = unnumpy(data)
    if used_numpy[0]:
        if not isinstance(data, list):
            data = [data]
        if compressed:
            data.append('numpy_zip')
        elif embed_arrays:
            data.append('numpy_embed')
        else:
            data.append('numpy')
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)
    if len(out_arrays) != 0:
        np_filename = filename.rsplit('.', 1)[0]
        if compressed:
            np.savez_compressed(np_filename + '_arrayzip.npz', **out_arrays)
        else:
            np.savez(np_filename + '_array.npz', **out_arrays)


def load_json(filename):
    """Load a JSON file, and re-form any numpy arrays if :func:`~gouda.save_json` was used to write them."""
    with open(filename, 'r') as f:
        data = json.load(f)
    if isinstance(data, list):
        if data[-1] == 'numpy':
            np_filename = filename.rsplit('.', 1)[0] + '_array.npz'
            arrays = np.load(np_filename)
        elif data[-1] == 'numpy_zip':
            np_filename = filename.rsplit('.', 1)[0] + '_arrayzip.npz'
            arrays = np.load(np_filename)
        elif data[-1] == 'numpy_embed':
            pass
        else:
            return data

        def renumpy(_data):
            if isinstance(_data, list):
                for i in range(len(_data)):
                    _data[i] = renumpy(_data[i])
            elif isinstance(_data, dict):
                if 'numpy_array' in _data:
                    if isinstance(_data['numpy_array'], list):
                        new_data = np.array(_data['numpy_array'], dtype=_data['dtype']).reshape(_data['shape'])
                    else:
                        new_data = arrays[_data['numpy_array']]
                        if new_data.dtype != _data['dtype'] or list(new_data.shape) != _data['shape']:
                            raise ValueError("Numpy array file doesn't match expected stored numpy array data")
                    return new_data
                else:
                    for key in _data.keys():
                        _data[key] = renumpy(_data[key])
            return _data

        data = data[:-1]
        if len(data) == 1:
            data = data[0]
        data = renumpy(data)
    return data


# Confusion Matrix utils
def underline(string):
    """Underline a string with ANSI escape characters"""
    return '\033[4m' + string + '\033[0m'


def get_specificities(confusion_matrix):
    """Return the specificity for each represented class in a 2D array as from :func:`~gouda.get_confusion_matrix`"""
    if confusion_matrix.ndim != 2:
        raise ValueError("Confusion matrix must be a 2D array")
    if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError("Confusion matrix must have the same height and width")
    tn = np.array([sum([confusion_matrix[j, :i].sum() + confusion_matrix[j, i + 1:].sum() for j in range(confusion_matrix.shape[0]) if j != i]) for i in range(confusion_matrix.shape[0])])
    fp = np.array([confusion_matrix[i, :].sum() - confusion_matrix[i, i].sum() for i in range(confusion_matrix.shape[0])])
    return tn / (tn + fp)


def get_sensitivities(confusion_matrix):
    """Return the sensitivity for each represented class in a 2D array as from :func:`~gouda.get_confusion_matrix`"""
    if confusion_matrix.ndim != 2:
        raise ValueError("Confusion matrix must be a 2D array")
    if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError("Confusion matrix must have the same height and width")
    return [confusion_matrix[i, i] / confusion_matrix[i, :].sum() if confusion_matrix[i, :].sum() > 0 else 0 for i in range(confusion_matrix.shape[0])]


def get_accuracy(confusion_matrix):
    """Return the accuracy from a 2D array as from :func:`~gouda.get_confusion_matrix`"""
    if confusion_matrix.ndim != 2:
        raise ValueError("Confusion matrix must be a 2D array")
    if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError("Confusion matrix must have the same height and width")
    return np.sum([confusion_matrix[i, i] for i in range(confusion_matrix.shape[0])]) / np.sum(confusion_matrix)


def get_binary_confusion_matrix(predictions, labels, threshold=0.5):
    """Get 2D array like a confusion matrix for 2-class predictions.

    Note
    ----
    * Predictions can be either boolean values or continuous probabilities
    * Rows represent expected class and columns represent predicted class
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    if predictions.ndim != 1 or labels.ndim != 1:
        raise ValueError("Predictions and labels must be lists or 1-dimensional arrays")
    if predictions.shape[0] != labels.shape[0]:
        raise ValueError("There must be an equal number of predictions and labels")
    if predictions.dtype != np.bool:
        rounded_pred = predictions > threshold
    else:
        rounded_pred = predictions
    output = np.zeros((2, 2), dtype=np.int)
    for i in range(labels.shape[0]):
        output[int(labels[i]), int(rounded_pred[i])] += 1
    return output


def get_confusion_matrix(predictions, labels, num_classes=None):
    """Get 2D array like a confusion matrix for multi-class predictions.

    Note
    ----
    * Predictions and labels will be treated as integer class labels and floats will be truncated.
    * Matrix is 0-indexed
    * Rows represent expected class and columns represent predicted class
    """
    predictions = np.array(predictions).astype(np.int)
    labels = np.array(labels).astype(np.int)
    if predictions.ndim != 1 or labels.ndim != 1:
        raise ValueError("Predictions and labels must be lists or 1-dimensional arrays")
    if predictions.shape[0] != labels.shape[0]:
        raise ValueError("There must be an equal number of predictions and labels")
    if num_classes is None:
        num_classes = labels.max() + 1
    confusion = np.zeros((num_classes, num_classes))
    for i in range(predictions.shape[0]):
        confusion[labels[i], predictions[i]] += 1
    return confusion


def print_confusion_matrix(confusion_matrix):
    """Format and print a 2D array like a confusion matrix as from :func:`~gouda.get_confusion_matrix`"""
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

    Note
    ----
    * Rows represent expected class and columns represent predicted class
    * Threshold only used for binary class probabilities
    * Matrix is 0-indexed
    * Dtype may be set to change memory usage, but will always be treated as an int. No checking is done to prevent overflow if dtype is manually set.

    """
    def __init__(self, predictions=None, labels=None, threshold=None, num_classes=None, dtype=np.int):

        self.matrix = None
        self._num_classes = 0
        self.threshold = threshold
        self.add_warned = False
        if num_classes is not None:
            self.reset(num_classes, dtype=dtype)
        elif predictions is None and labels is None:
            self.reset(2, dtype=dtype)
        if predictions is not None and labels is not None:
            self.add(predictions, labels, threshold=threshold)
            if self.matrix.dtype != dtype:
                self.matrix = self.matrix.astype(dtype)

    @property
    def shape(self):
        """The shape of the confusion matrix"""
        return self.matrix.shape

    @property
    def size(self):
        """The size of the confusion matrix"""
        return self.matrix.size

    @property
    def dtype(self):
        """The datatype of the values stored in the confusion matrix

        :getter: Return the datatype
        :setter: Re-cast the data in the matrix to a new type
        :type: numpy.dtype
        """
        return self.matrix.dtype

    @dtype.setter
    def dtype(self, dtype):
        self.matrix = self.matrix.astype(dtype)

    @property
    def num_classes(self):
        """Number of classes represented in the confusion matrix"""
        return self._num_classes

    def reset(self, num_classes=None, dtype=None):
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
        if dtype is None:
            dtype = self.matrix.dtype
        self._num_classes = num_classes
        self.matrix = np.zeros((self._num_classes, self._num_classes), dtype=dtype)

    def __iadd__(self, data):
        """Add single datapoint (predicted, expected)"""
        self.add(data[0], data[1])
        return self

    def __add__(self, matrix):
        """Add two matrices together.

        NOTE: Output dtype defaults to first matrix type."""
        incoming_matrix = np.copy(matrix.matrix)
        if self.matrix.dtype != matrix.dtype:
            warnings.warn("Second matrix converted from {} to {} in order to match first matrix.".format(matrix.dtype, self.matrix.dtype), UserWarning)
            incoming_matrix = incoming_matrix.astype(self.matrix.dtype)
        output_size = max(self._num_classes, matrix.num_classes)
        output = np.zeros((output_size, output_size), dtype=self.dtype)
        output[:self._num_classes, :self._num_classes] += self.matrix
        output[:matrix.num_classes, :matrix.num_classes] += incoming_matrix
        output_mat = ConfusionMatrix(num_classes=output_size, dtype=self.matrix.dtype)
        output_mat.matrix = output
        return output_mat

    def __str__(self):
        return str(self.matrix)

    def count(self):
        """Count the number of items in the matrix"""
        return self.matrix.sum()

    def accuracy(self):
        """Get the total accuracy in the matrix"""
        return np.sum([self.matrix[i, i] for i in range(self._num_classes)]) / np.sum(self.matrix) if np.sum(self.matrix) > 0 else 0

    def specificity(self, class_index=None):
        """Return the specificity of all classes or a single class.

        NOTE
        ----
        specificity = (true negative) / (true negative + false positive) for each class.
        """
        if class_index is None:
            tn = np.array([sum([self.matrix[j, :i].sum() + self.matrix[j, i + 1:].sum() for j in range(self._num_classes) if j != i]) for i in range(self._num_classes)])
            fp = np.array([self.matrix[i, :].sum() - self.matrix[i, i].sum() for i in range(self._num_classes)])
            return np.divide(tn, tn + fp, where=(tn + fp) > 0)

        else:
            tn = sum([self.matrix[j, :class_index].sum() + self.matrix[j, class_index + 1:].sum() for j in range(self._num_classes) if j != class_index])
            fp = self.matrix[class_index, :].sum() - self.matrix[class_index, class_index].sum()
            return np.divide(tn, tn + fp, where=(tn + fp) > 0)

    def sensitivity(self, class_index=None):
        """Return the sensitivity of all classes or a single class. AKA recall

        NOTE
        ----
        sensitivity = (true positive) / (true positive + false negative) for each class.
        """
        if class_index is None:
            return [self.matrix[i, i] / self.matrix[i, :].sum() if self.matrix[i, :].sum() > 0 else 0 for i in range(self._num_classes)]
        else:
            return self.matrix[class_index, class_index] / self.matrix[class_index, :].sum() if self.matrix[class_index, :].sum() > 0 else 0

    def precision(self, class_index=None):
        """Return the precision of all classes or a single class.

        NOTE
        ----
        precision = (true positive) / (true positive + false positive)
        """
        if class_index is None:
            return [self.matrix[i, i] / self.matrix[:, i].sum() if self.matrix[:, i].sum() > 0 else 0 for i in range(self._num_classes)]
        else:
            return self.matrix[class_index, class_index] / self.matrix[:, class_index].sum() if self.matrix[class_index, :].sum() > 0 else 0

    def mcc(self):
        """Return the Matthews correlation coefficient of a binary confusion matrix.

        NOTE
        ----
        mcc = ((tp * tn) - (fp * fn)) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        """
        if self._num_classes != 2:
            raise ValueError("Matthews correlation coefficient only applies to binary classifications")
        tp = self.matrix[1, 1]
        tn = sum([self.matrix[j, j] for j in range(self._num_classes) if j != 1])
        fp = sum([self.matrix[j, 1] for j in range(self._num_classes) if j != 1])
        fn = sum([self.matrix[:, j].sum() - self.matrix[j, j] for j in range(self._num_classes) if j != 1])
        return ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    @staticmethod
    def from_array(predicted, expected, threshold=None):
        mat = ConfusionMatrix()
        mat.add_array(predicted, expected, threshold=threshold)
        return mat

    def add_array(self, predicted, expected, threshold=None):
        """Add data to the confusion matrix as numpy arrays
        Parameters
        ----------
        predicted : numpy.ndarray
            Predicted values to add to the matrix either in same shape as expected or with shape [samples, classes] for probabilities
        expected : numpy.ndarray
            Expected values to add to the matrix
        threshold : type
            Threshold to use for predicted probabilities of binary classes. Defaults to self.threshold

        """
        if threshold is None:
            threshold = self.threshold
        if not isinstance(predicted, np.ndarray):
            raise ValueError("predicted and expected must be arrays, not {}".format(type(predicted)))
        if not isinstance(expected, np.ndarray):
            raise ValueError("predicted and expected must be arrays, not {}".format(type(expected)))
        if 'float' in predicted.dtype.name:
            if predicted.ndim == 2:
                # Assumes predicted samples as [samples, classes]
                predicted = np.argmax(predicted, axis=1)
            else:
                if threshold is None:
                    predicted = np.round(predicted)
                else:
                    predicted = predicted > threshold
        if 'float' in expected.dtype.name:
            warnings.warn("Float type labels will be automatically rounded to the nearest integer", UserWarning)
            expected = np.round(expected).astype(np.int)
        if not ('int' in expected.dtype.name or 'bool' in expected.dtype.name):
            raise ValueError("Expected must be either an int or a bool, not {}".format(expected.dtype))
        max_in = max(expected.max(), predicted.max()) + 1
        if self.matrix is None:
            self.reset(max_in, dtype=expected.dtype)
        if self._num_classes < max_in:
            new_matrix = np.zeros((max_in, max_in), dtype=self.dtype)
            new_matrix[:self._num_classes, :self._num_classes] += self.matrix
            self.matrix = new_matrix
            self._num_classes = max_in
        expected = expected.astype(self.dtype).flatten()
        predicted = predicted.astype(self.dtype).flatten()
        if expected.shape != predicted.shape:
            raise ValueError("Expected and predicted must have same shape")
        merged = np.stack([expected, predicted])
        points, counts = np.unique(merged, axis=1, return_counts=True)
        for i in range(points.shape[1]):
            self.matrix[points[0, i], points[1, i]] += counts[i]

    def add(self, predicted, expected, threshold=None):
        """Add data to the confusion Matrix

        Parameters
        ----------
        predicted : [bool, float, int, list]
            Predicted value(s) to add to the matrix
        expected : [bool, float, int, list]
            Expected value(s) to add to matrix
        threshold : type
            Threshold used for predicted probabilities of binary classes. Defaults to self.threshold

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
        elif isinstance(predicted, (bool, np.bool_)):
            predicted_class = 1 if predicted else 0
        elif isinstance(predicted, (int, np.int_)):
            # Single value: class label
            predicted_class = predicted
        elif isinstance(expected, (float, int, bool, np.bool_)) and isinstance(predicted, (list, np.ndarray)):
            # Class probabilities with single expected label
            predicted_class = np.argmax(predicted).astype(int)
        elif isinstance(predicted, (list, np.ndarray)) and isinstance(expected, (list, np.ndarray))and len(predicted) == len(expected):
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
            self.reset(max_in, dtype=np.array(expected).dtype)
        if self._num_classes < max_in:
            new_matrix = np.zeros((max_in, max_in), dtype=self.dtype)
            new_matrix[:self._num_classes, :self._num_classes] += self.matrix
            self.matrix = new_matrix
            self._num_classes = max_in
        self.matrix[expected_class, predicted_class] += 1

    def print(self, show_specificities=True, show_sensitivities=True, show_accuracy=True, return_string=False):
        """Format and print the confusion matrix

        Parameters
        ----------
        show_specificities : bool
            Whether to include specificities at end of columns
        show_sensitivities : bool
            Whether to include sensitivities at end of rows
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
        item_width = str(np.ceil(np.log10(self.matrix.max())).astype(np.int))
        header_string = "        " + "".join([('| {:^' + item_width + 'd} ').format(i) for i in range(self._num_classes)])
        if show_sensitivities:
            header_string += "| Sensitivity"
        confusion_string += predicted_string + "\n" + expected_string + "  " + underline(header_string) + '\n'
        for i in range(self._num_classes):
            line_string = "    {:1d}   |"
            for j in range(self._num_classes):
                if i == j:
                    line_string += colorama.Fore.GREEN + " {:" + item_width + "d} " + colorama.Style.RESET_ALL + "|"
                else:
                    line_string += " {:" + item_width + "d} |"
            if show_sensitivities:
                line_string += " {:.4f}"
                line_string = line_string.format(i, *self.matrix[i], sensitivities[i])
            else:
                line_string = line_string.format(i, *self.matrix[i])
            if i == self.num_classes - 1:
                line_string = underline(line_string)
            confusion_string += leading_space + line_string + '\n'

        if show_specificities:
            specificity_string = '        Specificity '
            for _ in range(self._num_classes):
                specificity_string += '| {:>' + item_width + '.4f} '
            confusion_string += (specificity_string + '\n').format(*specificities)

        if show_accuracy:
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
        self._mean = 0.0
        self._count = 0

    @property
    def count(self):
        """The count of items included in the mean"""
        return self._count

    def __iadd__(self, value):
        """Update the mean, including the given value."""
        self._count += 1
        self._mean += (1.0 / self._count) * (value - self._mean)
        return self

    def __add__(self, value):
        """Add a new value to the mean, does not update class values."""
        return self._mean + value

    def __sub__(self, value):
        """Subtract a new value from the mean, does not update class values."""
        return self._mean - value

    def __str__(self):
        """Return the mean as a string."""
        return str(self._mean)

    def __call__(self):
        """Alternate method for self.mean()"""
        return self._mean

    def mean(self):
        """Return the mean."""
        return self._mean


class MStddev(object):
    """Class to hold a moving standard deviation with constant-time update and memory."""

    def __init__(self):
        self._count = 0.0
        self._mean = 0.0
        self._variance = 0.0

    @property
    def count(self):
        """The count of items included in the standard deviation"""
        return self._count

    def __iadd__(self, value):
        """Update the mean and stddev, including the new value."""
        self._count += 1
        prev_mean = self._mean
        self._mean += (1.0 / self._count) * (value - self._mean)
        self._variance += (value - self._mean) * (value - prev_mean)
        return self

    def __add__(self, value):
        """Add a value to the stddev, does not update class values."""
        return self.stddev() + value

    def __call__(self):
        """Alternate method for self.stddev"""
        return self.stddev()

    def __sub__(self, value):
        """Subtract a value from the stddev, does not update class values."""
        return self.stddev() - value

    def __str__(self):
        """Return the stddev as a string."""
        if self._count == 0:
            return str(0)
        stddev = np.sqrt(self._variance / self._count)
        return str(stddev)

    def mean(self):
        """Return the mean"""
        return self._mean

    def stddev(self):
        """Return the current stddev."""
        if self._count == 0:
            return 0
        return np.sqrt(self._variance / self._count)


class MMeanArray(object):
    """Class to hold an array of element-wise independent means that update in constant-time and memory.

    Note
    ----
        Value shape must be the same or broadcastable to the shape of the
        mean array for all operations.
    """
    def __init__(self, shape, dtype=np.float):
        self._mean = np.zeros(shape, dtype=dtype)
        self._count = 0

    @property
    def shape(self):
        """The shape of the array"""
        return self._mean.shape

    @property
    def dtype(self):
        """The type of data stored in the array"""
        return self._mean.dtype

    @property
    def count(self):
        """The number of examples used for the mean of each item in the array"""
        return self._count

    def __iadd__(self, value):
        """Update the _mean, including the given value."""
        if value.shape != self.shape:
            raise ValueError('Input values must have the same shape as the MMeanArray')
        self._count += 1
        self._mean += (value - self._mean) * (1.0 / self._count)
        return self

    def __add__(self, value):
        """Add a new value to the mean, does not update class values."""
        return self._mean + value

    def __sub__(self, value):
        """Subtract a new value from the mean, does not update class values."""
        return self._mean - value

    def __str__(self):
        """Return the _mean as a string."""
        return str(self._mean)

    def __call__(self):
        """Alternate method for self.mean()"""
        return self._mean

    def mean(self):
        """Return the _mean."""
        return self._mean


class MStddevArray(object):
    """Class to hold an array of element-wise independent standard deviations that update in constant-time and memory.

    Note
    ----
        Value shape must be the same or broadcastable to the shape of the
        mean array for all operations.
    """

    def __init__(self, shape, dtype=np.float):
        self._mean = np.zeros(shape, dtype=dtype)
        self._variance = np.zeros(shape, dtype=dtype)
        self._count = 0

    @property
    def shape(self):
        """The shape of the array"""
        return self._variance.shape

    @property
    def dtype(self):
        """The type of data stored in the array"""
        return self._mean.dtype

    @property
    def count(self):
        """The number of examples used for the standard deviation of each item in the array"""
        return self._count

    def mean(self):
        """Return the mean of the array"""
        return self._mean

    def variance(self):
        """Return the variance of the array"""
        if self._count == 0:
            return np.zeros_like(self._variance)
        return self._variance / self._count

    def __iadd__(self, value):
        """Update the mean and stddev, including the new value."""
        if value.shape != self.shape:
            raise ValueError('Input values must have the same shape as the MStddevArray')
        self._count += 1
        prev_mean = np.copy(self._mean)
        self._mean += (1.0 / self._count) * (value - self._mean)
        self._variance += (value - self._mean) * (value - prev_mean)
        return self

    def __add__(self, value):
        """Add a value to the stddev, does not update class values."""
        return self.stddev() + value

    def __sub__(self, value):
        """Subtract a value from the stddev, does not update class values."""
        return self.stddev() - value

    def __str__(self):
        """Return the stddev as a string."""
        return str(self.stddev())

    def __call__(self):
        """Alternate method for self.stddev()"""
        return self.stddev()

    def stddev(self):
        """Return the current stddev."""
        if self._count == 0:
            return np.zeros_like(self._variance)
        return np.sqrt(self._variance / self._count)
