"""Binary confusion matrix class"""
import warnings

import colorama
import numpy as np

from .data_methods import num_digits

__author__ = "Sean Mullan"
__copyright__ = "Sean Mullan"
__license__ = "mit"


def underline(string):
    """Shortcut to underline ANSI text"""
    return '\033[4m' + string + '\033[0m'


class BinaryConfusionMatrix():
    """
    2D array to represent and evaluate a 2-class confusion matrix.

    Parameters
    ----------
    predictions : type
        Optional initial predictions for the matrix. (the default is None)
    labels : type
        Optional initial lables for the matrix. (the default is None)
    threshold : float
        Threshold to use for binary labels with continuous predictions (the default is 0.5)
    dtype : numpy.dtype
        Numpy variable type to use for the matrix. (the default is np.int)

    Note
    ----
    * Rows represent expected class and columns represent predicted class
    * Matrix is 0-indexed
    * Dtype may be set to change memory usage, but will always be treated as an int. No checking is done to prevent overflow if dtype is manually set.

    """
    def __init__(self, predictions=None, labels=None, threshold=0.5, dtype=np.int, pos_label='True', neg_label='False'):
        self.threshold = threshold
        self.reset(dtype)
        if predictions is not None:
            self.add(predictions, labels)
        self.pos_label = pos_label
        self.neg_label = neg_label

    @property
    def dtype(self):
        """The datatype of the values stored in the confusion matrix

        :getter: Return the datatype
        :setter: Re-cast the data in the matrix to a new type
        :type: numpy.dtype
        """
        return self.__matrix.dtype

    @dtype.setter
    def dtype(self, dtype):
        self.__matrix = self.__matrix.astype(dtype)

    @property
    def false_negative(self):
        """The count of incorrectly predicted negative values"""
        return self.__matrix[1, 0]

    @property
    def false_positive(self):
        """The count of incorrectly predicted positive values"""
        return self.__matrix[0, 1]

    @property
    def matrix(self):
        return self.__matrix

    @property
    def true_negative(self):
        """The count of correctly predicted negative values"""
        return self.__matrix[0, 0]

    @property
    def true_positive(self):
        """The count of correctly predicted positive values"""
        return self.__matrix[1, 1]

    def __array__(self, dtype=None):
        if dtype is None:
            return self.__matrix
        else:
            return self.__matrix.astype(dtype)

    def __add__(self, data):
        """Create a new matrix that is the sum of the current one and a new one"""
        new_matrix = self.copy()
        if isinstance(data, BinaryConfusionMatrix):
            new_matrix.add_matrix(data)
        else:
            new_matrix.add(data)
        return new_matrix

    def __getitem__(self, key):
        """Get items from the matrix - Indexing is [expected_class, predicted_class]"""
        return self.__matrix[key]

    def __iadd__(self, data):
        """Add to the current matrix"""
        if isinstance(data, BinaryConfusionMatrix):
            self.add_matrix(data)
        else:
            self.add(data)
        return self

    def __setitem__(self, index, value):
        """Manually set a matrix value"""
        self.__matrix[index] = value

    def __str__(self):
        return str(self.__matrix)

    def __repr__(self):
        digits = max([num_digits(item) for item in np.nditer(self.__matrix)])
        row_str = '{:' + str(digits) + 'd}, {:' + str(digits) + 'd}'
        spacer = " " * 22
        format_string = "BinaryConfusionMatrix([" + row_str + "]\n" + spacer + "[" + row_str + "])"
        return format_string.format(*self.__matrix.flatten())

    def accuracy(self):
        """The accuracy of the samples"""
        if self.__matrix.sum() == 0:
            return 0
        return (self.__matrix[0, 0] + self.__matrix[1, 1]) / self.__matrix.sum()

    def add(self, predictions, labels=None):
        """Add a set of predictions and labels to the matrix

        Parameters
        ----------
        predictions : one or two list/numpy.ndarrays
            Either a list/array of predictions, or a list of length 2 where the first item is predictions and the second item is labels
        labels : list/numpy.ndarray or None
            The ground truth labels that match the given predictions (the default is None)
        """
        if labels is None and isinstance(predictions, list):
            # Predictions is actually [predictions, labels]
            if len(predictions) == 2:
                predictions, labels = predictions
            else:
                raise ValueError('If passing a single list, it must be [predictions, labels]')
        elif labels is None:
            predictions = np.array(predictions)
            if 2 not in predictions.shape:
                raise ValueError("At least one dimension must be a stack of prediction/label values".format(predictions.shape))
            axis_to_split = np.where(np.array(predictions.shape) == 2)[0]
            if len(axis_to_split) > 1:
                warnings.warn("Multiple dimensions have size 2. Splitting the first one into prediction/label arrays.", UserWarning)
            predictions, labels = np.split(predictions, 2, axis=axis_to_split[0])
        labels = np.array(labels)
        predictions = np.array(predictions)
        if predictions.max() > 1 or predictions.min() < 0 or labels.max() > 1 or labels.min() < 0:
            raise ValueError("All values must be between 0 and 1")
        labels = np.round(labels).astype(np.int)
        predictions = (predictions >= self.threshold).astype(np.int)
        if labels.shape != predictions.shape:
            raise ValueError("Predictions and labels must have the same length/shape")
        labels = np.reshape(labels, [-1])
        predictions = np.reshape(predictions, [-1])
        # merged = np.stack([labels, predictions])
        # points, counts = np.unique(merged, axis=1, return_counts=True)
        # for i in range(points.shape[1]):
        #     self.__matrix[points[0, i], points[1, i]] += counts[i]
        tp = np.logical_and(labels, predictions).sum()
        tn = np.logical_and(1 - labels, 1 - predictions).sum()
        fp = predictions.sum() - tp
        fn = (1 - predictions).sum() - tn
        self.__matrix += np.array([[tn, fp], [fn, tp]])

    def add_matrix(self, binary_matrix):
        """Add another matrix or 2x2 array to the current matrix"""
        if isinstance(binary_matrix, BinaryConfusionMatrix):
            self.__matrix += binary_matrix.matrix
        elif isinstance(binary_matrix, np.ndarray):
            self.__matrix += binary_matrix
        else:
            raise ValueError("Unknown matrix type {} cannot be added as a matrix".format(type(binary_matrix)))

    def copy(self):
        """Create a copy of the current matrix"""
        new_mat = BinaryConfusionMatrix(threshold=self.threshold, dtype=self.dtype)
        new_mat[0, 0] = self.__matrix[0, 0]
        new_mat[0, 1] = self.__matrix[0, 1]
        new_mat[1, 0] = self.__matrix[1, 0]
        new_mat[1, 1] = self.__matrix[1, 1]
        return new_mat

    def count(self):
        """Return the number of items currently in the matrix"""
        return self.__matrix.sum()

    def mcc(self):
        """Return the Matthews correlation coefficient"""
        tp = self.__matrix[1, 1]
        tn = self.__matrix[0, 0]
        fp = self.__matrix[0, 1]
        fn = self.__matrix[1, 0]
        with warnings.catch_warnings(record=False):
            warnings.filterwarnings('error')
            try:
                top = ((tp * tn) - (fp * fn))
                bottom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
                if bottom == 0:
                    return 0
                result = top / np.sqrt(bottom)
            except RuntimeWarning:
                # Defaults to this in case of overflow issues with large matrices
                n = tn + tp + fn + fp + 0.0
                s = (tp + fn) / n
                p = (tp + fp) / n
                top = (tp / n) - (s * p)
                bottom = p * s * (1 - s) * (1 - p)
                if bottom == 0:
                    return 0
                result = top / np.sqrt(bottom)
        return result

    def precision(self):
        """Return the precision for the true or 1 class [TP / (TP + FP)]"""
        if self.__matrix[:, 1].sum() == 0:
            return 0
        return self.__matrix[1, 1] / self.__matrix[:, 1].sum()

    def sensitivity(self):
        """Return the sensitivity for the true or 1 class [TP / (TP + FN)]"""
        if self.__matrix[1].sum() == 0:
            return 0
        return self.__matrix[1, 1] / self.__matrix[1].sum()

    def specificity(self):
        """Return the specificity for the true or 1 class [TN / (TN + FP)]"""
        if self.__matrix[0].sum() == 0:
            return 0
        return self.__matrix[0, 0] / self.__matrix[0].sum()

    def zero_rule(self):
        """Return the accuracy for guessing the majority class"""
        if self.__matrix.sum() == 0:
            return 0
        return np.max(self.__matrix.sum(axis=1) / self.__matrix.sum())

    def reset(self, dtype=None):
        """Reset all matrix entries to 0"""
        if dtype is None:
            dtype = self.__matrix.dtype
        self.__matrix = np.zeros([2, 2], dtype=dtype)

    def save(self, path, title='BinaryConfusionMatrix'):
        """Save the current binary confusion matrix to a text file"""
        with open(path, 'w') as f:
            f.write(title + '\n')
            f.write('Threshold: {}\n'.format(self.threshold))
            f.write('Datatype: {}\n'.format(str(self.dtype)))
            f.write(self.print(return_string=True))

    @staticmethod
    def load(path):
        """Load a binary confusion matrix that was saved as a text file."""
        with open(path, 'r') as f:
            title = f.readline().strip()
            threshold = float(f.readline()[11:].strip())
            datatype = np.dtype(f.readline()[10:].strip())
            f.readline()
            next_line = f.readline()
            _, neg_label, pos_label = next_line.split('|')
            neg_label = neg_label.strip()
            pos_label = pos_label.strip()
            next_line = f.readline().split('|')
            tn = int(next_line[1].strip())
            fp = int(next_line[2].strip())
            next_line = f.readline().split('|')
            fn = int(next_line[1].strip())
            tp = int(next_line[2].strip())
        new_matrix = BinaryConfusionMatrix(threshold=threshold, pos_label=pos_label, neg_label=neg_label, dtype=datatype)
        new_matrix.add_matrix(np.array([[tn, fp], [fn, tp]]))
        return new_matrix

    def print(self, show_specificity=False, show_sensitivity=False, show_accuracy=False, as_label=True, pos_label=None, neg_label=None, return_string=False):
        """Format and print the confusion matrix

        Parameters
        ----------
        show_specificity : bool
            Whether to include specificities at end of columns (the default is False)
        show_sensitivity : bool
            Whether to include sensitivities at end of rows (the default is False)
        show_accuracy : bool
            Whether to include accuracies at the end of rows (the default is False)
        as_label : bool
            Whether to print class labels instead of 0/1 (the default is True)
        pos_label : str
            The label to print for class 1 when as_label is true (the default is None)
        neg_label : str
            The label to print for class 0 when as_label is true (the default is None)
        return_string : bool
            Whether to return a plain-text version of the matrix (the default is False).

        Note
        ----
        pos_label and neg_label default the class variables if they are None

        Returns
        -------
        str
            Confusion matrix formatted for plain-text printing if return_string is True.

        """
        expected_string = u"\u2193" + " Expected"
        predicted_string = u"\u2192" + "  Predicted"
        leading_space = "            "
        confusion_string = "         "

        if pos_label is None:
            pos_label = self.pos_label
        if neg_label is None:
            neg_label = self.neg_label

        if self.__matrix.max() == 0:
            item_width = 1
        else:
            item_width = np.ceil(np.log10(self.__matrix.max())).astype(np.int)
        if as_label:
            item_width = max(item_width, len(pos_label), len(neg_label))
        item_width = str(item_width)
        if as_label:
            header_string = " " * (int(item_width) + 3) + ('| {:^' + item_width + '} ').format(neg_label) + ('| {:^' + item_width + '} ').format(pos_label)
        else:
            header_string = "        " + ('| {:^' + item_width + 'd} ').format(0) + ('| {:^' + item_width + 'd} ').format(1)
        confusion_string += predicted_string + "\n" + expected_string + "  " + underline(header_string) + '\n'
        if as_label:
            line_string_1 = ("  {:^" + item_width + "} |").format(neg_label)
            line_string_2 = ("  {:^" + item_width + "} |").format(pos_label)
        else:
            line_string_1 = "    0   |"
            line_string_2 = "    1   |"
        # TN
        line_string_1 += colorama.Fore.GREEN + " {:" + item_width + "d} " + colorama.Style.RESET_ALL + "|"
        # FP
        line_string_1 += " {:" + item_width + "d} |"
        line_string_1 = line_string_1.format(*self.__matrix[0]) + '\n'
        # FN
        line_string_2 += " {:" + item_width + "d} |"
        # TP
        line_string_2 += colorama.Fore.GREEN + " {:" + item_width + "d} " + colorama.Style.RESET_ALL + "|"
        line_string_2 = line_string_2.format(*self.__matrix[1])
        line_string_2 = underline(line_string_2) + '\n'

        confusion_string += leading_space + line_string_1 + leading_space + line_string_2

        if show_accuracy:
            confusion_string += "\nAccuracy:    {:.4f}".format(self.accuracy())
        if show_sensitivity:
            confusion_string += '\nSensitivity: {:.4f}'.format(self.sensitivity())
        if show_specificity:
            confusion_string += '\nSpecificity: {:.4f}'.format(self.specificity())

        if return_string:
            for item in [colorama.Fore.GREEN, colorama.Style.RESET_ALL, '\033[4m', '\033[0m']:
                confusion_string = confusion_string.replace(item, '')
            return confusion_string
        else:
            print(confusion_string)
