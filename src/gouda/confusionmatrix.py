"""Confusion matrix class."""

from __future__ import annotations

import warnings
from typing import Any

import colorama
import numpy as np
import numpy.typing as npt

from gouda.symbols import underline

__author__ = "Sean Mullan"
__copyright__ = "Sean Mullan"
__license__ = "mit"


class ConfusionMatrix:
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

    def __init__(
        self,
        predictions: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        threshold: float | None = None,
        num_classes: int | None = None,
        dtype: npt.DTypeLike = int,
    ) -> None:
        self.matrix: npt.NDArray[np.integer]
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
    def shape(self) -> tuple[int, ...]:
        """The shape of the confusion matrix."""
        shape: tuple[int, ...] = self.matrix.shape
        return shape

    @property
    def size(self) -> int:
        """The size of the confusion matrix."""
        return self.matrix.size

    @property
    def dtype(self) -> npt.DTypeLike:
        """The datatype of the values stored in the confusion matrix.

        :getter: Return the datatype
        :setter: Re-cast the data in the matrix to a new type
        :type: numpy.dtype
        """
        return self.matrix.dtype

    @dtype.setter
    def dtype(self, dtype: npt.DTypeLike) -> None:
        self.matrix = self.matrix.astype(dtype)

    @property
    def num_classes(self) -> int:
        """Number of classes represented in the confusion matrix."""
        return self._num_classes

    def reset(self, num_classes: int | None = None, dtype: npt.DTypeLike | None = None) -> None:
        """Reset all matrix entries.

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

    # ignore - mypy bug if __iadd__ and __add__ have different signatures
    def __iadd__(  # type: ignore[misc]
        self, data: tuple[bool | float | int | npt.ArrayLike, bool | float | int | npt.ArrayLike]
    ) -> ConfusionMatrix:
        """Add single datapoint (predicted, expected)."""
        self.add(data[0], data[1])
        return self

    def __add__(self, matrix: ConfusionMatrix) -> ConfusionMatrix:
        """Add two matrices together.

        NOTE: Output dtype defaults to first matrix type.
        """
        incoming_matrix = np.copy(matrix.matrix)
        if self.matrix.dtype != matrix.dtype:
            warnings.warn(
                f"Second matrix converted from {matrix.dtype} to {self.matrix.dtype} in order to match first matrix.",
                UserWarning,
            )
            incoming_matrix = incoming_matrix.astype(self.matrix.dtype)
        output_size = max(self._num_classes, matrix.num_classes)
        output = np.zeros((output_size, output_size), dtype=self.dtype)
        output[: self._num_classes, : self._num_classes] += self.matrix
        output[: matrix.num_classes, : matrix.num_classes] += incoming_matrix
        output_mat = ConfusionMatrix(num_classes=output_size, dtype=self.matrix.dtype)
        output_mat.matrix = output
        return output_mat

    def __str__(self) -> str:
        """Return a string representation of the confusion matrix."""
        return str(self.matrix)

    def __getitem__(self, key: Any) -> Any:  # noqa: ANN401
        """Access values of the confusion matrix (similar to np.ndarray.__getitem__)."""
        return self.matrix[key]

    def __setitem__(self, key: Any, value: Any) -> None:  # noqa: ANN401
        """Manually set values of the confusion matrix. (NOT RECOMMENDED - USE ADDING/RESET METHODS)."""
        self.matrix[key] = value

    def __len__(self) -> int:
        """Get length of confusion matrix (number of classes)."""
        return self._num_classes

    def count(self) -> int:
        """Count the number of items in the matrix."""
        result: int = self.matrix.sum()
        return result

    def accuracy(self) -> float:
        """Get the total accuracy in the matrix."""
        return (
            np.sum([self.matrix[i, i] for i in range(self._num_classes)]) / np.sum(self.matrix)
            if np.sum(self.matrix) > 0
            else 0
        )

    def specificity(self, class_index: int | None = None) -> npt.NDArray[np.floating]:
        """Return the specificity of all classes or a single class.

        NOTE
        ----
        specificity = (true negative) / (true negative + false positive) for each class.
        """
        result: npt.NDArray[np.floating]
        if class_index is None:
            tn = np.array(
                [
                    sum(
                        [
                            self.matrix[j, :i].sum() + self.matrix[j, i + 1 :].sum()
                            for j in range(self._num_classes)
                            if j != i
                        ]
                    )
                    for i in range(self._num_classes)
                ]
            )
            fp = np.array([self.matrix[i, :].sum() - self.matrix[i, i].sum() for i in range(self._num_classes)])
            result = np.divide(tn, tn + fp, where=(tn + fp) > 0)
            return result

        else:
            tn = sum(
                [
                    self.matrix[j, :class_index].sum() + self.matrix[j, class_index + 1 :].sum()
                    for j in range(self._num_classes)
                    if j != class_index
                ]
            )
            fp = self.matrix[class_index, :].sum() - self.matrix[class_index, class_index].sum()
            result = np.divide(tn, tn + fp, where=(tn + fp) > 0)
            return result

    def sensitivity(self, class_index: int | None = None) -> list[float] | float:
        """Return the sensitivity of all classes or a single class. AKA recall.

        Notes
        -----
        sensitivity = (true positive) / (true positive + false negative) for each class.
        """
        if class_index is None:
            return [
                self.matrix[i, i] / self.matrix[i, :].sum() if self.matrix[i, :].sum() > 0 else 0
                for i in range(self._num_classes)
            ]
        else:
            return (
                self.matrix[class_index, class_index] / self.matrix[class_index, :].sum()
                if self.matrix[class_index, :].sum() > 0
                else 0.0
            )

    def precision(self, class_index: int | None = None) -> list[float] | float:
        """Return the precision of all classes or a single class.

        Notes
        -----
        precision = (true positive) / (true positive + false positive)
        """
        if self.matrix is None:
            raise RuntimeError("Matrix has not been initialized")
        if class_index is None:
            return [
                self.matrix[i, i] / self.matrix[:, i].sum() if self.matrix[:, i].sum() > 0 else 0
                for i in range(self._num_classes)
            ]
        else:
            return (
                self.matrix[class_index, class_index] / self.matrix[:, class_index].sum()
                if self.matrix[class_index, :].sum() > 0
                else 0
            )

    def mcc(self) -> float:
        """Return the Matthews correlation coefficient of a binary confusion matrix.

        Notes
        -----
        mcc = ((tp * tn) - (fp * fn)) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        """
        if self.matrix is None:
            raise RuntimeError("Matrix has not been initialized")
        if self._num_classes != 2:
            raise ValueError("Matthews correlation coefficient only applies to binary classifications")
        tp = self.matrix[1, 1]
        tn = self.matrix[0, 0]
        fp = self.matrix[0, 1]
        fn = self.matrix[1, 0]
        # tn = sum([self.matrix[j, j] for j in range(self._num_classes) if j != 1])
        # fp = sum([self.matrix[j, 1] for j in range(self._num_classes) if j != 1])
        # fn = sum([self.matrix[:, j].sum() - self.matrix[j, j] for j in range(self._num_classes) if j != 1])
        result: float
        with warnings.catch_warnings(record=False):  # TODO - reevaluate the need of this catch
            warnings.filterwarnings("error")
            try:
                result = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            except RuntimeWarning:  # pragma: no cover
                # Defaults to this in case of overflow issues with large matrices
                n = tn + tp + fn + fp + 0.0
                s = (tp + fn) / n
                p = (tp + fp) / n
                top = (tp / n) - (s * p)
                bottom = p * s * (1 - s) * (1 - p)
                result = top / np.sqrt(bottom)
        return result

    def zero_rule(self) -> float:
        """Return the accuracy as if only the most common class is predicted."""
        result: float = float(np.max(self.matrix.sum(axis=1) / self.matrix.sum()))
        return result

    @classmethod
    def from_array(
        cls,
        predicted: npt.NDArray[np.integer | np.floating],
        expected: npt.NDArray[np.integer | np.floating],
        threshold: float | None = None,
    ) -> ConfusionMatrix:
        """Create a confusion matrix from numpy arrays.

        Parameters
        ----------
        predicted : npt.NDArray[np.integer | np.floating]
            Predicted values to add to the matrix either in same shape as expected or with shape [samples, classes] for probabilities
        expected : npt.NDArray[np.integer | np.floating]
            Expected values to add to the matrix
        threshold : float | None
            Threshold to use for predicted probabilities of binary classes. Defaults to self.threshold


        Returns
        -------
        ConfusionMatrix
            The generated confusion matrix
        """
        mat = cls()
        mat.add_array(predicted, expected, threshold=threshold)
        return mat

    def add_array(
        self,
        predicted: npt.NDArray[np.integer | np.floating],
        expected: npt.NDArray[np.integer | np.floating],
        threshold: float | None = None,
    ) -> None:
        """Add data to the confusion matrix as numpy arrays.

        Parameters
        ----------
        predicted : npt.NDArray[np.integer | np.floating]
            Predicted values to add to the matrix either in same shape as expected or with shape [samples, classes] for probabilities
        expected : npt.NDArray[np.integer | np.floating]
            Expected values to add to the matrix
        threshold : float | None
            Threshold to use for predicted probabilities of binary classes. Defaults to self.threshold

        """
        if threshold is None:
            threshold = self.threshold
        if not isinstance(predicted, np.ndarray):
            raise ValueError(f"predicted and expected must be arrays, not {type(predicted)}")
        if not isinstance(expected, np.ndarray):
            raise ValueError(f"predicted and expected must be arrays, not {type(expected)}")
        if "float" in predicted.dtype.name:
            if predicted.ndim == 2 and predicted.shape[1] > 1:
                # Assumes predicted samples as [samples, classes]
                predicted = np.argmax(predicted, axis=1).astype(self.dtype)
            else:
                predicted = (
                    np.round(predicted, decimals=0).astype(self.dtype)
                    if threshold is None
                    else (predicted > threshold).astype(self.dtype)
                )
        if "float" in expected.dtype.name:
            warnings.warn("Float type labels will be automatically rounded to the nearest integer", UserWarning)
            expected = np.round(expected).astype(int)
        if not ("int" in expected.dtype.name or "bool" in expected.dtype.name):
            raise ValueError(f"Expected must be either an int or a bool, not {expected.dtype}")
        max_in: int = max(expected.max(), predicted.max()) + 1
        if self.matrix is None:
            self.reset(max_in, dtype=expected.dtype)
        if self._num_classes < max_in:
            new_matrix = np.zeros((max_in, max_in), dtype=self.dtype)
            new_matrix[: self._num_classes, : self._num_classes] += self.matrix
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

    def add(
        self,
        predicted: bool | float | int | npt.ArrayLike,
        expected: bool | float | int | np.number | npt.ArrayLike,
        threshold: float | None = None,
    ) -> None:
        """Add data to the Confusion Matrix.

        Parameters
        ----------
        predicted : bool | float | int | npt.ArrayLike
            Predicted value(s) to add to the matrix
        expected : bool | float | int | npt.ArrayLike
            Expected value(s) to add to matrix
        threshold : float | None
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
        predicted_class: int
        expected_class: int
        if isinstance(predicted, float | np.floating):
            # Single value: prediction of True (class 1)
            if threshold is not None:
                predicted_class = 1 if predicted > threshold else 0
            else:
                if not self.add_warned:
                    self.add_warned = True
                    print("Warning: Float predicted classes without a threshold are rounded to the nearest integer.")
                predicted_class = int(np.round(predicted))
        elif isinstance(predicted, bool):
            predicted_class = 1 if predicted else 0
        elif isinstance(predicted, int | np.integer):
            # Single value: class label
            predicted_class = int(predicted)
        elif isinstance(expected, float | int | bool | np.integer | np.floating) and isinstance(
            predicted, list | tuple | np.ndarray
        ):
            # Class probabilities with single expected label
            predicted_class = int(np.argmax(predicted))
        elif (
            isinstance(predicted, list | tuple | np.ndarray)
            and isinstance(expected, list | tuple | np.ndarray)
            and len(predicted) == len(expected)
        ):
            # Paired lists
            for x, y in zip(predicted, expected):
                self.add(x, y, threshold=threshold)
            return
        else:
            raise ValueError("Unsupported input format")
        if not isinstance(expected, float | int | bool | np.number):
            print(type(expected))
            raise ValueError("Only 1 expected value per prediction is supported")
        if isinstance(expected, bool):
            expected_class = 1 if expected else 0
        elif float(expected) % 1 == 0:
            expected_class = int(np.round(expected))
        else:
            raise ValueError("Expected values must be class label integers or boolean")

        max_in = max(predicted_class, expected_class) + 1
        if self.matrix is None:
            self.reset(max_in, dtype=np.array(expected).dtype)
        if self._num_classes < max_in:
            new_matrix = np.zeros((max_in, max_in), dtype=self.dtype)
            new_matrix[: self._num_classes, : self._num_classes] += self.matrix
            self.matrix = new_matrix
            self._num_classes = max_in
        self.matrix[expected_class, predicted_class] += 1

    def print(
        self,
        show_specificities: bool = True,
        show_sensitivities: bool = True,
        show_accuracy: bool = True,
        return_string: bool = False,
    ) -> str | None:
        """Format and print the confusion matrix.

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
        if not isinstance(sensitivities, list):
            raise ValueError("Sensitivity must be a list")
        expected_string = "\u2193" + " Expected"
        predicted_string = "\u2192" + "  Predicted"
        leading_space = "            "
        confusion_string = "         "
        item_width = str(np.ceil(np.log10(self.matrix.max())).astype(int))
        header_string = "        " + "".join(
            [("| {:^" + item_width + "d} ").format(i) for i in range(self._num_classes)]
        )
        if show_sensitivities:
            header_string += "| Sensitivity"
        confusion_string += predicted_string + "\n" + expected_string + "  " + underline(header_string) + "\n"
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
            confusion_string += leading_space + line_string + "\n"

        if show_specificities:
            specificity_string = "        Specificity "
            for _ in range(self._num_classes):
                specificity_string += "| {:>" + item_width + ".4f} "
            confusion_string += (specificity_string + "\n").format(*specificities)

        if show_accuracy:
            confusion_string += f"\nAccuracy: {self.accuracy():.4f}"

        print(confusion_string)
        if return_string:
            for item in [colorama.Fore.GREEN, colorama.Style.RESET_ALL, "\033[4m", "\033[0m"]:
                confusion_string = confusion_string.replace(item, "")
            return confusion_string
        else:
            return None
