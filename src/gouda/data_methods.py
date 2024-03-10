"""Methods for working with data and numpy arrays"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import warnings
from typing import Any, Iterable, List, Optional, Tuple, Union

from gouda.general import is_iter
from gouda.typing import FloatArrayType, LabelArrayType, ShapeType


def to_uint8(x: npt.NDArray, allow_rescale: bool = False) -> npt.NDArray[np.uint8]:
    """Convert an image/array to a uint8 type with range [0, 255] based on inferred normalization type.

    Parameters
    ----------
    x : npt.NDArray
        Input array to convert
    allow_rescale : bool, optional
        Whether to allow rescaling values (see NOTES), by default False

    Returns
    -------
    npt.NDArray[np.uint8]
        Converted array

    NOTES
    -----
    input range [0, 255] -> cast to uint8
    input range [-1, 1] -> x * 127.5 + 127.5 -> cast to uint8
    input range [0, 1] -> x * 255 -> cast to uint8
    input range OTHER -> (x - x_min) / (x_max - x_min) -> cast to uint8
    """
    if allow_rescale:
        x = rescale(x, 0, 255)  # rescale to [0, 255] for any input range
    elif x.dtype == np.uint8:
        return x
    elif x.max() > 1 and x.max() <= 255 and x.min() >= 0:  # input range [0, 255]
        pass
    elif x.min() < 0 and x.min() >= -1 and x.max() <= 1:  # input range [-1, 1]
        x = ((x * 127.5) + 127.5)
    elif x.min() >= 0 and x.max() <= 1:  # input range [0, 1]
        x = (x * 255.0)
    else:
        warnings.warn("Cannot determine input range. Rescaling to [0, 1]")
        x = rescale(x, 0, 255)
    return x.astype(np.uint8)


def arr_sample(arr: np.ndarray, rate: float) -> np.ndarray:
    """Return an array linearly sampled from the input array at the given rate.

    Parameters
    ----------
    arr : np.ndarray
        The 1-dimensional array to sample from
    rate : float
        The step size for each sample

    Returns
    -------
    np.ndarray
        The new array of samples

    Examples
    --------
    * [1, 2, 3, 4] and rate 2   -> [1, 3]
    * [1, 2, 3, 4] and rate 0.5 -> [1, 1, 2, 2, 3, 3, 4, 4]

    Raises
    ------
    ValueError
        If the input array is not 1-dimensional
    """
    if arr.ndim != 1:
        raise ValueError("Only 1d arrays can be sampled from.")
    i = 0
    out = []
    while i < arr.shape[0]:
        out.append(arr[np.floor(i).astype(int)])
        i += rate
    return np.array(out)


def factors(x: int) -> set[int]:
    """Returns the factors of x

    Parameters
    ----------
    x : int
        The number to factorize. Must be a non-zero integer

    Returns
    -------
    set[int]
        The set of factors for x

    Raises
    ------
    ValueError
        If x is equal to 0 or is not an integer value
    """
    if x == 0 or x % 1 != 0:
        raise ValueError("Factors can only be found with non-zero integers")
    if x < 0:
        x = np.abs(x)
        warnings.warn("Only positive factors will be returned, but negative numbers have a positive and negative factor for each.", UserWarning)
    factors = set([1, int(x)])
    for i in range(2, int(np.sqrt(x) + 1)):
        if (x / float(i)) == int(x / i):
            factors.add(int(i))
            factors.add(int(x / i))
    return factors


def flip_dict(dict: dict, unique_items: bool = False, force_list_values: bool = False) -> dict:
    """Swap keys and values in a dictionary

    Parameters
    ----------
    dict : dict
        dictionary object to flip
    unique_items : bool, optional
        whether to assume that all items in dict are unique and hashable - potential speedup but repeated items will be lost, by default False
    force_list_values : bool, optional
        whether to force all items in the result to be lists or to let unique items have unwrapped values. Doesn't apply if unique_items is true., by default False

    Returns
    -------
    dict
        The flipped dictionary
    """
    if unique_items:
        return {v: k for k, v in dict.items()}
    elif force_list_values:
        new_dict = {}
        for k, v in dict.items():
            new_dict.setdefault(v, []).append(k)
        return new_dict
    else:
        new_dict = {}
        for k, v in dict.items():
            if v in new_dict:
                if isinstance(new_dict[v], list):
                    new_dict[v].append(k)
                else:
                    new_dict[v] = [new_dict[v], k]
            else:
                new_dict[v] = k
        return new_dict


def num_digits(x: Union[int, float]) -> int:
    """Return the number of integer digits"""
    if x == 0:
        return 1
    return int(np.ceil(np.log10(np.abs(x) + 1)))


def prime_factors(x: int) -> list[int]:
    """Return the prime factorization of x.

    Parameters
    ----------
    x : int
        The number to factorize. Must be a non-zero integer

    Returns
    -------
    list[int]
        The list of prime factors. Repeated factors will occur multiple times in the list.

    Raises
    ------
    ValueError
        If x is 0 or
    """
    if x == 0 or x % 1 != 0:
        raise ValueError("Factors can only be found with non-zero integers")
    if x < 0:
        x = np.abs(x)
        warnings.warn("Only positive factors will be returned, but negative numbers have a positive and negative factor for each.", UserWarning)
    factors = [int(x)]
    prime_factors = []
    while len(factors) > 0:
        check = factors.pop()
        found = False
        for i in range(2, int(np.sqrt(check) + 1)):
            if (check / float(i)) == int(check / i):
                factors.extend([i, int(check / i)])
                found = True
                break
        if not found:
            prime_factors.append(check)
    return sorted(prime_factors)


def prime_overlap(x: int, y: int) -> list[int]:
    """Return the prime factors x and y have in common.

    Parameters
    ----------
    x : int
        The first number to factorize
    y: int
        The second number to factorize

    Returns
    -------
    list[int]
        The list of common factors. Repeated factors are included for the number of common repeats.
    """
    fact_x = prime_factors(x)
    fact_y = prime_factors(y)
    overlap = []
    for i in range(len(fact_x)):  # pragma: no branch
        item = fact_x.pop()
        if item in fact_y:
            overlap.append(item)
            fact_y.remove(item)
        if len(fact_x) == 0 or len(fact_y) == 0:
            break
    return sorted(overlap)


def rescale(data: npt.ArrayLike, output_min: float = 0, output_max: float = 1, input_min: Optional[float] = None, input_max: Optional[float] = None, axis: Optional[ShapeType] = None) -> npt.NDArray[np.floating]:
    """Rescales data to have range [new_min, new_max] along axis or axes indicated

    Parameters
    ----------
    data : npt.ArrayLike
        Input array-like to rescale
    output_min : float, optional
        The minimum output value, by default 0
    output_max : float, optional
        The maximum output value, by default 1
    input_min : float, optional
        The minimum input value, by default None (if None, inferred from data along axis)
    input_max : float, optional
        The maximum input value, by default None (if None, inferred from data along axis)
    axis : Optional[ShapeType], optional
        Axis or axes along which to infer input min/max if needed, by default None

    Returns
    -------
    FloatArrayType
        Rescaled array

    NOTE
    ----
    For flexibility, there is no checking that input_min and input_max are actually the minimum and maximum values in data along axis. If they are not, the output values are rescaled as if they were and may lie outside of [output_min, output_max]. For enforced bounds, use `gouda.data_methods.clip`.
    """
    data = np.asarray(data)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(float)
    min_val = np.min(data, axis=axis, keepdims=True) if input_min is None else np.asarray(input_min)
    max_val = np.max(data, axis=axis, keepdims=True) if input_max is None else np.asarray(input_max)
    data_range = max_val - min_val  # If max_val < min_val, the output will have flipped signs
    x = np.divide(data - min_val, data_range, where=data_range != 0, out=np.zeros_like(data))
    new_range = output_max - output_min
    return (x * new_range) + output_min


def order_normalization(data: npt.ArrayLike, order: int = 2, axis: Optional[ShapeType] = None) -> npt.NDArray[np.floating]:
    """Normalize data by its matrix or vector norm

    Parameters
    ----------
    data : npt.ArrayLike
        Input array-like to normalize
    order : int, optional
        Order of the norm to use (see :func:`numpy.linalg.norm`), by default 2
    axis : Optional[ShapeType], optional
        The axis or axes to compute the norm over (see :func:`numpy.linalg.norm`), by default None

    Returns
    -------
    numpy.typing.NDArray[np.floating]
        The normalized data
    """
    norm = np.linalg.norm(data, order, axis)
    norm = np.atleast_1d(norm)
    norm[norm == 0] = 1
    if axis is None:
        return np.divide(data, norm)
    return np.divide(data, np.expand_dims(norm, axis))


def clip(data: npt.ArrayLike, output_min: float = 0, output_max: float = 1, input_min: float = 0, input_max: float = 255) -> FloatArrayType:
    """Clip an array to a given range, then rescale the clipped array from the input range to the output range.

    Parameters
    ----------
    data : npt.ArrayLike
        The data to rescale
    output_min : float, optional
        The minimum value for the output data, by default 0
    output_max : float, optional
        The maximum value for the output data, by default 1
    input_min : float, optional
        The lower value to clip the input data to, by default 0
    input_max : float, optional
        The upper value to clip the input data to, by default 255

    Returns
    -------
    FloatArrayType
        The rescaled output array
    """
    data = np.clip(data, input_min, input_max)
    input_range = input_max - input_min
    output_range = output_max - output_min
    if input_range == 0:
        return np.zeros_like(data) + output_min
    scaler = output_range / input_range
    bias = -input_min * scaler + output_min
    return np.multiply(data, scaler) + bias


def percentile_rescale(x: npt.ArrayLike, low_percentile: float = 0.5, high_percentile: Optional[float] = None, output_min: float = 0, output_max: float = 1) -> FloatArrayType:
    """Clip an array to given percentiles, then rescale it to an output range

    Parameters
    ----------
    x : npt.ArrayLike
        The data to rescale
    low_percentile : float, optional
        The lower percentile to clip the input to, by default 0.5
    high_percentile : Optional[float], optional
        The upper percentile to clip the input to - uses `100 - low_percentile` if None, by default None
    output_min : float, optional
        The minimum value for the output data, by default 0
    output_max : float, optional
        The maximum value for the output data, by default 1

    Returns
    -------
    FloatArrayType
        The rescaled output array
    """
    x = np.asarray(x)
    if high_percentile is None:
        high_percentile = 100 - low_percentile
    low_percentile, high_percentile = sorted([low_percentile, high_percentile])
    low_val, high_val = np.percentile(x, (low_percentile, high_percentile))
    return clip(x, output_min, output_max, low_val, high_val)


def percentile_normalize(x: npt.ArrayLike, low_percentile: float = 0.5, high_percentile: Optional[float] = None) -> FloatArrayType:
    """Normalize data after clipping to a percentile value

    Parameters
    ----------
    x : npt.ArrayLike
        The data to normalize
    low_percentile : float, optional
        The lower percentile to clip data to, by default 0.5
    high_percentile : Optional[float], optional
        The upper percentile to clip the input to - uses `100 - low_percentile` if None, by default None

    Note
    ----
    A percentile of 0.5 is the value at the bottom 0.5% of the data NOT the value at the bottom 50%.

    Returns
    -------
    npt.NDArray[np.float_]
        The normalized output array
    """
    x = np.asarray(x)
    if high_percentile is None:
        high_percentile = 100 - low_percentile
    low_percentile, high_percentile = sorted([low_percentile, high_percentile])
    low_val, high_val = np.percentile(x, (low_percentile, high_percentile))
    x = np.clip(x, low_val, high_val)
    std_vals = np.std(x)
    return np.divide(x - np.mean(x), std_vals, where=std_vals > 0, out=np.zeros_like(x))


def relu(data: npt.ArrayLike) -> Any:
    """Return the rectified linear - max(data, 0)"""
    return np.maximum(data, 0)


def sigmoid(x: npt.NDArray, epsilon: float = 1e-7) -> Any:
    """Return the sigmoid of the given value/array."""
    return (1.0 + epsilon) / (1.0 + np.exp(-x) + epsilon)


def inv_sigmoid(x: npt.NDArray, epsilon: float = 1e-7) -> Any:
    """Return the inverse of the sigmoid function for the given value/array."""
    if x > 1 or x < 0:
        raise ValueError('Inverse sigmoid input must be in range [0, 1]')
    elif x == 0:
        return -np.inf
    elif x == 1:
        return np.inf
    return np.log(x / ((1 + epsilon) - ((1 + epsilon) * x)))


def softmax(x: npt.ArrayLike, axis: Optional[ShapeType] = None) -> FloatArrayType:
    """Return the softmax of the array

    Parameters
    ----------
    x : npt.ArrayLike
        The data to apply the softmax to
    axis : Optional[ShapeType], optional
        The axis or axes to apply softmax across

    Returns
    -------
    FloatArrayType
        The output array
    """
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(float)
    s = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - s)
    div = np.sum(e_x, axis=axis, keepdims=True)  # type: ignore  - np.sum typing doesn't allow None types for some reason
    return np.divide(e_x, div, where=div != 0, out=np.zeros_like(x))


def normalize(data: npt.ArrayLike, axis: Optional[ShapeType] = None) -> FloatArrayType:
    """Return data normalized to have zero mean and unit variance along axis or axes indicated.

    Parameters
    ----------
    data : npt.ArrayLike
        The data to normalize
    axis : Optional[ShapeType], optional
        The axis or axes to apply normalization across, by default None

    Returns
    -------
    FloatArrayType
        The normalized output array
    """
    data = np.asarray(data)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(float)
    mean = np.mean(data, axis=axis, keepdims=True)
    stddev = np.std(data, axis=axis, keepdims=True)
    return np.divide(data - mean, stddev, where=stddev != 0, out=np.zeros_like(data))


def roc_curve(label: npt.ArrayLike, pred: npt.ArrayLike, as_rates: bool = True) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """Get the ROC curve for the data.

    Parameters
    ----------
    label : npt.ArrayLike
        The ground truth values
    pred : npt.ArrayLike
        The predicted values
    as_rates : bool, optional
        Whether to return true/false positive rates or scores, by default True

    Returns
    -------
    Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]]
        The false positive rates/scores, true positive rates/scores, and the thresholds for each fps/tps
    """
    label = np.asarray(label)
    pred = np.asarray(pred)
    label = np.ravel(label)
    pred = np.ravel(pred)
    desc_score_indices = np.argsort(pred, kind='mergesort')[::-1]
    y_score = pred[desc_score_indices]
    y_true = label[desc_score_indices]

    distinct_idx = np.where(np.diff(y_score))[0]
    thresh_idx = np.concatenate([distinct_idx, np.array([y_true.size - 1])])

    tps = np.cumsum(y_true)
    # expected = np.sum(y_true)

    tps = tps[thresh_idx]
    fps = 1 + thresh_idx - tps
    thresh = y_score[thresh_idx]

    tps = np.concatenate((np.array([0]), tps))
    fps = np.concatenate((np.array([0]), fps))
    thresh = np.concatenate((np.array([1]), thresh))
    if as_rates:
        fpr = fps / fps[-1]
        tpr = tps / tps[-1]
        return fpr, tpr, thresh
    else:
        return fps, tps, thresh


def mcc_curve(label: npt.ArrayLike, pred: npt.ArrayLike, optimal_only: bool = False) -> Tuple[FloatArrayType, FloatArrayType]:
    """Get the Matthew's Correlation Coefficient for different thresholds

    Parameters
    ----------
    label : npt.ArrayLike
        Expected labels for the data samples
    pred : npt.ArrayLike
        Predicted labels for the data samples
    optimal_only : bool, optional
        If true, returns only the value and threshold for the greatest MCC value, by default False

    Returns
    -------
    Tuple[FloatArrayType, FloatArrayType]
        Either the optimal MCC and threshold or arrays of all MCCs and thresholds
    """
    fps, tps, thresh = roc_curve(label, pred, as_rates=False)
    return optimal_mcc_from_roc(fps, tps, thresh, optimal_only=optimal_only)


def optimal_mcc_from_roc(fps: npt.NDArray[np.floating], tps: npt.NDArray[np.floating], thresholds: npt.NDArray[np.floating], optimal_only=True) -> Tuple[FloatArrayType, FloatArrayType]:
    """Get the Matthew's Correlation Coefficient for different thresholds

    Parameters
    ----------
    fps : npt.NDArray[np.floating]
        False positive scores from the roc curve
    tps : npt.NDArray[np.floating]
        True positive scores from the roc curve
    thresholds : npt.NDArray[np.floating]
        Thresholds from the roc curve
    optimal_only : bool, optional
        If true, returns only the value and threshold for the greatest MCC value

    Returns
    -------
    Tuple[FloatArrayType, FloatArrayType]
        Either the optimal MCC and threshold or arrays of all MCCs and thresholds
    """
    N = tps[-1] + fps[-1]
    S = tps[-1] / N
    P = (fps + tps) / N
    top = (tps / N) - (S * P)
    bottom = np.sqrt(P * S * (1 - S) * (1 - P))
    mcc = np.divide(top, bottom, out=np.zeros_like(top), where=bottom != 0)
    if optimal_only:
        best = np.argmax(mcc)
        return mcc[best], thresholds[best]
    return mcc, thresholds


def accuracy_curve(label: npt.ArrayLike, pred: npt.ArrayLike, return_peak: bool = False) -> Union[Tuple[FloatArrayType, FloatArrayType, float, float], Tuple[FloatArrayType, FloatArrayType]]:
    """Get the accuracy values for each possible threshold in the predictions.

    Parameters
    ----------
    label : npt.ArrayLike
        The true values for each sample in the data.
    pred : npt.ArrayLike
        The predicted values for each sample in the data
    return_peak : bool, optional
        Whether to return the peak accuracy and best threshold for the data as well as the curve, by default False

    Returns
    -------
    Union[Tuple[FloatArrayType, FloatArrayType, float, float], Tuple[FloatArrayType, FloatArrayType]]
        The accuracy and thresholds - optionally, also the peak accuracy and threshold if return_peak is True
    """
    label = np.asarray(label)
    pred = np.asarray(pred)

    desc_score_indices = np.argsort(pred, kind='mergesort')[::-1]
    y_score = pred[desc_score_indices]
    y_true = label[desc_score_indices]

    distinct_idx = np.where(np.diff(y_score))[0]
    thresh_idx = np.concatenate([distinct_idx, np.array([y_true.size - 1])])
    thresh = y_score[thresh_idx]

    tps = np.cumsum(y_true)[thresh_idx]
    tns = np.cumsum((1 - y_true)[::-1])[::-1][thresh_idx]
    correct = tps + tns
    acc = correct / label.size
    if return_peak:
        peak = np.argmax(acc)
        return acc, thresh, acc[peak], thresh[peak]
    return acc, thresh


def spec_at_sens(label: npt.ArrayLike, pred: npt.ArrayLike, sensitivities: Union[List[float], FloatArrayType] = [0.95]) -> List[float]:
    """Get the peak specificity for each sensitivity.

    Parameters
    ----------
    label : npt.ArrayLike
        The true values for each sample in the data.
    pred : npt.ArrayLike
        The predicted values for each sample in the data
    sensitivities : Union[List[float], FloatArrayType], optional
        The sensitivity/sensitivities to find the specificities for, by default [0.95]

    Returns
    -------
    List[float]
        The list of specificities for the given sensitivities
    """
    if not hasattr(sensitivities, '__iter__'):
        sensitivities = [sensitivities]  # type: ignore - we only reach here if type = float
    fpr, tpr, thresholds = roc_curve(label, pred)
    specs = [np.max((1 - fpr)[tpr >= min_sens]) for min_sens in sensitivities]  # type: ignore - type is iterable by this point
    return specs


def get_confusion_stats(label: LabelArrayType, pred: npt.ArrayLike, threshold: float = 0.5) -> Tuple[int, int, int, int]:
    """Get the true positive, false positive, true negative, and false negative values for the given data

    Parameters
    ----------
    label : LabelArrayType
        The true values for each sample in the data.
    pred : npt.ArrayLike
        The predicted values for each sample in the data
    threshold : float, optional
        The threshold to use to separate Positive/Negative predictions, by default 0.5

    Returns
    -------
    Tuple[int, int, int, int]
        The true positive, false positive, true negative, and false negative counts
    """
    label = np.squeeze(label)
    pred = np.squeeze(pred)

    label_bool = label.astype(bool)
    pred_bool = pred >= threshold
    true_pos = np.logical_and(label_bool, pred_bool).sum()
    true_neg = np.logical_and(~label_bool, ~pred_bool).sum()
    false_pos = pred_bool.sum() - true_pos
    false_neg = (~pred_bool).sum() - true_neg
    return true_pos, false_pos, true_neg, false_neg


def dice_coef(label: LabelArrayType, pred: npt.ArrayLike, threshold: float = 0.5) -> float:
    """Get the Sorenson Dice Coefficient for the given data

    Parameters
    ----------
    label : LabelArrayType
        The true values for each sample in the data.
    pred : npt.ArrayLike
        The predicted values for each sample in the data
    threshold : float, optional
        The threshold to use to separate Positive/Negative predictions, by default 0.5

    Returns
    -------
    float
        The Dice coefficient
    """
    tp, fp, tn, fn = get_confusion_stats(label, pred, threshold)
    denom = tp * 2 + fp + fn
    if denom == 0:
        return 0
    return (tp * 2) / denom


def jaccard_coef(label: LabelArrayType, pred: npt.ArrayLike, threshold=0.5) -> float:
    """Get the Jaccard Coefficient for the given data

    Parameters
    ----------
    label : LabelArrayType
        The true values for each sample in the data.
    pred : npt.ArrayLike
        The predicted values for each sample in the data
    threshold : float, optional
        The threshold to use to separate Positive/Negative predictions, by default 0.5

    Returns
    -------
    float
        The Jaccard coefficient
    """
    tp, fp, tn, fn = get_confusion_stats(label, pred, threshold)
    denom = tp + fn + fp
    if denom == 0:
        return 0
    return tp / denom


def value_crossing(array: npt.NDArray[Any], threshold: float = 0, positive_crossing: bool = True, negative_crossing: bool = True, return_indices: bool = False) -> int:
    """Get the count of instances where a series crosses a value.

    Parameters
    ----------
    array : npt.ArrayLike
        A sequential array of values
    threshold : float, optional
        The value used as a crossing point, by default 0
    positive_crossing : bool, optional
        Whether to count when the sequence goes from less than to greater than the threshold value, by default True
    negative_crossing : bool, optional
        Whether to count when the sequence goes from greater than to less than the threshold value, by default True
    return_indices : bool, optional
        Whether to return the indices of the points immediately before the crossings, by default False

    Returns
    -------
    int
        The number of crossings found

    Raises
    ------
    ValueError
        Either positive_crossing or negative_crossing must be true
    """
    array = np.asarray(array)
    idxs = np.arange(array.size)[array != threshold] if return_indices else []
    array = array[array != threshold]
    pos = array > threshold
    npos = ~pos
    if positive_crossing and negative_crossing:
        crossing = (pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])
    elif negative_crossing:
        crossing = (pos[:-1] & npos[1:])
    elif positive_crossing:
        crossing = (npos[:-1] & pos[1:])
    else:
        raise ValueError('Either positive and/or negative crossings must be used')
    if return_indices:
        return idxs[np.concatenate([crossing, np.array([False])])]
    return crossing.sum()


def center_of_mass(input_arr: npt.ArrayLike) -> npt.NDArray[np.float_]:
    """Find the continuous index of the center of mass for the input n-dimensional array"""
    input_arr = np.asarray(input_arr)
    flat_mass = np.reshape(input_arr, [-1, 1])
    total_mass = np.sum(flat_mass)
    if total_mass == 0:
        raise ValueError("Cannot find the center if the total mass is 0")
    grids = np.meshgrid(*[np.arange(axis_length) for axis_length in input_arr.shape], indexing='ij')
    coords = np.stack([np.reshape(grid, [-1]) for grid in grids], axis=-1)

    center_of_mass = np.sum(flat_mass * coords, axis=0) / total_mass
    return center_of_mass


def max_signal(data: npt.ArrayLike, axis: Optional[ShapeType] = None) -> Any:
    """Return the signed value with the largest absolute value along the given axis

    Parameters
    ----------
    data : npt.ArrayLike
        Input array
    axis : Optional[ShapeType], optional
        The axis to check across, otherwise the result uses the flattened array, by default None

    NOTE
    ----
    If axis is None, the first item with the largest absolute value will be returned. Otherwise, the first value with the largest absolute value will returned, with positive values taking precedence over negative
    """
    data = np.asarray(data)
    if axis is None:
        data = data.ravel()
        return data[np.argmax(np.abs(data), axis=None)]
    else:
        maxes = np.max(data, axis=axis)
        mins = np.min(data, axis=axis)
        return np.where(np.abs(mins) > maxes, mins, maxes)


def argmax_signal(data: npt.ArrayLike, axis: Optional[int] = None) -> Union[Tuple[np.int_, ...], npt.NDArray[np.int_]]:
    """Return the index of the signed value with the largest absolute value along an axis

    Parameters
    ----------
    data : npt.ArrayLike
        Input array
    axis : Optional[int], optional
        The axis to check across, otherwise the result uses the flattened array

    NOTE
    ----
    If axis is None, the index of the first item with the largest absolute value will be returned. Otherwise, the index of the first value with the largest absolute value will be returned, with positive values taking precedence over negative
    """
    data = np.asarray(data)
    if axis is None:
        idx = np.argmax(np.abs(data).ravel(), axis=None)
        return np.unravel_index(idx, data.shape)
    else:
        max_idx = np.argmax(data, axis=axis)
        min_idx = np.argmin(data, axis=axis)
        return np.where(np.abs(data.flat[min_idx]) > data.flat[max_idx], min_idx, max_idx)


def benjamini_hochberg(p_vals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Performs the Benjamini-Hochberg procedure for multiple hypothesis testing.

    Parameters
    ----------
    p_vals : np.ndarray
        An array of p-values to check
    alpha : float, optional
        The baseline significance level, by default 0.05

    Returns
    -------
    np.ndarray
        list of booleans, True if the null hypothesis can be rejected

    Note
    ----
    See `Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing <https://www.jstor.org/stable/2346101>`_ for more information
    Or the Wikipedia page: https://en.wikipedia.org/wiki/False_discovery_rate#Benjamini%E2%80%93Hochberg_procedure
    """
    # TODO - Add unit tests
    p_vals = np.asarray(p_vals)
    rank = np.argsort(p_vals)
    p_vals = p_vals[rank]
    reject = np.zeros(p_vals.size, dtype=bool)
    scalar = alpha / p_vals.size
    for idx in range(p_vals.size):
        if p_vals[idx] <= (idx + 1) * scalar:
            reject[rank[idx]] = True
        else:
            break
    return reject


def segment_line(x1: float, x2: float, y1: float, y2: float, segment_size: float = 0.01, num_segments: Optional[int] = None):
    """Divide a line into smaller segments

    Parameters
    ----------
    x1 : float
        X value of the first point
    x2 : _type_
        X value of the second point
    y1 : _type_
        Y value of the first point
    y2 : _type_
        Y value of the second point
    segment_size : float, optional
        Approximate size of each output segment. If greater than the line length, returns a single segment. By default 0.01
    num_segments : Optional[int], optional
        Exact number of output segments (overwrites `segment_size` if given). If 1 or less, returns a single segment. By default None

    Returns
    -------
    npt.NDArray[np.floating]
        Line segments as an array with form [[[x1, y1], [x1 + step, y1 + step]], ... [[x2 - step, y2 - step], [x2, y2]]]

    Notes
    -----
    `segment_size` is approximate because the number of segments is truncated to the nearest integer. This means that any remainder when dividing the line length by `segment_size` is evenly distributed among the resulting segments, making them slightly larger. Setting `num_segments` allows for exact control over number of result segments.
    """
    if num_segments is None:
        dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        num_segments = int(dist / segment_size)
    if num_segments <= 1:
        return np.asarray([[[x1, y1], [x2, y2]]])

    x = np.linspace(x1, x2, num_segments + 1)
    y = np.linspace(y1, y2, num_segments + 1)
    points = np.asarray([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def line_dist(x: Iterable, y: Iterable) -> float:
    """Find the total distance along a line of points

    Parameters
    ----------
    x : Iterable
        A 1D list/array of x values along the line
    y : Iterable
        A 1D list/array of y values along the line

    Returns
    -------
    float
        Total distance along the line
    """
    if not (is_iter(x) and is_iter(y)):
        raise ValueError('x and y must be iterables of values to plot')
    x = np.squeeze(np.asarray(x))
    y = np.squeeze(np.asarray(y))
    if x.shape != y.shape or x.ndim > 1 or y.ndim > 1:
        raise ValueError('x and y must be 1D iterables of values along a line but found shapes {} and {}'.format(x.shape, y.shape))

    dists = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)
    return np.sum(dists)
