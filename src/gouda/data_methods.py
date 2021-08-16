"""Methods for working with data and numpy arrays"""
import warnings

import numpy as np


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


def factors(x):
    """Return the factors of x.

    Parameters
    ----------
    x : int
        The number to factorize. Must be a non-zero integer

    Returns
    -------
    factors : set
        The set of factors for x
    """
    if x == 0 or x % 1 != 0:
        raise ValueError("Factors can only be found with non-zero integers")
    if x < 0:
        x = np.abs(x)
        warnings.warn("Only positive factors will be returned, but negative numbers have a positive and negative factor for each.", UserWarning)
    factors = set([1, x])
    for i in range(2, int(np.sqrt(x) + 1)):
        if (x / float(i)) == int(x / i):
            factors.add(int(i))
            factors.add(int(x / i))
    return factors


def flip_dict(dict, unique_items=False, force_list_values=False):
    """Swap keys and values in a dictionary

    Parameters
    ----------
    dict: dictionary
        dictionary object to flip
    unique_items: bool
        whether to assume that all items in dict are unique, potential speedup but repeated items will be lost
    force_list_values: bool
        whether to force all items in the result to be lists or to let unique items have unwrapped values. Doesn't apply if unique_items is true.
    """
    if unique_items:
        return {v: k for k, v in dict.items()}
    elif force_list_values:
        new_dict = {}
        for k, v in dict.items():
            if v not in new_dict:
                new_dict[v] = []
            new_dict[v].append(k)
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


def num_digits(x):
    if x == 0:
        return 1
    return int(np.ceil(np.log10(np.abs(x) + 1)))


def prime_factors(x):
    """Return the prime factorization of x.

    Parameters
    ----------
    x : int
        The number to factorize. Must be a non-zero integer

    Returns
    -------
    prime_factors : list
        The list of prime factors. Repeated factors will occur multiple times in the list.
    """
    if x == 0 or x % 1 != 0:
        raise ValueError("Factors can only be found with non-zero integers")
    if x < 0:
        x = np.abs(x)
        warnings.warn("Only positive factors will be returned, but negative numbers have a positive and negative factor for each.", UserWarning)
    factors = [x]
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


def prime_overlap(x, y):
    """Return the prime factors x and y have in common.

    Parameters
    ----------
    x : int
        The first number to factorize
    y: int
        The second number to factorize

    Returns
    -------
    overlap : list
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


def rescale(data, new_min=0, new_max=1, axis=None):
    """Rescales data to have range [new_min, new_max] along axis or axes indicated."""
    data = np.asarray(data)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float)
    data_range = np.max(data, axis=axis, keepdims=True) - np.min(data, axis=axis, keepdims=True)
    x = np.divide(data - np.min(data, axis=axis, keepdims=True), data_range, where=data_range > 0, out=np.zeros_like(data))
    new_range = new_max - new_min
    return (x * new_range) + new_min


def clip(data, output_min=0, output_max=1, input_min=0, input_max=255):
    """Clip an array to a given range, then rescale the clipped array from the input range to the output range.

    Parameters
    ----------
    image : numpy.ndarray
        The data to rescale
    output_min : int | float | np.number
        The minimum value for the output data (the default is 0)
    output_max : int | float | np.number
        The maximum value for the output data (the default is 1)
    input_min : int | float | np.number
        The minimum value for the input data range (the default is 0)
    input_max : int | float | np.number
        The maximum value for the input data range (the default is 255)
    """
    # TODO - Add tests for this
    data = np.clip(data, input_min, input_max)
    scaler = (output_max - output_min) / (input_max - input_min)
    bias = (input_min * output_min) / (input_max - input_min) - (input_min * output_max) / (input_max - input_min) + output_min
    return data * scaler + bias


def sigmoid(x):
    """Return the sigmoid of the given value/array."""
    return 1.0 / (1.0 + np.exp(-x) + 1e-7)


def softmax(x, axis=None):
    """Return the softmax of the array

    Parameters
    ----------
    x : numpy.ndarray
        The data to apply the softmax to
    axis : int | list of ints
        The axis or axes to apply softmax across
    """
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float)
    s = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - s)
    div = np.sum(e_x, axis=axis, keepdims=True)
    return np.divide(e_x, div, where=div != 0, out=np.zeros_like(x))


def normalize(data, axis=None):
    """Return data normalized to have zero mean and unit variance along axis or axes indicated."""
    data = np.asarray(data)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float)
    mean = np.mean(data, axis=axis, keepdims=True)
    stddev = np.std(data, axis=axis, keepdims=True)
    return np.divide(data - mean, stddev, where=stddev != 0, out=np.zeros_like(data))


def roc_curve(label, pred, as_rates=True):
    """Get the ROC curve for the data.

    Parameters
    ----------
    label : numpy.ndarray
        The ground truth values
    pred : numpy.ndarray
        The predicted values
    as_rate : bool
        Whether to return true/false positive rates or scores (the default is True)

    Returns
    -------
    fps : numpy.ndarray
        The false positive rates/scores
    tps : numpy.ndarray
        The true positive rates/scores
    thresh : numpy.ndarray
        The thresholds for each fps/tps
    """
    if not isinstance(label, np.ndarray):
        label = np.array(label)
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    label = np.ravel(label)
    pred = np.ravel(pred)
    desc_score_indices = np.argsort(pred, kind='mergesort')[::-1]
    y_score = pred[desc_score_indices]
    y_true = label[desc_score_indices]

    distinct_idx = np.where(np.diff(y_score))[0]
    thresh_idx = np.concatenate([distinct_idx, [y_true.size - 1]])

    tps = np.cumsum(y_true)
    # expected = np.sum(y_true)

    tps = tps[thresh_idx]
    fps = 1 + thresh_idx - tps
    thresh = y_score[thresh_idx]

    tps = np.concatenate(([0], tps))
    fps = np.concatenate(([0], fps))
    thresh = np.concatenate(([1], thresh))
    if as_rates:
        fpr = fps / fps[-1]
        tpr = tps / tps[-1]
        return fpr, tpr, thresh
    else:
        return fps, tps, thresh


def mcc_curve(label, pred, optimal_only=False):
    """Get the Matthew's Correlation Coefficient for different thresholds

    Parameters
    ----------
    label : numpy.ndarray
        Expected labels for the data samples
    pred : numpy.ndarray
        Predicted labels for the data samples
    optimal_only : bool
        If true, returns only the value and threshold for the greatest MCC value
    """
    fps, tps, thresh = roc_curve(label, pred, as_rates=False)
    return optimal_mcc_from_roc(fps, tps, thresh, optimal_only=optimal_only)


def optimal_mcc_from_roc(fps, tps, thresholds, optimal_only=True):
    """Get the Matthew's Correlation Coefficient for different thresholds

    Parameters
    ----------
    fps : numpy.ndarray
        False positive scores from the roc curve
    tps : numpy.ndarray
        True positive scores from the roc curve
    thresholds : numpy.ndarray
        Thresholds from the roc curve
    optimal_only : bool
        If true, returns only the value and threshold for the greatest MCC value
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


def accuracy_curve(label, pred, return_peak=False):
    """Get the accuracy values for each possible threshold in the predictions.

    Parameters
    ----------
    label : numpy.ndarray
        The true values for each sample in the data.
    pred : numpy.ndarray
        The predicted values for each sample in the data
    return_peak : bool
        Whether to return the peak accuracy and best threshold for the data as well as the curve
    """
    if not isinstance(label, np.ndarray):
        label = np.array(label)
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)

    desc_score_indices = np.argsort(pred, kind='mergesort')[::-1]
    y_score = pred[desc_score_indices]
    y_true = label[desc_score_indices]

    distinct_idx = np.where(np.diff(y_score))[0]
    thresh_idx = np.concatenate([distinct_idx, [y_true.size - 1]])
    thresh = y_score[thresh_idx]

    tps = np.cumsum(y_true)[thresh_idx]
    tns = np.cumsum((1 - y_true)[::-1])[::-1][thresh_idx]
    correct = tps + tns
    acc = correct / label.size
    if return_peak:
        peak = np.argmax(acc)
        return acc, thresh, acc[peak], thresh[peak]
    return acc, thresh


def spec_at_sens(expected, predicted, sensitivities=[0.95]):
    """Get the peak specificity for each sensitivity."""
    if not hasattr(sensitivities, '__iter__'):
        sensitivities = [sensitivities]
    fpr, tpr, thresholds = roc_curve(expected, predicted)
    specs = [np.max((1 - fpr)[tpr >= min_sens]) for min_sens in sensitivities]
    return specs


def get_confusion_stats(label, pred, threshold=0.5):
    """Get the true positive, false positive, true negative, and false negative values for the given data"""
    label = np.squeeze(label)
    pred = np.squeeze(pred)

    label_bool = label.astype(bool)
    pred_bool = pred >= threshold
    true_pos = np.logical_and(label_bool, pred_bool).sum()
    true_neg = np.logical_and(~label_bool, ~pred_bool).sum()
    false_pos = pred_bool.sum() - true_pos
    false_neg = (~pred_bool).sum() - true_neg
    return true_pos, false_pos, true_neg, false_neg


def dice_coef(label, pred, threshold=0.5):
    """Get the Sorenson Dice Coefficient for the given data"""
    tp, fp, tn, fn = get_confusion_stats(label, pred, threshold)
    denom = tp * 2 + fp + fn
    if denom == 0:
        return 0
    return (tp * 2) / denom


def jaccard_coef(label, pred, threshold=0.5):
    """Get the Jaccard Coefficient for the given data"""
    tp, fp, tn, fn = get_confusion_stats(label, pred, threshold)
    denom = tp + fn + fp
    if denom == 0:
        return 0
    return tp / denom


def value_crossing(array, threshold=0, positive_crossing=True, negative_crossing=True, return_indices=False):
    """Get the count of instances where a series crosses a value.

    Parameters
    ----------
    array : np.ndarray
        A sequential array of values
    threshold : int | float
        The value used as a crossing point (the default is 0)
    positive_crossing : bool
        Whether to count when the sequence goes from less than to greater than the threshold value (the default is True)
    negative_crossing : bool
        Whether to count when the sequence goes from greater than to less than the threshold value (the default is True)
    return_indices : bool
        Whether to return the indices of the points immediately before the crossings
    """

    if not isinstance(array, np.ndarray):
        array = np.array(array)
    if return_indices:
        idxs = np.arange(array.size)[array != threshold]
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
        return idxs[np.concatenate([crossing, [False]])]
    return crossing.sum()


def center_of_mass(input_arr):
    """Find the continuous index of the center of mass for the input n-dimensional array"""
    flat_mass = np.reshape(input_arr, [-1, 1])
    total_mass = np.sum(flat_mass)
    if total_mass == 0:
        raise ValueError("Cannot find the center if the total mass is 0")
    grids = np.meshgrid(*[np.arange(axis_length) for axis_length in input_arr.shape], indexing='ij')
    coords = np.stack([np.reshape(grid, [-1]) for grid in grids], axis=-1)

    center_of_mass = np.sum(flat_mass * coords, axis=0) / total_mass
    return center_of_mass
