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


def normalize(data, axis=None):
    """Return data normalized to a z-score along axis or axes indicated."""
    mean = data.mean(axis=axis, keepdims=True)
    stddev = data.std(axis=axis, keepdims=True)
    return np.divide(data - mean, stddev, where=stddev != 0)


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
    data_range = data.max(axis=axis, keepdims=True) - data.min(axis=axis, keepdims=True)
    x = np.divide(data - data.min(axis=axis, keepdims=True), data_range, where=data_range > 0)
    new_range = new_max - new_min
    return (x * new_range) + new_min


def sigmoid(x):
    """Return the sigmoid of the given value/array."""
    return 1.0 / (1.0 + np.exp(-x) + 1e-7)


def softmax(x, axis=None):
    s = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - s)
    div = np.sum(e_x, axis=axis, keepdims=True)
    return np.divide(e_x, div, where=div != 0)


def standardize(data, axis=None):
    """Standardize data to have zero mean and unit variance along axis or axes indicated."""
    stds = data.std(axis=axis, keepdims=True)
    return np.divide(data - data.mean(axis=axis, keepdims=True), stds, where=stds > 0)
