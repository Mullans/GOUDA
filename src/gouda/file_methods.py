"""General file method and JSON handling"""
import copy
import json
import os
import warnings

import numpy as np

from .goudapath import GoudaPath

__author__ = "Sean Mullan"
__copyright__ = "Sean Mullan"
__license__ = "mit"


def ensure_dir(*paths):
    """Check if a given directory exists, and create it if it doesn't. Multiple directories can be passed as a top-to-bottom path structure.

    Parameters
    ----------
    *paths : iterable
        One or more nested directories to ensure

    Returns
    -------
    str
        Joined filepath of all ensured directories

    """
    full_path = ''
    for path in paths:
        full_path = os.path.join(full_path, path)
        if os.path.exists(full_path) and os.path.isdir(full_path):
            continue
        elif os.path.exists(full_path):
            raise ValueError("A file without an extension is blocking directory creation at {}".format(full_path))
        else:
            os.mkdir(full_path)
    if isinstance(paths[0], GoudaPath):
        return GoudaPath(full_path, use_absolute=paths[0].use_absolute)
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


# JSON methods
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
