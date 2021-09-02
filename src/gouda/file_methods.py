"""General file method and JSON handling"""
import copy
import glob
import json
import imghdr
import os
import warnings

import numpy as np

from .data_methods import num_digits

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
        full_path = os.path.join(full_path, str(path))
        if os.path.exists(full_path) and os.path.isdir(full_path):
            continue
        elif os.path.exists(full_path):
            raise ValueError("A file without an extension is blocking directory creation at {}".format(full_path))
        else:
            os.makedirs(full_path, exist_ok=True)
    return full_path


def next_filename(filename):
    """Check if a given file exists, and return a new filename for a numbered copy if it does."""
    filename = str(filename)
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
    if isinstance(data, dict) and 'slice_start' in data:
        data = slice(data['slice_start'], data['slice_stop'], data['slice_step'])
    elif isinstance(data, list):
        if data[-1] == 'numpy':
            np_filename = filename.rsplit('.', 1)[0] + '_array.npz'
            arrays = np.load(np_filename)
            data = data[0]
        elif data[-1] == 'numpy_zip':
            np_filename = filename.rsplit('.', 1)[0] + '_arrayzip.npz'
            arrays = np.load(np_filename)
            data = data[0]
        elif data[-1] == 'numpy_embed':
            data = data[0]
        # else:
        #     return data

    def renumpy(_data):
        if isinstance(_data, list):
            if len(_data) == 2 and 'numpy.' in _data[0]:
                _data = np.dtype(_data[0][6:]).type(_data[1])
            elif len(_data) == 2 and 'set.' == _data[0]:
                _data = set(renumpy(_data[1]))
            else:
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
            elif 'slice_start' in _data:
                _data = slice(_data['slice_start'], _data['slice_stop'], _data['slice_step'])
            else:
                for key in _data.keys():
                    _data[key] = renumpy(_data[key])
        return _data

        # if len(data) == 1:
        #     data = data[0]
    data = renumpy(data)
    return data


def is_jsonable(data):
    """Check to see if data is JSON serializable"""
    try:
        json.dumps(data)
        return True
    except (TypeError, OverflowError):
        return False


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
    used_arrays = [False]
    if embed_arrays and compressed:
        warnings.warn('Cannot compress an array that is embedded in a JSON', UserWarning)
        compressed = False

    def unnumpy(_data):
        if isinstance(_data, list):
            new_data = []
            for i in range(len(_data)):
                new_data.append(unnumpy(_data[i]))
        elif isinstance(_data, set):
            new_data = ['set.', unnumpy(list(_data))]
        elif isinstance(_data, dict):
            new_data = {}
            for key in _data.keys():
                new_data[key] = unnumpy(_data[key])
        elif isinstance(_data, slice):
            new_data = {'slice_start': _data.start, 'slice_stop': _data.stop, 'slice_step': _data.step}
        elif isinstance(_data, np.ndarray):
            used_arrays[0] = True
            if embed_arrays:
                new_data = {"numpy_array": _data.tolist(), "dtype": str(_data.dtype), "shape": _data.shape}
            else:
                new_data = {"numpy_array": 'array_{}'.format(len(out_arrays)), 'dtype': str(_data.dtype), 'shape': _data.shape}
                out_arrays['array_{}'.format(len(out_arrays))] = _data
        elif 'numpy' in str(type(_data)):
            dtype = str(_data.dtype)
            if np.issubdtype(_data, np.integer):
                _data = int(_data)
            elif np.issubdtype(_data, np.floating):
                _data = float(_data)
            new_data = ['numpy.' + dtype, _data]
        else:
            new_data = copy.copy(_data)
        return new_data

    data = unnumpy(data)
    if used_arrays[0]:
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


def is_image(path):
    """Check if the path is an image file"""
    path = str(path)
    try:
        return imghdr.what(path) is not None
    except IsADirectoryError:
        return False


def basicname(path):
    """Return the basename of the path without the extension"""
    return os.path.splitext(os.path.basename(path))[0]


def get_sorted_filenames(pattern, sep='_', ending=True, reverse=False):
    """Sort filenames based on ending digits

    Parameters
    ----------
    pattern : str
        The pattern used with glob to find files
    sep : str
        The separator between the filename and the indexing value (the default is '_')
    ending_index : bool
        Whether the indexing value is at the end of the filename or the start (the default is True)
    reverse : bool
        Whether to reverse the order of the returned filenames

    NOTESs
    -----
    ending_index=True with sep='_' would look like 'filename_1.txt', and ending_index=False would look like '1_filename.txt'

    This method is only useful in the case where you have file_2.txt and file_10.txt where file_10 would be sorted first with other methods because the 1 is at the same inde as the 2.
    """
    def get_copy_num(x):
        x = basicname(x)
        item = x.rsplit(sep, 1) if ending else x.split(sep, 1)
        if len(item) != 2:
            return -1
        item = item[int(ending)]
        if str.isdigit(item):
            return int(item)
        else:
            return -1

    pattern = str(pattern)
    files = glob.glob(pattern)
    max_num = -1
    for item in files:
        max_num = max(max_num, get_copy_num(item))
    digits = num_digits(max_num)
    key_string = "{:0" + str(digits) + "d}"

    def get_copy_key(x):
        x = basicname(x)
        item = x.rsplit(sep, 1) if ending else x.split(sep, 1)
        if len(item) != 2:
            return x
        key = item[int(ending)]
        path = item[int(~ending)]
        if str.isdigit(key):
            key = key_string.format(int(key))
        return sep.join([path, key]) if ending else sep.join([key, path])

    return sorted(files, key=get_copy_key, reverse=False)
