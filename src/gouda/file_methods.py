"""General file method and JSON handling"""
import copy
import fnmatch
import glob
import gzip
import importlib
import json
import numpy as np
import os
import re
import warnings
from contextlib import nullcontext
from typing import Union

from gouda.data_methods import num_digits

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
    # TODO - Can this all be replaced with `os.makedirs(os.path.join([str(path) for path in paths]))`? The only difference is the non-dir check
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


def next_filename(filename: str, first_idx: int = 2, path_fmt: str = '{base_name}{sep}{idx}{ext}', default_sep: str = '_') -> str:
    """Check if a given file exists, and return the next numbered filename in the order if it does.

    Parameters
    ----------
    filename : str
        The filename to check and iterate on
    first_idx : int, optional
        The first index to check if `filename` exists, by default 2
    path_fmt : str, optional
        The format of the basename of the numbered filenames, by default '{base_name}{sep}{idx}{ext}'
    default_sep : str, optional
        The default separator, by default '_'

    Returns
    -------
    str
        The next available filename in the sequence

    Notes
    -----
    The `path_fmt` parameter should be a format string with the following named fields:
        - "{base_name}": The base name of the file
        - "{sep}": The separator between the base name and the index, this is optional
        - "{idx}": The index of the file, this can be "{idx:03d}" for zero-padded indices
        - "{ext}": The file extension, this should include the period
    `path_fmt` does not take into account the path of the file, only the base name and extension.

    Examples
    --------
    >>> next_filename('test.txt')
    'test_2.txt'
    >>> next_filename('test.txt')
    'test.txt'  # 'test.txt' does not exist
    >>> next_filename('/path/to/test.txt')
    '/path/to/test_2.txt'
    >>> next_filename('test.txt', path_fmt='{idx:03}{sep}{base_name}{ext}')
    '002_test.txt'
    >>> next_filename('1_test-2.txt', path_fmt='{idx}{sep}{base_name}{ext}')
    '2_test-2.txt'
    """
    filename = str(filename)
    if not os.path.exists(filename):
        return filename
    path, base_name, extension = fullsplit(filename)
    sep = default_sep

    # Convert format to regex pattern
    pattern = re.escape(path_fmt)
    pattern = pattern.replace(r'\{idx\}', r'(?P<idx>\d+)')
    pattern = pattern.replace(r'\{sep\}', r'(?P<sep>[_\-]+)')
    pattern = pattern.replace(r'\{base_name\}', r'(?P<base_name>.+)')
    pattern = pattern.replace(r'\{ext\}', r'(?P<ext>\.[^.]+)')
    pattern = re.compile(f"^{pattern}$")

    # Identify current index, base name, and separator
    match = pattern.match(f"{base_name}{extension}")
    if match:
        current_idx = int(match.group("idx")) if match.group("idx") else None
        if 'sep' in match.groupdict():
            sep = match.group("sep") or "_"
        base_name = match.group("base_name")
    else:
        current_idx = None
    idx = 1 + current_idx if current_idx is not None else first_idx

    while True:
        next_check = os.path.join(
            path,
            path_fmt.format(base_name=base_name, sep=sep, idx=idx, ext=extension)
        )
        if os.path.isfile(next_check):
            idx += 1
        else:
            return next_check


# JSON methods
def load_json(filename):
    """Load a JSON file, and re-form any numpy arrays if :func:`~gouda.save_json` was used to write them."""
    filename = str(filename)
    with open(filename, 'r') as f:
        data = json.load(f)
    np_filename = None
    if isinstance(data, dict) and 'slice_start' in data:
        data = slice(data['slice_start'], data['slice_stop'], data['slice_step'])
    elif isinstance(data, list):
        if data[-1] == 'numpy':
            np_filename = filename.rsplit('.', 1)[0] + '_array.npz'
            data = data[0]
        elif data[-1] == 'numpy_zip':
            np_filename = filename.rsplit('.', 1)[0] + '_arrayzip.npz'
            data = data[0]
        elif data[-1] == 'numpy_embed':
            data = data[0]
        # else:
        #     return data
    with open(np_filename, 'rb') if np_filename is not None else nullcontext() as numpy_file:
        if np_filename is not None:
            arrays = np.load(numpy_file)

        def renumpy(_data):
            if isinstance(_data, list):
                if len(_data) == 2 and isinstance(_data[0], str):
                    if 'numpy.' in _data[0]:
                        _data = np.dtype(_data[0][6:]).type(_data[1])
                    elif 'set.' in _data[0]:
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
    filename = str(filename)
    out_arrays = {}
    used_arrays = [False]
    if embed_arrays and compressed:
        warnings.warn('Cannot compress an array that is embedded in a JSON', UserWarning)
        compressed = False

    def unnumpy(_data):
        if isinstance(_data, (list, tuple)):
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
            if np.issubdtype(_data.dtype, np.integer):
                _data = int(_data)
            elif np.issubdtype(_data.dtype, np.floating):
                _data = float(_data)
            else:
                raise ValueError('Un-JSON-able dtype {} found'.format(dtype))
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


def create_image_checker():
    """Create the :meth:`gouda.is_image` method that can be used to check if a file is an image based on available libraries"""
    if importlib.util.find_spec('PIL.Image'):
        import PIL.Image
        def is_image(path):
            """Check if the path is an image file"""
            path = str(path)
            if os.path.isdir(path):
                return False
            try:
                PIL.Image.open(path)
                return True
            except PIL.UnidentifiedImageError:
                return False
    elif importlib.util.find_spec('puremagic'):
        import puremagic
        def is_image(path):
            """Check if the path is an image file"""
            path = str(path)
            if os.path.isdir(path):
                return False
            return puremagic.magic_file(path)[0].mime_type.startswith('image/')
    elif importlib.util.find_spec('imghdr'):
        import imghdr
        def is_image(path):
            """Check if the path is an image file"""
            path = str(path)
            if os.path.isdir(path):
                return False
            return imghdr.what(path) is not None
    else:
        def is_image(path):
            """Check if the path is an image file"""
            raise ImportError("No image checking libraries found - install PIL, puremagic, or imghdr")
    return is_image

is_image = create_image_checker()

def fullsplit(path):
    """Split the path into head, basename, and extension

    NOTE
    ----
    * This splits at the first non-leading period of the basename, compared to os.path.splitext which splits the whole path at the last period.
    * Leading periods are considered to be part of the basename
    """
    # return os.path.split(path)
    head, tail = os.path.split(path)
    splitpath = tail.split('.')
    to_add = ''
    while splitpath[0] == '':
        splitpath = splitpath[1:]
        to_add += '.'
    splitpath[0] = to_add + splitpath[0]
    if len(splitpath) == 1:
        return head, splitpath[0], ''
    elif len(splitpath) > 2:
        return head, splitpath[0], '.' + '.'.join(splitpath[1:])
    else:
        return head, splitpath[0], '.' + splitpath[1]


def basicname(path):
    """Return the basename of the path without the extension"""
    return fullsplit(path)[1]


def fast_glob(base_path: str,
              glob_pattern: Union[str, re.Pattern],
              regex_flags: int = 0,
              sort: bool = False,
              basenames: bool = False,
              recursive: bool = False,
              follow_symlinks: bool = False,
              iter: bool = False) -> list[str]:
    """Fast globbing method that uses scandir to find files and directories whose basename match a given pattern.

    Parameters
    ----------
    base_path : str
        The base directory to search in
    glob_pattern : str | re.Pattern
        The pattern to check against the basenames of the files and directories
    regex_flags : int, optional
        Flags to pass to `re.compile(glob_pattern)`, by default 0
    sort : bool, optional
        If True, sort results by basename, by default False
    basenames : bool, optional
        If True, only return basenames of files, by default False
    recursive : bool, optional
        If True, recurse into sub-directories, by default False
    follow_symlinks : bool, optional
        If True and `recursive` is True, follow directory symlinks when recursing, by default False
    iter : bool, optional
        If True, return an iterator instead of a list, by default False

    Note
    ----
    * `sort` is only applied if `iter` is False

    Returns
    -------
    list[str]
        The list of files and directories that match the glob pattern
    """
    if not isinstance(glob_pattern, re.Pattern):
        glob_pattern = fnmatch.translate(glob_pattern)
        glob_pattern = re.compile(glob_pattern, flags=regex_flags)

    def search_generator():
        for item in os.scandir(base_path):
            if recursive and item.is_dir(follow_symlinks=follow_symlinks):
                yield from fast_glob(item.path,
                                     glob_pattern,
                                     regex_flags=regex_flags,
                                     sort=False,
                                     basenames=basenames,
                                     recursive=recursive,
                                     follow_symlinks=follow_symlinks,
                                     iter=iter)
            if glob_pattern.match(item.name):
                yield item.name if basenames else item.path
    if iter:
        return search_generator()
    else:
        results = list(search_generator())
        if sort:
            results = sorted(results, key=lambda x: basicname(x))
        return results


def find_images(base_path: str,
                sort: bool = False,
                basenames: bool = False,
                recursive: bool = False,
                follow_symlinks: bool = False,
                fast_check: bool = False,
                iter: bool = False) -> list[str]:
    """_summary_

    Parameters
    ----------
    base_path : str
        The base directory to search in
    sort : bool, optional
        If True, sort results by basename, by default False
    basenames : bool, optional
        If True, only return basenames of files, by default False
    recursive : bool, optional
        If True, recurse into sub-directories, by default False
    follow_symlinks : bool, optional
        If True and `recursive` is True, follow directory symlinks when recursing, by default False
    fast_check : bool, optional
        If True, check files by extension. If False, use imghdr to check for image bytes. by default False
    iter : bool, optional
        If True, return an iterator instead of a list, by default False

    Note
    ----
    * `sort` is only applied if `iter` is False
    * `fast_check` only checks for common image extensions, while :meth:`gouda.file_methods.is_image` checks for image bytes

    Returns
    -------
    list[str]
        The list of images
    """
    def image_generator():
        if fast_check:
            pattern = '|'.join([r'\.jpe?g', r'\.png', r'\.tiff', r'\.gif', r'\.bmp', r'\.webp'])  # noqa W605
            regex = re.compile(r'.*({})$'.format(pattern), re.I)
            yield from fast_glob(base_path, regex, regex_flags=re.I, sort=False, basenames=basenames, recursive=recursive, follow_symlinks=follow_symlinks, iter=True)
        else:
            for item in os.scandir(base_path):
                if recursive and item.is_dir(follow_symlinks=follow_symlinks):
                    yield from find_images(item.path,
                                           sort=False,
                                           basenames=basenames,
                                           recursive=recursive,
                                           follow_symlinks=follow_symlinks,
                                           fast_check=False, iter=iter)
                elif is_image(item.path):
                    yield item.name if basenames else item.path
    if iter:
        return image_generator()
    else:
        images = list(image_generator())
        if sort:
            images = sorted(images, key=lambda x: basicname(x))
        return images


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

    This method is only useful in the case where you have file_2.txt and file_10.txt where file_10 would be sorted first with other methods because the 1 is at the same index as the 2.
    """
    def get_copy_num(x):
        x = basicname(x)
        item = x.rsplit(sep, 1) if ending else x.split(sep, 1)
        if len(item) != 2:
            return -1
        item = item[-1 if ending else 0]
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
        path = item[int(not ending)]
        if str.isdigit(key):
            key = key_string.format(int(key))
        return sep.join([path, key]) if ending else sep.join([key, path])

    return sorted(files, key=get_copy_key, reverse=False)


def save_arr(path, arr):
    path = str(path)
    if path.endswith('.gz'):
        with gzip.open(path, 'wb') as f:
            np.save(f, arr)
    else:
        np.save(path, arr)


def read_arr(path):
    path = str(path)
    if path.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            data = np.load(f)
    else:
        with open(path, 'rb') as f:
            data = np.load(f)
    return data
