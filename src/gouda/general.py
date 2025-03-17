"""General purpose methods that don't fit other categories."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Generator, Iterable
from typing import Any


def getattr_recursive(item: Any, attr_string: str) -> Any:  # noqa: ANN401
    """Getattr for nested attributes.

    Parameters
    ----------
    item : object
        Any python object with attributes
    attr_string : string
        A string of attributes separated by periods (ex: first.second)

    Note
    ----
    An example would be a module with submodules: `getattr_recursive(os, 'path.basename')` would return os.path.basenamed
    """
    nested_type = type(item).__name__
    cur_item = item
    for key in attr_string.split("."):
        if hasattr(cur_item, key):
            cur_item = getattr(cur_item, key)
        else:
            raise AttributeError(f"'{nested_type}' object has no attribute '{key}'")
        nested_type += "." + type(cur_item).__name__
    return cur_item


def hasattr_recursive(item: Any, attr_string: str) -> Any:  # noqa: ANN401
    """Hasattr for nested attributes.

    Parameters
    ----------
    item : object
        Any python object with attributes
    attr_string : string
        A string of attributes separated by periods (ex: first.second)

    Note
    ----
    An example would be a module with submodules: `hasattr_recursive(os, 'path.basename')` would return True
    """
    cur_item = item
    for key in attr_string.split("."):
        if hasattr(cur_item, key):
            cur_item = getattr(cur_item, key)
        else:
            return False
    return True


def capped_cycle(iterable: Any) -> Generator[Any, None, None]:  # noqa: ANN401
    """Do the same thing as itertools.cycle, but with the StopIteration at the end of each cycle."""
    saved = []
    for item in iterable:
        yield item
        saved.append(item)
    yield StopIteration
    saved.append(StopIteration)
    while saved:  # pragma: no branch
        yield from saved


def nestit(*iterators: Any) -> Generator[Any, None, None]:  # noqa: ANN401
    """Combine iterators into a single nested iterator.

    WARNING
    -------
    Cycling iterators requires saving a copy of each element from sub-iterators and can require significant auxiliary storage (see `itertools.cycle`)
    """
    capped_iterators: list[Any] = [capped_cycle(item) for item in iterators]
    return_object = [next(it) for it in capped_iterators]
    if any(item is StopIteration for item in return_object):
        raise ValueError("Can't nest with an empty iterator")
    yield return_object

    next_up = len(capped_iterators) - 1
    while next_up != -1:
        return_object[next_up] = next(capped_iterators[next_up])
        # print(next_up, return_object)
        if return_object[next_up] is StopIteration:
            next_up -= 1
            continue
        for idx in range(next_up + 1, len(capped_iterators)):
            return_object[idx] = next(capped_iterators[idx])
        next_up = len(capped_iterators) - 1
        yield return_object


def is_iter(x: Any, non_iter: Iterable[type] = (str, bytes, bytearray)) -> bool:  # noqa: ANN401
    """Check if x is iterable.

    Parameters
    ----------
    x : Any
        The variable to check
    non_iter : Iterable[type], optional
        Types to not count as iterable types, by default (str, bytes, bytearray)
    """
    if isinstance(x, tuple(non_iter)):
        return False

    try:
        iter(x)
    except TypeError:
        return False
    else:
        return True


def force_len(x: Any, count: int, pad: str = "wrap") -> Iterable:  # noqa: ANN401
    """Force the length of x to a given count.

    Parameters
    ----------
    x : Any
        The item/iterable to force to a length of count
    count : int
        The output tuple length
    pad : str, optional
        Padding method for extending x. Can be either 'wrap' or 'reflect', by default 'wrap'
    """
    if not is_iter(x):
        return (x,) * count
    else:
        result: Iterable
        if len(x) == count:
            result = x
        elif len(x) < count:
            if pad == "wrap":
                result = list(x)
                while len(result) < count:
                    diff = count - len(result)
                    result.extend(x[:diff])
                result = type(x)(result)
            elif pad == "reflect":
                if len(x) * 2.0 < count:
                    raise ValueError("Cannot reflect enough to force length.")
                result = tuple(list(x) + list(reversed(x))[: count - len(x)])
            else:
                raise ValueError(f"Unknown padding method: {pad}.")
        else:
            result = x[:count]
        return result


def match_len(*args: Any, count: int | None = None, pad: str = "wrap") -> tuple[tuple[Iterable[Any], ...], int]:  # noqa: ANN401
    """Force all input items to the same length.

    Parameters
    ----------
    count : Optional[int], optional
        The length to set all items to. If None, uses the length of the longest item, by default None
    pad : str
        The padding to use to extend an item. Can be either 'wrap' or 'reflect', by default 'wrap'
    """
    if count is None:
        count = max([len(item) if is_iter(item) else 1 for item in args])
    return tuple(force_len(item, count, pad=pad) for item in args), count


def extract_method_kwargs(kwargs: dict, method: Callable, remove: bool = True) -> dict[str, Any]:
    """Extract keyword arguments related to a given method.

    Parameters
    ----------
    kwargs : dict
        A dictionary of keyword arguments
    method : func
        A function with keyword arguments
    remove : bool
        Whether to remove the extracted keyword arguments from the original dict (the default is True)
    """
    method_params = inspect.signature(method).parameters
    method_kwargs = {}
    to_remove = []
    for key, val in kwargs.items():
        if key in method_params:
            method_kwargs[key] = val
            to_remove.append(key)
    if remove:
        for key in to_remove:
            del kwargs[key]
    return method_kwargs
