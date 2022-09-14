"""General purpose methods that don't fit other categories"""


def getattr_recursive(item, attr_string):
    """getattr for nested attributes

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
    for key in attr_string.split('.'):
        try:
            cur_item = getattr(cur_item, key)
            nested_type += '.' + type(cur_item).__name__
        except AttributeError:
            raise AttributeError("'{}' object has no attribute '{}'".format(nested_type, key))
    return cur_item


def hasattr_recursive(item, attr_string):
    """hasattr for nested attributes

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
    for key in attr_string.split('.'):
        try:
            cur_item = getattr(cur_item, key)
        except AttributeError:
            return False
    return True


def capped_cycle(iterable):
    """Same thing as itertools.cycle, but with the StopIteration at the end of each cycle"""
    saved = []
    for item in iterable:
        yield item
        saved.append(item)
    yield StopIteration
    saved.append(StopIteration)
    while saved:  # pragma: no branch
        for element in saved:
            yield element


def nestit(*iterators):
    """Combine iterators into a single nested iterator.

    WARNING
    -------
    Cycling iterators requires saving a copy of each element from sub-iterators and can require significant auxiliary storage (see itertools.cycle)
    """
    iterators = [capped_cycle(item) for item in iterators]
    return_object = [next(it) for it in iterators]
    if any([item == StopIteration for item in return_object]):
        raise ValueError("Can't nest with an empty iterator")
    yield return_object

    next_up = len(iterators) - 1
    while next_up != -1:
        return_object[next_up] = next(iterators[next_up])
        # print(next_up, return_object)
        if return_object[next_up] == StopIteration:
            next_up -= 1
            continue
        for idx in range(next_up + 1, len(iterators)):
            return_object[idx] = next(iterators[idx])
        next_up = len(iterators) - 1
        yield return_object
