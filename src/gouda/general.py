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
