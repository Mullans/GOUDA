"""Path-like class for easier file navigation"""
import glob
import imghdr
import os

__author__ = "Sean Mullan"
__copyright__ = "Sean Mullan"
__license__ = "mit"


class GoudaPath(os.PathLike):
    """Note: requires python 3.6+"""
    def __init__(self, *path, use_absolute=True):
        """PathLike class for easier file traversal

        Parameters
        ----------
        path : str
            One or more strings to join as a filepath
        use_absolute : bool
            Whether to convert the path to the absolute path (the default is True)
        """
        if len(path) == 1 and isinstance(path[0], GoudaPath):
            path = path[0].path
        elif len(path) == 0:
            path = '.'
        else:
            path = os.path.join(*path)

        self.use_absolute = use_absolute
        if use_absolute:
            self.__path = os.path.abspath(path)
        else:
            self.__path = path

    def __call__(self, *path_args, use_absolute=None):
        """Add to the current path.

        Parameters
        ----------
        *path_args : os.PathLike
            One or more strings or PathLike objects to create new GoudaPaths from
        use_absolute : bool
            Whether to convert the path to the absolute path (defaults to the object parameter value)

        Returns
        -------
        path | paths : GoudaPath | list
            Returns one or more GoudaPaths that are children of the current object path
        """
        if use_absolute is None:
            use_absolute = self.use_absolute
        if len(path_args) == 0:
            return GoudaPath(self, use_absolute=use_absolute)

        if isinstance(path_args[0], list) and len(path_args) == 1:
            path_args = path_args[0]

        if len(path_args) == 1:
            return GoudaPath(os.path.join(self.__path, path_args[0]), use_absolute=use_absolute)
        else:
            return [GoudaPath(os.path.join(self.__path, path_args[i]), use_absolute=use_absolute) for i in range(len(path_args))]

    def __str__(self):
        return self.__path

    def __repr__(self):
        return "GoudaPath('{}')".format(self.__path)

    def __bytes__(self):
        return self.abspath.encode()

    def __truediv__(self, path):
        """Shortcut method for __call__ with a single child path and use_absolute set to False"""
        return GoudaPath(self(path), use_absolute=False)

    def __floordiv__(self, path):
        """Shortcut method for __call__ with a single child path and use_absolute set to True"""
        return GoudaPath(self(path), use_absolute=True)

    def __add__(self, path):
        """Appends a path to the end of the current path.

        Note
        ----
        This appends the strings. This does not join the paths, so it is not a child path of the current path
        """
        return GoudaPath(self.__path + path, use_absolute=self.use_absolute)

    def __getitem__(self, index):
        """Get part of the current path hierarchy."""
        split_path = self.__path.split(os.path.sep)
        if len(split_path[0]) == 0:
            split_path = split_path[1:]
        if isinstance(index, slice):
            return os.path.join(*split_path[index])
        return split_path[index]

    def __len__(self):
        """Get the length of the current path hierarchy."""
        split_path = self.__path.split(os.path.sep)
        if len(split_path[0]) == 0:
            return len(split_path) - 1
        else:
            return len(split_path)

    def __contains__(self, value):
        return value in self.__path

    def __fspath__(self):
        """Note: fspath is always absolute path"""
        return os.fspath(os.path.abspath(self.__path))

    def glob(self, pattern, as_gouda=False, basenames=False, recursive=False, sort=False):
        """Make a glob call starting from the current path.

        Parameters
        ----------
        pattern : str
            Pattern to match with the glob
        as_gouda : bool
            Whether to return results as GoudaPath objects (the default is False)
        basenames : bool
            Whether to return only the basenames of results (the default is False)
        recursive : bool
            The setting for the glob recursive argument (the default is False)
        sort : bool
            Whether to sort the results (the default is False)
        """
        results = glob.glob(os.path.join(self.__path, pattern), recursive=recursive)
        if basenames:
            results = [os.path.basename(item) for item in results]
        if sort:
            results = sorted(results)
        if as_gouda:
            results = [GoudaPath(item, use_absolute=self.use_absolute) for item in results]
        return results

    def parent_dir(self):
        """Return the parent directory of the current path."""
        parent_dir = os.path.dirname(self.__path)
        if len(parent_dir) == 0:
            parent_dir = os.path.join(os.pardir, os.path.basename(os.path.abspath(parent_dir)))
        return GoudaPath(parent_dir, use_absolute=self.use_absolute)

    def num_children(self, dirs_only=True, files_only=False, include_hidden=False):
        """If the path is a directory, return a count of the child paths"""
        if not self.is_dir():
            raise NotADirectoryError("Not a directory: {}".format(self.__path))
        children = [self(child) for child in os.listdir(self.__path)]
        if not include_hidden:
            children = [child for child in children if not child.is_hidden()]
        if dirs_only:
            children = [child for child in children if os.path.isdir(child)]
        if files_only:
            children = [child for child in children if os.path.isfile(child)]
        return len(children)

    def children(self, dirs_only=True, files_only=False, basenames=False, include_hidden=False):
        """If the path is a directory, get the child paths contained by it.

        Parameters
        ----------
        dirs_only: bool (default=True)
            Return only child directories
        files_only: bool (default=False)
            Return only child non-directories
        basenames: bool (default=False)
            Return only the basename of the child paths

        Returns
        -------
        list: a list of the children of the current path
        """

        if not self.is_dir():
            raise NotADirectoryError("Not a directory: {}".format(self.__path))
        children = os.listdir(self.__path)
        children = [self(child) for child in children]
        if not include_hidden:
            children = [child for child in children if not child.is_hidden()]
        if dirs_only:
            children = list(filter(lambda x: os.path.isdir(x), children))
        if files_only:
            children = list(filter(lambda x: os.path.isfile(x), children))
        if basenames:
            children = [child.basename() for child in children]
        return children

    def get_images(self, sort=False, basenames=False):
        """Return all images contained in the directory of the path"""
        if not self.is_dir():
            raise NotADirectoryError("Not a directory: {}".format(self.__path))
        images = []
        for item in os.listdir(self.__path):
            try:
                check_item = os.path.join(self.__path, item)
                if imghdr.what(check_item) is not None:
                    if basenames:
                        images.append(item)
                    else:
                        images.append(check_item)
            except IsADirectoryError:
                continue
        if sort:
            images = sorted(images, key=lambda x: os.path.basename(x))
        return images

    def resolve_links(self):
        """Resolve any symbolic links in the path using realpath"""
        self.__path = os.path.realpath(self.__path)

    @property
    def path(self):
        return self.__path

    @property
    def abspath(self):
        return os.path.abspath(self.__path)

    @property
    def realpath(self):
        return os.path.realpath(self.__path)

    def basename(self):
        return os.path.basename(self.__path)

    def basicname(self):
        return os.path.splitext(os.path.basename(self.__path))[0]

    def is_dir(self):
        """Check if the path is a directory"""
        return os.path.isdir(self.__path)

    def is_image(self):
        """Check if the path is an image (see imghdr.what for image types)"""
        try:
            return imghdr.what(self.__path) is not None
        except (IsADirectoryError, FileNotFoundError):
            return False

    def is_hidden(self):
        return self.basename().startswith('.')

    def extension(self):
        return '.' + self.__path.rsplit('.', 1)[-1]

    def exists(self):
        return os.path.exists(self)

    def endswith(self, suffix):
        return self.__path.endswith(suffix)

    def rstrip(self, chars=None):
        return GoudaPath(self.__path.rstrip(chars), use_absolute=self.use_absolute)

    def startswith(self, prefix):
        return self.__path.startswith(prefix)

    def add_basename(self, path):
        return self(os.path.basename(path))
