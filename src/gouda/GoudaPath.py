"""Path-like class for easier file navigation"""
import glob
import os

__author__ = "Sean Mullan"
__copyright__ = "Sean Mullan"
__license__ = "mit"


class GoudaPath(os.PathLike):
    """Note: requires python 3.6+"""
    def __init__(self, *path, use_absolute=True):
        path = os.path.join(*path)
        self.use_absolute = use_absolute
        if use_absolute:
            self.__path = os.path.abspath(path)
        else:
            self.__path = path

    def __call__(self, *path_args, use_absolute=None):
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

    def __truediv__(self, path):
        return GoudaPath(self(path), use_absolute=False)

    def __floordiv__(self, path):
        return GoudaPath(self(path), use_absolute=True)

    def __add__(self, path):
        return GoudaPath(self.__path + path, use_absolute=self.use_absolute)

    def __getitem__(self, index):
        split_path = self.__path.split(os.path.sep)
        if len(split_path[0]) == 0:
            split_path = split_path[1:]
        if isinstance(index, slice):
            return os.path.join(*split_path[index])
        return split_path[index]

    def __len__(self):
        split_path = self.__path.split(os.path.sep)
        if len(split_path[0]) == 0:
            return len(split_path) - 1
        else:
            return len(split_path)

    def __fspath__(self):
        return os.fspath(self.__path)

    def glob(self, pattern, basenames=False, recursive=False, sort=False):
        results = glob.glob(os.path.join(self.__path, pattern), recursive=recursive)
        if basenames:
            results = [os.path.basename(item) for item in results]
        if sort:
            return sorted(results)
        return results

    def parent_dir(self):
        parent_dir = os.path.dirname(self.__path)
        if len(parent_dir) == 0:
            parent_dir = os.path.join(os.pardir, os.path.basename(os.path.abspath(parent_dir)))
        return GoudaPath(parent_dir, use_absolute=self.use_absolute)

    def children(self, dirs_only=True, exclude_dirs=False, basenames=False):
        """If the path is a directory, get the child paths contained by it.

        Parameters
        ----------
        dirs_only: bool (default=True)
            Return only child directories
        exclude_dirs: bool (default=False)
            Return only child non-directories
        basenames: bool (default=False)
            Return only the basename of the child paths

        Returns
        -------
        list: a list of the children of the current path
        """

        if not self.is_dir():
            raise NotADirectoryError("Not a directory: {}".format(self.__path))
        children = [self(child, use_absolute=self.use_absolute) for child in os.listdir(self.__path)]
        if dirs_only:
            children = list(filter(lambda x: os.path.isdir(x), children))
        if exclude_dirs:
            children = list(filter(lambda x: not os.path.isdir(x), children))
        if basenames:
            children = [child.basename() for child in children]
        return children

    @property
    def path(self):
        return self.__path

    @property
    def abspath(self):
        return os.path.abspath(self.__path)

    def basename(self):
        return os.path.basename(self.__path)

    def is_dir(self):
        return os.path.isdir(self.__path)

    def extension(self):
        return '.' + self.__path.rsplit('.', 1)[-1]

    def exists(self):
        return os.path.exists(self)
