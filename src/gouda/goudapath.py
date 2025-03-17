"""Path-like class for easier file navigation"""

import glob
import os
import pathlib
import re
import warnings
from collections.abc import Generator
from typing import TypeVar, Union

from gouda.file_methods import ensure_dir, fast_glob, find_images, is_image

__author__ = "Sean Mullan"
__copyright__ = "Sean Mullan"
__license__ = "mit"


TGoudaPath = TypeVar("TGoudaPath", bound="GoudaPath")
GPathLike = Union[str, TGoudaPath, os.PathLike]


class GoudaPath(os.PathLike):
    # Eventually, subclass pathlib.Path (available in 3.12+)
    __slots__ = ("__path", "_hash", "_parsed_parts", "_parts_normcase_cached", "use_absolute")

    def __init__(self: TGoudaPath, *path: GPathLike, use_absolute: bool = False, ensure_dir: bool = False) -> None:
        """PathLike class for easier file traversal

        Parameters
        ----------
        path : str
            One or more strings to join as a filepath
        use_absolute : bool
            Whether to convert the path to the absolute path (the default is False)
        """
        if len(path) == 1 and isinstance(path[0], GoudaPath):
            path = path[0].path
        elif len(path) == 0:
            path = "."
        else:
            path = os.path.join(*path)

        self.use_absolute = use_absolute
        if self.use_absolute:
            self.__path = os.path.abspath(path)
        else:
            self.__path = path
        self.__path = os.path.normpath(self.__path)  # Normalize the path, mostly needed for os separators

        if ensure_dir:
            self.ensure_dir()

    def __call__(
        self: TGoudaPath, *path_args: GPathLike, use_absolute: bool | None = None
    ) -> TGoudaPath | list[TGoudaPath]:
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
            return GoudaPath(os.path.join(self.path, path_args[0]), use_absolute=use_absolute)
        else:
            return [
                GoudaPath(os.path.join(self.path, path_args[i]), use_absolute=use_absolute)
                for i in range(len(path_args))
            ]

    def __str__(self: TGoudaPath) -> str:
        return self.path

    def __repr__(self: TGoudaPath) -> str:
        return f"GoudaPath('{self.path}')"

    def __bytes__(self: TGoudaPath) -> bytes:
        return self.abspath.encode()

    def __truediv__(self: TGoudaPath, path: GPathLike) -> TGoudaPath:
        """Shortcut method for __call__ with a single child path and use_absolute set to False"""
        return GoudaPath(self(path), use_absolute=False)

    def __floordiv__(self: TGoudaPath, path: GPathLike) -> TGoudaPath:
        """Shortcut method for __call__ with a single child path and use_absolute set to True"""
        return GoudaPath(self(path), use_absolute=True)

    def __add__(self: TGoudaPath, path: GPathLike) -> TGoudaPath:
        """Appends a path to the end of the current path.

        Note
        ----
        This appends the strings. This does not join the paths, so it is not a child path of the current path
        """
        return GoudaPath(self.path + path, use_absolute=self.use_absolute)

    @property
    def parts(self: TGoudaPath) -> list[str]:
        try:
            return self._parsed_parts
        except AttributeError:
            self._parsed_parts = pathlib.Path(self.path).parts
            return self._parsed_parts

    def __getitem__(self: TGoudaPath, index: int) -> str:
        """Get part of the current path hierarchy."""
        split_path = self.parts
        if isinstance(index, slice):
            return os.path.join(*split_path[index])
        return split_path[index]

    def __setitem__(self: TGoudaPath, index: int, value: str) -> None:
        """Change part of the current path hierarchy."""
        split_path = list(self.parts)
        split_path[index] = value
        self.__update_path(*split_path)

    def insert(self: TGoudaPath, index: int, value: str) -> None:
        """Insert a part into the current path hierarchy."""
        split_path = list(self.parts)
        split_path.insert(index, value)
        self.__update_path(*split_path)

    def __len__(self: TGoudaPath) -> int:
        """Get the length of the current path hierarchy."""
        return len(self.parts)

    def __contains__(self: TGoudaPath, value: str) -> bool:
        return value in self.__path

    def __fspath__(self: TGoudaPath) -> str | bytes:
        """Note: fspath is always absolute path
        Should it be?
        """
        return os.fspath(self.__path)

    def ensure_dir(self: TGoudaPath) -> TGoudaPath:
        if "." in os.path.basename(self.path):
            ensure_dir(self.parent_dir())
        else:
            ensure_dir(self.path)
        return self

    def glob(
        self: TGoudaPath,
        pattern: GPathLike,
        as_gouda: bool = True,
        basenames: bool = False,
        recursive: bool = False,
        sort: bool = False,
    ) -> list[str | TGoudaPath]:
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
            Whether to sort the results using the sorted function (the default is False)
        """
        results = glob.glob(os.path.join(self.__path, pattern), recursive=recursive)
        if basenames:
            results = [os.path.basename(item) for item in results]
        if sort:
            results = sorted(results)
        if as_gouda:
            results = [GoudaPath(item, use_absolute=self.use_absolute) for item in results]
        return results

    def search(
        self: TGoudaPath,
        pattern: GPathLike | re.Pattern,
        as_gouda: bool = True,
        basenames: bool = False,
        recursive: bool = False,
        sort: bool = False,
        iter: bool = False,
    ) -> list[str | TGoudaPath]:
        """Make a fast_glob call starting from the current path (matches basenames against pattern).

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
            Whether to sort the results using the sorted function (the default is False)
        iter : bool
            If True, return a generator instead of a list (the default is False)
        """

        def search_gen():
            for item in fast_glob(self.__path, pattern, basenames=basenames, sort=sort, recursive=recursive):
                if as_gouda:
                    item = GoudaPath(item, use_absolute=self.use_absolute)
                yield item

        if iter:
            return search_gen()
        else:
            return list(search_gen())

    def globfirst(
        self: TGoudaPath, pattern: GPathLike, as_gouda: bool = True, basename: bool = False, recursive: bool = False
    ) -> str | TGoudaPath:
        """Make a glob call starting from the current path and return only the first result in glob order.

        Parameters
        ----------
        pattern : str
            Pattern to match with the glob
        as_gouda : bool
            Whether to return the result as a GoudaPath object (the default is False)
        basename : bool
            Whether to return only the basename of the result (the default is False)
        recursive : bool
            The setting for the glob recursive argument (the default is False)
        """
        for item in glob.iglob(os.path.join(self.__path, pattern), recursive=recursive):
            if basename:
                item = os.path.basename(item)
            if as_gouda:
                item = GoudaPath(item, use_absolute=self.use_absolute)
            return item

    def parent_dir(self: TGoudaPath) -> TGoudaPath:
        """Return the parent directory of the current path."""
        parent_dir = os.path.dirname(self.__path)
        if len(parent_dir) == 0:
            parent_dir = os.path.join(os.pardir, os.path.basename(os.path.abspath(parent_dir)))
        return GoudaPath(parent_dir, use_absolute=self.use_absolute)

    def num_children(
        self: TGoudaPath, dirs_only: bool = True, files_only: bool = False, include_hidden: bool = False
    ) -> int:
        """If the path is a directory, return a count of the child paths"""
        return len(self.children(dirs_only=dirs_only, files_only=files_only, include_hidden=include_hidden))

    def children(
        self: TGoudaPath,
        dirs_only: bool = True,
        files_only: bool = False,
        basenames: bool = False,
        include_hidden: bool = False,
    ) -> list[GPathLike]:
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

        Note
        ----
        Returns the same as iterdir, but as a list
        """
        if not self.is_dir():
            raise NotADirectoryError(f"Not a directory: {self.__path}")
        return list(
            self.iterdir(dirs_only=dirs_only, files_only=files_only, basenames=basenames, include_hidden=include_hidden)
        )

    def iterdir(
        self: TGoudaPath,
        dirs_only: bool = True,
        files_only: bool = False,
        basenames: bool = False,
        include_hidden: bool = False,
    ) -> Generator[GPathLike, None, None]:
        """If the path is a directory, iterate through the child paths contained by it.

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
            raise NotADirectoryError(f"Not a directory: {self.__path}")
        for child in os.scandir(self.__path):
            if not include_hidden and child.name.startswith("."):
                continue
            if dirs_only and not child.is_dir():
                continue
            if files_only and not child.is_file():
                continue
            if basenames:
                yield child.name
            else:
                yield GoudaPath(child.path)

    def get_images(
        self: TGoudaPath,
        sort: bool = False,
        basenames: bool = False,
        recursive: bool = False,
        fast_check: bool = True,
        iter: bool = False,
    ) -> list[str]:
        """Return all images contained in the directory of the path

        Parameters
        ----------
        sort : bool
            Whether to sort the results by basename
        basenames : bool
            Whether to return the image paths as basenames or fullpaths
        fast_check : bool
            If true, this only checks the file extension for jpg, jpeg, png,
            tiff, gif, and bmp extensions. If false, this uses imghdr to check
            the content of each file for image data.
        iter : bool, optional
            If true, return a generator instead of a list (the default is False)
        """
        if not self.is_dir():
            raise NotADirectoryError(f"Not a directory: {self.__path}")
        return find_images(
            self.__path, sort=sort, basenames=basenames, recursive=recursive, fast_check=fast_check, iter=iter
        )

    def resolve_links(self: TGoudaPath) -> TGoudaPath:
        """Resolve any symbolic links in the path using realpath"""
        return GoudaPath(os.path.realpath(self.__path))

    @property
    def path(self: TGoudaPath) -> str:
        """Returns the relative path if use_absolute is False, otherwise returns the absolute path"""
        return self.__path

    def __update_path(self: TGoudaPath, *args) -> str:
        self.__path = os.path.join(*args)
        self._clear_cache()

    @property
    def abspath(self: TGoudaPath) -> str:
        """Equivalent to os.path.abspath"""
        return os.path.abspath(self.__path)

    @property
    def realpath(self: TGoudaPath) -> str:
        """Equivalent to os.path.realpath"""
        return os.path.realpath(self.__path)

    def basename(self: TGoudaPath) -> str:
        """Equivalent to os.path.basename"""
        return os.path.basename(self.__path)

    def basicname(self: TGoudaPath) -> str:
        """Return the base name of the file without the extension"""
        warnings.warn("`.basicname()` is deprecated and will be removed, use `.stem()` instead", DeprecationWarning)
        return self.stem()

    def is_dir(self: TGoudaPath) -> bool:
        """Check if the path is a directory"""
        return os.path.isdir(self.__path)

    def is_image(self: TGoudaPath) -> bool:
        """Check if the path is an image (see imghdr.what for image types)"""
        return is_image(self.__path)

    def is_hidden(self: TGoudaPath) -> bool:
        """Check if the file is hidden in the filesystem (starts with .)"""
        if not self.exists():
            raise FileNotFoundError(f"File not found: {self.__path}")
        return self.basename().startswith(".")

    def extension(self: TGoudaPath) -> str:
        """Return just the extension of the file"""
        return self.fullsplit()[2]

    def exists(self: TGoudaPath) -> bool:
        """Check if the file indicated by the GoudaPath exists"""
        return os.path.exists(self)

    def endswith(self: TGoudaPath, suffix: str) -> bool:
        """Check if the the path ends with the given suffix"""
        return self.__path.endswith(suffix)

    def replace(self: TGoudaPath, old: str, new: str) -> TGoudaPath:
        """Replace part of the path with a new string"""
        return GoudaPath(self.__path.replace(old, new), use_absolute=self.use_absolute)

    def rstrip(self: TGoudaPath, chars: str | None = None) -> TGoudaPath:
        """Remove leading and trailing characters from the path

        Parameters
        ----------
        chars : string
            The set of characters to strip from either end of the path. Using None defauts to whitespace (the default is None)
        """
        return GoudaPath(self.__path.rstrip(chars), use_absolute=self.use_absolute)

    def fullsplit(self: TGoudaPath) -> tuple[str, str]:
        """Split the path into basename and extension

        NOTE
        ----
        * This splits at the first non-leading period of the basename, compared to os.path.splitext which splits the whole path at the last period.
        * Leading periods are considered to be part of the basename
        """
        head, tail = os.path.split(self.__path)
        splitpath = tail.split(".")
        to_add = ""
        while splitpath[0] == "":
            splitpath = splitpath[1:]
            to_add += "."
        splitpath[0] = to_add + splitpath[0]
        if len(splitpath) == 1:
            return head, splitpath[0], ""
        elif len(splitpath) > 2:
            return head, splitpath[0], "." + ".".join(splitpath[1:])
        else:
            return head, splitpath[0], "." + splitpath[1]

    def startswith(self: TGoudaPath, prefix: str) -> bool:
        """Check if the path starts with the given suffix"""
        return self.__path.startswith(prefix)

    def add_basename(self: TGoudaPath, path: str) -> bool:
        """Add the basename of the given path to the end of the GoudaPath"""
        return self(os.path.basename(path))

    def stem(self: TGoudaPath) -> str:
        return self.fullsplit()[1]

    def with_stem(self: TGoudaPath, stem: str) -> TGoudaPath:
        """Return a new GoudaPath with the same path but a new stem"""
        return GoudaPath(self.fullsplit()[0], stem + self.fullsplit()[2], use_absolute=self.use_absolute)

    def with_basename(self: TGoudaPath, basename: str) -> TGoudaPath:
        """Return a new GoudaPath with the same path but a new basename"""
        return GoudaPath(self.fullsplit()[0], basename, use_absolute=self.use_absolute)

    def with_extension(self: TGoudaPath, ext: str) -> TGoudaPath:
        """Return a new GoudaPath with the same path but a new extension"""
        if not ext.startswith("."):
            ext = "." + ext
        return GoudaPath(self.fullsplit()[0], self.fullsplit()[1] + ext, use_absolute=self.use_absolute)

    def as_posix(self: TGoudaPath) -> str:
        """Return the string path with forward slashes."""
        return (self.__path).replace(os.path.sep, "/")

    def as_pathlib(self: TGoudaPath) -> pathlib.Path:
        """Return the path as a pathlib.Path object"""
        return pathlib.Path(self.__path)

    def _opener(self, name, flags, mode=0o666):
        return os.open(self, flags, mode)

    def open(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        return open(self, mode, buffering, encoding, errors, newline, opener=self._opener)

    def read_bytes(self):
        with self.open(mode="rb") as f:
            return f.read()

    def read_text(self, encoding=None, errors=None):
        with self.open(mode="r", encoding=encoding, errors=errors) as f:
            return f.read()

    def write_bytes(self, data):
        view = memoryview(data)
        with self.open(mode="wb") as f:
            return f.write(view)

    def write_text(self, data, encoding=None, errors=None):
        if not isinstance(data, str):
            raise TypeError(f"data must be str, not {type(data).__name__}")
        with self.open(mode="w", encoding=encoding, errors=errors) as f:
            return f.write(data)

    def _raw_open(self, flags, mode=0o777):
        return os.open(self, flags, mode)

    def touch(self, mode=0o666, exist_ok=True):
        if exist_ok:
            try:
                os.utime(self, None)
            except OSError:
                pass
            else:
                return
        flags = os.O_CREAT | os.O_WRONLY
        if not exist_ok:
            flags |= os.O_EXCL
        fd = self._raw_open(flags, mode)
        os.close(fd)

    @classmethod
    def cwd(cls):
        """Return a new GoudaPath pointing to the current working directory using :func:`os.getcwd`."""
        return cls(os.getcwd(), use_absolute=False)

    @classmethod
    def home(cls) -> "GoudaPath":
        """Return a new GoudaPath pointing to the user's home directory using :func:`os.path.expanduser`."""
        return cls(os.path.expanduser("~"))

    def normpath(self: TGoudaPath) -> "GoudaPath":
        return GoudaPath(os.path.normpath(self))

    def expanduser(self: TGoudaPath) -> "GoudaPath":
        return GoudaPath(os.path.expanduser(self.path))

    def _clear_cache(self: TGoudaPath) -> None:
        if hasattr(self, "_parts_normcase_cached"):
            del self._parts_normcase_cached
        if hasattr(self, "_hash"):
            del self._hash
        if hasattr(self, "_parsed_parts"):
            del self._parsed_parts

    @property
    def _cparts(self: TGoudaPath):
        try:
            return self._parts_normcase_cached
        except AttributeError:
            self._parts_normcase_cached = pathlib.Path(os.path.normcase(self.__path)).parts
            return self._parts_normcase_cached

    def __hash__(self: TGoudaPath):
        try:
            return self._hash
        except AttributeError:
            self._hash = hash(tuple(self._cparts))
            return self._hash

    def __eq__(self: TGoudaPath, other: GPathLike) -> bool:
        """Check if the paths are equal.

        Note
        ----
        Compares case-folded paths if both are GoudaPaths, otherwise compares the string paths
        """
        if isinstance(other, (GoudaPath, pathlib.Path)):
            return self._cparts == _get_path_parts_normcase(other)
        return self.__path == str(other)

    def __lt__(self: TGoudaPath, other: GPathLike) -> bool:
        if isinstance(other, (GoudaPath, pathlib.Path)):
            return self._cparts < _get_path_parts_normcase(other)
        else:
            return self.__path < str(other)

    def __le__(self: TGoudaPath, other: GPathLike) -> bool:
        if isinstance(other, (GoudaPath, pathlib.Path)):
            return self._cparts <= _get_path_parts_normcase(other)
        else:
            return self.__path <= str(other)

    def __gt__(self: TGoudaPath, other: GPathLike) -> bool:
        if isinstance(other, (GoudaPath, pathlib.Path)):
            return self._cparts > _get_path_parts_normcase(other)
        else:
            return self.__path > str(other)

    def __ge__(self: TGoudaPath, other: GPathLike) -> bool:
        if isinstance(other, (GoudaPath, pathlib.Path)):
            return self._cparts >= _get_path_parts_normcase(other)
        else:
            return self.__path >= str(other)


def _get_path_parts_normcase(other: pathlib.Path) -> tuple[str]:
    if hasattr(other, "_cparts"):  # Python 3.9
        return other._cparts
    elif hasattr(other, "_parts_normcase"):  # Python 3.12
        return other._parts_normcase
    else:
        raise NotImplementedError("Cannot get casefolded/normed parts from Path object")
