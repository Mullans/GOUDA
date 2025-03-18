"""Utilities for data science and machine learning."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "gouda"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError


from gouda.binaryconfusionmatrix import BinaryConfusionMatrix
from gouda.color_lists import find_color_hex
from gouda.confusionmatrix import ConfusionMatrix
from gouda.constants import *
from gouda.data_methods import *
from gouda.display import print_grid, print_image, squarify
from gouda.file_methods import *
from gouda.general import *
from gouda.goudapath import GoudaPath
from gouda.moving_stats import *
