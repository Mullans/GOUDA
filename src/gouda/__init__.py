# -*- coding: utf-8 -*-
"""Miscellaneous data utilities."""
from pkg_resources import DistributionNotFound, get_distribution

from .binaryconfusionmatrix import BinaryConfusionMatrix  # noqa: F401
from .confusionmatrix import ConfusionMatrix  # noqa: F401
from .constants import *  # noqa: F401, F403
from .data_methods import *  # noqa: F401, F403
from .display import print_grid, print_image  # noqa: F401
from .file_methods import *  # noqa: F401, F403
from .general import *  # noqa: F401, F403
from .goudapath import GoudaPath  # noqa: F401
from .moving_stats import *  # noqa: F401, F403

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = 'GOUDA'
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:  # pragma: no cover
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound
