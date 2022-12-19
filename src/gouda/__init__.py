import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "GOUDA"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError


from gouda.binaryconfusionmatrix import BinaryConfusionMatrix  # noqa: F401
from gouda.confusionmatrix import ConfusionMatrix  # noqa: F401
from gouda.constants import *  # noqa: F401, F403
from gouda.data_methods import *  # noqa: F401, F403
from gouda.display import print_grid, print_image  # noqa: F401
from gouda.file_methods import *  # noqa: F401, F403
from gouda.general import *  # noqa: F401, F403
from gouda.goudapath import GoudaPath  # noqa: F401
from gouda.moving_stats import *  # noqa: F401, F403
