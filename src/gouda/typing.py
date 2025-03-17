"""Generic types used in Gouda."""

from collections.abc import Sequence
from typing import TypeVar

import numpy as np
import numpy.typing as npt

ShapeType = TypeVar("ShapeType", int, Sequence[int])
FloatArrayType = TypeVar("FloatArrayType", float, np.floating, npt.NDArray[np.floating])
LabelArrayType = TypeVar("LabelArrayType", bool, int, npt.NDArray[np.integer])
ImageArrayType = TypeVar("ImageArrayType", npt.NDArray[np.uint8], npt.NDArray[np.uint16])
ColorType = TypeVar("ColorType", str, float, int, tuple[float, float, float], tuple[int, int, int])
