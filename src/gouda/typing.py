"""Generic types used in Gouda."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as np
import numpy.typing as npt

ShapeType = Union[int, tuple[int, ...]]
"""Type for numpy shape/axis values (e.g. int or tuple of ints)"""
FloatArrayType = Union[float, np.floating, npt.NDArray[np.floating]]
"""Type for bare floats or numpy floating point arrays"""
LabelArrayType = Union[bool, int, np.integer, npt.NDArray[np.integer]]
"""Type for bare labels (e.g. bool, int, np.integer) or label arrays (e.g. numpy integer arrays)"""
NumberType = Union[int, float, np.integer, np.floating]
"""Type for non-complex numbers (e.g. int, float, np.integer, np.floating)"""
ImageArrayType = Union[npt.NDArray[np.uint8], npt.NDArray[np.uint16]]
"""Type for numpy image arrays (e.g. uint8 or uint16)"""
ImageLikeType = Union[npt.ArrayLike, dict[str, str, int, float, npt.ArrayLike], None]
"""Typing for image-like objects used in :func:`gouda.display.print_grid`"""
ColorType = Union[Sequence[float, int], npt.NDArray[np.integer], npt.NDArray[np.floating]]
"""Type for full color values (e.g. str, RGB, RGBA, etc.)"""
