"""Generic types used in Gouda."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack  # noqa: F401

ShapeType = int | tuple[int]
"""Type for numpy shape/axis values (e.g. int or tuple of ints)"""
FloatArrayType = float | np.floating | npt.NDArray[np.floating]
"""Type for bare floats or numpy floating point arrays"""
LabelArrayType = bool | int | np.integer | npt.NDArray[np.integer]
"""Type for bare labels (e.g. bool, int, np.integer) or label arrays (e.g. numpy integer arrays)"""
NumberType = int | float | np.integer | np.floating
"""Type for non-complex numbers (e.g. int, float, np.integer, np.floating)"""
ImageArrayType = npt.NDArray[np.uint8] | npt.NDArray[np.uint16]
"""Type for numpy image arrays (e.g. uint8 or uint16)"""
ImageLikeType = npt.ArrayLike | dict[str, str | int | float | npt.ArrayLike] | None
"""Typing for image-like objects used in :func:`gouda.display.print_grid`"""
FullColorType = Sequence[float | int] | npt.NDArray[np.integer] | npt.NDArray[np.floating]
"""Type for full color values (e.g. RGB, RGBA, etc.)"""
ColorType = str | float | int | FullColorType
"""Type for color values (e.g. str, float, int, or full color values)"""
