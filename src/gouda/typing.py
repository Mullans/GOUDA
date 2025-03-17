"""Generic types used in Gouda."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack  # noqa: F401

ShapeType = int | Sequence[int]
FloatArrayType = float | np.floating | npt.NDArray[np.floating]
LabelArrayType = bool | int | npt.NDArray[np.integer]
ImageArrayType = npt.NDArray[np.uint8] | npt.NDArray[np.uint16]
"""Type for numpy image arrays (e.g. uint8 or uint16)"""
FullColorType = Sequence[float | int] | npt.NDArray[np.integer] | npt.NDArray[np.floating]
"""Type for full color values (e.g. RGB, RGBA, etc.)"""
ColorType = str | float | int | FullColorType
"""Type for color values (e.g. str, float, int, or full color values)"""
