import numpy as np
import numpy.typing as npt
from typing import Tuple, Union

ShapeType = Union[int, Tuple[int, ...]]
FloatArrayType = Union[float, np.floating, npt.NDArray[np.floating]]
LabelArrayType = Union[bool, int, npt.NDArray[np.integer], npt.NDArray[np.bool_]]
ImageArrayType = Union[npt.NDArray[np.uint8], npt.NDArray[np.uint16]]

ColorType = Union[str, float, int, tuple[float, float, float]]
