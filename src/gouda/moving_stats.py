"""Classes to handle constant-time mean/stddev updates."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from gouda.typing import ShapeType


class ParallelStats:
    """Object for aggregation of stats across multiple arrays/values.

    Parameters
    ----------
    stabilize : bool
        Should a potentially more stable method be used to update the mean

    Note
    ----------
    Stabilize should be set to True for cases where self._count is rougly equal to the incoming count and both are large. In practice, there did not seem to be much error (<1e-10) even when repeatedly using arrays with 100k values each.

    The algorithm used is adapted from:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """

    def __init__(self, stabilize: bool = False) -> None:
        self._count = 0
        self._mean = 0.0
        self._ssd = 0.0  # sum of square differences
        self._min = float("inf")
        self._max = float("-inf")
        self.stabilize = stabilize

    def __add__(self, value: ParallelStats) -> ParallelStats:
        """Add two ParallelStats objects together."""
        stats_a = self.copy()
        stats_a += value
        return stats_a

    def __iadd__(self, value: Union[ParallelStats, npt.ArrayLike]) -> ParallelStats:
        """Update the stats with the given value or ParallelStats object."""
        count_a = self._count
        mean_a = self._mean
        ssd_a = self._ssd
        if isinstance(value, ParallelStats):
            count_b = value.count()
            mean_b = value.mean()
            ssd_b = value.ssd()
        else:
            value = np.asarray(value)
            count_b = value.size
            mean_b = value.mean()
            ssd_b = ((value - mean_b) ** 2).sum()

        self._min = min(self._min, value.min())
        self._max = max(self._max, value.max())

        self._count = count_a + count_b
        count_denom = 1 / self._count
        delta = mean_b - mean_a
        if self.stabilize:
            self._mean = (count_a * mean_a + count_b * mean_b) / self._count
        else:
            self._mean = mean_a + delta * count_b * count_denom
        self._ssd = ssd_a + ssd_b + delta**2 * count_a * count_b * count_denom
        return self

    def __call__(
        self, value: Union[int, float, ParallelStats, npt.ArrayLike], stabilize: Optional[bool] = None
    ) -> None:
        """Wrap the __iadd__ method to allow for a more natural interface."""
        if stabilize is not None:
            self.stabilize = stabilize
        self += value

    def copy(self) -> ParallelStats:
        """Return a copy of the current stats."""
        stats_copy = ParallelStats()
        stats_copy.copy_stats(self)
        return stats_copy

    def copy_stats(self, stats: ParallelStats) -> None:
        """Copy the stats from another ParallelStats object."""
        self._count = stats.count()
        self._mean = stats.mean()
        self._ssd = stats.ssd()
        self._min = stats.min()
        self._max = stats.max()
        self.stabilize = stats.stabilize

    def count(self) -> int:
        """Return the count of items included in the stats."""
        return self._count

    def mean(self) -> float:
        """Return the mean."""
        return self._mean

    def summary(self) -> None:  # pragma: no cover
        """Print a summary of the stats."""
        print(f"Count: {self._count:d}, Min: {self._min:.2f}, Max: {self._max:.2f}")
        print(f"Mean: {self._mean:.4f}, Std: {self.std():.4f}, Var: {self.var():.4f}")

    def ssd(self) -> float:
        """Return the sum of squared differences."""
        return self._ssd

    def std(self, sample_std: bool = False) -> float:
        """Return the standard deviation.

        Parameters
        ----------
        sample_std : bool
            Whether to return the sample standard deviation (use n-1 in the denominator)
        """
        if self._count <= 1:
            return 0
        return float(np.sqrt(self.var(sample_variance=sample_std)))

    def var(self, sample_variance: bool = False) -> float:
        """Return the variance.

        Parameters
        ----------
        sample_variance : bool
            Whether to return the sample variance (use n-1 in the denominator)
        """
        if self._count <= 1:
            return 0
        if sample_variance:
            return self._ssd / (self._count - 1)
        return self._ssd / self._count

    def min(self) -> Union[float, int]:
        """Return the minimum value."""
        return self._min

    def max(self) -> Union[float, int]:
        """Return the maximum value."""
        return self._max


class MMean:
    """Class to hold a moving mean with constant-time update and memory."""

    def __init__(self) -> None:
        self._mean = 0.0
        self._count = 0

    @property
    def count(self) -> int:
        """The count of items included in the mean."""
        return self._count

    def __iadd__(self, value: Union[int, float]) -> MMean:
        """Update the mean, including the given value."""
        self._count += 1
        self._mean += (1.0 / self._count) * (value - self._mean)
        return self

    def __add__(self, value: Union[int, float]) -> float:
        """Add a new value to the mean, does not update class values."""
        return self._mean + value

    def __sub__(self, value: Union[int, float]) -> float:
        """Subtract a new value from the mean, does not update class values."""
        return self._mean - value

    def __str__(self) -> str:
        """Return the mean as a string."""
        return str(self._mean)

    def __call__(self) -> float:
        """Alternate method for self.mean()."""
        return self._mean

    def mean(self) -> float:
        """Return the mean."""
        return self._mean


class MStddev:
    """Class to hold a moving standard deviation with constant-time update and memory."""

    def __init__(self) -> None:
        self._count: int = 0
        self._mean: float = 0.0
        self._variance: float = 0.0

    @property
    def count(self) -> int:
        """The count of items included in the standard deviation."""
        return self._count

    def __iadd__(self, value: Union[int, float]) -> MStddev:
        """Update the mean and stddev, including the new value."""
        self._count += 1
        prev_mean = self._mean
        self._mean += (1.0 / self._count) * (value - self._mean)
        self._variance += (value - self._mean) * (value - prev_mean)
        return self

    def __add__(self, value: Union[int, float]) -> float:
        """Add a value to the stddev, does not update class values."""
        return self.stddev() + value

    def __call__(self) -> float:
        """Alternate method for self.stddev."""
        return self.stddev()

    def __sub__(self, value: Union[int, float]) -> float:
        """Subtract a value from the stddev, does not update class values."""
        return self.stddev() - value

    def __str__(self) -> str:
        """Return the stddev as a string."""
        if self._count == 0:
            return str(0)
        stddev = np.sqrt(self._variance / self._count)
        return str(stddev)

    def mean(self) -> float:
        """Return the mean."""
        return self._mean

    def stddev(self) -> float:
        """Return the current stddev."""
        if self._count == 0:
            return 0
        return float(np.sqrt(self._variance / self._count))


class MMeanArray:
    """Class to hold an array of element-wise independent means that update in constant-time and memory.

    Note
    ----
        Value shape must be the same or broadcastable to the shape of the
        mean array for all operations.
    """

    def __init__(self, shape: ShapeType, dtype: npt.DTypeLike = float) -> None:
        self._mean: npt.NDArray[np.floating] = np.zeros(shape, dtype=dtype)
        self._count: int = 0

    @property
    def shape(self) -> ShapeType:
        """The shape of the array."""
        shape: tuple[int, ...] = self._mean.shape
        return shape

    @property
    def dtype(self) -> npt.DTypeLike:
        """The type of data stored in the array."""
        return self._mean.dtype

    @property
    def count(self) -> int:
        """The number of examples used for the mean of each item in the array."""
        return self._count

    def __iadd__(self, value: npt.NDArray[np.floating]) -> MMeanArray:
        """Update the mean, including the given value."""
        if value.shape != self.shape:
            raise ValueError("Input values must have the same shape as the MMeanArray")
        self._count += 1
        self._mean += (value - self._mean) * (1.0 / self._count)
        return self

    def __add__(self, value: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Add a new value to the mean, does not update class values."""
        result: npt.NDArray[np.floating] = self._mean + value
        return result

    def __sub__(self, value: npt.NDArray) -> npt.NDArray[np.floating]:
        """Subtract a new value from the mean, does not update class values."""
        result: npt.NDArray[np.floating] = self._mean - value
        return result

    def __str__(self) -> str:
        """Return the mean as a string."""
        return str(self._mean)

    def __call__(self) -> npt.NDArray[np.floating]:
        """Alternate method for self.mean()."""
        return self._mean

    def mean(self) -> npt.NDArray[np.floating]:
        """Return the mean."""
        return self._mean


class MStddevArray:
    """Class to hold an array of element-wise independent standard deviations that update in constant-time and memory.

    Note
    ----
        Value shape must be the same or broadcastable to the shape of the
        mean array for all operations.
    """

    def __init__(self, shape: ShapeType, dtype: npt.DTypeLike = float) -> None:
        self._mean = np.zeros(shape, dtype=dtype)
        self._variance = np.zeros(shape, dtype=dtype)
        self._count = 0

    @property
    def shape(self) -> ShapeType:
        """The shape of the array."""
        shape: tuple[int, ...] = self._variance.shape
        return shape

    @property
    def dtype(self) -> npt.DTypeLike:
        """The type of data stored in the array."""
        return self._mean.dtype

    @property
    def count(self) -> int:
        """The number of examples used for the standard deviation of each item in the array."""
        return self._count

    def mean(self) -> npt.NDArray[np.floating]:
        """Return the mean of the array."""
        return self._mean

    def variance(self) -> npt.NDArray[np.floating]:
        """Return the variance of the array."""
        if self._count == 0:
            return np.zeros_like(self._variance)
        return self._variance / self._count

    def __iadd__(self, value: npt.NDArray[np.floating]) -> MStddevArray:
        """Update the mean and stddev, including the new value."""
        if value.shape != self.shape:
            raise ValueError("Input values must have the same shape as the MStddevArray")
        self._count += 1
        prev_mean = np.copy(self._mean)
        self._mean += (1.0 / self._count) * (value - self._mean)
        self._variance += (value - self._mean) * (value - prev_mean)
        return self

    def __add__(self, value: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Add a value to the stddev, does not update class values."""
        result: npt.NDArray[np.floating] = self.stddev() + value
        return result

    def __sub__(self, value: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Subtract a value from the stddev, does not update class values."""
        result: npt.NDArray[np.floating] = self.stddev() - value
        return result

    def __str__(self) -> str:
        """Return the stddev as a string."""
        return str(self.stddev())

    def __call__(self) -> npt.NDArray[np.floating]:
        """Alternate method for self.stddev()."""
        return self.stddev()

    def stddev(self) -> npt.NDArray[np.floating]:
        """Return the current stddev."""
        if self._count == 0:
            return np.zeros_like(self._variance)
        result: npt.NDArray[np.floating] = np.sqrt(self._variance / self._count)
        return result
