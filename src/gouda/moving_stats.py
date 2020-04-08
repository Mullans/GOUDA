"""Classes to handle constant-time mean/stddev updates"""
import numpy as np


class MMean(object):
    """Class to hold a moving mean with constant-time update and memory."""

    def __init__(self):
        self._mean = 0.0
        self._count = 0

    @property
    def count(self):
        """The count of items included in the mean"""
        return self._count

    def __iadd__(self, value):
        """Update the mean, including the given value."""
        self._count += 1
        self._mean += (1.0 / self._count) * (value - self._mean)
        return self

    def __add__(self, value):
        """Add a new value to the mean, does not update class values."""
        return self._mean + value

    def __sub__(self, value):
        """Subtract a new value from the mean, does not update class values."""
        return self._mean - value

    def __str__(self):
        """Return the mean as a string."""
        return str(self._mean)

    def __call__(self):
        """Alternate method for self.mean()"""
        return self._mean

    def mean(self):
        """Return the mean."""
        return self._mean


class MStddev(object):
    """Class to hold a moving standard deviation with constant-time update and memory."""

    def __init__(self):
        self._count = 0.0
        self._mean = 0.0
        self._variance = 0.0

    @property
    def count(self):
        """The count of items included in the standard deviation"""
        return self._count

    def __iadd__(self, value):
        """Update the mean and stddev, including the new value."""
        self._count += 1
        prev_mean = self._mean
        self._mean += (1.0 / self._count) * (value - self._mean)
        self._variance += (value - self._mean) * (value - prev_mean)
        return self

    def __add__(self, value):
        """Add a value to the stddev, does not update class values."""
        return self.stddev() + value

    def __call__(self):
        """Alternate method for self.stddev"""
        return self.stddev()

    def __sub__(self, value):
        """Subtract a value from the stddev, does not update class values."""
        return self.stddev() - value

    def __str__(self):
        """Return the stddev as a string."""
        if self._count == 0:
            return str(0)
        stddev = np.sqrt(self._variance / self._count)
        return str(stddev)

    def mean(self):
        """Return the mean"""
        return self._mean

    def stddev(self):
        """Return the current stddev."""
        if self._count == 0:
            return 0
        return np.sqrt(self._variance / self._count)


class MMeanArray(object):
    """Class to hold an array of element-wise independent means that update in constant-time and memory.

    Note
    ----
        Value shape must be the same or broadcastable to the shape of the
        mean array for all operations.
    """
    def __init__(self, shape, dtype=np.float):
        self._mean = np.zeros(shape, dtype=dtype)
        self._count = 0

    @property
    def shape(self):
        """The shape of the array"""
        return self._mean.shape

    @property
    def dtype(self):
        """The type of data stored in the array"""
        return self._mean.dtype

    @property
    def count(self):
        """The number of examples used for the mean of each item in the array"""
        return self._count

    def __iadd__(self, value):
        """Update the _mean, including the given value."""
        if value.shape != self.shape:
            raise ValueError('Input values must have the same shape as the MMeanArray')
        self._count += 1
        self._mean += (value - self._mean) * (1.0 / self._count)
        return self

    def __add__(self, value):
        """Add a new value to the mean, does not update class values."""
        return self._mean + value

    def __sub__(self, value):
        """Subtract a new value from the mean, does not update class values."""
        return self._mean - value

    def __str__(self):
        """Return the _mean as a string."""
        return str(self._mean)

    def __call__(self):
        """Alternate method for self.mean()"""
        return self._mean

    def mean(self):
        """Return the _mean."""
        return self._mean


class MStddevArray(object):
    """Class to hold an array of element-wise independent standard deviations that update in constant-time and memory.

    Note
    ----
        Value shape must be the same or broadcastable to the shape of the
        mean array for all operations.
    """

    def __init__(self, shape, dtype=np.float):
        self._mean = np.zeros(shape, dtype=dtype)
        self._variance = np.zeros(shape, dtype=dtype)
        self._count = 0

    @property
    def shape(self):
        """The shape of the array"""
        return self._variance.shape

    @property
    def dtype(self):
        """The type of data stored in the array"""
        return self._mean.dtype

    @property
    def count(self):
        """The number of examples used for the standard deviation of each item in the array"""
        return self._count

    def mean(self):
        """Return the mean of the array"""
        return self._mean

    def variance(self):
        """Return the variance of the array"""
        if self._count == 0:
            return np.zeros_like(self._variance)
        return self._variance / self._count

    def __iadd__(self, value):
        """Update the mean and stddev, including the new value."""
        if value.shape != self.shape:
            raise ValueError('Input values must have the same shape as the MStddevArray')
        self._count += 1
        prev_mean = np.copy(self._mean)
        self._mean += (1.0 / self._count) * (value - self._mean)
        self._variance += (value - self._mean) * (value - prev_mean)
        return self

    def __add__(self, value):
        """Add a value to the stddev, does not update class values."""
        return self.stddev() + value

    def __sub__(self, value):
        """Subtract a value from the stddev, does not update class values."""
        return self.stddev() - value

    def __str__(self):
        """Return the stddev as a string."""
        return str(self.stddev())

    def __call__(self):
        """Alternate method for self.stddev()"""
        return self.stddev()

    def stddev(self):
        """Return the current stddev."""
        if self._count == 0:
            return np.zeros_like(self._variance)
        return np.sqrt(self._variance / self._count)
