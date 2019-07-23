import pytest

import numpy as np

import gouda


def test_MMean():
    item = gouda.MMean()
    assert item._mean == 0
    assert item.count == 0

    item += 3
    assert item._mean == 3
    assert item.val() == item._mean
    assert item.count == 1
    item += 1
    assert item._mean == (3 + 1) / 2
    assert item.val() == item._mean
    assert item.count == 2
    assert str(item) == str(2.0)

    assert item + 2 == 4
    assert item - 2 == 0


def test_MStddev():
    item = gouda.MStddev()
    assert item.count == 0
    assert item._mean == 0
    assert item.val() == 0
    assert str(item) == '0'

    item += 3
    assert item._mean == 3
    assert item.count == 1
    assert item.val() == np.std([3])

    item += 4
    item += 8
    assert item._mean == np.mean([3, 4, 8])
    assert item.count == 3
    assert item.val() == np.std([3, 4, 8])

    assert item + 1 == item.val() + 1
    assert item - 1 == item.val() - 1

    assert str(item) == str(item.val())


def test_MMeanArray():
    item = gouda.MMeanArray([3, 3])
    np.testing.assert_array_equal(item._mean, np.zeros([3, 3]))
    assert item.count == 0
    assert item.shape == (3, 3)
    assert item.dtype == 'float'

    item += np.ones([3, 3])
    np.testing.assert_array_equal(item._mean, np.ones([3, 3]))
    assert item.count == 1

    with pytest.raises(ValueError):
        assert item.__iadd__(np.ones([4, 3]))

    np.testing.assert_array_equal(item + 1, np.ones([3, 3]) * 2)
    np.testing.assert_array_equal(item - 1, np.zeros([3, 3]))

    assert str(item) == str(np.ones([3, 3]))

    np.testing.assert_array_equal(item.val(), np.ones([3, 3]))
    item += np.ones([3, 3]) * 3
    item += np.ones([3, 3]) * 16
    np.testing.assert_almost_equal(item.val()[0, 0], np.mean([1, 3, 16]))
    np.testing.assert_almost_equal(item.val()[-1, -1], np.mean([1, 3, 16]))


def test_MStddevArray():
    item = gouda.MStddevArray([3, 3])
    assert item.count == 0
    assert item.shape == (3, 3)
    assert item.dtype == 'float'
    np.testing.assert_array_equal(item.val(), np.zeros([3, 3]))

    item += np.ones([3, 3])
    np.testing.assert_array_equal(item.val(), np.zeros([3, 3]))
    assert item.count == 1
    assert str(item) == str(np.zeros([3, 3]))

    with pytest.raises(ValueError):
        assert item.__iadd__(np.ones([4, 3]))

    np.testing.assert_array_equal(item + 1, np.ones([3, 3]))
    np.testing.assert_array_equal(item - 1, np.ones([3, 3]) * -1)

    item += np.ones([3, 3]) * 3
    item += np.ones([3, 3]) * 16
    np.testing.assert_almost_equal(item.val()[0, 0], np.std([1, 3, 16]))
