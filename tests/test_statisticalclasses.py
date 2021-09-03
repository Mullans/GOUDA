import pytest

import numpy as np

import gouda


def test_MMean():
    item = gouda.MMean()
    assert item._mean == 0
    assert item.count == 0
    assert item() == item.mean()

    item += 3
    assert item._mean == 3
    assert item.mean() == item._mean
    assert item.count == 1
    item += 1
    assert item._mean == (3 + 1) / 2
    assert item.mean() == item._mean
    assert item.count == 2
    assert str(item) == str(2.0)
    assert item() == item.mean()

    assert item + 2 == 4
    assert item - 2 == 0


def test_MStddev():
    item = gouda.MStddev()
    assert item.count == 0
    assert item._mean == 0
    assert item.mean() == 0
    assert item.stddev() == 0
    assert str(item) == '0'
    assert item() == item.stddev()

    item += 3
    assert item._mean == 3
    assert item.count == 1
    assert item.stddev() == np.std([3])

    item += 4
    item += 8
    assert item._mean == np.mean([3, 4, 8])
    assert item.count == 3
    assert item.stddev() == np.std([3, 4, 8])
    assert item.mean() == np.mean([3, 4, 8])
    assert item() == item.stddev()

    assert item + 1 == item.stddev() + 1
    assert item - 1 == item.stddev() - 1

    assert str(item) == str(item.stddev())


def test_MMeanArray():
    item = gouda.MMeanArray([3, 3])
    np.testing.assert_array_equal(item._mean, np.zeros([3, 3]))
    assert item.count == 0
    assert item.shape == (3, 3)
    assert item.dtype == 'float'
    np.testing.assert_array_equal(item(), item.mean())

    item += np.ones([3, 3])
    np.testing.assert_array_equal(item._mean, np.ones([3, 3]))
    assert item.count == 1

    with pytest.raises(ValueError):
        assert item.__iadd__(np.ones([4, 3]))

    np.testing.assert_array_equal(item + 1, np.ones([3, 3]) * 2)
    np.testing.assert_array_equal(item - 1, np.zeros([3, 3]))

    assert str(item) == str(np.ones([3, 3]))

    np.testing.assert_array_equal(item.mean(), np.ones([3, 3]))
    item += np.ones([3, 3]) * 3
    item += np.ones([3, 3]) * 16
    np.testing.assert_almost_equal(item.mean()[0, 0], np.mean([1, 3, 16]))
    np.testing.assert_almost_equal(item.mean()[-1, -1], np.mean([1, 3, 16]))
    np.testing.assert_array_equal(item(), item.mean())


def test_MStddevArray():
    item = gouda.MStddevArray([3, 3])
    assert item.count == 0
    assert item.shape == (3, 3)
    assert item.dtype == 'float'
    np.testing.assert_array_equal(item.stddev(), np.zeros([3, 3]))
    np.testing.assert_array_equal(item.mean(), np.zeros([3, 3]))
    np.testing.assert_array_equal(item.variance(), np.zeros([3, 3]))
    np.testing.assert_array_equal(item(), item.stddev())

    item += np.ones([3, 3])
    np.testing.assert_array_equal(item.stddev(), np.zeros([3, 3]))
    assert item.count == 1
    assert str(item) == str(np.zeros([3, 3]))

    with pytest.raises(ValueError):
        assert item.__iadd__(np.ones([4, 3]))

    np.testing.assert_array_equal(item + 1, np.ones([3, 3]))
    np.testing.assert_array_equal(item - 1, np.ones([3, 3]) * -1)

    item += np.ones([3, 3]) * 3
    item += np.ones([3, 3]) * 16
    np.testing.assert_almost_equal(item.stddev()[0, 0], np.std([1, 3, 16]))
    np.testing.assert_almost_equal(item.variance()[0, 0], np.var([1, 3, 16]))
    np.testing.assert_almost_equal(item.mean()[0, 0], np.mean([1, 3, 16]))
    np.testing.assert_array_equal(item(), item.stddev())


def test_ParallelStats():
    to_use = np.random.randint(0, 1000, 100000)
    stats = gouda.ParallelStats(stabilize=False)
    assert stats.count() == 0
    assert stats.mean() == 0
    assert stats.ssd() == 0
    assert stats.std() == 0
    assert stats.var() == 0
    stats += to_use
    assert stats.count() == to_use.size
    np.testing.assert_allclose(stats.mean(), np.mean(to_use), atol=1e-10, rtol=0)
    np.testing.assert_allclose(stats.ssd(), ((to_use - np.mean(to_use)) ** 2).sum(), atol=1e-10, rtol=0)
    np.testing.assert_allclose(stats.std(), np.std(to_use), atol=1e-10, rtol=0)
    np.testing.assert_allclose(stats.var(), np.var(to_use), atol=1e-10, rtol=0)
    np.testing.assert_allclose(stats.min(), np.min(to_use), atol=1e-10)
    np.testing.assert_allclose(stats.max(), np.max(to_use), atol=1e-10)

    to_use = np.random.randint(0, 1000, 100000)
    stats = gouda.ParallelStats(stabilize=False)
    stats += to_use[:100]
    stats += to_use[100:5000]
    stats(to_use[5000:10000])
    stats += to_use[10000:]
    assert stats.count() == to_use.size
    np.testing.assert_allclose(stats.mean(), np.mean(to_use), atol=1e-10, rtol=0)
    np.testing.assert_allclose(stats.ssd(), ((to_use - np.mean(to_use)) ** 2).sum(), rtol=1e-10)
    np.testing.assert_allclose(stats.std(), np.std(to_use), atol=1e-10, rtol=0)
    np.testing.assert_allclose(stats.var(), np.var(to_use), atol=1e-10, rtol=0)
    np.testing.assert_allclose(stats.min(), np.min(to_use), atol=1e-10)
    np.testing.assert_allclose(stats.max(), np.max(to_use), atol=1e-10)

    new_stats = stats.copy()
    assert stats.count() == to_use.size
    np.testing.assert_allclose(stats.mean(), new_stats.mean(), atol=1e-10, rtol=0)
    np.testing.assert_allclose(stats.ssd(), new_stats.ssd(), rtol=1e-10)
    np.testing.assert_allclose(stats.std(), new_stats.std(), atol=1e-10, rtol=0)
    np.testing.assert_allclose(stats.var(), new_stats.var(), atol=1e-10, rtol=0)
    np.testing.assert_allclose(stats.min(), new_stats.min(), atol=1e-10)
    np.testing.assert_allclose(stats.max(), new_stats.max(), atol=1e-10)

    to_use = np.random.randint(0, 1000, 100000)
    stats = gouda.ParallelStats(stabilize=True)
    assert stats.count() == 0
    assert stats.mean() == 0
    assert stats.ssd() == 0
    assert stats.std() == 0
    assert stats.var() == 0
    stats += to_use
    assert stats.count() == to_use.size
    np.testing.assert_allclose(stats.mean(), np.mean(to_use), atol=1e-10, rtol=0)
    np.testing.assert_allclose(stats.ssd(), ((to_use - np.mean(to_use)) ** 2).sum(), rtol=1e-10)
    np.testing.assert_allclose(stats.std(), np.std(to_use), atol=1e-10, rtol=0)
    np.testing.assert_allclose(stats.var(), np.var(to_use), atol=1e-10, rtol=0)
    np.testing.assert_allclose(stats.min(), np.min(to_use), atol=1e-10)
    np.testing.assert_allclose(stats.max(), np.max(to_use), atol=1e-10)

    to_use2 = np.random.randint(0, 1000, 100000)
    stats2 = gouda.ParallelStats(stabilize=True)
    stats2 += to_use2[:100]
    stats2(to_use2[100:5000])
    stats2 += to_use2[5000:10000]
    stats2 += to_use2[10000:]
    assert stats2.count() == to_use2.size
    np.testing.assert_allclose(stats2.mean(), np.mean(to_use2), atol=1e-10, rtol=0)
    np.testing.assert_allclose(stats2.ssd(), ((to_use2 - np.mean(to_use2)) ** 2).sum(), rtol=1e-10)
    np.testing.assert_allclose(stats2.std(), np.std(to_use2), atol=1e-10, rtol=0)
    np.testing.assert_allclose(stats2.var(), np.var(to_use2), atol=1e-10, rtol=0)
    np.testing.assert_allclose(stats2.min(), np.min(to_use2), atol=1e-10)
    np.testing.assert_allclose(stats2.max(), np.max(to_use2), atol=1e-10)

    stats3 = stats + stats2
    to_use3 = np.concatenate([to_use, to_use2])
    assert stats3.count() == to_use3.size
    np.testing.assert_allclose(stats3.mean(), np.mean(to_use3), atol=1e-10, rtol=0)
    np.testing.assert_allclose(stats3.ssd(), ((to_use3 - np.mean(to_use3)) ** 2).sum(), rtol=1e-10)
    np.testing.assert_allclose(stats3.std(), np.std(to_use3), atol=1e-10, rtol=0)
    np.testing.assert_allclose(stats3.var(), np.var(to_use3), atol=1e-10, rtol=0)
    np.testing.assert_allclose(stats3.min(), np.min(to_use3), atol=1e-10)
    np.testing.assert_allclose(stats3.max(), np.max(to_use3), atol=1e-10)
