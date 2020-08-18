import pytest

import numpy as np

import gouda


def test_num_digits():
    assert gouda.num_digits(5) == 1
    assert gouda.num_digits(9) == 1
    assert gouda.num_digits(10) == 2
    assert gouda.num_digits(99) == 2
    assert gouda.num_digits(100) == 3
    assert gouda.num_digits(0) == 1
    assert gouda.num_digits(-1) == 1
    assert gouda.num_digits(-10) == 2


def test_arr_sample():
    temp_data = np.arange(1, 5)
    resampled = gouda.arr_sample(temp_data, 2)
    np.testing.assert_array_equal(resampled, np.array([1, 3]))
    resampled2 = gouda.arr_sample(temp_data, 0.5)
    np.testing.assert_array_equal(resampled2, np.array([1, 1, 2, 2, 3, 3, 4, 4]))


def test_arr_sample_exception():
    temp_data = np.arange(1, 5).reshape([2, 2])
    with pytest.raises(ValueError):
        assert gouda.arr_sample(temp_data, 1)


def test_sigmoid():
    np.testing.assert_almost_equal(gouda.sigmoid(0), 0.5)
    np.testing.assert_almost_equal(gouda.sigmoid(-100), 0)
    np.testing.assert_almost_equal(gouda.sigmoid(100), 1)


def test_normalize():
    test_data = np.arange(1000).reshape([10, 10, 10])
    normed_1 = gouda.normalize(test_data)
    np.testing.assert_almost_equal(normed_1.std(), 1)
    assert normed_1.mean() == 0

    normed_2 = gouda.normalize(test_data, axis=1)
    np.testing.assert_array_almost_equal(normed_2.mean(axis=1), np.zeros([10, 10]))
    np.testing.assert_equal(normed_2.std(axis=1), np.ones([10, 10]))

    normed_3 = gouda.normalize(test_data, axis=(0, 1))
    np.testing.assert_array_almost_equal(normed_3.mean(axis=(0, 1)), np.zeros([10,]))
    np.testing.assert_equal(normed_3.std(axis=(0, 1)), 1)


def test_rescale():
    test_data = np.arange(100).reshape([10, 10])
    scaled_1 = gouda.rescale(test_data, new_min=0, new_max=1, axis=1)
    manual = (test_data - test_data.min(axis=1, keepdims=True)) / (test_data.max(axis=1, keepdims=True) - test_data.min(axis=1, keepdims=True))
    np.testing.assert_array_equal(scaled_1, manual)

    scaled_2 = gouda.rescale(test_data, new_min=-1, new_max=2)
    assert scaled_2.max() == 2
    assert scaled_2.min() == -1


def test_factors():
    result1 = gouda.factors(100)
    expected1 = set([1, 2, 4, 5, 10, 20, 25, 50, 100])
    assert len(result1.symmetric_difference(expected1)) == 0

    result2 = gouda.factors(6)
    expected2 = set([1, 2, 3, 6])
    assert len(result2.symmetric_difference(expected2)) == 0

    result3 = gouda.factors(7)
    expected3 = set([1, 7])
    assert len(result3.symmetric_difference(expected3)) == 0

    with pytest.raises(ValueError):
        assert gouda.factors(0)

    with pytest.warns(UserWarning):
        result4 = gouda.factors(-1)
        expected4 = set([1])
        assert len(result4.symmetric_difference(expected4)) == 0


def test_prime_factors():
    result1 = gouda.prime_factors(100)
    expected1 = [2, 2, 5, 5]
    assert result1 == expected1

    result2 = gouda.prime_factors(7)
    expected2 = [7]
    assert result2 == expected2

    with pytest.raises(ValueError):
        assert gouda.prime_factors(0)

    with pytest.warns(UserWarning):
        result3 = gouda.prime_factors(-1)
        expected3 = [1]
        assert result3 == expected3


def test_prime_overlap():
    result1 = gouda.prime_overlap(2, 5)
    assert len(result1) == 0

    result2 = gouda.prime_overlap(4, 10)
    assert result2 == [2]

    result3 = gouda.prime_overlap(672, 42)
    assert result3 == [2, 3, 7]

    result4 = gouda.prime_overlap(42, 672)
    assert result4 == [2, 3, 7]

    result5 = gouda.prime_overlap(7, 5)
    assert result5 == []


def test_flip_dict():
    one2one_dict = {'a': 1, 'b': 2, 'c': 3}
    many2one_dict = {'a': 1, 'b': 1, 'c': 2}
    all2one_dict = {'a': 1, 'b': 1, 'c': 1}

    flip1 = gouda.flip_dict(one2one_dict)
    assert flip1 == {1: 'a', 2: 'b', 3: 'c'}

    flip2 = gouda.flip_dict(one2one_dict, unique_items=True)
    assert flip2 == {1: 'a', 2: 'b', 3: 'c'}

    flip3 = gouda.flip_dict(one2one_dict, force_list_values=True)
    assert flip3 == {1: ['a'], 2: ['b'], 3: ['c']}

    flip4 = gouda.flip_dict(many2one_dict)
    assert flip4 == {1: ['a', 'b'], 2: 'c'}

    flip5 = gouda.flip_dict(many2one_dict, unique_items=True)
    assert flip5 == {1: 'b', 2: 'c'}

    flip6 = gouda.flip_dict(many2one_dict, force_list_values=True)
    assert flip6 == {1: ['a', 'b'], 2: ['c']}

    flip7 = gouda.flip_dict(all2one_dict)
    assert flip7 == {1: ['a', 'b', 'c']}


def test_softmax():
    data = np.arange(10).reshape([5, 2])
    assert gouda.softmax(data).sum() == 1
    assert gouda.softmax(data, axis=0).sum() == 2
    assert gouda.softmax(data, axis=1).sum() == 5
