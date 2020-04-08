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
    assert normed_1.max() == 1
    assert normed_1.min() == 0
    assert pytest.approx(normed_1.min(axis=2).sum(), 0.1) == 50
    normed_2 = gouda.normalize(test_data, axis=1)
    assert normed_2.max(axis=1).sum() == 100
    assert normed_2.min(axis=1).sum() == 0
    assert pytest.approx(normed_2.min(axis=2).sum(), 0.1) == 50
    normed_3 = gouda.normalize(test_data, axis=(0, 1))
    assert normed_3.max(axis=(0, 1)).sum() == 10
    assert normed_3.min(axis=(0, 1)).sum() == 0
    assert pytest.approx(normed_3.min(axis=2).sum(), 0.1) == 50


def test_rescale():
    test_data = np.arange(100).reshape([10, 10])
    scaled_1 = gouda.rescale(test_data, new_min=0, new_max=1, axis=1)
    np.testing.assert_array_equal(scaled_1, gouda.normalize(test_data, axis=1))

    scaled_2 = gouda.rescale(test_data, new_min=-1, new_max=2)
    assert scaled_2.max() == 2
    assert scaled_2.min() == -1


def test_standardize():
    test_data = np.arange(1000).reshape([10, 10, 10])
    standard_1 = gouda.standardize(test_data)
    assert pytest.approx(standard_1.mean(), 0.0001) == 0
    assert pytest.approx(standard_1.std(), 0.0001) == 1
    standard_2 = gouda.standardize(test_data, axis=1)
    assert pytest.approx(standard_2.mean(axis=1), 0.0001) == 0
    assert pytest.approx(standard_2.std(axis=1), 0.0001) == 1
    standard_3 = gouda.standardize(test_data, axis=(0, 1))
    assert pytest.approx(standard_3.mean(axis=(0, 1)), 0.0001) == 0
    assert pytest.approx(standard_3.std(axis=(0, 1)), 0.0001) == 1


# def test_get_specificities():
#     data = np.array([[10, 7, 5], [7, 10, 7], [5, 7, 10]])
#     specificities = gouda.get_specificities(data)
#     assert specificities[0] == (10 + 7 + 7 + 10) / (10 + 7 + 7 + 10 + 7 + 5)
#     assert specificities[1] == (10 + 5 + 5 + 10) / (10 + 5 + 5 + 10 + 7 + 7)
#     assert specificities[2] == (10 + 7 + 7 + 10) / (10 + 7 + 7 + 10 + 7 + 5)
#
#
# def test_get_specificities_exception():
#     data = np.arange(10)
#     with pytest.raises(ValueError):
#         assert gouda.get_specificities(data)
#
#     data = np.arange(27).reshape([3, 3, 3])
#     with pytest.raises(ValueError):
#         assert gouda.get_specificities(data)
#
#     data = np.arange(8).reshape([2, 4])
#     with pytest.raises(ValueError):
#         assert gouda.get_specificities(data)
#
#
# def test_get_sensitivities():
#     data = np.array([[10, 7, 5], [7, 10, 7], [5, 7, 10]])
#     sensitivities = gouda.get_sensitivities(data)
#     assert sensitivities[0] == 10 / (10 + 7 + 5)
#     assert sensitivities[1] == 10 / (7 + 10 + 7)
#     assert sensitivities[2] == 10 / (5 + 7 + 10)
#
#
# def test_get_sensitivities_exception():
#     data = np.arange(10)
#     with pytest.raises(ValueError):
#         assert gouda.get_sensitivities(data)
#
#     data = np.arange(27).reshape([3, 3, 3])
#     with pytest.raises(ValueError):
#         assert gouda.get_sensitivities(data)
#
#     data = np.arange(8).reshape([2, 4])
#     with pytest.raises(ValueError):
#         assert gouda.get_sensitivities(data)
#
#
# def test_get_accuracy():
#     data = np.array([[10, 7, 5], [7, 10, 7], [5, 7, 10]])
#     accuracy = gouda.get_accuracy(data)
#     assert accuracy == (10 + 10 + 10) / data.sum()
#
#
# def test_get_accuracy_exception():
#     data = np.arange(10)
#     with pytest.raises(ValueError):
#         assert gouda.get_accuracy(data)
#
#     data = np.arange(27).reshape([3, 3, 3])
#     with pytest.raises(ValueError):
#         assert gouda.get_accuracy(data)
#
#     data = np.arange(8).reshape([2, 4])
#     with pytest.raises(ValueError):
#         assert gouda.get_accuracy(data)
#
#
# def test_get_confusion_matrix():
#     predictions = [0, 1, 2, 1, 1, 1]
#     labels = [0, 1, 2, 0, 0, 0]
#     matrix = gouda.get_confusion_matrix(predictions, labels)
#     expected = np.array([[1, 3, 0], [0, 1, 0], [0, 0, 1]])
#     assert matrix.shape[0] == matrix.shape[1]
#     assert matrix.shape[0] == 3
#     assert matrix.sum() == 6
#     np.testing.assert_array_equal(matrix, expected)
#     matrix2 = gouda.get_confusion_matrix(predictions, labels, num_classes=3)
#     np.testing.assert_array_equal(matrix, matrix2)
#
#
# def test_get_confusion_matrix_exception():
#     predictions = np.arange(8).reshape([4, 2])
#     labels = np.arange(8)
#     with pytest.raises(ValueError):
#         assert gouda.get_confusion_matrix(predictions, labels)
#
#     predictions = [0, 1, 2]
#     labels = [0, 1]
#     with pytest.raises(ValueError):
#         assert gouda.get_confusion_matrix(predictions, labels)
#
#
# def test_get_binary_confusion_matrix():
#     predictions = [False, False, True, True]
#     labels = [True, False, True, False]
#     matrix = gouda.get_binary_confusion_matrix(predictions, labels)
#     expected = np.ones([2, 2])
#     assert matrix.shape[0] == matrix.shape[1]
#     assert matrix.shape[0] == 2
#     assert matrix.sum() == 4
#     np.testing.assert_array_equal(matrix, expected)
#
#
# def test_get_binary_confusion_matrix_threshold():
#     predictions = [-1, 0.4, 0.5, 1.1]
#     labels = [1, 0, 1, 0]
#     matrix = gouda.get_binary_confusion_matrix(predictions, labels, threshold=0.4)
#     expected = np.ones([2, 2])
#     assert matrix.shape[0] == matrix.shape[1]
#     assert matrix.shape[0] == 2
#     assert matrix.sum() == 4
#     np.testing.assert_array_equal(matrix, expected)
#
#
# def test_get_binary_confusion_matrix_exception():
#     predictions = np.arange(8).reshape([4, 2])
#     labels = np.arange(8)
#     with pytest.raises(ValueError):
#         assert gouda.get_binary_confusion_matrix(predictions, labels)
#
#     predictions = [0, 1, 2]
#     labels = [0, 1]
#     with pytest.raises(ValueError):
#         assert gouda.get_binary_confusion_matrix(predictions, labels)
#
#
# def test_print_confusion_matrix_and_underline():
#     data = np.arange(16).reshape([4, 4])
#     gouda.print_confusion_matrix(data)


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
