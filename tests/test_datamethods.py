import pytest

import numpy as np

import gouda


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


def test_get_specificities():
    data = np.array([[10, 7, 5], [7, 10, 7], [5, 7, 10]])
    specificities = gouda.get_specificities(data)
    assert specificities[0] == (10 + 7 + 7 + 10) / (10 + 7 + 7 + 10 + 7 + 5)
    assert specificities[1] == (10 + 5 + 5 + 10) / (10 + 5 + 5 + 10 + 7 + 7)
    assert specificities[2] == (10 + 7 + 7 + 10) / (10 + 7 + 7 + 10 + 7 + 5)


def test_get_specificities_exception():
    data = np.arange(10)
    with pytest.raises(ValueError):
        assert gouda.get_specificities(data)

    data = np.arange(27).reshape([3, 3, 3])
    with pytest.raises(ValueError):
        assert gouda.get_specificities(data)

    data = np.arange(8).reshape([2, 4])
    with pytest.raises(ValueError):
        assert gouda.get_specificities(data)


def test_get_sensitivities():
    data = np.array([[10, 7, 5], [7, 10, 7], [5, 7, 10]])
    sensitivities = gouda.get_sensitivities(data)
    assert sensitivities[0] == 10 / (10 + 7 + 5)
    assert sensitivities[1] == 10 / (7 + 10 + 7)
    assert sensitivities[2] == 10 / (5 + 7 + 10)


def test_get_sensitivities_exception():
    data = np.arange(10)
    with pytest.raises(ValueError):
        assert gouda.get_sensitivities(data)

    data = np.arange(27).reshape([3, 3, 3])
    with pytest.raises(ValueError):
        assert gouda.get_sensitivities(data)

    data = np.arange(8).reshape([2, 4])
    with pytest.raises(ValueError):
        assert gouda.get_sensitivities(data)


def test_get_accuracy():
    data = np.array([[10, 7, 5], [7, 10, 7], [5, 7, 10]])
    accuracy = gouda.get_accuracy(data)
    assert accuracy == (10 + 10 + 10) / data.sum()


def test_get_accuracy_exception():
    data = np.arange(10)
    with pytest.raises(ValueError):
        assert gouda.get_accuracy(data)

    data = np.arange(27).reshape([3, 3, 3])
    with pytest.raises(ValueError):
        assert gouda.get_accuracy(data)

    data = np.arange(8).reshape([2, 4])
    with pytest.raises(ValueError):
        assert gouda.get_accuracy(data)


def test_get_confusion_matrix():
    predictions = [0, 1, 2, 1, 1, 1]
    labels = [0, 1, 2, 0, 0, 0]
    matrix = gouda.get_confusion_matrix(predictions, labels)
    expected = np.array([[1, 3, 0], [0, 1, 0], [0, 0, 1]])
    assert matrix.shape[0] == matrix.shape[1]
    assert matrix.shape[0] == 3
    assert matrix.sum() == 6
    np.testing.assert_array_equal(matrix, expected)
    matrix2 = gouda.get_confusion_matrix(predictions, labels, num_classes=3)
    np.testing.assert_array_equal(matrix, matrix2)


def test_get_confusion_matrix_exception():
    predictions = np.arange(8).reshape([4, 2])
    labels = np.arange(8)
    with pytest.raises(ValueError):
        assert gouda.get_confusion_matrix(predictions, labels)

    predictions = [0, 1, 2]
    labels = [0, 1]
    with pytest.raises(ValueError):
        assert gouda.get_confusion_matrix(predictions, labels)


def test_get_binary_confusion_matrix():
    predictions = [False, False, True, True]
    labels = [True, False, True, False]
    matrix = gouda.get_binary_confusion_matrix(predictions, labels)
    expected = np.ones([2, 2])
    assert matrix.shape[0] == matrix.shape[1]
    assert matrix.shape[0] == 2
    assert matrix.sum() == 4
    np.testing.assert_array_equal(matrix, expected)


def test_get_binary_confusion_matrix_threshold():
    predictions = [-1, 0.4, 0.5, 1.1]
    labels = [1, 0, 1, 0]
    matrix = gouda.get_binary_confusion_matrix(predictions, labels, threshold=0.4)
    expected = np.ones([2, 2])
    assert matrix.shape[0] == matrix.shape[1]
    assert matrix.shape[0] == 2
    assert matrix.sum() == 4
    np.testing.assert_array_equal(matrix, expected)


def test_get_binary_confusion_matrix_exception():
    predictions = np.arange(8).reshape([4, 2])
    labels = np.arange(8)
    with pytest.raises(ValueError):
        assert gouda.get_binary_confusion_matrix(predictions, labels)

    predictions = [0, 1, 2]
    labels = [0, 1]
    with pytest.raises(ValueError):
        assert gouda.get_binary_confusion_matrix(predictions, labels)


def test_print_confusion_matrix_and_underline():
    data = np.arange(16).reshape([4, 4])
    gouda.print_confusion_matrix(data)