import pytest

import numpy as np

from gouda import BinaryConfusionMatrix


def test_init():
    test_mat = BinaryConfusionMatrix()
    np.testing.assert_array_equal(test_mat.matrix, np.array([[0, 0], [0, 0]]))

    test_arr = np.array([[0, 1, 0], [1, 1, 0]])
    test_mat_2 = BinaryConfusionMatrix(test_arr)
    np.testing.assert_array_equal(test_mat_2.matrix, np.array([[1, 0], [1, 1]]))

    assert test_mat_2.dtype == np.int

    test_mat_2.dtype = np.uint8
    assert test_mat_2.dtype == np.uint8

    test_mat_2.reset()
    np.testing.assert_array_equal(test_mat_2.matrix, np.array([[0, 0], [0, 0]]))
    assert test_mat_2.dtype == np.uint8

    test_mat_2.reset(dtype=np.float32)
    assert test_mat_2.dtype == np.float32


def test_add():
    test_mat = BinaryConfusionMatrix()
    np.testing.assert_array_equal(test_mat.matrix, np.array([[0, 0], [0, 0]]))

    test_arr = np.array([[0, 1, 0], [1, 1, 0]])
    test_mat.add(test_arr)
    np.testing.assert_array_equal(test_mat.matrix, np.array([[1, 0], [1, 1]]))

    test_arr2 = np.array([[0, 1], [1, 1], [0, 0]])
    test_mat.add(test_arr2)
    np.testing.assert_array_equal(test_mat.matrix, np.array([[2, 0], [2, 2]]))

    with pytest.raises(ValueError):
        test_mat.add(np.ones([3, 3]))

    test_mat2 = BinaryConfusionMatrix()
    test_mat2.add(1, 1)
    np.testing.assert_array_equal(test_mat2.matrix, np.array([[0, 0], [0, 1]]))

    test_mat3 = test_mat2 + [0, 0]
    np.testing.assert_array_equal(test_mat3.matrix, np.array([[1, 0], [0, 1]]))

    test_mat4 = test_mat3 + test_mat2
    np.testing.assert_array_equal(test_mat4.matrix, np.array([[1, 0], [0, 2]]))
    np.testing.assert_array_equal(test_mat3.matrix, np.array([[1, 0], [0, 1]]))
    np.testing.assert_array_equal(test_mat2.matrix, np.array([[0, 0], [0, 1]]))

    test_mat3 += test_mat2
    np.testing.assert_array_equal(test_mat3.matrix, np.array([[1, 0], [0, 2]]))

    test_mat3 += [0, 0]
    np.testing.assert_array_equal(test_mat3.matrix, np.array([[2, 0], [0, 2]]))

    with pytest.raises(ValueError):
        test_mat.add([1, 2])

    with pytest.raises(ValueError):
        test_mat.add(np.ones(10), np.ones(5))

    with pytest.raises(ValueError):
        test_mat.add(np.ones([3, 3, 3]))

    with pytest.raises(ValueError):
        test_mat.add([1, 2, 3])

    test_mat3.add(predictions=[np.ones(3), [0, 0, 0]], labels=None)
    np.testing.assert_array_equal(test_mat3.matrix, np.array([[2, 3], [0, 2]]))

    test_mat3.threshold = 0.6
    test_mat3.add(predictions=[0.3, 0.5, 0.7], labels=[1, 1, 1])
    np.testing.assert_array_equal(test_mat3.matrix, np.array([[2, 3], [2, 3]]))


def test_parameters():
    test_mat = BinaryConfusionMatrix()
    assert test_mat.print(return_string=True, as_bool=False) == '         →  Predicted\n↓ Expected          | 0 | 1 \n                0   | 0 | 0 |\n                1   | 0 | 0 |\n'

    test_mat.add(np.array([[1, 1, 1, 1], [1, 1, 1, 1]]))
    test_mat.add(np.array([[1, 1, 1], [0, 0, 0]]))
    with pytest.warns(UserWarning):
        test_mat.add(np.array([[0, 0], [0, 0]]))
    test_mat.add(np.array([[0], [1]]))

    assert test_mat.true_positive == 4
    assert test_mat.true_negative == 2
    assert test_mat.false_positive == 3
    assert test_mat.false_negative == 1

    assert test_mat[1, 1] == 4
    assert test_mat[0, 0] == 2

    assert str(test_mat) == '[[2 3]\n [1 4]]'
    assert repr(test_mat) == 'BinaryConfusionMatrix([2, 3]\n                      [1, 4])'
    test_mat.print()
    assert test_mat.print(as_bool=True, return_string=True, show_specificity=True, show_accuracy=True, show_sensitivity=True) == '         →  Predicted\n↓ Expected          | False  |  True  \n              False |      2 |      3 |\n              True  |      1 |      4 |\n\nAccuracy:    0.6000\nSensitivity: 0.8000\nSpecificity: 0.4000'


def test_math():
    test_mat = BinaryConfusionMatrix()
    assert test_mat.mcc() == 0
    assert test_mat.sensitivity() == 0
    assert test_mat.specificity() == 0
    assert test_mat.precision() == 0
    assert test_mat.accuracy() == 0
    assert test_mat.zero_rule() == 0
    test_mat.add(np.array([[1, 1, 1, 1], [1, 1, 1, 1]]))
    test_mat.add(np.array([[1, 1, 1], [0, 0, 0]]))
    with pytest.warns(UserWarning):
        test_mat.add(np.array([[0, 0], [0, 0]]))
    test_mat.add(np.array([[0], [1]]))

    assert test_mat.accuracy() == 6 / 10
    assert test_mat.count() == 10
    assert test_mat.sensitivity() == 4 / 5
    assert test_mat.specificity() == 2 / 5
    assert test_mat.precision() == 4 / 7
    assert test_mat.zero_rule() == 5 / 10

    n = 10
    s = (4 + 1) / n
    p = (4 + 3) / n
    top = (4 / n) - (s * p)
    bottom = p * s * (1 - p) * (1 - s)
    manual_mcc = top / np.sqrt(bottom)

    np.testing.assert_almost_equal(test_mat.mcc(), manual_mcc)

    test_mat2 = BinaryConfusionMatrix()
    test_mat2[1, 1] = 99999999999
    assert test_mat2.mcc() == 0
    test_mat2[0, 0] = 99999999999
    assert test_mat2.mcc() == 1


def test_array():
    test_mat = BinaryConfusionMatrix()
    assert test_mat.__array__().dtype == test_mat.dtype
    assert test_mat.__array__(np.uint8).dtype == np.uint8
