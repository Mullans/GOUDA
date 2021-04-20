import pytest

import numpy as np

import gouda


def test_init():
    test_mat = gouda.BinaryConfusionMatrix()
    np.testing.assert_array_equal(test_mat.matrix, np.array([[0, 0], [0, 0]]))

    test_arr = np.array([[0, 1, 0], [1, 1, 0]])
    test_mat_2 = gouda.BinaryConfusionMatrix(test_arr)
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
    test_mat = gouda.BinaryConfusionMatrix()
    np.testing.assert_array_equal(test_mat.matrix, np.array([[0, 0], [0, 0]]))

    test_arr = np.array([[0, 1, 0], [1, 1, 0]])
    test_mat.add(test_arr)
    np.testing.assert_array_equal(test_mat.matrix, np.array([[1, 0], [1, 1]]))

    test_arr2 = np.array([[0, 1], [1, 1], [0, 0]])
    test_mat.add(test_arr2)
    np.testing.assert_array_equal(test_mat.matrix, np.array([[2, 0], [2, 2]]))

    with pytest.raises(ValueError):
        test_mat.add(np.ones([3, 3]))

    test_mat2 = gouda.BinaryConfusionMatrix()
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


def test_add_matrix():
    test_mat = gouda.BinaryConfusionMatrix()
    test_arr = np.array([[0, 1, 0], [1, 1, 0]])
    test_mat.add(test_arr)
    np.testing.assert_array_equal(test_mat.matrix, np.array([[1, 0], [1, 1]]))

    test_mat2 = gouda.BinaryConfusionMatrix()
    test_arr2 = np.array([[0, 0, 0], [0, 0, 0]])
    test_mat2.add(test_arr2)
    np.testing.assert_array_equal(test_mat2.matrix, np.array([[3, 0], [0, 0]]))

    test_mat.add_matrix(test_mat2)
    np.testing.assert_array_equal(test_mat.matrix, np.array([[4, 0], [1, 1]]))
    np.testing.assert_array_equal(test_mat2.matrix, np.array([[3, 0], [0, 0]]))

    test_mat3 = gouda.BinaryConfusionMatrix()
    test_arr3 = np.array([[1, 1, 1], [1, 1, 1]])
    test_mat3.add(test_arr3)
    np.testing.assert_array_equal(test_mat3.matrix, np.array([[0, 0], [0, 3]]))
    test_mat.add_matrix(test_mat3)
    np.testing.assert_array_equal(test_mat.matrix, np.array([[4, 0], [1, 4]]))
    np.testing.assert_array_equal(test_mat3.matrix, np.array([[0, 0], [0, 3]]))

    with pytest.raises(ValueError):
        test_mat.add_matrix([[1, 1, 1], [1, 0, 1]])


def test_parameters():
    test_mat = gouda.BinaryConfusionMatrix()
    assert test_mat.print(return_string=True, as_label=False) == '         →  Predicted\n↓ Expected          | 0 | 1 \n                0   | 0 | 0 |\n                1   | 0 | 0 |\n'

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
    test_string = '         →  Predicted\n↓ Expected          | False | True  \n              False |     2 |     3 |\n              True  |     1 |     4 |\n\nAccuracy:    0.6000\nSensitivity: 0.8000\nSpecificity: 0.4000'
    assert test_mat.print(as_label=True, return_string=True, show_specificity=True, show_accuracy=True, show_sensitivity=True) == test_string

    test_string2 = '         →  Predicted\n↓ Expected        | No  | Yes \n              No  |   2 |   3 |\n              Yes |   1 |   4 |\n\nAccuracy:    0.6000\nSensitivity: 0.8000\nSpecificity: 0.4000'
    assert test_mat.print(pos_label='Yes', neg_label='No', as_label=True, return_string=True, show_specificity=True, show_accuracy=True, show_sensitivity=True) == test_string2


def test_math():
    test_mat = gouda.BinaryConfusionMatrix()
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

    test_mat2 = gouda.BinaryConfusionMatrix()
    test_mat2[1, 1] = 99999999999
    assert test_mat2.mcc() == 0
    test_mat2[0, 0] = 99999999999
    assert test_mat2.mcc() == 1


def test_array():
    test_mat = gouda.BinaryConfusionMatrix()
    assert test_mat.__array__().dtype == test_mat.dtype
    assert test_mat.__array__(np.uint8).dtype == np.uint8


def test_underline():
    test_string = 'hello'
    underlined = gouda.binaryconfusionmatrix.underline(test_string)
    assert underlined[:4] == '\033[4m'
    assert underlined[-4:] == '\033[0m'
    assert underlined[4:-4] == test_string


def test_save_load():
    test_mat = gouda.BinaryConfusionMatrix()
    test_arr = np.array([[1, 1, 0], [1, 1, 0]])
    test_mat.add(test_arr)
    np.testing.assert_array_equal(test_mat.matrix, np.array([[1, 0], [0, 2]]))
    test_mat.save('ScratchFiles/test_mat.txt')

    test_mat2 = gouda.BinaryConfusionMatrix.load('ScratchFiles/test_mat.txt')
    np.testing.assert_array_equal(test_mat.matrix, test_mat2.matrix)
