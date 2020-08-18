import pytest

import numpy as np

import gouda


def test_ConfusionMatrix_init_properties():
    mat = gouda.ConfusionMatrix()
    np.testing.assert_array_equal(mat.matrix, np.zeros([2, 2]))
    assert mat._num_classes == 2

    mat = gouda.ConfusionMatrix(num_classes=2, dtype=np.uint8)
    assert isinstance(mat.matrix, np.ndarray)
    assert mat.shape == mat.matrix.shape
    assert mat.shape == (2, 2)
    assert mat.dtype == mat.matrix.dtype
    assert mat.matrix.dtype == 'uint8'
    assert mat.size == mat.matrix.size
    assert mat.size == 4
    assert mat.num_classes == 2

    mat.dtype = np.float32
    assert mat.matrix.dtype == mat.dtype
    assert mat.dtype == 'float32'


def test_ConfusionMatrix_init_withData():
    predictions = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    labels = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    mat = gouda.ConfusionMatrix(predictions=predictions, labels=labels)
    assert mat.threshold is None
    assert mat.shape == (3, 3)
    assert mat.size == 9
    assert mat.count() == 9
    np.testing.assert_array_equal(mat.matrix, np.ones([3, 3]))


def test_ConfusionMatrix_init_withThreshold():
    predictions = [0.1, 0.2, 0.6, 0.8, 0.9, 1]
    labels = [0, 0, 0, 1, 1, 1]
    mat = gouda.ConfusionMatrix(predictions=predictions, labels=labels, threshold=0.7)
    assert mat.threshold == 0.7
    assert mat.num_classes == 2
    assert mat.shape == (2, 2)
    assert mat.count() == 6
    np.testing.assert_array_equal(mat.matrix, np.array([[3, 0], [0, 3]]))


def test_ConfusionMatrix_reset():
    predictions = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    labels = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    mat = gouda.ConfusionMatrix(predictions=predictions, labels=labels)
    np.testing.assert_array_equal(mat.matrix, np.ones([3, 3]))

    mat.reset()
    assert mat.num_classes == 3
    np.testing.assert_array_equal(mat.matrix, np.zeros([3, 3]))

    mat.reset(num_classes=2)
    assert mat.num_classes == 2
    np.testing.assert_array_equal(mat.matrix, np.zeros([2, 2]))

    with pytest.raises(ValueError):
        assert mat.reset(-1)


def test_ConfusionMatrix_iadd():
    predictions = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    labels = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    mat = gouda.ConfusionMatrix(predictions=predictions, labels=labels)
    np.testing.assert_array_equal(mat.matrix, np.ones([3, 3]))
    mat += [0, 0]
    np.testing.assert_array_equal(mat.matrix, np.array([[2, 1, 1], [1, 1, 1], [1, 1, 1]]))
    mat += [[0.0, 1.0], [0, 0]]
    np.testing.assert_array_equal(mat.matrix, np.array([[3, 2, 1], [1, 1, 1], [1, 1, 1]]))
    assert mat.count() == 12
    mat += [True, False]
    assert mat.count() == 13
    np.testing.assert_array_equal(mat.matrix, np.array([[3, 3, 1], [1, 1, 1], [1, 1, 1]]))

    mat += [[0.1, 0.2, 0.8], 2]
    assert mat.count() == 14
    np.testing.assert_array_equal(mat.matrix, np.array([[3, 3, 1], [1, 1, 1], [1, 1, 2]]))


def test_ConfusionMatrix_iadd_exception():
    predictions = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    labels = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    mat = gouda.ConfusionMatrix(predictions=predictions, labels=labels)
    with pytest.raises(ValueError):
        assert mat.__iadd__(['test', False])

    with pytest.raises(ValueError):
        assert mat.__iadd__([False, 'test'])

    with pytest.raises(ValueError):
        assert mat.__iadd__([False, 1.2])


def test_ConfusionMatrix_add():
    predictions = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    labels = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    mat1 = gouda.ConfusionMatrix(predictions=predictions, labels=labels, dtype=np.int)
    mat2 = gouda.ConfusionMatrix(predictions=predictions, labels=labels, dtype=np.float)
    assert mat2.dtype == 'float'
    np.testing.assert_array_equal(mat1.matrix, np.ones([3, 3]))
    np.testing.assert_array_equal(mat2.matrix, np.ones([3, 3]))

    with pytest.warns(UserWarning):
        mat3 = mat1 + mat2
    assert mat3.dtype == 'int'
    np.testing.assert_array_equal(mat3.matrix, np.ones([3, 3]) * 2)

    assert mat1.dtype == 'int'
    assert mat2.dtype == 'float'
    with pytest.warns(UserWarning):
        mat4 = mat2 + mat1
    assert mat4.dtype == 'float'
    np.testing.assert_array_equal(mat4.matrix, np.ones([3, 3]) * 2)

    mat5 = gouda.ConfusionMatrix(num_classes=4)
    mat6 = mat5 + mat1
    assert mat6.dtype == 'int'
    np.testing.assert_array_equal(mat6.matrix, np.array([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]]))


def test_ConfusionMatrix_str():
    predictions = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    labels = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    mat1 = gouda.ConfusionMatrix(predictions=predictions, labels=labels, dtype=np.int)
    assert str(mat1) == str(mat1.matrix)


def test_ConfusionMatrix_accuracy():
    predictions = [0, 1, 0, 1, 1, 1, 2, 1, 2]
    labels = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    mat1 = gouda.ConfusionMatrix(predictions=predictions, labels=labels, dtype=np.int)
    assert mat1.accuracy() == 5 / 9


def test_ConfusionMatrix_specificity():
    predictions = [0, 1, 0, 1, 1, 1, 2, 1, 2]
    labels = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    mat1 = gouda.ConfusionMatrix(predictions=predictions, labels=labels, dtype=np.int)
    specs = mat1.specificity()
    for i in range(3):
        assert specs[i] == mat1.specificity(i)
    np.testing.assert_array_equal(specs, np.array([5 / 7, 1, 5 / 7]))


def test_ConfusionMatrix_sensitivity():
    predictions = [0, 1, 0, 1, 1, 1, 2, 1, 2]
    labels = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    mat1 = gouda.ConfusionMatrix(predictions=predictions, labels=labels, dtype=np.int)
    sens = mat1.sensitivity()
    for i in range(3):
        assert sens[i] == mat1.sensitivity(i)
    np.testing.assert_array_equal(sens, np.array([1 / 3, 1, 1 / 3]))


def test_ConfusionMatrix_print():
    # Needs a visual check?
    predictions = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    labels = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    mat1 = gouda.ConfusionMatrix(predictions=predictions, labels=labels, dtype=np.int)
    mat1.print()
    confusion_string = mat1.print(return_string=True)
    assert confusion_string is not None
    assert len(confusion_string) > 10
    assert 'Specificity' in confusion_string
    assert 'Sensitivity' in confusion_string
    assert 'Accuracy' in confusion_string

    confusion_string2 = mat1.print(show_specificities=False, return_string=True)
    assert 'Specificity' not in confusion_string2
    confusion_string3 = mat1.print(show_sensitivities=False, return_string=True)
    assert 'Sensitivity' not in confusion_string3
    confusion_string4 = mat1.print(show_accuracy=False, return_string=True)
    assert 'Accuracy' not in confusion_string4


def test_ConfusionMatrix_from_array_add_array():
    predictions = np.zeros([100])
    labels = np.zeros([100])
    predictions[25:100] = 1
    labels[50:] = 1
    mat = gouda.ConfusionMatrix()
    with pytest.warns(UserWarning):
        mat.add_array(predictions, labels)
    np.testing.assert_array_equal(mat.matrix, np.array([[25, 25], [0, 50]]))

    mat2 = gouda.ConfusionMatrix.from_array(predictions, labels.astype(np.int))
    np.testing.assert_array_equal(mat2.matrix, mat.matrix)

    mat3 = gouda.ConfusionMatrix()
    mat3.matrix = None
    mat3.add_array(predictions, labels.astype(np.int))
    np.testing.assert_array_equal(mat3.matrix, np.array([[25, 25], [0, 50]]))

    mat4 = gouda.ConfusionMatrix(num_classes=1)
    mat4.add_array(predictions, labels.astype(np.int))
    np.testing.assert_array_equal(mat4.matrix, np.array([[25, 25], [0, 50]]))


def test_ConfusionMatrix_add_array_exception():
    predictions = np.zeros([10])
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    mat = gouda.ConfusionMatrix()
    with pytest.raises(ValueError):
        mat.add_array(predictions, labels)

    predictions = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    labels = np.zeros([10])
    mat = gouda.ConfusionMatrix()
    with pytest.raises(ValueError):
        mat.add_array(predictions, labels)

    predictions = np.zeros([10])
    labels = np.zeros([10]).astype(np.object)
    mat = gouda.ConfusionMatrix()
    with pytest.raises(ValueError):
        mat.add_array(predictions, labels)

    predictions = np.zeros([10], dtype=np.int)
    labels = np.zeros([8], dtype=np.int)
    with pytest.raises(ValueError):
        mat = gouda.ConfusionMatrix.from_array(predictions, labels)


def test_ConfusionMatrix_add_array_argmax():
    predictions = np.array([[0.1, 0.9], [0.3, 0.7], [0.6, 0.4], [1.0, 0.0]])
    labels = np.array([1, 1, 0, 1])
    mat = gouda.ConfusionMatrix.from_array(predictions, labels)
    np.testing.assert_array_equal(mat.matrix, np.array([[1, 0], [1, 2]]))


def test_ConfusionMatrix_add_array_threshold():
    predictions = np.array([0.1, 0.2, 0.6, 0.8])
    labels = np.array([0, 1, 0, 1])
    mat = gouda.ConfusionMatrix.from_array(predictions, labels)
    np.testing.assert_array_equal(mat.matrix, np.ones([2, 2]))

    predictions = np.array([0.1, 0.2, 0.6, 0.8])
    labels = np.array([0, 1, 0, 1])
    mat = gouda.ConfusionMatrix.from_array(predictions, labels, threshold=0.7)
    np.testing.assert_array_equal(mat.matrix, np.array([[2, 0], [1, 1]]))


def test_ConfusionMatrix_precision_mcc():
    predictions = np.array([0, 0, 1, 1])
    labels = np.array([0, 1, 0, 1])
    mat = gouda.ConfusionMatrix.from_array(predictions, labels)
    np.testing.assert_array_equal(mat.precision(), np.array([0.5, 0.5]))
    np.testing.assert_array_equal(np.array([mat.precision(0), mat.precision(1)]), mat.precision())
    assert mat.mcc() == 0.0

    predictions = np.array([0, 0, 1, 1, 1])
    labels = np.array([0, 0, 0, 1, 1])
    mat2 = gouda.ConfusionMatrix.from_array(predictions, labels)
    np.testing.assert_array_equal(mat2.precision(), np.array([2.0 / 2.0, 2.0 / 3.0]))
    assert mat2.mcc() == 2.0 / 3.0

    mat3 = gouda.ConfusionMatrix.from_array(np.array([1, 2, 3]), np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        mat3.mcc()
