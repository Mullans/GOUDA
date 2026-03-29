from __future__ import annotations

import cv2
import numpy as np

from gouda import image as gimage


def test_rotate():
    image_test = np.zeros([50, 50, 3], dtype=np.uint8)
    image_test[:25, :25] = 1
    assert image_test.sum() == 25 * 25 * 3
    rotate_test = gimage.rotate(image_test, degrees=90)
    assert rotate_test.shape == (50, 50, 3)
    assert rotate_test.sum() == image_test.sum()
    assert rotate_test[:25, :25].sum() == 0
    assert rotate_test[:25, 25:].sum() == 25 * 25 * 3

    rotate_test_2 = gimage.rotate(image_test, degrees=180)
    np.testing.assert_array_equal(rotate_test_2, image_test[::-1, ::-1])

    np.testing.assert_array_equal(gimage.rotate(image_test, degrees=-90), gimage.rotate(image_test, degrees=270))
    np.testing.assert_array_equal(image_test, gimage.rotate(image_test, degrees=360))


def test_rotate_allow_resize():
    image_test = np.zeros([100, 50, 3], dtype=np.uint8)
    rotated_1 = gimage.rotate(image_test, degrees=90, allow_resize=True)
    assert rotated_1.shape == (50, 100, 3)
    rotated_2 = gimage.rotate(image_test, degrees=90, allow_resize=False)
    assert rotated_2.shape == (100, 50, 3)


def test_padded_resize():
    image_test = np.ones([50, 50, 3], dtype=np.uint8)
    pad_test = gimage.padded_resize(image_test, size=[50, 50])
    assert pad_test.shape == (50, 50, 3)
    np.testing.assert_array_equal(image_test, pad_test)

    pad_test_2 = gimage.padded_resize(image_test, size=[50, 75])
    assert pad_test_2.shape == (50, 75, 3)
    assert pad_test_2[:, :12].sum() == 0
    assert pad_test_2[:, -12:].sum() == 0

    pad_test_3 = gimage.padded_resize(image_test, size=[75, 75])
    assert pad_test_3.shape == (75, 75, 3)
    assert pad_test_3.sum() == pad_test_3.size

    image_test_2 = np.ones([50, 50, 1], dtype=np.uint8)
    pad_test_4 = gimage.padded_resize(image_test_2, size=[50, 50])
    np.testing.assert_array_equal(pad_test_4, image_test_2)

    image_test_3 = np.ones([50, 50], dtype=np.uint8)
    pad_test_5 = gimage.padded_resize(image_test_3, size=[50, 50])
    assert pad_test_5.shape == (50, 50)
    np.testing.assert_array_equal(pad_test_4[:, :, 0], pad_test_5)

    image_test_4 = np.zeros([25, 50, 3], dtype=np.uint8)
    image_test_4[:10] = 1
    pad_test_6 = gimage.padded_resize(image_test_4, size=(50, 25, 3))
    assert pad_test_6.shape == (50, 25, 3)
    np.testing.assert_array_equal(pad_test_6, gimage.rotate(image_test_4, degrees=90))

    pad_test_7 = gimage.padded_resize(image_test, size=[75, 50])
    assert pad_test_7.shape == (75, 50, 3)
    assert pad_test_7[:12].sum() == 0
    assert pad_test_7[-12:].sum() == 0

    pad_test_7 = gimage.padded_resize(image_test[:, :, 0], size=[75, 50])
    assert pad_test_7.shape == (75, 50)
    assert pad_test_7[:12].sum() == 0
    assert pad_test_7[-12:].sum() == 0


def test_adjust_gamma():
    image_test = np.ones([50, 50, 3], dtype=np.uint8) * np.linspace(
        0, 255, num=50, endpoint=True, dtype=np.uint8
    ).reshape([50, 1, 1])
    assert image_test.dtype == "uint8"
    gamma_test_1 = gimage.adjust_gamma(image_test, gamma=2)
    gamma_test_2 = gimage.adjust_gamma(image_test, gamma=0.5)
    assert gamma_test_1.sum() > image_test.sum()
    assert gamma_test_2.sum() < image_test.sum()


def test_polar_to_cartesian():
    x = np.zeros([200, 200, 3])
    color1 = (255, 0, 0)
    color2 = (255, 255, 0)
    color3 = (255, 0, 255)
    cv2.circle(x, (100, 100), 50, color1, -1)
    cv2.circle(x, (100, 100), 25, color2, -1)
    cv2.circle(x, (100, 100), 10, color3, -1)

    y = gimage.polar_to_cartesian(x)
    z = gimage.polar_to_cartesian(x, output_shape=[400, 100, 3])

    assert y.shape == x.shape
    assert z.shape == tuple([400, 100, 3])
    assert (y[0, 0] == color3).all()
    assert (y[50, 0] == color2).all()
    assert (y[100, 0] == color1).all()
    assert (y[0, 0] == z[0, 0]).all()
    assert (y[50, 0] == z[100, 0]).all()
    assert (y[100, 0] == z[200, 0]).all()
