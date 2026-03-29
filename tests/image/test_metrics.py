from __future__ import annotations

import numpy as np

from gouda import image as gimage


def test_laplacian_var():
    image_test = np.zeros([100, 100, 3], dtype=np.uint8)
    assert gimage.laplacian_var(image_test) == 0

    image_test[:50] += 255
    assert gimage.laplacian_var(image_test) == 161.3


def test_sobel_var():
    image_test = np.zeros([100, 100, 3], dtype=np.uint8)
    assert gimage.sobel_var(image_test) == 0

    image_test[:50] += 255
    assert gimage.sobel_var(image_test) == 629.1456
