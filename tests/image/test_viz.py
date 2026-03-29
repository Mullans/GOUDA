from __future__ import annotations

import numpy as np
import pytest

from gouda import image as gimage
from gouda.plotting import parse_color


def test_stack_label():
    test_label = np.ones([10, 10])
    r_label = gimage.stack_label(test_label, label_channel=0)
    g_label = gimage.stack_label(test_label, label_channel=1)
    b_label = gimage.stack_label(test_label, label_channel=2)
    all_label = gimage.stack_label(test_label, label_channel=-1)
    np.testing.assert_array_equal(r_label + g_label + b_label, all_label)
    np.testing.assert_array_equal(r_label[:, :, 0], g_label[:, :, 1])
    np.testing.assert_array_equal(r_label[:, :, 0], b_label[:, :, 2])
    with pytest.raises(ValueError):
        gimage.stack_label(test_label, label_channel=4)


def test_masked_lineup():
    image_test = np.ones([100, 100, 3], dtype=np.uint8)
    image_test[:50] -= 1
    image_test *= 255
    label_test = np.ones([100, 100], dtype=np.uint8)
    label_test[:, :50] -= 1
    lineup_test = gimage.masked_lineup(image_test, label_test)
    np.testing.assert_array_equal(lineup_test[0], image_test)
    np.testing.assert_array_equal(lineup_test[1], gimage.add_mask(image_test, label_test))
    np.testing.assert_array_equal(lineup_test[2][:, :, 0], label_test * 255)


def test_split_signs():
    mask = np.zeros([100, 100], dtype=np.float32)
    mask[:50] = 1
    mask[50:] = -1
    check = gimage.split_signs(mask)

    np.testing.assert_array_equal(check[0, 0], (0, 1, 0))
    np.testing.assert_array_equal(check[99, 99], (1, 0, 0))

    check = gimage.split_signs(mask, positive_color="orange", negative_color="purple")
    np.testing.assert_array_almost_equal(
        check[0, 0],
        parse_color("orange"),
    )
    np.testing.assert_array_almost_equal(check[99, 99], parse_color("purple"))
