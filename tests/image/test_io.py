from __future__ import annotations

import os
import stat

import cv2
import numpy as np
import pytest

from gouda import GRAYSCALE, RGB, UNCHANGED, GoudaPath
from gouda import image as gimage


def test_imwrite_imread(scratch_path):
    with pytest.raises(ValueError):
        gimage.imread("alskdfjalsdjf/asdf.png")

    image_test = np.ones([100, 100, 3], dtype=np.uint8)
    image_test[:50] *= 0
    image_test[:, :, 2] *= 0
    image_test *= 255

    assert image_test[:, :, 2].sum() == 0
    gimage.imwrite(GoudaPath(scratch_path / "test_RGB.png"), image_test)
    gimage.imwrite(scratch_path / "test_BGR.png", image_test, as_rgb=False)
    gimage.imwrite(scratch_path / "test_singleChannel.png", image_test[:, :, :1])
    gimage.imwrite(scratch_path / "test_2D.png", image_test[:, :, 0])
    gimage.imwrite(scratch_path / "test_uint16.png", image_test.astype(np.uint16))
    with pytest.raises(ValueError):
        assert gimage.imwrite(scratch_path / "failure.png", image_test[:, :, :2])

    assert os.path.isfile(scratch_path / "test_RGB.png")
    assert os.path.isfile(scratch_path / "test_BGR.png")
    assert os.path.isfile(scratch_path / "test_singleChannel.png")
    assert os.path.isfile(scratch_path / "test_2D.png")
    assert os.path.isfile(scratch_path / "test_uint16.png")
    assert not os.path.isfile(scratch_path / "failure.png")

    image_test_in_1 = gimage.imread(GoudaPath(scratch_path / "test_RGB.png"), flag=RGB)
    image_test_in_2 = gimage.imread(scratch_path / "test_RGB.png", flag=None)
    image_test_in_3 = gimage.imread(scratch_path / "test_BGR.png", flag=RGB)
    image_test_in_4 = gimage.imread(scratch_path / "test_BGR.png", flag=None)
    np.testing.assert_array_equal(image_test_in_1, image_test_in_4)
    np.testing.assert_array_equal(image_test_in_2, image_test_in_3)

    image_test_in_5 = gimage.imread(scratch_path / "test_singleChannel.png")
    image_test_in_6 = gimage.imread(scratch_path / "test_singleChannel.png")

    np.testing.assert_array_equal(image_test_in_5, image_test_in_6)

    image_test_in_7 = gimage.imread(scratch_path / "test_uint16.png", flag=UNCHANGED)
    assert image_test_in_7.dtype == np.uint16

    image_test_in_8 = gimage.imread(scratch_path / "test_RGB.png", flag=GRAYSCALE)
    assert image_test_in_8.shape == (100, 100)
    np.testing.assert_allclose(image_test_in_8, cv2.cvtColor(image_test, cv2.COLOR_RGB2GRAY), rtol=0, atol=1)


def test_imread_unknown_flag(scratch_path):
    image_test = np.ones([10, 10, 3], dtype=np.uint8) * 128
    gimage.imwrite(scratch_path / "test_flag.png", image_test)
    result = gimage.imread(scratch_path / "test_flag.png", flag=99)
    assert result is not None
    assert result.shape == (10, 10, 3)


@pytest.mark.skipif(os.getuid() == 0, reason="chmod restrictions don't apply to root")
def test_imread_unreadable_file(scratch_path):
    img_path = scratch_path / "test_unreadable.png"
    gimage.imwrite(img_path, np.ones([10, 10, 3], dtype=np.uint8))
    os.chmod(img_path, 0o000)
    try:
        with pytest.raises(ValueError, match="not readable"):
            gimage.imread(img_path)
    finally:
        os.chmod(img_path, stat.S_IRUSR | stat.S_IWUSR)
