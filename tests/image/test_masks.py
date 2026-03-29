from __future__ import annotations

import cv2
import numpy as np
import pytest

import gouda
from gouda import image as gimage


def test_grabCut():
    image_test = np.ones([100, 100, 3], dtype=np.uint8)
    image_test[:50] -= 1
    image_test *= 255
    label_test = np.ones([100, 100], dtype=np.uint8)
    label_test[:25] = 0
    label_test[25:50] = 2
    label_test[50:75] = 3
    label_test[75:] = 1
    gc_img, gc_mask = gimage.grabCut(image_test, label_test, iterations=1)
    np.testing.assert_array_equal(gc_img, cv2.bitwise_and(image_test, image_test, mask=gc_mask))

    label_test_2 = np.ones([100, 100], dtype=float)
    label_test_2[:25] = 0
    label_test_2[25:50] = 0.25
    label_test_2[50:75] = 0.5
    label_test_2[75:] = 0.75
    gc_img_2, gc_mask_2 = gimage.grabCut(image_test, label_test_2, thresholds=(0.2, 0.4, 0.7), iterations=1)
    np.testing.assert_array_equal(gc_img, gc_img_2)
    np.testing.assert_array_equal(gc_mask, gc_mask_2)
    gc_img_3, gc_mask_3 = gimage.grabCut(image_test, label_test_2, thresholds=(0.2, 0.4, 0.7), iterations=1, clean=True)
    clean_test = gimage.clean_grabCut_mask(gc_mask)
    np.testing.assert_array_equal(clean_test, gc_mask_3)
    clean_test_2 = gimage.clean_grabCut_mask(gc_mask[:, :, np.newaxis])
    np.testing.assert_array_equal(clean_test_2, gc_mask_3[:, :, np.newaxis])


def test_grabCut_exception():
    image_test = np.ones([100, 100, 3], dtype=np.uint8)
    image_test[:50] -= 1
    image_test *= 255
    label_test = np.ones([100, 100], dtype=np.uint8)
    label_test[:50] = 0
    label_test[50:] = 2
    with pytest.raises(ValueError):
        assert gimage.grabCut(image_test, label_test)
    label_test[:50] = 3
    label_test[50:] = 1
    with pytest.raises(ValueError):
        assert gimage.grabCut(image_test, label_test)


def test_crop_to_mask():
    image_test = np.ones([100, 100, 3], dtype=np.uint8)
    image_test[:50] = 0
    image_test *= 255
    label_test = np.zeros([100, 100])
    label_test[25:76, 25:76] = 1
    crop_test, mask_test = gimage.crop_to_mask(image_test, label_test, with_label=True, smoothing=True)
    assert crop_test.shape == (50, 50, 3)
    assert mask_test.shape == (50, 50)
    assert np.all(crop_test[:25] == 0)
    assert np.all(crop_test[25:] == 255)

    crop_test_2 = gimage.crop_to_mask(image_test, label_test, with_label=False, smoothing=False)
    assert crop_test_2.shape == (50, 50, 3)
    assert crop_test_2.shape == (50, 50, 3)


def test_get_bounds():
    label_test = np.zeros([100, 100])
    label_test[25:75, 25:75] = 1
    (x0, x1), (y0, y1) = gimage.get_bounds(label_test)
    print(gimage.get_bounds(label_test))
    assert x0 == 25
    assert x1 == 75
    assert y0 == 25
    assert y1 == 75

    # non-zero background value
    label_bg = np.full([100, 100], 2)
    label_bg[25:75, 25:75] = 1
    (x0, x1), (y0, y1) = gimage.get_bounds(label_bg, bg_val=2)
    assert x0 == 25
    assert x1 == 75
    assert y0 == 25
    assert y1 == 75

    # as_slice=True
    bounds_slice = gimage.get_bounds(label_test, as_slice=True)
    assert bounds_slice == (slice(25, 75), slice(25, 75))
    assert label_test[bounds_slice].shape == (50, 50)


def test_crop_to_content():
    image_test = np.zeros([100, 100, 3], dtype=np.uint8)
    image_test[25:75, 25:75] = 1
    crop_test = gimage.crop_to_content(image_test)
    assert crop_test.shape == (50, 50, 3)
    assert crop_test.sum() == crop_test.size

    image, bounds = gimage.crop_to_content(image_test, return_bounds=True)
    assert bounds == [(25, 75), (25, 75), (0, 3)]


def test_get_mask_border():
    x = np.zeros([500, 500], dtype=bool)
    x[200:300, 200:300] = 1
    border = gimage.get_mask_border(x, inside_border=True, border_thickness=2)
    assert border.dtype == bool
    expected = np.zeros([500, 500], dtype=bool)
    expected[200:202, 200:300] = 1
    expected[298:300, 200:300] = 1
    expected[200:300, 200:202] = 1
    expected[200:300, 298:300] = 1
    np.testing.assert_array_equal(border, expected)

    x = np.zeros([500, 500], dtype=bool)
    x[200:300, 200:300] = 1
    border = gimage.get_mask_border(x, inside_border=False, border_thickness=2)
    assert border.dtype == bool
    expected = np.zeros([500, 500], dtype=bool)
    expected[198, 200:300] = 1
    expected[199, 198:302] = 1
    expected[300, 198:302] = 1
    expected[301, 200:300] = 1
    expected[199:301, 198:200] = 1
    expected[199:301, 300:302] = 1
    np.testing.assert_array_equal(border, expected)

    border = gimage.get_mask_border(x, inside_border=False, border_thickness=2, kernel=cv2.MORPH_CROSS)
    assert border.dtype == bool
    expected = np.zeros([500, 500], dtype=bool)
    expected[198, 200:300] = 1
    expected[199, 198:302] = 1
    expected[300, 198:302] = 1
    expected[301, 200:300] = 1
    expected[199:301, 198:200] = 1
    expected[199:301, 300:302] = 1
    np.testing.assert_array_equal(border, expected)


def test_add_mask():
    image = np.zeros([100, 100], dtype=np.float32)
    label = np.zeros([100, 100], dtype=np.uint8)
    label[:50] = 229  # "red" isn't exactly 255, it's slightly less

    overlay = gimage.add_mask(image, label, color="g", opacity=1)
    assert overlay[:, :, [0, 2]].sum() == 0
    expected = np.zeros([100, 100, 3], dtype=np.float32)
    expected[:50, :, 1] = 0.5
    np.testing.assert_array_equal(overlay, expected)

    image = image.astype(np.uint8)
    overlay = gimage.add_mask(image, label, color="red", opacity=1)
    assert overlay[:, :, 1:].sum() == 0
    np.testing.assert_array_equal(overlay[:, :, 0], label)

    label_2 = np.zeros([100, 100, 1], dtype=np.uint8)
    label_2[:50] = 255

    image = np.dstack([image] * 3)
    overlay_2 = gimage.add_mask(image, label_2, color="red", opacity=1)

    np.testing.assert_array_equal(overlay, overlay_2)
    overlays = gimage.add_mask([image, image], label_2, color="red", opacity=1)
    np.testing.assert_array_equal(overlays[0], overlays[1])
    np.testing.assert_array_equal(overlays[0], overlay_2)

    label = label > 0.5
    overlay_2 = gimage.add_mask(image, label, color="red", opacity=1)
    np.testing.assert_array_equal(overlay_2, overlay)
    overlay_2b = gimage.add_mask(image.astype(bool), label, color="red", opacity=1)
    np.testing.assert_allclose(overlay_2b, overlay / 255.0, atol=1e-7)  # Use atol for floating point comparison

    label_3 = np.zeros([100, 100, 3], dtype=np.uint8)
    label_3[:50] = 255
    with pytest.raises(ValueError):
        gimage.add_mask(image, label_3, color="red", opacity=1)
    with pytest.raises(ValueError):
        gimage.add_mask(image, label, color="red", opacity=1.5)
    label = np.zeros([103, 100, 1], dtype=np.float32)
    with pytest.raises(ValueError):
        gimage.add_mask(image, label, color="red", opacity=1)

    label = np.zeros([100, 100], dtype=np.float32)
    label[:50] = 255
    with pytest.raises(ValueError):
        gimage.split_signs(np.dstack([label, label, label]))

    label *= 10
    with pytest.warns(UserWarning):
        gouda.to_uint8(label)


def test_mask_by_triplet():
    pred = np.zeros([100, 100], dtype=np.float32)
    # Large blob: base at 0.5, peak at 0.9 (50x50 = 2500 pixels > area_thresh=2000)
    pred[5:65, 5:65] = 0.5
    pred[10:60, 10:60] = 0.9
    # Small blob: base at 0.5, peak at 0.9 (5x5 = 25 pixels < area_thresh=2000)
    pred[75:90, 75:90] = 0.5
    pred[80:85, 80:85] = 0.9

    result_fast = gimage.mask_by_triplet(pred, lower_thresh=0.3, upper_thresh=0.75, area_thresh=2000, fast=True)
    assert result_fast[35, 35]  # inside large base
    assert result_fast[10, 10]  # inside large peak
    assert not result_fast[82, 82]  # inside small blob

    result_slow = gimage.mask_by_triplet(pred, lower_thresh=0.3, upper_thresh=0.75, area_thresh=2000, fast=False)
    np.testing.assert_array_equal(result_fast, result_slow)


def test_fast_label():
    x = np.zeros([100, 100])
    x[5:10, 5:10] = 1
    x[50:55, 50:55] = -1
    x[20:23, 90:91] = 50
    label = gimage.fast_label(x)
    np.testing.assert_array_equal(np.unique(label), [0, 1, 2, 3])
    assert label[5, 5] > 0
    assert label[50, 50] > 0
    assert label[20, 90] > 0
