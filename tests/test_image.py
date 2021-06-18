import os

import pytest

import cv2
import numpy as np

from gouda import GoudaPath, image as gimage, RGB, GRAYSCALE, UNCHANGED


def test_imwrite_imread():
    with pytest.raises(ValueError):
        gimage.imread('alskdfjalsdjf/asdf.png')

    image_test = np.ones([100, 100, 3], dtype=np.uint8)
    image_test[:50] *= 0
    image_test[:, :, 2] *= 0
    image_test *= 255

    assert image_test[:, :, 2].sum() == 0
    gimage.imwrite(GoudaPath('ScratchFiles/test_RGB.png'), image_test)
    gimage.imwrite('ScratchFiles/test_BGR.png', image_test, as_RGB=False)
    gimage.imwrite('ScratchFiles/test_singleChannel.png', image_test[:, :, :1])
    gimage.imwrite('ScratchFiles/test_2D.png', image_test[:, :, 0])
    gimage.imwrite('ScratchFiles/test_uint16.png', image_test.astype(np.uint16))
    with pytest.raises(ValueError):
        assert gimage.imwrite('ScratchFiles/failure.png', image_test[:, :, :2])

    assert os.path.isfile('ScratchFiles/test_RGB.png')
    assert os.path.isfile('ScratchFiles/test_BGR.png')
    assert os.path.isfile('ScratchFiles/test_singleChannel.png')
    assert os.path.isfile('ScratchFiles/test_2D.png')
    assert os.path.isfile('ScratchFiles/test_uint16.png')
    assert not os.path.isfile('ScratchFiles/failure.png')

    image_test_in_1 = gimage.imread(GoudaPath('ScratchFiles/test_RGB.png'), flag=RGB)
    image_test_in_2 = gimage.imread('ScratchFiles/test_RGB.png', flag=None)
    image_test_in_3 = gimage.imread('ScratchFiles/test_BGR.png', flag=RGB)
    image_test_in_4 = gimage.imread('ScratchFiles/test_BGR.png', flag=None)
    np.testing.assert_array_equal(image_test_in_1, image_test_in_4)
    np.testing.assert_array_equal(image_test_in_2, image_test_in_3)

    image_test_in_5 = gimage.imread('ScratchFiles/test_singleChannel.png')
    image_test_in_6 = gimage.imread('ScratchFiles/test_singleChannel.png')

    np.testing.assert_array_equal(image_test_in_5, image_test_in_6)

    image_test_in_7 = gimage.imread('ScratchFiles/test_uint16.png', flag=UNCHANGED)
    assert image_test_in_7.dtype == np.uint16

    image_test_in_8 = gimage.imread('ScratchFiles/test_RGB.png', flag=GRAYSCALE)
    assert image_test_in_8.shape == (100, 100)
    np.testing.assert_allclose(image_test_in_8, cv2.cvtColor(image_test, cv2.COLOR_RGB2GRAY), rtol=0, atol=1)

    os.remove('ScratchFiles/test_RGB.png')
    os.remove('ScratchFiles/test_BGR.png')
    os.remove('ScratchFiles/test_singleChannel.png')
    os.remove('ScratchFiles/test_2D.png')
    os.remove('ScratchFiles/test_uint16.png')


def test_rescale():
    image_test = np.ones([10, 10, 3], dtype=np.int)
    image_test[:5] -= 1
    image_test[-2:] += 1
    image_test *= 100
    rescaled_1 = np.unique(gimage.rescale(image_test))
    rescaled_2 = np.unique(gimage.rescale(image_test, max_val=2, min_val=0))
    rescaled_3 = np.unique(gimage.rescale(image_test, max_val=6, min_val=3))
    np.testing.assert_array_equal(rescaled_1, (0, 0.5, 1))
    np.testing.assert_array_equal(rescaled_2, (0, 1, 2))
    np.testing.assert_array_equal(rescaled_3, (3, 4.5, 6))

    image_test_2 = np.ones([10, 10], dtype=np.int)
    rescale_4 = gimage.rescale(image_test_2, return_type=np.int)
    assert rescale_4.max() == rescale_4.min()
    assert rescale_4[0, 0] == 0
    assert rescale_4.dtype == 'int'
    rescale_5 = gimage.rescale(image_test_2, return_type=np.float32)
    assert rescale_5.max() == rescale_5.min()
    assert rescale_5[0, 0] == 0
    assert rescale_5.dtype == 'float32'


def test_rescale_columnwise():
    image_test = np.ones([5, 5])
    temp_1 = np.array([0, 0, 2, 4, 4])
    temp_1 = np.stack([temp_1, temp_1], axis=0).transpose([1, 0])
    image_test[:, :2] = image_test[:, :2] * temp_1
    np.testing.assert_array_equal(image_test[:, :2], temp_1)
    temp_2 = np.array([1, 1, 2, 2, 2])
    temp_2 = np.stack([temp_2, temp_2, temp_2], axis=0).transpose([1, 0])
    image_test[:, 2:] *= temp_2
    np.testing.assert_array_equal(image_test[:, 2:], temp_2)
    rescaled_1 = gimage.rescale(image_test, column_wise=True)
    np.testing.assert_array_equal(image_test, np.array([[0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [2, 2, 2, 2, 2], [4, 4, 2, 2, 2], [4, 4, 2, 2, 2]]))
    np.testing.assert_array_equal(np.unique(rescaled_1[:, :2]), (0, 0.5, 1))
    np.testing.assert_array_equal(np.unique(rescaled_1[:, 2:]), (0, 1))

    image_test_2 = np.ones([5, 5])
    rescaled_2 = gimage.rescale(image_test_2, column_wise=True)
    assert rescaled_2.max() == rescaled_2.min()
    assert rescaled_2[0, 0] == 0


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


def test_add_overlay():
    image_1 = np.zeros([100, 100], dtype=np.uint8)
    label_1 = np.zeros([100, 100], dtype=np.uint8)
    label_1[:50] = 255
    overlay1 = gimage.add_overlay(image_1, label_1)
    np.testing.assert_array_equal(np.unique(overlay1), np.array([0, 128]))

    overlay1b = gimage.add_overlay(image_1, np.dstack([label_1, np.zeros([100, 100]), np.zeros([100, 100])]))
    np.testing.assert_array_equal(overlay1, overlay1b)

    label_2 = np.copy(label_1).astype(np.float)
    label_2[:50] = 1
    overlay2 = gimage.add_overlay(image_1[:, :, np.newaxis], label_2)
    np.testing.assert_array_equal(overlay1, overlay2)

    overlay2b = gimage.add_overlay(image_1.astype(np.float32), label_2.astype(np.float32))
    np.testing.assert_array_equal(np.sign(overlay2) * 0.5, overlay2b)

    image_2 = np.zeros([100, 100, 3], dtype=np.uint8)
    overlay3 = gimage.add_overlay(image_2, label_2)
    np.testing.assert_array_equal(overlay1, overlay3)

    label_3 = np.copy(label_2)
    label_3[:50] = 255
    overlay4 = gimage.add_overlay(image_2, label_3)
    np.testing.assert_array_equal(overlay1, overlay4)

    label_4 = np.copy(label_2)
    label_4[50:] = -1
    overlay5 = gimage.add_overlay(image_2, label_4)
    np.testing.assert_array_equal(overlay5[0, 0], np.array([128, 64, 64]))
    np.testing.assert_array_equal(overlay5[-1, -1], np.array([0, 64, 64]))

    overlay6 = gimage.add_overlay(image_2, label_4, separate_signs=True)
    np.testing.assert_array_equal(overlay6[0, 0], np.array([0, 128, 0]))
    np.testing.assert_array_equal(overlay6[-1, -1], np.array([128, 0, 0]))

    label_5 = label_3 * 10
    with pytest.warns(UserWarning):
        gimage.to_uint8(label_5)

    with pytest.raises(ValueError):
        bad_label = np.ones([50, 50])
        gimage.add_overlay(image_1, bad_label)

    with pytest.raises(ValueError):
        bad_label = np.ones([100, 100, 50])
        gimage.add_overlay(image_1, bad_label)

    with pytest.raises(ValueError):
        gimage.split_signs(np.dstack([label_1, label_1, label_1]))


def test_masked_lineup():
    image_test = np.ones([100, 100, 3], dtype=np.uint8)
    image_test[:50] -= 1
    image_test *= 255
    label_test = np.ones([100, 100], dtype=np.uint8)
    label_test[:, :50] -= 1
    lineup_test = gimage.masked_lineup(image_test, label_test)
    np.testing.assert_array_equal(lineup_test[0], image_test)
    np.testing.assert_array_equal(lineup_test[1], gimage.add_overlay(image_test, label_test))
    np.testing.assert_array_equal(lineup_test[2][:, :, 0], label_test * 255)


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

    # assert (cv2.GC_BGD, cv2.GC_PR_BGD, cv2.GC_PR_FGD, cv2.GC_FGD) == 1

    label_test_2 = np.ones([100, 100], dtype=np.float)
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
    label_test[25:76, 25:76] = 1
    (x0, y0), (x1, y1) = gimage.get_bounds(label_test)
    assert x0 == 25
    assert x1 == 75
    assert y0 == 25
    assert y1 == 75


def test_crop_to_content():
    image_test = np.zeros([100, 100, 3], dtype=np.uint8)
    image_test[25:76, 25:76] = 1
    crop_test = gimage.crop_to_content(image_test)
    assert crop_test.shape == (50, 50, 3)
    assert crop_test.sum() == crop_test.size

    bounds = gimage.crop_to_content(image_test, return_bounds=True)
    assert bounds == ((25, 75), (25, 75))


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
    gimage.imwrite("ScratchFiles/test.png", image_test)
    pad_test = gimage.padded_resize(image_test, size=[50, 50])
    pad_test_file = gimage.padded_resize("ScratchFiles/test.png", size=[50, 50])
    assert pad_test.shape == (50, 50, 3)
    np.testing.assert_array_equal(image_test, pad_test)
    np.testing.assert_array_equal(pad_test_file, pad_test)
    os.remove('ScratchFiles/test.png')

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


def test_flips():
    image_test = np.ones([50, 50, 3], dtype=np.uint8)
    image_test[:25] = 0
    flip_1 = gimage.horizontal_flip(image_test)
    np.testing.assert_array_equal(image_test, flip_1)
    flip_2 = gimage.vertical_flip(image_test)
    assert flip_2[:25].sum() == flip_2[:25].size

    image_test_2 = np.ones([50, 50, 3], dtype=np.uint8)
    image_test_2[:, :25] = 0
    flip_3 = gimage.horizontal_flip(image_test_2)
    assert flip_3[:, :25].sum() == flip_3[:, :25].size
    flip_4 = gimage.vertical_flip(image_test_2)
    np.testing.assert_array_equal(flip_4, image_test_2)


def test_adjust_gamma():
    image_test = np.ones([50, 50, 3], dtype=np.uint8) * np.linspace(0, 255, num=50, endpoint=True, dtype=np.uint8).reshape([50, 1, 1])
    assert image_test.dtype == 'uint8'
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
    assert(y[50, 0] == z[100, 0]).all()
    assert(y[100, 0] == z[200, 0]).all()


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
    label[:50] = 255

    overlay = gimage.add_mask(image, label, color='g', opacity=1)
    assert overlay[:, :, [0, 2]].sum() == 0
    expected = np.zeros([100, 100, 3], dtype=np.float32)
    expected[:50, :, 1] = 0.5
    np.testing.assert_array_equal(overlay, expected)

    image = image.astype(np.uint8)
    overlay = gimage.add_mask(image, label, color='red', opacity=1)
    assert overlay[:, :, 1:].sum() == 0
    np.testing.assert_array_equal(overlay[:, :, 0], label)

    label_2 = np.zeros([100, 100, 1], dtype=np.uint8)
    label_2[:50] = 255

    image = np.dstack([image] * 3)
    overlay_2 = gimage.add_mask(image, label_2, color='red', opacity=1)

    np.testing.assert_array_equal(overlay, overlay_2)
    overlays = gimage.add_mask([image, image], label_2, color='red', opacity=1)
    np.testing.assert_array_equal(overlays[0], overlays[1])
    np.testing.assert_array_equal(overlays[0], overlay_2)

    label = label > 0.5
    overlay_2 = gimage.add_mask(image, label, color='red', opacity=1)
    np.testing.assert_array_equal(overlay_2, overlay)

    label_3 = np.zeros([100, 100, 3], dtype=np.uint8)
    label_3[:50] = 255
    with pytest.raises(ValueError):
        gimage.add_mask(image, label_3, color='red', opacity=1)
    with pytest.raises(ValueError):
        gimage.add_mask(image, label, color='red', opacity=1.5)
    label = np.zeros([103, 100, 1], dtype=np.float32)
    with pytest.raises(ValueError):
        gimage.add_mask(image, label, color='red', opacity=1)

    label = np.zeros([100, 100], dtype=np.float32)
    label[:50] = 255
    with pytest.raises(ValueError):
        gimage.split_signs(np.dstack([label, label, label]))

    label *= 10
    with pytest.warns(UserWarning):
        gimage.to_uint8(label)
