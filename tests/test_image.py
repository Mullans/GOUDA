import os

import pytest

import cv2
import numpy as np

from gouda import GoudaPath, image


def test_imwrite_imread():
    image_test = np.ones([100, 100, 3], dtype=np.uint8)
    image_test[:50] *= 0
    image_test[:, :, 2] *= 0
    image_test *= 255

    assert image_test[:, :, 2].sum() == 0
    image.imwrite(GoudaPath('test_RGB.png'), image_test)
    image.imwrite('test_BGR.png', image_test, as_RGB=False)
    image.imwrite('test_singleChannel.png', image_test[:, :, :1])
    image.imwrite('test_2D.png', image_test[:, :, 0])
    image.imwrite('test_uint16.png', image_test.astype(np.uint16))
    with pytest.raises(ValueError):
        assert image.imwrite('failure.png', image_test[:, :, :2])

    assert os.path.isfile('test_RGB.png')
    assert os.path.isfile('test_BGR.png')
    assert os.path.isfile('test_singleChannel.png')
    assert os.path.isfile('test_2D.png')
    assert os.path.isfile('test_uint16.png')
    assert not os.path.isfile('failure.png')

    image_test_in_1 = image.imread(GoudaPath('test_RGB.png'), as_RGB=True, unchanged=False)
    image_test_in_2 = image.imread('test_RGB.png', as_RGB=False, unchanged=False)
    image_test_in_3 = image.imread('test_BGR.png', as_RGB=True, unchanged=False)
    image_test_in_4 = image.imread('test_BGR.png', as_RGB=False, unchanged=False)
    np.testing.assert_array_equal(image_test_in_1, image_test_in_4)
    np.testing.assert_array_equal(image_test_in_2, image_test_in_3)

    image_test_in_5 = image.imread('test_singleChannel.png')
    image_test_in_6 = image.imread('test_singleChannel.png')

    np.testing.assert_array_equal(image_test_in_5, image_test_in_6)

    image_test_in_7 = image.imread('test_uint16.png', unchanged=True)
    assert image_test_in_7.dtype == np.uint16

    image_test_in_8 = image.imread('test_RGB.png', as_greyscale=True, unchanged=False)
    assert image_test_in_8.shape == (100, 100)
    np.testing.assert_allclose(image_test_in_8, cv2.cvtColor(image_test, cv2.COLOR_RGB2GRAY), rtol=0, atol=1)

    os.remove('test_RGB.png')
    os.remove('test_BGR.png')
    os.remove('test_singleChannel.png')
    os.remove('test_2D.png')
    os.remove('test_uint16.png')


def test_val_type():
    image_test = np.ones([100, 100, 3])
    image_test[:50] -= 1
    assert image.val_type(image_test) == image.SIGMOID
    image_test[:50] -= 1
    assert image.val_type(image_test) == image.TANH
    image_test[:50] += 1
    image_test *= 255
    assert image.val_type(image_test) == image.FULL_RANGE

    image_test += 1
    with pytest.raises(ValueError):
        assert image.val_type(image_test)


def test_denorm():
    image_test = np.ones([100, 100, 3], dtype=np.int)
    image_test[:50] -= 1
    assert image_test.dtype == 'int'
    denorm_image_1 = image.denorm(image_test)
    denorm_image_2 = image.denorm(image_test, norm_type=image.SIGMOID)
    np.testing.assert_array_equal(denorm_image_1, denorm_image_2)
    assert denorm_image_1.dtype == 'uint8'
    assert denorm_image_2.dtype == 'uint8'
    assert denorm_image_1.max() == 255
    assert denorm_image_1.min() == 0
    assert image_test.dtype == 'int'

    image_test[:50] -= 1
    denorm_image_3 = image.denorm(image_test)
    denorm_image_4 = image.denorm(image_test, norm_type=image.TANH)
    np.testing.assert_array_equal(denorm_image_3, denorm_image_4)
    np.testing.assert_array_equal(denorm_image_1, denorm_image_3)
    assert denorm_image_3.dtype == 'uint8'
    assert denorm_image_4.dtype == 'uint8'
    assert image_test.dtype == 'int'

    image_test[:50] += 1
    image_test *= 255
    denorm_image_5 = image.denorm(image_test)
    denorm_image_6 = image.denorm(image_test, norm_type=image.FULL_RANGE)
    np.testing.assert_array_equal(denorm_image_5, denorm_image_6)
    np.testing.assert_array_equal(denorm_image_1, denorm_image_5)
    assert denorm_image_5.dtype == 'uint8'
    assert denorm_image_6.dtype == 'uint8'
    assert image_test.dtype == 'int'

    image_test = image_test.astype(np.int)
    image_test[:50] = image_test[:50] + 265
    with pytest.raises(ValueError):
        assert image.denorm(image_test)


def test_rescale():
    image_test = np.ones([10, 10, 3], dtype=np.int)
    image_test[:5] -= 1
    image_test[-2:] += 1
    image_test *= 100
    rescaled_1 = np.unique(image.rescale(image_test))
    rescaled_2 = np.unique(image.rescale(image_test, max_val=2, min_val=0))
    rescaled_3 = np.unique(image.rescale(image_test, max_val=6, min_val=3))
    np.testing.assert_array_equal(rescaled_1, (0, 0.5, 1))
    np.testing.assert_array_equal(rescaled_2, (0, 1, 2))
    np.testing.assert_array_equal(rescaled_3, (3, 4.5, 6))

    image_test_2 = np.ones([10, 10], dtype=np.int)
    rescale_4 = image.rescale(image_test_2, return_type=np.int)
    assert rescale_4.max() == rescale_4.min()
    assert rescale_4[0, 0] == 0
    assert rescale_4.dtype == 'int'
    rescale_5 = image.rescale(image_test_2, return_type=np.float32)
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
    rescaled_1 = image.rescale(image_test, column_wise=True)
    np.testing.assert_array_equal(image_test, np.array([[0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [2, 2, 2, 2, 2], [4, 4, 2, 2, 2], [4, 4, 2, 2, 2]]))
    np.testing.assert_array_equal(np.unique(rescaled_1[:, :2]), (0, 0.5, 1))
    np.testing.assert_array_equal(np.unique(rescaled_1[:, 2:]), (0, 1))

    image_test_2 = np.ones([5, 5])
    rescaled_2 = image.rescale(image_test_2, column_wise=True)
    assert rescaled_2.max() == rescaled_2.min()
    assert rescaled_2[0, 0] == 0


def test_stack_label():
    label_test = np.ones([10, 10])
    stacked = image.stack_label(label_test, should_denorm=False)
    assert stacked.shape == (10, 10, 3)

    stacked_2 = image.stack_label(label_test, should_denorm=True)
    denormed = image.denorm(label_test)
    np.testing.assert_array_equal(stacked_2[:, :, 0], denormed)


def test_laplacian_var():
    image_test = np.zeros([100, 100, 3], dtype=np.uint8)
    assert image.laplacian_var(image_test) == 0

    image_test[:50] += 255
    assert image.laplacian_var(image_test) == 161.3


def test_sobel_var():
    image_test = np.zeros([100, 100, 3], dtype=np.uint8)
    assert image.sobel_var(image_test) == 0

    image_test[:50] += 255
    assert image.sobel_var(image_test) == 629.1456


def test_mask():
    image_test = np.ones([100, 100, 3], dtype=np.uint8)
    image_test[:50] -= 1
    image_test *= 255
    label_test = np.ones([100, 100], dtype=np.uint8)
    label_test[:, :50] -= 1
    mask_test_1 = image.mask(image_test, label_test, renorm=True, norm_type=image.FULL_RANGE)
    assert mask_test_1.max() == 255
    assert mask_test_1.min() == 0
    layer_test = cv2.addWeighted(image_test[:, :, 0], 0.8, label_test * 255, 0.2, 0)
    overlay_test = np.where(label_test > 0, layer_test, image_test[:, :, 0])
    np.testing.assert_array_equal(mask_test_1[:, :, 0], overlay_test)

    image_test_2 = np.ones([100, 100, 3], dtype=np.float32)
    image_test_2[:50] -= 1
    assert image_test_2.max() == 1
    assert image_test_2.min() == 0
    mask_test_2 = image.mask(image_test_2, label_test[:, :, np.newaxis], norm_type=image.SIGMOID)
    assert mask_test_2.min() == 0
    assert mask_test_2.max() == 1
    np.testing.assert_array_equal(mask_test_2 * 255, mask_test_1)

    image_test_3 = np.ones([100, 100, 3], dtype=np.float32)
    image_test_3[:50] -= 2
    assert image_test_3.max() == 1
    assert image_test_3.min() == -1
    mask_test_3 = image.mask(image_test_3, label_test[:, :, np.newaxis], norm_type=image.TANH)
    assert mask_test_3.max() == 1
    assert mask_test_3.min() == -1
    np.testing.assert_array_equal(mask_test_3, (mask_test_1 - 127.5) / 127.5)

    mask_test_4 = image.mask(image_test_2, label_test, renorm=False)
    assert mask_test_4.dtype == 'uint8'
    np.testing.assert_array_equal(mask_test_4, mask_test_1)

    mask_test_5 = image.mask(image_test, label_test, overlay=False)
    np.testing.assert_array_equal(mask_test_5[:, :, 0] / 255, label_test)
    np.testing.assert_array_equal(mask_test_5[:, :, 1:], image_test[:, :, 1:])

    mask_test_6 = image.mask(image_test[:, :, 0], label_test)
    np.testing.assert_array_equal(mask_test_6, mask_test_1)


def test_mask_exception():
    image_test = np.ones([100, 100, 3], dtype=np.uint8)
    image_test[:50] -= 1
    image_test *= 255
    label_test = np.ones([100, 100], dtype=np.uint8)
    label_test[:, :50] -= 1
    with pytest.raises(ValueError):
        assert image.mask(image_test, np.dstack([label_test, label_test]))
    with pytest.raises(ValueError):
        assert image.mask(image_test, label_test * 2)


def test_masked_lineup():
    image_test = np.ones([100, 100, 3], dtype=np.uint8)
    image_test[:50] -= 1
    image_test *= 255
    label_test = np.ones([100, 100], dtype=np.uint8)
    label_test[:, :50] -= 1
    lineup_test = image.masked_lineup(image_test, label_test)
    np.testing.assert_array_equal(lineup_test[0], image_test)
    np.testing.assert_array_equal(lineup_test[1], image.mask(image_test, label_test))
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
    gc_img, gc_mask = image.grabCut(image_test, label_test, iterations=1)
    np.testing.assert_array_equal(gc_img, cv2.bitwise_and(image_test, image_test, mask=gc_mask))

    # assert (cv2.GC_BGD, cv2.GC_PR_BGD, cv2.GC_PR_FGD, cv2.GC_FGD) == 1

    label_test_2 = np.ones([100, 100], dtype=np.float)
    label_test_2[:25] = 0
    label_test_2[25:50] = 0.25
    label_test_2[50:75] = 0.5
    label_test_2[75:] = 0.75
    gc_img_2, gc_mask_2 = image.grabCut(image_test, label_test_2, thresholds=(0.2, 0.4, 0.7), iterations=1)
    np.testing.assert_array_equal(gc_img, gc_img_2)
    np.testing.assert_array_equal(gc_mask, gc_mask_2)
    gc_img_3, gc_mask_3 = image.grabCut(image_test, label_test_2, thresholds=(0.2, 0.4, 0.7), iterations=1, clean=True)
    clean_test = image.clean_grabCut_mask(gc_mask)
    np.testing.assert_array_equal(clean_test, gc_mask_3)
    clean_test_2 = image.clean_grabCut_mask(gc_mask[:, :, np.newaxis])
    np.testing.assert_array_equal(clean_test_2, gc_mask_3[:, :, np.newaxis])


def test_grabCut_exception():
    image_test = np.ones([100, 100, 3], dtype=np.uint8)
    image_test[:50] -= 1
    image_test *= 255
    label_test = np.ones([100, 100], dtype=np.uint8)
    label_test[:50] = 0
    label_test[50:] = 2
    with pytest.raises(ValueError):
        assert image.grabCut(image_test, label_test)
    label_test[:50] = 3
    label_test[50:] = 1
    with pytest.raises(ValueError):
        assert image.grabCut(image_test, label_test)


def test_crop_to_mask():
    image_test = np.ones([100, 100, 3], dtype=np.uint8)
    image_test[:50] = 0
    image_test *= 255
    label_test = np.zeros([100, 100])
    label_test[25:76, 25:76] = 1
    crop_test, mask_test = image.crop_to_mask(image_test, label_test, with_label=True, smoothing=True)
    assert crop_test.shape == (50, 50, 3)
    assert mask_test.shape == (50, 50)
    assert np.all(crop_test[:25] == 0)
    assert np.all(crop_test[25:] == 255)

    crop_test_2 = image.crop_to_mask(image_test, label_test, with_label=False, smoothing=False)
    assert crop_test_2.shape == (50, 50, 3)
    assert crop_test_2.shape == (50, 50, 3)


def test_get_bounds():
    label_test = np.zeros([100, 100])
    label_test[25:76, 25:76] = 1
    (x0, y0), (x1, y1) = image.get_bounds(label_test)
    assert x0 == 25
    assert x1 == 75
    assert y0 == 25
    assert y1 == 75


def test_crop_to_content():
    image_test = np.zeros([100, 100, 3], dtype=np.uint8)
    image_test[25:76, 25:76] = 1
    crop_test = image.crop_to_content(image_test)
    assert crop_test.shape == (50, 50, 3)
    assert crop_test.sum() == crop_test.size


def test_rotate():
    image_test = np.zeros([50, 50, 3], dtype=np.uint8)
    image_test[:25, :25] = 1
    assert image_test.sum() == 25 * 25 * 3
    rotate_test = image.rotate(image_test, degrees=90)
    assert rotate_test.shape == (50, 50, 3)
    assert rotate_test.sum() == image_test.sum()
    assert rotate_test[:25, :25].sum() == 0
    assert rotate_test[:25, 25:].sum() == 25 * 25 * 3

    rotate_test_2 = image.rotate(image_test, degrees=180)
    np.testing.assert_array_equal(rotate_test_2, image_test[::-1, ::-1])

    np.testing.assert_array_equal(image.rotate(image_test, degrees=-90), image.rotate(image_test, degrees=270))
    np.testing.assert_array_equal(image_test, image.rotate(image_test, degrees=360))


def test_rotate_allow_resize():
    image_test = np.zeros([100, 50, 3], dtype=np.uint8)
    rotated_1 = image.rotate(image_test, degrees=90, allow_resize=True)
    assert rotated_1.shape == (50, 100, 3)
    rotated_2 = image.rotate(image_test, degrees=90, allow_resize=False)
    assert rotated_2.shape == (100, 50, 3)


def test_padded_resize():
    image_test = np.ones([50, 50, 3], dtype=np.uint8)
    image.imwrite("test.png", image_test)
    pad_test = image.padded_resize(image_test, size=[50, 50])
    pad_test_file = image.padded_resize("test.png", size=[50, 50])
    assert pad_test.shape == (50, 50, 3)
    np.testing.assert_array_equal(image_test, pad_test)
    np.testing.assert_array_equal(pad_test_file, pad_test)
    os.remove('test.png')

    pad_test_2 = image.padded_resize(image_test, size=[50, 75])
    assert pad_test_2.shape == (50, 75, 3)
    assert pad_test_2[:, :12].sum() == 0
    assert pad_test_2[:, -12:].sum() == 0

    pad_test_3 = image.padded_resize(image_test, size=[75, 75])
    assert pad_test_3.shape == (75, 75, 3)
    assert pad_test_3.sum() == pad_test_3.size

    image_test_2 = np.ones([50, 50, 1], dtype=np.uint8)
    pad_test_4 = image.padded_resize(image_test_2, size=[50, 50])
    np.testing.assert_array_equal(pad_test_4, image_test_2)

    image_test_3 = np.ones([50, 50], dtype=np.uint8)
    pad_test_5 = image.padded_resize(image_test_3, size=[50, 50])
    assert pad_test_5.shape == (50, 50)
    np.testing.assert_array_equal(pad_test_4[:, :, 0], pad_test_5)

    image_test_4 = np.zeros([25, 50, 3], dtype=np.uint8)
    image_test_4[:10] = 1
    pad_test_6 = image.padded_resize(image_test_4, size=(50, 25, 3))
    assert pad_test_6.shape == (50, 25, 3)
    np.testing.assert_array_equal(pad_test_6, image.rotate(image_test_4, degrees=90))

    pad_test_7 = image.padded_resize(image_test, size=[75, 50])
    assert pad_test_7.shape == (75, 50, 3)
    assert pad_test_7[:12].sum() == 0
    assert pad_test_7[-12:].sum() == 0

    pad_test_7 = image.padded_resize(image_test[:, :, 0], size=[75, 50])
    assert pad_test_7.shape == (75, 50)
    assert pad_test_7[:12].sum() == 0
    assert pad_test_7[-12:].sum() == 0


def test_flips():
    image_test = np.ones([50, 50, 3], dtype=np.uint8)
    image_test[:25] = 0
    flip_1 = image.horizontal_flip(image_test)
    np.testing.assert_array_equal(image_test, flip_1)
    flip_2 = image.vertical_flip(image_test)
    assert flip_2[:25].sum() == flip_2[:25].size

    image_test_2 = np.ones([50, 50, 3], dtype=np.uint8)
    image_test_2[:, :25] = 0
    flip_3 = image.horizontal_flip(image_test_2)
    assert flip_3[:, :25].sum() == flip_3[:, :25].size
    flip_4 = image.vertical_flip(image_test_2)
    np.testing.assert_array_equal(flip_4, image_test_2)


def test_adjust_gamma():
    image_test = np.ones([50, 50, 3], dtype=np.uint8) * np.linspace(0, 255, num=50, endpoint=True, dtype=np.uint8).reshape([50, 1, 1])
    assert image_test.dtype == 'uint8'
    gamma_test_1 = image.adjust_gamma(image_test, gamma=2)
    gamma_test_2 = image.adjust_gamma(image_test, gamma=0.5)
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

    y = image.polar_to_cartesian(x)
    z = image.polar_to_cartesian(x, output_shape=[400, 100, 3])

    assert y.shape == x.shape
    assert z.shape == tuple([400, 100, 3])
    assert (y[0, 0] == color3).all()
    assert (y[50, 0] == color2).all()
    assert (y[100, 0] == color1).all()
    assert (y[0, 0] == z[0, 0]).all()
    assert(y[50, 0] == z[100, 0]).all()
    assert(y[100, 0] == z[200, 0]).all()
