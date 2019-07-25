import numpy as np
from matplotlib.testing.decorators import image_comparison

from gouda import display


def test_denorm_full():
    full_range = np.ones([10, 10], dtype=np.uint8) * 255
    full_range[:5] -= 255
    denormed = display._denorm(full_range)
    assert np.max(denormed) == 255
    assert np.min(denormed) == 0
    assert denormed.dtype == 'uint8'

    denormed1 = display._denorm(full_range, norm_type=display.FULL_RANGE)
    assert np.max(denormed1) == 255
    assert np.min(denormed1) == 0
    assert denormed1.dtype == 'uint8'

    np.testing.assert_array_equal(denormed, denormed1)


def test_denorm_tanh():
    full_range = np.ones([10, 10], dtype=np.float32)
    full_range[:5] -= 2
    denormed = display._denorm(full_range)
    assert np.max(denormed) == 255
    assert np.min(denormed) == 0
    assert denormed.dtype == 'uint8'

    denormed1 = display._denorm(full_range, norm_type=display.TANH)
    assert np.max(denormed1) == 255
    assert np.min(denormed1) == 0
    assert denormed1.dtype == 'uint8'

    np.testing.assert_array_equal(denormed, denormed1)


def test_denorm_sigmoid():
    full_range = np.ones([10, 10], dtype=np.float32)
    full_range[:5] -= 1
    denormed = display._denorm(full_range)
    assert np.max(denormed) == 255
    assert np.min(denormed) == 0
    assert denormed.dtype == 'uint8'

    denormed1 = display._denorm(full_range, norm_type=display.SIGMOID)
    assert np.max(denormed1) == 255
    assert np.min(denormed1) == 0
    assert denormed1.dtype == 'uint8'

    np.testing.assert_array_equal(denormed, denormed1)


@image_comparison(baseline_images=['baseline'], remove_text=True, extensions=['png'])
def test_show_image():
    image = np.ones([100, 100, 3], dtype=np.uint8)
    image[:100] -= 1
    image *= 127
    display.show_image(image)
