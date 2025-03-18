from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest

from gouda import display


@pytest.mark.skipif(sys.platform == "win32", reason="Bug in matplotlib/tkinter on Windows")
def test_print_grid():
    # List Tests
    image = np.random.randint(0, 255, size=[100, 100])

    # Image
    assert display.print_grid(image, show=False, return_grid_shape=True) == (1, 1)
    plt.close()

    fig, shape = display.print_grid(image, show=False, return_grid_shape=True, return_fig=True)
    assert shape == (1, 1)
    assert isinstance(fig, plt.Figure)
    plt.close()

    # # Multiple images
    assert display.print_grid(image, image, show=False, return_grid_shape=True) == (1, 2)
    plt.close()

    # # List of Images
    assert display.print_grid([image], show=False, return_grid_shape=True) == (1, 1)
    plt.close()
    assert display.print_grid([image, image], show=False, return_grid_shape=True) == (1, 2)
    plt.close()

    # # List of List of Images
    assert display.print_grid([image], show=False, return_grid_shape=True) == (1, 1)
    plt.close()
    assert display.print_grid([[image, image]], show=False, return_grid_shape=True) == (1, 2)
    plt.close()
    assert display.print_grid([[image], [image]], show=False, return_grid_shape=True) == (2, 1)
    plt.close()
    assert display.print_grid([[image, image], [image]], show=False, return_grid_shape=True) == (2, 2)
    plt.close()
    assert display.print_grid([[image, image], [None, image]], show=False, return_grid_shape=True) == (2, 2)
    plt.close()

    # Array Tests
    image_0c = np.random.randint(0, 255, size=[100, 100])
    image_1c = np.random.randint(0, 255, size=[100, 100, 1])
    image_3c = np.random.randint(0, 255, size=[100, 100, 3])

    # [x, y], [x, y, 1], [x, y, 3]
    assert display.print_grid(image_0c, show=False, return_grid_shape=True) == (1, 1)
    plt.close()
    assert display.print_grid(image_1c, show=False, return_grid_shape=True) == (1, 1)
    plt.close()
    assert display.print_grid(image_3c, show=False, return_grid_shape=True) == (1, 1)
    plt.close()

    # [cols, x, y], [cols, x, y, 1], [cols, x, y, 3]
    assert display.print_grid(image_0c[np.newaxis], show=False, return_grid_shape=True) == (1, 1)
    plt.close()
    assert display.print_grid(np.stack([image_0c, image_0c], axis=0), show=False, return_grid_shape=True) == (1, 2)
    plt.close()
    assert display.print_grid(image_1c[np.newaxis], show=False, return_grid_shape=True) == (1, 1)
    plt.close()
    assert display.print_grid(np.stack([image_1c, image_1c], axis=0), show=False, return_grid_shape=True) == (1, 2)
    plt.close()
    assert display.print_grid(image_3c[np.newaxis], show=False, return_grid_shape=True) == (1, 1)
    plt.close()
    assert display.print_grid(np.stack([image_3c, image_3c], axis=0), show=False, return_grid_shape=True) == (1, 2)
    plt.close()

    # [rows, cols, x, y], [rows, cols, x, y, 1], [rows, cols, x, y, 3]
    assert display.print_grid(image_0c[np.newaxis, np.newaxis], show=False, return_grid_shape=True) == (1, 1)
    plt.close()
    assert display.print_grid(
        np.stack([image_0c, image_0c], axis=0)[np.newaxis], show=False, return_grid_shape=True
    ) == (1, 2)
    plt.close()
    assert display.print_grid(
        np.stack([image_0c[np.newaxis], image_0c[np.newaxis]], axis=0), show=False, return_grid_shape=True
    ) == (2, 1)
    plt.close()
    assert display.print_grid(
        np.stack([np.stack([image_0c, image_0c], axis=0), np.stack([image_0c, image_0c], axis=0)]),
        show=False,
        return_grid_shape=True,
    ) == (2, 2)
    plt.close()

    assert display.print_grid(image_1c[np.newaxis, np.newaxis], show=False, return_grid_shape=True) == (1, 1)
    plt.close()
    assert display.print_grid(
        np.stack([image_1c, image_1c], axis=0)[np.newaxis], show=False, return_grid_shape=True
    ) == (1, 2)
    plt.close()
    assert display.print_grid(
        np.stack([image_1c[np.newaxis], image_1c[np.newaxis]], axis=0), show=False, return_grid_shape=True
    ) == (2, 1)
    plt.close()
    assert display.print_grid(
        np.stack([np.stack([image_1c, image_1c], axis=0), np.stack([image_1c, image_1c], axis=0)]),
        show=False,
        return_grid_shape=True,
    ) == (2, 2)
    plt.close()

    assert display.print_grid(image_3c[np.newaxis, np.newaxis], show=False, return_grid_shape=True) == (1, 1)
    plt.close()
    assert display.print_grid(
        np.stack([image_3c, image_3c], axis=0)[np.newaxis], show=False, return_grid_shape=True
    ) == (1, 2)
    plt.close()
    assert display.print_grid(
        np.stack([image_3c[np.newaxis], image_3c[np.newaxis]], axis=0), show=False, return_grid_shape=True
    ) == (2, 1)
    plt.close()
    assert display.print_grid(
        np.stack([np.stack([image_3c, image_3c], axis=0), np.stack([image_3c, image_3c], axis=0)]),
        show=False,
        return_grid_shape=True,
    ) == (2, 2)
    plt.close()

    test_image = {"image": image_3c}
    assert display.print_grid(test_image, show=False, return_grid_shape=True, suptitle="test_sup") == (1, 1)
    test_image = {"image": image_1c}
    assert display.print_grid(test_image, show=False, return_grid_shape=True) == (1, 1)
    test_image = {"image": image_1c, "cmap": "viridis", "title": "test", "xlabel": "testx", "ylabel": "testy"}
    assert display.print_grid(test_image, show=False, return_grid_shape=True, left=0.1) == (1, 1)
    plt.close()

    display.print_grid(np.ones([10, 10], dtype=bool), show=False)
    with pytest.raises(ValueError):
        assert display.print_grid([])


@pytest.mark.skipif(sys.platform == "win32", reason="Bug in matplotlib/tkinter on Windows")
def test_print_grid_squarify():
    image_0c = np.random.randint(0, 255, size=[100, 100])
    image_1c = np.random.randint(0, 255, size=[100, 100, 1])
    image_3c = np.random.randint(0, 255, size=[100, 100, 3])
    images = [image_0c, image_1c, image_3c, image_1c, image_0c, image_3c]
    size = display.print_grid(*images, return_grid_shape=True, return_fig=False, do_squarify=False, show=False)
    assert size == (1, 6)
    plt.close()
    size = display.print_grid(*images, return_grid_shape=True, return_fig=False, do_squarify=True, show=False)
    assert size == (2, 3)
    plt.close()
    size = display.print_grid(
        *images, return_grid_shape=True, return_fig=False, do_squarify=True, num_cols=2, show=False
    )
    assert size == (3, 2)
    plt.close()


@pytest.mark.skipif(sys.platform == "win32", reason="Bug in matplotlib/tkinter on Windows")
def test_print_grid_tofile(scratch_path):
    test_image = np.ones([10, 10])
    display.print_grid(test_image, show=False, file_name=scratch_path / "testprintgrid.png")
    assert os.path.exists(scratch_path / "testprintgrid.png")
    plt.close()


@pytest.mark.skipif(sys.platform == "win32", reason="Bug in matplotlib/tkinter on Windows")
def test_print_grid_exceptions():
    test_image = np.ones([5, 5, 100, 100, 5])
    with pytest.raises(ValueError):
        assert display.print_grid(test_image)

    with pytest.raises(ValueError):
        assert display.print_grid("string")


@pytest.mark.skipif(sys.platform == "win32", reason="Bug in matplotlib/tkinter on Windows")
def test_print_image(scratch_path):
    test_image = np.ones([100, 100])
    test_image[25:75, 25:75] = 0
    display.print_image(test_image, show=False, left=0.1)
    plt.close()
    display.print_image(test_image, show=False)
    plt.close()

    test_image = np.ones([100, 100, 3], dtype=np.uint8) * 255
    test_image[25:75, 25:75] = 0
    display.print_image(test_image, file_name=scratch_path / "test_image.png", show=False)
    plt.close()
    assert os.path.exists(scratch_path / "test_image.png")


@pytest.mark.skipif(sys.platform == "win32", reason="Bug in matplotlib/tkinter on Windows")
def test_squarify():
    test = np.random.randint(0, 255, size=[8, 200, 200])
    result = display.squarify(test, primary_axis=0, as_array=False)
    assert len(result) == 3
    for item in result:
        assert len(item) == len(result)
    assert result[-1][-1] is None

    result = display.squarify(test, primary_axis=0, as_array=True)
    assert result.shape == (3, 3, 200, 200)
    assert result[-1, -1].sum() == 0

    test = np.random.randint(0, 255, size=[200, 200, 8])
    result = display.squarify(test, primary_axis=2, as_array=True)
    assert result.shape == (3, 3, 200, 200)
    assert result[-1, -1].sum() == 0

    test = [np.random.randint(0, 255, size=[200, 200]) for i in range(8)]
    result = display.squarify(test, primary_axis=2, as_array=True)
    assert result.shape == (3, 3, 200, 200)
    assert result[-1, -1].sum() == 0

    test = [np.random.randint(0, 255, size=[200, 200]) for i in range(8)]
    result = display.squarify(test, primary_axis=2, as_array=False)
    assert len(result) == 3
    for item in result:
        assert len(item) == len(result)
    assert result[-1][-1] is None
