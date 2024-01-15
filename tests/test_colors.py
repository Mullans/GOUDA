import numpy as np

import gouda.colors


def test_sRGB_conversions():
    check_pairs = [([255, 255, 255], [255, 255, 255]), ([0, 0, 0], [0, 0, 0]), ([100, 150, 200], [32, 78, 147])]
    for start_color, mid_color in check_pairs:
        color_check = np.asarray(start_color).astype(np.uint8)
        forward_check = gouda.colors.sRGB2linearRGB(color_check)
        np.testing.assert_allclose(forward_check, np.asarray(mid_color), atol=1)
        reverse_check = gouda.colors.linearRGB2sRGB(forward_check)
        np.testing.assert_allclose(reverse_check, np.asarray(start_color), atol=1)


def test_LAB_conversions():
    check_pairs = [([255, 255, 255], [100, 0, 0]),
                   ([0, 0, 0], [0, 0, 0]),
                   ([255, 0, 0], [54.29, 80.81, 69.89]),
                   ([0, 255, 0], [87.82, -79.29, 80.99]),
                   ([0, 0, 255], [29.57, 69.30, -112.03])]
    for start_color, mid_color in check_pairs:
        color_check = np.asarray(start_color).astype(np.uint8)
        forward_check = gouda.colors.rgb2lab(color_check)
        np.testing.assert_allclose(forward_check, mid_color, atol=10)
        reverse_check = gouda.colors.lab2rgb(forward_check)
        np.testing.assert_allclose(reverse_check, color_check, atol=1)


def test_CVD_simulation():
    sim = gouda.colors.CVD_Simulator()
    test_palette = np.asarray([[198, 224, 0], [227, 1, 79], [1, 40, 133], [255, 133, 57], [0, 81, 67]])

    true_protan = np.asarray([[252, 216, 0], [83, 82, 80], [0, 45, 133], [172, 150, 58], [79, 76, 66]])
    true_deutan = np.asarray([[239, 206, 24], [141, 125, 70], [0, 53, 132], [198, 172, 45], [68, 68, 68]])
    true_tritan = np.asarray([[214, 209, 209], [226, 6, 73], [0, 60, 80], [254, 122, 141], [22, 76, 91]])

    protan_color = [sim.simulate_cvd_color(rgb, gouda.colors.Deficiency.PROTAN) for rgb in test_palette]
    deutan_color = [sim.simulate_cvd_color(rgb, gouda.colors.Deficiency.DEUTAN) for rgb in test_palette]
    tritan_color = [sim.simulate_cvd_color(rgb, gouda.colors.Deficiency.TRITAN) for rgb in test_palette]
    np.testing.assert_allclose(protan_color, true_protan)
    np.testing.assert_allclose(deutan_color, true_deutan)
    np.testing.assert_allclose(tritan_color, true_tritan)

    test_swatches = [np.ones([100, 100, 3]) * rgb.reshape([1, 1, 3]) for rgb in test_palette]
    protan_image_color = [sim.simulate_cvd_image(rgb, gouda.colors.Deficiency.PROTAN)[50, 50] for rgb in test_swatches]
    deutan_image_color = [sim.simulate_cvd_image(rgb, gouda.colors.Deficiency.DEUTAN)[50, 50] for rgb in test_swatches]
    tritan_image_color = [sim.simulate_cvd_image(rgb, gouda.colors.Deficiency.TRITAN)[50, 50] for rgb in test_swatches]
    np.testing.assert_allclose(protan_image_color, true_protan)
    np.testing.assert_allclose(deutan_image_color, true_deutan)
    np.testing.assert_allclose(tritan_image_color, true_tritan)
