import copy
import numpy as np
from enum import Enum
from typing import Callable, Union

# NOTE - Color distance and color palette generation functions in this file are adapted from media lab's javascript iwanthue library - https://medialab.github.io/iwanthue/js/libs/chroma.palette-gen.js
# NOTE - Color distance conversion functions are adapted from the chroma.js library - https://github.com/gka/chroma.js
# NOTE - Color vision deficiency adapted from the DaltonLens implementation of the Brettel, Viénot & Mollon 1997 algorithm - https://daltonlens.org/colorblindness-simulator


# CONSTANTS

LAB_CONSTANTS = {
    # Corresponds roughly to RGB brighter/darker
    'Kn': 18.0,

    # D65 standard XYZ values normalized to Y = 1
    'Xn': 0.950470,
    'Yn': 1.0,
    'Zn': 1.088830,

    't0': 0.137931034,  # 4 / 29
    't1': 0.206896552,  # 6 / 29
    't2': 0.12841855,   # 3 * t1 * t1
    't3': 0.008856452,  # t1 * t1 * t1
}


class Deficiency(Enum):
    PROTAN = 0
    DEUTAN = 1
    TRITAN = 2


# COLOR CONVERSIONS #################

def sRGB2linearRGB(arr: np.ndarray) -> np.ndarray:
    """Convert sRGB to linearRGB, removing the gamma correction.

    Parameters
    ----------
    arr : np.ndarray
        sRGB color to convert to linear RGB, should be float in range [0., 1.] or uint8 in range [0, 255]
    """
    arr = np.asarray(arr)
    convert_back = False
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
        convert_back = True
    out = np.zeros_like(arr)
    small_mask = arr < 0.04045
    large_mask = np.logical_not(small_mask)
    out[small_mask] = arr[small_mask] / 12.92
    out[large_mask] = np.power((arr[large_mask] + 0.055) / 1.055, 2.4)
    if convert_back:
        out = np.clip(out, 0.0, 1.0)
        out = np.round(out * 255.0).astype(np.uint8)
    return out


def linearRGB2sRGB(arr: np.ndarray) -> np.ndarray:
    """Convert linearRGB to sRGB, applying the gamma correction.

    Parameters
    ----------
    arr : np.ndarray
        linear RGB color to convert to sRGB, should be float in range [0., 1.] or uint8 in range [0, 255]
    """
    arr = np.asarray(arr)
    convert_back = False
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
        convert_back = True
    out = np.zeros_like(arr)
    arr = np.clip(arr, 0., 1.)
    small_mask = arr < 0.0031308
    large_mask = np.logical_not(small_mask)
    out[small_mask] = arr[small_mask] * 12.92
    out[large_mask] = np.power(arr[large_mask], 1.0 / 2.4) * 1.055 - 0.055
    if convert_back:
        out = np.clip(out, 0.0, 1.0)
        out = np.round(out * 255.0).astype(np.uint8)
    return out


def lab2rgb(lab: np.ndarray) -> np.ndarray:
    """Convert a CIE L*a*b* color to RGB

    Parameters
    ----------
    lab : np.ndarray
        CIE L*a*b* color to convert to RGB, should have shape [3,]
    """
    l, a, b = lab
    y = (l + 16.0) / 116.0
    x = y + a / 500.0  # if y is not Nan else y
    z = y - b / 200.0  # if y is not Nan else y

    y = LAB_CONSTANTS['Yn'] * lab_xyz(y)
    x = LAB_CONSTANTS['Xn'] * lab_xyz(x)
    z = LAB_CONSTANTS['Zn'] * lab_xyz(z)

    r = _xyz_rgb(3.2404542 * x - 1.5371385 * y - 0.4985314 * z)  # D65 -> sRGB
    g = _xyz_rgb(-0.9692660 * x + 1.8760108 * y + 0.0415560 * z)
    b_ = _xyz_rgb(0.0556434 * x - 0.2040259 * y + 1.0572252 * z)
    return np.round([r, g, b_]).astype(np.uint8)


def rgb2lab(rgb):
    """Convert an RGB color to CIE L*a*b*

    Parameters
    ----------
    rgb : np.ndarray
        RGB color to convert to CIE L*a*b*, should have shape [3,]
    """
    r, g, b = rgb
    x, y, z = _rgb2xyz(r, g, b)
    light = 116 * y - 16
    return max(0, light), 500 * (x - y), 200 * (y - z)


def _rgb2xyz(r, g, b):
    r = _rgb_xyz(r)
    g = _rgb_xyz(g)
    b = _rgb_xyz(b)
    x = _xyz_lab((0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / LAB_CONSTANTS['Xn'])
    y = _xyz_lab((0.2126729 * r + 0.7151522 * g + 0.0721750 * b) / LAB_CONSTANTS['Yn'])
    z = _xyz_lab((0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / LAB_CONSTANTS['Zn'])
    return x, y, z


def _rgb_xyz(r):
    r /= 255
    if r <= 0.04045:
        return r / 12.92
    else:
        return np.power((r + 0.055) / 1.055, 2.4)


def _xyz_rgb(r: float) -> float:
    if r <= 0.00304:
        result = 12.92 * r
    else:
        result = 1.055 * np.power(r, 1 / 2.4) - 0.055
    return 255 * result


def lab_xyz(t):
    if t > LAB_CONSTANTS['t1']:
        return t ** 3.0
    else:
        return LAB_CONSTANTS['t2'] * (t - LAB_CONSTANTS['t0'])


def _xyz_lab(t):
    if t > LAB_CONSTANTS['t3']:
        return np.power(t, 1 / 3.0)
    else:
        return t / LAB_CONSTANTS['t2'] + LAB_CONSTANTS['t0']


def validate_lab(lab: np.ndarray) -> bool:
    """Validate that a CIE L*a*b* color is within the valid range for both L*a*b* and RGB colors

    Parameters
    ----------
    lab : np.ndarray
        The CIE L*a*b* color to validate
    """
    l, a, b = lab
    l_check = l >= 0 and l <= 100
    a_check = a >= -128 and a <= 128
    b_check = b >= -128 and b <= 128
    if not (l_check and a_check and b_check):
        return False
    y = (l + 16) / 116
    x = y + a / 500
    z = y - b / 200
    y = LAB_CONSTANTS['Yn'] * lab_xyz(y)
    x = LAB_CONSTANTS['Xn'] * lab_xyz(x)
    z = LAB_CONSTANTS['Zn'] * lab_xyz(z)

    r = _xyz_rgb(3.2404542 * x - 1.5371385 * y - 0.4985314 * z)
    g = _xyz_rgb(-0.9692660 * x + 1.8760108 * y + 0.0415560 * z)
    b = _xyz_rgb(0.0556434 * x - 0.2040259 * y + 1.0572252 * z)

    r_check = r >= 0 and r <= 255
    g_check = g >= 0 and g <= 255
    b_check = b >= 0 and b <= 255
    return r_check and g_check and b_check


# COLORBLIND SIMULATION #############

def _plane_projection_matrix(normal, deficiency: Deficiency):
    if deficiency == Deficiency.PROTAN:
        return np.array([
            [0., -normal[1] / normal[0], -normal[2] / normal[0]],
            [0, 1, 0],
            [0, 0, 1]
        ])
    elif deficiency == Deficiency.DEUTAN:
        return np.array([
            [1, 0, 0],
            [-normal[0] / normal[1], 0, -normal[2] / normal[1]],
            [0, 0, 1]
        ])
    elif deficiency == Deficiency.TRITAN:
        return np.array([
            [1, 0, 0],
            [0, 1, 0],
            [-normal[0] / normal[2], -normal[1] / normal[2], 0]
        ])
    else:
        raise ValueError("Unknown color deficiency")


class CVD_Simulator(object):
    """Color vision deficiency simulator adapted from the DaltonLens implementation of (Brettel, Viénot & Mollon, 1997) 'Computerized simulation of color appearance for dichromats' - https://daltonlens.org/colorblindness-simulator

    NOTE
    ----
    The DaltonLens implementation offers significantly more configurations. This is only their Brettel1997 implementation without Vischeck anchors, using white as the neutral color, and using similar values as their SmithPokorny75 LMS color model. A lot of values here have been pre-computed, so check out https://github.com/DaltonLens for their work.
    """

    def __init__(self):
        # XYZJuddVos_from_linearRGB_BT709 @ LMS_from_XYZJuddVos_Smith_Pokorny_1975
        self._LMS_from_linearRGB = np.array([[1.78824041e-01, 4.35160906e-01, 4.11934969e-02],
                                             [3.45564232e-02, 2.71553825e-01, 3.86713084e-02],
                                             [2.99565576e-04, 1.84308960e-03, 1.46708614e-02]])
        self._linearRGB_from_LMS = np.linalg.inv(self._LMS_from_linearRGB)
        self._lms_neutral = self._LMS_from_linearRGB @ np.array([1.0, 1.0, 1.0])
        self._precomputed = {}

    def simulate_cvd_color(self, base_color, deficiency: Deficiency, severity: float = 1.0):
        """Simulate the appearance of a color for the given color vision deficiency

        Parameters
        ----------
        base_color : np.ndarray
            The input sRGB color as a uint8 array with shape (3,) and values in [0,255]
        deficiency : Deficiency
            The color vision deficiency to simulate.
        severity : float, optional
            The color blindness severity between 0 (normal vision) and 1 (complete dichromacy), by default 1.0

        Returns
        -------
        np.ndarray
            The simulated sRGB color as a uint8 array with shape (3,) and values in [0,255]
        """
        linear_rgb = np.asarray(base_color).astype(np.float32) / 255.0
        linear_rgb = sRGB2linearRGB(linear_rgb)

        cvd_linear_rgb = self._simulate_cvd_linear_rgb(linear_rgb, deficiency, severity)

        cvd_linear_rgb = np.clip(cvd_linear_rgb, 0.0, 1.0)
        cvd_float = linearRGB2sRGB(cvd_linear_rgb)

        return (np.clip(cvd_float, 0.0, 1.0) * 255.0).astype(np.uint8)

    def simulate_cvd_image(self, image, deficiency: Deficiency, severity: float = 1.0):
        """Simulate the appearance of an image for the given color vision deficiency

        Parameters
        ----------
        image : np.ndarray
            The input sRGB image as a uint8 array with shape (X, Y, 3) and values in [0,255]
        deficiency : Deficiency
            The color vision deficiency to simulate.
        severity : float, optional
            The color blindness severity between 0 (normal vision) and 1 (complete dichromacy), by default 1.0

        Returns
        -------
        np.ndarray
            The simulated sRGB image as a uint8 array with shape (X, Y, 3) and values in [0,255]
        """
        im_linear_rgb = np.asarray(image).astype(np.float32) / 255.0
        im_linear_rgb = sRGB2linearRGB(im_linear_rgb)

        im_cvd_linear_rgb = self._simulate_cvd_linear_rgb(im_linear_rgb, deficiency, severity)

        im_cvd_float = linearRGB2sRGB(im_cvd_linear_rgb)

        return (np.clip(im_cvd_float, 0.0, 1.0) * 255.0).astype(np.uint8)

    def _simulate_cvd_linear_rgb(self, image_linear_rgb_float32, deficiency: Deficiency, severity: float):
        if deficiency == Deficiency.PROTAN or deficiency == Deficiency.DEUTAN:
            H1, H2, n_sep_plane = self._compute_matrices(deficiency)
        elif deficiency == Deficiency.TRITAN:
            H1, H2, n_sep_plane = self._compute_matrices(deficiency)
        else:
            raise ValueError("Unknown color deficiency")

        im_lms = image_linear_rgb_float32 @ self._LMS_from_linearRGB.T
        im_H1 = im_lms @ H1.T
        im_H2 = im_lms @ H2.T
        H2_indices = np.dot(im_lms, n_sep_plane) < 0

        im_H1[H2_indices] = im_H2[H2_indices]
        im_dichromacy = im_H1 @ self._linearRGB_from_LMS.T

        if severity < 1.0:
            return im_dichromacy * severity + image_linear_rgb_float32 * (1.0 - severity)
        else:
            return im_dichromacy

    def _compute_matrices(self, deficiency):
        if deficiency in self._precomputed:
            return self._precomputed[deficiency]

        if deficiency == Deficiency.PROTAN or deficiency == Deficiency.DEUTAN:
            lms_on_wing1 = np.array([0.05093842, 0.06189707, 0.01515058])  # lms_475
            lms_on_wing2 = np.array([6.28133927e-01, 2.87409450e-01, 3.16776000e-05])  # lms_575
        else:
            lms_on_wing1 = np.array([0.08183212, 0.08803109, 0.00942931])  # lms_485
            lms_on_wing2 = np.array([5.82021668e-02, 2.79539320e-03, 1.60800000e-07])  # lms_660

        n1 = np.cross(self._lms_neutral, lms_on_wing1)
        n2 = np.cross(self._lms_neutral, lms_on_wing2)
        confusion_axis = np.array([0.0, 0.0, 0.0])
        confusion_axis[deficiency.value] = 1.0
        n_sep_plane = np.cross(self._lms_neutral, confusion_axis)
        if np.dot(n_sep_plane, lms_on_wing1) < 0:
            n1, n2 = n2, n1
            lms_on_wing1, lms_on_wing2 = lms_on_wing2, lms_on_wing1
        H1 = _plane_projection_matrix(n1, deficiency)
        H2 = _plane_projection_matrix(n2, deficiency)
        self._precomputed[deficiency] = (H1, H2, n_sep_plane)
        return (H1, H2, n_sep_plane)


# COLOR DISTANCES ###################

def get_color_distance(color1: np.ndarray, color2: np.ndarray, dist_type: str = 'euclidean', colorblind_simulator: CVD_Simulator = None, from_rgb: bool = False) -> float:
    """Compute the distance between two CIE L*a*b* colors

    Parameters
    ----------
    color1 : np.ndarray
        The first color to compare
    color2 : np.ndarray
        The second color to compare
    dist_type : str, optional
        The type of distance metric to use, by default 'euclidean'
    from_rgb: bool, optional
        If true, converts the colors from RGB to L*a*b before computing distance, by default False

    Returns
    -------
    float
        The distance between the colors
    """
    color1 = np.asarray(color1)
    color2 = np.asarray(color2)
    if from_rgb:
        color1 = rgb2lab(color1)
        color2 = rgb2lab(color2)
    dist_type = dist_type.lower()
    if dist_type == 'euclidean':
        return euclidean_distance(color1, color2)
    elif dist_type == 'cmc':
        return cmc_distance(color1, color2, 2, 1)
    elif dist_type == 'compromise':
        return compromise_distance(color1, color2, colorblind_simulator)
    elif isinstance(dist_type, Deficiency):
        return colorblind_distance(color1, color2, dist_type, colorblind_simulator)
    else:
        raise ValueError('Unknown color distance type')


def euclidean_distance(color1: np.ndarray, color2: np.ndarray) -> float:
    """Return the euclidean distance between two colors

    Parameters
    ----------
    color1 : np.ndarray
        The first color to compare
    color2 : np.ndarray
        The second color to compare

    Returns
    -------
    float
        The distance between the colors
    """
    color1 = np.asarray(color1)
    color2 = np.asarray(color2)
    return np.sqrt(np.sum((color1 - color2) ** 2))


def cmc_distance(lab_color1: np.ndarray, lab_color2: np.ndarray, lightness: int = 2, chroma: int = 1) -> float:
    """Return the CIE Delta E distance between two CIE L*a*b* colors, often referred to as the Color Measurement Committee (CMC).

    Parameters
    ----------
    color1 : np.ndarray
        The first color to compare
    color2 : np.ndarray
        The second color to compare
    lightness : int
        The lightness parameter for the CMC calculation, by default 2
    c : int
        The chroma parameter for the CMC calculation, by default 1

    Notes
    -----
    This is the 1984 CMC l:c version - see https://en.wikipedia.org/wiki/Color_difference#CMC_l:c_(1984)
    See http://www.brucelindbloom.com/index.html?Eqn_DeltaE_CMC.html for more information.
    General lightness:chroma value combinations are either 2:1 (acceptability) or 1:1 (perceptibility)

    Returns
    -------
    float
        The distance between the colors
    """
    L1, a1, b1 = lab_color1
    L2, a2, b2 = lab_color2

    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    deltaC = C1 - C2
    deltaL = L1 - L2
    deltaa = a1 - a2
    deltab = b1 - b2
    deltaH = np.sqrt(deltaa ** 2 + deltab ** 2 - deltaC ** 2)
    H1 = np.arctan2(b1, a1) * (180 / np.pi)
    while H1 < 0:
        H1 += 360
    F = np.sqrt(C1 ** 4 / (C1 ** 4 + 1900))
    T = 0.56 + np.abs(0.2 * np.cos(H1 + 168)) if 164 <= H1 and H1 <= 345 else 0.36 + np.abs(0.4 * np.cos(H1 + 35))
    S_L = 0.511 if lab_color1[0] < 16 else (0.040975 * L1 / (1 + 0.01765 * L1))
    S_C = (0.0638 * C1 / (1 + 0.0131 * C1)) + 0.638
    S_H = S_C * (F * T + 1 - F)
    return np.sqrt((deltaL / (lightness * S_L)) ** 2 + (deltaC / (chroma * S_C)) ** 2 + (deltaH / S_H) ** 2)


def compromise_distance(color1: np.ndarray, color2: np.ndarray, colorblind_simulator: CVD_Simulator) -> float:
    """Return the CMC distance between two CIE L*a*b* colors weighted across their colorblind representations

    Parameters
    ----------
    color1 : np.ndarray
        The first color to compare
    color2 : np.ndarray
        The second color to compare

    Returns
    -------
    float
        The distance between the colors
    """
    distances = []
    coeffs = []
    distances.append(cmc_distance(color1, color2, 2, 1))
    coeffs.append(1000)
    rgb_color1 = lab2rgb(color1)
    rgb_color2 = lab2rgb(color2)
    for deficiency in Deficiency:
        cvd_color1 = colorblind_simulator.simulate_cvd_color(rgb_color1, deficiency)
        cvd_color2 = colorblind_simulator.simulate_cvd_color(rgb_color2, deficiency)
        lab_color1 = rgb2lab(cvd_color1)
        lab_color2 = rgb2lab(cvd_color2)
        if lab_color1 is None or lab_color2 is None:
            if type == 'Protanope':
                c = 100
            elif type == 'Deuteranope':
                c = 500
            elif type == 'Tritanope':
                c = 1
            else:
                raise ValueError('Impossible situation')
            distances.append(cmc_distance(lab_color1, lab_color2, 2, 1))
            coeffs.append(c)
    total = 0
    count = 0
    for idx, dist in enumerate(distances):
        total += coeffs[idx] * dist
        count += coeffs[idx]
    return total / count


def colorblind_distance(color1: np.ndarray, color2: np.ndarray, deficiency: Deficiency, colorblind_simulator: CVD_Simulator) -> float:
    """Return the CMC distance between two CIE L*a*b* colors when converted to a completely colorblind representation

    Parameters
    ----------
    color1 : np.ndarray
        The first color to compare
    color2 : np.ndarray
        The second color to compare
    type : str
        The tpye of color blindness to consider (protanope, deuteranope, tritanope)

    Returns
    -------
    float
        The distance between the colors
    """
    cvd_color1 = lab2rgb(color1)
    cvd_color2 = lab2rgb(color2)
    cvd_color1 = colorblind_simulator.simulate_cvd_color(cvd_color1, deficiency)
    cvd_color2 = colorblind_simulator.simulate_cvd_color(cvd_color2, deficiency)
    color1 = rgb2lab(cvd_color1)
    color2 = rgb2lab(cvd_color2)
    return cmc_distance(color1, color2, 2, 1)


# COLOR PALETTES

def generate_palette(num_colors: int, check_color: Callable[[np.ndarray], bool] = lambda x: True, force_mode: bool = True, quality: int = 50, ultra_precision: bool = False, distance_type: Union[str, Deficiency] = 'compromise', as_rgb: bool = True, seed: Union[None, int, np.random.Generator] = None) -> list[np.ndarray]:
    """Generate a perceptually differentiable color palette

    Parameters
    ----------
    num_colors : int
        The number of colors to generate
    check_color : Callable[[np.ndarray], bool], optional
        An additional function to validate selected colors
    force_mode : bool, optional
        Whether to use force vectors for color selection (if false, uses k-means), by default True
    quality : int, optional
        A scaler for the number of steps taken when optimizing colors, by default 50
    ultra_precision : bool, optional
        If true, samples 20 times more potential colors during k-means (no effect on force vectors), by default False
    distance_type : str | Deficiency, optional
        The method to use when computing color distance (can be "euclidean", "cmc", "compromise" or a Deficiency enum value), by default 'compromise'
    as_rgb : bool, optional
        Whether to return colors as RGB (True) or CIE L*a*b* (False), by default True
    seed : Union[None, int, np.random.Generator], optional
        A seed for the random generator, by default None

    Returns
    -------
    list[np.ndarray]
        A list of generated colors

    Note
    ----
    If a Deficiency enum value is passed for `distance_type`, colors will be optimized for that colorblindness type using "cmc" distance
    """
    random = np.random.default_rng(seed)
    colorblind_simulator = CVD_Simulator()

    def check_lab(x):
        return validate_lab(x) and check_color(x)
    if force_mode:  # force vector mode
        colors = []
        vectors = {}
        for i in range(num_colors):
            color = [100 * random.random(), 100 * (2 * random.random() - 1), 100 * (2 * random.random() - 1)]
            while not check_lab(color):
                color = [100 * np.random.random(), 100 * (2 * random.random() - 1), 100 * (2 * random.random() - 1)]
            colors.append(color)

        repulsion = 100
        speed = 100
        steps = quality * 20
        for _ in range(steps):
            for i in range(len(colors)):
                vectors[i] = {'dl': 0, 'da': 0, 'db': 0}
            for i in range(len(colors)):
                color_a = colors[i]
                for j in range(i):
                    color_b = colors[j]

                    dl = color_a[0] - color_b[0]
                    da = color_a[1] - color_b[1]
                    db = color_a[2] - color_b[2]
                    try:
                        d = get_color_distance(color_a, color_b, distance_type, colorblind_simulator)
                    except TypeError:
                        print(color_a, color_b)
                        raise
                    if d > 0:
                        force = repulsion / (d ** 2)
                        vectors[i]['dl'] += dl * force / d
                        vectors[i]['da'] += da * force / d
                        vectors[i]['db'] += db * force / d

                        vectors[j]['dl'] += -dl * force / d
                        vectors[j]['da'] += -da * force / d
                        vectors[j]['db'] += -db * force / d
                    else:
                        vectors[j]['dl'] += 2 - 4 * random.random()
                        vectors[j]['da'] += 2 - 4 * random.random()
                        vectors[j]['db'] += 2 - 4 * random.random()

            for i in range(len(colors)):
                color = colors[i]
                displacement = speed * np.sqrt(np.power(vectors[i]['dl'], 2) + np.power(vectors[i]['da'], 2) + np.power(vectors[i]['db'], 2))
                if displacement > 0:
                    ratio = speed * min(0.1, displacement) / displacement
                    candidate_lab = [color[0] + vectors[i]['dl'] * ratio, color[1] + vectors[i]['da'] * ratio, color[2] + vectors[i]['db'] * ratio]
                    if check_lab(candidate_lab):
                        colors[i] = candidate_lab
    else:
        k_means = []
        for i in range(num_colors):
            lab = [100 * random.random(), 100 * (2 * random.random() - 1), 100 * (2 * random.random() - 1)]
            failsafe = 10
            while not check_color(lab) and failsafe > 0:
                failsafe -= 1
                lab = [100 * random.random(), 100 * (2 * random.random() - 1), 100 * (2 * random.random() - 1)]
            k_means.append(lab)

        color_samples = []
        samples_closest = []
        if ultra_precision:
            for light in range(101):
                for a in range(-100, 101, 5):
                    for b in range(-100, 101, 5):
                        if check_color([light, a, b]):
                            color_samples.append([light, a, b])
                            samples_closest.append(-1)
        else:
            for light in range(0, 101, 5):
                for a in range(-100, 101, 10):
                    for b in range(-100, 101, 10):
                        if check_color([light, a, b]):
                            color_samples.append([light, a, b])
                            samples_closest.append(-1)

        steps = quality
        while steps > 0:
            steps -= 1
            for i in range(len(color_samples)):
                lab = color_samples[i]
                min_distance = np.inf
                for j in range(len(k_means)):
                    k_mean = k_means[j]
                    distance = get_color_distance(lab, k_mean, distance_type, colorblind_simulator)
                    if distance < min_distance:
                        min_distance = distance
                        samples_closest[i] = j

            free_color_samples = copy.deepcopy(color_samples)
            for j in range(len(k_means)):
                count = 0
                candidate_kmean = [0, 0, 0]
                for i in range(len(color_samples)):
                    if samples_closest[i] == j:
                        count += 1
                        candidate_kmean[0] += color_samples[i][0]
                        candidate_kmean[1] += color_samples[i][1]
                        candidate_kmean[2] += color_samples[i][2]
                if count != 0:
                    candidate_kmean[0] /= count
                    candidate_kmean[1] /= count
                    candidate_kmean[2] /= count
                if count != 0 and check_color(candidate_kmean):  # and candidate_kmean  # Is this an existance check? Why?
                    k_means[j] = candidate_kmean
                else:
                    if len(free_color_samples) > 0:
                        min_distance = np.inf
                        closest = -1
                        for i in range(len(color_samples)):
                            distance = get_color_distance(color_samples[1], candidate_kmean, distance_type, colorblind_simulator)
                            if distance < min_distance:
                                min_distance = distance
                                closest = i
                        if closest >= 0:
                            k_means[j] = color_samples[closest]
                free_color_samples = filter(lambda x: x[0] != k_means[j][0] or x[1] != k_means[j][1] or x[2] != k_means[j][2], free_color_samples)
                free_color_samples = list(free_color_samples)
        colors = k_means
    if as_rgb:
        colors = [lab2rgb(color) for color in colors]
    return colors
