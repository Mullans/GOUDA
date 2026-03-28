from __future__ import annotations

import sys

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")

import gouda.plotting as plotting
from gouda.plotting import annotate_arrows, colorplot, parse_color, plot_accuracy_curve, plot_joint_arrow, quick_line


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ── parse_color ──────────────────────────────────────────────────────────────


class TestParseColor:
    def test_named_custom_color_red(self):
        # "red" maps to the xkcd color dict, not pure (1, 0, 0)
        result = parse_color("red")
        assert len(result) == 3
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(0.0)

    def test_named_custom_color_white(self):
        result = parse_color("White")
        assert result == pytest.approx((1.0, 1.0, 1.0))

    def test_named_custom_color_blue(self):
        result = parse_color("Blue")
        assert result == pytest.approx((0.0, 0.0, 1.0))

    def test_hex_string(self):
        result = parse_color("#ff0000")
        assert result == pytest.approx((1.0, 0.0, 0.0))

    def test_returns_three_tuple(self):
        result = parse_color("Blue")
        assert len(result) == 3

    def test_string_comma_separated(self):
        result = parse_color("0.5, 0.25, 0.75")
        assert result == pytest.approx((0.5, 0.25, 0.75))

    def test_string_comma_no_space(self):
        result = parse_color("0.5,0.25,0.75")
        assert result == pytest.approx((0.5, 0.25, 0.75))

    def test_string_space_separated(self):
        result = parse_color("0.5 0.25 0.75")
        assert result == pytest.approx((0.5, 0.25, 0.75))

    def test_string_comma_separated_uint8(self):
        result = parse_color("255, 0, 0")
        assert result == pytest.approx((1.0, 0.0, 0.0))

    def test_rgb_tuple(self):
        result = parse_color((0.2, 0.4, 0.6))
        assert result == pytest.approx((0.2, 0.4, 0.6))

    def test_rgba_tuple_drops_alpha(self):
        result = parse_color((0.2, 0.4, 0.6, 1.0))
        assert result == pytest.approx((0.2, 0.4, 0.6))

    def test_uint8_list_normalizes(self):
        result = parse_color([255, 0, 0])
        assert result == pytest.approx((1.0, 0.0, 0.0))

    def test_float_list_unchanged(self):
        result = parse_color([0.5, 0.5, 0.5])
        assert result == pytest.approx((0.5, 0.5, 0.5))

    def test_float_scalar_maps_to_colormap(self):
        result = parse_color(0.0)
        assert len(result) == 3
        assert all(0.0 <= v <= 1.0 for v in result)

    def test_int_scalar_maps_to_colormap(self):
        result = parse_color(3)
        assert len(result) == 3
        assert all(0.0 <= v <= 1.0 for v in result)

    def test_float_scalar_custom_cmap_string(self):
        result = parse_color(0.5, cmap="plasma")
        assert len(result) == 3

    def test_float_scalar_custom_cmap_object(self):
        cmap_obj = matplotlib.colormaps.get_cmap("plasma")
        result_str = parse_color(0.5, cmap="plasma")
        result_obj = parse_color(0.5, cmap=cmap_obj)
        assert result_str == pytest.approx(result_obj)

    def test_invalid_cmap_type_raises(self):
        with pytest.raises((ValueError, TypeError)):
            parse_color(0.5, cmap=12345)  # type: ignore


# ── quick_line ───────────────────────────────────────────────────────────────


class TestQuickLine:
    def test_returns_path_patch(self):
        from matplotlib.patches import PathPatch

        patch = quick_line(0.0, 1.0, 0.0, 1.0, 0.0)
        assert isinstance(patch, PathPatch)

    def test_not_filled(self):
        patch = quick_line(0.0, 1.0, 0.0, 1.0, 0.5)
        assert not patch.get_fill()

    def test_correct_vertices(self):
        patch = quick_line(0.1, 0.9, 0.2, 0.8, 0.3)
        verts = patch.get_path().vertices
        assert verts[0] == pytest.approx([0.1, 0.2])
        assert verts[1] == pytest.approx([0.1, 0.8])
        assert verts[2] == pytest.approx([0.9, 0.8])
        assert verts[3] == pytest.approx([0.9, 0.3])


# ── plot_accuracy_curve ───────────────────────────────────────────────────────


class TestPlotAccuracyCurve:
    def setup_method(self):
        self.acc = np.array([0.5, 0.7, 0.9, 0.8, 0.6])
        self.thresh = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    def test_returns_none(self):
        result = plot_accuracy_curve(self.acc, self.thresh)
        assert result is None

    def test_no_label(self):
        # Should not raise with label_height=None
        plot_accuracy_curve(self.acc, self.thresh, label_height=None)

    def test_custom_line_args(self):
        plot_accuracy_curve(self.acc, self.thresh, line_args={"color": "blue"})

    def test_custom_thresh_args(self):
        plot_accuracy_curve(self.acc, self.thresh, thresh_args={"color": "green"})

    def test_axes_limits(self):
        plot_accuracy_curve(self.acc, self.thresh)
        ax = plt.gca()
        assert ax.get_xlim() == pytest.approx((0.0, 1.0))
        assert ax.get_ylim() == pytest.approx((0.0, 1.0))


# ── annotate_arrows ───────────────────────────────────────────────────────────


class TestAnnotateArrows:
    def setup_method(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)

    def test_basic_call(self):
        annotate_arrows(self.ax, "p<0.05", (5.0, 3.0), [(2.0, 2.0), (8.0, 2.0)])

    def test_direction_right(self):
        annotate_arrows(self.ax, "*", (2.0, 3.0), [(5.0, 2.0), (8.0, 2.0)], direction="right")

    def test_repeat_text_false(self):
        annotate_arrows(self.ax, "ns", (5.0, 3.0), [(2.0, 2.0), (8.0, 2.0)], repeat_text=False)

    def test_explicit_y_top(self):
        annotate_arrows(self.ax, "*", (5.0, 3.0), [(2.0, 2.0), (8.0, 2.0)], y_top=7.0)


# ── colorplot ────────────────────────────────────────────────────────────────


class TestColorplot:
    def setup_method(self):
        self.x = [0.0, 1.0, 2.0, 3.0]
        self.y = [0.0, 1.0, 0.0, 1.0]

    def test_returns_axes(self):
        result = colorplot(self.x, self.y)
        assert isinstance(result, matplotlib.axes.Axes)

    def test_accepts_existing_axes(self):
        fig, ax = plt.subplots()
        result = colorplot(self.x, self.y, ax=ax)
        assert result is ax

    def test_string_cmap(self):
        result = colorplot(self.x, self.y, cmap="plasma")
        assert isinstance(result, matplotlib.axes.Axes)

    def test_colormap_object(self):
        cmap = matplotlib.colormaps.get_cmap("viridis")
        result = colorplot(self.x, self.y, cmap=cmap)
        assert isinstance(result, matplotlib.axes.Axes)

    def test_step_as_percent_false(self):
        result = colorplot(self.x, self.y, step_size=0.1, step_as_percent=False)
        assert isinstance(result, matplotlib.axes.Axes)

    def test_custom_start_end_val(self):
        result = colorplot(self.x, self.y, start_val=0.2, end_val=0.8)
        assert isinstance(result, matplotlib.axes.Axes)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            colorplot([0.0, 1.0], [0.0, 1.0, 2.0])

    def test_non_iterable_raises(self):
        with pytest.raises((ValueError, TypeError)):
            colorplot(1.0, 2.0)  # type: ignore

    def test_numpy_arrays(self):
        x = np.linspace(0, 2 * np.pi, 50)
        y = np.sin(x)
        result = colorplot(x, y)
        assert isinstance(result, matplotlib.axes.Axes)


# ── plot_joint_arrow ─────────────────────────────────────────────────────────


class TestPlotJointArrow:
    def setup_method(self):
        self.x = [0.0, 1.0, 2.0, 3.0]
        self.y = [0.0, 1.0, 0.5, 0.0]

    def test_returns_axes(self):
        result = plot_joint_arrow(self.x, self.y)
        assert isinstance(result, matplotlib.axes.Axes)

    def test_headlength_none_defaults_to_headwidth(self):
        # headlength=None should default to headwidth without raising
        result = plot_joint_arrow(self.x, self.y, headlength=None)
        assert isinstance(result, matplotlib.axes.Axes)

    def test_accepts_existing_axes(self):
        fig, ax = plt.subplots()
        result = plot_joint_arrow(self.x, self.y, ax=ax)
        assert result is ax

    def test_with_color(self):
        result = plot_joint_arrow(self.x, self.y, color="red")
        assert isinstance(result, matplotlib.axes.Axes)

    def test_with_label(self):
        result = plot_joint_arrow(self.x, self.y, label="trajectory")
        assert isinstance(result, matplotlib.axes.Axes)

    def test_custom_linewidth_headwidth(self):
        result = plot_joint_arrow(self.x, self.y, linewidth=3.0, headwidth=3.0)
        assert isinstance(result, matplotlib.axes.Axes)

    def test_custom_headlength(self):
        result = plot_joint_arrow(self.x, self.y, headlength=4.0)
        assert isinstance(result, matplotlib.axes.Axes)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            plot_joint_arrow([0.0, 1.0], [0.0, 1.0, 2.0])

    def test_non_iterable_raises(self):
        with pytest.raises((ValueError, TypeError)):
            plot_joint_arrow(1.0, 2.0)  # type: ignore

    def test_numpy_arrays(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 0.5, 1.0])
        result = plot_joint_arrow(x, y)
        assert isinstance(result, matplotlib.axes.Axes)
