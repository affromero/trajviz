"""Tests for 3D to 2D orthographic projection."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from trajviz.projection import build_projection_matrix, project_to_pixels


class TestBuildProjectionMatrix:
    """Tests for build_projection_matrix."""

    def test_output_shape(self) -> None:
        proj = build_projection_matrix(30.0, 45.0)
        assert proj.shape == (3, 2)
        assert proj.dtype == np.float64

    def test_zero_angles(self) -> None:
        """Zero elevation/azimuth gives identity-like projection."""
        proj = build_projection_matrix(0.0, 0.0)
        # With no rotation, XY columns of identity: [[1,0],[0,1],[0,0]]
        expected = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        assert_allclose(proj, expected, atol=1e-10)

    def test_90_azimuth_swaps_xy(self) -> None:
        """90 deg azimuth rotates X->Y, Y->-X."""
        proj = build_projection_matrix(0.0, 90.0)
        # Rz(90): [[0,-1,0],[1,0,0],[0,0,1]], take cols 0,1
        expected = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 0.0]])
        assert_allclose(proj, expected, atol=1e-10)

    def test_elevation_tilts_z_into_y(self) -> None:
        """90 deg elevation maps Z fully into screen Y."""
        proj = build_projection_matrix(90.0, 0.0)
        # Rx(90): [[1,0,0],[0,0,-1],[0,1,0]]
        # Take cols 0,1: [[1,0],[0,0],[0,1]]
        expected = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
        assert_allclose(proj, expected, atol=1e-10)

    def test_columns_are_orthogonal(self) -> None:
        """Projection columns should be orthogonal."""
        proj = build_projection_matrix(30.0, 45.0)
        dot = np.dot(proj[:, 0], proj[:, 1])
        assert_allclose(dot, 0.0, atol=1e-10)

    def test_columns_are_unit_length(self) -> None:
        """Projection columns should be unit vectors."""
        proj = build_projection_matrix(30.0, 45.0)
        assert_allclose(np.linalg.norm(proj[:, 0]), 1.0, atol=1e-10)
        assert_allclose(np.linalg.norm(proj[:, 1]), 1.0, atol=1e-10)

    @pytest.mark.parametrize("elev", [0.0, 15.0, 30.0, 45.0, 60.0, 90.0])
    @pytest.mark.parametrize("azim", [0.0, 45.0, 90.0, 180.0, 270.0])
    def test_orthogonality_various_angles(
        self, elev: float, azim: float
    ) -> None:
        """Columns stay orthogonal and unit-length for all angles."""
        proj = build_projection_matrix(elev, azim)
        assert proj.shape == (3, 2)
        dot = np.dot(proj[:, 0], proj[:, 1])
        assert_allclose(dot, 0.0, atol=1e-10)
        assert_allclose(np.linalg.norm(proj[:, 0]), 1.0, atol=1e-10)
        assert_allclose(np.linalg.norm(proj[:, 1]), 1.0, atol=1e-10)

    def test_negative_angles(self) -> None:
        proj = build_projection_matrix(-30.0, -45.0)
        assert proj.shape == (3, 2)
        dot = np.dot(proj[:, 0], proj[:, 1])
        assert_allclose(dot, 0.0, atol=1e-10)


class TestProjectToPixels:
    """Tests for project_to_pixels."""

    @pytest.fixture
    def proj(self) -> np.ndarray:
        return build_projection_matrix(30.0, 45.0)

    def test_output_shape(self, proj: np.ndarray) -> None:
        rng = np.random.default_rng(42)
        positions = rng.standard_normal((50, 3)).astype(np.float32)
        pixels = project_to_pixels(positions, proj, 720, 720, 0.1)
        assert pixels.shape == (50, 2)
        assert pixels.dtype == np.int32

    def test_single_point_centered(self) -> None:
        """Single point projects to canvas center."""
        proj = build_projection_matrix(0.0, 0.0)
        positions = np.array([[5.0, 3.0, 0.0]], dtype=np.float32)
        pixels = project_to_pixels(positions, proj, 720, 720, 0.1)
        assert_allclose(pixels[0], [360, 360], atol=1)

    def test_two_points_symmetric(self) -> None:
        """Two symmetric points project symmetrically."""
        proj = build_projection_matrix(0.0, 0.0)
        positions = np.array(
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
        )
        pixels = project_to_pixels(positions, proj, 720, 720, 0.1)
        # Midpoint should be canvas center
        mid = pixels.mean(axis=0)
        assert_allclose(mid, [360.0, 360.0], atol=1)
        # Symmetric around center
        assert pixels[0, 0] < 360
        assert pixels[1, 0] > 360

    def test_pixels_within_canvas(self, proj: np.ndarray) -> None:
        """All projected pixels are within canvas bounds."""
        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 3)).astype(np.float32)
        w, h = 720, 720
        pixels = project_to_pixels(positions, proj, w, h, 0.12)
        assert (pixels[:, 0] >= 0).all()
        assert (pixels[:, 0] < w).all()
        assert (pixels[:, 1] >= 0).all()
        assert (pixels[:, 1] < h).all()

    def test_degenerate_single_point(self, proj: np.ndarray) -> None:
        """Single point doesn't cause division by zero."""
        positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        pixels = project_to_pixels(positions, proj, 720, 720, 0.1)
        assert pixels.shape == (1, 2)
        assert np.isfinite(pixels).all()

    def test_collinear_points(self) -> None:
        """Points along one axis produce valid pixels."""
        proj = build_projection_matrix(0.0, 0.0)
        positions = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        pixels = project_to_pixels(positions, proj, 720, 720, 0.1)
        assert pixels.shape == (3, 2)
        # All on same Y (flipped) row
        assert pixels[0, 1] == pixels[1, 1] == pixels[2, 1]
        # X increases monotonically
        assert pixels[0, 0] < pixels[1, 0] < pixels[2, 0]

    def test_padding_ratio_affects_spread(self, proj: np.ndarray) -> None:
        """Larger padding shrinks the projected spread."""
        positions = np.array(
            [[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32
        )
        px_small = project_to_pixels(positions, proj, 720, 720, 0.05)
        px_large = project_to_pixels(positions, proj, 720, 720, 0.4)
        spread_small = np.abs(px_small[1] - px_small[0]).max()
        spread_large = np.abs(px_large[1] - px_large[0]).max()
        assert spread_small > spread_large

    def test_rectangular_canvas(self, proj: np.ndarray) -> None:
        """Non-square canvas produces valid pixels."""
        rng = np.random.default_rng(42)
        positions = rng.standard_normal((20, 3)).astype(np.float32)
        w, h = 1280, 720
        pixels = project_to_pixels(positions, proj, w, h, 0.1)
        assert pixels.shape == (20, 2)
        assert (pixels[:, 0] >= 0).all()
        assert (pixels[:, 0] < w).all()
        assert (pixels[:, 1] >= 0).all()
        assert (pixels[:, 1] < h).all()

    def test_y_axis_flipped(self) -> None:
        """Higher Y in world projects to lower Y on screen."""
        proj = build_projection_matrix(0.0, 0.0)
        positions = np.array(
            [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        )
        pixels = project_to_pixels(positions, proj, 720, 720, 0.1)
        # Screen Y is flipped: world Y=1 -> smaller screen Y
        assert pixels[1, 1] < pixels[0, 1]
