"""Tests for frame rendering."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from trajviz.config import FrustumStyle, TrajectoryRenderConfig
from trajviz.renderer import _compute_frustum_pixels, render_trajectory_frames


class TestComputeFrustumPixels:
    """Tests for _compute_frustum_pixels."""

    def test_output_shape(self) -> None:
        fwd = np.array([1.0, 0.0], dtype=np.float64)
        tri = _compute_frustum_pixels(fwd, (360, 360), 50.0, math.radians(30))
        assert tri.shape == (3, 2)
        assert tri.dtype == np.int32

    def test_first_vertex_is_center(self) -> None:
        fwd = np.array([1.0, 0.0], dtype=np.float64)
        center = (100, 200)
        tri = _compute_frustum_pixels(fwd, center, 50.0, math.radians(30))
        assert tri[0, 0] == 100
        assert tri[0, 1] == 200

    def test_degenerate_zero_forward(self) -> None:
        """Zero forward vector produces all-center triangle."""
        fwd = np.array([0.0, 0.0], dtype=np.float64)
        center = (360, 360)
        tri = _compute_frustum_pixels(fwd, center, 50.0, math.radians(30))
        for row in range(3):
            assert tri[row, 0] == 360
            assert tri[row, 1] == 360

    def test_near_zero_forward(self) -> None:
        """Near-zero forward also degenerates gracefully."""
        fwd = np.array([1e-8, 1e-8], dtype=np.float64)
        tri = _compute_frustum_pixels(fwd, (360, 360), 50.0, math.radians(30))
        assert tri.shape == (3, 2)

    def test_vertices_at_correct_distance(self) -> None:
        """Non-center vertices are roughly `length_px` away."""
        fwd = np.array([1.0, 0.0], dtype=np.float64)
        center = (200, 200)
        length = 100.0
        half_fov = math.radians(30)
        tri = _compute_frustum_pixels(fwd, center, length, half_fov)

        for v in (1, 2):
            dist = np.linalg.norm(tri[v].astype(float) - np.array(center))
            # Distance = length_px (exact for unit forward)
            assert_allclose(dist, length, atol=1.5)

    def test_angle_between_edges_matches_fov(self) -> None:
        """Angle between the two outer vertices matches the FOV."""
        fwd = np.array([0.0, -1.0], dtype=np.float64)
        center = np.array([500, 500], dtype=np.float64)
        half_fov = math.radians(45)
        tri = _compute_frustum_pixels(fwd, (500, 500), 200.0, half_fov)

        v1 = tri[1].astype(np.float64) - center
        v2 = tri[2].astype(np.float64) - center
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = math.acos(np.clip(cos_angle, -1, 1))
        assert_allclose(angle, 2 * half_fov, atol=0.02)

    def test_symmetry_around_forward(self) -> None:
        """Left/right vertices are symmetric about forward axis."""
        fwd = np.array([1.0, 0.0], dtype=np.float64)
        center = np.array([300, 300], dtype=np.float64)
        tri = _compute_frustum_pixels(fwd, (300, 300), 80.0, math.radians(30))
        v1 = tri[1].astype(np.float64) - center
        v2 = tri[2].astype(np.float64) - center
        # Y components should be opposite
        assert_allclose(v1[1], -v2[1], atol=1.5)
        # X components should be equal
        assert_allclose(v1[0], v2[0], atol=1.5)


class TestRenderTrajectoryFrames:
    """Tests for render_trajectory_frames."""

    @pytest.fixture
    def simple_trajectory(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """10-point diagonal trajectory with distinct colors."""
        n = 10
        pixels = np.stack(
            [
                np.linspace(100, 600, n),
                np.linspace(100, 600, n),
            ],
            axis=1,
        ).astype(np.int32)
        colors = np.stack(
            [
                np.linspace(0, 255, n),
                np.full(n, 100),
                np.linspace(255, 0, n),
            ],
            axis=1,
        ).astype(np.int32)
        return pixels, colors

    def test_yields_correct_count(
        self,
        simple_trajectory: tuple[np.ndarray, np.ndarray],
    ) -> None:
        pixels, colors = simple_trajectory
        config = TrajectoryRenderConfig(
            width=720, height=720, frustum_style=FrustumStyle.NONE
        )
        frames = list(render_trajectory_frames(pixels, colors, config))
        assert len(frames) == 10

    def test_frame_shape_and_dtype(
        self,
        simple_trajectory: tuple[np.ndarray, np.ndarray],
    ) -> None:
        pixels, colors = simple_trajectory
        config = TrajectoryRenderConfig(
            width=720, height=480, frustum_style=FrustumStyle.NONE
        )
        frame = next(iter(render_trajectory_frames(pixels, colors, config)))
        assert frame.shape == (480, 720, 3)
        assert frame.dtype == np.uint8

    def test_empty_trajectory_yields_nothing(self) -> None:
        pixels = np.empty((0, 2), dtype=np.int32)
        colors = np.empty((0, 3), dtype=np.int32)
        config = TrajectoryRenderConfig(frustum_style=FrustumStyle.NONE)
        frames = list(render_trajectory_frames(pixels, colors, config))
        assert len(frames) == 0

    def test_single_point(self) -> None:
        pixels = np.array([[360, 360]], dtype=np.int32)
        colors = np.array([[255, 0, 0]], dtype=np.int32)
        config = TrajectoryRenderConfig(
            width=720, height=720, frustum_style=FrustumStyle.NONE
        )
        frames = list(render_trajectory_frames(pixels, colors, config))
        assert len(frames) == 1
        assert frames[0].shape == (720, 720, 3)

    def test_background_color_present(self) -> None:
        """First frame corners should be close to background color."""
        pixels = np.array([[360, 360]], dtype=np.int32)
        colors = np.array([[255, 0, 0]], dtype=np.int32)
        bg = (50, 50, 50)
        config = TrajectoryRenderConfig(
            width=720,
            height=720,
            background_rgb=bg,
            grid_enabled=False,
            frustum_style=FrustumStyle.NONE,
        )
        frame = next(iter(render_trajectory_frames(pixels, colors, config)))
        # Corner pixel far from marker should be background
        corner = frame[0, 0]
        assert_allclose(corner, bg, atol=5)

    def test_grid_disabled(self) -> None:
        """With grid disabled, frame is just background + marker."""
        pixels = np.array([[360, 360]], dtype=np.int32)
        colors = np.array([[255, 255, 255]], dtype=np.int32)
        bg = (0, 0, 0)
        config = TrajectoryRenderConfig(
            width=100,
            height=100,
            background_rgb=bg,
            grid_enabled=False,
            frustum_style=FrustumStyle.NONE,
        )
        frame = next(iter(render_trajectory_frames(pixels, colors, config)))
        # Most pixels should be near-black (background)
        dark_fraction = (frame.mean(axis=2) < 10).mean()
        assert dark_fraction > 0.8

    def test_with_frustum_wedge(
        self,
        simple_trajectory: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Rendering with wedge frustum doesn't crash."""
        pixels, colors = simple_trajectory
        forward = np.tile(np.array([1.0, 0.0], dtype=np.float64), (10, 1))
        config = TrajectoryRenderConfig(
            width=720,
            height=720,
            frustum_style=FrustumStyle.WEDGE,
        )
        frames = list(
            render_trajectory_frames(pixels, colors, config, forward)
        )
        assert len(frames) == 10

    def test_with_frustum_pyramid(
        self,
        simple_trajectory: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Rendering with pyramid frustum doesn't crash."""
        pixels, colors = simple_trajectory
        forward = np.tile(np.array([0.0, -1.0], dtype=np.float64), (10, 1))
        config = TrajectoryRenderConfig(
            width=720,
            height=720,
            frustum_style=FrustumStyle.PYRAMID,
        )
        frames = list(
            render_trajectory_frames(pixels, colors, config, forward)
        )
        assert len(frames) == 10

    def test_frustum_none_ignores_forward(
        self,
        simple_trajectory: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """NONE frustum style works even with forward vectors."""
        pixels, colors = simple_trajectory
        forward = np.ones((10, 2), dtype=np.float64)
        config = TrajectoryRenderConfig(
            frustum_style=FrustumStyle.NONE,
        )
        frames = list(
            render_trajectory_frames(pixels, colors, config, forward)
        )
        assert len(frames) == 10

    def test_frames_differ_over_time(
        self,
        simple_trajectory: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Successive frames differ (trail is being drawn)."""
        pixels, colors = simple_trajectory
        config = TrajectoryRenderConfig(
            frustum_style=FrustumStyle.NONE,
        )
        frames = list(render_trajectory_frames(pixels, colors, config))
        # First and last frame must differ
        assert not np.array_equal(frames[0], frames[-1])

    def test_custom_dimensions(self) -> None:
        """Non-default width/height produces correct frame size."""
        pixels = np.array([[50, 50], [150, 150]], dtype=np.int32)
        colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.int32)
        config = TrajectoryRenderConfig(
            width=200,
            height=300,
            frustum_style=FrustumStyle.NONE,
        )
        frame = next(iter(render_trajectory_frames(pixels, colors, config)))
        assert frame.shape == (300, 200, 3)
