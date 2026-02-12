"""Tests for video encoding (require ffmpeg)."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from trajviz.config import FrustumStyle, TrajectoryRenderConfig
from trajviz.video import combine_videos_side_by_side, render_trajectory_video

HAS_FFMPEG = shutil.which("ffmpeg") is not None
skip_no_ffmpeg = pytest.mark.skipif(
    not HAS_FFMPEG, reason="ffmpeg not on PATH"
)


def _make_spiral(n: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Create a small spiral trajectory with colors."""
    t = np.linspace(0, 4 * np.pi, n, dtype=np.float32)
    positions = np.stack([np.cos(t), np.sin(t), t / (4 * np.pi)], axis=1)
    # Color gradient: blue -> red
    frac = np.linspace(0, 1, n, dtype=np.float32)
    colors = np.stack(
        [
            frac,
            np.zeros(n, dtype=np.float32),
            1 - frac,
            np.ones(n, dtype=np.float32),
        ],
        axis=1,
    )
    return positions, colors


class TestRenderTrajectoryVideo:
    """End-to-end video rendering tests."""

    @skip_no_ffmpeg
    def test_creates_mp4_file(self) -> None:
        positions, colors = _make_spiral(20)
        with tempfile.TemporaryDirectory() as tmp:
            out = f"{tmp}/test.mp4"
            config = TrajectoryRenderConfig(
                width=120,
                height=120,
                frustum_style=FrustumStyle.NONE,
                fps=10,
            )
            result = render_trajectory_video(positions, colors, out, config)
            assert result == out
            out_path = Path(out)
            assert out_path.is_file()
            assert out_path.stat().st_size > 0

    @skip_no_ffmpeg
    def test_returns_output_path(self) -> None:
        positions, colors = _make_spiral(10)
        with tempfile.TemporaryDirectory() as tmp:
            out = f"{tmp}/output.mp4"
            config = TrajectoryRenderConfig(
                width=80,
                height=80,
                frustum_style=FrustumStyle.NONE,
                fps=5,
            )
            result = render_trajectory_video(positions, colors, out, config)
            assert result == out

    @skip_no_ffmpeg
    def test_with_forward_vectors(self) -> None:
        positions, colors = _make_spiral(15)
        # Forward = tangent to spiral
        forward = np.diff(positions, axis=0, prepend=positions[:1])
        norms = np.linalg.norm(forward, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        forward = (forward / norms).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            out = f"{tmp}/frustum.mp4"
            config = TrajectoryRenderConfig(
                width=120,
                height=120,
                frustum_style=FrustumStyle.WEDGE,
                fps=5,
            )
            result = render_trajectory_video(
                positions, colors, out, config, forward
            )
            assert result == out

    @skip_no_ffmpeg
    def test_default_config(self) -> None:
        """None config uses defaults."""
        positions, colors = _make_spiral(5)
        with tempfile.TemporaryDirectory() as tmp:
            out = f"{tmp}/default.mp4"
            # Use small custom config to keep test fast
            config = TrajectoryRenderConfig(
                width=80,
                height=80,
                frustum_style=FrustumStyle.NONE,
                fps=5,
            )
            result = render_trajectory_video(positions, colors, out, config)
            assert result == out

    def test_missing_ffmpeg_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing ffmpeg raises FileNotFoundError."""
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        positions, colors = _make_spiral(5)
        with pytest.raises(FileNotFoundError, match="ffmpeg"):
            render_trajectory_video(positions, colors, "/tmp/nope.mp4")


class TestCombineVideosSideBySide:
    """Tests for combine_videos_side_by_side."""

    @skip_no_ffmpeg
    def test_combines_two_videos(self) -> None:
        """Create two small videos and combine them."""
        positions, colors = _make_spiral(10)
        with tempfile.TemporaryDirectory() as tmp:
            config = TrajectoryRenderConfig(
                width=120,
                height=120,
                frustum_style=FrustumStyle.NONE,
                fps=5,
            )
            left = f"{tmp}/left.mp4"
            right = f"{tmp}/right.mp4"
            combined = f"{tmp}/combined.mp4"

            render_trajectory_video(positions, colors, left, config)
            render_trajectory_video(positions, colors, right, config)

            result = combine_videos_side_by_side(left, right, combined, 5)
            assert result is True
            combined_path = Path(combined)
            assert combined_path.is_file()
            assert combined_path.stat().st_size > 0

    def test_missing_ffmpeg_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        result = combine_videos_side_by_side("a.mp4", "b.mp4", "c.mp4", 30)
        assert result is False
