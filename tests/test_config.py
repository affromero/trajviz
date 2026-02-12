"""Tests for TrajectoryRenderConfig and FrustumStyle."""

import pytest
from pydantic import ValidationError

from trajviz.config import FrustumStyle, TrajectoryRenderConfig


class TestFrustumStyle:
    """FrustumStyle enum tests."""

    def test_values(self) -> None:
        assert FrustumStyle.NONE.value == "none"
        assert FrustumStyle.WEDGE.value == "wedge"
        assert FrustumStyle.PYRAMID.value == "pyramid"

    def test_from_string(self) -> None:
        assert FrustumStyle("none") is FrustumStyle.NONE
        assert FrustumStyle("wedge") is FrustumStyle.WEDGE
        assert FrustumStyle("pyramid") is FrustumStyle.PYRAMID

    def test_invalid_string(self) -> None:
        with pytest.raises(ValueError, match="invalid"):
            FrustumStyle("invalid")

    def test_is_str(self) -> None:
        """FrustumStyle is a str enum, usable as string."""
        assert isinstance(FrustumStyle.WEDGE, str)


class TestTrajectoryRenderConfig:
    """TrajectoryRenderConfig model tests."""

    def test_defaults(self) -> None:
        config = TrajectoryRenderConfig()
        assert config.width == 720
        assert config.height == 720
        assert config.elevation_deg == 30.0
        assert config.azimuth_deg == 45.0
        assert config.padding_ratio == 0.12
        assert config.fps == 30
        assert config.crf == 18
        assert config.frustum_style is FrustumStyle.WEDGE

    def test_custom_values(self) -> None:
        config = TrajectoryRenderConfig(
            width=1080,
            height=1080,
            elevation_deg=60.0,
            azimuth_deg=90.0,
            fps=60,
            frustum_style=FrustumStyle.PYRAMID,
        )
        assert config.width == 1080
        assert config.elevation_deg == 60.0
        assert config.frustum_style is FrustumStyle.PYRAMID

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            TrajectoryRenderConfig.model_validate({"nonexistent_field": 42})

    def test_rgb_tuples(self) -> None:
        config = TrajectoryRenderConfig(
            background_rgb=(0, 0, 0),
            ghost_color_rgb=(255, 255, 255),
            marker_edge_color_rgb=(128, 128, 128),
            grid_color_rgb=(10, 20, 30),
        )
        assert config.background_rgb == (0, 0, 0)
        assert config.ghost_color_rgb == (255, 255, 255)

    def test_serialization_roundtrip(self) -> None:
        config = TrajectoryRenderConfig(width=512, fps=60)
        data = config.model_dump()
        restored = TrajectoryRenderConfig(**data)
        assert restored == config

    def test_frustum_style_from_string_in_model(self) -> None:
        config = TrajectoryRenderConfig(
            frustum_style=FrustumStyle("pyramid"),
        )
        assert config.frustum_style is FrustumStyle.PYRAMID
