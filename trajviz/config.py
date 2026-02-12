"""Rendering configuration for trajectory videos."""

from enum import Enum

from pydantic import BaseModel, ConfigDict


class FrustumStyle(str, Enum):
    """Camera frustum visualization style."""

    NONE = "none"
    """No frustum drawn."""

    WEDGE = "wedge"
    """Filled triangle showing horizontal FOV angle."""

    PYRAMID = "pyramid"
    """Wireframe pyramid showing full 3D FOV."""


class TrajectoryRenderConfig(BaseModel):
    """Rendering configuration for trajectory videos."""

    model_config = ConfigDict(extra="forbid")

    width: int = 720
    """Output frame width in pixels."""

    height: int = 720
    """Output frame height in pixels."""

    elevation_deg: float = 30.0
    """Camera elevation angle in degrees."""

    azimuth_deg: float = 45.0
    """Camera azimuth angle in degrees."""

    padding_ratio: float = 0.12
    """Fraction of bounding-box range used as padding."""

    background_rgb: tuple[int, int, int] = (26, 26, 46)
    """Background color as (R, G, B) 0-255."""

    ghost_color_rgb: tuple[int, int, int] = (80, 80, 80)
    """Color for the ghost (full-path) trajectory."""

    ghost_alpha: float = 0.15
    """Blend alpha for ghost trajectory overlay."""

    ghost_thickness: int = 1
    """Line thickness for ghost trajectory."""

    trail_thickness: int = 2
    """Line thickness for active trail segments."""

    marker_radius: int = 8
    """Radius of the current-position marker."""

    marker_edge_thickness: int = 2
    """Edge thickness for current-position marker."""

    marker_edge_color_rgb: tuple[int, int, int] = (255, 255, 255)
    """Edge color for current-position marker."""

    glow_radius: int = 18
    """Radius of the glow circle behind the marker."""

    glow_alpha: float = 0.25
    """Blend alpha for the marker glow."""

    grid_enabled: bool = True
    """Whether to draw a background grid."""

    grid_color_rgb: tuple[int, int, int] = (50, 45, 45)
    """Color for grid lines."""

    grid_divisions: int = 5
    """Number of grid divisions per axis."""

    frustum_style: FrustumStyle = FrustumStyle.WEDGE
    """Camera frustum visualization style."""

    frustum_length: float = 0.06
    """Frustum length as fraction of canvas diagonal."""

    frustum_fov_deg: float = 60.0
    """Horizontal field of view for the frustum wedge/pyramid."""

    frustum_alpha: float = 0.35
    """Blend alpha for the filled frustum wedge."""

    frustum_line_thickness: int = 1
    """Line thickness for pyramid frustum wireframe."""

    fps: int = 30
    """Output video frame rate."""

    crf: int = 18
    """H.264 constant rate factor (lower = higher quality)."""
