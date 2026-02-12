"""3D to 2D orthographic projection math."""

import numpy as np
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped


@jaxtyped(typechecker=beartype)
def build_projection_matrix(
    elevation_deg: float,
    azimuth_deg: float,
) -> Float[np.ndarray, "3 2"]:
    """Build a 3x2 orthographic projection matrix.

    Applies Rz(azimuth) then Rx(elevation) and extracts the
    first two columns (XY) for screen projection.

    Args:
        elevation_deg: Elevation angle in degrees.
        azimuth_deg: Azimuth angle in degrees.

    Returns:
        Projection matrix with shape (3, 2).

    """
    el = np.radians(elevation_deg)
    az = np.radians(azimuth_deg)

    cos_el, sin_el = np.cos(el), np.sin(el)
    cos_az, sin_az = np.cos(az), np.sin(az)

    # Rz(azimuth)
    rz = np.array(
        [
            [cos_az, -sin_az, 0.0],
            [sin_az, cos_az, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Rx(elevation)
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_el, -sin_el],
            [0.0, sin_el, cos_el],
        ]
    )

    # Combined rotation: Rz @ Rx, then take XY columns.
    # This order ensures Z (up) projects straight up on screen,
    # cleanly separating all three axes. The previous Rx @ Rz
    # collapsed Y and Z to the same screen direction.
    rot = rz @ rx
    return rot[:, :2].astype(np.float64)


@jaxtyped(typechecker=beartype)
def project_to_pixels(
    positions: Float[np.ndarray, "n 3"],
    projection_matrix: Float[np.ndarray, "3 2"],
    width: int,
    height: int,
    padding_ratio: float,
) -> Int[np.ndarray, "n 2"]:
    """Project 3D positions to integer pixel coordinates.

    Applies orthographic projection, uniform scaling to fit
    the canvas with padding, centering, and Y-axis flip.

    Args:
        positions: 3D positions with shape (n, 3).
        projection_matrix: Projection matrix (3, 2) from
            ``build_projection_matrix``.
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        padding_ratio: Fraction of range used as border padding.

    Returns:
        Integer pixel coordinates with shape (n, 2).

    """
    # Project to 2D
    xy = positions @ projection_matrix

    # Compute bounding box
    xy_min = xy.min(axis=0)
    xy_max = xy.max(axis=0)
    xy_range = xy_max - xy_min

    # Prevent division by zero for degenerate trajectories
    xy_range = np.maximum(xy_range, 1e-6)

    # Uniform scale to fit both axes with padding
    usable_w = width * (1.0 - 2.0 * padding_ratio)
    usable_h = height * (1.0 - 2.0 * padding_ratio)
    scale = min(usable_w / xy_range[0], usable_h / xy_range[1])

    # Center in canvas
    center_xy = (xy_min + xy_max) / 2.0
    center_px = np.array([width / 2.0, height / 2.0])

    # Transform: center, scale, flip Y
    pixels = (xy - center_xy) * scale
    pixels[:, 1] *= -1  # Flip Y (screen Y goes down)
    pixels += center_px

    return np.round(pixels).astype(np.int32)
