"""Frame rendering using OpenCV drawing primitives."""

import math
from collections.abc import Iterator

import cv2
import numpy as np
from beartype import beartype
from difflogtest import get_logger
from jaxtyping import Float, Int, UInt8, jaxtyped

from trajviz.config import AxisPosition, FrustumStyle, TrajectoryRenderConfig

logger = get_logger()


@jaxtyped(typechecker=beartype)
def _compute_frustum_pixels(
    forward_2d: Float[np.ndarray, " 2"],
    center: tuple[int, int],
    length_px: float,
    half_fov_rad: float,
) -> Int[np.ndarray, "3 2"]:
    """Compute the 3 vertices of a FOV wedge triangle.

    Args:
        forward_2d: Projected 2D forward direction (unnormalized).
        center: Marker center pixel (cx, cy).
        length_px: Length of the frustum in pixels.
        half_fov_rad: Half of the horizontal FOV in radians.

    Returns:
        Triangle vertices (3, 2) as int32 pixel coords.

    """
    norm = np.linalg.norm(forward_2d)
    if norm < 1e-6:
        # Degenerate: forward vector projects to zero
        return np.array([center, center, center], dtype=np.int32)

    fwd = forward_2d / norm
    cos_h, sin_h = math.cos(half_fov_rad), math.sin(half_fov_rad)

    # Rotate forward by +/- half_fov
    left = np.array(
        [
            fwd[0] * cos_h - fwd[1] * sin_h,
            fwd[0] * sin_h + fwd[1] * cos_h,
        ]
    )
    right = np.array(
        [
            fwd[0] * cos_h + fwd[1] * sin_h,
            -fwd[0] * sin_h + fwd[1] * cos_h,
        ]
    )

    cx, cy = center
    origin = np.array([cx, cy], dtype=np.float64)
    p_left = origin + left * length_px
    p_right = origin + right * length_px

    return np.array(
        [
            [cx, cy],
            np.round(p_left).astype(np.int32),
            np.round(p_right).astype(np.int32),
        ],
        dtype=np.int32,
    )


@jaxtyped(typechecker=beartype)
def _draw_axis_indicator(
    canvas: UInt8[np.ndarray, "h w 3"],
    axis_directions: Float[np.ndarray, "3 2"],
    config: TrajectoryRenderConfig,
) -> None:
    """Draw an XYZ axis indicator in a corner of the frame.

    All axes use a single neutral color so they don't compete
    with trajectory colors. Each arrow is labelled X, Y, or Z.

    Args:
        canvas: RGB frame to draw on (modified in-place).
        axis_directions: Projected 2D directions for X, Y, Z axes
            with shape (3, 2). Y component should already be
            flipped for screen coordinates.
        config: Rendering configuration.

    """
    h, w = canvas.shape[:2]
    length = config.axis_length
    margin = config.axis_margin
    color = config.axis_color_rgb

    # Compute origin based on configured position
    pos = config.axis_position
    if pos == AxisPosition.BOTTOM_LEFT:
        ox = margin + length
        oy = h - margin - length
    elif pos == AxisPosition.BOTTOM_RIGHT:
        ox = w - margin - length
        oy = h - margin - length
    elif pos == AxisPosition.TOP_LEFT:
        ox = margin + length
        oy = margin + length
    else:  # TOP_RIGHT
        ox = w - margin - length
        oy = margin + length

    labels = ["X", "Y", "Z"]

    for i in range(3):
        direction = axis_directions[i]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            continue

        # Normalize and scale to pixel length
        dx = float(direction[0] / norm * length)
        dy = float(direction[1] / norm * length)
        end_x = int(ox + dx)
        end_y = int(oy + dy)

        cv2.arrowedLine(
            canvas,
            (ox, oy),
            (end_x, end_y),
            color,
            config.axis_thickness,
            cv2.LINE_AA,
            tipLength=0.2,
        )

        # Place label slightly beyond the arrow tip
        label_x = int(ox + dx * 1.25) + 2
        label_y = int(oy + dy * 1.25) + 4
        cv2.putText(
            canvas,
            labels[i],
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.axis_label_scale,
            color,
            1,
            cv2.LINE_AA,
        )


@jaxtyped(typechecker=beartype)
def render_trajectory_frames(
    pixel_positions: Int[np.ndarray, "n 2"],
    colors_rgb: Int[np.ndarray, "n 3"],
    config: TrajectoryRenderConfig,
    forward_pixels: Float[np.ndarray, "n 2"] | None = None,
    axis_directions: Float[np.ndarray, "3 2"] | None = None,
) -> Iterator[UInt8[np.ndarray, "h w 3"]]:
    """Yield rendered trajectory frames as RGB uint8 arrays.

    Each frame shows:
    1. Background fill
    2. Ghost trajectory (full path, faint)
    3. Grid lines (optional)
    4. Active trail up to the current frame
    5. Glow circle behind the marker
    6. Current-position marker with edge
    7. Camera frustum (optional wedge or pyramid)

    Args:
        pixel_positions: Pixel coordinates (n, 2) as int32.
        colors_rgb: Per-position RGB colors (n, 3) as int32.
        config: Rendering configuration.
        forward_pixels: Projected 2D forward directions (n, 2).
            Required when ``frustum_style`` is not NONE.
        axis_directions: Projected 2D directions for X, Y, Z axes
            with shape (3, 2). Drawn when ``axis_enabled`` is True.

    Yields:
        RGB uint8 frames with shape (height, width, 3).

    """
    n = len(pixel_positions)
    if n == 0:
        return

    h, w = config.height, config.width
    bg = np.array(config.background_rgb, dtype=np.uint8)

    # ----------------------------------------------------------
    # Frustum setup
    # ----------------------------------------------------------
    draw_frustum = (
        config.frustum_style != FrustumStyle.NONE
        and forward_pixels is not None
    )
    diag = math.sqrt(w * w + h * h)
    frustum_len_px = config.frustum_length * diag
    half_fov = math.radians(config.frustum_fov_deg / 2.0)

    # ----------------------------------------------------------
    # Pre-render ghost trajectory into a reusable layer
    # ----------------------------------------------------------
    ghost_layer = np.zeros((h, w, 3), dtype=np.uint8)
    ghost_layer[:] = bg
    ghost_color = config.ghost_color_rgb
    for j in range(n - 1):
        pt1 = tuple(pixel_positions[j].tolist())
        pt2 = tuple(pixel_positions[j + 1].tolist())
        cv2.line(
            ghost_layer,
            pt1,
            pt2,
            ghost_color,
            config.ghost_thickness,
            cv2.LINE_AA,
        )

    # ----------------------------------------------------------
    # Pre-compute grid line endpoints
    # ----------------------------------------------------------
    grid_lines: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if config.grid_enabled:
        divs = config.grid_divisions
        for k in range(1, divs):
            x = int(w * k / divs)
            y = int(h * k / divs)
            grid_lines.append(((x, 0), (x, h)))
            grid_lines.append(((0, y), (w, y)))

    ghost_alpha = config.ghost_alpha
    ghost_beta = 1.0 - ghost_alpha

    for i in logger.track(range(n), description="Rendering trajectory"):
        # Start from background
        canvas = np.empty((h, w, 3), dtype=np.uint8)
        canvas[:] = bg

        # Blend ghost trajectory
        cv2.addWeighted(
            ghost_layer,
            ghost_alpha,
            canvas,
            ghost_beta,
            0.0,
            canvas,
        )

        # Grid lines
        grid_color = config.grid_color_rgb
        for pt1, pt2 in grid_lines:
            cv2.line(canvas, pt1, pt2, grid_color, 1, cv2.LINE_AA)

        # Active trail segments (0..i)
        for j in range(i):
            pt1 = tuple(pixel_positions[j].tolist())
            pt2 = tuple(pixel_positions[j + 1].tolist())
            color = tuple(colors_rgb[j].tolist())
            cv2.line(
                canvas,
                pt1,
                pt2,
                color,
                config.trail_thickness,
                cv2.LINE_AA,
            )

        # Current position
        cx, cy = pixel_positions[i].tolist()
        marker_color = tuple(colors_rgb[i].tolist())

        # Glow circle (blended overlay)
        overlay = canvas.copy()
        cv2.circle(
            overlay,
            (cx, cy),
            config.glow_radius,
            marker_color,
            -1,
            cv2.LINE_AA,
        )
        cv2.addWeighted(
            overlay,
            config.glow_alpha,
            canvas,
            1.0 - config.glow_alpha,
            0.0,
            canvas,
        )

        # Camera frustum
        if draw_frustum and forward_pixels is not None:
            fwd_2d = forward_pixels[i]
            tri = _compute_frustum_pixels(
                fwd_2d, (cx, cy), frustum_len_px, half_fov
            )

            if config.frustum_style == FrustumStyle.WEDGE:
                # Filled semi-transparent wedge
                frustum_overlay = canvas.copy()
                cv2.fillPoly(
                    frustum_overlay,
                    [tri],
                    marker_color,
                    cv2.LINE_AA,
                )
                cv2.addWeighted(
                    frustum_overlay,
                    config.frustum_alpha,
                    canvas,
                    1.0 - config.frustum_alpha,
                    0.0,
                    canvas,
                )
                # Edge lines
                cv2.polylines(
                    canvas,
                    [tri],
                    isClosed=True,
                    color=config.marker_edge_color_rgb,
                    thickness=config.frustum_line_thickness,
                    lineType=cv2.LINE_AA,
                )
            else:
                # Pyramid wireframe
                cv2.line(
                    canvas,
                    (cx, cy),
                    tuple(tri[1].tolist()),
                    config.marker_edge_color_rgb,
                    config.frustum_line_thickness,
                    cv2.LINE_AA,
                )
                cv2.line(
                    canvas,
                    (cx, cy),
                    tuple(tri[2].tolist()),
                    config.marker_edge_color_rgb,
                    config.frustum_line_thickness,
                    cv2.LINE_AA,
                )
                cv2.line(
                    canvas,
                    tuple(tri[1].tolist()),
                    tuple(tri[2].tolist()),
                    config.marker_edge_color_rgb,
                    config.frustum_line_thickness,
                    cv2.LINE_AA,
                )

        # Marker: filled circle + edge
        cv2.circle(
            canvas,
            (cx, cy),
            config.marker_radius,
            marker_color,
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(
            canvas,
            (cx, cy),
            config.marker_radius,
            config.marker_edge_color_rgb,
            config.marker_edge_thickness,
            cv2.LINE_AA,
        )

        # XYZ axis indicator (drawn last so it's always on top)
        if config.axis_enabled and axis_directions is not None:
            _draw_axis_indicator(canvas, axis_directions, config)

        yield canvas
