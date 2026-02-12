"""Generate demo trajectory video for README."""

import numpy as np

from trajviz import (
    FrustumStyle,
    TrajectoryRenderConfig,
    render_trajectory_video,
)


def main() -> None:
    """Render a spiral trajectory demo to assets/demo.mp4."""
    n = 200
    t = np.linspace(0, 6 * np.pi, n, dtype=np.float32)

    # 3D spiral: expanding radius, rising Z
    positions = np.stack(
        [
            np.cos(t) * (1 + t / (6 * np.pi)),
            np.sin(t) * (1 + t / (6 * np.pi)),
            t / (6 * np.pi) * 2,
        ],
        axis=1,
    )

    # Color gradient: cyan -> magenta
    frac = np.linspace(0, 1, n, dtype=np.float32)
    colors = np.stack(
        [
            frac,
            0.3 * (1 - frac),
            0.8 + 0.2 * (1 - frac),
            np.ones(n, dtype=np.float32),
        ],
        axis=1,
    )

    # Forward vectors = tangent to spiral
    forward = np.diff(positions, axis=0, prepend=positions[:1])
    norms = np.linalg.norm(forward, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    forward = (forward / norms).astype(np.float32)

    config = TrajectoryRenderConfig(
        width=480,
        height=480,
        elevation_deg=25.0,
        azimuth_deg=35.0,
        frustum_style=FrustumStyle.WEDGE,
        fps=30,
        crf=22,
        background_rgb=(18, 18, 36),
        grid_color_rgb=(40, 40, 60),
    )

    render_trajectory_video(
        positions=positions,
        colors_rgba=colors,
        output_path="assets/demo.mp4",
        config=config,
        forward_vectors=forward,
    )


if __name__ == "__main__":
    main()
