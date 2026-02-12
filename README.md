# trajviz

Fast 3D camera trajectory video renderer using NumPy + OpenCV.

Renders animated top-down trajectory visualizations from 3D camera positions, with configurable ghost trails, color-coded markers, glow effects, and camera frustum overlays. Frames are piped directly to ffmpeg as raw RGB — no intermediate files.

## Installation

```bash
pip install git+https://github.com/afromero/trajviz.git
```

## Usage

```python
import numpy as np
from trajviz import render_trajectory_video, TrajectoryRenderConfig

# Random 3D trajectory
positions = np.random.randn(120, 3).astype(np.float32)
positions = np.cumsum(positions * 0.1, axis=0)

# RGBA colors per frame (e.g., gradient from blue to red)
t = np.linspace(0, 1, 120)
colors = np.stack([t, np.zeros(120), 1 - t, np.ones(120)], axis=1).astype(np.float32)

render_trajectory_video(
    positions=positions,
    colors_rgba=colors,
    output_path="trajectory.mp4",
)
```

### Configuration

All rendering options are controlled via `TrajectoryRenderConfig`:

```python
from trajviz import TrajectoryRenderConfig, FrustumStyle

config = TrajectoryRenderConfig(
    width=1080,
    height=1080,
    elevation_deg=45.0,
    frustum_style=FrustumStyle.WEDGE,
    fps=30,
)

render_trajectory_video(positions, colors, "out.mp4", config=config)
```

### Side-by-side comparison

Combine a recording with its trajectory overlay:

```python
from trajviz import combine_videos_side_by_side

combine_videos_side_by_side(
    left_path="recording.mp4",
    right_path="trajectory.mp4",
    output_path="combined.mp4",
    fps=30,
)
```

## Requirements

- Python >= 3.10
- ffmpeg on PATH
- numpy, opencv-python-headless, pydantic, jaxtyping, beartype, difflogtest
