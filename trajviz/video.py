"""FFmpeg-based video output via stdin pipe."""

import shutil
import subprocess

import numpy as np
from beartype import beartype
from difflogtest import get_logger
from jaxtyping import Float, jaxtyped

from trajviz.config import TrajectoryRenderConfig
from trajviz.projection import (
    build_projection_matrix,
    project_to_pixels,
)
from trajviz.renderer import render_trajectory_frames

logger = get_logger()


@jaxtyped(typechecker=beartype)
def render_trajectory_video(
    positions: Float[np.ndarray, "n 3"],
    colors_rgba: Float[np.ndarray, "n 4"],
    output_path: str,
    config: TrajectoryRenderConfig | None = None,
    forward_vectors: Float[np.ndarray, "n 3"] | None = None,
) -> str:
    """Render 3D trajectory to MP4 via ffmpeg stdin pipe.

    Projects 3D positions to 2D, renders each frame with
    OpenCV, and pipes raw RGB bytes directly to ffmpeg —
    no intermediate PNG files written to disk.

    Args:
        positions: Camera positions (n, 3) in world coords.
        colors_rgba: RGBA colors per position (n, 4), 0-1.
        output_path: Path for the output MP4 file.
        config: Rendering configuration (uses defaults if None).
        forward_vectors: Unit forward directions (n, 3) for
            frustum rendering. Required when frustum_style
            is not NONE.

    Returns:
        The output_path on success.

    Raises:
        FileNotFoundError: If ffmpeg is not installed.
        RuntimeError: If ffmpeg exits with non-zero code.

    """
    if config is None:
        config = TrajectoryRenderConfig()

    if shutil.which("ffmpeg") is None:
        msg = "ffmpeg not found on PATH"
        raise FileNotFoundError(msg)

    n = len(positions)
    logger.info(f"Rendering {n} frames to {output_path}")

    # ----------------------------------------------------------
    # Project 3D -> 2D pixel coordinates
    # ----------------------------------------------------------
    proj = build_projection_matrix(config.elevation_deg, config.azimuth_deg)
    pixels = project_to_pixels(
        positions,
        proj,
        config.width,
        config.height,
        config.padding_ratio,
    )

    # Convert RGBA float [0,1] to RGB int [0,255]
    colors_rgb = np.clip(colors_rgba[:, :3] * 255, 0, 255).astype(np.int32)

    # ----------------------------------------------------------
    # Project forward vectors to 2D (direction only)
    # ----------------------------------------------------------
    forward_pixels: Float[np.ndarray, "n 2"] | None = None
    if forward_vectors is not None:
        fwd_2d = (forward_vectors @ proj).astype(np.float64)
        # Flip Y to match screen coordinates (Y goes down)
        fwd_2d[:, 1] *= -1
        forward_pixels = fwd_2d

    # ----------------------------------------------------------
    # Open ffmpeg subprocess with stdin pipe
    # ----------------------------------------------------------
    w, h = config.width, config.height
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(config.fps),
        "-i",
        "pipe:",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(config.crf),
        "-preset",
        "fast",
        output_path,
    ]

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    stdin = process.stdin
    if stdin is None:
        msg = "ffmpeg stdin pipe not available"
        raise RuntimeError(msg)

    for frame in render_trajectory_frames(
        pixels, colors_rgb, config, forward_pixels
    ):
        stdin.write(frame.tobytes())

    stdin.close()
    return_code = process.wait()

    if return_code != 0:
        msg = f"ffmpeg exited with code {return_code}"
        raise RuntimeError(msg)

    logger.success(f"Trajectory video saved: {output_path}")
    return output_path


def combine_videos_side_by_side(
    left_path: str,
    right_path: str,
    output_path: str,
    fps: int,
) -> bool:
    """Combine two videos side-by-side with ffmpeg hstack.

    Scales the right video to match the left video height
    before horizontal stacking.

    Args:
        left_path: Path to left video (recording).
        right_path: Path to right video (trajectory).
        output_path: Path to output combined video.
        fps: Output video frame rate.

    Returns:
        True if combination succeeded.

    """
    if shutil.which("ffmpeg") is None:
        logger.error("ffmpeg not found")
        return False

    # Probe the left video height (scale2ref fails on
    # variable-resolution recordings common with HaxScan).
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=height",
        "-of",
        "csv=p=0",
        left_path,
    ]
    probe = subprocess.run(
        probe_cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    target_h = int(probe.stdout.strip()) if probe.returncode == 0 else 720
    # Ensure even for H.264
    target_h = target_h - (target_h % 2)

    filter_complex = (
        f"[0:v]scale=-2:{target_h}[left];"
        f"[1:v]scale=-2:{target_h}[right];"
        "[left][right]hstack=inputs=2[out]"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        left_path,
        "-i",
        right_path,
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-r",
        str(fps),
        output_path,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    if result.returncode != 0:
        logger.error(f"ffmpeg hstack failed (code {result.returncode})")
        return False

    return True
