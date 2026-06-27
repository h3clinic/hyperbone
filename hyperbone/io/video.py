"""Video loading and frame extraction using OpenCV."""

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple


def get_video_info(video_path: str) -> dict:
    """Return basic metadata for a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    info = {
        "path": str(Path(video_path).resolve()),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration_sec"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info


def sample_frames(
    video_path: str,
    sample_fps: float = 1.0,
    skip_start_sec: float = 0.0,
    skip_end_sec: float = 0.0,
) -> Iterator[Tuple[int, float, np.ndarray]]:
    """Yield (frame_idx, timestamp_sec, bgr_frame) at the given sample rate.

    Args:
        skip_start_sec: Skip this many seconds from the start.
        skip_end_sec: Skip this many seconds from the end.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        cap.release()
        raise ValueError(f"Invalid FPS in video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps if video_fps > 0 else 0
    end_cutoff_sec = max(0, duration_sec - skip_end_sec)

    frame_interval = max(1, int(round(video_fps / sample_fps)))
    start_frame = int(skip_start_sec * video_fps)

    # Seek to start frame if skipping
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = frame_idx / video_fps
        if timestamp >= end_cutoff_sec:
            break
        if frame_idx % frame_interval == 0:
            yield frame_idx, timestamp, frame
        frame_idx += 1

    cap.release()
