"""Dataset index — per-accepted-object row for downstream training."""

import json
from pathlib import Path
from typing import Dict, Any, List


class DatasetIndexWriter:
    """Write dataset_index.jsonl — one row per accepted object/frame."""

    def __init__(self, output_dir: str, filename: str = "dataset_index.jsonl"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / filename
        self._file = open(self.path, "w", encoding="utf-8")

    def write(self, row: Dict[str, Any]):
        self._file.write(json.dumps(row, separators=(",", ":")) + "\n")

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def build_index_row(
    video_id: str,
    frame_idx: int,
    timestamp_sec: float,
    object_id: int,
    frame_path: str = None,
    mask_path: str = None,
    overlay_path: str = None,
    graph_path: str = None,
    npz_path: str = None,
    quality_score: float = 0.0,
) -> Dict[str, Any]:
    """Build a single dataset index row."""
    return {
        "video_id": video_id,
        "frame_idx": frame_idx,
        "timestamp_sec": round(timestamp_sec, 4),
        "object_id": object_id,
        "frame_path": frame_path,
        "mask_path": mask_path,
        "overlay_path": overlay_path,
        "graph_path": graph_path,
        "npz_path": npz_path,
        "quality_score": quality_score,
    }
