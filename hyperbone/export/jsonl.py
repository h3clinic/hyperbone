"""JSONL export for skeleton graphs."""

import json
from pathlib import Path
from typing import Dict, Any


def graph_to_record(
    video_id: str,
    frame_idx: int,
    timestamp_sec: float,
    object_id: int,
    graph: Dict,
    bbox: tuple = None,
    object_class: str = "unknown",
) -> Dict[str, Any]:
    """Build a single JSONL record from a graph dict."""
    record = {
        "video_id": video_id,
        "frame_idx": frame_idx,
        "timestamp_sec": round(timestamp_sec, 4),
        "object_id": object_id,
        "object_class": object_class,
        "coord_system": "image_xy",
        "nodes": graph.get("nodes", []),
        "edges": graph.get("edges", []),
        "bbox": list(bbox) if bbox else None,
        "quality": {
            "node_count": len(graph.get("nodes", [])),
            "edge_count": len(graph.get("edges", [])),
            "accepted": len(graph.get("nodes", [])) >= 2,
        },
    }
    return record


class JSONLWriter:
    """Append JSONL records to a file."""

    def __init__(self, output_dir: str, filename: str = "graphs.jsonl"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / filename
        self._file = open(self.path, "w", encoding="utf-8")

    def write(self, record: Dict[str, Any]):
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
