"""Manifest utilities — track pipeline inputs and outputs."""

import json
from pathlib import Path
from typing import Any


def write_manifest(output_dir: str, info: dict[str, Any]) -> Path:
    """Write a pipeline run manifest to output_dir/manifest.json."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.json"
    path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return path


def read_manifest(output_dir: str) -> dict[str, Any]:
    """Read a previously written manifest."""
    path = Path(output_dir) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))
