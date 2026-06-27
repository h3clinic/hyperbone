"""Fetch glTF sample assets from KhronosGroup/glTF-Sample-Assets."""

import sys, json, shutil, urllib.request, zipfile, tempfile
from pathlib import Path
from datetime import datetime, timezone
import argparse


REPO_URL = "https://github.com/KhronosGroup/glTF-Sample-Assets"
RAW_BASE = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models"


# Known asset metadata
ASSET_META = {
    "Fox": {
        "license_note": "CC0 1.0 base model, CC-BY 4.0 rigging/animation by PixelMannen",
        "attribution_note": (
            "Fox model from glTF-Sample-Assets by KhronosGroup. "
            "Base mesh: CC0 1.0 Universal. "
            "Rigging and animation: CC-BY 4.0 by PixelMannen (@ARebelSpy)."
        ),
        "glb_subpath": "glTF-Binary/Fox.glb",
    },
}


def fetch_asset(asset_name: str, out_dir: str) -> Path:
    """Fetch a glTF sample asset.

    Downloads the .glb file directly from GitHub raw content.
    """
    if asset_name not in ASSET_META:
        raise ValueError(f"Unknown asset: {asset_name}. Known: {list(ASSET_META.keys())}")

    meta = ASSET_META[asset_name]
    out = Path(out_dir) / asset_name
    out.mkdir(parents=True, exist_ok=True)

    glb_subpath = meta["glb_subpath"]
    glb_dir = out / Path(glb_subpath).parent
    glb_dir.mkdir(parents=True, exist_ok=True)

    glb_path = out / glb_subpath
    if not glb_path.exists():
        url = f"{RAW_BASE}/{asset_name}/{glb_subpath}"
        print(f"[Fetch] Downloading {asset_name} from {url}")
        urllib.request.urlretrieve(url, str(glb_path))
        print(f"[Fetch] Saved: {glb_path}")
    else:
        print(f"[Fetch] Already exists: {glb_path}")

    # Verify
    if not glb_path.exists() or glb_path.stat().st_size < 1000:
        raise RuntimeError(f"Failed to fetch {asset_name}: {glb_path}")

    # Write manifest
    manifest = {
        "asset_name": asset_name,
        "source_repo": REPO_URL,
        "source_url": f"{RAW_BASE}/{asset_name}/{glb_subpath}",
        "local_model_path": str(glb_path.resolve()),
        "license_note": meta["license_note"],
        "attribution_note": meta["attribution_note"],
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "file_size_bytes": glb_path.stat().st_size,
    }

    manifest_path = out / "asset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Fetch] Manifest: {manifest_path}")

    return glb_path


def main():
    parser = argparse.ArgumentParser(description="Fetch glTF sample asset")
    parser.add_argument("--asset", required=True, help="Asset name (e.g. Fox)")
    parser.add_argument("--out", default="assets/gltf_samples", help="Output base directory")
    args = parser.parse_args()

    path = fetch_asset(args.asset, args.out)
    print(f"[Fetch] Done. Model: {path}")


if __name__ == "__main__":
    main()
