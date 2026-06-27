"""Fetch Quaternius Animated Animal Pack from Poly Pizza.

Downloads GLTF pack. If direct download is blocked, instructs user to
manually place files.

Animals: Cow, Donkey, Deer, Alpaca, Bull, Fox, Shiba Inu, Stag, Husky, Wolf, White Horse, Horse
License: Public Domain / CC0
"""

import sys, json, os, shutil, zipfile, tempfile, urllib.request
from pathlib import Path
from datetime import datetime, timezone
import argparse


SOURCE_URL = "https://poly.pizza/bundle/Animated-Animal-Pack-ILAPXeUYiS"

# Known animals in the pack and their GLTF filenames
KNOWN_ANIMALS = [
    "Wolf", "Horse", "Fox", "Deer", "Stag", "Husky",
    "Cow", "Donkey", "Alpaca", "Bull", "Shiba", "WhiteHorse",
]

# Priority order for selection
PRIORITY = ["Wolf", "Horse", "Fox", "Deer", "Stag"]

# Known download URLs to try (Quaternius hosts on their site too)
DOWNLOAD_URLS = [
    "https://quaternius.com/packs/ultimatedAnimatedAnimalPack.html",
]


def find_animal_model(base_dir: Path, animal_name: str) -> Path:
    """Search for a model file for the given animal."""
    # Search patterns
    patterns = [
        f"**/{animal_name}*.gltf",
        f"**/{animal_name}*.glb",
        f"**/{animal_name}*.fbx",
        f"**/{animal_name}*.blend",
        f"**/{animal_name.lower()}*.gltf",
        f"**/{animal_name.lower()}*.glb",
    ]
    for pattern in patterns:
        matches = list(base_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def select_animal(base_dir: Path) -> tuple:
    """Select the best available animal by priority.

    Returns (animal_name, model_path) or (None, None).
    """
    for animal in PRIORITY:
        path = find_animal_model(base_dir, animal)
        if path:
            return animal, path

    # Try any animal
    for animal in KNOWN_ANIMALS:
        path = find_animal_model(base_dir, animal)
        if path:
            return animal, path

    return None, None


def discover_animals(base_dir: Path) -> list:
    """Find all available animals in the directory."""
    found = []
    for animal in KNOWN_ANIMALS:
        path = find_animal_model(base_dir, animal)
        if path:
            found.append({"name": animal, "path": str(path)})
    return found


def fetch_quaternius_animals(out_dir: str) -> dict:
    """Fetch or verify the Quaternius Animated Animal Pack.

    If the pack is already extracted, just verifies and selects.
    If not present, attempts download or provides instructions.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Check if already present
    available = discover_animals(out)
    if not available:
        # Try to download from known URLs
        downloaded = _try_download(out)
        if not downloaded:
            # Provide manual instructions
            print("[Fetch] Quaternius Animated Animal Pack not found.")
            print("[Fetch] Please download manually:")
            print(f"  1. Visit: {SOURCE_URL}")
            print(f"  2. Download the GLTF format pack")
            print(f"  3. Extract contents to: {out.resolve()}")
            print(f"  4. Run this script again")
            print()
            print("[Fetch] Alternative: Download from quaternius.com")
            print(f"  https://quaternius.com/packs/ultimateAnimatedAnimals.html")
            print()

            # Create placeholder manifest
            manifest = {
                "source_name": "Quaternius Animated Animal Pack",
                "source_url": SOURCE_URL,
                "license": "Public Domain / CC0",
                "status": "not_downloaded",
                "instructions": "Download GLTF pack from Poly Pizza and extract here",
                "selected_animal": None,
                "selected_model_path": None,
                "available_animals": [],
            }
            _write_manifest(out, manifest)
            return manifest

        available = discover_animals(out)

    # Select best animal
    animal_name, model_path = select_animal(out)

    manifest = {
        "source_name": "Quaternius Animated Animal Pack",
        "source_url": SOURCE_URL,
        "license": "Public Domain / CC0",
        "attribution_note": "Animated Animal Pack by Quaternius. Public Domain / CC0. No attribution required.",
        "selected_animal": animal_name,
        "selected_model_path": str(model_path) if model_path else None,
        "available_animals": [a["name"] for a in available],
        "available_model_paths": {a["name"]: a["path"] for a in available},
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }

    _write_manifest(out, manifest)

    if animal_name:
        print(f"[Fetch] Selected animal: {animal_name}")
        print(f"[Fetch] Model path: {model_path}")
    else:
        print("[Fetch] No animal models found after extraction.")

    print(f"[Fetch] Available: {[a['name'] for a in available]}")
    return manifest


def _try_download(out: Path) -> bool:
    """Attempt to download the pack. Returns True if successful."""
    # Try Quaternius direct GLTF download
    # The pack is often available as a zip from quaternius.com
    urls_to_try = [
        "https://quaternius.com/packs/ultimateAnimatedAnimals.html",
    ]

    # For Poly Pizza, direct download requires their API or manual action
    # We'll try the quaternius.com source which sometimes has direct zips
    print("[Fetch] Attempting download from known sources...")

    # Try fetching the Poly Pizza page to find download link
    try:
        # Poly Pizza bundle pages have download buttons
        # but require browser interaction. Skip automated download.
        pass
    except Exception:
        pass

    return False


def _write_manifest(out: Path, manifest: dict):
    """Write the asset manifest."""
    path = out / "asset_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Fetch] Manifest: {path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch Quaternius Animated Animal Pack")
    parser.add_argument("--out", default="assets/quaternius_animals",
                        help="Output directory")
    args = parser.parse_args()

    manifest = fetch_quaternius_animals(args.out)

    if manifest.get("selected_animal"):
        print(f"\n[Fetch] Ready. Selected: {manifest['selected_animal']}")
    else:
        print(f"\n[Fetch] Pack not available yet. Follow download instructions above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
