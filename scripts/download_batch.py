"""
HyperBone Movie Batch Downloader
Replicates the Crawly/Elixir pipeline in Python:
  1. Reads movie list from priv/hyperbone_movies.json
  2. For each movie, resolves an HLS stream URL via Playwright (headless browser)
  3. Downloads the movie using yt-dlp
  4. Trims 10 minutes from front and back using ffmpeg

Usage:
    python scripts/download_batch.py [--dry-run] [--category animals] [--limit 5] [--skip-trim]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


MOVIE_LIST = Path(__file__).resolve().parent.parent / "HyperVid" / "priv" / "hyperbone_movies.json"
DOWNLOAD_DIR = Path.home() / "Downloads" / "HyperBone_Movies"
TRIMMED_DIR = Path.home() / "Downloads" / "HyperBone_Movies_Trimmed"

STREAM_PROVIDERS = [
    ("xpass", "https://play.xpass.top/e/movie/{id}"),
    ("zxcstream", "https://www.zxcstream.xyz/player/movie/{id}"),
    ("vidcore", "https://vidcore.net/movie/{id}"),
    ("cinemaos", "https://cinemaos.tech/player/{id}"),
    ("airflix", "https://airflix1.com/embed/movie/{id}"),
    ("peachify", "https://peachify.top/embed/movie/{id}"),
    ("vidzen", "https://vidzen.fun/movie/{id}"),
    ("vidplays", "https://vidplays.fun/embed/movie/{id}"),
    ("videasy", "https://player.videasy.net/movie/{id}"),
    ("screenscape", "https://screenscape.me/embed?tmdb={id}&type=movie"),
    ("modocine", "https://play.modocine.com/play.php/embed/movie/{id}"),
    ("frembed", "https://frembed.buzz/api/film.php?id={id}"),
]

FFMPEG_PATH = None


def find_ffmpeg():
    """Find ffmpeg binary (imageio_ffmpeg fallback)."""
    global FFMPEG_PATH
    # Check PATH first
    import shutil
    path = shutil.which("ffmpeg")
    if path:
        FFMPEG_PATH = path
        return

    # Fallback to imageio_ffmpeg
    try:
        import imageio_ffmpeg
        FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass


def resolve_stream_url(player_url: str, timeout: int = 30000) -> str | None:
    """Use Playwright to resolve HLS stream URL from a player page."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": 1440, "height": 900},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        )

        candidate_urls = set()

        def on_response(response):
            url = response.url
            if re.search(r'\.m3u8(\?|$)|/playlist/|\.txt(\?|$)', url):
                candidate_urls.add(url)

        page.on("response", on_response)

        try:
            page.goto(player_url, wait_until="domcontentloaded", timeout=timeout)
            page.wait_for_timeout(8000)

            # Try extracting playlist from page HTML
            html = page.content()
            match = re.search(r'"playlist":"([^"]+)"', html)
            if match:
                playlist_url = match.group(1)
                if not playlist_url.startswith("http"):
                    playlist_url = f"https://play.xpass.top{playlist_url}"
                try:
                    resp = page.request.get(playlist_url, headers={
                        "Referer": "https://play.xpass.top/"
                    })
                    if resp.ok:
                        data = resp.json()
                        sources = data.get("playlist", [{}])[0].get("sources", [])
                        for src in sources:
                            if src.get("file"):
                                candidate_urls.add(src["file"])
                except Exception:
                    pass

        except Exception as e:
            print(f"    [resolve] Error navigating to {player_url}: {e}")
        finally:
            browser.close()

        # Sort: prefer cf-master / index patterns
        ordered = sorted(candidate_urls, key=lambda u: (
            0 if re.search(r'cf-master|index-f\d+-v1-a1\.txt', u) else 1,
            u
        ))

        return ordered[0] if ordered else None


def download_movie(title: str, stream_url: str, output_dir: Path) -> Path | None:
    """Download a movie using yt-dlp."""
    clean_title = re.sub(r'[^\w\s\-]', '', title).strip()
    output_template = str(output_dir / f"{clean_title}.%(ext)s")

    args = [
        sys.executable, "-m", "yt_dlp",
        stream_url,
        "-o", output_template,
        "--referer", "https://play.xpass.top/",
        "--no-playlist",
        "--abort-on-error",
        "--socket-timeout", "60",
        "--retries", "5",
        "--fragment-retries", "10",
        "--concurrent-fragments", "4",
    ]

    if FFMPEG_PATH:
        args.extend(["--ffmpeg-location", FFMPEG_PATH])

    print(f"    [download] Starting: {clean_title}")
    # Stream output live so progress is visible; no timeout (movies can take 1hr+)
    try:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        last_line = ""
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                last_line = line
                # Print progress lines (contains %)
                if "%" in line or "Destination" in line or "Merging" in line:
                    print(f"    [yt-dlp] {line}")
        proc.wait()
        returncode = proc.returncode
    except Exception as e:
        print(f"    [download] Exception: {e}")
        return None

    if returncode == 0:
        # Find the downloaded file
        for ext in [".mp4", ".mkv", ".webm", ".ts"]:
            candidate = output_dir / f"{clean_title}{ext}"
            if candidate.exists():
                return candidate
        # Glob fallback
        matches = sorted(output_dir.glob(f"{clean_title}.*"), key=lambda p: p.stat().st_size, reverse=True)
        matches = [m for m in matches if m.suffix in (".mp4", ".mkv", ".webm", ".ts")]
        if matches:
            return matches[0]
        print(f"    [download] Completed but file not found for: {clean_title}")
        return None
    else:
        print(f"    [download] FAILED (exit {returncode}): {last_line}")
        return None


def trim_video(input_path: Path, output_path: Path, trim_front_sec: int = 600, trim_back_sec: int = 600) -> bool:
    """Trim front and back from a video using ffmpeg stream copy."""
    if not FFMPEG_PATH:
        print("    [trim] ffmpeg not found, skipping trim")
        return False

    import cv2
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps <= 0 or frame_count <= 0:
        print(f"    [trim] Cannot read video metadata: {input_path.name}")
        return False

    duration = frame_count / fps
    content_duration = duration - trim_front_sec - trim_back_sec

    if content_duration < 60:
        print(f"    [trim] Video too short ({duration:.0f}s) to trim {trim_front_sec+trim_back_sec}s. Copying as-is.")
        import shutil
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    args = [
        FFMPEG_PATH,
        "-y",
        "-ss", str(trim_front_sec),
        "-i", str(input_path),
        "-t", str(content_duration),
        "-c", "copy",
        str(output_path)
    ]

    result = subprocess.run(args, capture_output=True, text=True, timeout=600)
    if result.returncode == 0 or output_path.exists():
        print(f"    [trim] OK: {output_path.name} ({content_duration/60:.1f} min)")
        return True
    else:
        print(f"    [trim] FAILED: {result.stderr[:200]}")
        return False


def process_movie(movie: dict, output_dir: Path, trimmed_dir: Path, dry_run: bool, skip_trim: bool) -> dict:
    """Process a single movie: resolve → download → trim."""
    title = movie["title"]
    tmdb_id = movie["tmdb_id"]
    category = movie.get("category", "unknown")

    result = {
        "title": title,
        "tmdb_id": tmdb_id,
        "category": category,
        "status": "pending",
        "stream_url": None,
        "file_path": None,
        "trimmed_path": None,
    }

    print(f"\n  [{category}] {title} (TMDB {tmdb_id})")

    if dry_run:
        result["status"] = "dry_run"
        providers = [name for name, _ in STREAM_PROVIDERS[:3]]
        print(f"    [dry-run] Would try providers: {', '.join(providers)}")
        return result

    # Check if already downloaded
    clean_title = re.sub(r'[^\w\s\-]', '', title).strip()
    existing = list(output_dir.glob(f"{clean_title}.*"))
    # Filter out partial/temp files
    existing = [f for f in existing if not f.suffix in (".part", ".ytdl") and f.stat().st_size > 10_000_000]
    if existing:
        print(f"    [skip] Already downloaded: {existing[0].name}")
        result["status"] = "already_exists"
        result["file_path"] = str(existing[0])

        if not skip_trim:
            trimmed_path = trimmed_dir / existing[0].name
            if not trimmed_path.exists():
                trim_video(existing[0], trimmed_path)
            result["trimmed_path"] = str(trimmed_path)
        return result

    # Try providers in order
    for provider_name, url_template in STREAM_PROVIDERS:
        player_url = url_template.format(id=tmdb_id)
        print(f"    [resolve] Trying {provider_name}...")

        stream_url = resolve_stream_url(player_url)
        if stream_url:
            print(f"    [resolve] Got stream from {provider_name}")
            result["stream_url"] = stream_url

            downloaded = download_movie(title, stream_url, output_dir)
            if downloaded:
                result["status"] = "downloaded"
                result["file_path"] = str(downloaded)

                if not skip_trim:
                    trimmed_path = trimmed_dir / downloaded.name
                    if trim_video(downloaded, trimmed_path):
                        result["trimmed_path"] = str(trimmed_path)
                return result
            else:
                print(f"    [resolve] Download failed from {provider_name}, trying next...")
                continue
        else:
            continue

    result["status"] = "failed"
    print(f"    [FAILED] Could not resolve any stream for: {title}")
    return result


def main():
    parser = argparse.ArgumentParser(description="HyperBone batch movie downloader")
    parser.add_argument("--dry-run", action="store_true", help="Don't download, just show what would happen")
    parser.add_argument("--category", type=str, help="Filter by category (e.g. animals, robots, dreamworks)")
    parser.add_argument("--limit", type=int, help="Max movies to process")
    parser.add_argument("--skip-trim", action="store_true", help="Skip 10-min trim step")
    parser.add_argument("--start-from", type=int, default=0, help="Start from Nth movie (0-indexed)")
    args = parser.parse_args()

    if not MOVIE_LIST.exists():
        print(f"ERROR: Movie list not found: {MOVIE_LIST}")
        sys.exit(1)

    find_ffmpeg()
    if FFMPEG_PATH:
        print(f"[ffmpeg] {FFMPEG_PATH}")
    else:
        print("[ffmpeg] NOT FOUND — trimming will be skipped")

    with open(MOVIE_LIST) as f:
        movies = json.load(f)

    if args.category:
        movies = [m for m in movies if m.get("category") == args.category]

    movies = movies[args.start_from:]

    if args.limit:
        movies = movies[:args.limit]

    print(f"\n{'='*60}")
    print(f"HyperBone Batch Download")
    print(f"Movies: {len(movies)}")
    print(f"Output: {DOWNLOAD_DIR}")
    print(f"Trimmed: {TRIMMED_DIR}")
    print(f"Trim: 10 min front + 10 min back")
    print(f"{'='*60}")

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    TRIMMED_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for i, movie in enumerate(movies):
        print(f"\n[{i+1}/{len(movies)}] ", end="")
        result = process_movie(movie, DOWNLOAD_DIR, TRIMMED_DIR, args.dry_run, args.skip_trim)
        results.append(result)

        # Save progress after each movie
        progress_path = DOWNLOAD_DIR / "download_progress.json"
        with open(progress_path, "w") as f:
            json.dump(results, f, indent=2)

    # Final summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    by_status = {}
    for r in results:
        by_status.setdefault(r["status"], []).append(r["title"])

    for status, titles in sorted(by_status.items()):
        print(f"\n  {status}: {len(titles)}")
        for t in titles[:10]:
            print(f"    - {t}")
        if len(titles) > 10:
            print(f"    ... and {len(titles)-10} more")

    print(f"\nProgress saved to: {DOWNLOAD_DIR / 'download_progress.json'}")


if __name__ == "__main__":
    main()
