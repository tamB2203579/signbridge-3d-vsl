"""
Kaggle Vietnamese Sign Language (VSL) Dataset Manager.
Handles metadata caching, deterministic random sampling, and on-demand video downloading.
"""

import os
import csv
import time
import random
import logging
import subprocess
import urllib.parse
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests

logger = logging.getLogger("dataset_manager")

DEFAULT_KAGGLE_DATASET = "aresusayhi/vsl-vietnamese-sign-languages"
DEFAULT_MANIFEST_URL = (
    f"https://www.kaggle.com/api/v1/datasets/download/{DEFAULT_KAGGLE_DATASET}/Dataset%2FLabels%2Flabel.csv"
)
DEFAULT_VIDEO_BASE_URL = (
    f"https://www.kaggle.com/api/v1/datasets/download/{DEFAULT_KAGGLE_DATASET}/Dataset%2FVideos%2F"
)

MIN_VALID_VIDEO_BYTES = 10 * 1024  # 10 KB threshold to verify valid MP4


class KaggleDatasetManager:
    def __init__(
        self,
        base_dir: Optional[str | Path] = None,
        manifest_url: str = DEFAULT_MANIFEST_URL,
        video_base_url: str = DEFAULT_VIDEO_BASE_URL,
    ):
        if base_dir is None:
            # Default to repo root / data
            self.base_dir = Path(__file__).resolve().parent.parent / "data"
        else:
            self.base_dir = Path(base_dir)

        self.manifest_path = self.base_dir / "label.csv"
        self.videos_dir = self.base_dir / "kaggle_vsl" / "videos"
        self.manifest_url = manifest_url
        self.video_base_url = video_base_url

        self._labels: List[Dict[str, Any]] = []
        self._labels_by_video: Dict[str, Dict[str, Any]] = {}

        self.ensure_directories()

    def ensure_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)

    def ensure_labels_downloaded(self, force: bool = False) -> Path:
        """Download label.csv from Kaggle if missing or forced."""
        if not force and self.manifest_path.exists() and self.manifest_path.stat().st_size > 100:
            return self.manifest_path

        logger.info("Downloading Kaggle VSL manifest from %s", self.manifest_url)
        temp_dest = self.manifest_path.with_suffix(".tmp")
        try:
            # Attempt with requests first
            response = requests.get(self.manifest_url, allow_redirects=True, timeout=30)
            if response.status_code == 200 and len(response.content) > 100:
                with open(temp_dest, "wb") as f:
                    f.write(response.content)
            else:
                # Fallback to curl
                cmd = ["curl.exe" if os.name == "nt" else "curl", "-L", "-s", "-o", str(temp_dest), self.manifest_url]
                subprocess.run(cmd, check=True, timeout=60)

            if temp_dest.exists() and temp_dest.stat().st_size > 100:
                if self.manifest_path.exists():
                    self.manifest_path.unlink()
                temp_dest.rename(self.manifest_path)
                logger.info("Successfully saved label manifest to %s", self.manifest_path)
            else:
                raise RuntimeError("Downloaded manifest is missing or invalid size")
        finally:
            if temp_dest.exists():
                temp_dest.unlink()

        return self.manifest_path

    def load_labels(self, force_reload: bool = False) -> List[Dict[str, Any]]:
        """Load and parse records from label.csv."""
        if self._labels and not force_reload:
            return self._labels

        self.ensure_labels_downloaded()

        records = []
        with open(self.manifest_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalise keys: ID, VIDEO, LABEL
                rec_id = row.get("ID", "").strip()
                video = row.get("VIDEO", "").strip()
                label = row.get("LABEL", "").strip()
                if video:
                    record = {
                        "id": rec_id,
                        "video": video,
                        "label": label,
                    }
                    records.append(record)

        self._labels = records
        self._labels_by_video = {r["video"].lower(): r for r in records}
        logger.info("Loaded %d label records from %s", len(self._labels), self.manifest_path)
        return self._labels

    def is_video_cached(self, video_filename: str) -> bool:
        """Check if video file exists locally and has valid size."""
        target_path = self.videos_dir / video_filename
        return target_path.exists() and target_path.stat().st_size >= MIN_VALID_VIDEO_BYTES

    def get_dataset_info(self) -> Dict[str, Any]:
        """Return dataset statistics and cache details."""
        records = self.load_labels()
        cached_count = 0
        cached_bytes = 0

        if self.videos_dir.exists():
            for f in self.videos_dir.iterdir():
                if f.is_file() and f.suffix.lower() == ".mp4" and f.stat().st_size >= MIN_VALID_VIDEO_BYTES:
                    cached_count += 1
                    cached_bytes += f.stat().st_size

        return {
            "total_samples": len(records),
            "cached_videos_count": cached_count,
            "cached_size_mb": round(cached_bytes / (1024 * 1024), 2),
            "manifest_path": str(self.manifest_path),
            "videos_dir": str(self.videos_dir),
        }

    def search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search records by label or video name."""
        records = self.load_labels()
        q_lower = query.strip().lower()

        results = []
        for r in records:
            if q_lower in r["label"].lower() or q_lower in r["video"].lower():
                results.append({
                    **r,
                    "is_cached": self.is_video_cached(r["video"]),
                })
                if len(results) >= limit:
                    break

        return results

    def get_by_video_name(self, video_name: str) -> Optional[Dict[str, Any]]:
        """Find record by video name (matches 'D0001N.mp4' or 'D0001N')."""
        self.load_labels()
        v_norm = video_name.strip().lower()
        if not v_norm.endswith(".mp4"):
            v_norm += ".mp4"

        record = self._labels_by_video.get(v_norm)
        if record:
            return {
                **record,
                "is_cached": self.is_video_cached(record["video"]),
            }
        return None

    def sample(self, n: int, seed: int = 42) -> List[Dict[str, Any]]:
        """
        Draw n deterministic random samples using Python random.Random(seed).
        Clamps n to available dataset count.
        """
        records = self.load_labels()
        if not records:
            return []

        count = min(max(1, n), len(records))
        rng = random.Random(seed)
        sampled = rng.sample(records, count)

        return [
            {
                **item,
                "is_cached": self.is_video_cached(item["video"]),
            }
            for item in sampled
        ]

    def download_video(self, video_filename: str, retries: int = 3) -> Path:
        """
        Download a single video file on-demand using curl / requests with retries.
        Skips download if file is already cached and valid.
        """
        self.ensure_directories()
        video_name = video_filename.strip()
        if not video_name.endswith(".mp4"):
            video_name += ".mp4"

        dest_path = self.videos_dir / video_name
        if dest_path.exists() and dest_path.stat().st_size >= MIN_VALID_VIDEO_BYTES:
            logger.info("Video %s already cached (%d bytes)", video_name, dest_path.stat().st_size)
            return dest_path

        encoded_filename = urllib.parse.quote(video_name)
        url = f"{self.video_base_url}{encoded_filename}"
        temp_dest = dest_path.with_suffix(".tmp")

        last_error = None
        for attempt in range(1, retries + 1):
            try:
                logger.info(
                    "Downloading video %s (attempt %d/%d) from %s",
                    video_name, attempt, retries, url
                )
                if temp_dest.exists():
                    temp_dest.unlink()

                # Stream with requests
                with requests.get(url, stream=True, timeout=60, allow_redirects=True) as r:
                    r.raise_for_status()
                    with open(temp_dest, "wb") as f:
                        for chunk in r.iter_content(chunk_size=64 * 1024):
                            if chunk:
                                f.write(chunk)

                # Validate size and MP4 signature
                if temp_dest.exists() and temp_dest.stat().st_size >= MIN_VALID_VIDEO_BYTES:
                    # Check first 16 bytes for valid file type signature (ftyp)
                    with open(temp_dest, "rb") as f:
                        header = f.read(16)
                    if b"ftyp" in header or b"\x00\x00\x00" in header:
                        if dest_path.exists():
                            dest_path.unlink()
                        temp_dest.rename(dest_path)
                        logger.info(
                            "Video %s successfully downloaded to %s (%d bytes)",
                            video_name, dest_path, dest_path.stat().st_size
                        )
                        return dest_path
                    else:
                        raise ValueError(f"Downloaded content for {video_name} is not an MP4 video")
                else:
                    raise IOError(
                        f"Downloaded file too small ({temp_dest.stat().st_size if temp_dest.exists() else 0} bytes)"
                    )
            except Exception as e:
                last_error = e
                logger.warning("Attempt %d failed to download %s: %s", attempt, video_name, e)
                if temp_dest.exists():
                    temp_dest.unlink()
                time.sleep(2 ** (attempt - 1))

        raise RuntimeError(f"Failed to download video {video_name} after {retries} attempts: {last_error}")
