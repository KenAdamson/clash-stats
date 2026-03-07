"""Client for the SAMv2 tracking sidecar container.

Prepares frame windows from replay recordings, sends tracking requests
to the SAMv2 API, and maps results back to replay-guided label format.

The SAMv2 video predictor expects:
  - A directory of JPEG frames named as <integer>.jpg (e.g., 00000.jpg)
  - Bounding box prompts in pixel coordinates
  - All frames loaded into GPU memory at once

To keep VRAM usage manageable, we track in windows (e.g., 50-100 frames)
rather than feeding the entire 1900+ frame recording at once.
"""

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_SAMV2_URL = "http://localhost:8079"


@dataclass
class SpawnPrompt:
    """A unit spawn to track, derived from replay events."""
    object_id: int
    card_name: str
    team: str  # "friendly" or "opponent"
    spawn_frame: int  # 1-indexed frame in the original video
    bbox: tuple[float, float, float, float]  # normalized [x1, y1, x2, y2]


@dataclass
class TrackingResult:
    """Tracking result for one object at one frame."""
    object_id: int
    card_name: str
    team: str
    frame_number: int  # original video frame number
    bbox: tuple[float, float, float, float]  # normalized
    confidence: float


def prepare_frame_window(
    source_dir: Path,
    start_frame: int,
    end_frame: int,
    work_dir: Optional[Path] = None,
) -> Path:
    """Create a SAMv2-compatible frame directory from a window of frames.

    SAMv2 expects frames named as <integer>.jpg sorted numerically.
    Our source frames are frame_NNNN.jpg.

    Args:
        source_dir: directory with frame_NNNN.jpg files
        start_frame: first frame number (1-indexed)
        end_frame: last frame number (inclusive)
        work_dir: if None, creates a temp directory

    Returns:
        Path to the prepared frame directory (with 00000.jpg naming)
    """
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="samv2_window_"))
    else:
        work_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    for frame_num in range(start_frame, end_frame + 1):
        src = source_dir / f"frame_{frame_num:04d}.jpg"
        if not src.exists():
            logger.warning("Missing frame %d, skipping", frame_num)
            continue
        dst = work_dir / f"{idx:05d}.jpg"
        # Symlink to save disk space
        dst.symlink_to(src.resolve())
        idx += 1

    logger.info(
        "Prepared %d frames (video %d-%d) in %s",
        idx, start_frame, end_frame, work_dir,
    )
    return work_dir


def track_units(
    frame_dir: Path,
    prompts: list[SpawnPrompt],
    window_start_frame: int,
    samv2_url: str = DEFAULT_SAMV2_URL,
    confidence_threshold: float = 0.2,
    timeout: int = 300,
) -> list[TrackingResult]:
    """Send a tracking request to the SAMv2 sidecar.

    Args:
        frame_dir: SAMv2-compatible frame directory (00000.jpg naming)
        prompts: spawn prompts with frame numbers relative to ORIGINAL video
        window_start_frame: first original video frame in this window
        samv2_url: base URL of the SAMv2 sidecar
        confidence_threshold: minimum confidence to include results
        timeout: request timeout in seconds

    Returns:
        List of TrackingResult with frame numbers mapped back to original video
    """
    # Convert original frame numbers to window-relative (0-indexed for SAMv2)
    api_prompts = []
    for p in prompts:
        window_frame = p.spawn_frame - window_start_frame + 1  # 1-indexed within window
        if window_frame < 1:
            logger.warning(
                "Prompt %d (%s) spawn frame %d is before window start %d, skipping",
                p.object_id, p.card_name, p.spawn_frame, window_start_frame,
            )
            continue
        api_prompts.append({
            "object_id": p.object_id,
            "spawn_frame": window_frame,
            "bbox": list(p.bbox),
            "card_name": p.card_name,
            "team": p.team,
        })

    if not api_prompts:
        logger.warning("No valid prompts for this window")
        return []

    payload = {
        "frame_dir": str(frame_dir),
        "prompts": api_prompts,
        "confidence_threshold": confidence_threshold,
    }

    logger.info(
        "Tracking %d objects through %s (threshold=%.2f)",
        len(api_prompts), frame_dir, confidence_threshold,
    )

    resp = requests.post(
        f"{samv2_url}/track",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    # Map frame numbers back to original video coordinates
    results = []
    for r in data["results"]:
        original_frame = r["frame_number"] + window_start_frame - 1
        results.append(TrackingResult(
            object_id=r["object_id"],
            card_name=r["card_name"],
            team=r["team"],
            frame_number=original_frame,
            bbox=tuple(r["bbox"]),
            confidence=r["confidence"],
        ))

    logger.info(
        "Got %d tracking results across %d frames in %.1fs",
        len(results), data["frames_processed"], data["elapsed_seconds"],
    )
    return results


def cleanup_window(frame_dir: Path) -> None:
    """Remove a temporary frame window directory."""
    if frame_dir.exists() and "samv2_window_" in str(frame_dir):
        shutil.rmtree(frame_dir)
        logger.info("Cleaned up %s", frame_dir)
