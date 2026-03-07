"""Full-game SAMv2 tracking pipeline.

Processes all troop events in a battle replay:
  1. Load replay events from the database
  2. For each troop spawn, create a tracking window
  3. Send to SAMv2 sidecar for frame-to-frame tracking
  4. Merge results into per-frame labels

Spells and buildings are handled by replay_guided_labels.py (fixed positions).
Only troops need SAMv2 tracking because they move.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from tracker.ml.card_metadata import kebab_to_title
from tracker.models import Battle, ReplayEvent
from tracker.vision.card_properties import get_properties
from tracker.vision.replay_guided_labels import arena_to_screen
from tracker.vision.samv2_client import SpawnPrompt, TrackingResult, track_units

logger = logging.getLogger(__name__)

# Units that are invisible or hard to track at spawn time
SKIP_TRACKING = {
    "Miner",  # burrows underground, appears ~1s later at target position
}

TICKS_PER_SECOND = 20
DEFAULT_FPS = 10.0
DEFAULT_WINDOW_SECONDS = 8.0  # track each unit for 8 seconds after spawn


@dataclass
class GameTrackingConfig:
    """Configuration for full-game tracking."""
    frame_dir: Path  # directory with frame_NNNN.jpg files
    battle_id: str
    fps: float = DEFAULT_FPS
    window_seconds: float = DEFAULT_WINDOW_SECONDS
    samv2_url: str = "http://localhost:8079"
    confidence_threshold: float = 0.1
    # Container path mapping: frame_dir on host → this path inside container
    container_replay_base: str = "/app/replays"
    host_replay_base: str = "replays"
    # Working directory for SAMv2-compatible frame windows
    window_dir: Optional[Path] = None


def tick_to_frame(tick: int, fps: float = DEFAULT_FPS) -> int:
    """Convert a replay tick to a frame number."""
    return round(tick / TICKS_PER_SECOND * fps)


def make_spawn_bbox(arena_x: int, arena_y: int, card_name: str) -> tuple:
    """Create a normalized bbox from arena coordinates and card properties."""
    screen_x, screen_y = arena_to_screen(arena_x, arena_y)
    props = get_properties(card_name)
    bbox_w, bbox_h = props.bbox_size

    x1 = max(0.0, screen_x - bbox_w / 2)
    y1 = max(0.0, screen_y - bbox_h / 2)
    x2 = min(1.0, screen_x + bbox_w / 2)
    y2 = min(1.0, screen_y + bbox_h / 2)

    return (round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4))


def prepare_window(
    source_dir: Path,
    window_dir: Path,
    start_frame: int,
    end_frame: int,
) -> int:
    """Create hardlinked SAMv2-compatible frame window. Returns frame count."""
    window_dir.mkdir(parents=True, exist_ok=True)
    for f in window_dir.glob("*.jpg"):
        f.unlink()

    idx = 0
    for frame_num in range(start_frame, end_frame + 1):
        src = source_dir / f"frame_{frame_num:04d}.jpg"
        if not src.exists():
            continue
        dst = window_dir / f"{idx:05d}.jpg"
        os.link(src, dst)
        idx += 1
    return idx


def get_troop_events(session: Session, battle_id: str) -> list[ReplayEvent]:
    """Get all troop-type replay events for a battle."""
    events = list(session.execute(
        select(ReplayEvent)
        .where(ReplayEvent.battle_id == battle_id)
        .order_by(ReplayEvent.game_tick)
    ).scalars())

    troop_events = []
    for event in events:
        card_name = kebab_to_title(event.card_name)
        props = get_properties(card_name)
        if props.card_type == "troop" and card_name not in SKIP_TRACKING:
            troop_events.append(event)

    return troop_events


def track_full_game(
    session: Session,
    config: GameTrackingConfig,
) -> list[TrackingResult]:
    """Track all troops in a game using SAMv2.

    Each troop is tracked in its own short window for quality and speed.
    Results are merged with frame numbers in the original video coordinate space.

    Args:
        session: SQLAlchemy session for loading replay events
        config: tracking configuration

    Returns:
        List of all tracking results across all units
    """
    troop_events = get_troop_events(session, config.battle_id)
    if not troop_events:
        logger.warning("No troop events found for battle %s", config.battle_id)
        return []

    window_dir = config.window_dir or config.frame_dir.parent / "_samv2_tracking"
    container_window = config.container_replay_base + "/" + window_dir.name

    logger.info(
        "Tracking %d troop spawns for battle %s",
        len(troop_events), config.battle_id,
    )

    all_results = []
    total_time = 0.0

    for i, event in enumerate(troop_events):
        card_name = kebab_to_title(event.card_name)
        team = "friendly" if event.side == "team" else "opponent"
        spawn_frame = tick_to_frame(event.game_tick, config.fps)

        # Window: a few frames before spawn to end of tracking period
        window_start = max(1, spawn_frame - 3)
        window_end = spawn_frame + int(config.window_seconds * config.fps)

        bbox = make_spawn_bbox(event.arena_x, event.arena_y, card_name)
        prompt = SpawnPrompt(
            object_id=1,  # single object per window
            card_name=card_name,
            team=team,
            spawn_frame=spawn_frame,
            bbox=bbox,
        )

        n_frames = prepare_window(config.frame_dir, window_dir, window_start, window_end)
        if n_frames == 0:
            logger.warning(
                "No frames for %s spawn at frame %d, skipping",
                card_name, spawn_frame,
            )
            continue

        logger.info(
            "[%d/%d] Tracking %s (%s) tick=%d frame=%d window=%d-%d (%d frames)",
            i + 1, len(troop_events), card_name, team,
            event.game_tick, spawn_frame, window_start, window_end, n_frames,
        )

        t0 = time.time()
        try:
            results = track_units(
                frame_dir=Path(container_window),
                prompts=[prompt],
                window_start_frame=window_start,
                samv2_url=config.samv2_url,
                confidence_threshold=config.confidence_threshold,
                timeout=120,
            )
        except Exception as e:
            logger.error("Tracking failed for %s: %s", card_name, e)
            continue

        elapsed = time.time() - t0
        total_time += elapsed

        # Tag results with the event tick for downstream correlation
        for r in results:
            r.object_id = event.game_tick  # reuse object_id as event identifier

        all_results.extend(results)
        logger.info(
            "  → %d points in %.1fs (%.2fs/frame)",
            len(results), elapsed, elapsed / max(n_frames, 1),
        )

    logger.info(
        "Tracking complete: %d total points from %d units in %.0fs",
        len(all_results), len(troop_events), total_time,
    )

    # Cleanup window dir
    for f in window_dir.glob("*.jpg"):
        f.unlink()

    return all_results


def save_tracking_results(
    results: list[TrackingResult],
    output_path: Path,
) -> None:
    """Save tracking results to JSON."""
    data = [
        {
            "object_id": r.object_id,
            "card_name": r.card_name,
            "team": r.team,
            "frame_number": r.frame_number,
            "bbox": list(r.bbox),
            "confidence": r.confidence,
        }
        for r in results
    ]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved %d tracking results to %s", len(data), output_path)
