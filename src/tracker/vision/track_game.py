"""Full-game SAMv2 tracking pipeline.

Processes all troop events in a battle replay:
  1. Load replay events from the database
  2. For each troop spawn, create a tracking window
  3. Send to SAMv2 sidecar for frame-to-frame tracking
  4. Merge results into per-frame labels

Spells and buildings are handled by replay_guided_labels.py (fixed positions).
Only troops need SAMv2 tracking because they move.

Supports concurrent tracking via the SAMv2 predictor pool.
"""

import json
import logging
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from tracker.ml.card_metadata import kebab_to_title
from tracker.models import Battle, ReplayEvent
from tracker.vision.card_properties import get_properties
from tracker.vision.replay_guided_labels import arena_to_screen, estimate_unit_position
from tracker.vision.samv2_client import SpawnPrompt, TrackingResult, track_units

logger = logging.getLogger(__name__)

# Units that are invisible or hard to track at spawn time
SKIP_TRACKING = {
    "Miner",  # burrows underground, appears ~1s later at target position
}

# Units too small/fast for reliable SAMv2 tracking
UNRELIABLE_TRACKING = {
    "Bats",  # 5 tiny fast-moving units, SAMv2 locks on static features
}

TICKS_PER_SECOND = 20
DEFAULT_FPS = 10.0
DEFAULT_WINDOW_SECONDS = 8.0  # track each unit for 8 seconds after spawn
DEFAULT_CONCURRENCY = 1  # XPU can't parallelize — concurrent sessions cause 3x slowdown

# Tower screen positions (normalized) for overlap detection
# SAMv2 grabs towers when spawn bbox overlaps them
_TOWER_POSITIONS = [
    (0.5000, 0.1301, 0.08),  # opponent king tower, radius
    (0.2120, 0.1875, 0.06),  # opponent princess L
    (0.7880, 0.1875, 0.06),  # opponent princess R
    (0.5000, 0.6705, 0.08),  # friendly king tower
    (0.2120, 0.6216, 0.06),  # friendly princess L
    (0.7880, 0.6216, 0.06),  # friendly princess R
]


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
    # Working directory base for SAMv2-compatible frame windows
    window_dir: Optional[Path] = None
    # Downscale factor for frames (0.5 = half resolution, ~24% faster)
    downscale: float = 1.0
    # Number of concurrent tracking requests
    concurrency: int = DEFAULT_CONCURRENCY


@dataclass
class _TrackingTask:
    """Internal: a planned tracking task ready for execution."""
    index: int  # position in event list (for logging)
    total: int  # total events (for logging)
    card_name: str
    team: str
    game_tick: int
    prompt: SpawnPrompt
    window_start: int
    window_end: int


def tick_to_frame(tick: int, fps: float = DEFAULT_FPS) -> int:
    """Convert a replay tick to a frame number."""
    return round(tick / TICKS_PER_SECOND * fps)


def _overlaps_tower(screen_x: float, screen_y: float) -> bool:
    """Check if a screen position overlaps any tower."""
    for tx, ty, radius in _TOWER_POSITIONS:
        if abs(screen_x - tx) < radius and abs(screen_y - ty) < radius:
            return True
    return False


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
    downscale: float = 1.0,
) -> int:
    """Create SAMv2-compatible frame window. Returns frame count.

    Args:
        source_dir: directory with frame_NNNN.jpg files.
        window_dir: output directory for sequential NNNNN.jpg files.
        start_frame: first frame number (inclusive).
        end_frame: last frame number (inclusive).
        downscale: resize factor (0.5 = half resolution). 1.0 = hardlink original.
    """
    window_dir.mkdir(parents=True, exist_ok=True)
    for f in window_dir.glob("*.jpg"):
        f.unlink()

    idx = 0
    for frame_num in range(start_frame, end_frame + 1):
        src = source_dir / f"frame_{frame_num:04d}.jpg"
        if not src.exists():
            continue
        dst = window_dir / f"{idx:05d}.jpg"
        if downscale == 1.0:
            os.link(src, dst)
        else:
            from PIL import Image
            img = Image.open(src)
            w, h = img.size
            img = img.resize((int(w * downscale), int(h * downscale)), Image.LANCZOS)
            img.save(dst, quality=85)
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


def _plan_tasks(
    troop_events: list[ReplayEvent],
    config: GameTrackingConfig,
) -> list[_TrackingTask]:
    """Build the list of tracking tasks, applying filters and tower delay."""
    tasks = []
    total = len(troop_events)

    for i, event in enumerate(troop_events):
        card_name = kebab_to_title(event.card_name)
        team = "friendly" if event.side == "team" else "opponent"

        if card_name in UNRELIABLE_TRACKING:
            logger.info(
                "[%d/%d] Skipping %s (%s) — unreliable tracking",
                i + 1, total, card_name, team,
            )
            continue

        spawn_frame = tick_to_frame(event.game_tick, config.fps)
        prompt_frame = spawn_frame

        # Check if spawn position overlaps a tower
        screen_x, screen_y = arena_to_screen(event.arena_x, event.arena_y)
        props = get_properties(card_name)
        if _overlaps_tower(screen_x, screen_y) and props.walk_speed > 0:
            cleared = False
            for step in range(1, 11):
                delay_sec = step * 0.5
                est_x, est_y, _ = estimate_unit_position(
                    event.arena_x, event.arena_y,
                    event.side, props, delay_sec,
                )
                sx, sy = arena_to_screen(int(est_x), int(est_y))
                if not _overlaps_tower(sx, sy):
                    cleared = True
                    break

            if not cleared:
                logger.warning(
                    "[%d/%d] Skipping %s (%s) — spawn path stays over tower",
                    i + 1, total, card_name, team,
                )
                continue

            delay_frames = int(delay_sec * config.fps)
            prompt_frame = spawn_frame + delay_frames
            bbox = make_spawn_bbox(int(est_x), int(est_y), card_name)
            logger.info(
                "  Tower overlap: %s delayed %.1fs to frame %d",
                card_name, delay_sec, prompt_frame,
            )
        else:
            bbox = make_spawn_bbox(event.arena_x, event.arena_y, card_name)

        window_start = max(1, prompt_frame - 3)
        window_end = prompt_frame + int(config.window_seconds * config.fps)

        prompt = SpawnPrompt(
            object_id=1,
            card_name=card_name,
            team=team,
            spawn_frame=prompt_frame,
            bbox=bbox,
        )

        tasks.append(_TrackingTask(
            index=i + 1,
            total=total,
            card_name=card_name,
            team=team,
            game_tick=event.game_tick,
            prompt=prompt,
            window_start=window_start,
            window_end=window_end,
        ))

    return tasks


def _execute_task(
    task: _TrackingTask,
    config: GameTrackingConfig,
    slot_queue: queue.Queue,
) -> tuple[_TrackingTask, list[TrackingResult], float]:
    """Execute a single tracking task in a thread.

    Acquires a slot from the queue for its exclusive window directory.
    """
    slot = slot_queue.get()
    try:
        return _execute_task_with_slot(task, config, slot)
    finally:
        slot_queue.put(slot)


def _execute_task_with_slot(
    task: _TrackingTask,
    config: GameTrackingConfig,
    slot: int,
) -> tuple[_TrackingTask, list[TrackingResult], float]:
    """Execute tracking with an assigned slot."""
    # Per-slot window directory
    base_window = config.window_dir or config.frame_dir.parent / "_samv2_tracking"
    window_dir = base_window.parent / f"{base_window.name}_{slot}"
    container_window = config.container_replay_base + "/" + window_dir.name

    n_frames = prepare_window(
        config.frame_dir, window_dir,
        task.window_start, task.window_end,
        downscale=config.downscale,
    )
    if n_frames == 0:
        logger.warning(
            "[%d/%d] No frames for %s at frame %d",
            task.index, task.total, task.card_name, task.window_start,
        )
        return (task, [], 0.0)

    logger.info(
        "[%d/%d] Tracking %s (%s) tick=%d window=%d-%d (%d frames) [slot %d]",
        task.index, task.total, task.card_name, task.team,
        task.game_tick, task.window_start, task.window_end, n_frames, slot,
    )

    t0 = time.time()
    try:
        results = track_units(
            frame_dir=Path(container_window),
            prompts=[task.prompt],
            window_start_frame=task.window_start,
            samv2_url=config.samv2_url,
            confidence_threshold=config.confidence_threshold,
            timeout=180,
        )
    except Exception as e:
        logger.error("Tracking failed for %s: %s", task.card_name, e)
        return (task, [], time.time() - t0)

    elapsed = time.time() - t0

    # Tag results with the event tick for downstream correlation
    for r in results:
        r.object_id = task.game_tick

    logger.info(
        "  → %s: %d points in %.1fs (%.2fs/frame) [slot %d]",
        task.card_name, len(results), elapsed,
        elapsed / max(n_frames, 1), slot,
    )

    # Cleanup window
    for f in window_dir.glob("*.jpg"):
        f.unlink()

    return (task, results, elapsed)


def track_full_game(
    session: Session,
    config: GameTrackingConfig,
) -> list[TrackingResult]:
    """Track all troops in a game using SAMv2.

    Each troop is tracked in its own short window. With concurrency > 1,
    multiple windows are tracked simultaneously using the SAMv2 predictor pool.

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

    # Plan all tasks (filters, tower delay, etc.)
    tasks = _plan_tasks(troop_events, config)
    if not tasks:
        logger.warning("No trackable units for battle %s", config.battle_id)
        return []

    logger.info(
        "Tracking %d units for battle %s (concurrency=%d)",
        len(tasks), config.battle_id, config.concurrency,
    )

    all_results = []
    total_time = 0.0
    t_wall_start = time.time()

    if config.concurrency <= 1:
        # Sequential mode
        for task in tasks:
            _, results, elapsed = _execute_task_with_slot(task, config, slot=0)
            all_results.extend(results)
            total_time += elapsed
    else:
        # Concurrent mode: slot queue ensures exclusive window dirs
        slot_queue: queue.Queue = queue.Queue()
        for s in range(config.concurrency):
            slot_queue.put(s)

        with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
            futures = {}
            for task in tasks:
                future = pool.submit(_execute_task, task, config, slot_queue)
                futures[future] = task

            for future in as_completed(futures):
                task, results, elapsed = future.result()
                all_results.extend(results)
                total_time += elapsed

    wall_time = time.time() - t_wall_start
    logger.info(
        "Tracking complete: %d points from %d units — "
        "%.0fs wall time (%.0fs compute, %.1fx speedup)",
        len(all_results), len(tasks), wall_time, total_time,
        total_time / max(wall_time, 1),
    )

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
