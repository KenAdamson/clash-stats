"""Convert SAMv2 tracking results into PredictedFrameLabel format.

Merges SAMv2-tracked bounding boxes with replay event metadata to produce
labels in the same format as replay_guided_labels.py, but with actual
tracked positions instead of walk-speed estimates.

Usage:
    from tracker.vision.tracking_labels import tracking_to_labels
    labels = tracking_to_labels(tracking_results, session, battle_id)
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from tracker.ml.card_metadata import kebab_to_title
from tracker.models import Battle, ReplayEvent
from tracker.vision.card_properties import get_properties
from tracker.vision.replay_guided_labels import (
    PredictedFrameLabel,
    PredictedUnit,
    ReplayExclusiveSignals,
    get_game_period,
    load_battle_context,
    TICKS_PER_SECOND,
    FLYING_SPRITE_OFFSET_X,
    FLYING_SPRITE_OFFSET_Y,
)
from tracker.vision.samv2_client import TrackingResult

logger = logging.getLogger(__name__)

DEFAULT_FPS = 10.0


def tracking_to_labels(
    results: list[TrackingResult],
    session: Session,
    battle_id: str,
    fps: float = DEFAULT_FPS,
) -> list[PredictedFrameLabel]:
    """Convert SAMv2 tracking results to PredictedFrameLabel format.

    Each TrackingResult has object_id set to the spawn event's game_tick,
    which links it back to the ReplayEvent for metadata.

    Args:
        results: SAMv2 tracking results from track_full_game().
        session: SQLAlchemy session for loading battle context.
        battle_id: battle ID for metadata lookup.
        fps: video frame rate.

    Returns:
        List of PredictedFrameLabel, one per frame that has tracking data.
    """
    battle, events, deck_info = load_battle_context(session, battle_id)
    if battle is None:
        raise ValueError(f"Battle {battle_id} not found")

    # Index events by game_tick for metadata lookup
    event_by_tick = {e.game_tick: e for e in events}

    # Build deck lists
    player_deck = [
        info["card_name"] for info in deck_info.values()
        if info["team"] == "friendly"
    ]
    opponent_deck = [
        info["card_name"] for info in deck_info.values()
        if info["team"] == "opponent"
    ]

    # Group tracking results by frame
    by_frame: dict[int, list[TrackingResult]] = defaultdict(list)
    for r in results:
        by_frame[r.frame_number].append(r)

    labels = []
    for frame_num in sorted(by_frame.keys()):
        game_time_sec = frame_num / fps
        period = get_game_period(game_time_sec, battle.battle_duration)

        label = PredictedFrameLabel(
            frame_number=frame_num,
            game_time_seconds=round(game_time_sec, 2),
            period=period,
            battle_id=battle_id,
            player_tag=battle.player_tag,
            player_deck=player_deck,
            opponent_deck=opponent_deck,
            replay_signals=ReplayExclusiveSignals(),
        )

        for r in by_frame[frame_num]:
            event = event_by_tick.get(r.object_id)
            card_name = r.card_name
            team = r.team
            props = get_properties(card_name)

            # Compute elapsed time from spawn
            if event:
                elapsed_sec = (frame_num / fps) - (event.game_tick / TICKS_PER_SECOND)
            else:
                elapsed_sec = 0.0

            # Determine action from tracking data
            action = "walking"
            if elapsed_sec < 0.5:
                action = "deployed"
            elif elapsed_sec > props.lifespan_sec * 0.8:
                action = "attacking"  # likely engaged by now

            x1, y1, x2, y2 = r.bbox

            # Sprite bbox for flying units
            sprite_bbox = None
            if props.is_flying:
                sx1 = max(0.0, x1 + FLYING_SPRITE_OFFSET_X)
                sy1 = max(0.0, y1 + FLYING_SPRITE_OFFSET_Y)
                sx2 = min(1.0, x2 + FLYING_SPRITE_OFFSET_X)
                sy2 = min(1.0, y2 + FLYING_SPRITE_OFFSET_Y)
                sprite_bbox = (round(sx1, 4), round(sy1, 4), round(sx2, 4), round(sy2, 4))

            # Lookup deck info
            deck_key = f"{card_name}:{team}"
            card_info = deck_info.get(deck_key, {})

            unit = PredictedUnit(
                card_name=card_name,
                team=team,
                arena_x=0.0,  # not recoverable from screen bbox
                arena_y=0.0,
                screen_bbox=(round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)),
                confidence=r.confidence,
                card_type=props.card_type,
                action=action,
                time_since_play=round(elapsed_sec, 2),
                level=card_info.get("level"),
                is_evo=card_info.get("is_evo", False),
                is_flying=props.is_flying,
                card_elixir=card_info.get("elixir"),
                play_tick=r.object_id,
                notes="samv2_tracked",
                sprite_bbox=sprite_bbox,
            )
            label.units.append(unit)

        labels.append(label)

    logger.info(
        "Converted %d tracking points to %d frame labels",
        len(results), len(labels),
    )
    return labels


def save_tracking_labels(
    labels: list[PredictedFrameLabel],
    output_dir: Path,
) -> None:
    """Save tracking-derived labels as JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for label in labels:
        out_path = output_dir / f"label_{label.frame_number:04d}.json"
        with open(out_path, "w") as f:
            json.dump(label.to_dict(), f, indent=2)
    logger.info("Saved %d label files to %s", len(labels), output_dir)


def load_tracking_results(path: Path) -> list[TrackingResult]:
    """Load tracking results from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return [
        TrackingResult(
            object_id=r["object_id"],
            card_name=r["card_name"],
            team=r["team"],
            frame_number=r["frame_number"],
            bbox=tuple(r["bbox"]),
            confidence=r["confidence"],
        )
        for r in data
    ]
