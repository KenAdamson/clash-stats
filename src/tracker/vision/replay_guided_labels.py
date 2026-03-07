"""Replay-guided label generation for visual game state recognition (ADR-009).

Uses replay event data (card plays with tick timing and arena coordinates)
plus deck composition to generate predicted bounding boxes for each video frame.
These predicted labels bootstrap the training pipeline — Claude Vision or DINOv2
refines them, but the replay data provides strong priors that dramatically reduce
the labeling search space.

For a given frame at game_time T:
  1. Query all replay events for the battle
  2. Determine which units should be alive and where
  3. Map arena coordinates to screen pixel coordinates
  4. Generate predicted bounding boxes with metadata
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from tracker.ml.card_metadata import kebab_to_title, CARD_TYPES
from tracker.models import Battle, DeckCard, ReplayEvent
from tracker.vision.card_properties import get_properties, CardVisualProps

logger = logging.getLogger(__name__)

# Arena coordinate system (from API replay events)
ARENA_X_MIN = 0
ARENA_X_MAX = 17500
ARENA_Y_MIN = 0      # opponent king tower
ARENA_Y_MAX = 31500  # player king tower
ARENA_BRIDGE_Y = 15750  # approximate bridge line

# Screen mapping for 1284x2778 phone capture
# The arena occupies a sub-region of the screen (UI bars at top/bottom)
# These are normalized [0,1] coordinates of the arena bounds within the image
SCREEN_ARENA_LEFT = 0.02
SCREEN_ARENA_RIGHT = 0.98
SCREEN_ARENA_TOP = 0.12     # below opponent info bar
SCREEN_ARENA_BOTTOM = 0.82  # above player info bar

# Ticks per second (from replay event timing)
TICKS_PER_SECOND = 20

# Flying unit sprite offset: visual sprite renders above and slightly behind
# the shadow/hitbox on the ground. Spells target the shadow position.
# Values are in normalized screen coordinates (empirical from replay observation).
FLYING_SPRITE_OFFSET_Y = -0.06  # upward on screen (negative Y = toward top)
FLYING_SPRITE_OFFSET_X = 0.0    # no significant lateral shift


@dataclass
class PredictedUnit:
    """A unit predicted to be visible in a frame based on replay data."""

    card_name: str           # Title Case: "P.E.K.K.A", "Witch", etc.
    team: str                # "friendly" or "opponent"
    arena_x: float           # current estimated arena X
    arena_y: float           # current estimated arena Y
    screen_bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized
    confidence: float        # how confident we are this unit is still here
    card_type: str           # "troop", "spell", "building"
    action: str              # "walking", "attacking", "stationary", "deployed", "fading"
    time_since_play: float   # seconds since this card was played
    level: Optional[int] = None
    is_evo: bool = False
    is_flying: bool = False
    card_elixir: Optional[int] = None
    play_tick: int = 0
    notes: str = ""
    # Flying units: hitbox is at shadow (ground), sprite renders offset upward.
    # screen_bbox is always the hitbox/shadow position (where spells target).
    # sprite_bbox is the visual render position (where the detector sees the unit).
    sprite_bbox: Optional[tuple[float, float, float, float]] = None


@dataclass
class ReplayExclusiveSignals:
    """Signals only visible in replay recordings (not live gameplay).

    Replays show information that is hidden during live play:
    - Opponent's elixir bar (exact count, top-left of screen)
    - Opponent's hand (4 visible cards with hover/selection highlighting)
    - Card selection intent (opponent touching a card before playing it)

    These signals are OCR/detection targets for Claude Vision or a dedicated
    classifier — not derivable from replay event data alone.
    """

    # Opponent elixir: OCR from the purple bar at top-left (0-10)
    opponent_elixir: Optional[int] = None
    opponent_elixir_confidence: float = 0.0

    # Opponent's visible hand (up to 4 card names, left to right)
    opponent_hand: list[str] = field(default_factory=list)

    # Card the opponent is hovering/about to play (highlighted in hand)
    opponent_selected_card: Optional[str] = None


# Screen regions for detection targets (normalized coordinates)
# These are constant across all replays — annotated here for detectors
ELIXIR_BAR_REGION = (0.02, 0.02, 0.18, 0.06)
OPPONENT_HAND_REGION = (0.20, 0.01, 0.80, 0.08)


@dataclass
class PredictedFrameLabel:
    """Complete predicted label for a single video frame."""

    frame_number: int
    game_time_seconds: float
    period: str              # "regular", "double", "triple", "overtime"
    battle_id: str
    player_tag: str
    units: list[PredictedUnit] = field(default_factory=list)

    # Known from battle data
    player_deck: list[str] = field(default_factory=list)
    opponent_deck: list[str] = field(default_factory=list)

    # Replay-exclusive signals (opponent elixir, hand, selection intent)
    replay_signals: Optional[ReplayExclusiveSignals] = None

    def to_dict(self) -> dict:
        """Serialize to dict for JSON export."""
        d = asdict(self)
        return d

    def to_yolo_lines(self, class_map: dict[str, int]) -> list[str]:
        """Convert to YOLO format: class_id x_center y_center width height (normalized).

        Args:
            class_map: mapping of "card_name:team" to integer class ID.

        Returns:
            List of YOLO annotation lines.
        """
        lines = []
        for unit in self.units:
            key = f"{unit.card_name}:{unit.team}"
            if key not in class_map:
                continue
            # For flying units, use sprite_bbox (visual position) for detector training
            bbox = unit.sprite_bbox if unit.sprite_bbox else unit.screen_bbox
            x1, y1, x2, y2 = bbox
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                continue
            lines.append(f"{class_map[key]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        return lines


def arena_to_screen(arena_x: float, arena_y: float) -> tuple[float, float]:
    """Map arena coordinates to normalized screen coordinates.

    Arena Y=0 is opponent side (screen top), Y=31500 is player side (screen bottom).
    The screen has a slight perspective effect but we use linear mapping as baseline.

    Returns:
        (screen_x, screen_y) in [0, 1] normalized coordinates.
    """
    # Normalize arena coords to [0, 1]
    norm_x = (arena_x - ARENA_X_MIN) / (ARENA_X_MAX - ARENA_X_MIN)
    norm_y = (arena_y - ARENA_Y_MIN) / (ARENA_Y_MAX - ARENA_Y_MIN)

    # Map to screen arena region
    screen_x = SCREEN_ARENA_LEFT + norm_x * (SCREEN_ARENA_RIGHT - SCREEN_ARENA_LEFT)
    # Y is inverted: arena Y=0 (opponent) -> screen top, Y=31500 (player) -> screen bottom
    screen_y = SCREEN_ARENA_TOP + norm_y * (SCREEN_ARENA_BOTTOM - SCREEN_ARENA_TOP)

    return (screen_x, screen_y)


def estimate_unit_position(
    play_x: int,
    play_y: int,
    team: str,
    props: CardVisualProps,
    elapsed_sec: float,
) -> tuple[float, float, str]:
    """Estimate a unit's current arena position based on play position and time elapsed.

    Args:
        play_x: arena X where card was played.
        play_y: arena Y where card was played.
        team: "team" or "opponent".
        props: card visual properties.
        elapsed_sec: seconds since the card was played.

    Returns:
        (arena_x, arena_y, action) tuple.
    """
    if props.card_type in ("spell", "building"):
        action = "stationary" if props.card_type == "building" else "active"
        if props.card_type == "spell" and elapsed_sec > props.lifespan_sec:
            action = "fading"
        return (float(play_x), float(play_y), action)

    # Troops walk toward the opponent's side
    walk_speed = props.walk_speed
    if walk_speed == 0:
        return (float(play_x), float(play_y), "stationary")

    # Direction: friendly troops walk toward Y=0 (opponent), opponent troops toward Y=31500
    if team == "team":
        direction = -1  # decrease Y (toward opponent)
        target_y = 0.0
    else:
        direction = 1   # increase Y (toward player)
        target_y = float(ARENA_Y_MAX)

    # Ranged units stop when in range of the nearest tower
    if props.is_ranged:
        stop_distance = props.range_tiles * 500  # tiles to arena units
    else:
        stop_distance = 500  # melee range ~1 tile

    # Simple walk: move Y toward target at walk_speed
    distance_walked = walk_speed * elapsed_sec
    new_y = play_y + direction * distance_walked

    # Clamp to arena bounds and determine action
    action = "walking"
    if team == "team":
        # Friendly: don't walk past opponent king tower area
        min_y = stop_distance
        if new_y <= min_y:
            new_y = min_y
            action = "attacking"
    else:
        max_y = ARENA_Y_MAX - stop_distance
        if new_y >= max_y:
            new_y = max_y
            action = "attacking"

    # After ~3 seconds of walking, if near bridge, likely engaging
    if elapsed_sec > 3.0:
        bridge_dist = abs(new_y - ARENA_BRIDGE_Y)
        if bridge_dist < 3000:
            action = "attacking"

    return (float(play_x), float(new_y), action)


def estimate_unit_alive(
    props: CardVisualProps,
    elapsed_sec: float,
    card_name: str,
) -> tuple[bool, float]:
    """Estimate if a unit is still alive and confidence in the prediction.

    Returns:
        (is_alive, confidence) tuple.
    """
    if elapsed_sec < 0:
        return (False, 0.0)

    # Spells have fixed durations
    if props.card_type == "spell":
        if elapsed_sec > props.lifespan_sec + 1.0:  # 1s grace for fading animation
            return (False, 0.0)
        if elapsed_sec > props.lifespan_sec:
            return (True, 0.3)  # might be fading
        return (True, 0.85)

    # Buildings have fixed lifetime
    if props.card_type == "building":
        if elapsed_sec > props.lifespan_sec:
            return (False, 0.0)
        # Confidence decays as building ages
        remaining_fraction = 1.0 - (elapsed_sec / props.lifespan_sec)
        return (True, max(0.3, remaining_fraction))

    # Troops: confidence decays over time (they get killed)
    if elapsed_sec > props.lifespan_sec:
        return (False, 0.0)

    # Troops near full lifespan are probably dead
    life_fraction = elapsed_sec / props.lifespan_sec
    if life_fraction < 0.3:
        confidence = 0.9  # just deployed, almost certainly alive
    elif life_fraction < 0.6:
        confidence = 0.7  # probably alive
    elif life_fraction < 0.8:
        confidence = 0.4  # maybe alive
    else:
        confidence = 0.2  # probably dead

    # Miner has a known travel time (~1s burrow), adjust
    if card_name == "Miner" and elapsed_sec < 1.0:
        return (True, 0.5)  # underground, not yet visible

    return (True, confidence)


def get_game_period(game_time_sec: float, battle_duration_sec: Optional[int]) -> str:
    """Determine the game period from elapsed time.

    Standard game: 180s regular, 60s double elixir (at 120s remaining = 60s elapsed... no)
    Actually: 3:00 regulation. Double elixir at 1:00 remaining (= 120s elapsed).
    Overtime: 2:00 with double elixir, then triple at 1:00 remaining.
    """
    if battle_duration_sec and game_time_sec > 180:
        # Overtime
        ot_time = game_time_sec - 180
        if ot_time > 60:
            return "triple"
        return "overtime"
    if game_time_sec > 120:
        return "double"
    return "regular"


def load_battle_context(
    session: Session,
    battle_id: str,
) -> tuple[Optional[Battle], list[ReplayEvent], dict[str, dict]]:
    """Load all data needed for replay-guided labeling.

    Returns:
        (battle, events, deck_info) where deck_info maps card_name to
        {level, is_evo, elixir, team} for both player and opponent decks.
    """
    battle = session.execute(
        select(Battle).where(Battle.battle_id == battle_id)
    ).scalar_one_or_none()

    if battle is None:
        return (None, [], {})

    events = list(session.execute(
        select(ReplayEvent)
        .where(ReplayEvent.battle_id == battle_id)
        .order_by(ReplayEvent.game_tick)
    ).scalars())

    # Build deck info from deck_cards
    deck_cards = list(session.execute(
        select(DeckCard).where(DeckCard.battle_id == battle_id)
    ).scalars())

    deck_info: dict[str, dict] = {}
    for dc in deck_cards:
        team = "friendly" if dc.is_player_deck else "opponent"
        deck_info[f"{dc.card_name}:{team}"] = {
            "card_name": dc.card_name,
            "level": dc.card_level,
            "is_evo": dc.evolution_level > 0,
            "elixir": dc.card_elixir,
            "team": team,
            "variant": dc.card_variant,
        }

    return (battle, events, deck_info)


def generate_frame_labels(
    session: Session,
    battle_id: str,
    frame_number: int,
    fps: float = 10.0,
    video_start_offset: float = 0.0,
) -> PredictedFrameLabel:
    """Generate predicted labels for a single video frame.

    Args:
        session: SQLAlchemy session.
        battle_id: battle to label.
        frame_number: 1-indexed frame number.
        fps: video frame rate.
        video_start_offset: seconds into the video where gameplay starts
            (skip intro/loading screen).

    Returns:
        PredictedFrameLabel with predicted unit positions.
    """
    battle, events, deck_info = load_battle_context(session, battle_id)
    if battle is None:
        raise ValueError(f"Battle {battle_id} not found")

    game_time_sec = (frame_number / fps) - video_start_offset
    if game_time_sec < 0:
        game_time_sec = 0.0

    game_tick = int(game_time_sec * TICKS_PER_SECOND)
    period = get_game_period(game_time_sec, battle.battle_duration)

    # Build deck lists
    player_deck = [
        info["card_name"] for key, info in deck_info.items()
        if info["team"] == "friendly"
    ]
    opponent_deck = [
        info["card_name"] for key, info in deck_info.items()
        if info["team"] == "opponent"
    ]

    label = PredictedFrameLabel(
        frame_number=frame_number,
        game_time_seconds=round(game_time_sec, 2),
        period=period,
        battle_id=battle_id,
        player_tag=battle.player_tag,
        player_deck=player_deck,
        opponent_deck=opponent_deck,
        # Replay-exclusive signals placeholder — populated by Claude Vision
        # or a dedicated OCR/classifier during label refinement
        replay_signals=ReplayExclusiveSignals(),
    )

    # Process each replay event to predict visible units
    for event in events:
        if event.game_tick > game_tick:
            break  # future event, hasn't happened yet

        elapsed_sec = (game_tick - event.game_tick) / TICKS_PER_SECOND
        card_name = kebab_to_title(event.card_name)
        props = get_properties(card_name)
        team_label = "friendly" if event.side == "team" else "opponent"

        # Check if unit is still alive
        is_alive, confidence = estimate_unit_alive(props, elapsed_sec, card_name)
        if not is_alive:
            continue

        # Estimate current position
        arena_x, arena_y, action = estimate_unit_position(
            event.arena_x, event.arena_y,
            event.side, props, elapsed_sec,
        )

        # Map to screen coordinates
        screen_x, screen_y = arena_to_screen(arena_x, arena_y)
        bbox_w, bbox_h = props.bbox_size

        # Build screen bbox (clamped to [0, 1])
        x1 = max(0.0, screen_x - bbox_w / 2)
        y1 = max(0.0, screen_y - bbox_h / 2)
        x2 = min(1.0, screen_x + bbox_w / 2)
        y2 = min(1.0, screen_y + bbox_h / 2)

        # Lookup deck info for level/evo
        deck_key = f"{card_name}:{team_label}"
        card_info = deck_info.get(deck_key, {})

        # Compute sprite_bbox for flying units (visual render offset from shadow)
        sprite_bbox = None
        if props.is_flying:
            sx1 = max(0.0, x1 + FLYING_SPRITE_OFFSET_X)
            sy1 = max(0.0, y1 + FLYING_SPRITE_OFFSET_Y)
            sx2 = min(1.0, x2 + FLYING_SPRITE_OFFSET_X)
            sy2 = min(1.0, y2 + FLYING_SPRITE_OFFSET_Y)
            sprite_bbox = (round(sx1, 4), round(sy1, 4), round(sx2, 4), round(sy2, 4))

        unit = PredictedUnit(
            card_name=card_name,
            team=team_label,
            arena_x=arena_x,
            arena_y=arena_y,
            screen_bbox=(round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)),
            confidence=round(confidence, 2),
            card_type=props.card_type,
            action=action,
            time_since_play=round(elapsed_sec, 2),
            level=card_info.get("level"),
            is_evo=card_info.get("is_evo", False),
            is_flying=props.is_flying,
            card_elixir=card_info.get("elixir"),
            play_tick=event.game_tick,
            sprite_bbox=sprite_bbox,
        )

        label.units.append(unit)

    return label


def generate_batch_labels(
    session: Session,
    battle_id: str,
    frame_dir: Path,
    fps: float = 10.0,
    video_start_offset: float = 0.0,
    output_dir: Optional[Path] = None,
) -> list[PredictedFrameLabel]:
    """Generate predicted labels for all frames in a directory.

    Args:
        session: SQLAlchemy session.
        battle_id: battle to label.
        frame_dir: directory containing extracted frames (frame_NNNN.jpg).
        fps: video frame rate.
        video_start_offset: seconds offset to first gameplay frame.
        output_dir: if provided, write JSON label files here.

    Returns:
        List of PredictedFrameLabel for all frames.
    """
    frames = sorted(frame_dir.glob("frame_*.jpg"))
    if not frames:
        logger.warning("No frames found in %s", frame_dir)
        return []

    logger.info(
        "Generating replay-guided labels for %d frames (battle=%s)",
        len(frames), battle_id,
    )

    labels = []
    for frame_path in frames:
        # Extract frame number from filename: frame_0001.jpg -> 1
        frame_num = int(frame_path.stem.split("_")[1])
        label = generate_frame_labels(
            session, battle_id, frame_num, fps, video_start_offset,
        )
        labels.append(label)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        for label in labels:
            out_path = output_dir / f"label_{label.frame_number:04d}.json"
            with open(out_path, "w") as f:
                json.dump(label.to_dict(), f, indent=2)
        logger.info("Wrote %d label files to %s", len(labels), output_dir)

    return labels


def build_class_map(session: Session, battle_id: str) -> dict[str, int]:
    """Build a YOLO class map from the decks in a battle.

    Returns:
        Dict mapping "CardName:team" to integer class ID.
    """
    _, _, deck_info = load_battle_context(session, battle_id)
    class_map: dict[str, int] = {}
    idx = 0
    for key in sorted(deck_info.keys()):
        info = deck_info[key]
        class_map[f"{info['card_name']}:{info['team']}"] = idx
        idx += 1
    return class_map
