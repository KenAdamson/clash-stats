"""Video ingestion pipeline for visual game state recognition (ADR-009).

Drop a video, name the opponent, get labels. Full pipeline:
  1. Match the video to a battle by opponent name
  2. Extract frames at 10fps using VAAPI hardware acceleration
  3. Detect gameplay start (skip match screen / loading)
  4. Generate replay-guided labels for all gameplay frames
  5. Link the video path to the battle record

Usage:
    from tracker.vision.ingest import ingest_video
    ingest_video(session, video_path="/mnt/media/clash-videos/SK-game.mp4",
                 opponent_hint="SK")
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from PIL import Image
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from tracker.models import Battle, ReplayEvent
from tracker.vision.replay_guided_labels import generate_batch_labels, TICKS_PER_SECOND

logger = logging.getLogger(__name__)

# Frame extraction settings
DEFAULT_FPS = 10.0
VAAPI_DEVICE = "/dev/dri/renderD128"

# Gameplay detection: use the VS screen → gameplay transition.
#
# The VS/match screen has distinctive bright red/pink banners across the top
# (RGB R>150, G<50). When gameplay starts, this region becomes the dark
# opponent info bar (RGB R~100, G~30, B~36). This signal is:
#   - Arena-skin independent (UI chrome, not arena texture)
#   - Replay-control independent (top bar is never occluded by replay controls)
#   - Consistent across all trophy ranges and game modes
#
# Secondary signal: player's princess tower body is light blue (R>170, G>200, B>230)
# during gameplay but completely different on the match screen.
VS_BANNER_SAMPLE_POINTS = [
    (0.85, 0.025),  # top-right opponent banner area
    (0.50, 0.025),  # top-center banner area
    (0.15, 0.025),  # top-left banner area
]

TOWER_BODY_SAMPLE_POINTS = [
    (0.22, 0.72),  # player left princess tower area
    (0.78, 0.72),  # player right princess tower area
]

# Player king tower: the light-blue tower body at bottom-center of the arena.
# Present during gameplay, absent on VS screens / loading / phone UI.
# RGB ~(120-190, 170-225, 200-255) — consistently light blue across arenas.
# Sampled at the tower body itself (narrow column at x≈0.50, y≈0.85-0.86).
PLAYER_KING_TOWER_POINTS = [
    (0.48, 0.85),   # king tower body left
    (0.50, 0.855),  # king tower body center
    (0.52, 0.85),   # king tower body right
]


def find_battle_by_opponent(
    session: Session,
    opponent_hint: str,
    corpus: str = "personal",
    limit: int = 5,
) -> list[Battle]:
    """Find recent battles matching an opponent name hint.

    Args:
        session: SQLAlchemy session.
        opponent_hint: partial opponent name (case-insensitive).
        corpus: filter to this corpus (default "personal").
        limit: max results to return.

    Returns:
        List of matching battles, most recent first.
    """
    pattern = f"%{opponent_hint}%"
    battles = list(session.execute(
        select(Battle)
        .where(Battle.corpus == corpus)
        .where(Battle.opponent_name.ilike(pattern))
        .order_by(Battle.battle_time.desc())
        .limit(limit)
    ).scalars())
    return battles


def has_replay_data(session: Session, battle_id: str) -> bool:
    """Check if a battle has replay events."""
    count = session.execute(
        select(ReplayEvent.id)
        .where(ReplayEvent.battle_id == battle_id)
        .limit(1)
    ).scalar_one_or_none()
    return count is not None


def detect_gameplay_start(frame_dir: Path, sample_step: int = 10) -> int:
    """Find the first frame where gameplay is visible (not match screen).

    Uses pixel brightness sampling from the arena floor region. The arena
    has a bright tan/sandy ground that's easily distinguishable from the
    dark match screen overlay.

    Args:
        frame_dir: directory containing frame_NNNN.jpg files.
        sample_step: initial step size for coarse scan (refined with binary search).

    Returns:
        Frame number (1-indexed) of the first gameplay frame.
    """
    frames = sorted(frame_dir.glob("frame_*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No frames in {frame_dir}")

    def frame_number(path: Path) -> int:
        return int(path.stem.split("_")[1])

    def is_gameplay(frame_path: Path) -> bool:
        """Check if a frame shows gameplay (not VS/match screen or phone UI).

        Two-signal detection:
        1. Reject VS screen: bright red/pink banners (R>150, G<50) at top
        2. Require player info bar: light blue bar (G>140, B>180) at y≈0.86

        The player info bar is the most reliable positive signal — it's
        the name/clan bar just below the arena. Present during gameplay,
        absent on VS screens, loading screens, and phone UI overlays.
        """
        img = Image.open(frame_path)
        width, height = img.size

        # Signal 1: reject if VS banner is present (bright red/pink at top)
        vs_banner_hits = 0
        for nx, ny in VS_BANNER_SAMPLE_POINTS:
            px, py = int(nx * width), int(ny * height)
            r, g, b = img.getpixel((px, py))[:3]
            if r > 150 and g < 50:
                vs_banner_hits += 1

        if vs_banner_hits >= 2:
            return False  # still on VS screen

        # Signal 2: require player king tower body (light blue at bottom-center)
        # Gameplay: RGB ~(120-190, 170-225, 200-255)
        tower_hits = 0
        for nx, ny in PLAYER_KING_TOWER_POINTS:
            px, py = int(nx * width), int(ny * height)
            r, g, b = img.getpixel((px, py))[:3]
            if g > 140 and b > 180:
                tower_hits += 1

        return tower_hits >= 2

    # Coarse scan: find approximate transition
    first_gameplay_idx = len(frames) - 1  # default to last frame
    for i in range(0, len(frames), sample_step):
        if is_gameplay(frames[i]):
            first_gameplay_idx = i
            break

    # Binary search: refine to exact transition frame
    lo = max(0, first_gameplay_idx - sample_step)
    hi = first_gameplay_idx

    while lo < hi:
        mid = (lo + hi) // 2
        if is_gameplay(frames[mid]):
            hi = mid
        else:
            lo = mid + 1

    gameplay_frame = frame_number(frames[lo])
    logger.info(
        "Gameplay starts at frame %d (%.1fs into video at 10fps)",
        gameplay_frame, gameplay_frame / DEFAULT_FPS,
    )
    return gameplay_frame


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: float = DEFAULT_FPS,
    use_vaapi: bool = True,
) -> int:
    """Extract frames from a video using ffmpeg with optional VAAPI acceleration.

    Args:
        video_path: path to the video file.
        output_dir: directory to write frame_NNNN.jpg files.
        fps: output frame rate.
        use_vaapi: use Arc A770 hardware decode + encode.

    Returns:
        Number of frames extracted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = str(output_dir / "frame_%04d.jpg")

    if use_vaapi:
        cmd = [
            "ffmpeg", "-y",
            "-hwaccel", "vaapi",
            "-hwaccel_device", VAAPI_DEVICE,
            "-hwaccel_output_format", "vaapi",
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-c:v", "mjpeg_vaapi",
            output_pattern,
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            output_pattern,
        ]

    logger.info("Extracting frames: %s", " ".join(cmd))
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
    )

    if result.returncode != 0:
        # If VAAPI failed, retry without it
        if use_vaapi:
            logger.warning("VAAPI extraction failed, retrying with CPU: %s", result.stderr[-200:])
            return extract_frames(video_path, output_dir, fps, use_vaapi=False)
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")

    frame_count = len(list(output_dir.glob("frame_*.jpg")))
    logger.info("Extracted %d frames to %s", frame_count, output_dir)
    return frame_count


def ingest_video(
    session: Session,
    video_path: str,
    opponent_hint: str,
    frames_dir: Optional[str] = None,
    fps: float = DEFAULT_FPS,
    corpus: str = "personal",
    auto_link: bool = True,
) -> dict:
    """Full video ingestion pipeline.

    Args:
        session: SQLAlchemy session.
        video_path: path to the video file.
        opponent_hint: partial opponent name to match the battle.
        frames_dir: where to extract frames. Defaults to
            ~/clash-stats/replays/{video_stem}/
        fps: frame extraction rate.
        corpus: which corpus to search for the battle.
        auto_link: if True and exactly one battle matches, link automatically.

    Returns:
        Dict with pipeline results:
        {
            "battle_id": str,
            "opponent_name": str,
            "frames_extracted": int,
            "gameplay_start_frame": int,
            "video_start_offset": float,
            "labels_generated": int,
            "has_replay_data": bool,
        }
    """
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Step 1: Find the battle
    matches = find_battle_by_opponent(session, opponent_hint, corpus)
    if not matches:
        raise ValueError(
            f"No battles found matching opponent '{opponent_hint}'. "
            f"Check the name or try a different hint."
        )

    if len(matches) == 1:
        battle = matches[0]
        logger.info("Matched: %s vs %s (%s) — %s",
                     battle.player_name, battle.opponent_name,
                     battle.battle_time, battle.result)
    else:
        # Multiple matches — pick the most recent
        logger.info("Found %d matches for '%s':", len(matches), opponent_hint)
        for i, b in enumerate(matches):
            replay_tag = " [has replay]" if has_replay_data(session, b.battle_id) else ""
            logger.info("  [%d] %s vs %s — %s %s%s",
                        i, b.player_name, b.opponent_name,
                        b.battle_time, b.result, replay_tag)
        if auto_link:
            battle = matches[0]
            logger.info("Auto-selecting most recent: %s", battle.opponent_name)
        else:
            raise ValueError(
                f"Multiple matches for '{opponent_hint}'. "
                f"Be more specific or set auto_link=True."
            )

    # Load replay events (needed for offset calibration and label generation)
    from tracker.vision.replay_guided_labels import load_battle_context
    _, events, _ = load_battle_context(session, battle.battle_id)
    has_replays = len(events) > 0
    if not has_replays:
        logger.warning(
            "Battle %s has no replay events. Labels will be limited to "
            "deck composition only (no position/timing data).",
            battle.battle_id,
        )

    # Step 2: Extract frames
    if frames_dir:
        frame_dir = Path(frames_dir)
    else:
        frame_dir = Path.home() / "clash-stats" / "replays" / video.stem

    # Skip extraction if frames already exist
    existing = list(frame_dir.glob("frame_*.jpg"))
    if existing:
        frame_count = len(existing)
        logger.info("Using %d existing frames in %s", frame_count, frame_dir)
    else:
        frame_count = extract_frames(video, frame_dir, fps)

    # Step 3: Detect gameplay start and calibrate offset
    #
    # The video_start_offset maps frame numbers to game time.
    # gameplay_start_frame is the first frame showing the arena.
    # The game clock at that frame is NOT necessarily 3:00 — the
    # replay may have started mid-game or there's a loading delay.
    #
    # We calibrate using the first replay event as an anchor:
    #   first_event_tick / 20 = game_time of first card play
    #   We assume the first card play is visible within a few
    #   seconds of gameplay_start, so:
    #   video_start_offset ≈ gameplay_start_frame / fps
    #
    # For precise calibration, the clock at gameplay_start typically
    # reads ~2:52 to 2:55 (5-8 seconds of game time elapsed during
    # the VS screen + FIGHT animation). We add this correction.
    gameplay_start = detect_gameplay_start(frame_dir)

    # Estimate game time at gameplay_start using first replay event
    # The first event gives us a known (tick, game_time) anchor.
    # Game starts at tick 0 = game_time 0s = clock showing 3:00.
    # The VS screen + FIGHT animation typically takes 5-10s of real time
    # but 5-8s of game time pass during this period.
    if events:
        first_tick = events[0].game_tick
        first_game_time = first_tick / TICKS_PER_SECOND
        # Assume first card play happens a few seconds into visible gameplay
        # This is approximate — OCR calibration can refine it later
        estimated_game_time_at_start = max(0.0, first_game_time - 10.0)
    else:
        # No replay events — assume standard ~8s delay
        estimated_game_time_at_start = 5.0

    video_start_offset = (gameplay_start / fps) - estimated_game_time_at_start

    # Step 4: Link video to battle
    session.execute(
        update(Battle)
        .where(Battle.battle_id == battle.battle_id)
        .values(video_path=str(video_path))
    )
    session.commit()
    logger.info("Linked video to battle %s", battle.battle_id)

    # Step 5: Generate replay-guided labels (if replay data exists)
    labels_count = 0
    if has_replays:
        label_dir = frame_dir / "labels"
        labels = generate_batch_labels(
            session,
            battle.battle_id,
            frame_dir,
            fps=fps,
            video_start_offset=video_start_offset,
            output_dir=label_dir,
        )
        labels_count = len(labels)

    return {
        "battle_id": battle.battle_id,
        "opponent_name": battle.opponent_name,
        "result": battle.result,
        "frames_extracted": frame_count,
        "gameplay_start_frame": gameplay_start,
        "video_start_offset": round(video_start_offset, 2),
        "labels_generated": labels_count,
        "has_replay_data": has_replays,
        "frame_dir": str(frame_dir),
    }
