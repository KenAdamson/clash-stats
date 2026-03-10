"""Integration test for SAMv2 tracking with real replay frames.

Tests the full pipeline:
  1. For each unit, prepare a short frame window (~50 frames / 5 seconds)
  2. Track single unit per window for speed and accuracy
  3. Merge results across windows

Requires:
  - SAMv2 sidecar running (docker compose up cr-samv2)
  - SharpJedi replay frames in replays/ScreenRecording_03-06-2026 16-24-20_1/
"""

import json
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tracker.vision.samv2_client import (
    SpawnPrompt,
    track_units,
)
from tracker.vision.replay_guided_labels import arena_to_screen
from tracker.vision.card_properties import get_properties

REPLAY_DIR = Path("replays/ScreenRecording_03-06-2026 16-24-20_1")
WINDOW_DIR = Path("replays/_samv2_test_window")
CONTAINER_WINDOW_DIR = "/app/replays/_samv2_test_window"
FPS = 10.0
TICKS_PER_SECOND = 20


def tick_to_frame(tick: int) -> int:
    """Convert a replay tick to a frame number."""
    return round(tick / TICKS_PER_SECOND * FPS)


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


def prepare_window(start_frame: int, end_frame: int) -> int:
    """Create hardlinked frame window. Returns number of frames linked."""
    WINDOW_DIR.mkdir(parents=True, exist_ok=True)
    for f in WINDOW_DIR.glob("*.jpg"):
        f.unlink()

    idx = 0
    for frame_num in range(start_frame, end_frame + 1):
        src = REPLAY_DIR / f"frame_{frame_num:04d}.jpg"
        if not src.exists():
            continue
        dst = WINDOW_DIR / f"{idx:05d}.jpg"
        os.link(src, dst)
        idx += 1
    return idx


def track_single_unit(
    card_name: str,
    team: str,
    spawn_tick: int,
    arena_x: int,
    arena_y: int,
    window_seconds: float = 5.0,
) -> list:
    """Track a single unit in a short window after its spawn.

    Returns list of tracking results.
    """
    spawn_frame = tick_to_frame(spawn_tick)
    # Start window a few frames before spawn to give SAMv2 context
    window_start = max(1, spawn_frame - 3)
    window_end = spawn_frame + int(window_seconds * FPS)

    bbox = make_spawn_bbox(arena_x, arena_y, card_name)
    prompt = SpawnPrompt(
        object_id=1,
        card_name=card_name,
        team=team,
        spawn_frame=spawn_frame,
        bbox=bbox,
    )

    n_frames = prepare_window(window_start, window_end)
    print(f"\n  Tracking {card_name} ({team}): frames {window_start}-{window_end} ({n_frames} frames)")
    print(f"    Spawn frame: {spawn_frame}, bbox: {bbox}")

    t0 = time.time()
    results = track_units(
        frame_dir=Path(CONTAINER_WINDOW_DIR),
        prompts=[prompt],
        window_start_frame=window_start,
        confidence_threshold=0.1,
        timeout=120,
    )
    elapsed = time.time() - t0

    print(f"    Got {len(results)} tracking points in {elapsed:.1f}s ({elapsed/max(n_frames,1):.2f}s/frame)")

    if results:
        first = results[0]
        last = results[-1]
        print(f"    First: frame {first.frame_number}, bbox=({first.bbox[0]:.3f},{first.bbox[1]:.3f},{first.bbox[2]:.3f},{first.bbox[3]:.3f}), conf={first.confidence:.3f}")
        print(f"    Last:  frame {last.frame_number}, bbox=({last.bbox[0]:.3f},{last.bbox[1]:.3f},{last.bbox[2]:.3f},{last.bbox[3]:.3f}), conf={last.confidence:.3f}")

        # Show movement: first vs last bbox center
        cx1 = (first.bbox[0] + first.bbox[2]) / 2
        cy1 = (first.bbox[1] + first.bbox[3]) / 2
        cx2 = (last.bbox[0] + last.bbox[2]) / 2
        cy2 = (last.bbox[1] + last.bbox[3]) / 2
        dx = cx2 - cx1
        dy = cy2 - cy1
        print(f"    Movement: dx={dx:+.4f}, dy={dy:+.4f}")

    return results


def test_track_units_individually():
    """Track each unit in the SharpJedi game in its own short window.

    SharpJedi troop events (first half):
      tick=132 (6.6s)   opponent baby-dragon   x=1500 y=14500
      tick=202 (10.1s)  team     executioner    x=499  y=26499
      tick=307 (15.3s)  team     bats           x=2500 y=18500
      tick=521 (26.1s)  opponent ice-wizard     x=1500 y=6500
      tick=863 (43.1s)  opponent valkyrie       x=2499 y=4500
      tick=1144 (57.2s) team     pekka          x=8499 y=20499
    """
    print("=== SAMv2 Per-Unit Tracking Test ===")

    all_results = []

    # Baby Dragon — flies from behind opponent king tower toward bridge
    results = track_single_unit("Baby Dragon", "opponent", 132, 1500, 14500, window_seconds=8.0)
    all_results.extend(results)

    # Executioner — walks from player side toward bridge
    results = track_single_unit("Executioner", "friendly", 202, 499, 26499, window_seconds=8.0)
    all_results.extend(results)

    # Bats — fast flyers, tiny, move quickly to bridge
    results = track_single_unit("Bats", "friendly", 307, 2500, 18500, window_seconds=5.0)
    all_results.extend(results)

    # Ice Wizard — opponent deploys near king tower
    results = track_single_unit("Ice Wizard", "opponent", 521, 1500, 6500, window_seconds=8.0)
    all_results.extend(results)

    # Valkyrie — melee, walks toward bridge
    results = track_single_unit("Valkyrie", "opponent", 863, 2499, 4500, window_seconds=8.0)
    all_results.extend(results)

    # PEKKA — big unit, should be easy to track
    results = track_single_unit("P.E.K.K.A", "friendly", 1144, 8499, 20499, window_seconds=8.0)
    all_results.extend(results)

    # Save all results
    output_path = Path("replays/_samv2_test_results.json")
    with open(output_path, "w") as f:
        json.dump(
            [{"object_id": r.object_id, "card_name": r.card_name, "team": r.team,
              "frame_number": r.frame_number, "bbox": list(r.bbox), "confidence": r.confidence}
             for r in all_results],
            f, indent=2,
        )
    print(f"\n=== Total: {len(all_results)} tracking points saved to {output_path} ===")


if __name__ == "__main__":
    test_track_units_individually()
