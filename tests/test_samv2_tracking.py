"""Integration test for SAMv2 tracking with real replay frames.

Tests the full pipeline:
  1. Prepare a frame window from the SharpJedi recording
  2. Create spawn prompts from replay event data
  3. Send to SAMv2 sidecar for tracking
  4. Verify we get reasonable bounding boxes back

Requires:
  - SAMv2 sidecar running (docker compose up cr-samv2)
  - SharpJedi replay frames in replays/ScreenRecording_03-06-2026 16-24-20_1/
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tracker.vision.samv2_client import (
    SpawnPrompt,
    prepare_frame_window,
    track_units,
    cleanup_window,
)
from tracker.vision.replay_guided_labels import arena_to_screen
from tracker.vision.card_properties import get_properties

REPLAY_DIR = Path("replays/ScreenRecording_03-06-2026 16-24-20_1")
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


def test_track_early_game():
    """Track the first few units spawned in the SharpJedi game.

    Events at start of game:
      tick=132 (6.6s)  opponent baby-dragon   x=1500 y=14500
      tick=171 (8.6s)  opponent graveyard      x=2500 y=25500
      tick=202 (10.1s) team     executioner    x=499  y=26499
      tick=228 (11.4s) team     goblin-curse   x=3500 y=25500
      tick=307 (15.3s) team     bats           x=2500 y=18500
    """
    # Larger window: frame 60 to frame 200 (~14 seconds)
    # Covers first 5 spawn events:
    #   tick=132 (6.6s)  opponent baby-dragon   x=1500 y=14500  → frame 66
    #   tick=171 (8.6s)  opponent graveyard      x=2500 y=25500  → frame 86 (spell, skip)
    #   tick=202 (10.1s) team     executioner    x=499  y=26499  → frame 101
    #   tick=228 (11.4s) team     goblin-curse   x=3500 y=25500  → frame 114 (spell, skip)
    #   tick=307 (15.3s) team     bats           x=2500 y=18500  → frame 154
    window_start = 60
    window_end = 200

    prompts = [
        SpawnPrompt(
            object_id=1,
            card_name="Baby Dragon",
            team="opponent",
            spawn_frame=tick_to_frame(132),  # frame 66
            bbox=make_spawn_bbox(1500, 14500, "Baby Dragon"),
        ),
        SpawnPrompt(
            object_id=2,
            card_name="Executioner",
            team="friendly",
            spawn_frame=tick_to_frame(202),  # frame 101
            bbox=make_spawn_bbox(499, 26499, "Executioner"),
        ),
        SpawnPrompt(
            object_id=3,
            card_name="Bats",
            team="friendly",
            spawn_frame=tick_to_frame(307),  # frame 154
            bbox=make_spawn_bbox(2500, 18500, "Bats"),
        ),
    ]

    print(f"\n=== SAMv2 Tracking Test: Early Game Window ===")
    print(f"Frames: {window_start}-{window_end} ({window_end - window_start + 1} frames)")
    print(f"Prompts:")
    for p in prompts:
        print(f"  #{p.object_id} {p.card_name} ({p.team}) @ frame {p.spawn_frame}, bbox={p.bbox}")

    # Prepare frame window
    # The frames need to be accessible from inside the container at /app/replays/...
    # Since we volume-mount ./replays:/app/replays, use that path
    container_source = Path("/app/replays/ScreenRecording_03-06-2026 16-24-20_1")

    # Create window dir under replays so it's visible inside container
    window_dir = Path("replays/_samv2_test_window")
    window_dir.mkdir(parents=True, exist_ok=True)

    # Clean any previous test
    for f in window_dir.glob("*.jpg"):
        f.unlink()

    # Create hardlinks with SAMv2-compatible naming
    # (symlinks don't work because they resolve to host paths inside the container)
    import os
    idx = 0
    for frame_num in range(window_start, window_end + 1):
        src = REPLAY_DIR / f"frame_{frame_num:04d}.jpg"
        if not src.exists():
            print(f"  WARNING: Missing frame {frame_num}")
            continue
        dst = window_dir / f"{idx:05d}.jpg"
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
        idx += 1

    print(f"\nPrepared {idx} frames in {window_dir}")

    # Container sees these at /app/replays/_samv2_test_window
    container_window_dir = "/app/replays/_samv2_test_window"

    # Adjust prompts for container path
    results = track_units(
        frame_dir=Path(container_window_dir),
        prompts=prompts,
        window_start_frame=window_start,
        confidence_threshold=0.1,
        timeout=600,
    )

    print(f"\n=== Results: {len(results)} tracking points ===")

    # Group by object
    by_object = {}
    for r in results:
        by_object.setdefault(r.object_id, []).append(r)

    for obj_id, obj_results in sorted(by_object.items()):
        prompt = next(p for p in prompts if p.object_id == obj_id)
        print(f"\n  #{obj_id} {prompt.card_name} ({prompt.team}):")
        print(f"    Tracked in {len(obj_results)} frames")
        if obj_results:
            first = obj_results[0]
            last = obj_results[-1]
            print(f"    First: frame {first.frame_number}, bbox={first.bbox}, conf={first.confidence:.3f}")
            print(f"    Last:  frame {last.frame_number}, bbox={last.bbox}, conf={last.confidence:.3f}")

            # Show every 10th frame
            for r in obj_results[::10]:
                print(f"    frame {r.frame_number:4d}: bbox=({r.bbox[0]:.3f},{r.bbox[1]:.3f},{r.bbox[2]:.3f},{r.bbox[3]:.3f}) conf={r.confidence:.3f}")

    # Save full results
    output_path = Path("replays/_samv2_test_results.json")
    with open(output_path, "w") as f:
        json.dump(
            [{"object_id": r.object_id, "card_name": r.card_name, "team": r.team,
              "frame_number": r.frame_number, "bbox": list(r.bbox), "confidence": r.confidence}
             for r in results],
            f, indent=2,
        )
    print(f"\nFull results saved to {output_path}")

    # Cleanup
    # Don't clean up yet so we can inspect
    # cleanup_window(window_dir)


if __name__ == "__main__":
    test_track_early_game()
