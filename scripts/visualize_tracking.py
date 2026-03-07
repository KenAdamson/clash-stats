"""Overlay SAMv2 tracking bboxes on replay frames for visual verification.

Generates annotated frames and optionally stitches them into a video.

Usage:
    python scripts/visualize_tracking.py [--video] [--compare]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

RESULTS_PATH = Path("replays/_samv2_full_game_results.json")
FRAME_DIR = Path("replays/ScreenRecording_03-06-2026 16-24-20_1")
OUTPUT_DIR = Path("replays/_samv2_overlay")
LABEL_DIR = Path("replays/ScreenRecording_03-06-2026 16-24-20_1/labels_samv2")

# Colors per team (RGB)
TEAM_COLORS = {
    "friendly": (0, 140, 255),   # blue
    "opponent": (255, 60, 60),   # red
}

# Sample every Nth frame to keep output manageable
SAMPLE_INTERVAL = 5


def load_results():
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    # Group by frame number
    by_frame = defaultdict(list)
    for r in data:
        by_frame[r["frame_number"]].append(r)
    return by_frame


def draw_overlays(by_frame, make_video=False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get all frames that have tracking data
    frame_nums = sorted(by_frame.keys())
    sampled = [f for f in frame_nums if f % SAMPLE_INTERVAL == 0]

    if not sampled:
        sampled = frame_nums[:50]  # fallback: first 50

    print(f"Rendering {len(sampled)} overlay frames (every {SAMPLE_INTERVAL}th tracked frame)...")

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
        small_font = font

    rendered = []
    for i, frame_num in enumerate(sampled):
        src = FRAME_DIR / f"frame_{frame_num:04d}.jpg"
        if not src.exists():
            continue

        img = Image.open(src)
        draw = ImageDraw.Draw(img)
        w, h = img.size

        for r in by_frame[frame_num]:
            color = TEAM_COLORS.get(r["team"], (255, 255, 0))
            x1 = int(r["bbox"][0] * w)
            y1 = int(r["bbox"][1] * h)
            x2 = int(r["bbox"][2] * w)
            y2 = int(r["bbox"][3] * h)

            # Draw bbox (thick outline)
            for offset in range(6):
                draw.rectangle(
                    [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                    outline=color,
                )

            # Label with dark background
            label = f"{r['card_name']} ({r['confidence']:.2f})"
            text_bbox = draw.textbbox((x1, y1 - 40), label, font=font)
            draw.rectangle(
                [text_bbox[0] - 4, text_bbox[1] - 4, text_bbox[2] + 4, text_bbox[3] + 4],
                fill=(0, 0, 0),
            )
            draw.text((x1, y1 - 40), label, fill=color, font=font)

        # Frame number watermark
        draw.text((10, 10), f"Frame {frame_num}", fill=(255, 255, 255), font=font)

        out_path = OUTPUT_DIR / f"overlay_{frame_num:04d}.jpg"
        img.save(out_path, quality=85)
        rendered.append(out_path)

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(sampled)} frames rendered")

    print(f"Done: {len(rendered)} overlay frames in {OUTPUT_DIR}")

    if make_video and rendered:
        try:
            import subprocess
            video_path = OUTPUT_DIR / "tracking_overlay.mp4"
            # Use ffmpeg with glob pattern
            subprocess.run([
                "ffmpeg", "-y",
                "-framerate", "2",  # slow enough to inspect
                "-pattern_type", "glob",
                "-i", str(OUTPUT_DIR / "overlay_*.jpg"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-vf", "scale=642:1389",  # half res for smaller file
                str(video_path),
            ], check=True, capture_output=True)
            print(f"Video: {video_path}")
        except Exception as e:
            print(f"Video generation failed (ffmpeg): {e}")

    return rendered


def main():
    make_video = "--video" in sys.argv
    by_frame = load_results()
    print(f"Loaded {sum(len(v) for v in by_frame.values())} tracking points across {len(by_frame)} frames")
    draw_overlays(by_frame, make_video=make_video)


if __name__ == "__main__":
    main()
