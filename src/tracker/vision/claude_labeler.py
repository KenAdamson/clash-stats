"""Claude Vision label refinement for ADR-009 Phase 1.

Sends gameplay frames + replay-guided predicted labels to Claude Vision API.
Claude refines bounding boxes, confirms/rejects predictions, adds missing
detections (spawned sub-units, spell effects), and reads replay-exclusive
signals (opponent elixir, hand composition, card selection intent).

Runs as a standalone batch job — no Claude Code required. Just needs
ANTHROPIC_API_KEY in environment.

Usage:
    from tracker.vision.claude_labeler import refine_labels_batch
    results = refine_labels_batch(frame_dir, label_dir, output_dir, battle_id)

    # Or from CLI:
    python -m tracker.vision.claude_labeler --frame-dir /path/to/frames \
        --label-dir /path/to/labels --output-dir /path/to/refined \
        --battle-id abc123 --sample 50
"""

import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

# Model for vision labeling — Haiku is fast and cheap for structured extraction
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
# Sonnet for higher quality when needed
QUALITY_MODEL = "claude-sonnet-4-6"

MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds

# Cost tracking
COST_PER_INPUT_MTOK = {"claude-haiku-4-5-20251001": 0.80, "claude-sonnet-4-6": 3.00}
COST_PER_OUTPUT_MTOK = {"claude-haiku-4-5-20251001": 4.00, "claude-sonnet-4-6": 15.00}


SYSTEM_PROMPT = """\
You are a Clash Royale visual analysis system. You receive gameplay frames from \
replay recordings along with predicted unit positions derived from replay event data.

Your job:
1. CONFIRM or REJECT each predicted unit — is it actually visible in the frame?
2. REFINE bounding boxes — adjust coordinates to tightly fit the visible unit.
3. ADD missing units — spawned sub-units (Witch skeletons, Graveyard skeletons, \
Tombstone skeletons, Goblin Drill goblins), spell visual effects, and any troops \
the predictions missed.
4. READ replay-exclusive signals visible at the top of the screen:
   - Opponent's elixir count (purple bar, top-left, 0-10)
   - Opponent's 4 hand cards (card icons at top-center)
   - Whether a card is highlighted/selected (glowing border = about to be played)

Return ONLY valid JSON matching the schema below. No markdown, no explanation."""

LABEL_SCHEMA = """\
{
  "units": [
    {
      "card_name": "P.E.K.K.A",
      "team": "friendly",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95,
      "action": "walking",
      "status": "confirmed",
      "notes": ""
    }
  ],
  "added_units": [
    {
      "card_name": "Skeleton",
      "team": "friendly",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.7,
      "action": "walking",
      "spawned_by": "Witch",
      "notes": "spawned skeleton near bridge"
    }
  ],
  "rejected_predictions": [
    {"card_name": "Miner", "team": "friendly", "reason": "not visible — likely dead"}
  ],
  "replay_signals": {
    "opponent_elixir": 7,
    "opponent_hand": ["Graveyard", "Poison", "Tornado", "Baby Dragon"],
    "opponent_selected_card": "Graveyard"
  },
  "frame_notes": "Double elixir, opponent pushing left lane with Graveyard"
}

Bbox coordinates are normalized [0,1] relative to the full image: [x1, y1, x2, y2].
Team is "friendly" (blue, bottom) or "opponent" (red, top).
Status: "confirmed" (prediction correct), "adjusted" (bbox refined), "rejected" (not visible).
Actions: "walking", "attacking", "deploying", "dying", "idle", "ability".
"""


def encode_frame(frame_path: Path) -> str:
    """Read and base64-encode a frame image."""
    with open(frame_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def build_user_message(
    frame_path: Path,
    predicted_label: dict,
    player_deck: list[str],
    opponent_deck: list[str],
) -> list[dict]:
    """Build the multimodal user message with frame + context."""
    image_data = encode_frame(frame_path)

    # Format predictions as compact text
    pred_text = f"Game time: {predicted_label['game_time_seconds']}s | Period: {predicted_label['period']}\n"
    pred_text += f"Player deck: {', '.join(player_deck)}\n"
    pred_text += f"Opponent deck: {', '.join(opponent_deck)}\n\n"
    pred_text += "Predicted units on field:\n"

    if predicted_label["units"]:
        for u in predicted_label["units"]:
            bbox = u["screen_bbox"]
            pred_text += (
                f"  - {u['team']} {u['card_name']} at bbox "
                f"[{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}] "
                f"conf={u['confidence']:.0%} action={u['action']} +{u['time_since_play']:.1f}s\n"
            )
    else:
        pred_text += "  (none predicted)\n"

    pred_text += f"\nOutput schema:\n{LABEL_SCHEMA}"

    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_data,
            },
        },
        {"type": "text", "text": pred_text},
    ]


@dataclass
class LabelingStats:
    """Track labeling run statistics."""

    frames_processed: int = 0
    frames_failed: int = 0
    units_confirmed: int = 0
    units_rejected: int = 0
    units_added: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    model: str = ""

    def summary(self) -> str:
        return (
            f"Processed {self.frames_processed} frames "
            f"({self.frames_failed} failed)\n"
            f"  Confirmed: {self.units_confirmed} | "
            f"Rejected: {self.units_rejected} | "
            f"Added: {self.units_added}\n"
            f"  Tokens: {self.total_input_tokens:,} in / "
            f"{self.total_output_tokens:,} out\n"
            f"  Cost: ${self.total_cost_usd:.4f} ({self.model})"
        )


def refine_single_frame(
    client: anthropic.Anthropic,
    frame_path: Path,
    label: dict,
    model: str = DEFAULT_MODEL,
) -> Optional[dict]:
    """Send one frame + predicted labels to Claude Vision for refinement.

    Returns:
        Refined label dict, or None on failure.
    """
    user_content = build_user_message(
        frame_path, label,
        label.get("player_deck", []),
        label.get("opponent_deck", []),
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )

            # Parse JSON response
            text = response.content[0].text.strip()
            # Strip markdown code fence if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            refined = json.loads(text)

            # Attach token usage
            refined["_usage"] = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            return refined

        except json.JSONDecodeError as e:
            logger.warning(
                "Frame %s: invalid JSON on attempt %d: %s",
                frame_path.name, attempt + 1, str(e)[:100],
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
        except anthropic.RateLimitError:
            wait = RETRY_DELAY * (2 ** attempt)
            logger.warning("Rate limited, waiting %.1fs", wait)
            time.sleep(wait)
        except anthropic.APIError as e:
            logger.error("API error on %s: %s", frame_path.name, e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    return None


def refine_labels_batch(
    frame_dir: str | Path,
    label_dir: str | Path,
    output_dir: str | Path,
    sample: Optional[int] = None,
    stride: int = 1,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    delay: float = 0.5,
) -> LabelingStats:
    """Refine a batch of frames through Claude Vision.

    Args:
        frame_dir: directory with frame_NNNN.jpg files.
        label_dir: directory with label_NNNN.json files (from replay-guided).
        output_dir: where to write refined label JSON files.
        sample: if set, only process this many frames (evenly spaced).
        stride: process every Nth frame (default: every frame).
        model: Claude model to use.
        api_key: Anthropic API key (default: ANTHROPIC_API_KEY env var).
        delay: seconds between API calls (rate limiting).

    Returns:
        LabelingStats with run summary.
    """
    frame_dir = Path(frame_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. Pass api_key= or set the environment variable."
        )

    client = anthropic.Anthropic(api_key=key)
    stats = LabelingStats(model=model)

    # Collect frame/label pairs
    label_files = sorted(label_dir.glob("label_*.json"))
    if not label_files:
        logger.warning("No label files in %s", label_dir)
        return stats

    # Apply stride
    if stride > 1:
        label_files = label_files[::stride]

    # Apply sample limit (evenly spaced)
    if sample and sample < len(label_files):
        step = len(label_files) / sample
        label_files = [label_files[int(i * step)] for i in range(sample)]

    logger.info(
        "Refining %d frames with %s (from %s)",
        len(label_files), model, label_dir,
    )

    input_cost_rate = COST_PER_INPUT_MTOK.get(model, 3.0)
    output_cost_rate = COST_PER_OUTPUT_MTOK.get(model, 15.0)

    for i, label_path in enumerate(label_files):
        # Match frame file
        frame_num = label_path.stem.replace("label_", "")
        frame_path = frame_dir / f"frame_{frame_num}.jpg"
        if not frame_path.exists():
            logger.warning("No frame for %s", label_path.name)
            stats.frames_failed += 1
            continue

        # Skip if already refined
        out_path = output_dir / f"refined_{frame_num}.json"
        if out_path.exists():
            logger.debug("Skipping %s (already refined)", frame_num)
            continue

        # Load predicted label
        with open(label_path) as f:
            label = json.load(f)

        # Send to Claude
        refined = refine_single_frame(client, frame_path, label, model)

        if refined is None:
            stats.frames_failed += 1
            continue

        # Track stats
        usage = refined.pop("_usage", {})
        stats.frames_processed += 1
        stats.total_input_tokens += usage.get("input_tokens", 0)
        stats.total_output_tokens += usage.get("output_tokens", 0)
        stats.total_cost_usd += (
            usage.get("input_tokens", 0) / 1_000_000 * input_cost_rate
            + usage.get("output_tokens", 0) / 1_000_000 * output_cost_rate
        )

        stats.units_confirmed += len([
            u for u in refined.get("units", [])
            if u.get("status") in ("confirmed", "adjusted")
        ])
        stats.units_rejected += len(refined.get("rejected_predictions", []))
        stats.units_added += len(refined.get("added_units", []))

        # Merge refined data with original label metadata
        refined["frame_number"] = label["frame_number"]
        refined["game_time_seconds"] = label["game_time_seconds"]
        refined["period"] = label["period"]
        refined["battle_id"] = label["battle_id"]
        refined["player_deck"] = label.get("player_deck", [])
        refined["opponent_deck"] = label.get("opponent_deck", [])

        # Write refined label
        with open(out_path, "w") as f:
            json.dump(refined, f, indent=2)

        if (i + 1) % 10 == 0:
            logger.info(
                "Progress: %d/%d frames (cost so far: $%.4f)",
                i + 1, len(label_files), stats.total_cost_usd,
            )

        # Rate limiting
        if delay > 0 and i < len(label_files) - 1:
            time.sleep(delay)

    logger.info("Refinement complete.\n%s", stats.summary())
    return stats


def main():
    """CLI entrypoint for batch label refinement."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Refine replay-guided labels with Claude Vision",
    )
    parser.add_argument("--frame-dir", required=True, help="Directory with frame_NNNN.jpg files")
    parser.add_argument("--label-dir", help="Directory with label_NNNN.json files (default: frame-dir/labels)")
    parser.add_argument("--output-dir", help="Output directory for refined labels (default: frame-dir/refined)")
    parser.add_argument("--sample", type=int, help="Process N evenly-spaced frames (default: all)")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame (default: 1)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        choices=[DEFAULT_MODEL, QUALITY_MODEL],
                        help=f"Claude model (default: {DEFAULT_MODEL})")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between API calls (default: 0.5)")
    parser.add_argument("--battle-id", help="Battle ID (for reference only)")
    args = parser.parse_args()

    frame_dir = Path(args.frame_dir)
    label_dir = Path(args.label_dir) if args.label_dir else frame_dir / "labels"
    output_dir = Path(args.output_dir) if args.output_dir else frame_dir / "refined"

    stats = refine_labels_batch(
        frame_dir=frame_dir,
        label_dir=label_dir,
        output_dir=output_dir,
        sample=args.sample,
        stride=args.stride,
        model=args.model,
        delay=args.delay,
    )
    print(stats.summary())


if __name__ == "__main__":
    main()
