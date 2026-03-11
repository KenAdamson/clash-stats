"""Analyze refined Claude Vision labels for troop observation statistics.

Shows how many confirmed observations we have per card, per team,
opponent elixir distribution, and hand card frequency.

Usage:
    python scripts/analyze_refined_labels.py [refined_dir]
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_DIR = Path("replays/ScreenRecording_03-06-2026 16-24-20_1/labels_refined")


def analyze(refined_dir: Path):
    files = sorted(refined_dir.glob("refined_*.json"))
    if not files:
        print(f"No refined labels in {refined_dir}")
        return

    # Counters
    unit_obs = Counter()  # (card_name, team) -> count of confirmed observations
    unit_status = Counter()  # status -> count
    added_units = Counter()  # card_name -> count
    added_spawners = Counter()  # spawned_by -> count
    rejected = Counter()  # card_name -> count
    elixir_readings = []
    hand_cards = Counter()  # card_name -> times seen in opponent hand
    selected_cards = Counter()  # card_name -> times selected
    actions = Counter()

    frames_processed = len(files)
    frames_with_units = 0

    for f in files:
        data = json.load(open(f))

        units = data.get("units", [])
        if units:
            frames_with_units += 1

        for u in units:
            key = (u["card_name"], u.get("team", "?"))
            unit_obs[key] += 1
            unit_status[u.get("status", "unknown")] += 1
            actions[u.get("action", "unknown")] += 1

        for u in data.get("added_units", []):
            added_units[u["card_name"]] += 1
            if u.get("spawned_by"):
                added_spawners[u["spawned_by"]] += 1

        for r in data.get("rejected_predictions", []):
            rejected[r.get("card_name", "unknown")] += 1

        signals = data.get("replay_signals", {})
        if signals.get("opponent_elixir") is not None:
            elixir_readings.append(signals["opponent_elixir"])
        for card in signals.get("opponent_hand", []):
            if card != "Unknown":
                hand_cards[card] += 1
        if signals.get("opponent_selected_card"):
            selected_cards[signals["opponent_selected_card"]] += 1

    # Print results
    print(f"{'=' * 60}")
    print(f"REFINED LABEL ANALYSIS — {frames_processed} frames")
    print(f"{'=' * 60}")
    print(f"Frames with units: {frames_with_units}/{frames_processed} "
          f"({100*frames_with_units/frames_processed:.0f}%)\n")

    # Unit observations by card
    print(f"{'TROOP OBSERVATIONS':^60}")
    print(f"{'-' * 60}")
    print(f"{'Card':<25} {'Team':<10} {'Count':>6} {'Avg/frame':>10}")
    print(f"{'-' * 60}")

    # Sort by count descending
    for (card, team), count in sorted(unit_obs.items(), key=lambda x: -x[1]):
        avg = count / frames_processed
        print(f"{card:<25} {team:<10} {count:>6} {avg:>10.2f}")

    print(f"\nTotal confirmed observations: {sum(unit_obs.values())}")

    # Status breakdown
    print(f"\n{'STATUS BREAKDOWN':^60}")
    print(f"{'-' * 60}")
    for status, count in sorted(unit_status.items(), key=lambda x: -x[1]):
        print(f"  {status:<20} {count:>6}")

    # Actions
    print(f"\n{'ACTION BREAKDOWN':^60}")
    print(f"{'-' * 60}")
    for action, count in sorted(actions.items(), key=lambda x: -x[1]):
        print(f"  {action:<20} {count:>6}")

    # Added sub-units
    if added_units:
        print(f"\n{'ADDED SUB-UNITS':^60}")
        print(f"{'-' * 60}")
        for card, count in sorted(added_units.items(), key=lambda x: -x[1]):
            print(f"  {card:<25} {count:>6}")
        print(f"\n  Spawned by:")
        for spawner, count in sorted(added_spawners.items(), key=lambda x: -x[1]):
            print(f"    {spawner:<23} {count:>6}")

    # Rejected
    if rejected:
        print(f"\n{'REJECTED PREDICTIONS':^60}")
        print(f"{'-' * 60}")
        for card, count in sorted(rejected.items(), key=lambda x: -x[1]):
            print(f"  {card:<25} {count:>6}")

    # Elixir
    if elixir_readings:
        print(f"\n{'OPPONENT ELIXIR':^60}")
        print(f"{'-' * 60}")
        elixir_dist = Counter(elixir_readings)
        for e in range(11):
            count = elixir_dist.get(e, 0)
            bar = '#' * (count // 5)
            print(f"  {e:>2} elixir: {count:>5} frames  {bar}")
        avg_e = sum(elixir_readings) / len(elixir_readings)
        print(f"\n  Average: {avg_e:.1f} elixir")
        print(f"  Readings: {len(elixir_readings)}/{frames_processed} frames "
              f"({100*len(elixir_readings)/frames_processed:.0f}%)")

    # Hand cards
    if hand_cards:
        print(f"\n{'OPPONENT HAND FREQUENCY':^60}")
        print(f"{'-' * 60}")
        for card, count in sorted(hand_cards.items(), key=lambda x: -x[1]):
            pct = 100 * count / frames_processed
            sel = selected_cards.get(card, 0)
            sel_str = f" (selected {sel}x)" if sel else ""
            print(f"  {card:<25} {count:>5} ({pct:>4.0f}%){sel_str}")


if __name__ == "__main__":
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    analyze(d)
