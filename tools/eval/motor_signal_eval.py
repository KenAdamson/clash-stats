"""C3 verdict: can TIMING ALONE re-identify a pilot across decks?

Builds per-(player, deck) motor signatures from the timing shards — the
nan-median of each feature across the group's games, z-scored across all
signatures, L2-normalized — and scores them on the EXACT Phase-0 harness
(same distance-≥3 positives, same same-deck hard negatives, same bars:
AUC_hard > 0.5 with CI excluding 0.5, flagship top decile). Nothing about
the evaluation moves; only the representation changed.

THE KILL-TEST PRINTS FIRST: main ↔ alt is the same human on distance-8
decks 9k trophies apart. If timing cannot link Ken to Ken, motor signal
does not survive the 20Hz capture and C3 is dead for this dataset.

Run with cwd=/app after motor_timing_extract.py:
  PYTHONPATH=/app/src python3 tools/eval/motor_signal_eval.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pilot_signal_eval import (  # the Phase-0 harness, unchanged
    KINDS as _EMB_KINDS,  # noqa: F401  (unused; documents what we replace)
    MIN_GROUP, MIN_DECK_DIST, MAIN, ALT,
    load_cardsets, deck_dist, evaluate, flagship,
)

MOTOR_DIR = Path("data/pilot_embed/motor")
OUT = Path("data/pilot_embed/verdict_c3_motor.json")


def load_motor():
    tags, decks, trophies, feats = [], [], [], []
    names = None
    for f in sorted(MOTOR_DIR.glob("motor_*.npz")):
        z = np.load(f, allow_pickle=False)
        tags.append(z["player_tags"]); decks.append(z["deck_hashes"])
        trophies.append(z["trophies"]); feats.append(z["features"])
        names = z["feature_names"].tolist()
    return (np.concatenate(tags), np.concatenate(decks),
            np.concatenate(trophies), np.concatenate(feats), names)


def motor_signatures(tags, decks, trophies, feats):
    """(player, deck) → z-scored, L2-normalized median-feature signature."""
    idx = defaultdict(list)
    for i, (t, d) in enumerate(zip(tags, decks)):
        if d and d != "None":
            idx[(t, d)].append(i)
    keys, mat, troph = [], [], {}
    for key, rows in idx.items():
        if len(rows) < MIN_GROUP:
            continue
        m = np.nanmedian(feats[rows], axis=0)
        keys.append(key); mat.append(m)
        troph[key] = float(np.mean(trophies[rows]))
    mat = np.asarray(mat, dtype=np.float64)
    # z-score per feature ACROSS signatures; NaN feature → population mean (0)
    mu = np.nanmean(mat, axis=0)
    sd = np.nanstd(mat, axis=0)
    sd[sd == 0] = 1.0
    z = (mat - mu) / sd
    z = np.nan_to_num(z, nan=0.0)
    z /= np.linalg.norm(z, axis=1, keepdims=True).clip(min=1e-9)
    return {k: z[i] for i, k in enumerate(keys)}, troph


def main():
    tags, decks, trophies, feats, names = load_motor()
    cs = load_cardsets()
    print(f"loaded {len(tags)} games, {feats.shape[1]} timing features: {names}")
    sig, troph = motor_signatures(tags, decks, trophies, feats)
    print(f"signatures: {len(sig)} (player,deck) groups at >= {MIN_GROUP} games\n")

    # ---- KILL-TEST FIRST ----
    fl = flagship("motor_timing", sig, troph, cs)
    print("== C3 KILL-TEST (flagship: same human, disjoint decks) ==")
    for a in fl.get("alt_decks", []):
        print(f"  alt {a['alt_deck']} dist_to_main={a['deck_dist_to_main']} "
              f"sim={a['sim_to_main']} pct={a['percentile_vs_controls']} "
              f"(n={a['controls']} {a['pool']})")
    print()

    summary = evaluate("motor_timing", sig, troph, cs)
    print("== C3 corpus verdict (Phase-0 harness) ==")
    print(json.dumps(summary, indent=1))
    OUT.write_text(json.dumps({"flagship": fl, "summary": summary,
                               "features": names}, indent=1))


if __name__ == "__main__":
    main()
