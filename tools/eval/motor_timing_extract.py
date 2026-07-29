"""C3 (Phase 1 of the pilot-style program): per-game MOTOR TIMING features.

Deck-blind by construction: the SQL selects only (side, game_tick) — card
identity and position never enter the pipeline, so the deck-leakage failure
mode that invalidated the raw-embedding eval cannot occur here, even by
accident. v1 is timing-only; micro-position jitter (magnitude-not-location)
is deferred to v2 if timing alone proves marginal.

Features per game (own side), all in seconds at the 20Hz tick rate:
  cadence   — inter-placement gap stats overall + per elixir phase
              (single <2:00, double 2:00-3:00, triple/OT >3:00)
  reaction  — opponent placement → own next placement latency: median, p25,
              snap-rate (<1.5s), ignore-rate (>6s)
  tempo     — own placements per 100 ticks, per phase; share of total
  opening   — first-placement tick (opening hesitation)

Cohort comes from the pilot-embed shards (same groups as Phase 0, so the
harness and bars are identical). Output: data/pilot_embed/motor/motor_NNNN.npz
(resumable). Read-only; run against the replica.

  DATABASE_URL=postgresql://...@192.168.7.62/clash_stats \
  PYTHONPATH=/app/src python3 tools/eval/motor_timing_extract.py
"""

import logging
import os
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("tracker.ml.motor_timing")

SHARD_DIR = Path("data/pilot_embed/wp_v9")
OUT_DIR = Path("data/pilot_embed/motor")
BATCH = int(os.environ.get("MOTOR_BATCH", "400"))  # small enough to finish
# inside the hot-standby's WAL-replay grace window; 2000-battle batches were
# cancelled with "conflict with recovery". The script resumes, so callers
# wrap it in a retry loop.
TPS = 20.0                      # ticks per second (verified 2026-07-09)
PH1, PH2 = 2400, 3600           # elixir phase boundaries in ticks

FEATURES = [
    "n_own", "n_opp", "own_share", "first_move_s",
    "gap_med", "gap_p25", "gap_p75", "gap_min", "burst_frac", "slow_frac",
    "gap_med_ph1", "gap_med_ph2", "gap_med_ph3",
    "tempo_ph1", "tempo_ph2", "tempo_ph3",
    "react_med", "react_p25", "snap_frac", "ignore_frac", "n_react",
]


def game_features(own_ticks: np.ndarray, opp_ticks: np.ndarray) -> list[float]:
    """Timing-only feature vector for one game. NaN where undefined."""
    f = dict.fromkeys(FEATURES, np.nan)
    f["n_own"], f["n_opp"] = len(own_ticks), len(opp_ticks)
    if len(own_ticks) + len(opp_ticks) > 0:
        f["own_share"] = len(own_ticks) / (len(own_ticks) + len(opp_ticks))
    if len(own_ticks) == 0:
        return [f[k] for k in FEATURES]
    f["first_move_s"] = own_ticks[0] / TPS
    end = max(own_ticks.max(), opp_ticks.max() if len(opp_ticks) else 0)

    gaps = np.diff(own_ticks) / TPS
    if len(gaps):
        f["gap_med"], f["gap_p25"], f["gap_p75"] = (
            float(np.median(gaps)), float(np.percentile(gaps, 25)),
            float(np.percentile(gaps, 75)))
        f["gap_min"] = float(gaps.min())
        f["burst_frac"] = float((gaps < 1.0).mean())
        f["slow_frac"] = float((gaps > 6.0).mean())
        mid = (own_ticks[:-1] + own_ticks[1:]) / 2
        for name, lo, hi in (("ph1", 0, PH1), ("ph2", PH1, PH2), ("ph3", PH2, 10**9)):
            sel = gaps[(mid >= lo) & (mid < hi)]
            if len(sel):
                f[f"gap_med_{name}"] = float(np.median(sel))
    for name, lo, hi in (("ph1", 0, PH1), ("ph2", PH1, PH2), ("ph3", PH2, 10**9)):
        span = min(end, hi) - lo
        if span > 200:          # >10s of that phase actually played
            n = int(((own_ticks >= lo) & (own_ticks < hi)).sum())
            f[f"tempo_{name}"] = 100.0 * n / span

    if len(opp_ticks):
        idx = np.searchsorted(own_ticks, opp_ticks, side="right")
        ok = idx < len(own_ticks)
        reacts = (own_ticks[idx[ok]] - opp_ticks[ok]) / TPS
        reacts = reacts[reacts <= 12.0]     # beyond 12s it's not a reaction
        f["n_react"] = len(reacts)
        if len(reacts) >= 3:
            f["react_med"] = float(np.median(reacts))
            f["react_p25"] = float(np.percentile(reacts, 25))
            f["snap_frac"] = float((reacts < 1.5).mean())
            f["ignore_frac"] = float((reacts > 6.0).mean())
    return [f[k] for k in FEATURES]


def main() -> int:
    meta: list[tuple[str, str, str, int]] = []   # battle_id, tag, deck, trophies
    for shard in sorted(SHARD_DIR.glob("shard_*.npz")):
        z = np.load(shard, allow_pickle=False)
        meta.extend(zip(z["battle_ids"].tolist(), z["player_tags"].tolist(),
                        z["deck_hashes"].tolist(), z["trophies"].tolist()))
    logger.info("cohort battles: %d", len(meta))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    for f in sorted(OUT_DIR.glob("motor_*.npz")):
        done.update(np.load(f, allow_pickle=False)["battle_ids"].tolist())
    todo = [m for m in meta if m[0] not in done]
    logger.info("resume: %d done, %d to extract", len(done), len(todo))

    engine = create_engine(os.environ["DATABASE_URL"])
    shard_no = len(list(OUT_DIR.glob("motor_*.npz")))
    with engine.connect() as conn:
        for start in range(0, len(todo), BATCH):
            chunk = todo[start:start + BATCH]
            ids = [m[0] for m in chunk]
            # timing only — card_name and coordinates deliberately NOT selected
            rows = conn.execute(text("""
                SELECT battle_id, side, game_tick FROM replay_events
                WHERE battle_id = ANY(:ids) AND game_tick IS NOT NULL
                ORDER BY battle_id, game_tick
            """), {"ids": ids}).all()
            own: dict[str, list[int]] = {}
            opp: dict[str, list[int]] = {}
            for bid, side, tick in rows:
                (own if side == "team" else opp).setdefault(bid, []).append(tick)
            feats, keep = [], []
            for m in chunk:
                o = np.asarray(own.get(m[0], []), dtype=np.float64)
                p = np.asarray(opp.get(m[0], []), dtype=np.float64)
                if len(o) + len(p) == 0:
                    continue
                feats.append(game_features(o, p))
                keep.append(m)
            np.savez_compressed(
                OUT_DIR / f"motor_{shard_no:04d}.npz",
                battle_ids=np.array([m[0] for m in keep]),
                player_tags=np.array([m[1] for m in keep]),
                deck_hashes=np.array([m[2] for m in keep]),
                trophies=np.array([m[3] for m in keep], dtype=np.int32),
                features=np.array(feats, dtype=np.float32),
                feature_names=np.array(FEATURES),
            )
            if shard_no % 10 == 0 or start + BATCH >= len(todo):
                logger.info("motor_%04d: %d games (%d/%d)",
                            shard_no, len(keep), start + len(chunk), len(todo))
            shard_no += 1
    logger.info("extraction complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
