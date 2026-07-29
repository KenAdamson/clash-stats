"""C5: distributional / trajectory geometry — stop using centroids.

A (player, deck) group is represented as the DISTRIBUTION of its per-game
vectors, compared by sliced-Wasserstein distance approximated with quantile
sketches: project every game vector onto K shared random directions, record
L quantiles per direction; L1 distance between K×L sketches ≈ sliced-W1.
This keeps full pairwise evaluation (including complete retrieval) at
vectorized-numpy cost.

Variants:
  sw_full   — embedding clouds as-is (location + shape)
  sw_shape  — clouds with the group CENTROID SUBTRACTED first: pure
              distribution shape. Centroid cosine already failed (0.594
              d6-8); this isolates exactly the information it destroyed —
              C5's thesis lives or dies here.
  sw_timing — 21-dim timing clouds, same sketching.

Same pair construction, strata, CIs and bars as the Phase-0 harness
(distance-≥3 positives, same-deck hard negatives, AUC_hard > 0.5 CI-clear,
flagship d=8 top decile); only the similarity function changes (−L1 between
sketches instead of cosine between centroids).

Run with cwd=/app:  PYTHONPATH=/app/src python3 tools/eval/c5_distributional_eval.py
Env: C5_SLICES (64), C5_QUANTILES (16), C5_MAX_GAMES (64), C5_SEED (7)
"""

import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from pilot_signal_eval import (
    MIN_GROUP, MIN_DECK_DIST, MAIN, ALT,
    load_cardsets, deck_dist, auc, auc_ci, STRATA,
)
from c1_adversarial_projection import load_games

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("tracker.ml.c5_dist")

K_SLICES = int(os.environ.get("C5_SLICES", "64"))
L_Q = int(os.environ.get("C5_QUANTILES", "16"))
MAX_GAMES = int(os.environ.get("C5_MAX_GAMES", "64"))
SEED = int(os.environ.get("C5_SEED", "7"))
OUT = Path("data/pilot_embed/verdict_c5_distributional.json")
QS = np.linspace(0.05, 0.95, L_Q)


def sketch(cloud: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """K×L quantile sketch of a point cloud (n×d) under shared projections."""
    proj = cloud @ theta                              # n × K
    return np.quantile(proj, QS, axis=0).T.ravel()    # K*L


def build_sketches(tags, decks, trophies, x, rng, center: bool):
    d = x.shape[1]
    theta = rng.standard_normal((d, K_SLICES))
    theta /= np.linalg.norm(theta, axis=0, keepdims=True)
    idx = defaultdict(list)
    for i, (t, dk) in enumerate(zip(tags.tolist(), decks.tolist())):
        if dk and dk != "None":
            idx[(t, dk)].append(i)
    sig, troph = {}, {}
    for key, rows in idx.items():
        if len(rows) < MIN_GROUP:
            continue
        rows = np.asarray(rows)
        if len(rows) > MAX_GAMES:
            rows = rng.choice(rows, MAX_GAMES, replace=False)
        cloud = x[rows]
        if center:
            cloud = cloud - cloud.mean(0, keepdims=True)
        sig[key] = sketch(cloud, theta)
        troph[key] = float(np.mean(trophies[rows]))
    return sig, troph


def evaluate_dist(kind, sig, troph, cs, rng):
    """Phase-0 pair construction with similarity = −L1 between sketches."""
    keys = list(sig)
    mat = np.stack([sig[k] for k in keys]).astype(np.float32)
    pos_k = {k: i for i, k in enumerate(keys)}

    def sim(a, b):
        return -float(np.abs(mat[pos_k[a]] - mat[pos_k[b]]).mean())

    by_player, by_deck = defaultdict(list), defaultdict(list)
    for (t, d) in keys:
        by_player[t].append(d); by_deck[d].append(t)

    pos_by_stratum = {s: [] for s in STRATA}
    for t, ds in by_player.items():
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                dist = deck_dist(cs, ds[i], ds[j])
                if dist is None:
                    continue
                for lo, hi in STRATA:
                    if lo <= dist <= hi:
                        pos_by_stratum[(lo, hi)].append(sim((t, ds[i]), (t, ds[j])))
    pos = [s for (lo, _), v in pos_by_stratum.items() if lo >= MIN_DECK_DIST for s in v]

    easy, tries = [], 0
    while len(easy) < min(max(len(pos), 100) * 20, 120_000) and tries < 600_000:
        tries += 1
        k1, k2 = (keys[rng.integers(len(keys))] for _ in range(2))
        if k1[0] == k2[0]:
            continue
        dist = deck_dist(cs, k1[1], k2[1])
        if dist is not None and dist >= MIN_DECK_DIST:
            easy.append(sim(k1, k2))
    hard = []
    for d, ts in by_deck.items():
        uts = list(dict.fromkeys(ts))
        for i in range(len(uts)):
            for j in range(i + 1, len(uts)):
                hard.append(sim((uts[i], d), (uts[j], d)))

    # full retrieval via batched pairwise L1 (mean-abs) distances
    rr, hits1, hits5, n_q = [], 0, 0, 0
    for t, ds in by_player.items():
        if len(ds) < 2:
            continue
        for d in ds:
            if not any((deck_dist(cs, d, d2) or 0) >= MIN_DECK_DIST for d2 in ds if d2 != d):
                continue
            q = mat[pos_k[(t, d)]]
            dists = np.abs(mat - q).mean(1)
            order = np.argsort(dists)
            rank = 0
            for kidx in order:
                t2, d2 = keys[kidx]
                dd = deck_dist(cs, d, d2)
                if dd is None or dd < MIN_DECK_DIST:
                    continue
                rank += 1
                if t2 == t:
                    rr.append(1.0 / rank); hits1 += rank == 1; hits5 += rank <= 5
                    n_q += 1
                    break
    strat = {f"d{lo}-{hi}": {"n": len(v), "auc_vs_easy": round(auc(v, easy), 4) if v else None}
             for (lo, hi), v in pos_by_stratum.items()}
    return {
        "kind": kind, "signatures": len(sig),
        "pos_pairs_d>=3": len(pos), "hard_pairs": len(hard),
        "auc_easy": round(auc(pos, easy), 4), "auc_easy_ci": auc_ci(pos, easy),
        "auc_hard": round(auc(pos, hard), 4), "auc_hard_ci": auc_ci(pos, hard),
        "by_deck_distance": strat,
        "retrieval_mrr": round(float(np.mean(rr)), 4) if rr else None,
        "recall@1": round(hits1 / n_q, 4) if n_q else None,
        "recall@5": round(hits5 / n_q, 4) if n_q else None,
        "queries": n_q,
    }


def flagship_dist(kind, sig, troph, cs, tags, decks, trophies, x, rng, center):
    """Ken: MAIN's whole cloud as one sketch; alt decks vs same-band controls."""
    main_rows = np.where(tags == MAIN)[0]
    if len(main_rows) < MIN_GROUP:
        return {"kind": kind, "note": "insufficient main games"}
    d = x.shape[1]
    theta = rng.standard_normal((d, K_SLICES))
    theta /= np.linalg.norm(theta, axis=0, keepdims=True)
    # rebuild all sketches under THIS theta so distances are comparable
    def mk(rows):
        rows = np.asarray(rows)
        if len(rows) > MAX_GAMES:
            rows = rng.choice(rows, MAX_GAMES, replace=False)
        cloud = x[rows]
        if center:
            cloud = cloud - cloud.mean(0, keepdims=True)
        return sketch(cloud, theta)
    main_sk = mk(main_rows)
    groups = defaultdict(list)
    for i, (t, dk) in enumerate(zip(tags.tolist(), decks.tolist())):
        if dk and dk != "None":
            groups[(t, dk)].append(i)
    out = []
    for (t, dk), rows in groups.items():
        if t != ALT or len(rows) < MIN_GROUP:
            continue
        s_alt = -float(np.abs(mk(rows) - main_sk).mean())
        band = float(np.mean(trophies[rows]))
        dists = sorted({deck_dist(cs, dk, mdk) for mdk in
                        {d2 for (t2, d2) in groups if t2 == MAIN}} - {None})
        ctrl = []
        for (t2, dk2), rows2 in groups.items():
            if t2 in (MAIN, ALT) or len(rows2) < MIN_GROUP:
                continue
            if abs(float(np.mean(trophies[rows2])) - band) < 1500:
                ctrl.append(-float(np.abs(mk(rows2) - main_sk).mean()))
        pct = float(np.mean([s_alt > c for c in ctrl])) if ctrl else float("nan")
        out.append({"alt_deck": dk[:10], "deck_dist_to_main": dists,
                    "sim_to_main": round(s_alt, 4), "controls": len(ctrl),
                    "percentile_vs_controls": round(100 * pct, 1)})
    return {"kind": kind, "alt_decks": out}


def main():
    rng = np.random.default_rng(SEED)
    cs = load_cardsets()
    bids, tags, decks, trophies, x = load_games()
    emb, timing = x[:, :512], x[:, 512:]

    report = {"summary": [], "flagship": []}
    for kind, feats, center in (("sw_full", emb, False),
                                ("sw_shape", emb, True),
                                ("sw_timing", timing, False)):
        r = np.random.default_rng(SEED)          # same theta per variant
        sig, troph = build_sketches(tags, decks, trophies, feats, r, center)
        logger.info("%s: %d sketches (K=%d L=%d)", kind, len(sig), K_SLICES, L_Q)
        fl = flagship_dist(kind, sig, troph, cs, tags, decks, trophies, feats,
                           np.random.default_rng(SEED), center)
        print(f"== C5 KILL-TEST [{kind}] ==")
        for a in fl.get("alt_decks", []):
            print(f"  alt {a['alt_deck']} dist={a['deck_dist_to_main']} "
                  f"pct={a['percentile_vs_controls']} (n={a['controls']})")
        summary = evaluate_dist(kind, sig, troph, cs, np.random.default_rng(SEED))
        print(json.dumps({k: v for k, v in summary.items()
                          if k not in ("by_deck_distance",)}, indent=1))
        report["summary"].append(summary)
        report["flagship"].append(fl)
    OUT.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
