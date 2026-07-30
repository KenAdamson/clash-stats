"""C6 Stage 1: is there a deck-scaled INTRINSIC in the timing features?

Per-feature ICC of (player, deck)-group residuals across each player's
decks, after removing deck-tempo and trophy-band effects two ways:

  rank : normal-scores within (deck-tempo-bin x band) cells — annihilates
         ANY monotone deck/band effect without estimating the curve.
  log  : log1p feature regressed on [avg_elixir, cycle4, band], residual
         studentized within cell — the parametric multiplicative model.

rank > log on ICC is evidence the true deck->timing curve is non-log
(the "inverted logistic" suspicion). ICC pairs are restricted to a
player's deck pairs at TRUE distance >= 3 (the Phase-0 rule), and group
residuals are precision-weighted with split-half reliability reported per
feature (the quadrature gate: odd/even half-estimates must agree).

Kill criterion (pre-registered): reaction-family ICC indistinguishable
from zero -> C6 falsified for this dataset.

Run with cwd=/app:  PYTHONPATH=/app/src python3 tools/eval/c6_tempo_normalization.py
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from pilot_signal_eval import MIN_GROUP, MIN_DECK_DIST, MAIN, ALT, load_cardsets, deck_dist

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("tracker.ml.c6")

MOTOR_DIR = Path("data/pilot_embed/motor")
VOCAB = Path("data/card_vocab.json")
OUT = Path("data/pilot_embed/verdict_c6_stage1.json")
N_ELIXIR_BINS = 5
BAND_W = 1500
SEED = 7


def load_motor():
    tags, decks, trophies, feats, names = [], [], [], [], None
    for f in sorted(MOTOR_DIR.glob("motor_*.npz")):
        z = np.load(f, allow_pickle=False)
        tags.append(z["player_tags"]); decks.append(z["deck_hashes"])
        trophies.append(z["trophies"]); feats.append(z["features"])
        names = z["feature_names"].tolist()
    return (np.concatenate(tags), np.concatenate(decks),
            np.concatenate(trophies), np.concatenate(feats).astype(np.float64), names)


def deck_covariates(cs):
    """deck_hash -> (avg_elixir, cheapest-4 cycle cost) via the vocab cache."""
    vocab = json.loads(VOCAB.read_text())
    cost = {name: el for name, el in vocab["rows"] if el is not None}
    cov = {}
    for dh, cards in cs.items():
        cs_ = [cost.get(c) for c in cards]
        cs_ = [c for c in cs_ if c is not None]
        if len(cs_) >= 6:
            cov[dh] = (float(np.mean(cs_)), float(np.sum(sorted(cs_)[:4])))
    return cov


def normal_scores(v):
    r = np.argsort(np.argsort(v)) + 1
    from scipy.stats import norm  # scipy available in the image (sklearn dep)
    return norm.ppf((r - 0.5) / len(v))


def icc_oneway(groups: list[np.ndarray]):
    """ICC(1): between-player consistency of group-level residual means."""
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 8:
        return None, len(groups)
    k = np.mean([len(g) for g in groups])
    gm = np.concatenate(groups).mean()
    msb = np.sum([len(g) * (g.mean() - gm) ** 2 for g in groups]) / (len(groups) - 1)
    msw = np.sum([((g - g.mean()) ** 2).sum() for g in groups]) / max(
        sum(len(g) for g in groups) - len(groups), 1)
    icc = (msb - msw) / (msb + (k - 1) * msw)
    return float(icc), len(groups)


def main():
    rng = np.random.default_rng(SEED)
    cs = load_cardsets()
    cov = deck_covariates(cs)
    tags, decks, trophies, feats, names = load_motor()
    n_feat = feats.shape[1]
    logger.info("games %d, features %d, decks w/ covariates %d",
                len(tags), n_feat, len(cov))

    have_cov = np.array([d in cov for d in decks.tolist()])
    tags, decks, trophies, feats = tags[have_cov], decks[have_cov], trophies[have_cov], feats[have_cov]
    avg_el = np.array([cov[d][0] for d in decks.tolist()])
    cyc4 = np.array([cov[d][1] for d in decks.tolist()])
    band = np.clip(trophies // BAND_W, 0, 9)
    el_bin = np.digitize(avg_el, np.quantile(avg_el, np.linspace(0, 1, N_ELIXIR_BINS + 1)[1:-1]))
    cell = el_bin * 100 + band

    # --- per-game residuals, both variants ---
    res_rank = np.full_like(feats, np.nan)
    res_log = np.full_like(feats, np.nan)
    for f in range(n_feat):
        v = feats[:, f]
        ok = np.isfinite(v)
        # rank variant: normal scores within cell
        for c in np.unique(cell):
            m = ok & (cell == c)
            if m.sum() >= 50:
                res_rank[m, f] = normal_scores(v[m])
        # log variant: log1p ~ [avg_el, cyc4, band], residual, studentized per cell
        m = ok & (v > -0.999)
        if m.sum() >= 500:
            y = np.log1p(np.clip(v[m], 0, None)) if v[m].min() >= 0 else v[m]
            Xd = np.column_stack([np.ones(m.sum()), avg_el[m], cyc4[m], band[m]])
            beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
            r = y - Xd @ beta
            tmp = np.full(m.sum(), np.nan)
            cm = cell[m]
            for c in np.unique(cm):
                s = cm == c
                if s.sum() >= 50 and np.nanstd(r[s]) > 0:
                    tmp[s] = (r[s] - np.nanmean(r[s])) / np.nanstd(r[s])
            res_log[m, f] = tmp

    # --- groups and player deck-pairs at distance >= 3 ---
    group_rows = defaultdict(list)
    for i, (t, d) in enumerate(zip(tags.tolist(), decks.tolist())):
        group_rows[(t, d)].append(i)
    groups = {k: np.asarray(v) for k, v in group_rows.items() if len(v) >= MIN_GROUP}
    by_player = defaultdict(list)
    for (t, d) in groups:
        by_player[t].append(d)
    eligible = {}
    for t, ds in by_player.items():
        keep = [d for d in ds if any(
            (deck_dist(cs, d, d2) or 0) >= MIN_DECK_DIST for d2 in ds if d2 != d)]
        if len(keep) >= 2:
            eligible[t] = keep
    logger.info("groups %d; players with distance->=3 deck pairs: %d",
                len(groups), len(eligible))

    report = {"features": [], "n_players": len(eligible)}
    print(f"{'feature':<14} {'ICC_rank':>9} {'ICC_log':>9} {'splithalf_r':>11} {'n_pl':>5}")
    for f in range(n_feat):
        row = {"feature": names[f]}
        for label, res in (("rank", res_rank), ("log", res_log)):
            per_player = []
            for t, ds in eligible.items():
                means = []
                for d in ds:
                    vals = res[groups[(t, d)], f]
                    vals = vals[np.isfinite(vals)]
                    if len(vals) >= 8:
                        means.append(vals.mean())
                if len(means) >= 2:
                    per_player.append(np.asarray(means))
            icc, n = icc_oneway(per_player)
            row[f"icc_{label}"], row[f"n_players_{label}"] = icc, n
        # split-half reliability of the rank residual (quadrature self-noise)
        halves = []
        for (t, d), rows_ in groups.items():
            vals = res_rank[rows_, f]
            fin = np.isfinite(vals)
            if fin.sum() >= 12:
                v_ = vals[fin]
                halves.append((v_[0::2].mean(), v_[1::2].mean()))
        if len(halves) >= 50:
            h = np.asarray(halves)
            r = float(np.corrcoef(h[:, 0], h[:, 1])[0, 1])
            row["splithalf_r"] = round(2 * r / (1 + r), 4)   # Spearman-Brown
        report["features"].append(row)
        print(f"{names[f]:<14} {str(round(row.get('icc_rank') or float('nan'), 4)):>9} "
              f"{str(round(row.get('icc_log') or float('nan'), 4)):>9} "
              f"{str(row.get('splithalf_r', '—')):>11} {row.get('n_players_rank', 0):>5}")

    # flagship sanity: Ken's rank-residuals, main vs alt, reaction family
    fam = [i for i, n in enumerate(names) if n in
           ("react_med", "react_p25", "snap_frac", "gap_med", "burst_frac")]
    for who in (MAIN, ALT):
        for d in by_player.get(who, []):
            vals = res_rank[groups[(who, d)]][:, fam]
            mm = np.nanmean(vals, axis=0)
            print(f"  {who} deck {d[:8]}: " + " ".join(
                f"{names[fam[j]]}={mm[j]:+.2f}" for j in range(len(fam))))
    OUT.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
