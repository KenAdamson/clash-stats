"""C6 addendum: why a reliable intrinsic still fails to identify.

Stage 1 measured RELIABILITY (is the value stably yours across decks).
Stage 2 measured DISCRIMINABILITY (does it separate you from everyone else).
They came apart. This asks whether the gap is a modeling shortfall or a
ceiling: how many pilots the timing channel can distinguish AT ALL.
"""
import json, numpy as np
from collections import defaultdict
from pilot_signal_eval import MIN_GROUP, MIN_DECK_DIST, load_cardsets, deck_dist
from c6_tempo_normalization import prepare, residualize, eligible_players

cs, tags, decks, trophies, feats, names, avg_el, cyc4, band, cell = prepare()
res, _ = residualize(feats, avg_el, cyc4, band, cell)
groups, by_player, eligible = eligible_players(tags, decks, cs)

# deck-level residual means (the Stage-2 signature, pre-normalization)
keys, mat, who = [], [], []
for (t, d), rows in groups.items():
    blk = res[rows]
    m = np.where(np.isfinite(blk).sum(0) >= 8, np.nanmean(blk, axis=0), np.nan)
    if np.isfinite(m).sum() >= 2:
        keys.append((t, d)); mat.append(m); who.append(t)
M = np.nan_to_num(np.asarray(mat), nan=0.0)
M = (M - M.mean(0)) / M.std(0).clip(min=1e-9)
who = np.asarray(who)

# 1. effective dimensionality (participation ratio of the eigenspectrum)
ev = np.linalg.eigvalsh(np.cov(M, rowvar=False))[::-1].clip(min=0)
pr = ev.sum()**2 / (ev**2).sum()
print(f"signatures {len(M)}  nominal dims {M.shape[1]}  EFFECTIVE dims {pr:.2f}")
print(f"  variance in top 3 PCs: {ev[:3].sum()/ev.sum():.1%}   top 5: {ev[:5].sum()/ev.sum():.1%}")

# 2. between-pilot vs within-pilot(across decks) spread in that space
wi, bw = [], []
multi = {t: [i for i, w in enumerate(who) if w == t] for t in eligible}
for t, idxs in multi.items():
    for a in range(len(idxs)):
        for b in range(a+1, len(idxs)):
            if (deck_dist(cs, keys[idxs[a]][1], keys[idxs[b]][1]) or 0) >= MIN_DECK_DIST:
                wi.append(np.linalg.norm(M[idxs[a]] - M[idxs[b]]))
rng = np.random.default_rng(7)
for _ in range(20000):
    i, j = rng.integers(len(M), size=2)
    if who[i] != who[j]:
        bw.append(np.linalg.norm(M[i] - M[j]))
wi, bw = np.asarray(wi), np.asarray(bw)
print(f"\nwithin-pilot across-deck distance: {wi.mean():.3f} +- {wi.std():.3f}  (n={len(wi)})")
print(f"between-pilot distance:            {bw.mean():.3f} +- {bw.std():.3f}")
sep = (bw.mean() - wi.mean()) / np.sqrt((bw.var() + wi.var()) / 2)
print(f"separation d' = {sep:.3f}")

# 3. capacity: distinguishable cells = (between-spread / within-spread)^eff_dims
ratio = bw.mean() / wi.mean()
print(f"\nspread ratio {ratio:.3f} over {pr:.1f} effective dims")
print(f"  order-of-magnitude distinguishable pilots ~ {ratio**pr:,.0f}")
print(f"  pilots in this evaluation: {len(set(who)):,}")

json.dump({"signatures": len(M), "nominal_dims": int(M.shape[1]),
           "effective_dims": round(float(pr), 3),
           "var_top3": round(float(ev[:3].sum()/ev.sum()), 4),
           "within_pilot_dist": round(float(wi.mean()), 4),
           "between_pilot_dist": round(float(bw.mean()), 4),
           "d_prime": round(float(sep), 4),
           "capacity_est": round(float(ratio**pr), 2),
           "pilots_present": int(len(set(who)))},
          open("data/pilot_embed/verdict_c6_capacity.json", "w"), indent=1)
