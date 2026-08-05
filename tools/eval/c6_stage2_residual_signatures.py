"""C6 Stage 2: does the deck-normalized intrinsic IDENTIFY a pilot?

Stage 1 established that a deck-scaled intrinsic EXISTS — per-feature ICC of
(player, deck) residuals is well clear of zero (first_move_s 0.55 at
split-half 0.93; the reaction family 0.14-0.28). Consistency is not identity,
though: a feature can be reliably yours and still not distinguish you from
the thousand other players who share the value. Stage 2 asks the question
that actually matters by feeding those residuals to the UNCHANGED Phase-0
harness — same distance->=3 positives, same same-deck hard negatives, same
bootstrap CIs, same flagship.

The comparison that tests C6's thesis directly is against C3. C3 built
signatures from the RAW timing features and scored auc_hard 0.371 — below
chance, because cadence is causally deck-shaped (elixir costs drive it), so
same-deck strangers looked alike. C6's claim is that dividing out deck tempo
leaves a deck-invariant intrinsic. If that claim is true, auc_hard must move
up substantially from 0.371. If the residuals identify no better than the raw
features did, then the intrinsic Stage 1 found is real but not personal —
consistency without discriminative power.

PRE-REGISTERED, criteria fixed before the first run:
  PASS      auc_hard >= 0.55 with bootstrap CI excluding 0.50 (program bar,
            unchanged since Phase 0)
  PARTIAL   auc_hard in [0.50, 0.55) with CI excluding 0.50 — normalization
            removed the deck confound but the residue is too weak to use
  KILL      auc_hard < 0.50, or CI spanning 0.50
Secondary (not sufficient on its own): d6-8 auc_vs_easy vs the ladder peak
0.692 (C1-weak), and retrieval R@1 vs C3's floor.

Four representations, all from the same residual matrix:
  rank_all  every feature's rank residual
  rank_top  features with Stage-1 ICC_rank >= TOP_ICC
  rank_wt   every feature, scaled by sqrt(ICC x split-half) — Ken's quadrature
            gate: a feature earns weight only if it is BOTH consistent and
            self-reproducible, so a high-ICC/low-reliability feature is damped
  log_top   the parametric variant on the same top set; rank beat log on ICC,
            so this checks whether that ordering survives to identification

Run with cwd=/app after c6_tempo_normalization.py:
  PYTHONPATH=/app/src python3 tools/eval/c6_stage2_residual_signatures.py
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from pilot_signal_eval import (  # the Phase-0 harness, unchanged
    MIN_GROUP, MAIN, ALT, evaluate, flagship,
)
from c6_tempo_normalization import prepare, residualize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("tracker.ml.c6s2")

STAGE1 = Path("data/pilot_embed/verdict_c6_stage1.json")
C3 = Path("data/pilot_embed/verdict_c3_motor.json")
OUT = Path("data/pilot_embed/verdict_c6_stage2.json")
TOP_ICC = 0.20
MIN_FINITE = 8          # residuals per group needed before a feature counts
PASS_BAR, LADDER_PEAK = 0.55, 0.692


def stage1_weights(names):
    """(icc, splithalf) per feature from the Stage-1 verdict, aligned to names."""
    rows = {r["feature"]: r for r in json.loads(STAGE1.read_text())["features"]}
    icc = np.array([max(rows.get(n, {}).get("icc_rank") or 0.0, 0.0) for n in names])
    rel = np.array([max(rows.get(n, {}).get("splithalf_r") or 0.0, 0.0) for n in names])
    return icc, rel


def residual_signatures(tags, decks, trophies, res, cols, weights=None):
    """(player, deck) -> z-scored, L2-normalized mean-residual signature.

    Mirrors C3's motor_signatures exactly except that the input is the
    deck-normalized residual rather than the raw feature, so any difference in
    the verdict is attributable to normalization and nothing else.
    """
    idx = defaultdict(list)
    for i, (t, d) in enumerate(zip(tags.tolist(), decks.tolist())):
        if d and d != "None":
            idx[(t, d)].append(i)
    keys, mat, troph = [], [], {}
    for key, rows in idx.items():
        if len(rows) < MIN_GROUP:
            continue
        block = res[np.asarray(rows)][:, cols]
        with np.errstate(invalid="ignore"):
            m = np.where(np.isfinite(block).sum(0) >= MIN_FINITE,
                         np.nanmean(block, axis=0), np.nan)
        if np.isfinite(m).sum() < 2:
            continue
        keys.append(key); mat.append(m)
        troph[key] = float(np.mean(trophies[np.asarray(rows)]))
    mat = np.asarray(mat, dtype=np.float64)
    mu, sd = np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)
    sd[sd == 0] = 1.0
    z = np.nan_to_num((mat - mu) / sd, nan=0.0)   # missing feature -> population mean
    if weights is not None:
        z = z * weights[None, :]
    z /= np.linalg.norm(z, axis=1, keepdims=True).clip(min=1e-9)
    return {k: z[i] for i, k in enumerate(keys)}, troph


def main():
    cs, tags, decks, trophies, feats, names, avg_el, cyc4, band, cell = prepare()
    res_rank, res_log = residualize(feats, avg_el, cyc4, band, cell)
    icc, rel = stage1_weights(names)
    top = np.where(icc >= TOP_ICC)[0]
    allc = np.arange(len(names))
    logger.info("top-ICC features (>= %.2f): %s", TOP_ICC, [names[i] for i in top])

    variants = [
        ("c6_rank_all", res_rank, allc, None),
        ("c6_rank_top", res_rank, top, None),
        ("c6_rank_wt", res_rank, allc, np.sqrt(icc * rel)),
        ("c6_log_top", res_log, top, None),
    ]

    baseline = {}
    if C3.exists():
        c3 = json.loads(C3.read_text())["summary"]
        baseline = {"auc_hard": c3["auc_hard"], "auc_easy": c3["auc_easy"],
                    "d6-8": (c3["by_deck_distance"].get("d6-8") or {}).get("auc_vs_easy"),
                    "recall@1": c3["recall@1"]}
        logger.info("C3 raw-timing baseline: %s", baseline)

    out = {"baseline_c3_raw": baseline, "top_icc_features": [names[i] for i in top],
           "pass_bar": PASS_BAR, "variants": {}}
    for kind, res, cols, w in variants:
        sig, troph = residual_signatures(tags, decks, trophies, res, cols,
                                         None if w is None else w[cols])
        s = evaluate(kind, sig, troph, cs)
        f = flagship(kind, sig, troph, cs)
        out["variants"][kind] = {"summary": s, "flagship": f}
        d68 = (s["by_deck_distance"].get("d6-8") or {}).get("auc_vs_easy")
        print(f"\n== {kind} ({len(sig)} sigs, {len(cols)} features) ==")
        print(f"  auc_hard {s['auc_hard']}  CI {s['auc_hard_ci']}"
              f"   auc_easy {s['auc_easy']}  CI {s['auc_easy_ci']}")
        print(f"  d6-8 {d68}   R@1 {s['recall@1']}   MRR {s['retrieval_mrr']}")
        for a in f.get("alt_decks", []):
            print(f"  flagship alt d={a['deck_dist_to_main']} sim={a['sim_to_main']} "
                  f"pct={a['percentile_vs_controls']}")

    best = max(out["variants"].values(), key=lambda v: v["summary"]["auc_hard"])
    bh, bci = best["summary"]["auc_hard"], best["summary"]["auc_hard_ci"]
    lo = bci[0] if isinstance(bci, (list, tuple)) else None
    verdict = ("PASS" if bh >= PASS_BAR and lo and lo > 0.5 else
               "PARTIAL" if bh >= 0.5 and lo and lo > 0.5 else "KILL")
    out["verdict"] = {"best_kind": best["summary"]["kind"], "auc_hard": bh,
                      "ci": bci, "result": verdict,
                      "vs_c3_raw": None if not baseline else round(bh - baseline["auc_hard"], 4)}
    print(f"\n== C6 STAGE 2 VERDICT: {verdict} ==")
    print(f"  best {best['summary']['kind']} auc_hard {bh} (bar {PASS_BAR}, CI {bci})")
    if baseline:
        print(f"  vs C3 raw timing {baseline['auc_hard']}: {bh - baseline['auc_hard']:+.4f}")
    print(f"  ladder peak for reference (d6-8): {LADDER_PEAK}")
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
