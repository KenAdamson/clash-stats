"""Pilot-signal verdict: is play style emergent in the WP latent space?

Consumes the shards from pilot_embed_extract.py. The question, sharpened:
given only HOW someone plays (placements, timing, positions, tempo — the
encoder's inputs), does the space recognize the same pilot across DIFFERENT
decks better than chance, and how does that signal compare to the deck
signal that is trivially present (card ids are inputs)?

Design:
  signature = normalized centroid of a (player, deck) group's embeddings
              (>= MIN_GROUP games), computed per embedding kind
              (mean / own-side / last-tick).
  positives = same pilot, different deck        (the style claim)
  easy negs = different pilot, different deck   (chance floor)
  hard negs = different pilot, SAME deck hash   (the deck confound itself:
              if style < deck, these score HIGH and AUC_hard drops)

  AUC_easy  : style signal exists at all (vs chance)
  AUC_hard  : style vs deck — does "same pilot" beat "same deck" as the
              organizing principle? >0.5 means a pilot's other-deck games
              are closer than a stranger playing the IDENTICAL deck.
  retrieval : query with a pilot's deck-A signature against ALL other
              signatures on OTHER decks; rank of the pilot's own deck-B.

  flagship  : Ken. main (#L90009GPP, PEKKA-GY, ~12k) vs alt (#VRVR9Q2QP,
              underleveled, low trophies, disjoint decks). For each alt
              deck: percentile of sim(alt-deck, main) among
              sim(control-on-similar-decks, main). High percentile = the
              space sees Ken through the deck and the trophy gap.

Run with cwd=/app after extraction:  PYTHONPATH=/app/src python3 tools/eval/pilot_signal_eval.py
Env: PILOT_MIN_GROUP (default 12)
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

SHARD_DIR = Path("data/pilot_embed/wp_v9")
MIN_GROUP = int(os.environ.get("PILOT_MIN_GROUP", "12"))
MAIN, ALT = "#L90009GPP", "#VRVR9Q2QP"
KINDS = ("emb_own", "emb_mean", "emb_last")
RNG = np.random.default_rng(7)


def load_shards():
    tags, decks, trophies = [], [], []
    embs = {k: [] for k in KINDS}
    for shard in sorted(SHARD_DIR.glob("shard_*.npz")):
        z = np.load(shard, allow_pickle=False)
        tags.append(z["player_tags"]); decks.append(z["deck_hashes"])
        trophies.append(z["trophies"])
        for k in KINDS:
            embs[k].append(z[k])
    tags = np.concatenate(tags); decks = np.concatenate(decks)
    trophies = np.concatenate(trophies)
    embs = {k: np.concatenate(v) for k, v in embs.items()}
    return tags, decks, trophies, embs


def signatures(tags, decks, trophies, emb):
    """(player, deck) -> normalized centroid + mean trophies, groups >= MIN_GROUP."""
    idx = defaultdict(list)
    for i, (t, d) in enumerate(zip(tags, decks)):
        if d and d != "None":
            idx[(t, d)].append(i)
    sig, troph = {}, {}
    for key, rows in idx.items():
        if len(rows) < MIN_GROUP:
            continue
        c = emb[rows].mean(0)
        n = np.linalg.norm(c)
        if n > 0:
            sig[key] = c / n
            troph[key] = float(np.mean(trophies[rows]))
    return sig, troph


def auc(pos, neg):
    """Rank-based AUC of pos scores over neg scores."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    rp = ranks[: len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def evaluate(kind, sig, troph):
    by_player = defaultdict(list)
    by_deck = defaultdict(list)
    for (t, d) in sig:
        by_player[t].append(d)
        by_deck[d].append(t)

    pos, easy, hard = [], [], []
    keys = list(sig.keys())
    for t, ds in by_player.items():
        if len(ds) < 2:
            continue
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                pos.append(float(sig[(t, ds[i])] @ sig[(t, ds[j])]))
    # easy negatives: random different-pilot different-deck pairs
    for _ in range(min(len(pos) * 20, 200_000)):
        (t1, d1), (t2, d2) = (keys[RNG.integers(len(keys))] for _ in range(2))
        if t1 != t2 and d1 != d2:
            easy.append(float(sig[(t1, d1)] @ sig[(t2, d2)]))
    # hard negatives: different pilots on the SAME deck
    for d, ts in by_deck.items():
        uts = list(dict.fromkeys(ts))
        for i in range(len(uts)):
            for j in range(i + 1, len(uts)):
                if uts[i] != uts[j]:
                    hard.append(float(sig[(uts[i], d)] @ sig[(uts[j], d)]))

    # retrieval: query deck A of a multi-deck pilot; candidates = all signatures
    # on OTHER decks; rank of pilot's own deck-B (best of their others).
    rr, hits1, hits5, n_q = [], 0, 0, 0
    key_arr = keys
    mat = np.stack([sig[k] for k in key_arr])
    for t, ds in by_player.items():
        if len(ds) < 2:
            continue
        for d in ds:
            q = sig[(t, d)]
            sims = mat @ q
            order = []
            for kidx in np.argsort(-sims):
                t2, d2 = key_arr[kidx]
                if d2 == d:                    # cross-deck candidates only
                    continue
                order.append((t2, d2))
            rank = next((r + 1 for r, (t2, _) in enumerate(order) if t2 == t), None)
            if rank:
                rr.append(1.0 / rank)
                hits1 += rank == 1
                hits5 += rank <= 5
                n_q += 1
    return {
        "kind": kind, "signatures": len(sig),
        "pos_pairs": len(pos), "hard_pairs": len(hard),
        "auc_easy": round(auc(pos, easy), 4),
        "auc_hard": round(auc(pos, hard), 4),
        "retrieval_mrr": round(float(np.mean(rr)), 4) if rr else None,
        "recall@1": round(hits1 / n_q, 4) if n_q else None,
        "recall@5": round(hits5 / n_q, 4) if n_q else None,
        "queries": n_q,
    }


def flagship(kind, sig, troph):
    """Ken: alt decks vs main, ranked against controls on similar decks."""
    main_keys = [k for k in sig if k[0] == MAIN]
    alt_keys = [k for k in sig if k[0] == ALT]
    if not main_keys or not alt_keys:
        return {"kind": kind, "note": f"main sigs={len(main_keys)} alt sigs={len(alt_keys)} — insufficient"}
    main_sig = np.mean([sig[k] for k in main_keys], axis=0)
    main_sig /= np.linalg.norm(main_sig)
    out = []
    for ak in alt_keys:
        s_alt = float(sig[ak] @ main_sig)
        # controls: everyone else's signatures on the SAME deck as this alt deck,
        # else (fallback) all non-Ken signatures in the alt group's trophy band.
        same_deck = [k for k in sig if k[1] == ak[1] and k[0] not in (MAIN, ALT)]
        pool = same_deck
        pool_name = "same-deck"
        if len(pool) < 8:
            band = troph[ak]
            pool = [k for k in sig
                    if k[0] not in (MAIN, ALT) and abs(troph[k] - band) < 1500]
            pool_name = "trophy-band"
        ctrl = [float(sig[k] @ main_sig) for k in pool]
        pct = float(np.mean([s_alt > c for c in ctrl])) if ctrl else float("nan")
        out.append({
            "alt_deck": ak[1][:10], "sim_to_main": round(s_alt, 4),
            "controls": len(ctrl), "pool": pool_name,
            "percentile_vs_controls": round(100 * pct, 1),
        })
    return {"kind": kind, "alt_decks": out}


def main():
    tags, decks, trophies, embs = load_shards()
    print(f"loaded {len(tags)} games, {len(set(tags))} players")
    report = {"summary": [], "flagship": []}
    for kind in KINDS:
        sig, troph = signatures(tags, decks, trophies, embs[kind])
        report["summary"].append(evaluate(kind, sig, troph))
        report["flagship"].append(flagship(kind, sig, troph))
    print(json.dumps(report, indent=1))
    Path("data/pilot_embed/verdict.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
