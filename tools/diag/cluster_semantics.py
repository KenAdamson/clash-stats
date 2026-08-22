"""Ask what the manifold clusters actually separate.

A cluster count and a noise fraction say nothing about meaning. The TCN is
trained on win/loss, so the dominant axis of its embedding space is very likely
outcome -- and if a "clustering" is just win versus loss re-derived, it is not an
archetype map and should not be presented as one.

For each candidate min_cluster_size this reports, per cluster: size, win rate,
and mean game length. A clustering whose win rates spread to 0/1 has rediscovered
the label. One whose win rates all sit near the corpus base rate while other
properties differ is separating something else, which is the useful case.

Run with cwd=/app:
  PYTHONPATH=/app/src python3 tools/diag/cluster_semantics.py [sizes]
"""

import os
import sys
from collections import defaultdict

import numpy as np
from sqlalchemy import text

sys.path.insert(0, "/app/src")
from tracker.database import get_engine, get_session      # noqa: E402

WORK = "/app/data/tcn_embed_work"
SAMPLE = 150_000


def main() -> None:
    sizes = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1000,2500,10000").split(",")]
    import hdbscan

    emb = np.load("%s/cluster_space_10d.npy" % WORK, mmap_mode="r")
    bids = open("%s/battle_ids.txt" % WORK).read().splitlines()
    n = min(emb.shape[0], len(bids))

    rng = np.random.default_rng(46)
    idx = np.sort(rng.choice(n, size=min(SAMPLE, n), replace=False))
    sample = np.asarray(emb[idx], dtype=np.float64)
    sample_bids = [bids[i] for i in idx]

    session = get_session(get_engine(os.environ["DATABASE_URL"]))
    meta = {}
    for s in range(0, len(sample_bids), 5000):
        chunk = tuple(sample_bids[s:s + 5000])
        for bid, res, dur in session.execute(text(
                "SELECT battle_id, result, battle_duration FROM battles "
                "WHERE battle_id IN :b"), {"b": chunk}):
            meta[bid] = (1 if res == "win" else 0, dur)
    base = np.mean([v[0] for v in meta.values()])
    print("sample %d games, corpus win rate %.3f\n" % (len(meta), base))

    for mcs in sizes:
        labels = hdbscan.HDBSCAN(
            min_cluster_size=mcs, min_samples=max(5, mcs // 10),
            cluster_selection_method="eom", core_dist_n_jobs=4).fit_predict(sample)
        agg = defaultdict(lambda: [0, 0, 0.0, 0])
        for lab, bid in zip(labels, sample_bids):
            if bid not in meta:
                continue
            w, dur = meta[bid]
            a = agg[int(lab)]
            a[0] += 1
            a[1] += w
            if dur:
                a[2] += dur
                a[3] += 1
        print("min_cluster_size=%d" % mcs)
        print("  %8s %9s %9s %10s" % ("cluster", "games", "win rate", "mean secs"))
        for lab in sorted(agg, key=lambda L: -agg[L][0])[:12]:
            cnt, wins, dsum, dn = agg[lab]
            print("  %8s %9d %9.3f %10s" % (
                "noise" if lab < 0 else lab, cnt, wins / max(cnt, 1),
                "%.0f" % (dsum / dn) if dn else "-"))
        wrs = [agg[L][1] / max(agg[L][0], 1) for L in agg if L >= 0 and agg[L][0] > 200]
        if wrs:
            print("  win-rate spread across clusters: %.3f .. %.3f  %s" % (
                min(wrs), max(wrs),
                "<-- looks like it rediscovered the win/loss label"
                if (max(wrs) > 0.85 or min(wrs) < 0.15) else "(not just outcome)"))
        print()


if __name__ == "__main__":
    main()
