"""Profile the final clustering against every game, not a sample.

Sampling is what made the earlier candidate look good: HDBSCAN's density
threshold is relative to the point cloud, so a 400k subsample opened gaps that
do not exist in 1.9M, and a min_cluster_size tuned there produced a 99.3% blob
at full scale. So this reads all of it.

Reports per cluster: size, win rate, mean duration -- and then whether the
clustering is separating OUTCOME (which the encoder was trained on, so
rediscovering it means the manifold adds nothing) or something else. The verdict
weights by cluster size: an earlier version flagged any extreme cluster at all,
which condemned a clustering whose extremes were tiny satellites around two
near-base-rate bulk clusters.

Run with cwd=/app:
  PYTHONPATH=/app/src python3 tools/diag/cluster_profile_full.py
"""

import os
import sys
from collections import defaultdict

import numpy as np
from sqlalchemy import text

sys.path.insert(0, "/app/src")
from tracker.database import get_engine, get_session      # noqa: E402

WORK = "/app/data/tcn_embed_work"


def main() -> None:
    labels = np.load("%s/cluster_ids_3d.npy" % WORK)
    bids = open("%s/battle_ids.txt" % WORK).read().splitlines()
    n = min(len(labels), len(bids))
    by_bid = {bids[i]: int(labels[i]) for i in range(n)}
    print("profiling %d games, %d distinct labels" % (
        n, len(set(labels[:n].tolist()))))

    session = get_session(get_engine(os.environ["DATABASE_URL"]))
    agg = defaultdict(lambda: {"n": 0, "w": 0, "dsum": 0.0, "dn": 0})
    seen = 0
    for bid, res, dur in session.execute(text(
            "SELECT battle_id, result, battle_duration FROM battles "
            "WHERE battle_type IN ('PvP','pathOfLegend') AND result IN ('win','loss')"
    ).execution_options(stream_results=True, yield_per=50000)):
        lab = by_bid.get(bid)
        if lab is None:
            continue
        a = agg[lab]
        a["n"] += 1
        a["w"] += 1 if res == "win" else 0
        if dur:
            a["dsum"] += dur
            a["dn"] += 1
        seen += 1

    total = sum(a["n"] for a in agg.values())
    base = sum(a["w"] for a in agg.values()) / max(total, 1)
    print("matched %d games, corpus win rate %.3f\n" % (seen, base))

    rows = []
    for lab, a in agg.items():
        if a["n"] == 0:
            continue
        rows.append((a["n"], lab, a["w"] / a["n"],
                     a["dsum"] / a["dn"] if a["dn"] else None))
    rows.sort(reverse=True)

    def name(l):
        return "noise" if l < 0 else ("sub-%d" % (l - 1000) if l >= 1000 else "c%d" % l)

    print("%10s %10s %7s %9s %10s" % ("cluster", "games", "share", "win rate", "mean secs"))
    for cnt, lab, wr, dur in rows[:18]:
        print("%10s %10d %6.1f%% %9.3f %10s" % (
            name(lab), cnt, 100.0 * cnt / total, wr,
            "%.0f" % dur if dur else "-"))

    # Outcome-contamination, weighted by games rather than by cluster count.
    extreme = sum(c for c, l, wr, _ in rows if l >= 0 and (wr > 0.80 or wr < 0.20))
    nearbase = sum(c for c, l, wr, _ in rows if l >= 0 and abs(wr - base) < 0.08)
    print("\nshare of games in outcome-extreme clusters (wr>.80 or <.20): %.1f%%"
          % (100.0 * extreme / total))
    print("share of games in near-base-rate clusters (|wr-base|<.08):    %.1f%%"
          % (100.0 * nearbase / total))

    durs = [d for c, l, wr, d in rows if l >= 0 and d and c > 5000]
    if durs:
        print("duration spread across major clusters: %.0fs .. %.0fs" % (min(durs), max(durs)))
    print("\n%s" % (
        "VERDICT: mostly a relabelling of win/loss."
        if extreme > 0.5 * total else
        "VERDICT: bulk of games sit in clusters near the base win rate, so the "
        "structure is NOT outcome. Check the duration spread for what it is."))


if __name__ == "__main__":
    main()
