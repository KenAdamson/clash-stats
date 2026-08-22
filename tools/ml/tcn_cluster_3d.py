"""Re-cluster the manifold in 3D UMAP space instead of the 128-dim encoder space.

Clustering the 128-dim embeddings forced a subsample: HDBSCAN could not take
1.9M x 128, so it was fit on 300k and the rest assigned with
approximate_predict. That is what produced 88.1% noise -- 2,047 tight clusters
fit on a 16% sample leave most of the remaining points outside any cluster's
region. (For scale, the previous 128-dim production clustering was already 77.3%
noise, so the subsampling made a bad situation worse rather than creating it.)

In 3D the constraint disappears. HDBSCAN's Boruvka KD-tree path is designed for
low dimensions, so all 1.9M points can be clustered directly with no subsample
and no approximate_predict -- the conservative-assignment problem is gone by
construction, not by tuning.

The tradeoff is real and worth stating: these are clusters in the PROJECTED
space, so they inherit UMAP's distortions. UMAP preserves local neighbourhoods
but not global distances, so a 3D cluster means "games that land near each other
in the visualisation", which is exactly what the dashboard shows, but is a
weaker claim than "games the encoder considers similar".

min_cluster_size also has to move. The inherited value of 10 is 0.0005% of a
1.9M corpus and fragments it into thousands of specks; --sweep measures several
candidates on a sample so the choice is made on evidence rather than taste.

Run with cwd=/app:
  # look first
  PYTHONPATH=/app/src python3 tools/ml/tcn_cluster_3d.py --sweep
  # then commit
  PYTHONPATH=/app/src python3 tools/ml/tcn_cluster_3d.py --apply --min-cluster-size N
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from sqlalchemy import text

sys.path.insert(0, "/app/src")
from tracker.database import get_engine, get_session          # noqa: E402
from tracker.ml.training import TCN_MODEL_VERSION             # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("tcn_cluster3d")

WORK = Path(os.environ.get("TCN_EMBED_WORK", "/app/data/tcn_embed_work"))
SWEEP_SAMPLE = int(os.environ.get("CLUSTER_SWEEP_SAMPLE", "400000"))
# min_samples controls how conservative the density estimate is, and it matters
# more here than min_cluster_size does. An earlier version derived it as
# min_cluster_size//10, which at mcs=5000 meant 500: smooth enough to erase every
# density gap, collapsing 1.9M games into 2 clusters. The same mcs with
# min_samples=5 yields 39 balanced clusters. Keep it small and explicit.
DEFAULT_MIN_SAMPLES = int(os.environ.get("CLUSTER_MIN_SAMPLES", "5"))
MIN_SAMPLES = DEFAULT_MIN_SAMPLES
SPACE_FILE = None   # set by --cluster-dims to a purpose-built clustering space
# Second-pass cluster ids are offset so the two levels stay distinguishable in a
# single column: 0..N are primary clusters, >=NOISE_OFFSET came from re-clustering
# the primary pass's leftovers at a finer density.
NOISE_OFFSET = 1000


def _load_3d() -> np.ndarray:
    p = WORK / "embeddings_3d.npy"
    if not p.exists():
        raise SystemExit("no 3D embeddings at %s — run tcn_embed_corpus.py first" % p)
    return np.load(p, mmap_mode="r")


def build_cluster_space(n_components: int, fit_sample: int) -> Path:
    """Fit a SECOND UMAP whose only job is to be clusterable.

    The 3D projection on disk exists to be looked at: UMAP_3D_PARAMS sets
    min_dist=0.3 and spread=1.5, which push points apart so the dashboard
    scatter does not overplot. That is precisely what destroys the density gaps
    HDBSCAN separates on, and it is why clustering it yields one blob holding
    99% of the corpus.

    min_dist=0.0 does the opposite -- it lets UMAP pack points as tightly as the
    topology allows -- and a handful of components rather than 3 leaves room for
    structure that 3 dimensions has to crush together. This restores the split
    the original 50->15->3 pipeline had: cluster in an intermediate space,
    visualise in 3D. The 3D coordinates are untouched.
    """
    out = WORK / ("cluster_space_%dd.npy" % n_components)
    if out.exists():
        logger.info("reusing cluster space %s", out)
        return out
    from umap import UMAP
    emb = np.load(WORK / "embeddings_128d.npy", mmap_mode="r")
    n = emb.shape[0]
    k = min(fit_sample, n)
    rng = np.random.default_rng(45)
    idx = np.sort(rng.choice(n, size=k, replace=False))
    logger.info("fitting %dD cluster space (min_dist=0) on %d of %d", n_components, k, n)
    t0 = time.time()
    reducer = UMAP(n_components=n_components, n_neighbors=30, min_dist=0.0,
                   metric="euclidean", random_state=42)
    reducer.fit(np.asarray(emb[idx], dtype=np.float32))
    tmp = out.with_suffix(".npy.tmp")
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32,
                                   shape=(n, n_components))
    for s0 in range(0, n, 50000):
        e0 = min(s0 + 50000, n)
        mm[s0:e0] = reducer.transform(np.asarray(emb[s0:e0], dtype=np.float32))
    mm.flush(); del mm
    os.replace(tmp, out)
    logger.info("cluster space built in %.1f min -> %s", (time.time() - t0) / 60, out)
    return out


def sweep(sizes: list[int]) -> None:
    """Report cluster count and noise for each candidate min_cluster_size.

    On a sample, because the point is to compare candidates cheaply; the winner
    is then run over everything. Noise fraction is the number that matters --
    a cluster label is only useful if most games have one.
    """
    import hdbscan
    emb = np.load(SPACE_FILE, mmap_mode="r") if SPACE_FILE else _load_3d()
    n = emb.shape[0]
    k = min(SWEEP_SAMPLE, n)
    rng = np.random.default_rng(44)
    idx = np.sort(rng.choice(n, size=k, replace=False))
    sample = np.asarray(emb[idx], dtype=np.float64)
    logger.info("sweep on %d of %d points, 3D space", k, n)

    print("\n%12s %10s %10s %12s %s" % (
        "min_cluster", "clusters", "noise %", "largest %", "fit"))
    for mcs in sizes:
        t0 = time.time()
        cl = hdbscan.HDBSCAN(min_cluster_size=mcs,
                             min_samples=MIN_SAMPLES,
                             cluster_selection_method="eom",
                             core_dist_n_jobs=4)
        labels = cl.fit_predict(sample)
        n_cl = len(set(labels.tolist())) - (1 if -1 in labels else 0)
        noise = 100.0 * (labels == -1).sum() / k
        sizes_ = np.bincount(labels[labels >= 0]) if n_cl else np.array([0])
        largest = 100.0 * sizes_.max() / k if n_cl else 0.0
        print("%12d %10d %9.1f%% %11.1f%% %6.1fs" % (
            mcs, n_cl, noise, largest, time.time() - t0))
    print("\nPick for LOW noise with clusters that are not one giant blob.")


def recurse_noise(emb, labels: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """Re-cluster the primary pass's noise at a finer density scale.

    HDBSCAN's notion of noise is relative to the density it settled on for the
    whole corpus: a group can be perfectly coherent among itself and still fall
    below the global threshold. Pulling those points out and clustering them
    alone lets a finer scale apply where it is appropriate, without loosening the
    threshold everywhere and fragmenting the main structure.

    Labels come back offset by NOISE_OFFSET so a reader can always tell which
    pass produced a given id. Points that are still noise at the finer scale stay
    at -1 -- some genuinely are outliers, and inventing a cluster for them would
    be worse than admitting it.
    """
    import hdbscan
    noise_idx = np.flatnonzero(labels < 0)
    if len(noise_idx) < min_cluster_size * 2:
        logger.info("second pass: only %d noise points — skipping", len(noise_idx))
        return labels
    logger.info("second pass: re-clustering %d noise points (%.1f%%) at "
                "min_cluster_size=%d", len(noise_idx),
                100.0 * len(noise_idx) / len(labels), min_cluster_size)
    t0 = time.time()
    sub = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=MIN_SAMPLES,
        cluster_selection_method="eom", core_dist_n_jobs=4,
    ).fit_predict(np.asarray(emb[noise_idx], dtype=np.float64))
    n_sub = len(set(sub.tolist())) - (1 if -1 in sub else 0)
    rescued = int((sub >= 0).sum())
    labels = labels.copy()
    labels[noise_idx[sub >= 0]] = sub[sub >= 0] + NOISE_OFFSET
    logger.info("second pass: %d sub-clusters, rescued %d of %d (%.1f%%) in "
                "%.1f min", n_sub, rescued, len(noise_idx),
                100.0 * rescued / len(noise_idx), (time.time() - t0) / 60)
    return labels


def apply(min_cluster_size: int, dry: bool, noise_mcs: int | None = None) -> None:
    """Cluster every point in 3D, then update cluster_id in place."""
    import hdbscan
    emb = np.load(SPACE_FILE, mmap_mode="r") if SPACE_FILE else _load_3d()
    n = emb.shape[0]
    logger.info("clustering all %d points in %dD (min_cluster_size=%d)",
                n, emb.shape[1], min_cluster_size)
    t0 = time.time()
    # No subsample and no approximate_predict: every point gets a real label
    # from the same fit, which is the entire reason for moving to 3D.
    labels = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=MIN_SAMPLES,
        cluster_selection_method="eom",
        core_dist_n_jobs=4,
    ).fit_predict(np.asarray(emb, dtype=np.float64))
    n_cl = len(set(labels.tolist())) - (1 if -1 in labels else 0)
    noise = 100.0 * (labels == -1).sum() / n
    logger.info("done in %.1f min: %d clusters, %.1f%% noise",
                (time.time() - t0) / 60, n_cl, noise)

    if noise_mcs:
        labels = recurse_noise(emb, labels, noise_mcs)
        final_noise = 100.0 * (labels == -1).sum() / n
        logger.info("after second pass: %.1f%% noise remains", final_noise)

    np.save(WORK / "cluster_ids_3d.npy", labels.astype(np.int32))
    if dry:
        logger.info("--dry: labels saved, database untouched")
        return

    bids = (WORK / "battle_ids.txt").read_text().splitlines()
    session = get_session(get_engine(os.environ["DATABASE_URL"]))
    sql = text("UPDATE game_embeddings SET cluster_id = :cid "
               "WHERE battle_id = :bid AND model_version = :ver")
    done = 0
    t0 = time.time()
    for s in range(0, min(len(bids), n), 5000):
        e = min(s + 5000, min(len(bids), n))
        session.execute(sql, [
            {"bid": bids[i], "cid": int(labels[i]) if labels[i] >= 0 else None,
             "ver": TCN_MODEL_VERSION}
            for i in range(s, e)])
        session.commit()
        done += e - s
        if (s // 5000) % 20 == 0:
            logger.info("  updated %d/%d", done, n)
    logger.info("updated %d rows in %.1f min", done, (time.time() - t0) / 60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry", action="store_true", help="cluster but do not write")
    ap.add_argument("--min-cluster-size", type=int, default=None)
    ap.add_argument("--sizes", type=str, default="50,200,500,1000,2500,5000")
    ap.add_argument("--cluster-dims", type=int, default=None,
                    help="build/use a min_dist=0 UMAP space of N dims for clustering")
    ap.add_argument("--fit-sample", type=int, default=300000)
    ap.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    ap.add_argument("--noise-min-cluster-size", type=int, default=None,
                    help="re-cluster the first pass's noise at this finer scale")
    args = ap.parse_args()

    global SPACE_FILE, MIN_SAMPLES
    MIN_SAMPLES = args.min_samples
    if args.cluster_dims:
        SPACE_FILE = build_cluster_space(args.cluster_dims, args.fit_sample)

    if args.sweep:
        sweep([int(x) for x in args.sizes.split(",")])
        return
    if args.apply or args.dry:
        if args.min_cluster_size is None:
            raise SystemExit("--min-cluster-size is required; run --sweep first")
        apply(args.min_cluster_size, dry=args.dry and not args.apply,
              noise_mcs=args.noise_min_cluster_size)
        return
    raise SystemExit("choose --sweep or --apply")


if __name__ == "__main__":
    main()
