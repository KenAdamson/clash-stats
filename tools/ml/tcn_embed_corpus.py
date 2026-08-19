"""Embed, project and cluster the whole corpus from a trained TCN checkpoint.

train_tcn() does this as steps 6-10 of one long function, but only reaches them
on the clean path: the tcn_v2 run ended by exhausting its NaN recoveries and
raised, so the encoder exists and none of the downstream artefacts do. This runs
those stages standalone, from whatever checkpoint is on disk.

It is not a copy of those steps. At 1.9M games the in-memory approach they use
does not fit:

  - train_tcn holds all embeddings in RAM. 1.9M x 128 float32 is ~970MB, which
    is survivable, but it then hands the whole array to UMAP.fit_transform and
    to HDBSCAN.fit_predict. The last successful fit in this project was 394,704
    points; this is 4.8x that, and both algorithms build neighbour graphs whose
    footprint grows far faster than the input.
  - So: fit the reducer and the clusterer on a SUBSAMPLE at a size already known
    to work, then transform/predict the rest in chunks. This is the standard way
    to scale both, and it is what makes the job finish at all rather than dying
    after an hour of graph construction.
  - Embeddings go to a disk memmap, so phase 1 survives a crash in phase 2 and
    the expensive part is not repeated.

Phases are separately resumable via files in the work dir; delete a file to
force that phase to recompute.

Run with cwd=/app:
  PYTHONPATH=/app/src python3 tools/ml/tcn_embed_corpus.py [--fit-sample N] [--limit N]
"""

import argparse
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sqlalchemy import text
from torch.utils.data import DataLoader

sys.path.insert(0, "/app/src")
from tracker.database import get_engine, get_session                    # noqa: E402
from tracker.ml.card_metadata import CardVocabulary                     # noqa: E402
from tracker.ml.sequence_dataset import SequenceDataset, collate_fn, MIN_EVENTS  # noqa: E402
from tracker.ml.tcn import GameEmbeddingModel                           # noqa: E402
from tracker.ml.training import (TCN_MODEL_VERSION, EMBEDDING_DIM, DROPOUT,
                                 BROKEN_REDUCER_NAN_FRACTION)  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("tcn_embed")

WORK = Path(os.environ.get("TCN_EMBED_WORK", "/app/data/tcn_embed_work"))
MODEL_DIR = Path("/app/data/ml_models")
# 300k is below the 394,704 that last fit successfully here, leaving headroom on
# a box that also runs Postgres and the scrapers.
DEFAULT_FIT_SAMPLE = int(os.environ.get("TCN_FIT_SAMPLE", "300000"))
CHUNK = 50_000


def _checkpoint_path() -> Path:
    p = MODEL_DIR / ("%s.pt" % TCN_MODEL_VERSION.replace("-", "_"))
    if not p.exists():
        raise SystemExit("no checkpoint at %s" % p)
    return p


def phase1_embed(session, limit: int | None) -> tuple[Path, list[str]]:
    """Encode every game to 128-dim, straight into a disk memmap."""
    ids_file = WORK / "battle_ids.txt"
    emb_file = WORK / "embeddings_128d.npy"
    if emb_file.exists() and ids_file.exists():
        bids = ids_file.read_text().splitlines()
        logger.info("phase 1: reusing %d embeddings from %s", len(bids), emb_file)
        return emb_file, bids

    ck_path = _checkpoint_path()
    ck = torch.load(ck_path, map_location="cpu", weights_only=True)
    bad = [k for k, v in ck["model_state_dict"].items()
           if torch.is_tensor(v) and v.dtype.is_floating_point
           and not torch.isfinite(v).all()]
    if bad:
        raise SystemExit("checkpoint %s has non-finite weights in %d tensors"
                         % (ck_path.name, len(bad)))
    logger.info("phase 1: %s epoch %s val_loss %.4f val_acc %.4f",
                ck_path.name, ck.get("epoch"), ck.get("val_loss", float("nan")),
                ck.get("val_acc", float("nan")))

    vocab = CardVocabulary(session)
    model = GameEmbeddingModel(vocab_size=ck["vocab_size"], dropout=DROPOUT,
                               embedding_dim=EMBEDDING_DIM)
    model.load_state_dict(ck["model_state_dict"])
    device = torch.device("xpu" if (hasattr(torch, "xpu") and torch.xpu.is_available())
                          else "cpu")
    model = model.to(device).eval()

    # --limit must scope the DATASET, not just the encode loop. Building all
    # 1.9M games takes ~80 minutes, so a limit that only capped the loop made a
    # smoke test as slow as the real run -- which defeats the point of having
    # one, and is how an untested script ends up launched for real.
    if limit is not None:
        scoped = [r[0] for r in session.execute(text("""
            SELECT b.battle_id
            FROM battles b
            JOIN (SELECT battle_id FROM replay_events WHERE card_name != '_invalid'
                  GROUP BY battle_id HAVING COUNT(*) >= :min_events) rc
              ON rc.battle_id = b.battle_id
            WHERE b.battle_type = 'PvP' AND b.result IN ('win','loss')
            ORDER BY b.battle_time LIMIT :n
        """), {"min_events": MIN_EVENTS, "n": limit})]
        ds = SequenceDataset(session, vocab, battle_ids=scoped)
    else:
        ds = SequenceDataset(session, vocab)
    n = len(ds) if limit is None else min(limit, len(ds))
    logger.info("phase 1: encoding %d games on %s", n, device)

    WORK.mkdir(parents=True, exist_ok=True)
    tmp = emb_file.with_suffix(".npy.tmp")
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32,
                                   shape=(n, EMBEDDING_DIM))
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate_fn,
                        num_workers=4)
    written = 0
    t0 = time.time()
    with torch.no_grad():
        for card_ids, features, lengths, labels, _di, _dv in loader:
            if written >= n:
                break
            emb, _ = model(card_ids.to(device), features.to(device), lengths.to(device))
            emb = emb.float().cpu().numpy()
            take = min(len(emb), n - written)
            mm[written:written + take] = emb[:take]
            written += take
            if written % (CHUNK * 4) < 512:
                logger.info("  %d/%d (%.0f/s)", written, n, written / (time.time() - t0))
    mm.flush(); del mm
    os.replace(tmp, emb_file)

    bids = ds.battle_ids_in_order[:written]
    ids_file.write_text("\n".join(bids))
    logger.info("phase 1: wrote %d embeddings in %.1f min", written,
                (time.time() - t0) / 60)
    return emb_file, bids


def phase2_project(emb_file: Path, fit_sample: int) -> Path:
    """UMAP 128d -> 3d: fit on a subsample, transform everything in chunks."""
    out = WORK / "embeddings_3d.npy"
    if out.exists():
        logger.info("phase 2: reusing %s", out)
        return out
    from umap import UMAP
    from tracker.ml.umap_embeddings import UMAP_3D_PARAMS

    emb = np.load(emb_file, mmap_mode="r")
    n = emb.shape[0]
    k = min(fit_sample, n)
    # Deterministic subsample: a fixed seed keeps the projection reproducible,
    # which matters because the dashboard's coordinates should not move when the
    # job is re-run.
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(n, size=k, replace=False))
    logger.info("phase 2: fitting UMAP on %d of %d points", k, n)
    t0 = time.time()
    reducer = UMAP(**UMAP_3D_PARAMS)
    reducer.fit(np.asarray(emb[idx], dtype=np.float32))
    logger.info("phase 2: fit in %.1f min; transforming %d in chunks of %d",
                (time.time() - t0) / 60, n, CHUNK)

    tmp = out.with_suffix(".npy.tmp")
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, 3))
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        mm[s:e] = reducer.transform(np.asarray(emb[s:e], dtype=np.float32))
        if (s // CHUNK) % 5 == 0:
            logger.info("  transformed %d/%d", e, n)
    mm.flush(); del mm
    os.replace(tmp, out)

    # Written to the WORK dir, never straight to model_dir. A --limit smoke run
    # fits on a couple of thousand points; publishing that over the production
    # reducer is exactly the kind of quiet damage a test should not be able to
    # do. Promotion is a separate, explicit step at the end of a full run.
    with open(WORK / "umap_3d_standalone.pkl", "wb") as f:
        pickle.dump(reducer, f)
    logger.info("phase 2: done in %.1f min; reducer -> %s",
                (time.time() - t0) / 60, WORK / "umap_3d_standalone.pkl")
    return out


def phase3_cluster(emb_file: Path, fit_sample: int) -> Path:
    """HDBSCAN on the 128-dim space: fit on a subsample, approximate the rest."""
    out = WORK / "cluster_ids.npy"
    if out.exists():
        logger.info("phase 3: reusing %s", out)
        return out
    import hdbscan
    from tracker.ml.clustering import HDBSCAN_PARAMS

    emb = np.load(emb_file, mmap_mode="r")
    n = emb.shape[0]
    k = min(fit_sample, n)
    rng = np.random.default_rng(43)
    idx = np.sort(rng.choice(n, size=k, replace=False))
    logger.info("phase 3: fitting HDBSCAN on %d of %d points", k, n)
    t0 = time.time()
    clusterer = hdbscan.HDBSCAN(prediction_data=True, core_dist_n_jobs=4,
                                **HDBSCAN_PARAMS)
    clusterer.fit(np.asarray(emb[idx], dtype=np.float64))
    n_fit_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
    logger.info("phase 3: fit in %.1f min, %d clusters on the sample",
                (time.time() - t0) / 60, n_fit_clusters)

    labels = np.empty(n, dtype=np.int32)
    labels[idx] = clusterer.labels_
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    rest = np.flatnonzero(mask)
    for s in range(0, len(rest), CHUNK):
        sl = rest[s:s + CHUNK]
        lab, _ = hdbscan.approximate_predict(
            clusterer, np.asarray(emb[sl], dtype=np.float64))
        labels[sl] = lab
        if (s // CHUNK) % 5 == 0:
            logger.info("  predicted %d/%d", s + len(sl), len(rest))

    noise = int((labels == -1).sum())
    logger.info("phase 3: %d clusters, %d noise (%.1f%%)",
                len(set(labels.tolist())) - (1 if -1 in labels else 0),
                noise, 100.0 * noise / n)
    np.save(out, labels)
    return out


def phase4_store(session, bids, emb_file, emb3d_file, clusters_file) -> int:
    """Upsert into game_embeddings, stamped with the current model version."""
    from tracker.ml.storage import to_blob

    emb = np.load(emb_file, mmap_mode="r")
    e3 = np.load(emb3d_file, mmap_mode="r")
    cl = np.load(clusters_file, mmap_mode="r")
    n = min(len(bids), emb.shape[0])
    logger.info("phase 4: storing %d rows as %s", n, TCN_MODEL_VERSION)

    sql = text("""
        INSERT INTO game_embeddings
            (battle_id, embedding_15d, embedding_3d, embedding_tcn_128d,
             embedding_vec_3d, cluster_id, model_version)
        VALUES (:bid, :b15, :b3, :v128, :v3, :cid, :ver)
        ON CONFLICT (battle_id) DO UPDATE SET
            embedding_15d = EXCLUDED.embedding_15d,
            embedding_3d = EXCLUDED.embedding_3d,
            embedding_tcn_128d = EXCLUDED.embedding_tcn_128d,
            embedding_vec_3d = EXCLUDED.embedding_vec_3d,
            cluster_id = EXCLUDED.cluster_id,
            model_version = EXCLUDED.model_version
    """)
    done = 0
    skipped = 0
    t0 = time.time()
    for s in range(0, n, 5000):
        e = min(s + 5000, n)
        rows = []
        for i in range(s, e):
            v128 = np.asarray(emb[i])
            v3 = np.asarray(e3[i])
            # pgvector rejects NaN outright, so a degenerate projection aborts
            # the whole write rather than losing one row. UMAP.transform yields
            # a handful of these on inputs it cannot place (measured: 474 of
            # 1.9M, 0.025%); the encoder output itself is clean. The existing
            # convention in training.py is to skip individually below
            # BROKEN_REDUCER_NAN_FRACTION and only abort if the reducer looks
            # wholesale broken, which is what this reproduces.
            if not (np.isfinite(v128).all() and np.isfinite(v3).all()):
                skipped += 1
                continue
            rows.append({
                "bid": bids[i],
                "b15": to_blob(v128),
                "b3": to_blob(v3),
                "v128": v128.tolist(),
                "v3": v3.tolist(),
                "cid": int(cl[i]) if cl[i] >= 0 else None,
                "ver": TCN_MODEL_VERSION,
            })
        if not rows:
            continue
        session.execute(sql, rows)
        session.commit()
        done += len(rows)
        if (s // 5000) % 10 == 0:
            logger.info("  stored %d/%d (%.0f rows/s)", done, n,
                        done / max(time.time() - t0, 1e-9))
    frac = skipped / max(n, 1)
    logger.info("phase 4: stored %d rows in %.1f min (%d skipped, %.3f%%, non-finite)",
                done, (time.time() - t0) / 60, skipped, 100.0 * frac)
    if frac > BROKEN_REDUCER_NAN_FRACTION:
        raise RuntimeError(
            "%.1f%% of rows were non-finite — that is a broken reducer, not "
            "degenerate inputs. Refit phase 2 before trusting this." % (100.0 * frac))
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-sample", type=int, default=DEFAULT_FIT_SAMPLE)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap games (smoke tests); implies no DB write")
    ap.add_argument("--store", action="store_true",
                    help="force the DB write even under --limit")
    args = ap.parse_args()

    session = get_session(get_engine(os.environ["DATABASE_URL"]))
    WORK.mkdir(parents=True, exist_ok=True)

    emb_file, bids = phase1_embed(session, args.limit)
    emb3d = phase2_project(emb_file, args.fit_sample)
    clusters = phase3_cluster(emb_file, args.fit_sample)
    if args.limit is not None and not args.store:
        logger.warning("--limit set without --store: skipping the database write. "
                       "A partial run must not overwrite production embeddings "
                       "with rows derived from an unrepresentative fit.")
        logger.info("DONE (dry)")
        return
    phase4_store(session, bids, emb_file, emb3d, clusters)
    if args.limit is None:
        import shutil
        shutil.copy2(WORK / "umap_3d_standalone.pkl",
                     MODEL_DIR / "umap_3d_standalone.pkl")
        logger.info("promoted reducer to %s", MODEL_DIR / "umap_3d_standalone.pkl")
    logger.info("DONE")


if __name__ == "__main__":
    main()
