"""Scan a corpus sample's extra features for non-finite values (the epoch-1 NaN)."""
import os
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from tracker.ml.card_metadata import CardVocabulary
from tracker.ml.sequence_dataset import SequenceDataset

e = create_engine(os.environ["DATABASE_URL"])
with Session(e) as s:
    vocab = CardVocabulary(s)
    # corpus games (NOT personal) — the ones the full train adds
    bids = [r[0] for r in s.execute(text("""
        SELECT b.battle_id FROM battles b
        JOIN (SELECT battle_id FROM replay_events WHERE card_name!='_invalid'
              GROUP BY battle_id HAVING count(*)>=4) rc ON rc.battle_id=b.battle_id
        WHERE b.battle_type='PvP' AND b.result IN ('win','loss')
          AND b.corpus='top_ladder'
        ORDER BY random() LIMIT 3000
    """)).fetchall()]
    ds = SequenceDataset(s, vocab, battle_ids=bids, extra_features=True)

col_names = ["own_elx", "opp_elx", "diff", "trophy_gap", "opp_eff"]
bad_games = 0
colmax = np.zeros(5); colmin = np.zeros(5); nonfinite = np.zeros(5, dtype=int)
for i in range(len(ds)):
    _, feat, _ = ds[i]
    f = feat.numpy() if hasattr(feat, "numpy") else np.asarray(feat)
    ex = f[:, 17:22]
    if not np.isfinite(ex).all():
        bad_games += 1
        nf = ~np.isfinite(ex)
        nonfinite += nf.any(axis=0).astype(int)
        if bad_games <= 3:
            cols = [col_names[c] for c in range(5) if nf[:, c].any()]
            print(f"  non-finite in {ds.battle_ids_in_order[i]}: cols={cols}")
    else:
        colmax = np.maximum(colmax, ex.max(axis=0))
        colmin = np.minimum(colmin, ex.min(axis=0))

print(f"\nscanned {len(ds)} corpus games | games with non-finite extra feats: {bad_games}")
print(f"non-finite count per col: {dict(zip(col_names, nonfinite.tolist()))}")
print(f"finite ranges: " + ", ".join(f"{n}[{lo:.2f},{hi:.2f}]" for n, lo, hi in zip(col_names, colmin, colmax)))
