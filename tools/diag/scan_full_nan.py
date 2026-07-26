"""Build the FULL WP training dataset and locate the non-finite source."""
import os
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from tracker.ml.card_metadata import CardVocabulary
from tracker.ml.sequence_dataset import SequenceDataset

e = create_engine(os.environ["DATABASE_URL"])
with Session(e) as s:
    vocab = CardVocabulary(s)
    ds = SequenceDataset(s, vocab, extra_features=True)  # full 378K, exactly as training

n_bad_base = 0
n_bad_extra = 0
extra_col_bad = np.zeros(5, dtype=int)
base_col_bad = np.zeros(17, dtype=int)
bad_ids = []
for i in range(len(ds)):
    _, feat, _ = ds[i]
    f = feat.numpy() if hasattr(feat, "numpy") else np.asarray(feat)
    base_nf = ~np.isfinite(f[:, :17])
    extra_nf = ~np.isfinite(f[:, 17:22])
    if base_nf.any():
        n_bad_base += 1
        base_col_bad += base_nf.any(axis=0).astype(int)
    if extra_nf.any():
        n_bad_extra += 1
        extra_col_bad += extra_nf.any(axis=0).astype(int)
    if (base_nf.any() or extra_nf.any()) and len(bad_ids) < 5:
        bad_ids.append(ds.battle_ids_in_order[i])

print(f"\nTOTAL games: {len(ds)}")
print(f"games w/ non-finite BASE (0:17):  {n_bad_base}  cols={base_col_bad.tolist()}")
print(f"games w/ non-finite EXTRA (17:22): {n_bad_extra}  cols={dict(zip(['own','opp','diff','gap','eff'], extra_col_bad.tolist()))}")
print(f"sample bad battle_ids: {bad_ids}")
