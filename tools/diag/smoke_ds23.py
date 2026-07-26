import os, torch
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from tracker.ml.card_metadata import CardVocabulary
from tracker.ml.sequence_dataset import SequenceDataset
e=create_engine(os.environ["DATABASE_URL"]); s=Session(e)
vocab=CardVocabulary(s)
# include the mirror game (has a spell connect) + some others
bids=[r[0] for r in s.execute(text("SELECT DISTINCT b.battle_id FROM battles b "
  "JOIN replay_events re ON re.battle_id=b.battle_id WHERE b.player_tag LIKE '%VRVR9Q2QP%' LIMIT 40")).fetchall()]
bids.append("ae756cf392b7d546387c7551c4022e86")
ds=SequenceDataset(s,vocab,battle_ids=bids,extra_features=True)
print("feature_dim:", ds.feature_dim, "(expect 23)  n:", len(ds))
# check dim 22 (spell_connect_value) has nonzero somewhere + finite
import numpy as np
col22_nonzero=0; allfinite=True; total=0
for cid,feat,label in ds._samples:
    c=feat[:,22]; total+=1
    if np.any(c>0): col22_nonzero+=1
    if not np.all(np.isfinite(feat)): allfinite=False
print(f"games with nonzero spell_connect_value: {col22_nonzero}/{total}  all-finite: {allfinite}")
