import os, numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from tracker.ml.card_metadata import CardVocabulary
from tracker.ml.sequence_dataset import SequenceDataset
e=create_engine(os.environ["DATABASE_URL"]); s=Session(e)
vocab=CardVocabulary(s)
bids=[r[0] for r in s.execute(text(
  "SELECT DISTINCT b.battle_id FROM battles b JOIN replay_events re ON re.battle_id=b.battle_id "
  "WHERE b.battle_type='pathOfLegend' AND b.result IN ('win','loss') LIMIT 40")).fetchall()]
ds=SequenceDataset(s,vocab,battle_ids=bids,extra_features=True)
print("pathOfLegend smoke: feature_dim=",ds.feature_dim," n=",len(ds))
if len(ds):
    print("all-finite:", all(np.all(np.isfinite(f)) for _,f,_ in ds._samples))
    print("nonzero spell-tower(dim23) games:", sum(1 for _,f,_ in ds._samples if np.any(f[:,23]>0)))
