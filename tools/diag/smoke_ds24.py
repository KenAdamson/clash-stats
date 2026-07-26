import os, numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from tracker.ml.card_metadata import CardVocabulary
from tracker.ml.sequence_dataset import SequenceDataset
e=create_engine(os.environ["DATABASE_URL"]); s=Session(e)
vocab=CardVocabulary(s)
# find corpus games where the TEAM plays rocket (rocket-cycle-ish) + have replays
rc=[r[0] for r in s.execute(text("""
  SELECT DISTINCT b.battle_id FROM battles b
  JOIN replay_events re ON re.battle_id=b.battle_id AND re.side='team' AND re.card_name='rocket'
  WHERE b.player_deck::text ILIKE '%Rocket%' LIMIT 25""")).fetchall()]
print(f"rocket-team games found: {len(rc)}")
ds=SequenceDataset(s,vocab,battle_ids=rc,extra_features=True)
print("feature_dim:", ds.feature_dim, "(expect 24)  n:", len(ds))
tower_nonzero=0; unit_nonzero=0; finite=True
for cid,feat,label in ds._samples:
    if np.any(feat[:,23]>0): tower_nonzero+=1
    if np.any(feat[:,22]>0): unit_nonzero+=1
    if not np.all(np.isfinite(feat)): finite=False
print(f"games w/ nonzero TOWER value (dim23): {tower_nonzero}/{len(ds)}")
print(f"games w/ nonzero UNIT value (dim22): {unit_nonzero}/{len(ds)}")
print(f"all-finite: {finite}")
