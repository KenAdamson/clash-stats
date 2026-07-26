import os, torch
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from tracker.ml.card_metadata import CardVocabulary
from tracker.ml.sequence_dataset import SequenceDataset
from tracker.ml.wp_dataset import wp_collate_fn
from tracker.ml.win_probability import WinProbabilityModel
from tracker.ml.model_registry import get_production
from torch.utils.data import DataLoader
e = create_engine(os.environ["DATABASE_URL"]); s = Session(e)
prod = get_production(s, "wp"); vocab = CardVocabulary(s)
bids = [r[0] for r in s.execute(text(
  "SELECT DISTINCT b.battle_id FROM battles b JOIN replay_events re ON re.battle_id=b.battle_id "
  "WHERE b.player_tag='#L90009GPP' AND b.battle_type='PvP' AND b.result IN ('win','loss') "
  "AND b.opponent_starting_trophies>0 LIMIT 40")).fetchall()]
m = WinProbabilityModel.from_pretrained_tcn(f"data/ml_models/{prod.filename}", vocab.size,
      torch.device("cpu"), freeze_encoder=True, dropout=0.2, extra_feature_dim=5)
ds = SequenceDataset(s, vocab, battle_ids=bids, extra_features=True)
dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=wp_collate_fn)
cid, feat, lengths, labels, mask = next(iter(dl))
with torch.no_grad(): out = m(cid, feat, lengths)
print(f"n={len(ds)} feat={tuple(feat.shape)} logits={tuple(out.shape)} finite={bool(torch.isfinite(out).all())} p_range=({torch.sigmoid(out).min():.3f},{torch.sigmoid(out).max():.3f})")
