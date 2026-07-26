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
ds = SequenceDataset(s, vocab, battle_ids=bids, extra_features=True)
dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=wp_collate_fn)
cid, feat, lengths, labels, mask = next(iter(dl))
print("full feat finite:", bool(torch.isfinite(feat).all()))
for d in range(feat.shape[2]):
    col = feat[:,:,d]
    if not torch.isfinite(col).all():
        print(f"  dim {d}: NaN/inf count = {int((~torch.isfinite(col)).sum())}")
# stage-by-stage
m = WinProbabilityModel.from_pretrained_tcn(f"data/ml_models/{prod.filename}", vocab.size,
      torch.device("cpu"), freeze_encoder=True, dropout=0.2, extra_feature_dim=5)
m.eval()
with torch.no_grad():
    ce = m.card_embedding(cid); print("card_emb finite:", bool(torch.isfinite(ce).all()))
    base = feat[:,:,:17]; combined = torch.cat([ce, base],2).transpose(1,2)
    print("base finite:", bool(torch.isfinite(base).all()))
    t = m.tcn(combined); print("tcn_out finite:", bool(torch.isfinite(t).all()))
