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
prod = get_production(s, "wp"); print("production wp:", prod.filename)
vocab = CardVocabulary(s)
bids = [r[0] for r in s.execute(text(
  "SELECT battle_id FROM battles WHERE player_tag='#L90009GPP' AND battle_type='PvP' "
  "AND result IN ('win','loss') AND opponent_starting_trophies>0 LIMIT 30")).fetchall()]
m = WinProbabilityModel.from_pretrained_tcn(f"data/ml_models/{prod.filename}", vocab.size,
      torch.device("cpu"), freeze_encoder=True, dropout=0.2, extra_feature_dim=5)
enc_frozen = all(not p.requires_grad for n,p in m.named_parameters() if n.startswith(('card_embedding','tcn')))
head_train = sum(p.numel() for n,p in m.named_parameters() if n.startswith('head') and p.requires_grad)
print(f"encoder frozen={enc_frozen}  head trainable={head_train}  head[0].in_channels={m.head[0].in_channels} (expect 261)")
ds = SequenceDataset(s, vocab, battle_ids=bids, extra_features=True)
print("dataset feature_dim:", ds.feature_dim, " n:", len(ds))
dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=wp_collate_fn)
cid, feat, lengths, labels, mask = next(iter(dl))
with torch.no_grad(): out = m(cid, feat, lengths)
print("feat shape:", tuple(feat.shape), "logits:", tuple(out.shape), "finite:", bool(torch.isfinite(out).all()))
