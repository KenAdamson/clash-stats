"""Reproduce the from-scratch WP training on a subset, log per-batch finiteness
to pinpoint where the NaN enters (init? logits growth? a specific batch?)."""
import os
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
import torch.nn as nn
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from tracker.ml.card_metadata import CardVocabulary
from tracker.ml.sequence_dataset import SequenceDataset
from tracker.ml.wp_dataset import wp_collate_fn
from tracker.ml.win_probability import WinProbabilityModel

e = create_engine(os.environ["DATABASE_URL"])
dev = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
with Session(e) as s:
    vocab = CardVocabulary(s)
    bids = [r[0] for r in s.execute(text("""
        SELECT b.battle_id FROM battles b
        JOIN (SELECT battle_id FROM replay_events WHERE card_name!='_invalid'
              GROUP BY battle_id HAVING count(*)>=4) rc ON rc.battle_id=b.battle_id
        WHERE b.battle_type='PvP' AND b.result IN ('win','loss')
        ORDER BY random() LIMIT 20000
    """)).fetchall()]
    ds = SequenceDataset(s, vocab, battle_ids=bids, extra_features=True)

m = WinProbabilityModel(vocab_size=vocab.size, feature_dim=ds.feature_dim, dropout=0.2).to(dev)
opt = AdamW([p for p in m.parameters() if p.requires_grad], lr=5e-4, weight_decay=1e-4)
crit = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([0.817], device=dev))
dl = DataLoader(ds, batch_size=512, shuffle=True, collate_fn=wp_collate_fn)

m.train()
for bidx, (cid, feat, lengths, labels, mask) in enumerate(dl):
    cid, feat, labels, mask = cid.to(dev), feat.to(dev), labels.to(dev), mask.to(dev)
    opt.zero_grad()
    logits = m(cid, feat, lengths)
    lpt = crit(logits, labels)
    loss = (lpt * mask).sum() / mask.sum().clamp(min=1)
    lg_max = float(logits.abs().max())
    in_finite = bool(torch.isfinite(feat).all())
    print(f"batch {bidx}: in_finite={in_finite} |logit|max={lg_max:.1f} "
          f"logits_finite={bool(torch.isfinite(logits).all())} loss={float(loss):.4f}")
    if not torch.isfinite(loss):
        print(f"  >>> NaN at batch {bidx}. weights finite before step? "
              f"{all(torch.isfinite(p).all() for p in m.parameters())}")
        break
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    if bidx >= 8:
        print("survived 9 batches — divergence is slower or subset missed it")
        break
