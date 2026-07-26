import os, torch
from torch.utils.data import DataLoader
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from tracker.ml.card_metadata import CardVocabulary
from tracker.ml.sequence_dataset import SequenceDataset
from tracker.ml.wp_dataset import wp_collate_fn
from tracker.ml.win_probability import WinProbabilityModel

e = create_engine(os.environ["DATABASE_URL"])
with Session(e) as s:
    vocab = CardVocabulary(s)
    bids = [r[0] for r in s.execute(text(
        "SELECT battle_id FROM battles WHERE player_tag='#L90009GPP' "
        "AND battle_type='PvP' ORDER BY battle_time DESC LIMIT 40")).fetchall()]
    ds = SequenceDataset(s, vocab, battle_ids=bids, extra_features=True)
    m = WinProbabilityModel(vocab_size=vocab.size, feature_dim=ds.feature_dim, dropout=0.1)
    print("model feature_dim:", m.feature_dim, "input_channels:", 16 + m.feature_dim)
    dl = DataLoader(ds, batch_size=8, collate_fn=wp_collate_fn)
    cid, feat, lengths, labels, mask = next(iter(dl))
    print("batch feat shape:", tuple(feat.shape))
    logits = m(cid, feat, lengths)
    print("forward OK, logits:", tuple(logits.shape))
    loss = (torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none") * mask).sum() / mask.sum()
    loss.backward()
    gnorm = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
    print(f"backward OK, loss={float(loss):.4f}, grad_norm={gnorm:.1f}")
