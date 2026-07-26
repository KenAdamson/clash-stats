"""Score the wp_v4 (22-dim board-truth) candidate over the main's personal PvP
games and compute calibration-by-tier — the Phase B verdict vs the frozen-76%."""
import glob
import os
import numpy as np
import torch
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from torch.utils.data import DataLoader
from tracker.ml.card_metadata import CardVocabulary
from tracker.ml.sequence_dataset import SequenceDataset
from tracker.ml.wp_dataset import wp_collate_fn
from tracker.ml.win_probability import WinProbabilityModel

ckpt = sorted(glob.glob("data/ml_models/wp_v*.pt"))[-1]
c = torch.load(ckpt, map_location="cpu", weights_only=True)
sd = c["model_state_dict"]
nan = sum(int(torch.isnan(v).sum()) for v in sd.values()
          if torch.is_tensor(v) and v.dtype.is_floating_point)
fd = c.get("feature_dim", 17)
ed = c.get("extra_feature_dim", 0)
print(f"checkpoint={ckpt} feature_dim={fd} extra_feature_dim={ed} val_acc={c.get('val_acc')} "
      f"val_loss={round(c.get('val_loss', 0), 4)} NaN_weights={nan}")

dev = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
e = create_engine(os.environ["DATABASE_URL"])
with Session(e) as s:
    vocab = CardVocabulary(s)
    rows = s.execute(text("""
        SELECT battle_id, result, opponent_starting_trophies tr
        FROM battles WHERE player_tag='#L90009GPP' AND battle_type='PvP'
          AND result IN ('win','loss') AND opponent_starting_trophies > 0
    """)).fetchall()
    meta = {r[0]: (1 if r[1] == 'win' else 0, r[2]) for r in rows}
    bids = list(meta)
    ds = SequenceDataset(s, vocab, battle_ids=bids, extra_features=(fd + ed) > 17)

m = WinProbabilityModel(vocab_size=c["vocab_size"], feature_dim=fd, extra_feature_dim=ed, dropout=0.0)
m.load_state_dict(sd); m.to(dev).eval()

# per-game max P(win)
dl = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=wp_collate_fn)
maxwp = []
order = ds.battle_ids_in_order
with torch.no_grad():
    for cid, feat, lengths, labels, mask in dl:
        logits = m(cid.to(dev), feat.to(dev), lengths)
        p = torch.sigmoid(logits).cpu().numpy()
        mk = mask.numpy().astype(bool)
        for i in range(p.shape[0]):
            vals = p[i][mk[i]]
            maxwp.append(float(vals.max()) if vals.size else 0.5)

y = np.array([meta[b][0] for b in order])
tr = np.array([meta[b][1] for b in order])
mw = np.array(maxwp)
print(f"scored {len(mw)} personal games")

def tier(t):
    return '3 11-12k' if t < 12000 else '4 12k+'
print(f"\n{'tier':<10}{'n':>6}{'actual_WR':>11}{'model_peak':>12}{'WR|peak>=.7':>13}")
for name in ['3 11-12k', '4 12k+']:
    idx = np.array([tier(t) == name for t in tr])
    if idx.sum() < 10:
        continue
    yy, mm = y[idx], mw[idx]
    conf = mm >= 0.7
    wr_conf = yy[conf].mean() if conf.sum() else float('nan')
    print(f"{name:<10}{idx.sum():>6}{yy.mean()*100:>10.0f}%{mm.mean()*100:>11.0f}%"
          f"{wr_conf*100:>12.0f}%")
print("\nBaseline (old wp_v3, from the 07-09 diagnostic): model_peak frozen ~76% "
      "both tiers; WR|confident 59% at 11-12k, 50% (coin flip) at 12k+.")
print("Phase B win = model_peak now LOWER at 12k+ than 11-12k (tier-aware), and "
      "WR|confident at 12k+ lifts above 50%.")
