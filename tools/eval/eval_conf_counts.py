"""Same calibration-by-tier, but reporting CONFIDENT-SUBSET SIZES so we can tell
a real signal from a small-sample artifact."""
import glob, os
import numpy as np, torch
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from torch.utils.data import DataLoader
from tracker.ml.card_metadata import CardVocabulary
from tracker.ml.sequence_dataset import SequenceDataset
from tracker.ml.wp_dataset import wp_collate_fn
from tracker.ml.win_probability import WinProbabilityModel

ckpt = os.environ.get("WP_EVAL_CKPT") or sorted(glob.glob("data/ml_models/wp_v*.pt"))[-1]
c = torch.load(ckpt, map_location="cpu", weights_only=True)
fd, ed = c.get("feature_dim",17), c.get("extra_feature_dim",0)
print(f"checkpoint={ckpt} feature_dim={fd} extra={ed} val_acc={round(c.get('val_acc',0),4)}")
dev = torch.device("xpu" if hasattr(torch,"xpu") and torch.xpu.is_available() else "cpu")
e = create_engine(os.environ["DATABASE_URL"])
with Session(e) as s:
    vocab = CardVocabulary(s)
    rows = s.execute(text("""
        SELECT battle_id, result, opponent_starting_trophies tr
        FROM battles WHERE player_tag='#L90009GPP' AND battle_type='PvP'
          AND result IN ('win','loss') AND opponent_starting_trophies > 0""")).fetchall()
    meta = {r[0]:(1 if r[1]=='win' else 0, r[2]) for r in rows}
    ds = SequenceDataset(s, vocab, battle_ids=list(meta), extra_features=(fd+ed)>17)
m = WinProbabilityModel(vocab_size=c["vocab_size"], feature_dim=fd, extra_feature_dim=ed, dropout=0.0, tcn_channels=c.get("tcn_channels"), card_embed_dim=c.get("card_embed_dim",16))
m.load_state_dict(c["model_state_dict"]); m.to(dev).eval()
dl = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=wp_collate_fn)
maxwp=[]
with torch.no_grad():
    for cid,feat,lengths,labels,mask in dl:
        p = torch.sigmoid(m(cid.to(dev),feat.to(dev),lengths)).cpu().numpy()
        mk = mask.numpy().astype(bool)
        for i in range(p.shape[0]):
            v = p[i][mk[i]]; maxwp.append(float(v.max()) if v.size else 0.5)
order = ds.battle_ids_in_order
y = np.array([meta[b][0] for b in order]); tr = np.array([meta[b][1] for b in order]); mw = np.array(maxwp)
print(f"scored {len(mw)} games\n")
print(f"{'tier':<10}{'n':>6}{'actualWR':>10}{'meanPeak':>10}{'n>=.7':>8}{'cover':>8}{'WR|>=.7':>9}{'n>=.6':>8}{'WR|>=.6':>9}")
for name,lo,hi in [("11-12k",0,12000),("12k+",12000,99999)]:
    idx = (tr>=lo)&(tr<hi)
    if idx.sum()<10: continue
    yy,mm = y[idx],mw[idx]
    c7 = mm>=0.7; c6 = mm>=0.6
    print(f"{name:<10}{idx.sum():>6}{yy.mean()*100:>9.0f}%{mm.mean()*100:>9.0f}%{c7.sum():>8}"
          f"{c7.mean()*100:>7.0f}%{(yy[c7].mean()*100 if c7.sum() else float('nan')):>8.0f}%"
          f"{c6.sum():>8}{(yy[c6].mean()*100 if c6.sum() else float('nan')):>8.0f}%")
