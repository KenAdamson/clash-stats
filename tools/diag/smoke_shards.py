"""Equivalence: shard-loader batches vs in-memory DataLoader batches on 3K games."""
import os, numpy as np, torch
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from torch.utils.data import DataLoader, Subset
from tracker.ml.card_metadata import CardVocabulary
from tracker.ml.sequence_dataset import SequenceDataset, select_training_battles
from tracker.ml.wp_dataset import wp_collate_fn
from tracker.ml.wp_shard_cache import build_shards, ShardDataset, ShardBatchLoader

e = create_engine(os.environ["DATABASE_URL"]); s = Session(e)
vocab = CardVocabulary(s)
rows = select_training_battles(s, max_games=3000)
meta = build_shards(s, "/app/data/wp_shards_smoke", vocab=vocab, extra_features=True, battle_rows=rows)
print("meta:", {k: meta[k] for k in ("n","n_allocated","max_len","feature_dim","truncated_games")})

# in-memory reference on the SAME pinned battles
ref = SequenceDataset(s, vocab, extra_features=True, battle_rows=rows)
sh = ShardDataset("/app/data/wp_shards_smoke")
print("counts: ref", len(ref), " shard", len(sh), " ids match:", ref.battle_ids_in_order == sh.battle_ids_in_order)

# batch equivalence, first 4 sequential batches
dl = DataLoader(Subset(ref, list(range(len(ref)))), batch_size=256, shuffle=False, collate_fn=wp_collate_fn)
sl = ShardBatchLoader(sh, list(range(len(sh))), batch_size=256, shuffle=False)
ok = True
for k, ((c1,f1,l1,y1,m1),(c2,f2,l2,y2,m2)) in enumerate(zip(dl, sl)):
    L = min(c1.shape[1], c2.shape[1])
    if not torch.equal(torch.clamp(l1, max=sh.max_len), l2): ok=False; print(f"batch {k}: lengths differ beyond truncation"); break
    if not torch.equal(c1[:,:L], c2[:,:L]): ok=False; print(f"batch {k}: card_ids differ"); break
    d = (f1[:,:L]-f2[:,:L]).abs().max().item()
    if d > 2e-3: ok=False; print(f"batch {k}: features max diff {d}"); break  # fp16 tolerance
    if not torch.equal(y1[:,:L], y2[:,:L]) or not torch.equal(m1[:,:L], m2[:,:L]): ok=False; print(f"batch {k}: labels/mask differ"); break
    if k >= 3: break
print("EQUIVALENCE:", "PASS" if ok else "FAIL")
