import torch, resource, sys
from tracker.ml.wp_shard_cache import ShardDataset, ShardBatchLoader
from tracker.ml.win_probability import WinProbabilityModel
bs=int(sys.argv[1])
ds=ShardDataset("/app/data/wp_shards")
loader=ShardBatchLoader(ds, list(range(1342318)), batch_size=bs, shuffle=True)
m=WinProbabilityModel(vocab_size=124, feature_dim=24, dropout=0.2).to("xpu")
opt=torch.optim.AdamW(m.parameters(), lr=1e-3); crit=torch.nn.BCEWithLogitsLoss(reduction="none")
n=0
for cid,feat,lengths,labels,mask in loader:
    logits=m(cid.to("xpu"),feat.to("xpu"),lengths)
    loss=(crit(logits,labels.to("xpu"))*mask.to("xpu")).sum()/mask.sum().clamp(min=1).to("xpu")
    loss.backward(); opt.step(); opt.zero_grad(); torch.xpu.synchronize()
    n+=1
    if n>=20: break
print(f"SURVIVED 20 steps @ batch {bs}", flush=True)
