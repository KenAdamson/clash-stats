import torch, time, resource, sys, traceback
from tracker.ml.wp_shard_cache import ShardDataset, ShardBatchLoader
from tracker.ml.win_probability import WinProbabilityModel
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024
ds=ShardDataset("/app/data/wp_shards")
print(f"shard opened, n={len(ds)}", flush=True)
loader=ShardBatchLoader(ds, list(range(1342318)), batch_size=2048, shuffle=True)
m=WinProbabilityModel(vocab_size=124, feature_dim=24, dropout=0.2).to("xpu")
opt=torch.optim.AdamW(m.parameters(), lr=1e-3)
crit=torch.nn.BCEWithLogitsLoss(reduction="none")
n=0
try:
    for cid,feat,lengths,labels,mask in loader:
        print(f"  batch {n}: shapes cid={tuple(cid.shape)} feat={tuple(feat.shape)} host_RSS={rss():.1f}GB", flush=True)
        cid=cid.to("xpu"); feat=feat.to("xpu")
        logits=m(cid,feat,lengths)
        loss=(crit(logits,labels.to("xpu"))*mask.to("xpu")).sum()/mask.sum().clamp(min=1).to("xpu")
        loss.backward(); opt.step(); opt.zero_grad(); torch.xpu.synchronize()
        print(f"    step {n} done, xpu_alloc={torch.xpu.memory_allocated()/1e9:.2f}GB", flush=True)
        n+=1
        if n>=6: break
    print(f"SUCCESS: {n} steps @2048", flush=True)
except Exception as e:
    print(f"EXCEPTION at batch {n}: {type(e).__name__}: {e}", flush=True); traceback.print_exc()
