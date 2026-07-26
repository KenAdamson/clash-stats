import torch, resource, time
from tracker.ml.win_probability import WinProbabilityModel
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024
# Variant B: 2x width + embed 32
m=WinProbabilityModel(vocab_size=124, feature_dim=24, extra_feature_dim=0, dropout=0.2,
                      tcn_channels=[128,128,256,256,512,512], card_embed_dim=32).to("xpu")
nparams=sum(p.numel() for p in m.parameters())
print(f"variant B params: {nparams/1e6:.2f}M", flush=True)
opt=torch.optim.AdamW(m.parameters(), lr=6e-4); crit=torch.nn.BCEWithLogitsLoss(reduction="none")
bs=512
t0=time.time()
try:
    for i in range(20):
        cid=torch.randint(0,124,(bs,192),device="xpu"); feat=torch.randn(bs,192,24,device="xpu")
        lengths=torch.full((bs,),192,dtype=torch.long)
        logits=m(cid,feat,lengths); loss=crit(logits,torch.ones_like(logits)).mean()
        loss.backward(); opt.step(); opt.zero_grad()
    torch.xpu.synchronize()
    dt=time.time()-t0
    print(f"B @ batch 512: SURVIVED 20 steps — {dt/20*1000:.0f} ms/step (concurrent w/ data-iso), host_RSS={rss():.1f}GB, xpu_alloc={torch.xpu.memory_allocated()/1e9:.2f}GB", flush=True)
except Exception as e:
    print(f"B @ batch 512: FAILED {type(e).__name__}: {str(e)[:100]}", flush=True)
