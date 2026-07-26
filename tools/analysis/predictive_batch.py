import os
from collections import Counter
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from predictive_placement import analyze
e=create_engine(os.environ["DATABASE_URL"]); s=Session(e)
bids=[r[0] for r in s.execute(text("SELECT DISTINCT b.battle_id FROM battles b "
  "JOIN replay_events re ON re.battle_id=b.battle_id WHERE b.player_tag LIKE '%VRVR9Q2QP%'")).fetchall()]
connects=0; pred=[]; pcls=Counter(); pspell=Counter(); ppull=0; games=set()
for bid in bids:
    for h in analyze(s,bid):
        t,card,oc,cls,dist,lead,tof,predictive,push=h
        connects+=1
        if predictive:
            pred.append((bid,)+h); pcls[cls]+=1; pspell[card]+=1; games.add(bid)
            if push: ppull+=1
print(f"scanned {len(bids)} games")
print(f"spell CONNECTS (value events, any timing): {connects}")
print(f"PREDICTIVE reads (spawned during flight): {len(pred)} across {len(games)} games")
print(f"  by class: {dict(pcls)}")
print(f"  by spell: {dict(pspell)}")
print(f"  spell-the-pull: {ppull}")
print("\nTightest predictive reads (dist, sorted):")
for bid,t,card,oc,cls,dist,lead,tof,predictive,push in sorted(pred,key=lambda x:x[4])[:12]:
    print(f"  {bid[:10]} {t:6.1f}s {card:10s}->{oc:13s}[{cls}] d={dist:.2f}t spawn=+{lead:.2f}s/{tof:.2f}{' pull' if push else ''}")
