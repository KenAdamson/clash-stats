"""Detect predictive spell placements with time-of-flight physics.
A spell CONNECTS if the target spawned before the spell's impact (placement+TOF);
it is PREDICTIVE if the target also spawned at/after the throw (you couldn't react).
The valid predictive window is exactly the flight time: generous for lobbed Fireball
/Arrows, near-zero for instant Zap (so Zaps that fire into empty grass are rejected)."""
import os, sys, math
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
TILE=1000.0
SPELL_RADIUS={"fireball":2.5,"the-log":2.0,"giant-snowball":2.0,"arrows":4.0,"poison":3.5,
 "lightning":3.5,"rocket":2.0,"earthquake":3.5,"zap":2.5,"barbarian-barrel":2.0}
TOF={"zap":0.15,"giant-snowball":0.5,"the-log":0.9,"arrows":1.1,"fireball":1.0,"poison":1.0,
 "lightning":0.5,"rocket":1.2,"earthquake":0.8,"barbarian-barrel":0.6}
BUILDINGS={"cannon","tesla","tombstone","bomb-tower","inferno-tower","goblin-cage","mortar",
 "x-bow","furnace","goblin-hut","elixir-collector","barbarian-hut","goblin-drill"}
SWARM={"skeletons","skeleton-army","goblins","goblin-gang","bats","spear-goblins","minions",
 "minion-horde","fire-spirit","electro-spirit","ice-spirit","heal-spirit","princess",
 "dart-goblin","wall-breakers","goblin-barrel","firecracker"}
MEDIUM={"barbarians","elite-barbarians","musketeer","three-musketeers","witch","wizard","archers",
 "night-witch","guards","rascals","zappies","flying-machine","mega-minion","bomber","hunter",
 "magic-archer","mother-witch","skeleton-dragons"}
SPELL_TROOPS={"zap":SWARM,"giant-snowball":SWARM,"the-log":{"skeletons","skeleton-army","goblins",
 "goblin-gang","spear-goblins","princess","dart-goblin","wall-breakers","goblin-barrel","barbarians"},
 "arrows":SWARM|{"witch"},"fireball":SWARM|MEDIUM,"poison":SWARM|MEDIUM,"lightning":MEDIUM,
 "rocket":MEDIUM,"barbarian-barrel":{"skeletons","skeleton-army","goblins","goblin-gang"}}
BUILDING_SPELLS={"fireball","poison","lightning","rocket","earthquake","zap","giant-snowball","the-log","arrows"}
WINCONS={"hog-rider","ram-rider","battle-ram","royal-hogs","balloon","wall-breakers","giant","golem",
 "royal-giant","goblin-giant","electro-giant","miner","graveyard"}
STALE=3.0
def _cls(c): return "building" if c in BUILDINGS else "swarm" if c in SWARM else "troop" if c in MEDIUM else "other"
def analyze(s, bid, verbose=False):
    rows=s.execute(text("SELECT game_tick,side,card_name,arena_x,arena_y FROM replay_events "
        "WHERE battle_id=:b ORDER BY game_tick"),{"b":bid}).fetchall()
    ev=[(t/20.0,side,c,x,y) for t,side,c,x,y in rows]
    opp=[(t,c,x,y) for t,side,c,x,y in ev if side=="opponent"]
    mywc=[t for t,side,c,x,y in ev if side=="team" and c in WINCONS]
    hits=[]
    for t,side,card,x,y in ev:
        if side!="team" or card not in SPELL_RADIUS: continue
        r=SPELL_RADIUS[card]*TILE; tof=TOF.get(card,1.0)
        targets=set(SPELL_TROOPS.get(card,set()))
        if card in BUILDING_SPELLS: targets|=BUILDINGS
        for (to,oc,ox,oy) in opp:
            if oc not in targets: continue
            lead=to-t
            if not (-STALE <= lead <= tof): continue
            if math.hypot(ox-x,oy-y) > r: continue
            predictive = lead >= -0.15
            push=any(0<=(t-wt)<=4.0 for wt in mywc)
            hits.append((t,card,oc,_cls(oc),math.hypot(ox-x,oy-y)/TILE,lead,tof,predictive,push))
    if verbose:
        for t,card,oc,cls,dist,lead,tof,pred,push in hits:
            tag="PREDICTIVE" if pred else "reactive"
            print(f"  {t:6.1f}s {card:12s}->{oc:14s}[{cls:8s}] d={dist:.2f}t spawn={lead:+.2f}s "
                  f"(impact@+{tof:.2f}) {tag}{'  spell-the-pull' if push else ''}")
    return hits
if __name__=="__main__":
    e=create_engine(os.environ["DATABASE_URL"]); s=Session(e)
    analyze(s, sys.argv[1] if len(sys.argv)>1 else "ae756cf392b7d546387c7551c4022e86", verbose=True)
