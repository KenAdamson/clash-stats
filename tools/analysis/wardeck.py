import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from tracker.archetypes import classify_archetype

POOL = {  # canonical name -> your level
 "Royal Delivery":12,"Firecracker":11,"Goblin Machine":11,"Vines":11,"Tesla":10,
 "Minion Horde":10,"Earthquake":10,"Barbarian Hut":10,"Zap":10,"Bandit":10,"X-Bow":10,
 "Heal Spirit":10,"Electro Wizard":10,"Wizard":10,"Skeleton Dragons":9,"Wall Breakers":9,
 "Ice Spirit":9,"Princess":9,"Inferno Dragon":9,"Mirror":9,"Night Witch":9,"Ram Rider":9,
 "Sparky":9,"Spirit Empress":9,"Lava Hound":9,"Electro Spirit":9,"Barbarians":9,"Dark Prince":9,
 "Balloon":9,"Electro Giant":9,"Skeleton Army":9,"Poison":9,"Battle Ram":9,"Goblin Barrel":9,
 "Mortar":8,"Electro Dragon":8,"Mega Minion":8,"Hunter":8,"Golem":8,"Dart Goblin":8,
 "Royal Giant":8,"Goblin Giant":8,"Lightning":8,"Minions":8,"Berserker":8,"Baby Dragon":8,
 "Inferno Tower":8,"Elite Barbarians":8,"Freeze":8,"Ice Golem":8,"Royal Recruits":7,"Zappies":7,
 "Furnace":7,"Bomber":7,"Battle Healer":7,"Spear Goblins":7,"Royal Hogs":7,"Elixir Collector":7,
 "Rocket":7,"Three Musketeers":7,"Skeleton Barrel":7,"Goblin Cage":7,"Bomb Tower":7,"Goblin Hut":7,
 "Tombstone":7,"Goblin Gang":7,"Guards":7,"Fire Spirit":6,"Flying Machine":6,"Giant":6,"Goblins":6,
 "Valkyrie":5,"Goblin Demolisher":5,"Rascals":5,"Giant Snowball":1,
}
e=create_engine(os.environ["DATABASE_URL"]); s=Session(e)
rows=s.execute(text("""
  WITH d AS (
    SELECT b.result,
      (SELECT array_agg(c->>'name' ORDER BY c->>'name')
       FROM jsonb_array_elements(b.opponent_deck::jsonb) c) AS cards
    FROM battles b
    WHERE b.battle_type IN ('PvP','pathOfLegend') AND b.result IN ('win','loss')
      AND b.opponent_deck IS NOT NULL AND b.opponent_starting_trophies > 7000
      AND b.battle_time > now() - interval '21 days'
  )
  SELECT cards, count(*) n, count(*) FILTER (WHERE result='loss') opp_wins
  FROM d GROUP BY cards HAVING count(*) >= 40 ORDER BY n DESC LIMIT 4000
""")).fetchall()
print(f"candidate decks pulled: {len(rows)}")

BEAT={"Golem","Lava Hound","Electro Giant","Giant","Goblin Giant","Royal Giant"}
SIEGE={"X-Bow","Mortar"}
cands=[]
for cards,n,opp_wins in rows:
    cards=list(cards)
    if len(cards)!=8: continue
    owned=[c for c in cards if c in POOL]
    if len(owned)<8: continue           # fully fieldable from your pool
    lvlsum=sum(POOL[c] for c in cards)
    wr=opp_wins/n
    arch=classify_archetype([{"name":c} for c in cards])
    style = "beatdown" if BEAT&set(cards) else ("siege" if SIEGE&set(cards) else arch)
    cands.append((lvlsum,wr,n,style,arch,cards))

cands.sort(key=lambda x:(-x[0],-x[1]))
print(f"fully-ownable decks: {len(cands)}\n")
def show(label,filt,k=6):
    print(f"===== {label} =====")
    seen=0
    for lvlsum,wr,n,style,arch,cards in cands:
        if not filt(style,arch,cards): continue
        print(f"  Σlvl={lvlsum} avglvl={lvlsum/8:.1f} corpusWR={wr*100:.0f}% n={n} [{arch}]")
        print(f"     {', '.join(cards)}")
        seen+=1
        if seen>=k: break
    if not seen: print("  (none fully ownable)")
    print()
show("BEATDOWN (Golem/Lava/E-Giant/etc)", lambda st,ar,c: st=="beatdown")
show("SIEGE/CYCLE (X-Bow/Mortar)", lambda st,ar,c: st=="siege")
