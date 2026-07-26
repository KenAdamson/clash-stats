import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from tracker.ml.spell_value import spell_connect_values
e=create_engine(os.environ["DATABASE_URL"]); s=Session(e)
bid="ae756cf392b7d546387c7551c4022e86"
rows=s.execute(text("SELECT side,game_tick,card_name,arena_x,arena_y FROM replay_events "
  "WHERE battle_id=:b ORDER BY game_tick"),{"b":bid}).fetchall()
events=[(r[0],r[1],r[2],r[3],r[4]) for r in rows]
vals=spell_connect_values(events)
print(f"events={len(events)}  nonzero spell-connect values:")
for (side,gt,card,x,y),v in zip(events,vals):
    if v>0: print(f"  {gt/20.0:6.1f}s  after event [{side} {card}]  spell_connect_value={v} elixir")
print(f"total connect value delivered: {sum(vals)}")
