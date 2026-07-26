"""Does mid-game elixir advantage (leakage-safe, first 90s) predict outcome?
Go/no-go on whether exact-elixir is worth adding to the WP model."""
import os
import numpy as np
from collections import defaultdict
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from tracker.ml.card_metadata import CardVocabulary, kebab_to_title
from tracker.ml.elixir_trace import elixir_trace

MID_TICK = 1800  # ~90s at 20Hz — well before most games are decided
e = create_engine(os.environ["DATABASE_URL"])
with Session(e) as s:
    vocab = CardVocabulary(s)
    rows = s.execute(text("""
        SELECT b.battle_id, b.result, re.side, re.game_tick, re.card_name
        FROM battles b JOIN replay_events re ON re.battle_id = b.battle_id
        WHERE b.player_tag='#L90009GPP' AND b.battle_type='PvP'
          AND b.result IN ('win','loss') AND re.card_name != '_invalid'
        ORDER BY b.battle_id, re.game_tick, re.id
    """)).fetchall()

games = defaultdict(lambda: {"ev": [], "y": None})
for bid, res, side, tick, card in rows:
    g = games[bid]; g["y"] = 1 if res == "win" else 0
    c = vocab.elixir(kebab_to_title(card))
    if c:
        g["ev"].append((side, tick, c))

adv, ys = [], []
for g in games.values():
    if not g["ev"]:
        continue
    trace, _ = elixir_trace(g["ev"])
    # time-weighted mean elixir differential over first 90s: sample each side's
    # elixir-after-play for plays with tick < MID_TICK
    tm = [a for (t, b, a, l) in trace["team"] if t < MID_TICK]
    op = [a for (t, b, a, l) in trace["opponent"] if t < MID_TICK]
    if not tm or not op:
        continue
    adv.append(np.mean(tm) - np.mean(op))
    ys.append(g["y"])

adv = np.array(adv); ys = np.array(ys)
print(f"games with mid-game elixir data: {len(adv)}")
# correlation of mid-game elixir advantage with eventual win
from scipy.stats import pointbiserialr
r, p = pointbiserialr(ys, adv)
print(f"mid-game (<90s) elixir advantage vs eventual win: point-biserial r={r:.3f}, p={p:.1e}")
# win rate by advantage tercile
import pandas as pd
df = pd.DataFrame({"adv": adv, "y": ys})
df["q"] = pd.qcut(df["adv"], 3, labels=["behind", "even", "ahead"], duplicates="drop")
print(df.groupby("q", observed=True).agg(n=("y", "size"), win_rate=("y", "mean")).to_string())
print("\nIf 'ahead' wins meaningfully more than 'behind', exact-elixir carries")
print("leakage-safe predictive signal -> worth adding as a per-tick WP feature.")
