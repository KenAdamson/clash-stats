"""Validate analytic elixir physics + empirically DERIVE the replay tick rate
by fitting computed leak to RoyaleAPI's ground-truth elixirLeaked."""
import os
import numpy as np
from collections import defaultdict
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from tracker.ml.card_metadata import CardVocabulary, kebab_to_title
from tracker.ml.elixir_trace import elixir_trace

e = create_engine(os.environ["DATABASE_URL"])
with Session(e) as s:
    vocab = CardVocabulary(s)
    rows = s.execute(text("""
        SELECT b.battle_id, b.player_elixir_leaked, b.opponent_elixir_leaked,
               re.side, re.game_tick, re.card_name
        FROM battles b JOIN replay_events re ON re.battle_id = b.battle_id
        WHERE b.player_tag = '#L90009GPP' AND b.battle_type = 'PvP'
          AND b.player_elixir_leaked IS NOT NULL
          AND re.card_name != '_invalid'
        ORDER BY b.battle_id, re.game_tick, re.id
    """)).fetchall()

games = defaultdict(lambda: {"events": [], "team_leak": None, "opp_leak": None})
n_cost_hits = 0; n_events = 0
for bid, pl, ol, side, tick, card in rows:
    g = games[bid]
    g["team_leak"] = pl; g["opp_leak"] = ol
    cost = vocab.elixir(kebab_to_title(card))
    n_events += 1
    if cost:
        n_cost_hits += 1
        g["events"].append((side, tick, cost))
print(f"games: {len(games)} | events: {n_events} | cost-resolved: {n_cost_hits} ({100*n_cost_hits/max(n_events,1):.0f}%)")

def run(start, tps, ae):
    pred, truth = [], []
    for g in games.values():
        if not g["events"]:
            continue
        _, leak = elixir_trace(g["events"], start_elixir=start, ticks_per_sec=tps, accrue_to_end=ae)
        pred += [leak["team"], leak["opponent"]]
        truth += [g["team_leak"] or 0.0, g["opp_leak"] or 0.0]
    pred = np.array(pred); truth = np.array(truth)
    if len(pred) < 3:
        return None
    return np.mean(np.abs(pred - truth)), np.mean(pred - truth), np.corrcoef(pred, truth)[0, 1], pred.mean(), truth.mean()

print(f"\nEnd-accrual test at 20Hz (truth_mean = RoyaleAPI ground truth):")
print(f"{'tps':>6}{'end':>5}{'start':>6}{'MAE':>8}{'bias':>8}{'corr':>8}{'pred_mn':>9}{'truth_mn':>9}")
best = None
for tps in [20.0, 20.5]:
    for ae in [True, False]:
        for start in [5, 6, 7]:
            r = run(start, tps, ae)
            if r is None: continue
            mae, bias, corr, pm, tm = r
            print(f"{tps:>6.1f}{str(ae)[0]:>5}{start:>6}{mae:>8.2f}{bias:>8.2f}{corr:>8.3f}{pm:>9.2f}{tm:>9.2f}")
            if best is None or mae < best[0]:
                best = (mae, tps, start, corr)
print(f"\nBEST: tps={best[1]} start={best[2]} → MAE {best[0]:.2f}, corr {best[3]:.3f}")
print("Corr high + MAE low + bias~0 ⇒ physics + tick-rate VALIDATED against independent ground truth.")
