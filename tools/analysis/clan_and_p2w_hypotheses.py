"""Test two field observations against the corpus (2026-07-26).

H1  Clanned players out-skill unclanned -- does it survive controlling for
    engagement, or is clan tag just a proxy for cares-about-the-game?
H2  Pay-to-win is rare -- does card level buy wins, and is it "genuinely
    rarer" or just "rarer at my current arena"?

Run with cwd=/app. Reads the corpus enrichment pickle (no DB needed for H1)
and the Beelink replica for H2 so the primary stays clear of ingest.

    docker exec cr-tracker bash -c 'cd /app && python3 tools/analysis/clan_and_p2w_hypotheses.py'

TWO TRAPS THIS SCRIPT EXISTS TO AVOID
-------------------------------------
1. BOT CONTAMINATION. The #VG8/#VGG fleet is 36% of the high-battleCount
   UNCLANNED cohort and 0.0% of the clanned one. Left in, it inflates the
   top-decile clan gap from +627 to +1221 -- i.e. half that headline number
   was bots, not clans. Any clanned-vs-unclanned comparison MUST exclude them.
2. WRONG JOIN KEY. battles.id is an integer surrogate; deck_cards.battle_id
   joins to battles.BATTLE_ID (a varchar hash). Joining on id silently returns
   zero rows rather than erroring.
"""

import os
import pickle
import re
import statistics as st

BOT_PREFIX = re.compile(r"^#(VG8|VGG)")
ENRICHMENT = "data/corpus_enrichment.pkl"


def load_players(drop_bots: bool = True) -> list[tuple[str, dict]]:
    """Enriched corpus players with the fields both hypotheses need.

    Args:
        drop_bots: Exclude known bot-fleet tag prefixes. Leave True for any
            clanned-vs-unclanned comparison -- see module docstring.

    Returns:
        (tag, record) pairs where record has bc (lifetime battleCount),
        best (bestTrophies) and optionally clan.
    """
    with open(ENRICHMENT, "rb") as fh:
        raw = pickle.load(fh)
    out = []
    for tag, rec in raw.items():
        if not isinstance(rec, dict) or not rec.get("bc") or not rec.get("best"):
            continue
        if drop_bots and BOT_PREFIX.match(tag):
            continue
        out.append((tag, rec))
    return out


def h1_clan_controlling_for_engagement(deciles: int = 10) -> None:
    """Stratify by lifetime games played, then compare peak trophies by clan.

    Engagement is the confound: clan membership correlates with investment and
    invested players are better. Stratifying by battleCount holds that roughly
    constant, so a surviving gap is not merely "clanned players try harder".
    """
    players = load_players(drop_bots=True)
    players.sort(key=lambda kv: kv[1]["bc"])
    n, size = len(players), len(players) // deciles
    print(f"H1  clan effect on peak trophies, bots excluded (n={n})")
    print(f"    {'games played':<20} {'n':<7} {'clanned':<12} {'unclanned':<12} gap")
    for i in range(deciles):
        grp = players[i * size:(i + 1) * size] if i < deciles - 1 else players[i * size:]
        clan = [r["best"] for _, r in grp if r.get("clan")]
        solo = [r["best"] for _, r in grp if not r.get("clan")]
        if len(clan) < 30 or len(solo) < 30:
            continue
        lo, hi = grp[0][1]["bc"], grp[-1][1]["bc"]
        print(f"    {f'{lo}-{hi}':<20} {len(grp):<7} "
              f"{f'{st.mean(clan):.0f}/{len(clan)}':<12} "
              f"{f'{st.mean(solo):.0f}/{len(solo)}':<12} "
              f"{st.mean(clan) - st.mean(solo):+.0f}")
    clan = [r["best"] for _, r in players if r.get("clan")]
    solo = [r["best"] for _, r in players if not r.get("clan")]
    print(f"    uncontrolled gap: {st.mean(clan) - st.mean(solo):+.0f} trophies")


def h1_quantify_bot_contamination() -> None:
    """Show why drop_bots matters: bots are ~a third of high-volume unclanned."""
    players = load_players(drop_bots=False)
    players.sort(key=lambda kv: kv[1]["bc"])
    top = players[9 * (len(players) // 10):]
    clan = [(t, r) for t, r in top if r.get("clan")]
    solo = [(t, r) for t, r in top if not r.get("clan")]
    for label, grp in (("clanned", clan), ("unclanned", solo)):
        bots = [t for t, _ in grp if BOT_PREFIX.match(t)]
        print(f"    {label:<10} bot-prefix {len(bots)}/{len(grp)} = "
              f"{100 * len(bots) / max(len(grp), 1):.1f}%")
    solo_all = [r["best"] for _, r in solo]
    solo_clean = [r["best"] for t, r in solo if not BOT_PREFIX.match(t)]
    clan_all = [r["best"] for _, r in clan]
    print(f"    gap with bots:    {st.mean(clan_all) - st.mean(solo_all):+.0f}")
    print(f"    gap without bots: {st.mean(clan_all) - st.mean(solo_clean):+.0f}")


H2_SQL = """
SET statement_timeout='260s';
WITH b AS (
  SELECT battle_id, result, opponent_starting_trophies AS otr
  FROM battles
  WHERE corpus='top_ladder' AND opponent_starting_trophies IS NOT NULL
    AND result IN ('win','loss') AND battle_time > now() - interval '10 days'
  LIMIT 250000
), lv AS (
  -- NOTE: joins on battles.battle_id (varchar hash), NOT battles.id
  SELECT d.battle_id,
         avg(d.card_level) FILTER (WHERE d.is_player_deck=1) AS p_lvl,
         avg(d.card_level) FILTER (WHERE d.is_player_deck=0) AS o_lvl,
         avg(d.card_max_level) FILTER (WHERE d.is_player_deck=0) AS o_max
  FROM deck_cards d JOIN b ON b.battle_id = d.battle_id
  GROUP BY d.battle_id
)
SELECT (width_bucket(b.otr,0,13000,13)-1)*1000 AS trophy_band,
       count(*) AS battles,
       round(avg(lv.o_lvl)::numeric,2) AS avg_card_level,
       round(avg(lv.o_max - lv.o_lvl)::numeric,3) AS deficit_from_max,
       round(100.0*count(*) FILTER (WHERE b.result='win')/count(*),1) AS win_pct
FROM b JOIN lv ON lv.battle_id = b.battle_id
GROUP BY 1 HAVING count(*) > 50 ORDER BY 1;
"""

if __name__ == "__main__":
    h1_clan_controlling_for_engagement()
    print()
    print("H1  bot contamination check (why the top decile looked huge)")
    h1_quantify_bot_contamination()
    print()
    print("H2  run H2_SQL against the replica (192.168.7.62) -- see module docstring")
