"""Formatted terminal output for all analytics views."""

import shutil

from sqlalchemy.orm import Session

from tracker import analytics


def print_overall_stats(session: Session) -> None:
    """Print comprehensive stats report."""
    # All-time from API
    api_stats = analytics.get_all_time_api_stats(session)
    if api_stats:
        print()
        print("=" * 70)
        print("ALL-TIME STATS (from Supercell)")
        print("=" * 70)
        print()
        print(f"  Player:        {api_stats.get('name')} ({api_stats.get('player_tag')})")
        print(f"  Clan:          {api_stats.get('clan_name') or 'None'}")
        print(f"  Trophies:      {api_stats.get('trophies'):,} (Best: {api_stats.get('best_trophies'):,})")
        print()

        wins = api_stats.get("wins", 0)
        losses = api_stats.get("losses", 0)
        battles = api_stats.get("battle_count", 0)
        three_crowns = api_stats.get("three_crown_wins", 0)

        win_rate = (wins / battles * 100) if battles > 0 else 0
        three_crown_rate = (three_crowns / wins * 100) if wins > 0 else 0

        print(f"  Battles:       {battles:,}")
        print(f"  Wins:          {wins:,} ({win_rate:.1f}%)")
        print(f"  Losses:        {losses:,}")
        print(f"  3-Crown Wins:  {three_crowns:,} ({three_crown_rate:.1f}% of wins)")
        print(f"  War Day Wins:  {api_stats.get('war_day_wins', 0):,}")

    # Tracked battles
    stats = analytics.get_overall_stats(session)
    total = stats.get("total", 0)

    print()
    print("=" * 70)
    print("TRACKED BATTLES (our SQLite database)")
    print("=" * 70)
    print()

    if total == 0:
        print("  No battles tracked yet. Run with --fetch to start!")
        return

    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    draws = stats.get("draws", 0)
    three_crowns = stats.get("three_crowns", 0)

    win_rate = (wins / total * 100) if total > 0 else 0
    three_crown_rate = (three_crowns / wins * 100) if wins > 0 else 0

    print(f"  Total Tracked:   {total:,}")
    print(f"  Date Range:      {stats.get('first_battle', 'N/A')[:10]} to {stats.get('last_battle', 'N/A')[:10]}")
    print()
    print(f"  Wins:            {wins:,} ({win_rate:.1f}%)")
    print(f"  Losses:          {losses:,} ({losses / total * 100:.1f}%)")
    print(f"  Draws:           {draws:,} ({draws / total * 100:.1f}%)")
    print()
    print(f"  3-Crown Wins:    {three_crowns:,} ({three_crown_rate:.1f}% of wins)")
    print(f"  Avg Crowns:      {stats.get('avg_crowns', 0):.2f} (opponent: {stats.get('avg_crowns_against', 0):.2f})")

    # By battle type
    type_stats = analytics.get_stats_by_battle_type(session)
    if type_stats:
        print()
        print("  BY MODE:")
        for ts in type_stats:
            print(f"    {ts['battle_type']:28} {ts['total']:4} games  {ts['win_rate']:5.1f}% WR")

    # Time of day analysis
    time_stats = analytics.get_time_of_day_stats(session)
    if time_stats and len(time_stats) >= 3:
        print()
        print("  BY TIME OF DAY (UTC):")
        sorted_times = sorted(time_stats, key=lambda x: x["win_rate"], reverse=True)
        best = sorted_times[0]
        worst = sorted_times[-1]
        print(f"    Best Hour:   {best['hour']:02d}:00 ({best['win_rate']:.1f}% WR, {best['total']} games)")
        print(f"    Worst Hour:  {worst['hour']:02d}:00 ({worst['win_rate']:.1f}% WR, {worst['total']} games)")
    print()


def print_deck_stats(session: Session) -> None:
    """Print per-deck statistics."""
    deck_stats = analytics.get_deck_stats(session, min_battles=3)

    print()
    print("=" * 70)
    print("DECK PERFORMANCE (min 3 battles)")
    print("=" * 70)
    print()

    if not deck_stats:
        print("  Not enough data per deck yet. Keep playing!")
        return

    for i, ds in enumerate(deck_stats[:10], 1):
        cards = ", ".join(ds["deck_cards"])
        print(f"  Deck #{i}")
        print(f"    Cards:     {cards}")
        print(f"    Battles:   {ds['total']}  |  Win Rate: {ds['win_rate']}%  |  3-Crowns: {ds['three_crowns']}")
        print()


def print_crown_distribution(session: Session) -> None:
    """Print crown distribution analysis."""
    dist = analytics.get_crown_distribution(session)

    print()
    print("=" * 70)
    print("CROWN DISTRIBUTION")
    print("=" * 70)
    print()

    print("  WINS by crowns earned:")
    total_wins = sum(dist["win"].values()) if dist["win"] else 0
    for crowns in [1, 2, 3]:
        count = dist["win"].get(crowns, 0)
        pct = (count / total_wins * 100) if total_wins > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"    {crowns}-crown: {count:4} ({pct:5.1f}%) {bar}")
    print()

    print("  LOSSES by crowns earned:")
    total_losses = sum(dist["loss"].values()) if dist["loss"] else 0
    for crowns in [0, 1, 2]:
        count = dist["loss"].get(crowns, 0)
        pct = (count / total_losses * 100) if total_losses > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"    {crowns}-crown: {count:4} ({pct:5.1f}%) {bar}")
    print()


def print_matchup_stats(session: Session) -> None:
    """Print card matchup analysis."""
    matchups = analytics.get_card_matchup_stats(session, min_battles=5)

    print()
    print("=" * 70)
    print("CARD MATCHUP ANALYSIS (min 5 games)")
    print("=" * 70)
    print()

    if not matchups:
        print("  Not enough data yet.")
        return

    problem_cards = sorted(matchups, key=lambda x: x["win_rate"])[:10]
    print("  YOUR TROUBLE CARDS (lowest WR against):")
    for m in problem_cards:
        print(f"    {m['card_name']:22} {m['times_faced']:3}x  →  {m['win_rate']:5.1f}% WR")
    print()

    best_matchups = sorted(matchups, key=lambda x: -x["win_rate"])[:10]
    print("  YOUR BEST MATCHUPS (highest WR against):")
    for m in best_matchups:
        print(f"    {m['card_name']:22} {m['times_faced']:3}x  →  {m['win_rate']:5.1f}% WR")
    print()


def print_recent_battles(session: Session, limit: int = 10) -> None:
    """Print last N battles."""
    battles = analytics.get_recent_battles(session, limit)

    print()
    print("=" * 70)
    print(f"LAST {len(battles)} BATTLES")
    print("=" * 70)
    print()

    if not battles:
        print("  No battles tracked yet.")
        return

    for b in battles:
        result_icon = "✓" if b["result"] == "win" else ("✗" if b["result"] == "loss" else "—")
        trophy_change = b["player_trophy_change"] or 0
        trophy_str = f"+{trophy_change}" if trophy_change >= 0 else str(trophy_change)
        print(f"  {result_icon} {b['player_crowns']}-{b['opponent_crowns']} vs {b['opponent_name']:15} | {trophy_str:>4} | {b['battle_type']}")
    print()


def print_streaks(session: Session) -> None:
    """Print win/loss streak analysis."""
    data = analytics.get_streaks(session)

    print()
    print("=" * 70)
    print("STREAK ANALYSIS")
    print("=" * 70)
    print()

    if not data["current_streak"]:
        print("  No battles tracked yet.")
        return

    cs = data["current_streak"]
    icon = "🔥" if cs["type"] == "win" else "❄️"
    label = f"{cs['type']}es" if cs["type"] == "loss" else f"{cs['type']}s"
    if cs["length"] == 1:
        label = cs["type"]
    print(f"  Current:       {icon} {cs['length']} {label} ({cs['start_trophies']} → {cs['end_trophies']})")

    if data["longest_win_streak"]:
        ws = data["longest_win_streak"]
        print(f"  Best Win Run:  {ws['length']} wins ({ws['start_trophies']} → {ws['end_trophies']})")

    if data["longest_loss_streak"]:
        ls = data["longest_loss_streak"]
        print(f"  Worst Tilt:    {ls['length']} losses ({ls['start_trophies']} → {ls['end_trophies']})")

    streaks = data["streaks"]
    win_streaks = sorted([s for s in streaks if s["type"] == "win"], key=lambda s: -s["length"])
    loss_streaks = sorted([s for s in streaks if s["type"] == "loss"], key=lambda s: -s["length"])

    if len(win_streaks) > 1:
        print()
        print("  TOP WIN STREAKS:")
        for s in win_streaks[:5]:
            print(f"    {s['length']} wins  {s['start_trophies']} → {s['end_trophies']}")

    if len(loss_streaks) > 1:
        print()
        print("  WORST LOSS STREAKS:")
        for s in loss_streaks[:5]:
            print(f"    {s['length']} losses  {s['start_trophies']} → {s['end_trophies']}")
    print()


def print_rolling_stats(session: Session, window: int = 35) -> None:
    """Print rolling window stats for last N games."""
    stats = analytics.get_rolling_stats(session, window)

    print()
    print("=" * 70)
    print(f"LAST {stats['total']} GAMES (rolling window: {window})")
    print("=" * 70)
    print()

    if stats["total"] == 0:
        print("  No battles tracked yet.")
        return

    wins = stats["wins"]
    losses = stats["losses"]
    draws = stats["draws"]
    total = stats["total"]
    three_crowns = stats["three_crowns"]

    three_crown_rate = (three_crowns / wins * 100) if wins > 0 else 0

    print(f"  Wins:          {wins} ({stats['win_rate']:.1f}%)")
    print(f"  Losses:        {losses} ({losses / total * 100:.1f}%)")
    if draws:
        print(f"  Draws:         {draws}")
    print(f"  3-Crown Wins:  {three_crowns} ({three_crown_rate:.1f}% of wins)")
    print(f"  Avg Crowns:    {stats['avg_crowns']:.2f}")

    trophy_change = stats["trophy_change"]
    trophy_str = f"+{trophy_change}" if trophy_change >= 0 else str(trophy_change)
    print(f"  Trophy Change: {trophy_str}")

    overall = analytics.get_overall_stats(session)
    overall_total = overall.get("total", 0)
    if overall_total > 0:
        overall_wins = overall.get("wins", 0)
        overall_wr = round(overall_wins / overall_total * 100, 1)
        diff = stats["win_rate"] - overall_wr
        direction = "above" if diff > 0 else "below"
        print(f"  vs Overall:    {abs(diff):.1f}pp {direction} ({overall_wr:.1f}% overall)")
    print()


def print_trophy_history(session: Session) -> None:
    """Print trophy progression as an ASCII chart."""
    history = analytics.get_trophy_history(session)

    print()
    print("=" * 70)
    print("TROPHY PROGRESSION")
    print("=" * 70)
    print()

    if not history:
        print("  No trophy data tracked yet.")
        return

    trophies = [h["trophies"] for h in history]
    min_t = min(trophies)
    max_t = max(trophies)

    try:
        term_width = shutil.get_terminal_size().columns
    except (AttributeError, ValueError):
        term_width = 80
    chart_width = max(20, term_width - 40)

    print(f"  Range: {min_t:,} - {max_t:,} ({max_t - min_t:+,} spread)")
    print(f"  Games: {len(history)}")
    print()

    max_rows = 30
    if len(history) > max_rows:
        step = len(history) / max_rows
        indices = [int(i * step) for i in range(max_rows)]
        indices[-1] = len(history) - 1
        display = [history[i] for i in indices]
    else:
        display = history

    span = max_t - min_t if max_t != min_t else 1
    for h in display:
        date = (h.get("battle_time") or "")[:8]
        result_icon = "W" if h["result"] == "win" else ("L" if h["result"] == "loss" else "D")
        bar_len = int((h["trophies"] - min_t) / span * chart_width)
        bar = "█" * bar_len
        print(f"  {date} {result_icon} {h['trophies']:>6,} |{bar}")
    print()


def print_archetype_stats(session: Session) -> None:
    """Print opponent archetype analysis."""
    stats = analytics.get_archetype_stats(session, min_battles=3)

    print()
    print("=" * 70)
    print("OPPONENT ARCHETYPE ANALYSIS (min 3 games)")
    print("=" * 70)
    print()

    if not stats:
        print("  Not enough data yet.")
        return

    for a in stats:
        print(f"  {a['archetype']:28} {a['total']:4} games  {a['win_rate']:5.1f}% WR  ({a['wins']}W-{a['losses']}L)")
    print()


def print_manifold(data: dict) -> None:
    """Print manifold leg profiles and comparisons."""
    print()
    print("=" * 70)
    print(f"TCN MANIFOLD ANALYSIS — {data['total_games']:,} games, 3 legs")
    print("=" * 70)

    legs = data.get("legs", [])
    for leg in legs:
        name = leg["leg_name"].upper()
        n = leg["game_count"]
        wr = leg["win_rate"]
        print()
        print(f"  LEG: {name}  ({n:,} games, {wr:.1%} WR)")
        print("  " + "─" * 50)

        # Duration / economy
        dur = leg.get("avg_duration")
        dur_str = f"{dur}s" if dur else "N/A"
        print(f"    Duration:      {dur_str}  |  "
              f"Leak: {leg['avg_player_leak']:.1f}e (team) / {leg['avg_opponent_leak']:.1f}e (opp)")
        print(f"    Crown diff:    {leg['avg_crown_diff']:+.2f}")

        # Phase distribution
        pf = leg.get("avg_phase_fraction", {})
        print(f"    Phases:        "
              f"reg {pf.get('regular', 0):.0%}  "
              f"dbl {pf.get('double', 0):.0%}  "
              f"OT {pf.get('overtime', 0):.0%}  "
              f"OT2x {pf.get('ot_double', 0):.0%}")

        # Tempo
        print(f"    Tempo:         "
              f"{leg['avg_plays_per_game']:.0f} plays/game, "
              f"median gap {leg['median_inter_play_gap']} ticks")

        # Spatial
        print(f"    Aggression:    {leg['aggression_index']:.1%} plays in opponent half")
        ld = leg.get("lane_distribution", {})
        print(f"    Lanes:         "
              f"L {ld.get('left', 0):.0%}  "
              f"R {ld.get('right', 0):.0%}  "
              f"C {ld.get('center', 0):.0%}")

        # Card types
        ct = leg.get("card_type_distribution", {})
        print(f"    Card types:    "
              f"troop {ct.get('troop', 0):.0%}  "
              f"spell {ct.get('spell', 0):.0%}  "
              f"bldg {ct.get('building', 0):.0%}")

        # Action-reaction
        alt = leg.get("alternation_rate", 0)
        style = "reactive" if alt > 0.55 else "committed" if alt < 0.45 else "balanced"
        print(f"    Play style:    {alt:.1%} alternation ({style})")

        # Top cards
        team_cards = leg.get("top_cards_team", [])
        if team_cards:
            cards_str = ", ".join(c["card"] for c in team_cards[:6])
            print(f"    Top team:      {cards_str}")
        opp_cards = leg.get("top_cards_opp", [])
        if opp_cards:
            cards_str = ", ".join(c["card"] for c in opp_cards[:6])
            print(f"    Top opp:       {cards_str}")

    # Comparisons
    comparisons = data.get("comparisons", [])
    if comparisons:
        print()
        print("  KEY DIFFERENCES")
        print("  " + "─" * 50)
        for c in comparisons:
            print(f"    → {c}")

    print()


def print_matchup_dive(data: dict) -> None:
    """Print full matchup deep dive analysis."""
    print()
    print("=" * 70)
    print(f"MATCHUP DEEP DIVE: vs {data['archetype']}")
    print("=" * 70)
    print()

    print(f"  Games:       {data['game_count']} ({data['win_count']}W / {data['loss_count']}L)")
    print(f"  Win Rate:    {data['win_rate']}%")
    if data.get("avg_duration"):
        print(f"  Avg Duration: {data['avg_duration']}s")
    print(f"  Avg Leak:    {data['avg_leak_win']}e (wins) / {data['avg_leak_loss']}e (losses)")
    if data.get("trophy_filter"):
        print(f"  Trophy Filter: >= {data['trophy_filter']}")

    # Opening analysis
    opening = data.get("opening", {})
    w = opening.get("win", {})
    l = opening.get("loss", {})
    if w.get("count") or l.get("count"):
        print()
        print("  OPENING (~30s)")
        print(f"  {'':18} {'WINS':>20} {'LOSSES':>20}")
        print(f"  {'First play tick':<18} {w.get('avg_first_play_tick', 'N/A'):>20} {l.get('avg_first_play_tick', 'N/A'):>20}")
        print(f"  {'Avg plays':<18} {w.get('avg_plays', 'N/A'):>20} {l.get('avg_plays', 'N/A'):>20}")
        print(f"  {'Aggression':<18} {w.get('aggression_index', 0):>19.1%} {l.get('aggression_index', 0):>19.1%}")

        for label, group in [("Win", w), ("Loss", l)]:
            cards = group.get("first_card_team", [])
            if cards:
                print(f"\n  {label} first cards:")
                for c in cards[:5]:
                    print(f"    {c['card']:28} {c['count']:>3}x ({c['pct']}%)")

    # Phase profile
    phases = data.get("phases", {}).get("phases", {})
    if phases:
        print()
        print("  PHASE BREAKDOWN")
        print(f"  {'Phase':<12} {'Win plays/100t':>15} {'Loss plays/100t':>16} {'Win opp spells':>15}")
        print("  " + "─" * 62)
        for phase in ("regular", "double", "overtime", "ot_double"):
            pd = phases.get(phase, {})
            wp = pd.get("win", {})
            lp = pd.get("loss", {})
            w_rate = wp.get("plays_per_100_ticks", 0)
            l_rate = lp.get("plays_per_100_ticks", 0)
            w_spell = wp.get("opp_card_type_mix", {}).get("spell", 0)
            print(f"  {phase:<12} {w_rate:>15.2f} {l_rate:>16.2f} {w_spell:>14.0%}")

    # Push timing
    pushes = data.get("push_timing", {})
    wp = pushes.get("win", {})
    lp = pushes.get("loss", {})
    if wp or lp:
        print()
        print("  PUSH TIMING")
        print(f"  {'':18} {'WINS':>20} {'LOSSES':>20}")
        w_fpt = wp.get("avg_first_push_tick")
        l_fpt = lp.get("avg_first_push_tick")
        print(f"  {'First push tick':<18} {(str(w_fpt) if w_fpt else 'N/A'):>20} {(str(l_fpt) if l_fpt else 'N/A'):>20}")
        print(f"  {'Avg pushes/game':<18} {wp.get('avg_push_count', 0):>20.2f} {lp.get('avg_push_count', 0):>20.2f}")
        print(f"  {'Avg push size':<18} {wp.get('avg_push_size', 0):>20.1f} {lp.get('avg_push_size', 0):>20.1f}")

    # Notable patterns
    patterns = data.get("notable_patterns", [])
    if patterns:
        print()
        print("  NOTABLE PATTERNS")
        for p in patterns:
            print(f"    → {p}")

    print()


def print_broken_cycle(results: list[dict]) -> None:
    """Print broken cycle analysis results."""
    print()
    print("=" * 70)
    print("BROKEN CYCLE ANALYSIS")
    print("=" * 70)
    print()

    if not results:
        print("  No results.")
        return

    for r in results:
        a, b = r["pair"]
        print(f"  {a} + {b}  (window: {r['window_ticks']} ticks)")
        print(f"    Games where {a} played: {r['total_games']}")
        print(f"    Intact (B within window): {r['intact_count']}  "
              f"WR {r['intact_win_rate']}%")
        print(f"    Broken (B missing/late):  {r['broken_count']}  "
              f"WR {r['broken_win_rate']}%")
        delta = r["delta_pp"]
        direction = "+" if delta >= 0 else ""
        print(f"    Delta: {direction}{delta}pp")
        print()
