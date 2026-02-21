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
