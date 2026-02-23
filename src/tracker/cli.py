"""CLI entrypoint for the Clash Royale Battle Tracker."""

import argparse
import logging
import os

# Must be set before prometheus_client is first imported (via tracker.metrics)
os.environ.setdefault("PROMETHEUS_DISABLE_CREATED_SERIES", "true")

from tracker import analytics, reporting
from tracker.api import ClashRoyaleAPI, DEFAULT_API_URL
from tracker.database import get_session, init_db
from tracker.export import export_data

DB_FILE = "clash_royale_history.db"


def fetch_and_store(
    api_key: str, player_tag: str, session, api_url: str = DEFAULT_API_URL
) -> None:
    """Fetch latest battles and store new ones.

    Args:
        api_key: Clash Royale API key.
        player_tag: Player tag (without '#').
        session: SQLAlchemy session.
        api_url: API base URL.
    """
    api = ClashRoyaleAPI(api_key, base_url=api_url)

    print(f"\n🔄 Fetching data for #{player_tag}...")

    try:
        player = api.get_player(player_tag)
        analytics.store_player_snapshot(session, player)
        print(f"  ✓ Profile: {player.get('name')} | {player.get('trophies'):,} trophies")
        print(f"  ✓ All-time: {player.get('wins'):,}W / {player.get('losses'):,}L / {player.get('threeCrownWins'):,} 3-crowns")

        diff = analytics.get_snapshot_diff(session)
        if diff:
            parts = []
            if diff["trophies"]:
                sign = "+" if diff["trophies"] > 0 else ""
                parts.append(f"{sign}{diff['trophies']} trophies")
            if diff["wins"]:
                parts.append(f"+{diff['wins']} wins")
            if diff["losses"]:
                parts.append(f"+{diff['losses']} losses")
            if parts:
                print(f"  ✓ Since last fetch: {', '.join(parts)}")
    except Exception as e:
        print(f"  ✗ Error fetching player: {e}")
        return

    try:
        battles = api.get_battle_log(player_tag)
        new_count = 0
        for battle in battles:
            battle_id, is_new = analytics.store_battle(session, battle, player.get("tag"))
            if is_new:
                new_count += 1

        print(f"  ✓ Fetched {len(battles)} battles, {new_count} NEW")
        print(f"  ✓ Total tracked: {analytics.get_total_battles(session):,} battles")
    except Exception as e:
        print(f"  ✗ Error fetching battles: {e}")


def main() -> int:
    """CLI entrypoint. Returns exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Clash Royale Battle Tracker - Build your historical match database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  clash-stats --fetch --api-key YOUR_KEY --player-tag ABC123
  clash-stats --stats
  clash-stats --deck-stats
  clash-stats --crowns
  clash-stats --matchups
  clash-stats --recent 20
  clash-stats --streaks
  clash-stats --rolling 35
  clash-stats --trophy-history
  clash-stats --archetypes
  clash-stats --stats --export json --output stats.json

Environment variables:
  CR_API_KEY     - Your API key from developer.clashroyale.com
  CR_PLAYER_TAG  - Your player tag (without #)
  CR_API_URL     - API base URL (default: https://api.clashroyale.com/v1)
        """,
    )
    parser.add_argument("--fetch", action="store_true", help="Fetch and store new battles")
    parser.add_argument("--stats", action="store_true", help="Show overall statistics")
    parser.add_argument("--deck-stats", action="store_true", help="Show per-deck statistics")
    parser.add_argument("--crowns", action="store_true", help="Show crown distribution")
    parser.add_argument("--matchups", action="store_true", help="Show card matchup analysis")
    parser.add_argument("--recent", type=int, metavar="N", help="Show last N battles")
    parser.add_argument("--streaks", action="store_true", help="Win/loss streak analysis")
    parser.add_argument("--rolling", type=int, metavar="N", help="Rolling window stats (last N games)")
    parser.add_argument("--trophy-history", action="store_true", help="Trophy progression over time")
    parser.add_argument("--archetypes", action="store_true", help="Opponent archetype analysis")
    parser.add_argument("--export", choices=["csv", "json"], help="Export data as CSV or JSON")
    parser.add_argument("--output", type=str, metavar="FILE", help="Export output file (default: stdout)")
    parser.add_argument("--replay-login", action="store_true",
                        help="Start RoyaleAPI login (complete via noVNC)")
    parser.add_argument("--replay-check", action="store_true",
                        help="Check if RoyaleAPI login is complete and save session")
    parser.add_argument("--fetch-replays", action="store_true",
                        help="Fetch replay data from RoyaleAPI for recent battles")
    parser.add_argument("--corpus-update", action="store_true",
                        help="Refresh top-ladder player tags for training corpus")
    parser.add_argument("--corpus-scrape", action="store_true",
                        help="Scrape battles for corpus players")
    parser.add_argument("--corpus-replays", action="store_true",
                        help="Scrape replays for corpus players")
    parser.add_argument("--corpus-stats", action="store_true",
                        help="Show training corpus statistics")
    parser.add_argument("--corpus-add-priority", type=str, nargs="+", metavar="TAG",
                        help="Add player tags to the priority replay queue")
    parser.add_argument("--sim-matchups", action="store_true",
                        help="Run Monte Carlo matchup analysis (ADR-002)")
    parser.add_argument("--sim-interactions", action="store_true",
                        help="Show card interaction matrix (P(win|card))")
    parser.add_argument("--sim-full", action="store_true",
                        help="Run full simulation suite and cache results")
    parser.add_argument("--corpus-limit", type=int, default=20, metavar="N",
                        help="Max corpus players to process per run (default: 20)")
    parser.add_argument("--replays-per-player", type=int, metavar="N",
                        help="Max replays to fetch per player (default: 25, env: REPLAYS_PER_PLAYER)")
    parser.add_argument("--max-pages", type=int, default=5, metavar="N",
                        help="Max pagination depth per player (1=fast/recent, 5=full; default: 5)")
    parser.add_argument("--api-key", type=str, help="CR API key")
    parser.add_argument("--player-tag", type=str, help="Player tag (without #)")
    parser.add_argument("--api-url", type=str, help="API base URL (default: https://api.clashroyale.com/v1)")
    parser.add_argument("--db", type=str, default=DB_FILE, help=f"Database file (default: {DB_FILE})")

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("CR_API_KEY")
    player_tag = args.player_tag or os.environ.get("CR_PLAYER_TAG")
    api_url = args.api_url or os.environ.get("CR_API_URL", DEFAULT_API_URL)

    engine = init_db(args.db)
    session = get_session(engine)

    try:
        if args.fetch:
            if not api_key or not player_tag:
                print("Error: --api-key and --player-tag required for fetching")
                print("       Or set CR_API_KEY and CR_PLAYER_TAG environment variables")
                return 1
            fetch_and_store(api_key, player_tag.replace("#", ""), session, api_url=api_url)
            from tracker.metrics import flush_metrics
            flush_metrics("fetch")

        export_fmt = args.export
        export_out = args.output

        if args.stats:
            if export_fmt:
                export_data(analytics.get_overall_stats(session), export_fmt, export_out)
            else:
                reporting.print_overall_stats(session)

        if args.deck_stats:
            if export_fmt:
                export_data(analytics.get_deck_stats(session, min_battles=1), export_fmt, export_out)
            else:
                reporting.print_deck_stats(session)

        if args.crowns:
            if export_fmt:
                export_data(analytics.get_crown_distribution(session), export_fmt, export_out)
            else:
                reporting.print_crown_distribution(session)

        if args.matchups:
            if export_fmt:
                export_data(analytics.get_card_matchup_stats(session, min_battles=1), export_fmt, export_out)
            else:
                reporting.print_matchup_stats(session)

        if args.recent:
            if export_fmt:
                export_data(analytics.get_recent_battles(session, args.recent), export_fmt, export_out)
            else:
                reporting.print_recent_battles(session, args.recent)

        if args.streaks:
            if export_fmt:
                export_data(analytics.get_streaks(session), export_fmt, export_out)
            else:
                reporting.print_streaks(session)

        if args.rolling:
            if export_fmt:
                export_data(analytics.get_rolling_stats(session, args.rolling), export_fmt, export_out)
            else:
                reporting.print_rolling_stats(session, args.rolling)

        if args.trophy_history:
            if export_fmt:
                export_data(analytics.get_trophy_history(session), export_fmt, export_out)
            else:
                reporting.print_trophy_history(session)

        if args.archetypes:
            if export_fmt:
                export_data(analytics.get_archetype_stats(session, min_battles=1), export_fmt, export_out)
            else:
                reporting.print_archetype_stats(session)

        if args.replay_login:
            from tracker.replays import run_start_login
            print("Starting RoyaleAPI login... Complete via noVNC.")
            run_start_login()

        if args.replay_check:
            from tracker.replays import run_check_login
            if run_check_login():
                print("RoyaleAPI session saved successfully.")
            else:
                print("Login not yet complete. Finish login via noVNC.")

        if args.fetch_replays:
            if not player_tag:
                print("Error: --player-tag required for replay fetching")
                return 1
            from tracker.replays import run_fetch_replays
            count = run_fetch_replays(session, player_tag.replace("#", ""))
            print(f"  ✓ Fetched {count} replays")

        if args.corpus_update:
            if not api_key:
                print("Error: --api-key required for corpus update")
                return 1
            from tracker.corpus import update_top_ladder
            api = ClashRoyaleAPI(api_key, base_url=api_url)
            added = update_top_ladder(session, api, limit=args.corpus_limit)
            print(f"  ✓ Corpus update: {added} new players added")

        if args.corpus_scrape:
            if not api_key:
                print("Error: --api-key required for corpus scraping")
                return 1
            from tracker.corpus_scraper import scrape_corpus_battles
            from tracker.metrics import flush_metrics
            api = ClashRoyaleAPI(api_key, base_url=api_url)
            result = scrape_corpus_battles(session, api, limit=args.corpus_limit)
            print(f"  ✓ Corpus battles: {result['total_players']} players, "
                  f"{result['total_new_battles']} new battles")
            flush_metrics("corpus_scrape")

        if args.corpus_replays:
            from tracker.corpus_scraper import run_scrape_corpus_replays
            from tracker.metrics import flush_metrics
            replays_per_player = (
                args.replays_per_player
                or int(os.environ.get("REPLAYS_PER_PLAYER", "25"))
            )
            result = run_scrape_corpus_replays(
                session, limit=args.corpus_limit,
                replays_per_player=replays_per_player,
                max_pages=args.max_pages,
            )
            print(f"  ✓ Corpus replays: {result['total_players']} players, "
                  f"{result['total_replays']} replays")
            flush_metrics("corpus_replays")

        if args.corpus_add_priority:
            from tracker.corpus import add_manual_player
            from tracker.models import PlayerCorpus
            for tag in args.corpus_add_priority:
                clean = f"#{tag.lstrip('#')}"
                existing = session.get(PlayerCorpus, clean)
                was_priority = existing and existing.source == "priority" if existing else False
                added = add_manual_player(session, tag, source="priority")
                if added:
                    print(f"  ✓ {tag}: added as priority")
                elif was_priority:
                    print(f"  · {tag}: already priority")
                else:
                    print(f"  ✓ {tag}: promoted to priority")

        if args.corpus_stats:
            from tracker.corpus import get_corpus_stats
            stats = get_corpus_stats(session)
            print("\n📊 Training Data Corpus")
            print(f"  Players tracked:   {stats['total_players']} "
                  f"({stats['active_players']} active)")
            print(f"  Source breakdown:   {stats['source_breakdown']}")
            print(f"  Battles by corpus: {stats['battles_by_corpus']}")
            print(f"  Total battles:     {stats['total_battles']:,}")
            print(f"  With replay data:  {stats['battles_with_replays']:,} "
                  f"({stats['replay_coverage_pct']}%)")

        if args.sim_matchups:
            from tracker.simulation.matchup_model import (
                compute_matchup_posteriors, compute_threat_ranking,
            )
            posteriors = compute_matchup_posteriors(
                session, player_tag=player_tag, min_battles=3,
                use_sub_archetypes=True,
            )
            threats = compute_threat_ranking(posteriors, min_battles=5)
            print("\n🎯 Matchup Posteriors (Beta-Binomial)")
            print(f"{'Archetype':<28} {'W':>4} {'L':>4} {'P(win)':>8} {'95% CI':>16}")
            print("─" * 66)
            for t in threats:
                ci = f"[{t['ci_low']:.2f}, {t['ci_high']:.2f}]"
                print(f"{t['archetype']:<28} {t['wins']:>4} {t['losses']:>4} "
                      f"{t['posterior_mean']:>7.1%} {ci:>16}")

            # Sub-archetypes for top threats
            for t in threats[:5]:
                archetype = t["archetype"]
                if "sub_archetypes" in posteriors.get(archetype, {}):
                    subs = posteriors[archetype]["sub_archetypes"]
                    print(f"\n  ↳ {archetype} sub-archetypes:")
                    for sa in subs:
                        sig = ", ".join(sa["signature_cards"][:4])
                        print(f"    {sig:<40} n={sa['count']:>3} "
                              f"P(win)={sa['posterior_mean']:.1%} "
                              f"[{sa['ci_low']:.2f}, {sa['ci_high']:.2f}]")

        if args.sim_interactions:
            from tracker.simulation.interaction_matrix import (
                build_card_interaction_matrix,
            )
            matrix = build_card_interaction_matrix(
                session, player_tag=player_tag, min_appearances=5,
            )
            print("\n⚔️  Card Interaction Matrix — P(win | opponent has card)")
            print(f"{'Card':<28} {'Faced':>6} {'Win%':>6} {'95% CI':>16}")
            print("─" * 60)
            for card, data in matrix.items():
                ci = f"[{data['ci_low']:.2f}, {data['ci_high']:.2f}]"
                print(f"{card:<28} {data['total']:>6} "
                      f"{data['win_rate']:>5.1%} {ci:>16}")

        if args.sim_full:
            from tracker.simulation.runner import run_full_simulation
            results = run_full_simulation(session, player_tag=player_tag)
            n_matchups = len(results.get("corpus_matchups", {}))
            n_cards = len(results.get("card_interactions", {}))
            n_subs = sum(len(v) for v in results.get("sub_archetypes", {}).values())
            print(f"  ✓ Simulation complete: {n_matchups} matchups, "
                  f"{n_cards} card interactions, {n_subs} sub-archetypes")
            print(f"  ✓ Results cached for dashboard")

        # Default: show help + db status
        has_action = any([
            args.fetch, args.stats, args.deck_stats, args.crowns,
            args.matchups, args.recent, args.streaks, args.rolling,
            args.trophy_history, args.archetypes,
            args.replay_login, args.replay_check, args.fetch_replays,
            args.corpus_update, args.corpus_scrape, args.corpus_replays,
            args.corpus_stats, args.corpus_add_priority,
            args.sim_matchups, args.sim_interactions, args.sim_full,
        ])
        if not has_action:
            parser.print_help()
            print()
            print(f"Database: {args.db}")
            print(f"Battles tracked: {analytics.get_total_battles(session):,}")

        return 0

    finally:
        session.close()
        engine.dispose()
