"""CLI entrypoint for the Clash Royale Battle Tracker."""

import argparse
import logging
import os

logger = logging.getLogger(__name__)

# Must be set before prometheus_client is first imported (via tracker.metrics)
os.environ.setdefault("PROMETHEUS_DISABLE_CREATED_SERIES", "true")

from tracker import analytics, reporting
from tracker.api import ClashRoyaleAPI, DEFAULT_API_URL
from tracker.database import get_session, init_db
from tracker.export import export_data

DB_FILE = os.environ.get("CR_DB_PATH", "data/clash_royale_history.db")
# If DATABASE_URL is set it takes precedence over --db for all operations.
_DATABASE_URL = os.environ.get("DATABASE_URL")


def fetch_and_store(
    api_key: str, player_tag: str, session, api_url: str = DEFAULT_API_URL
) -> int:
    """Fetch latest battles and store new ones.

    Args:
        api_key: Clash Royale API key.
        player_tag: Player tag (without '#').
        session: SQLAlchemy session.
        api_url: API base URL.

    Returns:
        Number of new battles stored, or -1 on error.
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
        return -1

    try:
        battles = api.get_battle_log(player_tag)
        new_count = 0
        for battle in battles:
            battle_id, is_new = analytics.store_battle(session, battle, player.get("tag"))
            if is_new:
                new_count += 1

        print(f"  ✓ Fetched {len(battles)} battles, {new_count} NEW")
        print(f"  ✓ Total tracked: {analytics.get_total_battles(session):,} battles")

        # Tilt detection after every fetch
        if new_count > 0:
            from tracker.ml.tilt_detector import detect_tilt, print_tilt_warning
            tilt_status = detect_tilt(session)
            print_tilt_warning(tilt_status)

        return new_count
    except Exception as e:
        print(f"  ✗ Error fetching battles: {e}")
        return -1


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
    parser.add_argument("--personal-combined", action="store_true",
                        help="Fetch battles + replays for personal tag in one pass")
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
    parser.add_argument("--corpus-discover", action="store_true",
                        help="Discover new players from opponent tags in existing battles")
    parser.add_argument("--corpus-locations", action="store_true",
                        help="Discover players from regional leaderboards (deeper than global)")
    parser.add_argument("--corpus-nemeses", action="store_true",
                        help="Add opponents you've lost to into the corpus")
    parser.add_argument("--corpus-combined", action="store_true",
                        help="Combined battle+replay scrape (chains CR API and RoyaleAPI per player)")
    parser.add_argument("--sim-matchups", action="store_true",
                        help="Run Monte Carlo matchup analysis (ADR-002)")
    parser.add_argument("--sim-interactions", action="store_true",
                        help="Show card interaction matrix (P(win|card))")
    parser.add_argument("--sim-elixir", type=str, nargs="?", const="all", metavar="ARCHETYPE",
                        help="Elixir economy analysis (optionally vs ARCHETYPE, e.g. 'Hog Cycle')")
    parser.add_argument("--sim-hands", action="store_true",
                        help="Opening hand analysis — first-card win rates")
    parser.add_argument("--sim-full", action="store_true",
                        help="Run full simulation suite and cache results")
    parser.add_argument("--corpus-limit", type=int, default=20, metavar="N",
                        help="Max corpus players to process per run (default: 20)")
    parser.add_argument("--replays-per-player", type=int, metavar="N",
                        help="Max replays to fetch per player (default: 25, env: REPLAYS_PER_PLAYER)")
    parser.add_argument("--max-pages", type=int, default=5, metavar="N",
                        help="Max pagination depth per player (1=fast/recent, 5=full; default: 5)")
    parser.add_argument("--concurrency", type=int, default=1, metavar="N",
                        help="Number of parallel browser tabs for replay scraping (default: 1)")
    # ML Phase 0 (ADR-001, ADR-003)
    parser.add_argument("--tilt-check", action="store_true",
                        help="Check recent games for tilt patterns")
    parser.add_argument("--train-tcn", action="store_true",
                        help="Train TCN embedding model (ADR-003 Phase 1)")
    parser.add_argument("--build-features", action="store_true",
                        help="Extract ML feature vectors for all replay games")
    parser.add_argument("--train-embeddings", action="store_true",
                        help="Fit UMAP embeddings and cluster games")
    parser.add_argument("--clusters", action="store_true",
                        help="Show game cluster profiles")
    parser.add_argument("--similar", type=str, metavar="BATTLE_ID",
                        help="Find games most similar to the given battle")
    parser.add_argument("--embed-new", action="store_true",
                        help="Embed new games using trained TCN (inference only, no retraining)")
    # Win Probability (ADR-004)
    parser.add_argument("--train-wp", action="store_true",
                        help="Train win probability model (ADR-004)")
    parser.add_argument("--wp-unfreeze", action="store_true",
                        help="Unfreeze TCN encoder during WP training (full fine-tune)")
    parser.add_argument("--wp-infer", action="store_true",
                        help="Run WP inference using existing checkpoint (no retraining)")
    parser.add_argument("--wp-infer-new", action="store_true",
                        help="Run WP inference only on games missing WP data (incremental)")
    parser.add_argument("--wp", type=str, metavar="BATTLE_ID",
                        help="Show P(win) curve for a game")
    parser.add_argument("--wp-critical", type=str, metavar="BATTLE_ID",
                        help="Show top critical plays (highest WPA) for a game")
    parser.add_argument("--wp-cards", action="store_true",
                        help="Show aggregate card WPA impact across all games")
    parser.add_argument("--train-activity-model", action="store_true",
                        help="Train activity prediction model for corpus scheduling")
    parser.add_argument("--mark-stale-replays", action="store_true",
                        help="Mark unfetched battles older than 7 days as permanently missed")
    parser.add_argument("--manifold", action="store_true",
                        help="Profile the 3-leg TCN embedding manifold")
    # Counterfactual Simulator (ADR-006)
    parser.add_argument("--train-cvae", action="store_true",
                        help="Train CVAE counterfactual model (ADR-006)")
    parser.add_argument("--counterfactual", nargs=3, metavar=("BATTLE_ID", "OLD_CARD", "NEW_CARD"),
                        help="Generate counterfactual for a game (swap OLD_CARD with NEW_CARD)")
    parser.add_argument("--deck-gradient", action="store_true",
                        help="Rank best single-card swaps by expected WR delta")
    parser.add_argument("--cf-samples", type=int, default=10, metavar="N",
                        help="Samples per counterfactual (default: 10)")
    # Temporal analysis
    parser.add_argument("--matchup-dive", type=str, metavar="ARCHETYPE",
                        help="Deep temporal analysis against an archetype (e.g. 'Hog Cycle')")
    parser.add_argument("--broken-cycle", type=str, nargs="+", metavar="A:B",
                        help="Detect broken synergy pairs (kebab-case, e.g. miner:graveyard)")
    parser.add_argument("--min-trophies", type=int, metavar="N",
                        help="Minimum opponent trophies (used with --matchup-dive)")

    parser.add_argument("--api-key", type=str, help="CR API key")
    parser.add_argument("--player-tag", type=str, help="Player tag (without #)")
    parser.add_argument("--api-url", type=str, help="API base URL (default: https://api.clashroyale.com/v1)")
    parser.add_argument("--db", type=str, default=DB_FILE, help=f"Database file (default: {DB_FILE})")

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("CR_API_KEY")
    player_tag = args.player_tag or os.environ.get("CR_PLAYER_TAG")
    api_url = args.api_url or os.environ.get("CR_API_URL", DEFAULT_API_URL)

    # DATABASE_URL env var overrides --db (enables PostgreSQL backend)
    _db_ref = _DATABASE_URL or args.db
    engine = init_db(_db_ref)
    session = get_session(engine)

    # model_dir lives next to the DB file for SQLite; fall back to data/ml_models
    # when using a URL-based backend (PostgreSQL).
    from pathlib import Path
    _model_dir = (
        Path("data/ml_models")
        if ("://" in _db_ref)
        else Path(args.db).parent / "ml_models"
    )

    try:
        if args.fetch:
            if not api_key or not player_tag:
                print("Error: --api-key and --player-tag required for fetching")
                print("       Or set CR_API_KEY and CR_PLAYER_TAG environment variables")
                return 1
            fetch_and_store(api_key, player_tag.replace("#", ""), session, api_url=api_url)
            from tracker.metrics import SCRAPE_RUNS, flush_metrics
            SCRAPE_RUNS.labels(scrape_type="battles", outcome="success").inc()
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
            from tracker.replay_http import run_fetch_replays_http
            count = run_fetch_replays_http(session, player_tag.replace("#", ""))
            if count == -1:
                print("  ⚠ HTTP session expired, falling back to browser")
                from tracker.replays import run_fetch_replays
                count = run_fetch_replays(session, player_tag.replace("#", ""))
            print(f"  ✓ Fetched {count} replays")

        if args.personal_combined:
            if not api_key or not player_tag:
                print("Error: --api-key and --player-tag required")
                print("       Or set CR_API_KEY and CR_PLAYER_TAG environment variables")
                return 1
            tag_clean = player_tag.replace("#", "")
            new_battles = fetch_and_store(api_key, tag_clean, session, api_url=api_url)
            from tracker.metrics import SCRAPE_RUNS, flush_metrics
            if new_battles >= 0:
                SCRAPE_RUNS.labels(scrape_type="battles", outcome="success").inc()
            # Fetch personal replays via HTTP (no browser needed).
            # Falls back to browser-based fetcher if cookies are expired.
            try:
                from tracker.replay_http import run_fetch_replays_http
                replay_count = run_fetch_replays_http(session, tag_clean)
                if replay_count == -1:
                    logger.warning("HTTP replay fetch: session expired, falling back to browser")
                    from tracker.replays import run_fetch_replays
                    replay_count = run_fetch_replays(session, tag_clean)
                if replay_count > 0:
                    print(f"  ✓ Fetched {replay_count} personal replays")
                    SCRAPE_RUNS.labels(scrape_type="personal_replays", outcome="success").inc()
            except Exception as e:
                logger.warning("Personal replay scrape failed: %s", e)
                SCRAPE_RUNS.labels(scrape_type="personal_replays", outcome="error").inc()
            # Incremental WP inference — process any games with replays but no WP data
            try:
                from tracker.ml.wp_training import infer_wp_incremental
                wp_new = infer_wp_incremental(session, model_dir=_model_dir)
                if wp_new and wp_new > 0:
                    print(f"  ✓ WP inference: {wp_new} new games")
            except Exception as e:
                logger.warning("Incremental WP inference failed: %s", e)
            flush_metrics("personal_combined")

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
            concurrency = args.concurrency or int(
                os.environ.get("REPLAY_CONCURRENCY", "1")
            )
            result = run_scrape_corpus_replays(
                session, limit=args.corpus_limit,
                replays_per_player=replays_per_player,
                max_pages=args.max_pages,
                concurrency=concurrency,
            )
            print(f"  ✓ Corpus replays: {result['total_players']} players, "
                  f"{result['total_replays']} replays")
            flush_metrics("corpus_replays")

        if args.corpus_combined:
            if not api_key:
                print("Error: --api-key required for combined scrape")
                print("       Or set CR_API_KEY environment variable")
                return 1
            from tracker.corpus_scraper import run_scrape_corpus_combined
            from tracker.metrics import flush_metrics
            api = ClashRoyaleAPI(api_key, base_url=api_url)
            replays_per_player = (
                args.replays_per_player
                or int(os.environ.get("REPLAYS_PER_PLAYER", "25"))
            )
            concurrency = args.concurrency or int(
                os.environ.get("REPLAY_CONCURRENCY", "1")
            )
            result = run_scrape_corpus_combined(
                session, api, limit=args.corpus_limit,
                replays_per_player=replays_per_player,
                max_pages=args.max_pages,
                concurrency=concurrency,
                personal_tag=player_tag or os.environ.get("CR_PLAYER_TAG"),
            )
            print(f"  ✓ Combined scrape: {result['total_players']} players, "
                  f"{result['total_new_battles']} battles, "
                  f"{result['total_replays']} replays")
            flush_metrics("corpus_combined")

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

        if args.corpus_discover:
            from tracker.corpus import discover_from_opponents
            added = discover_from_opponents(
                session, max_players=args.corpus_limit,
            )
            print(f"  ✓ Network discovery: {added} new players from opponent tags")

        if args.corpus_locations:
            if not api_key:
                print("Error: --api-key required for location discovery")
                return 1
            from tracker.corpus import update_location_leaderboards
            api = ClashRoyaleAPI(api_key, base_url=api_url)
            added = update_location_leaderboards(session, api, limit=args.corpus_limit)
            print(f"  ✓ Location discovery: {added} new players from regional leaderboards")

        if args.corpus_nemeses:
            if not player_tag:
                print("Error: --player-tag required for nemesis discovery")
                return 1
            from tracker.corpus import discover_nemeses
            added = discover_nemeses(session, player_tag)
            print(f"  ✓ Nemesis discovery: {added} new players from your losses")

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

        if args.sim_elixir:
            from tracker.simulation.elixir_economy import (
                build_exchange_distributions,
                compute_matchup_elixir_profile,
            )
            archetype_filter = None if args.sim_elixir == "all" else args.sim_elixir

            if archetype_filter and player_tag:
                # Detailed matchup profile
                profile = compute_matchup_elixir_profile(
                    session, player_tag, archetype_filter,
                )
                if profile:
                    print(f"\nElixir Economy vs {profile['archetype']}")
                    print(f"  Games: {profile['total_games']} "
                          f"({profile['wins']}W-{profile['losses']}L, "
                          f"{profile['win_rate']:.0%} WR)")
                    if profile["avg_leak_wins"] is not None:
                        print(f"  Avg elixir leak (wins):   {profile['avg_leak_wins']:.1f}")
                    if profile["avg_leak_losses"] is not None:
                        print(f"  Avg elixir leak (losses): {profile['avg_leak_losses']:.1f}")

                    # Card performance comparison
                    w_perf = profile["card_performance_wins"]
                    l_perf = profile["card_performance_losses"]
                    all_cards = sorted(set(w_perf.keys()) | set(l_perf.keys()))
                    if all_cards:
                        print(f"\n  {'Card':<24} {'Win net_e':>10} {'Loss net_e':>10} {'Delta':>8}")
                        print(f"  {'─' * 56}")
                        rows = []
                        for card in all_cards:
                            w = w_perf.get(card, {}).get("mean_net", None)
                            l = l_perf.get(card, {}).get("mean_net", None)
                            delta = (w or 0) - (l or 0)
                            rows.append((card, w, l, delta))
                        rows.sort(key=lambda r: r[3], reverse=True)
                        for card, w, l, delta in rows:
                            w_str = f"{w:>+.1f}" if w is not None else "   -"
                            l_str = f"{l:>+.1f}" if l is not None else "   -"
                            d_str = f"{delta:>+.1f}"
                            print(f"  {card:<24} {w_str:>10} {l_str:>10} {d_str:>8}")
                else:
                    print(f"  Not enough games vs {archetype_filter} with replay data.")
            else:
                # Global exchange distributions
                dists = build_exchange_distributions(
                    session, player_tag=player_tag, min_exchanges=10,
                )
                print(f"\nElixir Exchange Distributions")
                print(f"  Games: {dists['games_processed']}, "
                      f"Exchanges: {dists['total_exchanges']}")
                print(f"\n  {'Card':<24} {'Plays':>6} {'Net e':>7} "
                      f"{'Std':>5} {'WR':>6}  Top Responses")
                print(f"  {'─' * 80}")
                for card, data in dists["card_distributions"].items():
                    o = data["overall"]
                    resps = ", ".join(r[0] for r in o["top_responses"][:3])
                    print(f"  {card:<24} {o['total_plays']:>6} "
                          f"{o['mean_net_elixir']:>+6.1f} {o['std_net_elixir']:>5.1f} "
                          f"{o['win_rate']:>5.0%}  {resps}")

        if args.sim_hands:
            from tracker.simulation.opening_hand import analyze_opening_hands
            if not player_tag:
                print("Error: --player-tag required for opening hand analysis")
            else:
                results = analyze_opening_hands(session, player_tag)
                if "error" in results:
                    print(f"  Error: {results['error']}")
                else:
                    print(f"\nOpening Hand Analysis ({results['games_analyzed']} games)")
                    print(f"  Deck: {', '.join(results['deck_cards'])}")
                    cd = results["cost_distribution"]
                    print(f"  Hand cost range: {cd['min_hand']}-{cd['max_hand']}e "
                          f"(avg {cd['mean_hand']}e)")
                    print(f"  Cheapest hand: {', '.join(cd['cheapest_possible'])}")
                    print(f"  Heaviest hand: {', '.join(cd['most_expensive'])}")

                    print(f"\n  First-Card Win Rates:")
                    print(f"  {'Card':<24} {'Games':>6} {'WR':>6} {'Cost':>5}")
                    print(f"  {'─' * 44}")
                    for card, data in results["opener_performance"].items():
                        print(f"  {card:<24} {data['total']:>6} "
                              f"{data['win_rate']:>5.0%} {data['avg_cost']:>5.1f}e")

        if args.sim_full:
            from tracker.simulation.runner import run_full_simulation
            results = run_full_simulation(session, player_tag=player_tag)
            n_matchups = len(results.get("corpus_matchups", {}))
            n_cards = len(results.get("card_interactions", {}))
            n_subs = sum(len(v) for v in results.get("sub_archetypes", {}).values())
            print(f"  ✓ Simulation complete: {n_matchups} matchups, "
                  f"{n_cards} card interactions, {n_subs} sub-archetypes")
            print(f"  ✓ Results cached for dashboard")

        if args.tilt_check:
            from tracker.ml.tilt_detector import detect_tilt, print_tilt_warning
            tilt_status = detect_tilt(session)
            if tilt_status.level == "none":
                print(f"\n  {tilt_status.message}")
            else:
                print_tilt_warning(tilt_status)

        if args.train_tcn:
            from tracker.ml.training import train_tcn
            train_tcn(session, model_dir=_model_dir)

        if args.build_features:
            from tracker.ml.card_metadata import CardVocabulary
            from tracker.ml.features import build_feature_matrix
            vocab = CardVocabulary(session)
            battle_ids, matrix = build_feature_matrix(session, vocab)
            if len(battle_ids) > 0:
                print(f"  ✓ Features extracted: {len(battle_ids)} games, "
                      f"{matrix.shape[1]} dimensions")
            else:
                print("  · No new games to process")

        if args.train_embeddings:
            from tracker.ml.features import load_feature_matrix
            from tracker.ml.umap_embeddings import train_embeddings
            battle_ids, features = load_feature_matrix(session)
            if len(battle_ids) < 20:
                print(f"  ✗ Need at least 20 games with features (have {len(battle_ids)})")
                print("    Run --build-features first")
            else:
                train_embeddings(session, battle_ids, features, model_dir=_model_dir)
                print(f"  ✓ Embeddings trained: {len(battle_ids)} games, "
                      f"UMAP 15d + 3d, HDBSCAN clustered")

        if args.clusters:
            from tracker.ml.clustering import profile_clusters
            profiles = profile_clusters(session)
            if not profiles:
                print("  ✗ No embeddings found. Run --train-embeddings first.")
            else:
                print(f"\n{'Cluster':<12} {'Size':>6} {'Win%':>6} {'Personal':>10}")
                print("─" * 38)
                for p in profiles:
                    label = p["label"]
                    print(f"{label:<12} {p['size']:>6} "
                          f"{p['win_rate']:>5.1%} {p['personal_count']:>10}")

        if args.similar:
            from tracker.ml.similarity import find_similar
            results = find_similar(session, args.similar)
            if not results["corpus"] and not results["personal"]:
                print(f"  ✗ No embedding found for {args.similar}")
            else:
                header = (f"{'Opponent':>16} {'Result':>7} {'Score':>7} "
                          f"{'Rank':>9} {'Kernel':>7} {'Archetype':>22}")
                for label, games in [("Personal", results["personal"]), ("Corpus", results["corpus"])]:
                    if not games:
                        continue
                    print(f"\n{label} games similar to {args.similar}:")
                    print(header)
                    print("─" * 75)
                    for r in games:
                        score = f"{r.get('player_crowns', '?')}-{r.get('opponent_crowns', '?')}"
                        pct = f"Top {r['percentile']*100:.1f}%"
                        print(f"{r.get('opponent_name', '?'):>16} "
                              f"{r.get('result', '?'):>7} {score:>7} "
                              f"{pct:>9} {r['similarity']:>7.3f} "
                              f"{r.get('archetype', '?'):>22}")

        if args.embed_new:
            from tracker.ml.training import embed_new
            embed_new(session, model_dir=_model_dir)

        if args.train_wp:
            from tracker.ml.wp_training import train_wp
            train_wp(session, model_dir=_model_dir, unfreeze_encoder=args.wp_unfreeze)

        if args.wp_infer:
            from tracker.ml.wp_training import infer_wp
            infer_wp(session, model_dir=_model_dir)

        if args.wp_infer_new:
            from tracker.ml.wp_training import infer_wp_incremental
            n = infer_wp_incremental(session, model_dir=_model_dir)
            if n == -1:
                print("  ✗ No trained WP model found. Run --train-wp first.")
            elif n == 0:
                print("  · No new games to process")
            else:
                print(f"  ✓ WP inference: {n} new games processed")

        if args.wp:
            from tracker.ml.wp_storage import WinProbability
            rows = session.query(WinProbability).filter_by(
                battle_id=args.wp,
            ).order_by(WinProbability.game_tick).all()
            if not rows:
                print(f"  ✗ No win probability data for {args.wp}")
                print("    Run --train-wp first")
            else:
                reporting.print_wp_curve(rows, args.wp)

        if args.wp_critical:
            from tracker.ml.wp_storage import WinProbability
            rows = session.query(WinProbability).filter_by(
                battle_id=args.wp_critical,
            ).order_by(WinProbability.criticality.desc()).limit(10).all()
            if not rows:
                print(f"  ✗ No win probability data for {args.wp_critical}")
                print("    Run --train-wp first")
            else:
                reporting.print_wp_critical(rows, args.wp_critical)

        if args.wp_cards:
            from tracker.ml.wp_storage import GameWPSummary
            from tracker.models import Battle
            from sqlalchemy import func, select
            rows = session.execute(
                select(GameWPSummary)
                .join(Battle, Battle.battle_id == GameWPSummary.battle_id)
                .where(Battle.corpus == "personal")
            ).scalars().all()
            if not rows:
                print("  ✗ No WP data for personal games. Run --wp-infer first.")
            else:
                reporting.print_wp_cards(rows)

        if args.train_cvae:
            from tracker.ml.cvae_training import train_cvae
            train_cvae(session, model_dir=_model_dir)

        if args.counterfactual:
            battle_id, old_card, new_card = args.counterfactual
            from tracker.ml.counterfactual import CounterfactualGenerator
            gen = CounterfactualGenerator(session, model_dir=_model_dir)
            if gen.cvae is None:
                print("  ✗ No trained CVAE model. Run --train-cvae first.")
            elif gen.wp_model is None:
                print("  ✗ No trained WP model. Run --train-wp first.")
            else:
                result = gen.run_counterfactual(
                    battle_id, old_card, new_card, n_samples=args.cf_samples,
                )
                if result:
                    reporting.print_counterfactual(result)
                else:
                    print(f"  ✗ Could not generate counterfactual for {battle_id}")

        if args.deck_gradient:
            from tracker.ml.counterfactual import CounterfactualGenerator
            gen = CounterfactualGenerator(session, model_dir=_model_dir)
            if gen.cvae is None:
                print("  ✗ No trained CVAE model. Run --train-cvae first.")
            elif gen.wp_model is None:
                print("  ✗ No trained WP model. Run --train-wp first.")
            else:
                results = gen.compute_deck_gradient(
                    n_games=20, n_samples=args.cf_samples,
                )
                reporting.print_deck_gradient(results)

        if args.train_activity_model:
            from tracker.ml.activity_model import train_activity_model
            metrics = train_activity_model(session, model_dir=_model_dir)
            if metrics is None:
                print("  ✗ Insufficient data to train activity model")
            else:
                print(f"  ✓ Activity model trained: accuracy={metrics['accuracy']:.3f}, "
                      f"AUC={metrics['auc']:.3f}, "
                      f"{metrics['players_profiled']} players profiled")

        if args.mark_stale_replays:
            from tracker.corpus_scraper import mark_stale_replays
            count = mark_stale_replays(session)
            print(f"  ✓ Marked {count:,} stale battles as permanently missed")

        if args.manifold:
            from tracker.ml.cluster_profiler import profile_manifold
            data = profile_manifold(session)
            if "error" in data:
                print(f"\n  ✗ {data['error']}")
            elif export_fmt:
                export_data(data, export_fmt, export_out)
            else:
                reporting.print_manifold(data)

        if args.matchup_dive:
            from tracker.temporal_analysis import matchup_deep_dive
            data = matchup_deep_dive(session, args.matchup_dive, min_trophies=args.min_trophies)
            if "error" in data and data.get("game_count", -1) == 0:
                print(f"\n  ✗ {data['error']}")
            elif "error" in data:
                print(f"\n  ✗ {data['error']}")
                if "known" in data:
                    print(f"    Known archetypes: {', '.join(data['known'][:10])}...")
            elif export_fmt:
                export_data(data, export_fmt, export_out)
            else:
                reporting.print_matchup_dive(data)

        if args.broken_cycle:
            from tracker.temporal_analysis import broken_cycle
            pairs = []
            for spec in args.broken_cycle:
                parts = spec.split(":")
                if len(parts) != 2:
                    print(f"  ✗ Invalid pair format '{spec}' — use card-a:card-b")
                    continue
                pairs.append((parts[0], parts[1]))
            if pairs:
                results = broken_cycle(session, pairs)
                if export_fmt:
                    export_data(results, export_fmt, export_out)
                else:
                    reporting.print_broken_cycle(results)

        # Default: show help + db status
        has_action = any([
            args.fetch, args.stats, args.deck_stats, args.crowns,
            args.matchups, args.recent, args.streaks, args.rolling,
            args.trophy_history, args.archetypes,
            args.replay_login, args.replay_check, args.fetch_replays,
            args.personal_combined,
            args.corpus_update, args.corpus_scrape, args.corpus_replays,
            args.corpus_combined,
            args.corpus_stats, args.corpus_add_priority,
            args.corpus_discover, args.corpus_locations, args.corpus_nemeses,
            args.sim_matchups, args.sim_interactions, args.sim_elixir,
            args.sim_hands, args.sim_full,
            args.tilt_check, args.train_tcn, args.build_features, args.train_embeddings,
            args.clusters, args.similar, args.embed_new,
            args.train_wp, args.wp_infer, args.wp_infer_new, args.wp, args.wp_critical, args.wp_cards,
            args.train_cvae, args.counterfactual, args.deck_gradient,
            args.manifold, args.train_activity_model,
            args.matchup_dive, args.broken_cycle, args.mark_stale_replays,
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
