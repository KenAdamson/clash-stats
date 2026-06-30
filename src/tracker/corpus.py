"""Training data corpus management (ADR-007).

Collects top-ladder player tags from the CR API and manages the
player_corpus table for batch replay scraping.
"""

import logging
import os
import pickle
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func, select, text, update
from sqlalchemy.orm import Session

from tracker.api import APIError, ClashRoyaleAPI
from tracker.models import Battle, PlayerCorpus

logger = logging.getLogger(__name__)

# Players below this Trophy-Road floor (but above the alt range) are no longer
# relevant to a 12k+ main and just dilute scrape budget — corpus_hygiene prunes
# them and discovery stops adding them. Raise as the main climbs.
RELEVANT_TROPHY_FLOOR = 12000
ALT_TROPHY_FLOOR = 5000


def update_top_ladder(
    session: Session,
    api: ClashRoyaleAPI,
    location_id: str = "global",
    limit: int = 200,
) -> int:
    """Fetch top-ladder player tags and upsert into player_corpus.

    Args:
        session: SQLAlchemy session.
        api: ClashRoyaleAPI client.
        location_id: Location ID or 'global'.
        limit: Number of players to fetch (max 200).

    Returns:
        Number of new players added.
    """
    players = api.get_top_players(location_id=location_id, limit=limit)
    logger.info("Fetched %d players from %s leaderboard.", len(players), location_id)

    added = 0
    for p in players:
        tag = p.get("tag", "").lstrip("#")
        if not tag:
            continue

        existing = session.get(PlayerCorpus, f"#{tag}")
        # Use eloRating (Path of Legend) or trophies, whichever is available
        trophies = p.get("eloRating") or p.get("trophies", 0)
        if existing:
            existing.player_name = p.get("name")
            if existing.trophy_range_high is None or trophies > existing.trophy_range_high:
                existing.trophy_range_high = trophies
            if existing.trophy_range_low is None or trophies < existing.trophy_range_low:
                existing.trophy_range_low = trophies
        else:
            session.add(PlayerCorpus(
                player_tag=f"#{tag}",
                player_name=p.get("name"),
                source="top_ladder",
                trophy_range_low=trophies,
                trophy_range_high=trophies,
                active=1,
            ))
            added += 1

    session.commit()
    logger.info("Corpus update: %d new players, %d total.", added, len(players))
    return added


def add_manual_player(
    session: Session,
    player_tag: str,
    source: str = "manual",
    player_name: Optional[str] = None,
) -> bool:
    """Add a player to the corpus manually.

    Args:
        session: SQLAlchemy session.
        player_tag: Player tag (with or without #).
        source: Provenance label.
        player_name: Optional display name.

    Returns:
        True if added, False if already exists.
    """
    tag = f"#{player_tag.lstrip('#')}"
    existing = session.get(PlayerCorpus, tag)
    if existing:
        if existing.source != source:
            existing.source = source
            session.commit()
        return False

    session.add(PlayerCorpus(
        player_tag=tag,
        player_name=player_name,
        source=source,
        active=1,
    ))
    session.commit()
    return True


def get_corpus_players(
    session: Session,
    active_only: bool = True,
    source: Optional[str] = None,
    limit: Optional[int] = None,
    prioritize_active: bool = False,
    model_dir: Optional[str] = None,
) -> list[PlayerCorpus]:
    """Get corpus players, ordered by least recently scraped.

    Args:
        session: SQLAlchemy session.
        active_only: Only return active players.
        source: Filter by source type.
        limit: Maximum players to return.
        prioritize_active: If True and an activity model exists, sort by
            P(has_new_battles) descending instead of FIFO. Priority/nemesis
            players are still boosted to the top.
        model_dir: Directory containing trained ML models.

    Returns:
        List of PlayerCorpus objects.
    """
    stmt = select(PlayerCorpus)
    if active_only:
        stmt = stmt.where(PlayerCorpus.active == 1)
    if source:
        stmt = stmt.where(PlayerCorpus.source == source)

    # Default ordering: priority first, then never-scraped, then FIFO
    stmt = stmt.order_by(
        (PlayerCorpus.source == "priority").desc(),
        PlayerCorpus.last_scraped.is_(None).desc(),
        PlayerCorpus.last_scraped.asc(),
    )
    if limit:
        stmt = stmt.limit(limit)

    players = list(session.scalars(stmt).all())

    # ML-based reordering if requested and model exists
    if prioritize_active and players:
        try:
            from tracker.ml.activity_model import score_corpus_players
            _mdir = model_dir or "data/ml_models"
            logger.info("Activity model: scoring %d players...", len(players))
            scores = score_corpus_players(session, model_dir=_mdir)
            if scores is not None:
                score_map = dict(scores)

                # Partition: priority/nemesis players stay at the top
                priority = [p for p in players if p.source in ("priority", "nemesis")]
                rest = [p for p in players if p.source not in ("priority", "nemesis")]

                # Sort non-priority players by activity score descending
                rest.sort(
                    key=lambda p: score_map.get(p.player_tag, 0.0),
                    reverse=True,
                )

                players = priority + rest

                # Log score distribution
                if rest:
                    rest_scores = [score_map.get(p.player_tag, 0.0) for p in rest]
                    top_n = min(500, len(rest_scores))
                    bot_n = min(500, len(rest_scores))
                    logger.info(
                        "Activity model: scored %d players — "
                        "top %d have P(active) > %.2f, "
                        "bottom %d < %.2f",
                        len(rest_scores),
                        top_n, rest_scores[top_n - 1] if top_n <= len(rest_scores) else 0.0,
                        bot_n, rest_scores[-bot_n] if bot_n <= len(rest_scores) else 0.0,
                    )
        except Exception as e:
            logger.warning("Activity model scoring failed, using FIFO: %s", e)

    return players


def mark_player_scraped(
    session: Session,
    player_tag: str,
    games: int = 0,
    replays: int = 0,
) -> None:
    """Update scraping stats for a corpus player.

    Args:
        session: SQLAlchemy session.
        player_tag: Player tag (with #).
        games: Number of new games scraped.
        replays: Number of new replays scraped.
    """
    player = session.get(PlayerCorpus, player_tag)
    if player:
        player.games_scraped = (player.games_scraped or 0) + games
        player.replays_scraped = (player.replays_scraped or 0) + replays
        player.last_scraped = datetime.utcnow()
        session.commit()


def discover_from_opponents(
    session: Session,
    min_trophies: int = RELEVANT_TROPHY_FLOOR,
    max_players: int = 200,
) -> int:
    """Mine opponent tags from existing corpus battles and add to corpus.

    Finds players who appeared as opponents in corpus battles but aren't
    yet tracked. This grows the network organically — every player we
    track exposes 25 new opponents per scrape.

    Args:
        session: SQLAlchemy session.
        min_trophies: Minimum startingTrophies to include (0 = all).
        max_players: Maximum new players to add per run.

    Returns:
        Number of new players added.
    """
    # Map existing tags -> (active, source). Lets us REACTIVATE a player that
    # corpus_hygiene previously parked as 'dormant'/'dead' but who is now
    # reappearing as an opponent (i.e. they're playing again) — self-healing.
    # Bots are never revived.
    existing = {
        row[0]: (row[1], row[2])
        for row in session.execute(
            select(PlayerCorpus.player_tag, PlayerCorpus.active, PlayerCorpus.source)
        ).all()
    }

    # Get opponent tags from corpus battles, with their names and trophy data
    rows = session.execute(
        select(
            Battle.opponent_tag,
            Battle.opponent_name,
            func.MAX(Battle.opponent_starting_trophies).label("max_trophies"),
            func.MIN(Battle.opponent_starting_trophies).label("min_trophies"),
            func.COUNT(Battle.battle_id).label("appearances"),
        )
        .where(Battle.corpus.isnot(None))
        .where(Battle.opponent_tag.isnot(None))
        .group_by(Battle.opponent_tag)
        .order_by(func.COUNT(Battle.battle_id).desc())
    ).all()

    added = 0
    reactivated = 0
    for row in rows:
        if added >= max_players:
            break

        tag = row.opponent_tag
        if not tag:
            continue
        if tag in existing:
            act, src = existing[tag]
            # Revive a parked real player ONLY if they're reappearing at/above
            # the relevant tier (climbers come back; sub-tier and bots don't).
            if (act == 0 and src in ("dormant", "dead", "below_tier")
                    and (row.max_trophies or 0) >= min_trophies):
                session.execute(
                    update(PlayerCorpus)
                    .where(PlayerCorpus.player_tag == tag)
                    .values(active=1, source="network")
                )
                reactivated += 1
            continue  # already tracked (active / bot / just-revived) — don't re-add

        # Trophy filter (0 means unknown — include those too since
        # Path of Legend uses a different rating scale)
        max_trophy = row.max_trophies or 0
        if min_trophies > 0 and 0 < max_trophy < min_trophies:
            continue

        session.add(PlayerCorpus(
            player_tag=tag,
            player_name=row.opponent_name,
            source="network",
            trophy_range_low=row.min_trophies if row.min_trophies and row.min_trophies > 0 else None,
            trophy_range_high=row.max_trophies if row.max_trophies and row.max_trophies > 0 else None,
            active=1,
        ))
        added += 1

    session.commit()
    total_corpus = session.scalar(
        select(func.count()).select_from(PlayerCorpus).where(PlayerCorpus.active == 1)
    ) or 0
    logger.info(
        "Network discovery: %d new, %d reactivated from opponent tags (%d total active).",
        added, reactivated, total_corpus,
    )
    return added


def corpus_hygiene(
    session: Session,
    api: ClashRoyaleAPI,
    dormant_days: int = 14,
    min_trophy: int = RELEVANT_TROPHY_FLOOR,
    bot_eff_max: float = 0.3,
    bot_min_battles: int = 10000,
    cache_path: str = "/app/data/corpus_enrichment.pkl",
) -> dict:
    """Periodic corpus tidy — wired to ``--prune-corpus`` (weekly cron).

    Keeps the tracking list lean so the FIFO scraper re-polls the survivors
    more often (higher captured-games-per-player density). Three reversible
    passes; ``source='priority'`` is never touched:

    1. **Enrich** new active players with battleCount/bestTrophies/clan from
       ``/players`` (cached — only never-seen tags cost an API call).
    2. **Bots**: deactivate accounts that grind without progressing — high
       battleCount, low ``best/battleCount`` efficiency, *and* clanless (the
       clanless gate spares legit clanned grinders). ``source='bot'`` is
       permanent (never re-discovered).
    3. **Dormant**: deactivate accounts with no captured game in
       ``dormant_days``. ``source='dormant'`` — :func:`discover_from_opponents`
       revives them automatically if they start playing again.

    Returns counts: ``enriched``, ``bots``, ``dormant``, ``active``.
    """
    cache: dict = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            cache = {}

    active = [r[0] for r in session.execute(
        select(PlayerCorpus.player_tag).where(PlayerCorpus.active == 1)
    )]

    # 1. enrich never-seen active players
    enriched = 0
    for tag in active:
        if tag in cache:
            continue
        try:
            p = api.get_player(tag)
            cache[tag] = {
                "bc": p.get("battleCount", 0),
                "best": p.get("bestTrophies", 0),
                "clan": (p.get("clan") or {}).get("tag"),
            }
        except (APIError, Exception):
            cache[tag] = None
        enriched += 1
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
    except Exception:
        logger.warning("corpus_hygiene: could not persist enrichment cache")

    # 2. bot prune (clanless + high-volume + low progression efficiency)
    def _is_bot(tag: str) -> bool:
        v = cache.get(tag)
        if not v or not v.get("bc"):
            return False
        return (v["bc"] >= bot_min_battles
                and (v["best"] / v["bc"]) <= bot_eff_max
                and v["clan"] is None)

    bots = [t for t in active if _is_bot(t)]
    bot_n = 0
    for i in range(0, len(bots), 500):
        res = session.execute(
            update(PlayerCorpus)
            .where(PlayerCorpus.player_tag.in_(bots[i:i + 500]))
            .where(PlayerCorpus.source != "priority")
            .values(active=0, source="bot")
        )
        bot_n += res.rowcount or 0
    if bots:
        session.commit()

    # 3. dormant prune (latest captured game older than the cutoff)
    cutoff = datetime.now(timezone.utc) - timedelta(days=dormant_days)
    dormant = [r[0] for r in session.execute(text("""
        SELECT pc.player_tag
        FROM player_corpus pc
        JOIN (
            SELECT player_tag, max(battle_time) AS last_game
            FROM battles WHERE corpus = 'top_ladder'
            GROUP BY player_tag
        ) la ON la.player_tag = pc.player_tag
        WHERE pc.active = 1 AND pc.source <> 'priority' AND la.last_game < :cutoff
    """), {"cutoff": cutoff})]
    dorm_n = 0
    for i in range(0, len(dormant), 500):
        res = session.execute(
            update(PlayerCorpus)
            .where(PlayerCorpus.player_tag.in_(dormant[i:i + 500]))
            .values(active=0, source="dormant")
        )
        dorm_n += res.rowcount or 0
    if dormant:
        session.commit()

    # 4. trophy-tier prune: drop sub-tier players (median Trophy-Road standing in
    # [ALT_TROPHY_FLOOR, min_trophy)) — no longer relevant to a min_trophy+ main.
    # Uses median over PvP games only (ranked ratings are seasonally reset).
    # Self-heals: a climber reappears as a >=min_trophy opponent and discovery
    # reactivates them.
    tier_n = 0
    if min_trophy and min_trophy > 0:
        below = [r[0] for r in session.execute(text("""
            SELECT pc.player_tag
            FROM player_corpus pc
            JOIN (
                SELECT player_tag,
                       percentile_cont(0.5) within group (order by player_starting_trophies) tr
                FROM battles
                WHERE corpus = 'top_ladder' AND battle_type = 'PvP'
                  AND player_starting_trophies > 0
                GROUP BY player_tag
            ) pt ON pt.player_tag = pc.player_tag
            WHERE pc.active = 1 AND pc.source <> 'priority'
              AND pt.tr >= :lo AND pt.tr < :hi
        """), {"lo": ALT_TROPHY_FLOOR, "hi": min_trophy})]
        for i in range(0, len(below), 500):
            res = session.execute(
                update(PlayerCorpus)
                .where(PlayerCorpus.player_tag.in_(below[i:i + 500]))
                .values(active=0, source="below_tier")
            )
            tier_n += res.rowcount or 0
        if below:
            session.commit()

    remaining = session.scalar(
        select(func.count()).select_from(PlayerCorpus).where(PlayerCorpus.active == 1)
    ) or 0
    logger.info("corpus_hygiene: enriched %d, -%d bots, -%d dormant, -%d sub-tier, %d active remain",
                enriched, bot_n, dorm_n, tier_n, remaining)
    return {"enriched": enriched, "bots": bot_n, "dormant": dorm_n,
            "below_tier": tier_n, "active": remaining}


def discover_nemeses(
    session: Session,
    player_tag: str,
) -> int:
    """Add opponents the player has lost to into the corpus.

    Also promotes existing corpus players to 'nemesis' source so they
    get prioritized in scrape ordering.

    Args:
        session: SQLAlchemy session.
        player_tag: Player tag (with or without #).

    Returns:
        Number of new players added.
    """
    tag = f"#{player_tag.lstrip('#')}"

    # Find opponent tags from losses not already in corpus
    existing_tags = set(
        row[0] for row in session.execute(
            select(PlayerCorpus.player_tag)
        ).all()
    )

    rows = session.execute(
        select(
            Battle.opponent_tag,
            Battle.opponent_name,
        )
        .where(Battle.player_tag == tag)
        .where(Battle.result == "loss")
        .where(Battle.opponent_tag.isnot(None))
        .distinct()
    ).all()

    added = 0
    promoted = 0
    for row in rows:
        opp_tag = row.opponent_tag
        if not opp_tag:
            continue

        if opp_tag not in existing_tags:
            session.add(PlayerCorpus(
                player_tag=opp_tag,
                player_name=row.opponent_name,
                source="nemesis",
                active=1,
            ))
            added += 1
            existing_tags.add(opp_tag)
        else:
            # Promote existing non-priority players to nemesis
            existing = session.get(PlayerCorpus, opp_tag)
            if existing and existing.source not in ("priority", "nemesis"):
                existing.source = "nemesis"
                promoted += 1

    session.commit()
    logger.info(
        "Nemesis discovery for %s: %d new, %d promoted, %d total nemeses.",
        tag, added, promoted, added + promoted,
    )
    return added


def update_location_leaderboards(
    session: Session,
    api: ClashRoyaleAPI,
    location_ids: list[str] | None = None,
    limit: int = 200,
) -> int:
    """Fetch players from location-specific leaderboards.

    Location leaderboards go deeper than global — useful for finding
    players in the 8000-11000 trophy range who aren't on the global top 200.

    Args:
        session: SQLAlchemy session.
        api: ClashRoyaleAPI client.
        location_ids: List of location IDs. Defaults to major regions.
        limit: Players per leaderboard (max 200).

    Returns:
        Number of new players added across all locations.
    """
    if location_ids is None:
        # Major regions with deep ladder pools
        location_ids = [
            "57000249",  # United States
            "57000056",  # China
            "57000109",  # Japan
            "57000138",  # South Korea
            "57000034",  # Brazil
            "57000070",  # France
            "57000077",  # Germany
            "57000224",  # Turkey
            "57000183",  # Russia
            "57000094",  # Indonesia
        ]

    total_added = 0
    for loc_id in location_ids:
        try:
            players = api.get_top_players(location_id=loc_id, limit=limit)
            logger.info("Fetched %d players from location %s.", len(players), loc_id)

            for p in players:
                tag = p.get("tag", "").lstrip("#")
                if not tag:
                    continue

                existing = session.get(PlayerCorpus, f"#{tag}")
                trophies = p.get("eloRating") or p.get("trophies", 0)
                if existing:
                    existing.player_name = p.get("name")
                    if existing.trophy_range_high is None or trophies > existing.trophy_range_high:
                        existing.trophy_range_high = trophies
                    if existing.trophy_range_low is None or trophies < existing.trophy_range_low:
                        existing.trophy_range_low = trophies
                else:
                    session.add(PlayerCorpus(
                        player_tag=f"#{tag}",
                        player_name=p.get("name"),
                        source="location_ladder",
                        trophy_range_low=trophies,
                        trophy_range_high=trophies,
                        active=1,
                    ))
                    total_added += 1

            session.commit()
        except Exception as e:
            logger.warning("Error fetching location %s: %s", loc_id, e)
            continue

    logger.info("Location discovery: %d new players across %d regions.",
                total_added, len(location_ids))
    return total_added


def get_corpus_stats(session: Session) -> dict:
    """Get summary statistics about the corpus.

    Returns:
        Dict with counts, source breakdown, and coverage stats.
    """
    total_players = session.scalar(
        select(func.count()).select_from(PlayerCorpus)
    ) or 0
    active_players = session.scalar(
        select(func.count()).select_from(PlayerCorpus).where(PlayerCorpus.active == 1)
    ) or 0

    # Games by corpus type
    corpus_counts = {}
    rows = session.execute(
        select(Battle.corpus, func.count()).group_by(Battle.corpus)
    ).all()
    for row in rows:
        corpus_counts[row[0] or "personal"] = row[1]

    # Source breakdown
    source_counts = {}
    rows = session.execute(
        select(PlayerCorpus.source, func.count()).group_by(PlayerCorpus.source)
    ).all()
    for row in rows:
        source_counts[row[0]] = row[1]

    # Replay coverage
    total_battles = session.scalar(
        select(func.count()).select_from(Battle)
    ) or 0
    battles_with_replays = session.scalar(
        select(func.count()).select_from(Battle).where(Battle.replay_fetched == 1)
    ) or 0

    return {
        "total_players": total_players,
        "active_players": active_players,
        "source_breakdown": source_counts,
        "battles_by_corpus": corpus_counts,
        "total_battles": total_battles,
        "battles_with_replays": battles_with_replays,
        "replay_coverage_pct": round(
            battles_with_replays / total_battles * 100, 1
        ) if total_battles > 0 else 0.0,
    }
