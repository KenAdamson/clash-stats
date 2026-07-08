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

# Progression badges every account accrues passively; anything else that isn't
# a Mastery* badge is an event/mode badge (seasonal events, 2v2, Draft, ...).
# Fleet bots grind one mode forever and never touch events — see the
# badge-sheet fingerprint calibration (2026-07-06).
_CORE_BADGES = frozenset({
    "YearsPlayed", "BattleWins", "ClanWarWins", "ClanDonations",
    "CollectionLevel", "EmoteCollection", "BannerCollection", "ClanWarsVeteran",
})


# Top-of-band re-seed (2026-07-08): corpus = top 5% (by trophies) of each
# high ladder band, PvP-only. Rationale: bots park at each gate FLOOR, so the
# top slice of each band is structurally bot-free and tracks the genuinely
# skilled/climbing players; ladder (PvP) games have RoyaleAPI replays (ranked
# does not) → high replay density. Opponents stay in-tier (trophy matchmaking)
# so discovery converges rather than explodes.
BAND_EDGES = (12000, 12500, 13000, 13500, 14000)
TOP_OF_BAND_PCT = 0.05
TOP_OF_BAND_RECENCY_DAYS = 14


def select_top_of_band_tags(
    session: Session,
    pct: float = TOP_OF_BAND_PCT,
    recency_days: int = TOP_OF_BAND_RECENCY_DAYS,
) -> list[str]:
    """Tags in the top ``pct`` by current trophies within each ladder band.

    Uses each player's most recent PvP trophy observation from EITHER side of
    a battle (so opponents we've only seen are eligible), within the recency
    window. Returns the union across bands. Reused by the one-time re-seed and
    the periodic rising-star sampler.
    """
    rows = session.execute(
        text("""
            WITH obs AS (
                SELECT player_tag AS tag, player_starting_trophies AS tr,
                       battle_time bt
                FROM battles
                WHERE battle_type='PvP'
                  AND player_starting_trophies BETWEEN :lo AND :hi
                  AND battle_time > now() - (:days || ' days')::interval
                UNION ALL
                SELECT opponent_tag, opponent_starting_trophies, battle_time
                FROM battles
                WHERE battle_type='PvP'
                  AND opponent_starting_trophies BETWEEN :lo AND :hi
                  AND opponent_tag IS NOT NULL
                  AND battle_time > now() - (:days || ' days')::interval
            ),
            latest AS (
                SELECT DISTINCT ON (tag) tag, tr, width_bucket(tr, :edges) band
                FROM obs ORDER BY tag, bt DESC
            ),
            thresh AS (
                SELECT band, percentile_cont(1 - :pct)
                       WITHIN GROUP (ORDER BY tr) p
                FROM latest GROUP BY band
            )
            SELECT l.tag FROM latest l JOIN thresh t ON l.band = t.band
            WHERE l.tr >= t.p
        """),
        {
            "lo": BAND_EDGES[0], "hi": BAND_EDGES[-1] - 1,
            "days": str(recency_days), "pct": pct,
            "edges": list(BAND_EDGES[1:-1]),
        },
    ).scalars().all()
    return list(rows)


def reseed_top_of_band(
    session: Session,
    pct: float = TOP_OF_BAND_PCT,
    recency_days: int = TOP_OF_BAND_RECENCY_DAYS,
    archive: bool = True,
    protected_sources: tuple = ("nemesis", "priority", "watchlist"),
) -> dict:
    """Re-seed the active corpus to the top-of-band players.

    Idempotent: upserts the seed as active ``source='top_ladder'``; when
    ``archive`` is True, deactivates every currently-active player NOT in the
    seed and NOT in ``protected_sources`` (source→'archived', reversible via
    the pre-run snapshot; historical battles/replays untouched). Discovery
    then re-grows the in-tier opponent set organically.
    """
    seed = set(select_top_of_band_tags(session, pct, recency_days))
    if not seed:
        return {"seed": 0, "activated": 0, "archived": 0}

    # Archive FIRST (deactivate every active player except protected sources),
    # THEN activate the seed — so stale players carrying any source, including
    # a prior 'top_ladder', are correctly dropped unless they're in the fresh
    # seed. Seed players re-activate in the next step regardless of prior state.
    archived = 0
    if archive:
        res = session.execute(
            update(PlayerCorpus)
            .where(PlayerCorpus.active == 1)
            .where(PlayerCorpus.source.notin_(protected_sources))
            .values(active=0, source="archived")
        )
        archived = res.rowcount or 0
        session.commit()

    seed_list = list(seed)
    activated = 0
    for i in range(0, len(seed_list), 500):
        chunk = seed_list[i:i + 500]
        existing = {
            r[0] for r in session.execute(
                select(PlayerCorpus.player_tag)
                .where(PlayerCorpus.player_tag.in_(chunk))
            )
        }
        for tag in chunk:
            if tag in existing:
                session.execute(
                    update(PlayerCorpus)
                    .where(PlayerCorpus.player_tag == tag)
                    .values(active=1, source="top_ladder")
                )
            else:
                session.add(PlayerCorpus(
                    player_tag=tag, source="top_ladder", active=1,
                ))
            activated += 1
    session.commit()

    logger.info(
        "reseed_top_of_band: seed=%d activated=%d archived=%d",
        len(seed), activated, archived,
    )
    return {"seed": len(seed), "activated": activated, "archived": archived}


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


def _log_polling_batch(scored: list[str], floor: list[str]) -> None:
    """Append batch composition to a JSONL for scored-vs-floor yield A/B.

    The 20% FIFO-floor slots inside every scored batch are a built-in
    control group: joining these memberships against battles ingested in
    the run window measures the model's real hit-rate lift, immune to
    catch-up/transition effects. Measurement infrastructure only.
    """
    import json
    from datetime import datetime, timezone
    try:
        with open("data/polling_batches.jsonl", "a") as f:
            f.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                "scored": scored,
                "floor": floor,
            }) + "\n")
    except OSError:
        pass


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
        prioritize_active: If True and an activity model exists, select by
            P(has_new_battles): score the FULL candidate pool, then take the
            batch as ~80% top-scored + ~20% oldest-FIFO (exploration floor).
            Scoring must happen BEFORE the limit — the original post-limit
            reorder shuffled a batch whose membership FIFO had already fixed,
            so the model influenced nothing (root of the polling-efficiency
            gap found 2026-07-07). The FIFO floor guarantees every player is
            still visited eventually, so a bad model can slow the queue but
            never starve it. Priority-source players always lead the batch.
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
    if limit and not prioritize_active:
        stmt = stmt.limit(limit)

    players = list(session.scalars(stmt).all())

    # Score-then-limit selection when requested and a model exists
    if prioritize_active and players:
        try:
            from tracker.ml.activity_model import score_corpus_players
            _mdir = model_dir or "data/ml_models"
            scores = score_corpus_players(session, model_dir=_mdir)
            if scores is not None:
                score_map = dict(scores)

                priority = [p for p in players if p.source == "priority"]
                rest = [p for p in players if p.source != "priority"]
                by_score = sorted(
                    rest, key=lambda p: score_map.get(p.player_tag, 0.0),
                    reverse=True,
                )

                if limit and len(players) > limit:
                    n_explore = max(1, int(limit * 0.2))
                    n_scored = max(0, limit - len(priority) - n_explore)
                    batch = list(priority) + by_score[:n_scored]
                    chosen = {p.player_tag for p in batch}
                    # exploration floor: oldest-FIFO among the not-yet-chosen
                    # (`rest` still carries the SQL FIFO ordering)
                    explore = [p for p in rest if p.player_tag not in chosen]
                    batch += explore[:n_explore]
                    players = batch[:limit]
                    logger.info(
                        "Activity model: score-then-limit %d candidates -> "
                        "%d batch (%d priority + %d scored + %d FIFO floor)",
                        len(rest) + len(priority), len(players),
                        len(priority), n_scored, min(n_explore, len(explore)),
                    )
                    _log_polling_batch(
                        scored=[p.player_tag for p in by_score[:n_scored]],
                        floor=[p.player_tag for p in explore[:n_explore]],
                    )
                else:
                    players = priority + by_score
        except Exception as e:
            logger.warning("Activity model scoring failed, using FIFO: %s", e)
            if limit:
                players = players[:limit]
    elif prioritize_active and limit:
        players = players[:limit]

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
    badge_bot_eff_max: float = 0.85,
    badge_bot_min_battles: int = 5000,
    badge_bot_max_badges: int = 30,
    badge_backfill_max: int = 5000,
    cache_path: str = "/app/data/corpus_enrichment.pkl",
) -> dict:
    """Periodic corpus tidy — wired to ``--prune-corpus`` (weekly cron).

    Keeps the tracking list lean so the FIFO scraper re-polls the survivors
    more often (higher captured-games-per-player density). Three reversible
    passes; ``source='priority'`` and ``source='watchlist'`` are never touched:

    1. **Enrich** new active players with battleCount/bestTrophies/clan from
       ``/players`` (cached — only never-seen tags cost an API call).
    2. **Bots**: deactivate accounts that grind without progressing — high
       battleCount, low ``best/battleCount`` efficiency, *and* clanless (the
       clanless gate spares legit clanned grinders). ``source='bot'`` is
       permanent (never re-discovered).

       A second, badge-corroborated pass covers the efficiency gray zone
       (``bot_eff_max`` < eff <= ``badge_bot_eff_max``) where grinding humans
       and bots are indistinguishable by efficiency alone: a clanless
       gray-zone account is a bot if its badge sheet is skeletal — <=
       ``badge_bot_max_badges`` badges, or no YearsPlayed badge with <=2
       event badges. Calibration (2026-07-06, n=60/59): hits 40% of
       known-fleet accounts (the young-generation subtype) at 0-2% human
       false-positive rate; humans inevitably accrue event badges and age
       into YearsPlayed.
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

    # 1. enrich active players: never-seen tags, plus pre-badge-era entries
    # that lack badge fields (self-healing backfill, capped per run so a large
    # legacy cache amortizes over several weekly runs instead of one marathon).
    # Dead entries (None = profile fetch failed permanently) are not retried.
    enriched = 0
    backfilled = 0
    for tag in active:
        v = cache.get(tag)
        if tag in cache and (v is None or "n_badges" in v):
            continue
        if tag in cache:
            if backfilled >= badge_backfill_max:
                continue
            backfilled += 1
        try:
            p = api.get_player(tag)
            badge_names = [b.get("name", "") for b in (p.get("badges") or [])]
            cache[tag] = {
                "bc": p.get("battleCount", 0),
                "best": p.get("bestTrophies", 0),
                "clan": (p.get("clan") or {}).get("tag"),
                "n_badges": len(badge_names),
                "has_years_played": "YearsPlayed" in badge_names,
                "n_event_badges": sum(
                    1 for n in badge_names
                    if not n.startswith("Mastery") and n not in _CORE_BADGES
                ),
            }
        except (APIError, Exception):
            # keep a stale-but-real entry over clobbering it with None
            cache.setdefault(tag, None)
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

    # 2b. badge-corroborated pass for the efficiency gray zone: skeletal badge
    # sheets condemn accounts whose efficiency alone can't (see docstring).
    # Entries enriched before badge fields existed lack "n_badges" — skipped
    # until their next re-enrichment.
    def _is_badge_bot(tag: str) -> bool:
        v = cache.get(tag)
        if not v or not v.get("bc") or "n_badges" not in v:
            return False
        if v["clan"] is not None or v["bc"] < badge_bot_min_battles:
            return False
        eff = v["best"] / v["bc"]
        if not (bot_eff_max < eff <= badge_bot_eff_max):
            return False
        return (v["n_badges"] <= badge_bot_max_badges
                or (not v["has_years_played"] and v["n_event_badges"] <= 2))

    bots = [t for t in active if _is_bot(t) or _is_badge_bot(t)]
    bot_n = 0
    for i in range(0, len(bots), 500):
        res = session.execute(
            update(PlayerCorpus)
            .where(PlayerCorpus.player_tag.in_(bots[i:i + 500]))
            .where(PlayerCorpus.source.notin_(("priority", "watchlist")))
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
        WHERE pc.active = 1 AND pc.source NOT IN ('priority', 'watchlist') AND la.last_game < :cutoff
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
            WHERE pc.active = 1 AND pc.source NOT IN ('priority', 'watchlist')
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
