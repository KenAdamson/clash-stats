"""Derived analytics dimensions: clan_dim and player_dim.

Both tables are *fully derived* — repopulatable at any time, never a source of
truth:

* :func:`refresh_clan_dim` rebuilds ``clan_dim`` from the CR clan API for every
  clan we've actually seen as an opponent's clan in ``battles``.
* :func:`refresh_player_dim` rebuilds ``player_dim`` by aggregating the
  ``battles`` table over ``opponent_tag`` (each opponent is a player), enriched
  with the opponent's clan tag from battle ``raw_json``.

Both are wired to the ``--refresh-dims`` CLI flag and intended to run as a
daily cron job (see ``refresh_dims.sh`` sketched in ``entrypoint.sh``).
"""

import logging
from collections import defaultdict
from datetime import datetime, timezone
from statistics import median
from typing import Optional

from sqlalchemy import case, func, select, text
from sqlalchemy.orm import Session

from tracker.api import APIError, ClashRoyaleAPI
from tracker.models import Battle, ClanDim, LevelTrophyRef, PlayerDim, PlayerKing

logger = logging.getLogger(__name__)

# Trophy thresholds for the n_9k / n_11k / n_12k member buckets.
_TROPHY_BUCKETS = ((9000, "n_9k"), (11000, "n_11k"), (12000, "n_12k"))


def _now() -> datetime:
    """Return a timezone-aware UTC timestamp for the TIMESTAMPTZ columns."""
    return datetime.now(timezone.utc)


# Corpus labels for OUR accounts — clans seen as opponents here get resolver
# priority and on_our_accounts=true. The full corpus has ~490K distinct
# opponent clans; harvest covers all of them for free (no API), the resolver
# enriches them lazily, ours-first.
_OUR_CORPUSES = ("personal", "alt")

# Resolver defaults: how many clans to API-fetch per run, re-resolution age,
# and a give-up cap so deleted/perma-404 clans aren't retried forever.
_RESOLVE_BATCH = 300
_RESOLVE_MAX_AGE_DAYS = 30
_RESOLVE_MAX_ATTEMPTS = 4


def harvest_clan_dim(session: Session) -> int:
    """Harvest clan IDENTITY from battle raw_json — NO API calls.

    Upserts one row per clan ever seen as an opponent's clan (~490K), filling
    clan_tag, clan_name, first/last_seen, n_battles_seen, on_our_accounts. A
    single set-based INSERT...SELECT...ON CONFLICT — cheap relative to ~490K API
    calls. Deliberately does NOT touch the measure columns or ``resolved_at`` so
    prior enrichment is preserved. PostgreSQL only (JSON path + ON CONFLICT);
    SQLite (tests) is a no-op.

    Args:
        session: SQLAlchemy session.

    Returns:
        Number of clan identity rows upserted (insert + update).
    """
    if session.bind is not None and session.bind.dialect.name == "sqlite":
        return 0
    result = session.execute(text(
        """
        INSERT INTO clan_dim (
            clan_tag, clan_name, first_seen, last_seen,
            n_battles_seen, on_our_accounts, refreshed_at
        )
        SELECT
            raw_json->'opponent'->0->'clan'->>'tag' AS clan_tag,
            (array_agg(raw_json->'opponent'->0->'clan'->>'name'
                       ORDER BY battle_time DESC NULLS LAST))[1] AS clan_name,
            min(battle_time) AS first_seen,
            max(battle_time) AS last_seen,
            count(*) AS n_battles_seen,
            bool_or(corpus = ANY(:our_corpuses)) AS on_our_accounts,
            now() AS refreshed_at
        FROM battles
        WHERE raw_json->'opponent'->0->'clan'->>'tag' IS NOT NULL
        GROUP BY raw_json->'opponent'->0->'clan'->>'tag'
        ON CONFLICT (clan_tag) DO UPDATE SET
            clan_name       = EXCLUDED.clan_name,
            first_seen      = LEAST(clan_dim.first_seen, EXCLUDED.first_seen),
            last_seen       = GREATEST(clan_dim.last_seen, EXCLUDED.last_seen),
            n_battles_seen  = EXCLUDED.n_battles_seen,
            on_our_accounts = clan_dim.on_our_accounts OR EXCLUDED.on_our_accounts,
            refreshed_at    = now()
        """
    ), {"our_corpuses": list(_OUR_CORPUSES)})
    session.commit()
    n = result.rowcount or 0
    logger.info("harvest_clan_dim: upserted %d clan identities (no API)", n)
    return n


def resolve_clan_dim(
    session: Session,
    api: ClashRoyaleAPI,
    batch: int = _RESOLVE_BATCH,
    max_age_days: int = _RESOLVE_MAX_AGE_DAYS,
    max_attempts: int = _RESOLVE_MAX_ATTEMPTS,
) -> int:
    """Enrich a priority batch of clan_dim rows with CR clan API measures.

    Picks up to ``batch`` clans that are unresolved (``resolved_at IS NULL``) or
    stale (older than ``max_age_days``), skipping those that have already failed
    ``max_attempts`` times. Ordered ours-first, then unresolved, then most
    recently/frequently seen — so the clans that matter (our accounts'
    opponents) resolve first and the ~490K long tail drains opportunistically
    over many runs. PostgreSQL only; SQLite (tests) is a no-op.

    Args:
        session: SQLAlchemy session.
        api: ClashRoyaleAPI client (needs a valid CR_API_KEY).
        batch: Max clans to API-fetch this run.
        max_age_days: Re-resolve clans whose measures are older than this.
        max_attempts: Stop retrying a clan after this many failed attempts.

    Returns:
        Number of clans successfully resolved this run.
    """
    if session.bind is not None and session.bind.dialect.name == "sqlite":
        return 0
    candidates = session.execute(text(
        """
        SELECT clan_tag FROM clan_dim
        WHERE resolve_attempts < :max_attempts
          AND (resolved_at IS NULL
               OR resolved_at < now() - make_interval(days => :max_age_days))
        ORDER BY on_our_accounts DESC,
                 (resolved_at IS NULL) DESC,
                 last_seen DESC NULLS LAST,
                 n_battles_seen DESC NULLS LAST
        LIMIT :batch
        """
    ), {"max_attempts": max_attempts, "max_age_days": max_age_days, "batch": batch}
    ).scalars().all()

    logger.info("resolve_clan_dim: %d clans selected for enrichment", len(candidates))
    resolved = 0
    for clan_tag in candidates:
        try:
            clan = api.get_clan(clan_tag)
        except APIError as e:
            # No silent failures: bump attempt count (so perma-failures retire)
            # and continue rather than aborting the batch.
            logger.warning("resolve_clan_dim: %s failed: %s", clan_tag, e)
            session.execute(text(
                "UPDATE clan_dim SET resolve_attempts = resolve_attempts + 1, "
                "refreshed_at = now() WHERE clan_tag = :t"
            ), {"t": clan_tag})
            continue

        members = clan.get("memberList") or []
        trophies = [m.get("trophies", 0) for m in members
                    if m.get("trophies") is not None]
        buckets = {col: sum(1 for t in trophies if t >= thr)
                   for thr, col in _TROPHY_BUCKETS}
        top = max(members, key=lambda m: m.get("trophies", 0), default=None)

        session.execute(text(
            """
            UPDATE clan_dim SET
                clan_name = COALESCE(:name, clan_name),
                member_count = :mc, max_trophies = :mx, avg_trophies = :av,
                median_trophies = :md, n_9k = :n9, n_11k = :n11, n_12k = :n12,
                top_member_name = :tn, top_member_tag = :tt, top_member_trophies = :tr,
                resolved_at = now(), resolve_attempts = resolve_attempts + 1,
                refreshed_at = now()
            WHERE clan_tag = :t
            """
        ), {
            "t": clan_tag, "name": clan.get("name"),
            "mc": clan.get("members", len(members)),
            "mx": max(trophies) if trophies else None,
            "av": round(sum(trophies) / len(trophies)) if trophies else None,
            "md": round(median(trophies)) if trophies else None,
            "n9": buckets["n_9k"], "n11": buckets["n_11k"], "n12": buckets["n_12k"],
            "tn": top.get("name") if top else None,
            "tt": top.get("tag") if top else None,
            "tr": top.get("trophies") if top else None,
        })
        resolved += 1

    session.commit()
    logger.info("resolve_clan_dim: enriched %d clans", resolved)
    return resolved


def _opponent_clan_tags(session: Session) -> dict[str, Optional[str]]:
    """Map opponent_tag -> latest clan tag from battle raw_json.

    Uses the most recent battle per opponent (DISTINCT ON battle_time DESC) so
    a player's clan reflects where they last were. PostgreSQL only; returns an
    empty mapping on SQLite.

    Args:
        session: SQLAlchemy session.

    Returns:
        Dict of opponent_tag to clan tag (clan tag may be None).
    """
    if session.bind is not None and session.bind.dialect.name == "sqlite":
        return {}
    rows = session.execute(text(
        "SELECT DISTINCT ON (opponent_tag) opponent_tag, "
        "       raw_json->'opponent'->0->'clan'->>'tag' AS clan_tag "
        "FROM battles "
        "WHERE opponent_tag IS NOT NULL "
        "ORDER BY opponent_tag, battle_time DESC NULLS LAST"
    )).fetchall()
    return {r.opponent_tag: r.clan_tag for r in rows}


def refresh_level_trophy_ref(session: Session) -> int:
    """Recompute the empirical level→trophy reference from the whole corpus.

    For each deck-top displayed level, the median (and p10) of
    ``opponent_starting_trophies`` across all PvP battles — i.e. where decks of
    that top-level normally sit on the ladder. Drives ``implied_trophy_gap``.
    PostgreSQL only (percentile_cont + displayed-level arithmetic); SQLite
    (tests) is a no-op.

    Returns:
        Number of reference rows written.
    """
    if session.bind is not None and session.bind.dialect.name == "sqlite":
        return 0
    session.execute(text("DELETE FROM level_trophy_ref"))
    session.execute(text(
        """
        INSERT INTO level_trophy_ref
            (deck_top_level, median_trophy, p10_trophy, n_samples, refreshed_at)
        SELECT lvl,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY tr)::int,
               percentile_cont(0.1) WITHIN GROUP (ORDER BY tr)::int,
               COUNT(*), now()
        FROM (
            SELECT b.battle_id, b.opponent_starting_trophies AS tr,
                   MAX(dc.card_level + (16 - dc.card_max_level)) AS lvl
            FROM battles b
            JOIN deck_cards dc ON dc.battle_id = b.battle_id AND dc.is_player_deck = 0
            WHERE b.battle_type = 'PvP'
              AND b.opponent_starting_trophies BETWEEN 1 AND 13000
            GROUP BY b.battle_id, b.opponent_starting_trophies
        ) per
        WHERE lvl IS NOT NULL
        GROUP BY lvl
        """
    ))
    session.commit()
    n = session.execute(text("SELECT COUNT(*) FROM level_trophy_ref")).scalar() or 0
    logger.info("refresh_level_trophy_ref: %d levels in reference", n)
    return n


def resolve_player_king(
    session: Session,
    api: ClashRoyaleAPI,
    batch: int = _RESOLVE_BATCH,
    max_age_days: int = _RESOLVE_MAX_AGE_DAYS,
    max_attempts: int = _RESOLVE_MAX_ATTEMPTS,
) -> int:
    """Resolve king (experience) level for a priority batch of player_dim players.

    King level isn't in the battle log, so fetch ``/players/{tag}`` and cache
    king_level + best_trophies in ``player_king``. Prioritized by
    implied_trophy_gap DESC (confirm the suspected funded-smurfs first), then
    recency; skips players already resolved (within max_age) or repeatedly
    failed. PostgreSQL only; SQLite (tests) is a no-op.

    Returns:
        Number of players successfully resolved this run.
    """
    if session.bind is not None and session.bind.dialect.name == "sqlite":
        return 0
    candidates = session.execute(text(
        """
        SELECT pd.player_tag
        FROM player_dim pd
        LEFT JOIN player_king pk ON pk.player_tag = pd.player_tag
        WHERE pd.player_tag IS NOT NULL
          AND COALESCE(pk.resolve_attempts, 0) < :max_attempts
          AND (pk.resolved_at IS NULL
               OR pk.resolved_at < now() - make_interval(days => :max_age_days))
        ORDER BY pd.implied_trophy_gap DESC NULLS LAST,
                 pd.last_seen DESC NULLS LAST
        LIMIT :batch
        """
    ), {"max_attempts": max_attempts, "max_age_days": max_age_days, "batch": batch}
    ).scalars().all()

    logger.info("resolve_player_king: %d players selected", len(candidates))
    resolved = 0
    for tag in candidates:
        try:
            p = api.get_player(tag)
        except APIError as e:
            logger.warning("resolve_player_king: %s failed: %s", tag, e)
            session.execute(text(
                "INSERT INTO player_king (player_tag, resolve_attempts, refreshed_at) "
                "VALUES (:t, 1, now()) ON CONFLICT (player_tag) DO UPDATE SET "
                "resolve_attempts = player_king.resolve_attempts + 1, refreshed_at = now()"
            ), {"t": tag})
            continue
        session.execute(text(
            """
            INSERT INTO player_king
                (player_tag, king_level, best_trophies, resolved_at, resolve_attempts, refreshed_at)
            VALUES (:t, :kl, :bt, now(), 1, now())
            ON CONFLICT (player_tag) DO UPDATE SET
                king_level = :kl, best_trophies = :bt, resolved_at = now(),
                resolve_attempts = player_king.resolve_attempts + 1, refreshed_at = now()
            """
        ), {"t": tag, "kl": p.get("expLevel"), "bt": p.get("bestTrophies")})
        resolved += 1

    session.commit()
    logger.info("resolve_player_king: resolved %d players", resolved)
    return resolved


def _opponent_deck_top_levels(
    session: Session, corpuses: Optional[tuple[str, ...]],
) -> dict[str, int]:
    """Map opponent_tag -> max displayed card level they've fielded.

    displayed = card_level + (16 - card_max_level). PostgreSQL only; SQLite
    returns empty.
    """
    if session.bind is not None and session.bind.dialect.name == "sqlite":
        return {}
    sql = (
        "SELECT b.opponent_tag, MAX(dc.card_level + (16 - dc.card_max_level)) AS lvl "
        "FROM battles b JOIN deck_cards dc "
        "  ON dc.battle_id = b.battle_id AND dc.is_player_deck = 0 "
        "WHERE b.opponent_tag IS NOT NULL"
    )
    params: dict = {}
    if corpuses is not None:
        sql += " AND b.corpus = ANY(:corpuses)"
        params["corpuses"] = list(corpuses)
    sql += " GROUP BY b.opponent_tag"
    return {r.opponent_tag: r.lvl for r in session.execute(text(sql), params) if r.lvl is not None}


def refresh_player_dim(
    session: Session,
    corpuses: Optional[tuple[str, ...]] = ("personal", "alt"),
    alt_min_games: int = 3,
) -> int:
    """Repopulate ``player_dim`` by aggregating the ``battles`` table.

    Each opponent (``opponent_tag``) becomes one player_dim row. Win/loss are
    from the *opponent's* perspective (inverted from ``Battle.result``). Enriched
    with the opponent's latest clan tag, their deck-top displayed level, and the
    **implied_trophy_gap** (funded-smurf signal): where their card levels imply
    they belong (``level_trophy_ref``) minus their actual trophies.

    Scoped to our own accounts (``('personal','alt')``) by default — the full
    corpus has ~1.2M distinct opponents (heavier). Pass ``None`` for all battles.

    Args:
        session: SQLAlchemy session.
        corpuses: Battle corpus labels to aggregate, or None for all.
        alt_min_games: Minimum games before the alt-suspect heuristic applies.

    Returns:
        Number of player_dim rows written.
    """
    # Opponent win/loss is inverted: opponent wins when the main player lost.
    opp_wins = func.sum(case((Battle.result == "loss", 1), else_=0))
    opp_losses = func.sum(case((Battle.result == "win", 1), else_=0))

    stmt = (
        select(
            Battle.opponent_tag.label("player_tag"),
            func.max(Battle.opponent_name).label("name"),
            func.count().label("games"),
            opp_wins.label("wins"),
            opp_losses.label("losses"),
            func.min(Battle.battle_time).label("first_seen"),
            func.max(Battle.battle_time).label("last_seen"),
            func.max(Battle.opponent_starting_trophies).label("latest_trophies"),
        )
        .where(Battle.opponent_tag.isnot(None))
        .group_by(Battle.opponent_tag)
    )
    if corpuses is not None:
        stmt = stmt.where(Battle.corpus.in_(corpuses))

    agg_rows = session.execute(stmt).all()
    logger.info("refresh_player_dim: aggregating %d distinct opponents (corpuses=%s)",
                len(agg_rows), corpuses)

    # latest_deck_hash: the opponent's deck in their most recent battle.
    deck_stmt = (
        select(Battle.opponent_tag, Battle.opponent_deck_hash)
        .where(Battle.opponent_tag.isnot(None))
        .order_by(Battle.opponent_tag, Battle.battle_time.desc())
    )
    if corpuses is not None:
        deck_stmt = deck_stmt.where(Battle.corpus.in_(corpuses))
    last_deck: dict[str, Optional[str]] = {}
    for tag, deck_hash in session.execute(deck_stmt):
        last_deck.setdefault(tag, deck_hash)

    clan_map = _opponent_clan_tags(session)
    deck_top = _opponent_deck_top_levels(session, corpuses)
    # level -> median trophy reference for the implied-trophy gap
    ref = {lvl: med for lvl, med in session.execute(
        select(LevelTrophyRef.deck_top_level, LevelTrophyRef.median_trophy)
    )}
    # king (experience) level from the persistent cache (survives rebuilds)
    king_map = {tag: kl for tag, kl in session.execute(
        select(PlayerKing.player_tag, PlayerKing.king_level)
    )}

    # Full rebuild — derived data.
    session.query(PlayerDim).delete()
    session.flush()

    now = _now()
    written = 0
    for r in agg_rows:
        # Alt-suspect heuristic: a name that looks like the main account's
        # alts is out of scope here; instead flag players whose displayed name
        # is empty/anonymous yet who recur — a weak signal the orchestrator can
        # refine. Kept deliberately conservative (off unless clearly anomalous).
        is_alt_suspect = bool(
            r.games >= alt_min_games and (r.name is None or r.name.strip() == "")
        )
        # Funded-smurf signal: where their card levels imply they belong minus
        # where they actually are. Large positive = cards belong far above
        # placement. NULL if we lack their deck level or a reference for it.
        dtl = deck_top.get(r.player_tag)
        gap = None
        if dtl is not None and r.latest_trophies is not None and ref.get(dtl) is not None:
            gap = ref[dtl] - r.latest_trophies
        session.add(PlayerDim(
            player_tag=r.player_tag,
            name=r.name,
            latest_trophies=r.latest_trophies,
            exp_level=king_map.get(r.player_tag),  # king level from player_king cache
            clan_tag=clan_map.get(r.player_tag),
            first_seen=r.first_seen,
            last_seen=r.last_seen,
            games=r.games or 0,
            wins=r.wins or 0,
            losses=r.losses or 0,
            last_deck_hash=last_deck.get(r.player_tag),
            is_alt_suspect=is_alt_suspect,
            deck_top_level=dtl,
            implied_trophy_gap=gap,
            refreshed_at=now,
        ))
        written += 1

    session.commit()
    logger.info("refresh_player_dim: wrote %d player rows", written)
    return written


def refresh_dims(
    session: Session, api: ClashRoyaleAPI, resolve_batch: int = _RESOLVE_BATCH,
) -> dict[str, int]:
    """Refresh the dimension tables. Wired to the ``--refresh-dims`` CLI flag.

    Three phases, cheapest first: (1) harvest clan IDENTITY for all clans from
    battle data (no API); (2) resolve a priority batch of clan MEASURES via the
    clan API; (3) rebuild player_dim from battles. Designed for a daily cron —
    harvest is always complete, the resolver chips the ~490K enrichment backlog
    ours-first over many runs.

    Args:
        session: SQLAlchemy session.
        api: ClashRoyaleAPI client.
        resolve_batch: Max clans to API-enrich this run.

    Returns:
        Dict with ``harvested``, ``resolved`` and ``players`` counts.
    """
    harvested = harvest_clan_dim(session)
    resolved = resolve_clan_dim(session, api, batch=resolve_batch)
    levels = refresh_level_trophy_ref(session)   # must precede player_dim (feeds the gap)
    players = refresh_player_dim(session)
    # King level resolves AFTER player_dim (candidate selection reads it,
    # prioritized by the gap), then we reflect the freshly-resolved kings back
    # into player_dim so this run isn't a cycle behind.
    kings = resolve_player_king(session, api, batch=resolve_batch)
    if session.bind is not None and session.bind.dialect.name != "sqlite":
        session.execute(text(
            "UPDATE player_dim SET exp_level = pk.king_level "
            "FROM player_king pk WHERE pk.player_tag = player_dim.player_tag"
        ))
        session.commit()
    return {"harvested": harvested, "resolved": resolved, "level_ref": levels,
            "players": players, "kings": kings}
