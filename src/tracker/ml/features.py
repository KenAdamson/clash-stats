"""Feature extraction from replay data into ~50-dim per-game vectors.

Extracts card play counts, elixir economy, tempo, outcome, and matchup
context features from replay_events + replay_summaries + deck_cards + battles.
"""

import logging
from typing import Optional

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from tracker.models import Battle, DeckCard, ReplayEvent, ReplaySummary
from tracker.ml.card_metadata import CardVocabulary
from tracker.ml.storage import GameFeature, to_blob, from_blob

logger = logging.getLogger(__name__)

# Arena midpoints for lane/aggression calculations
ARENA_X_MID = 8750   # left lane < mid, right lane > mid
ARENA_Y_MID = 15750  # team half < mid, opponent half > mid

# Feature vector version — bump when the extraction logic changes
FEATURE_VERSION = "v2"


def extract_game_features(
    session: Session,
    battle_id: str,
    vocab: CardVocabulary,
) -> Optional[np.ndarray]:
    """Extract a ~50-dim feature vector for a single game.

    Returns None if the game lacks sufficient replay data.
    """
    battle = session.execute(
        select(Battle).where(Battle.battle_id == battle_id)
    ).scalar_one_or_none()
    if battle is None:
        return None

    events = session.execute(
        select(ReplayEvent)
        .where(ReplayEvent.battle_id == battle_id)
        .order_by(ReplayEvent.game_tick)
    ).scalars().all()

    summaries = session.execute(
        select(ReplaySummary).where(ReplaySummary.battle_id == battle_id)
    ).scalars().all()

    deck_cards = session.execute(
        select(DeckCard).where(DeckCard.battle_id == battle_id)
    ).scalars().all()

    return _extract_features_from_loaded(battle, events, summaries, deck_cards)


def _extract_features_from_loaded(
    battle: Battle,
    events: list[ReplayEvent],
    summaries: list[ReplaySummary],
    deck_cards: list[DeckCard],
) -> Optional[np.ndarray]:
    """Extract features from pre-loaded ORM objects (no DB queries)."""
    if len(events) < 4:
        return None

    summary_by_side = {s.side: s for s in summaries}
    player_cards = [dc for dc in deck_cards if dc.is_player_deck == 1]
    opponent_cards = [dc for dc in deck_cards if dc.is_player_deck == 0]

    features: list[float] = []

    # --- Card play counts (24 dim) ---
    player_card_names = sorted(dc.card_name for dc in player_cards)
    team_events = [e for e in events if e.side == "team"]
    opp_events = [e for e in events if e.side == "opponent"]

    for card_name in player_card_names[:8]:
        count = sum(1 for e in team_events if e.card_name == card_name and not e.ability_used)
        features.append(float(count))
    features.extend([0.0] * (8 - len(player_card_names[:8])))

    opponent_card_names = sorted(dc.card_name for dc in opponent_cards)
    for card_name in opponent_card_names[:8]:
        count = sum(1 for e in opp_events if e.card_name == card_name and not e.ability_used)
        features.append(float(count))
    features.extend([0.0] * (8 - len(opponent_card_names[:8])))

    player_abilities = [e for e in team_events if e.ability_used]
    ability_names = sorted(set(e.card_name for e in player_abilities))
    for i in range(4):
        if i < len(ability_names):
            features.append(float(sum(1 for e in player_abilities if e.card_name == ability_names[i])))
        else:
            features.append(0.0)

    opp_abilities = [e for e in opp_events if e.ability_used]
    opp_ability_names = sorted(set(e.card_name for e in opp_abilities))
    for i in range(4):
        if i < len(opp_ability_names):
            features.append(float(sum(1 for e in opp_abilities if e.card_name == opp_ability_names[i])))
        else:
            features.append(0.0)

    # --- Elixir economy (10 dim) ---
    team_summary = summary_by_side.get("team")
    opp_summary = summary_by_side.get("opponent")

    total_elixir_player = float(team_summary.total_elixir) if team_summary and team_summary.total_elixir else 0.0
    total_elixir_opp = float(opp_summary.total_elixir) if opp_summary and opp_summary.total_elixir else 0.0
    total_plays_player = float(team_summary.total_plays) if team_summary and team_summary.total_plays else 0.0
    total_plays_opp = float(opp_summary.total_plays) if opp_summary and opp_summary.total_plays else 0.0

    features.append(total_elixir_player)
    features.append(total_elixir_opp)
    features.append(total_elixir_player - total_elixir_opp)
    features.append(float(battle.player_elixir_leaked or 0.0))
    features.append(float(battle.opponent_elixir_leaked or 0.0))
    features.append(total_elixir_player / max(total_plays_player, 1))
    features.append(total_elixir_opp / max(total_plays_opp, 1))

    for attr in ("troop_elixir", "spell_elixir", "building_elixir"):
        val = getattr(team_summary, attr, None) if team_summary else None
        features.append(float(val or 0) / max(total_elixir_player, 1))

    # --- Tempo (8 dim) ---
    features.append(total_plays_player)
    features.append(total_plays_opp)

    max_tick = max(e.game_tick for e in events) if events else 1
    features.append(total_plays_player / max(max_tick, 1) * 1000)
    features.append(total_plays_opp / max(max_tick, 1) * 1000)

    first_play_tick = min((e.game_tick for e in team_events), default=0)
    features.append(float(first_play_tick) / max(max_tick, 1))

    right_plays = sum(1 for e in team_events if e.arena_x > ARENA_X_MID)
    features.append(right_plays / max(len(team_events), 1))

    team_ticks = sorted(e.game_tick for e in team_events)
    if len(team_ticks) > 1:
        spacings = [team_ticks[i+1] - team_ticks[i] for i in range(len(team_ticks) - 1)]
        features.append(float(np.mean(spacings)) / max(max_tick, 1))
    else:
        features.append(0.0)

    aggressive_plays = sum(1 for e in team_events if e.arena_y > ARENA_Y_MID)
    features.append(aggressive_plays / max(len(team_events), 1))

    # --- Outcome-adjacent (3 dim) ---
    features.append(float(battle.crown_differential or 0))
    features.append(float(battle.battle_duration or 180) / 300.0)
    features.append(float(battle.player_king_tower_hp or 0) / 10000.0)

    # --- Matchup context (5 dim) ---
    trophies = battle.player_starting_trophies or 5000
    features.append(float(trophies) / 10000.0)

    avg_elixir_player = sum(dc.card_elixir or 0 for dc in player_cards) / max(len(player_cards), 1)
    avg_elixir_opp = sum(dc.card_elixir or 0 for dc in opponent_cards) / max(len(opponent_cards), 1)
    features.append(avg_elixir_player)
    features.append(avg_elixir_opp)

    evo_count = sum(1 for dc in player_cards if dc.card_variant == "evo")
    hero_count = sum(1 for dc in player_cards if dc.card_variant == "hero")
    features.append(float(evo_count))
    features.append(float(hero_count))

    # --- Battle type indicator (1 dim) ---
    # 1.0 = pathOfLegend, 0.0 = PvP/other — lets UMAP separate populations
    features.append(1.0 if battle.battle_type == "pathOfLegend" else 0.0)

    return np.array(features, dtype=np.float32)


CHUNK_SIZE = 1000


def build_feature_matrix(
    session: Session,
    vocab: CardVocabulary,
    incremental: bool = True,
) -> tuple[list[str], np.ndarray]:
    """Build feature vectors for all games with replay data.

    Bulk-loads battles, events, summaries, and deck_cards in chunks
    to avoid per-game query overhead.

    Args:
        session: DB session.
        vocab: CardVocabulary for card encoding.
        incremental: If True, skip battles already in game_features.

    Returns:
        Tuple of (battle_ids, feature_matrix) where feature_matrix
        is shape (n_games, n_features).
    """
    # Find battles with replay events
    replay_battles = session.execute(
        text("""
            SELECT DISTINCT re.battle_id
            FROM replay_events re
            JOIN battles b ON b.battle_id = re.battle_id
            WHERE b.battle_type IN ('PvP', 'pathOfLegend')
            ORDER BY b.battle_time
        """)
    ).scalars().all()

    # Filter out already-processed battles if incremental
    if incremental:
        existing = set(
            session.execute(
                select(GameFeature.battle_id)
                .where(GameFeature.feature_version == FEATURE_VERSION)
            ).scalars().all()
        )
        replay_battles = [bid for bid in replay_battles if bid not in existing]

    if not replay_battles:
        logger.info("No new battles to process")
        return [], np.array([])

    logger.info("Extracting features for %d games (chunk size %d)", len(replay_battles), CHUNK_SIZE)

    battle_ids: list[str] = []
    vectors: list[np.ndarray] = []
    skipped = 0
    processed = 0

    for chunk_start in range(0, len(replay_battles), CHUNK_SIZE):
        chunk = replay_battles[chunk_start:chunk_start + CHUNK_SIZE]

        # Bulk-load all data for this chunk
        battles_by_id: dict[str, Battle] = {}
        for b in session.execute(
            select(Battle).where(Battle.battle_id.in_(chunk))
        ).scalars():
            battles_by_id[b.battle_id] = b

        events_by_id: dict[str, list[ReplayEvent]] = {bid: [] for bid in chunk}
        for e in session.execute(
            select(ReplayEvent)
            .where(ReplayEvent.battle_id.in_(chunk))
            .order_by(ReplayEvent.game_tick)
        ).scalars():
            events_by_id[e.battle_id].append(e)

        summaries_by_id: dict[str, list[ReplaySummary]] = {bid: [] for bid in chunk}
        for s in session.execute(
            select(ReplaySummary).where(ReplaySummary.battle_id.in_(chunk))
        ).scalars():
            summaries_by_id[s.battle_id].append(s)

        deck_cards_by_id: dict[str, list[DeckCard]] = {bid: [] for bid in chunk}
        for dc in session.execute(
            select(DeckCard).where(DeckCard.battle_id.in_(chunk))
        ).scalars():
            deck_cards_by_id[dc.battle_id].append(dc)

        # Extract features from loaded data
        for battle_id in chunk:
            battle = battles_by_id.get(battle_id)
            if battle is None:
                skipped += 1
                continue

            vec = _extract_features_from_loaded(
                battle,
                events_by_id[battle_id],
                summaries_by_id[battle_id],
                deck_cards_by_id[battle_id],
            )
            if vec is not None:
                battle_ids.append(battle_id)
                vectors.append(vec)
                session.merge(GameFeature(
                    battle_id=battle_id,
                    feature_vector=to_blob(vec),
                    feature_version=FEATURE_VERSION,
                ))
            else:
                skipped += 1

        session.flush()
        # Evict loaded objects to keep memory bounded
        session.expire_all()
        processed += len(chunk)
        logger.info("  processed %d / %d games (%d extracted, %d skipped)",
                     processed, len(replay_battles), len(vectors), skipped)

    session.commit()

    n_features = vectors[0].shape[0] if vectors else 0
    logger.info(
        "Feature extraction complete: %d games, %d features/game, %d skipped",
        len(vectors), n_features, skipped,
    )

    if not vectors:
        return [], np.array([])

    return battle_ids, np.stack(vectors)


def load_feature_matrix(session: Session) -> tuple[list[str], np.ndarray]:
    """Load all stored feature vectors from the database.

    Returns:
        Tuple of (battle_ids, feature_matrix).
    """
    rows = session.execute(
        select(GameFeature.battle_id, GameFeature.feature_vector)
        .where(GameFeature.feature_version == FEATURE_VERSION)
        .order_by(GameFeature.battle_id)
    ).all()

    if not rows:
        return [], np.array([])

    battle_ids = [r[0] for r in rows]
    # Infer dimension from first vector
    dim = len(from_blob(rows[0][1], -1))
    vectors = [from_blob(r[1], dim) for r in rows]

    return battle_ids, np.stack(vectors)
