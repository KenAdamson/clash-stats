"""Rich temporal profiling of TCN embedding clusters and manifold legs.

Analyzes the 3-leg manifold structure at two levels:
1. Macro-legs via k-means on 3D UMAP coordinates (the 3 visual legs)
2. HDBSCAN micro-clusters with full temporal replay feature profiles

Per-group metrics:
- Temporal: game phase distribution, mean game tick, game duration
- Spatial: arena Y distribution (defensive vs offensive), lane preference
- Card type: troop/spell/building ratios
- Tempo: plays per game, inter-play tick gaps, deployment density
- Action-reaction: team vs opponent play interleaving pattern
- Economy: mean elixir cost, elixir leaked
- Outcome: win rate, crown differential
"""

import logging
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from tracker.ml.card_metadata import CARD_TYPES, kebab_to_title
from tracker.ml.storage import GameEmbedding
from tracker.models import Battle, ReplayEvent
from tracker.temporal_analysis import (
    ARENA_X_MID, ARENA_Y_MID, PHASE_REGULAR_END, PHASE_DOUBLE_END,
    PHASE_OT_END, _classify_phase, _classify_lane, _card_type,
)

logger = logging.getLogger(__name__)

# K-means for macro-leg detection
N_LEGS = 3


def _load_embedding_map(session: Session) -> dict[str, dict]:
    """Load all TCN embeddings with 3D coordinates and cluster IDs.

    Returns:
        Dict mapping battle_id → {cluster_id, x, y, z}.
    """
    rows = session.execute(
        select(
            GameEmbedding.battle_id,
            GameEmbedding.cluster_id,
            GameEmbedding.embedding_vec_3d,
        ).where(GameEmbedding.model_version == "tcn-v1")
    ).all()

    result = {}
    for bid, cid, vec_3d in rows:
        if vec_3d is None or len(vec_3d) != 3:
            continue
        xyz = vec_3d
        result[bid] = {
            "cluster_id": cid if cid is not None else -1,
            "x": float(xyz[0]),
            "y": float(xyz[1]),
            "z": float(xyz[2]),
        }
    return result


def _load_battle_meta(
    session: Session, battle_ids: list[str],
) -> dict[str, dict]:
    """Load battle metadata for given IDs."""
    meta = {}
    chunk_size = 500
    for i in range(0, len(battle_ids), chunk_size):
        chunk = battle_ids[i : i + chunk_size]
        rows = session.execute(
            select(
                Battle.battle_id,
                Battle.result,
                Battle.corpus,
                Battle.battle_duration,
                Battle.player_elixir_leaked,
                Battle.opponent_elixir_leaked,
                Battle.player_crowns,
                Battle.opponent_crowns,
                Battle.opponent_starting_trophies,
            ).where(Battle.battle_id.in_(chunk))
        ).all()
        for r in rows:
            meta[r[0]] = {
                "result": r[1],
                "corpus": r[2],
                "duration": r[3],
                "player_leak": r[4] or 0,
                "opponent_leak": r[5] or 0,
                "player_crowns": r[6] or 0,
                "opponent_crowns": r[7] or 0,
                "opp_trophies": r[8],
            }
    return meta


def _load_replay_events(
    session: Session, battle_ids: list[str],
) -> dict[str, list]:
    """Load replay events grouped by battle_id."""
    events: dict[str, list] = defaultdict(list)
    chunk_size = 500
    for i in range(0, len(battle_ids), chunk_size):
        chunk = battle_ids[i : i + chunk_size]
        rows = session.execute(
            select(ReplayEvent)
            .where(
                ReplayEvent.battle_id.in_(chunk),
                ReplayEvent.card_name != "_invalid",
            )
            .order_by(ReplayEvent.battle_id, ReplayEvent.game_tick)
        ).scalars().all()
        for ev in rows:
            events[ev.battle_id].append(ev)
    return dict(events)


def _profile_group(
    battle_ids: list[str],
    battle_meta: dict[str, dict],
    replay_events: dict[str, list],
) -> dict:
    """Compute rich temporal profile for a group of games.

    Args:
        battle_ids: Games in this group.
        battle_meta: Pre-loaded battle metadata.
        replay_events: Pre-loaded replay events by battle_id.

    Returns:
        Dict with temporal, spatial, card type, tempo, and outcome profiles.
    """
    n = len(battle_ids)
    if n == 0:
        return {"game_count": 0}

    # --- Outcome metrics ---
    wins = 0
    durations = []
    player_leaks = []
    opponent_leaks = []
    crown_diffs = []

    for bid in battle_ids:
        m = battle_meta.get(bid, {})
        if m.get("result") == "win":
            wins += 1
        if m.get("duration"):
            durations.append(m["duration"])
        player_leaks.append(m.get("player_leak", 0))
        opponent_leaks.append(m.get("opponent_leak", 0))
        crown_diffs.append(m.get("player_crowns", 0) - m.get("opponent_crowns", 0))

    # --- Temporal / spatial / card features from replay events ---
    phase_counts = Counter()  # phase → count of events
    lane_counts = Counter()
    card_type_counts = Counter()
    side_sequence_patterns = []  # list of (side, side, ...) tuples per game
    all_ticks = []
    inter_play_gaps = []
    plays_per_game = []
    team_y_values = []
    opp_y_values = []
    elixir_costs = []
    top_cards_team = Counter()
    top_cards_opp = Counter()
    # Phase distribution per game (fraction of plays in each phase)
    game_phase_fracs = []
    # First play tick per game
    first_play_ticks = []
    # Last play tick per game (proxy for game length in replay terms)
    last_play_ticks = []

    for bid in battle_ids:
        evts = replay_events.get(bid, [])
        if not evts:
            continue

        plays_per_game.append(len(evts))
        first_play_ticks.append(evts[0].game_tick)
        last_play_ticks.append(evts[-1].game_tick)

        game_phases = Counter()
        sides = []

        prev_tick = None
        for ev in evts:
            phase = _classify_phase(ev.game_tick)
            phase_counts[phase] += 1
            game_phases[phase] += 1
            lane_counts[_classify_lane(ev.arena_x)] += 1
            card_type_counts[_card_type(ev.card_name)] += 1
            all_ticks.append(ev.game_tick)
            sides.append(ev.side)

            if ev.side == "team":
                team_y_values.append(ev.arena_y)
                top_cards_team[ev.card_name] += 1
            else:
                opp_y_values.append(ev.arena_y)
                top_cards_opp[ev.card_name] += 1

            title = kebab_to_title(ev.card_name)
            cost = CARD_TYPES.get(title)  # This is type, not cost — use separate lookup
            # We don't have elixir cost on replay events, skip

            if prev_tick is not None:
                gap = ev.game_tick - prev_tick
                if gap > 0:
                    inter_play_gaps.append(gap)
            prev_tick = ev.game_tick

        # Side alternation: count team→opp and opp→team transitions
        side_sequence_patterns.append(tuple(sides))

        # Phase fraction for this game
        total_evts = len(evts)
        game_phase_fracs.append({
            p: game_phases.get(p, 0) / total_evts
            for p in ("regular", "double", "overtime", "ot_double")
        })

    # --- Aggregate ---
    total_events = sum(phase_counts.values()) or 1

    # Phase distribution (fraction of all events)
    phase_dist = {
        p: round(phase_counts.get(p, 0) / total_events, 3)
        for p in ("regular", "double", "overtime", "ot_double")
    }

    # Average phase fraction per game
    avg_phase_frac = {}
    if game_phase_fracs:
        for p in ("regular", "double", "overtime", "ot_double"):
            avg_phase_frac[p] = round(
                sum(f.get(p, 0) for f in game_phase_fracs) / len(game_phase_fracs), 3
            )
    else:
        avg_phase_frac = {p: 0 for p in ("regular", "double", "overtime", "ot_double")}

    # Lane distribution
    total_lane = sum(lane_counts.values()) or 1
    lane_dist = {
        l: round(lane_counts.get(l, 0) / total_lane, 3)
        for l in ("left", "right", "center")
    }

    # Card type distribution
    total_ct = sum(card_type_counts.values()) or 1
    card_type_dist = {
        t: round(card_type_counts.get(t, 0) / total_ct, 3)
        for t in ("troop", "spell", "building")
    }

    # Spatial: mean arena Y for team/opponent
    mean_team_y = round(np.mean(team_y_values)) if team_y_values else None
    mean_opp_y = round(np.mean(opp_y_values)) if opp_y_values else None
    # Aggression: fraction of team plays in opponent half
    team_aggressive = sum(1 for y in team_y_values if y > ARENA_Y_MID)
    aggression = round(team_aggressive / max(len(team_y_values), 1), 3)

    # Tempo
    avg_plays = round(np.mean(plays_per_game), 1) if plays_per_game else 0
    median_gap = round(float(np.median(inter_play_gaps))) if inter_play_gaps else 0
    mean_gap = round(float(np.mean(inter_play_gaps)), 1) if inter_play_gaps else 0

    # Action-reaction: count transitions between sides
    transitions = 0
    same_side = 0
    for seq in side_sequence_patterns:
        for i in range(1, len(seq)):
            if seq[i] != seq[i - 1]:
                transitions += 1
            else:
                same_side += 1
    total_pairs = transitions + same_side
    alternation_rate = round(transitions / max(total_pairs, 1), 3)

    # Top cards
    top_team = [
        {"card": c, "count": cnt}
        for c, cnt in top_cards_team.most_common(8)
    ]
    top_opp = [
        {"card": c, "count": cnt}
        for c, cnt in top_cards_opp.most_common(8)
    ]

    return {
        "game_count": n,
        "win_rate": round(wins / n, 3),
        "avg_duration": round(np.mean(durations)) if durations else None,
        "avg_player_leak": round(np.mean(player_leaks), 1),
        "avg_opponent_leak": round(np.mean(opponent_leaks), 1),
        "avg_crown_diff": round(np.mean(crown_diffs), 2),
        "phase_distribution": phase_dist,
        "avg_phase_fraction": avg_phase_frac,
        "lane_distribution": lane_dist,
        "card_type_distribution": card_type_dist,
        "aggression_index": aggression,
        "mean_team_y": mean_team_y,
        "mean_opp_y": mean_opp_y,
        "avg_plays_per_game": avg_plays,
        "median_inter_play_gap": median_gap,
        "mean_inter_play_gap": mean_gap,
        "alternation_rate": alternation_rate,
        "avg_first_play_tick": round(np.mean(first_play_ticks)) if first_play_ticks else None,
        "avg_last_play_tick": round(np.mean(last_play_ticks)) if last_play_ticks else None,
        "top_cards_team": top_team,
        "top_cards_opp": top_opp,
    }


def identify_legs(
    session: Session, n_legs: int = N_LEGS,
) -> dict[int, list[str]]:
    """Use k-means on 3D UMAP coordinates to identify macro-legs.

    Args:
        session: SQLAlchemy session.
        n_legs: Number of legs to identify (default 3).

    Returns:
        Dict mapping leg_id (0, 1, 2) → list of battle_ids.
    """
    emb_map = _load_embedding_map(session)
    if not emb_map:
        return {}

    battle_ids = list(emb_map.keys())
    coords = np.array([[emb_map[b]["x"], emb_map[b]["y"], emb_map[b]["z"]] for b in battle_ids])

    kmeans = KMeans(n_clusters=n_legs, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)

    legs: dict[int, list[str]] = defaultdict(list)
    for i, bid in enumerate(battle_ids):
        legs[int(labels[i])].append(bid)

    # Sort legs by win rate (highest first) for consistent ordering
    # We need battle meta to compute win rates
    battle_meta = _load_battle_meta(session, battle_ids)
    leg_wr = {}
    for leg_id, bids in legs.items():
        wins = sum(1 for b in bids if battle_meta.get(b, {}).get("result") == "win")
        leg_wr[leg_id] = wins / max(len(bids), 1)

    # Remap: leg 0 = highest WR, leg 2 = lowest WR
    sorted_legs = sorted(leg_wr.keys(), key=lambda k: -leg_wr[k])
    remapped = {}
    for new_id, old_id in enumerate(sorted_legs):
        remapped[new_id] = legs[old_id]

    logger.info(
        "K-means legs: %s",
        {k: f"{len(v)} games ({leg_wr[sorted_legs[k]]:.1%} WR)" for k, v in remapped.items()},
    )

    return dict(remapped)


def profile_legs(session: Session) -> list[dict]:
    """Full temporal profile of the 3 manifold legs.

    Returns:
        List of 3 dicts (leg 0=wins, leg 1=mixed, leg 2=losses),
        each with rich temporal/spatial/card features.
    """
    legs = identify_legs(session)
    if not legs:
        return []

    all_ids = []
    for bids in legs.values():
        all_ids.extend(bids)

    # Batch-load everything once
    battle_meta = _load_battle_meta(session, all_ids)
    replay_events = _load_replay_events(session, all_ids)

    profiles = []
    leg_names = ["dominant", "contested", "overwhelmed"]
    for leg_id in sorted(legs.keys()):
        bids = legs[leg_id]
        profile = _profile_group(bids, battle_meta, replay_events)
        profile["leg_id"] = leg_id
        profile["leg_name"] = leg_names[leg_id] if leg_id < len(leg_names) else f"leg-{leg_id}"
        profiles.append(profile)

    return profiles


def profile_manifold(session: Session) -> dict:
    """Complete manifold analysis: legs + summary comparison.

    Returns:
        Dict with 'legs' list and 'comparison' highlighting differences.
    """
    legs = profile_legs(session)
    if not legs:
        return {"error": "No TCN embeddings found. Run --train-tcn first."}

    # Generate comparison insights
    comparisons = []

    if len(legs) >= 2:
        dom = legs[0]  # dominant (highest WR)
        ovr = legs[-1]  # overwhelmed (lowest WR)

        # Win rate spread
        wr_spread = dom["win_rate"] - ovr["win_rate"]
        comparisons.append(
            f"Win rate spread: {dom['leg_name']} {dom['win_rate']:.1%} vs "
            f"{ovr['leg_name']} {ovr['win_rate']:.1%} ({wr_spread:.1%} gap)"
        )

        # Tempo comparison
        if dom.get("avg_plays_per_game") and ovr.get("avg_plays_per_game"):
            comparisons.append(
                f"Tempo: {dom['leg_name']} {dom['avg_plays_per_game']:.0f} plays/game, "
                f"{ovr['leg_name']} {ovr['avg_plays_per_game']:.0f} plays/game"
            )

        # Aggression
        comparisons.append(
            f"Aggression: {dom['leg_name']} {dom['aggression_index']:.1%} offensive, "
            f"{ovr['leg_name']} {ovr['aggression_index']:.1%} offensive"
        )

        # Alternation (reactive vs committed)
        comparisons.append(
            f"Alternation: {dom['leg_name']} {dom['alternation_rate']:.1%}, "
            f"{ovr['leg_name']} {ovr['alternation_rate']:.1%} "
            f"({'more reactive' if dom['alternation_rate'] > ovr['alternation_rate'] else 'more committed'})"
        )

        # Phase distribution shift
        for phase in ("regular", "double", "overtime"):
            d_frac = dom.get("avg_phase_fraction", {}).get(phase, 0)
            o_frac = ovr.get("avg_phase_fraction", {}).get(phase, 0)
            diff = d_frac - o_frac
            if abs(diff) > 0.03:
                more = dom["leg_name"] if diff > 0 else ovr["leg_name"]
                comparisons.append(
                    f"{phase.title()}: {more} has {abs(diff):.0%} more plays in this phase"
                )

        # Leak comparison
        comparisons.append(
            f"Elixir leak: {dom['leg_name']} {dom['avg_player_leak']:.1f}e, "
            f"{ovr['leg_name']} {ovr['avg_player_leak']:.1f}e"
        )

        # Card type shift
        for ct in ("spell", "building"):
            d_pct = dom.get("card_type_distribution", {}).get(ct, 0)
            o_pct = ovr.get("card_type_distribution", {}).get(ct, 0)
            diff = d_pct - o_pct
            if abs(diff) > 0.02:
                more = dom["leg_name"] if diff > 0 else ovr["leg_name"]
                comparisons.append(
                    f"{ct.title()}s: {more} uses {abs(diff):.0%} more {ct}s"
                )

    return {
        "legs": legs,
        "comparisons": comparisons,
        "total_games": sum(l["game_count"] for l in legs),
    }
