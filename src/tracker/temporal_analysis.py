"""Time-based strategic analysis of replay event sequences.

Provides temporal queries: opening analysis, phase profiles, push timing,
broken cycle detection, and full matchup deep dives against opponent archetypes.
"""

import json
import logging
from collections import Counter, defaultdict
from typing import Optional

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from tracker.archetypes import ARCHETYPES, classify_archetype
from tracker.ml.card_metadata import CARD_TYPES, kebab_to_title
from tracker.models import Battle, ReplayEvent

logger = logging.getLogger(__name__)

# Game phase tick boundaries (mirrored from ml/sequence_dataset.py)
PHASE_REGULAR_END = 3360
PHASE_DOUBLE_END = 5280
PHASE_OT_END = 7920

# Arena coordinates
ARENA_X_MID = 8750
ARENA_Y_MID = 15750
LANE_LEFT = 6750
LANE_RIGHT = 10750

# Opening window: ~30 seconds at ~18.7 ticks/sec
OPENING_TICK_WINDOW = 560

# Push detection
PUSH_MIN_PLAYS = 3
PUSH_TICK_WINDOW = 300

PHASE_NAMES = ["regular", "double", "overtime", "ot_double"]


def _classify_phase(tick: int) -> str:
    """Classify a game tick into its phase."""
    if tick < PHASE_REGULAR_END:
        return "regular"
    if tick < PHASE_DOUBLE_END:
        return "double"
    if tick < PHASE_OT_END:
        return "overtime"
    return "ot_double"


def _classify_lane(arena_x: int) -> str:
    """Classify arena x-coordinate into lane."""
    if arena_x < LANE_LEFT:
        return "left"
    if arena_x > LANE_RIGHT:
        return "right"
    return "center"


def _card_type(card_name_kebab: str) -> str:
    """Get card type (troop/spell/building) from kebab-case name."""
    title = kebab_to_title(card_name_kebab)
    return CARD_TYPES.get(title, "troop")


def _match_archetype(name: str) -> Optional[str]:
    """Case-insensitive fuzzy match against ARCHETYPES keys.

    Args:
        name: User input like "hog", "Hog Cycle", "hog cycle".

    Returns:
        Matched archetype key or None.
    """
    name_lower = name.lower().strip()
    # Exact match (case-insensitive)
    for key in ARCHETYPES:
        if key.lower() == name_lower:
            return key
    # Substring match
    for key in ARCHETYPES:
        if name_lower in key.lower():
            return key
    return None


def _load_filtered_battles(
    session: Session,
    archetype: Optional[str] = None,
    min_trophies: Optional[int] = None,
    result: Optional[str] = None,
) -> list[dict]:
    """Load PvP battles with replay data, optionally filtered.

    Uses full corpus (not just personal games) to maximize sample size.

    Args:
        session: SQLAlchemy session.
        archetype: Filter to games against this opponent archetype.
        min_trophies: Minimum opponent starting trophies.
        result: Filter by 'win' or 'loss'.

    Returns:
        List of battle dicts with keys: battle_id, result,
        player_elixir_leaked, opponent_elixir_leaked, battle_duration,
        opponent_starting_trophies, opponent_deck.
    """
    stmt = select(
        Battle.battle_id,
        Battle.result,
        Battle.player_elixir_leaked,
        Battle.opponent_elixir_leaked,
        Battle.battle_duration,
        Battle.opponent_starting_trophies,
        Battle.opponent_deck,
    ).where(
        Battle.battle_type == "PvP",
        Battle.result.in_(["win", "loss"]),
        Battle.opponent_deck.isnot(None),
    )

    if result:
        stmt = stmt.where(Battle.result == result)
    if min_trophies:
        stmt = stmt.where(Battle.opponent_starting_trophies >= min_trophies)

    # Only battles with replay data
    stmt = stmt.where(
        text("""EXISTS (
            SELECT 1 FROM replay_events re
            WHERE re.battle_id = battles.battle_id
              AND re.card_name != '_invalid'
        )""")
    )

    rows = session.execute(stmt).all()
    battles = []
    for row in rows:
        d = row._asdict()
        # Apply archetype filter Python-side
        if archetype:
            try:
                deck = json.loads(d["opponent_deck"])
                if classify_archetype(deck) != archetype:
                    continue
            except (json.JSONDecodeError, TypeError):
                continue
        battles.append(d)

    logger.info(
        "Loaded %d battles (archetype=%s, min_trophies=%s, result=%s)",
        len(battles), archetype, min_trophies, result,
    )
    return battles


def _load_events_for_battles(
    session: Session,
    battle_ids: list[str],
    tick_min: Optional[int] = None,
    tick_max: Optional[int] = None,
    side: Optional[str] = None,
) -> dict[str, list]:
    """Load replay events for given battles, grouped by battle_id.

    Args:
        session: SQLAlchemy session.
        battle_ids: Battle IDs to load events for.
        tick_min: Minimum game_tick (inclusive).
        tick_max: Maximum game_tick (inclusive).
        side: Filter to "team" or "opponent".

    Returns:
        Dict mapping battle_id → list of ReplayEvent objects.
    """
    events_by_battle: dict[str, list] = defaultdict(list)
    chunk_size = 500

    for i in range(0, len(battle_ids), chunk_size):
        chunk = battle_ids[i : i + chunk_size]
        stmt = (
            select(ReplayEvent)
            .where(
                ReplayEvent.battle_id.in_(chunk),
                ReplayEvent.card_name != "_invalid",
            )
            .order_by(ReplayEvent.battle_id, ReplayEvent.game_tick)
        )
        if tick_min is not None:
            stmt = stmt.where(ReplayEvent.game_tick >= tick_min)
        if tick_max is not None:
            stmt = stmt.where(ReplayEvent.game_tick <= tick_max)
        if side:
            stmt = stmt.where(ReplayEvent.side == side)

        for ev in session.execute(stmt).scalars():
            events_by_battle[ev.battle_id].append(ev)

    return dict(events_by_battle)


def _lane_preference(events: list) -> dict[str, float]:
    """Compute lane preference fractions from events."""
    if not events:
        return {"left": 0.0, "right": 0.0, "center": 0.0}
    counts = Counter(_classify_lane(ev.arena_x) for ev in events)
    total = sum(counts.values())
    return {lane: counts.get(lane, 0) / total for lane in ("left", "right", "center")}


def _card_type_mix(events: list) -> dict[str, float]:
    """Compute card type distribution from events."""
    if not events:
        return {"troop": 0.0, "spell": 0.0, "building": 0.0}
    counts = Counter(_card_type(ev.card_name) for ev in events)
    total = sum(counts.values())
    return {t: counts.get(t, 0) / total for t in ("troop", "spell", "building")}


# ---------------------------------------------------------------------------
# Analysis Functions
# ---------------------------------------------------------------------------


def opening_analysis(
    session: Session,
    archetype: Optional[str] = None,
    min_trophies: Optional[int] = None,
    _battles: Optional[list[dict]] = None,
) -> dict:
    """Analyze the first ~30 seconds of games, comparing wins vs losses.

    Args:
        session: SQLAlchemy session.
        archetype: Filter to games against this opponent archetype.
        min_trophies: Minimum opponent starting trophies.
        _battles: Pre-filtered battles (internal optimization).

    Returns:
        Dict with game_count, archetype_filter, win/loss sub-dicts containing
        first card frequencies, timing, lane preference, and aggression index.
    """
    battles = _battles or _load_filtered_battles(
        session, archetype=archetype, min_trophies=min_trophies,
    )

    wins = [b for b in battles if b["result"] == "win"]
    losses = [b for b in battles if b["result"] == "loss"]

    all_ids = [b["battle_id"] for b in battles]
    events = _load_events_for_battles(
        session, all_ids, tick_max=OPENING_TICK_WINDOW,
    )

    def _analyze_group(group: list[dict]) -> dict:
        if not group:
            return {
                "count": 0, "first_card_team": [], "first_card_opponent": [],
                "avg_first_play_tick": 0, "avg_plays": 0,
                "lane_preference": {"left": 0, "right": 0, "center": 0},
                "aggression_index": 0,
            }

        first_cards_team: list[str] = []
        first_cards_opp: list[str] = []
        first_play_ticks: list[int] = []
        play_counts: list[int] = []
        all_team_events: list = []

        for b in group:
            evts = events.get(b["battle_id"], [])
            team_evts = [e for e in evts if e.side == "team"]
            opp_evts = [e for e in evts if e.side == "opponent"]

            if team_evts:
                first_cards_team.append(team_evts[0].card_name)
                first_play_ticks.append(team_evts[0].game_tick)
            if opp_evts:
                first_cards_opp.append(opp_evts[0].card_name)

            play_counts.append(len(team_evts))
            all_team_events.extend(team_evts)

        def _top_cards(cards: list[str], n: int = 5) -> list[dict]:
            counts = Counter(cards)
            total = len(cards) or 1
            return [
                {"card": c, "count": cnt, "pct": round(cnt / total * 100, 1)}
                for c, cnt in counts.most_common(n)
            ]

        aggressive = sum(1 for e in all_team_events if e.arena_y > ARENA_Y_MID)
        total_team = len(all_team_events) or 1

        return {
            "count": len(group),
            "first_card_team": _top_cards(first_cards_team),
            "first_card_opponent": _top_cards(first_cards_opp),
            "avg_first_play_tick": round(sum(first_play_ticks) / max(len(first_play_ticks), 1)),
            "avg_plays": round(sum(play_counts) / max(len(play_counts), 1), 1),
            "lane_preference": _lane_preference(all_team_events),
            "aggression_index": round(aggressive / total_team, 3),
        }

    return {
        "game_count": len(battles),
        "archetype_filter": archetype,
        "trophy_filter": min_trophies,
        "win": _analyze_group(wins),
        "loss": _analyze_group(losses),
    }


def phase_profile(
    session: Session,
    archetype: Optional[str] = None,
    result: Optional[str] = None,
    min_trophies: Optional[int] = None,
    _battles: Optional[list[dict]] = None,
) -> dict:
    """Per-phase breakdown of play patterns.

    Args:
        session: SQLAlchemy session.
        archetype: Filter by opponent archetype.
        result: Filter by 'win' or 'loss'. If None, returns both.
        min_trophies: Minimum opponent trophies.
        _battles: Pre-filtered battles (internal optimization).

    Returns:
        Dict with game_count and phases dict keyed by phase name,
        each containing win/loss sub-dicts with play rates, card type mix,
        and lane preference.
    """
    battles = _battles or _load_filtered_battles(
        session, archetype=archetype, result=result,
        min_trophies=min_trophies,
    )

    all_ids = [b["battle_id"] for b in battles]
    events = _load_events_for_battles(session, all_ids)

    # Group events by battle → phase → result → side
    phase_data: dict[str, dict[str, dict[str, list]]] = {
        phase: {"win": {"team": [], "opponent": []}, "loss": {"team": [], "opponent": []}}
        for phase in PHASE_NAMES
    }
    # Track game counts per phase (games that reach this phase)
    games_per_phase: dict[str, dict[str, int]] = {
        phase: {"win": 0, "loss": 0} for phase in PHASE_NAMES
    }

    result_map = {b["battle_id"]: b["result"] for b in battles}

    for bid, evts in events.items():
        r = result_map.get(bid)
        if not r:
            continue
        phases_seen = set()
        for ev in evts:
            phase = _classify_phase(ev.game_tick)
            phase_data[phase][r][ev.side].append(ev)
            phases_seen.add(phase)
        for p in phases_seen:
            games_per_phase[p][r] += 1

    # Compute metrics per phase
    phase_ticks = {
        "regular": PHASE_REGULAR_END,
        "double": PHASE_DOUBLE_END - PHASE_REGULAR_END,
        "overtime": PHASE_OT_END - PHASE_DOUBLE_END,
        "ot_double": 2000,  # approximate
    }

    phases_result = {}
    for phase in PHASE_NAMES:
        ticks = phase_ticks[phase]
        phase_result = {}
        for r in ("win", "loss"):
            n_games = games_per_phase[phase][r] or 1
            team_evts = phase_data[phase][r]["team"]
            opp_evts = phase_data[phase][r]["opponent"]

            phase_result[r] = {
                "games": games_per_phase[phase][r],
                "plays_per_100_ticks": round(len(team_evts) / n_games / ticks * 100, 2) if ticks else 0,
                "opp_plays_per_100_ticks": round(len(opp_evts) / n_games / ticks * 100, 2) if ticks else 0,
                "card_type_mix": _card_type_mix(team_evts),
                "opp_card_type_mix": _card_type_mix(opp_evts),
                "lane_preference": _lane_preference(team_evts),
            }
        phases_result[phase] = phase_result

    return {
        "game_count": len(battles),
        "archetype_filter": archetype,
        "phases": phases_result,
    }


def broken_cycle(
    session: Session,
    card_pairs: list[tuple[str, str]],
    window_ticks: int = 200,
) -> list[dict]:
    """Detect games where synergy card pairs are played apart.

    Args:
        session: SQLAlchemy session.
        card_pairs: List of (card_a, card_b) tuples in kebab-case.
        window_ticks: Maximum tick separation for "intact" cycle.

    Returns:
        List of dicts per pair with intact/broken counts and win rates.
    """
    # Load all battles with replay data (no archetype filter)
    battles = _load_filtered_battles(session)
    result_map = {b["battle_id"]: b["result"] for b in battles}

    all_ids = [b["battle_id"] for b in battles]
    events = _load_events_for_battles(session, all_ids, side="team")

    results = []
    for card_a, card_b in card_pairs:
        intact_wins = 0
        intact_losses = 0
        broken_wins = 0
        broken_losses = 0
        total_games = 0

        for bid, evts in events.items():
            # Find all plays of card_a
            a_ticks = [e.game_tick for e in evts if e.card_name == card_a]
            if not a_ticks:
                continue

            total_games += 1
            b_ticks = [e.game_tick for e in evts if e.card_name == card_b]

            # Check if any A play has a B play within window
            is_intact = False
            for a_t in a_ticks:
                for b_t in b_ticks:
                    if abs(a_t - b_t) <= window_ticks:
                        is_intact = True
                        break
                if is_intact:
                    break

            won = result_map.get(bid) == "win"
            if is_intact:
                if won:
                    intact_wins += 1
                else:
                    intact_losses += 1
            else:
                if won:
                    broken_wins += 1
                else:
                    broken_losses += 1

        intact_total = intact_wins + intact_losses
        broken_total = broken_wins + broken_losses
        intact_wr = round(intact_wins / intact_total * 100, 1) if intact_total else 0.0
        broken_wr = round(broken_wins / broken_total * 100, 1) if broken_total else 0.0

        results.append({
            "pair": (card_a, card_b),
            "total_games": total_games,
            "intact_count": intact_total,
            "broken_count": broken_total,
            "intact_wins": intact_wins,
            "broken_wins": broken_wins,
            "intact_win_rate": intact_wr,
            "broken_win_rate": broken_wr,
            "delta_pp": round(intact_wr - broken_wr, 1),
            "window_ticks": window_ticks,
        })

    return results


def push_timing(
    session: Session,
    archetype: Optional[str] = None,
    result: Optional[str] = None,
    _battles: Optional[list[dict]] = None,
) -> dict:
    """Analyze push timing — when do pushes into opponent territory happen?

    A push is 3+ team plays in opponent half (arena_y > ARENA_Y_MID)
    within PUSH_TICK_WINDOW ticks.

    Args:
        session: SQLAlchemy session.
        archetype: Filter by opponent archetype.
        result: Filter by 'win' or 'loss'. If None, compares both.
        _battles: Pre-filtered battles (internal optimization).

    Returns:
        Dict with push statistics and phase distribution.
    """
    battles = _battles or _load_filtered_battles(
        session, archetype=archetype, result=result,
    )

    all_ids = [b["battle_id"] for b in battles]
    events = _load_events_for_battles(session, all_ids, side="team")
    result_map = {b["battle_id"]: b["result"] for b in battles}

    def _detect_pushes(evts: list) -> list[dict]:
        """Detect push clusters in a game's team events."""
        aggressive = [e for e in evts if e.arena_y > ARENA_Y_MID]
        if len(aggressive) < PUSH_MIN_PLAYS:
            return []

        pushes = []
        push_start = 0
        for i in range(1, len(aggressive)):
            if aggressive[i].game_tick - aggressive[i - 1].game_tick > PUSH_TICK_WINDOW:
                # Gap too large — evaluate accumulated push
                cluster = aggressive[push_start:i]
                if len(cluster) >= PUSH_MIN_PLAYS:
                    pushes.append({
                        "start_tick": cluster[0].game_tick,
                        "end_tick": cluster[-1].game_tick,
                        "size": len(cluster),
                        "phase": _classify_phase(cluster[0].game_tick),
                    })
                push_start = i

        # Check last cluster
        cluster = aggressive[push_start:]
        if len(cluster) >= PUSH_MIN_PLAYS:
            pushes.append({
                "start_tick": cluster[0].game_tick,
                "end_tick": cluster[-1].game_tick,
                "size": len(cluster),
                "phase": _classify_phase(cluster[0].game_tick),
            })
        return pushes

    def _analyze_group(group_ids: list[str]) -> dict:
        all_pushes = []
        first_push_ticks = []
        push_counts = []

        for bid in group_ids:
            evts = events.get(bid, [])
            pushes = _detect_pushes(evts)
            push_counts.append(len(pushes))
            all_pushes.extend(pushes)
            if pushes:
                first_push_ticks.append(pushes[0]["start_tick"])

        n = len(group_ids) or 1
        phase_counts = Counter(p["phase"] for p in all_pushes)
        total_pushes = len(all_pushes) or 1

        return {
            "game_count": len(group_ids),
            "games_with_pushes": len(first_push_ticks),
            "avg_first_push_tick": round(sum(first_push_ticks) / max(len(first_push_ticks), 1)) if first_push_ticks else None,
            "avg_push_count": round(sum(push_counts) / n, 2),
            "avg_push_size": round(sum(p["size"] for p in all_pushes) / total_pushes, 1) if all_pushes else 0,
            "phase_distribution": {
                phase: round(phase_counts.get(phase, 0) / total_pushes, 3)
                for phase in PHASE_NAMES
            },
        }

    if result:
        ids = [b["battle_id"] for b in battles]
        data = _analyze_group(ids)
        data["archetype_filter"] = archetype
        data["result_filter"] = result
        return data

    win_ids = [b["battle_id"] for b in battles if b["result"] == "win"]
    loss_ids = [b["battle_id"] for b in battles if b["result"] == "loss"]

    return {
        "game_count": len(battles),
        "archetype_filter": archetype,
        "win": _analyze_group(win_ids),
        "loss": _analyze_group(loss_ids),
    }


def matchup_deep_dive(
    session: Session,
    archetype: str,
    min_trophies: Optional[int] = None,
) -> dict:
    """Comprehensive temporal profile against one archetype.

    Args:
        session: SQLAlchemy session.
        archetype: Opponent archetype (e.g., "Hog Cycle").
        min_trophies: Minimum opponent trophies.

    Returns:
        Dict with overall stats, opening analysis, phase profile,
        push timing, and notable patterns.
    """
    matched = _match_archetype(archetype)
    if not matched:
        return {"error": f"Unknown archetype: {archetype}", "known": list(ARCHETYPES.keys())}

    # Load battles once, pass to sub-functions
    battles = _load_filtered_battles(
        session, archetype=matched, min_trophies=min_trophies,
    )

    if not battles:
        return {
            "archetype": matched,
            "game_count": 0,
            "error": "No games with replay data against this archetype",
        }

    wins = [b for b in battles if b["result"] == "win"]
    losses = [b for b in battles if b["result"] == "loss"]

    win_leaks = [b["player_elixir_leaked"] or 0 for b in wins]
    loss_leaks = [b["player_elixir_leaked"] or 0 for b in losses]
    durations = [b["battle_duration"] for b in battles if b["battle_duration"]]

    opening = opening_analysis(session, _battles=battles)
    phases = phase_profile(session, _battles=battles)
    pushes = push_timing(session, _battles=battles)

    # Generate notable patterns
    patterns = _generate_patterns(opening, phases, pushes)

    return {
        "archetype": matched,
        "game_count": len(battles),
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate": round(len(wins) / len(battles) * 100, 1),
        "avg_duration": round(sum(durations) / max(len(durations), 1)) if durations else None,
        "avg_leak_win": round(sum(win_leaks) / max(len(win_leaks), 1), 1) if win_leaks else 0,
        "avg_leak_loss": round(sum(loss_leaks) / max(len(loss_leaks), 1), 1) if loss_leaks else 0,
        "trophy_filter": min_trophies,
        "opening": opening,
        "phases": phases,
        "push_timing": pushes,
        "notable_patterns": patterns,
    }


def _generate_patterns(opening: dict, phases: dict, pushes: dict) -> list[str]:
    """Generate heuristic notable pattern strings from analysis data."""
    patterns = []

    # Opening timing difference
    w_tick = opening.get("win", {}).get("avg_first_play_tick", 0)
    l_tick = opening.get("loss", {}).get("avg_first_play_tick", 0)
    if w_tick and l_tick:
        diff = l_tick - w_tick
        if abs(diff) > 20:
            if diff > 0:
                patterns.append(f"You open {diff} ticks faster in wins (avg tick {w_tick} vs {l_tick})")
            else:
                patterns.append(f"You open {-diff} ticks SLOWER in wins (avg tick {w_tick} vs {l_tick})")

    # Aggression index difference
    w_agg = opening.get("win", {}).get("aggression_index", 0)
    l_agg = opening.get("loss", {}).get("aggression_index", 0)
    if w_agg > 0 or l_agg > 0:
        diff = w_agg - l_agg
        if abs(diff) > 0.05:
            if diff > 0:
                patterns.append(f"More aggressive openings in wins ({w_agg:.1%} vs {l_agg:.1%} plays in opp half)")
            else:
                patterns.append(f"More aggressive openings in losses ({l_agg:.1%} vs {w_agg:.1%} plays in opp half)")

    # Push timing
    w_push = pushes.get("win", {}).get("avg_first_push_tick")
    l_push = pushes.get("loss", {}).get("avg_first_push_tick")
    if w_push and l_push:
        diff = l_push - w_push
        if abs(diff) > 100:
            if diff > 0:
                patterns.append(f"First push {diff} ticks earlier in wins (tick {w_push} vs {l_push})")
            else:
                patterns.append(f"First push {-diff} ticks LATER in wins (tick {w_push} vs {l_push})")

    # Push count difference
    w_pc = pushes.get("win", {}).get("avg_push_count", 0)
    l_pc = pushes.get("loss", {}).get("avg_push_count", 0)
    if w_pc > 0 or l_pc > 0:
        diff = w_pc - l_pc
        if abs(diff) > 0.3:
            if diff > 0:
                patterns.append(f"More pushes in wins ({w_pc:.1f}/game vs {l_pc:.1f}/game)")
            else:
                patterns.append(f"Fewer pushes in wins ({w_pc:.1f}/game vs {l_pc:.1f}/game)")

    # Phase profile: spell usage change in double elixir
    phase_data = phases.get("phases", {})
    for phase_name in ("double", "overtime"):
        w_phase = phase_data.get(phase_name, {}).get("win", {})
        l_phase = phase_data.get(phase_name, {}).get("loss", {})
        w_spell = w_phase.get("opp_card_type_mix", {}).get("spell", 0)
        l_spell = l_phase.get("opp_card_type_mix", {}).get("spell", 0)
        if w_spell > 0 and l_spell > 0:
            diff = w_spell - l_spell
            if abs(diff) > 0.05:
                direction = "more" if diff > 0 else "fewer"
                patterns.append(
                    f"Opponent plays {direction} spells in {phase_name} "
                    f"when they beat you ({w_spell:.0%} vs {l_spell:.0%})"
                )

    # Lane preference shift between phases
    reg_win = phase_data.get("regular", {}).get("win", {}).get("lane_preference", {})
    dbl_win = phase_data.get("double", {}).get("win", {}).get("lane_preference", {})
    if reg_win and dbl_win:
        for lane in ("left", "right"):
            reg_pct = reg_win.get(lane, 0)
            dbl_pct = dbl_win.get(lane, 0)
            diff = dbl_pct - reg_pct
            if abs(diff) > 0.1:
                direction = "more" if diff > 0 else "less"
                patterns.append(
                    f"Winning games shift {direction} {lane} lane in double elixir "
                    f"({reg_pct:.0%} → {dbl_pct:.0%})"
                )

    return patterns
