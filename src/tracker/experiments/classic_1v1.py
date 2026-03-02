#!/usr/bin/env python3
"""Classic 1v1 (Showdown_Friendly) investigation.

Extracts Showdown_Friendly games from the main DB, enriches opponent data
by cross-referencing all battle records and the CR API, and writes results
to a standalone experiment database.

Usage:
    # From repo root (package installed):
    python -m tracker.experiments.classic_1v1 --main-db data/clash_royale_history.db

    # Quick report only (no API calls):
    python -m tracker.experiments.classic_1v1 --main-db data/clash_royale_history.db --offline

    # Fetch fresh opponent profiles from CR API:
    python -m tracker.experiments.classic_1v1 --main-db data/clash_royale_history.db --fetch-profiles
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("classic_1v1")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OpponentProfile:
    """Everything we know about a Showdown_Friendly opponent."""
    tag: str
    name: str
    clan: str
    showdown_record: str  # "W 2-1", "L 0-3", etc.
    showdown_games: int
    # Cross-referenced from main DB
    max_trophies_as_opponent: Optional[int] = None
    max_trophies_as_player: Optional[int] = None  # If they're in corpus
    is_corpus_player: bool = False
    corpus_source: Optional[str] = None
    corpus_games_scraped: int = 0
    total_appearances_in_db: int = 0
    ladder_game_modes: list[str] = field(default_factory=list)
    # From CR API (if fetched)
    api_trophies: Optional[int] = None
    api_best_trophies: Optional[int] = None
    api_wins: Optional[int] = None
    api_losses: Optional[int] = None
    api_battle_count: Optional[int] = None
    api_three_crown_wins: Optional[int] = None
    api_challenge_max_wins: Optional[int] = None
    api_clan_name: Optional[str] = None


@dataclass
class ShowdownBattle:
    """A single Showdown_Friendly battle."""
    battle_id: str
    battle_time: str
    result: str
    player_crowns: int
    opponent_crowns: int
    opponent_tag: str
    opponent_name: str
    opponent_clan: str
    opponent_deck: list[str]
    opponent_archetype: str
    event_tag: Optional[str] = None
    raw_json: Optional[str] = None


# ---------------------------------------------------------------------------
# Archetype classification (inline, no import dependency)
# ---------------------------------------------------------------------------

ARCHETYPES = {
    "Golem Beatdown": ["Golem"],
    "Lava Hound": ["Lava Hound"],
    "Giant Beatdown": ["Giant"],
    "Royal Giant": ["Royal Giant"],
    "Hog Cycle": ["Hog Rider"],
    "X-Bow Siege": ["X-Bow"],
    "Mortar Siege": ["Mortar"],
    "Bridge Spam": ["Ram Rider", "Battle Ram"],
    "Graveyard Control": ["Graveyard"],
    "Miner Control": ["Miner"],
    "Three Musketeers": ["Three Musketeers"],
    "Sparky": ["Sparky"],
    "Balloon": ["Balloon"],
    "Elite Barbarians": ["Elite Barbarians"],
    "P.E.K.K.A Control": ["P.E.K.K.A"],
    "Mega Knight": ["Mega Knight"],
    "Goblin Barrel Bait": ["Goblin Barrel"],
    "Skeleton King": ["Skeleton King"],
    "Monk": ["Monk"],
    "Archer Queen": ["Archer Queen"],
    "Goblin Giant": ["Goblin Giant"],
    "Electro Giant": ["Electro Giant"],
    "Egiant": ["Elixir Golem"],
    "Golden Knight": ["Golden Knight"],
    "Lumberjack Rush": ["Lumberjack"],
    "Skeleton Barrel Bait": ["Skeleton Barrel"],
    "Wall Breakers": ["Wall Breakers"],
}


def classify_deck(card_names: list[str]) -> str:
    """Classify a deck by win condition."""
    cards = set(card_names)
    for archetype, wcs in ARCHETYPES.items():
        if any(wc in cards for wc in wcs):
            return archetype
    return "Unknown"


# ---------------------------------------------------------------------------
# Main DB extraction
# ---------------------------------------------------------------------------

def extract_showdown_battles(main_db: str) -> list[ShowdownBattle]:
    """Pull all Showdown_Friendly games from the main database."""
    conn = sqlite3.connect(main_db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Get battles
    cur.execute("""
        SELECT b.battle_id, b.battle_time, b.result, b.player_crowns,
               b.opponent_crowns, b.opponent_tag, b.opponent_name,
               b.raw_json
        FROM battles b
        WHERE b.game_mode_name = 'Showdown_Friendly'
        ORDER BY b.battle_time DESC
    """)
    rows = cur.fetchall()

    battles = []
    for row in rows:
        bid = row["battle_id"]

        # Get opponent deck
        cur.execute("""
            SELECT card_name FROM deck_cards
            WHERE battle_id = ? AND is_player_deck = 0
            ORDER BY card_name
        """, (bid,))
        deck = [r["card_name"] for r in cur.fetchall()]

        # Parse clan from raw_json
        clan = ""
        if row["raw_json"]:
            try:
                rj = json.loads(row["raw_json"])
                opp = rj.get("opponent", [{}])
                if isinstance(opp, list) and opp:
                    clan = opp[0].get("clan", {}).get("name", "")
            except (json.JSONDecodeError, KeyError, IndexError):
                pass

        # Parse event tag from raw_json
        event_tag = None
        if row["raw_json"]:
            try:
                rj = json.loads(row["raw_json"])
                event_tag = rj.get("challengeId") or rj.get("eventId")
            except (json.JSONDecodeError, KeyError):
                pass

        battles.append(ShowdownBattle(
            battle_id=bid,
            battle_time=row["battle_time"],
            result=row["result"],
            player_crowns=row["player_crowns"],
            opponent_crowns=row["opponent_crowns"],
            opponent_tag=row["opponent_tag"],
            opponent_name=row["opponent_name"],
            opponent_clan=clan,
            opponent_deck=deck,
            opponent_archetype=classify_deck(deck),
            event_tag=str(event_tag) if event_tag else None,
            raw_json=row["raw_json"],
        ))

    conn.close()
    return battles


def cross_reference_opponents(
    main_db: str, opponent_tags: set[str]
) -> dict[str, OpponentProfile]:
    """Look up opponent tags across the entire main DB."""
    conn = sqlite3.connect(main_db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    profiles: dict[str, OpponentProfile] = {}
    for tag in opponent_tags:
        profiles[tag] = OpponentProfile(
            tag=tag, name="", clan="", showdown_record="", showdown_games=0
        )

    tag_list = list(opponent_tags)
    placeholders = ",".join("?" * len(tag_list))

    # 1. Max trophies seen as opponent across ALL battles
    cur.execute(f"""
        SELECT opponent_tag,
               MAX(opponent_starting_trophies) as max_trophies,
               COUNT(*) as appearances
        FROM battles
        WHERE opponent_tag IN ({placeholders})
          AND opponent_starting_trophies IS NOT NULL
          AND opponent_starting_trophies > 0
        GROUP BY opponent_tag
    """, tag_list)
    for row in cur.fetchall():
        p = profiles[row["opponent_tag"]]
        p.max_trophies_as_opponent = row["max_trophies"]
        p.total_appearances_in_db = row["appearances"]

    # 2. Also count appearances where trophy data is null (like showdown)
    cur.execute(f"""
        SELECT opponent_tag, COUNT(*) as total
        FROM battles
        WHERE opponent_tag IN ({placeholders})
        GROUP BY opponent_tag
    """, tag_list)
    for row in cur.fetchall():
        profiles[row["opponent_tag"]].total_appearances_in_db = row["total"]

    # 3. Check if any are corpus players (player_tag in battles)
    cur.execute(f"""
        SELECT player_tag, MAX(trophies) as max_trophies
        FROM (
            SELECT player_tag,
                   MAX(player_starting_trophies) as trophies
            FROM battles
            WHERE player_tag IN ({placeholders})
              AND player_starting_trophies IS NOT NULL
              AND player_starting_trophies > 0
            GROUP BY player_tag
        )
        GROUP BY player_tag
    """, tag_list)
    for row in cur.fetchall():
        if row["player_tag"] in profiles:
            profiles[row["player_tag"]].max_trophies_as_player = row["max_trophies"]

    # 4. Check player_corpus table
    try:
        cur.execute(f"""
            SELECT player_tag, player_name, source, trophy_range_high,
                   games_scraped, active
            FROM player_corpus
            WHERE player_tag IN ({placeholders})
        """, tag_list)
        for row in cur.fetchall():
            if row["player_tag"] in profiles:
                p = profiles[row["player_tag"]]
                p.is_corpus_player = True
                p.corpus_source = row["source"]
                p.corpus_games_scraped = row["games_scraped"]
                if row["trophy_range_high"]:
                    p.max_trophies_as_player = max(
                        p.max_trophies_as_player or 0,
                        row["trophy_range_high"]
                    )
    except sqlite3.OperationalError:
        pass  # table might not exist

    # 5. What game modes do they appear in? (as opponent)
    cur.execute(f"""
        SELECT opponent_tag, game_mode_name,
               COUNT(*) as cnt,
               MAX(opponent_starting_trophies) as max_tr
        FROM battles
        WHERE opponent_tag IN ({placeholders})
          AND game_mode_name != 'Showdown_Friendly'
        GROUP BY opponent_tag, game_mode_name
        ORDER BY opponent_tag, cnt DESC
    """, tag_list)
    for row in cur.fetchall():
        if row["opponent_tag"] in profiles:
            mode = row["game_mode_name"] or "unknown"
            tr = row["max_tr"]
            profiles[row["opponent_tag"]].ladder_game_modes.append(
                f"{mode}(×{row['cnt']}, max={tr or '?'})"
            )

    # 6. Check if they appear as opponents of corpus players (gives trophy data)
    cur.execute(f"""
        SELECT b.opponent_tag, b.opponent_name,
               MAX(b.opponent_starting_trophies) as max_opp_tr,
               b.player_tag, b.game_mode_name
        FROM battles b
        WHERE b.opponent_tag IN ({placeholders})
          AND b.game_mode_name IN ('Ranked1v1_NewArena', 'Ladder', 'CW_Battle_1v1')
          AND b.opponent_starting_trophies > 0
        GROUP BY b.opponent_tag
    """, tag_list)
    for row in cur.fetchall():
        tag = row["opponent_tag"]
        if tag in profiles:
            tr = row["max_opp_tr"]
            if tr and (not profiles[tag].max_trophies_as_opponent or tr > profiles[tag].max_trophies_as_opponent):
                profiles[tag].max_trophies_as_opponent = tr

    conn.close()
    return profiles


# ---------------------------------------------------------------------------
# CR API enrichment (optional)
# ---------------------------------------------------------------------------

def fetch_opponent_profiles(profiles: dict[str, OpponentProfile]) -> None:
    """Fetch live player profiles from the CR API.

    Requires CR_API_KEY and CR_API_URL env vars.
    """
    api_key = os.environ.get("CR_API_KEY")
    api_url = os.environ.get("CR_API_URL", "https://proxy.royaleapi.dev/v1")
    if not api_key:
        logger.warning("CR_API_KEY not set — skipping API enrichment")
        return

    # Import here to avoid dependency when running offline
    from tracker.api import ClashRoyaleAPI, NotFoundError, AuthError

    api = ClashRoyaleAPI(api_key, api_url)

    for tag, prof in profiles.items():
        try:
            data = api.get_player(tag)
            prof.api_trophies = data.get("trophies")
            prof.api_best_trophies = data.get("bestTrophies")
            prof.api_wins = data.get("wins")
            prof.api_losses = data.get("losses")
            prof.api_battle_count = data.get("battleCount")
            prof.api_three_crown_wins = data.get("threeCrownWins")
            prof.api_challenge_max_wins = data.get("challengeMaxWins")
            clan = data.get("clan", {})
            prof.api_clan_name = clan.get("name") if clan else None
            logger.info(f"  Fetched {prof.name} ({tag}): {prof.api_trophies} trophies, best {prof.api_best_trophies}")
        except NotFoundError:
            logger.warning(f"  {tag}: player not found")
        except AuthError as e:
            logger.error(f"  API auth error — stopping: {e}")
            return
        except Exception as e:
            logger.warning(f"  {tag}: API error: {e}")


# ---------------------------------------------------------------------------
# Experiment database
# ---------------------------------------------------------------------------

EXPERIMENT_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS showdown_battles (
    battle_id TEXT PRIMARY KEY,
    battle_time TEXT,
    result TEXT,
    player_crowns INTEGER,
    opponent_crowns INTEGER,
    opponent_tag TEXT,
    opponent_name TEXT,
    opponent_clan TEXT,
    opponent_deck TEXT,
    opponent_archetype TEXT,
    event_tag TEXT,
    raw_json TEXT
);

CREATE TABLE IF NOT EXISTS opponent_profiles (
    tag TEXT PRIMARY KEY,
    name TEXT,
    clan TEXT,
    showdown_record TEXT,
    showdown_games INTEGER,
    max_trophies_as_opponent INTEGER,
    max_trophies_as_player INTEGER,
    is_corpus_player INTEGER DEFAULT 0,
    corpus_source TEXT,
    corpus_games_scraped INTEGER DEFAULT 0,
    total_appearances_in_db INTEGER DEFAULT 0,
    ladder_game_modes TEXT,
    api_trophies INTEGER,
    api_best_trophies INTEGER,
    api_wins INTEGER,
    api_losses INTEGER,
    api_battle_count INTEGER,
    api_three_crown_wins INTEGER,
    api_challenge_max_wins INTEGER,
    api_clan_name TEXT,
    last_updated TEXT
);

CREATE TABLE IF NOT EXISTS analysis_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT (datetime('now')),
    category TEXT,
    note TEXT
);
"""


def init_experiment_db(db_path: str) -> sqlite3.Connection:
    """Create/open the experiment database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(EXPERIMENT_DB_SCHEMA)
    conn.commit()
    return conn


def write_results(
    conn: sqlite3.Connection,
    battles: list[ShowdownBattle],
    profiles: dict[str, OpponentProfile],
) -> None:
    """Persist results to the experiment database."""
    cur = conn.cursor()

    for b in battles:
        cur.execute("""
            INSERT OR REPLACE INTO showdown_battles
            (battle_id, battle_time, result, player_crowns, opponent_crowns,
             opponent_tag, opponent_name, opponent_clan, opponent_deck,
             opponent_archetype, event_tag, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            b.battle_id, b.battle_time, b.result, b.player_crowns,
            b.opponent_crowns, b.opponent_tag, b.opponent_name,
            b.opponent_clan, json.dumps(b.opponent_deck),
            b.opponent_archetype, b.event_tag, b.raw_json,
        ))

    for tag, p in profiles.items():
        # Use INSERT OR IGNORE + UPDATE to avoid overwriting cached API data with None
        cur.execute("INSERT OR IGNORE INTO opponent_profiles (tag) VALUES (?)", (tag,))
        cur.execute("""
            UPDATE opponent_profiles SET
                name=?, clan=?, showdown_record=?, showdown_games=?,
                max_trophies_as_opponent=?, max_trophies_as_player=?,
                is_corpus_player=?, corpus_source=?, corpus_games_scraped=?,
                total_appearances_in_db=?, ladder_game_modes=?,
                last_updated=?
            WHERE tag=?
        """, (
            p.name, p.clan, p.showdown_record, p.showdown_games,
            p.max_trophies_as_opponent, p.max_trophies_as_player,
            int(p.is_corpus_player), p.corpus_source, p.corpus_games_scraped,
            p.total_appearances_in_db, json.dumps(p.ladder_game_modes),
            datetime.now().isoformat(), tag,
        ))
        # Only update API fields if we have fresh data (don't clobber cached values)
        if p.api_trophies is not None:
            cur.execute("""
                UPDATE opponent_profiles SET
                    api_trophies=?, api_best_trophies=?, api_wins=?, api_losses=?,
                    api_battle_count=?, api_three_crown_wins=?, api_challenge_max_wins=?,
                    api_clan_name=?
                WHERE tag=?
            """, (
                p.api_trophies, p.api_best_trophies, p.api_wins, p.api_losses,
                p.api_battle_count, p.api_three_crown_wins, p.api_challenge_max_wins,
                p.api_clan_name, tag,
            ))

    conn.commit()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(
    battles: list[ShowdownBattle],
    profiles: dict[str, OpponentProfile],
) -> None:
    """Print the investigation report to stdout."""
    wins = sum(1 for b in battles if b.result == "win")
    losses = len(battles) - wins

    print("=" * 72)
    print("  CLASSIC 1v1 (SHOWDOWN_FRIENDLY) INVESTIGATION")
    print("=" * 72)
    print(f"\n  Record: {wins}W-{losses}L ({len(battles)} games)")
    print(f"  Win rate: {wins/len(battles)*100:.0f}%")
    print()

    # Battles table
    print("  BATTLES")
    print("  " + "-" * 68)
    print(f"  {'Result':<8} {'Score':<7} {'Opponent':<20} {'Archetype':<25}")
    print("  " + "-" * 68)
    for b in battles:
        tag = "WIN" if b.result == "win" else "LOSS"
        score = f"{b.player_crowns}-{b.opponent_crowns}"
        print(f"  {tag:<8} {score:<7} {b.opponent_name:<20} {b.opponent_archetype:<25}")
    print()

    # Opponent deep-dive
    print("  OPPONENT PROFILES")
    print("  " + "-" * 68)

    # Separate stompers (losses) from victims (wins)
    loss_tags = {b.opponent_tag for b in battles if b.result != "win"}
    stompers = {t: p for t, p in profiles.items() if t in loss_tags}
    others = {t: p for t, p in profiles.items() if t not in loss_tags}

    if stompers:
        print("\n  Players who STOMPED you:")
        print()
        for tag, p in sorted(stompers.items(), key=lambda x: x[1].name):
            best_known = _best_trophy_estimate(p)
            elite_marker = " *** ELITE ***" if _is_elite(p) else ""
            print(f"    {p.name:<20} [{p.clan}]")
            print(f"      Tag: {tag}")
            print(f"      Showdown: {p.showdown_record}")
            print(f"      Best trophy estimate: {best_known}{elite_marker}")
            if p.api_best_trophies:
                print(f"      API best trophies: {p.api_best_trophies}")
                print(f"      API current trophies: {p.api_trophies}")
                print(f"      API record: {p.api_wins}W-{p.api_losses}L ({p.api_battle_count} total)")
                if p.api_three_crown_wins and p.api_wins:
                    pct = p.api_three_crown_wins / p.api_wins * 100
                    print(f"      3-crown rate: {pct:.0f}%")
                if p.api_challenge_max_wins:
                    print(f"      Challenge max wins: {p.api_challenge_max_wins}")
            if p.is_corpus_player:
                print(f"      CORPUS PLAYER: source={p.corpus_source}, scraped={p.corpus_games_scraped}")
            if p.ladder_game_modes:
                print(f"      Other modes: {', '.join(p.ladder_game_modes[:3])}")
            print(f"      Total DB appearances: {p.total_appearances_in_db}")
            print()

    if others:
        print("\n  Players you BEAT:")
        print()
        for tag, p in sorted(others.items(), key=lambda x: x[1].name):
            best_known = _best_trophy_estimate(p)
            print(f"    {p.name:<20} [{p.clan}]  — {best_known}")
            if p.api_best_trophies:
                print(f"      API: {p.api_trophies} current, {p.api_best_trophies} best")
            print()

    # Summary analysis
    print("  " + "=" * 68)
    print("  ANALYSIS: IS THIS WHERE ELITE PLAYERS PRACTICE?")
    print("  " + "=" * 68)
    print()

    elite_count = sum(1 for p in profiles.values() if _is_elite(p))
    known_count = sum(1 for p in profiles.values() if _best_trophy_estimate(p) != "unknown")

    print(f"  Opponents with known trophy data: {known_count}/{len(profiles)}")
    print(f"  Opponents confirmed elite (10K+): {elite_count}/{len(profiles)}")
    print()

    if elite_count > 0:
        print(f"  At least {elite_count}/{len(profiles)} opponents are elite-tier players.")
        print("  This mode appears to attract high-level players for level-capped practice.")
    else:
        print("  Insufficient trophy data to confirm elite player pool.")
        print("  Try --fetch-profiles to query the CR API for live trophy data.")
    print()

    # Score pattern analysis
    narrow_losses = sum(1 for b in battles if b.result != "win" and b.opponent_crowns - b.player_crowns == 1)
    blowout_losses = sum(1 for b in battles if b.result != "win" and b.opponent_crowns >= 3)
    if losses > 0:
        print(f"  Loss pattern: {narrow_losses} narrow (1-crown), {blowout_losses} blowout (3-crown)")
        if narrow_losses > blowout_losses:
            print("  Most losses are close — skill gap is marginal, not structural.")
        elif blowout_losses > 0:
            print("  Blowout losses suggest specific deck matchup problems.")
    print()

    # Archetype distribution
    loss_archetypes: dict[str, int] = {}
    for b in battles:
        if b.result != "win":
            loss_archetypes[b.opponent_archetype] = loss_archetypes.get(b.opponent_archetype, 0) + 1

    if loss_archetypes:
        print("  Loss archetype distribution:")
        for arch, cnt in sorted(loss_archetypes.items(), key=lambda x: -x[1]):
            print(f"    {arch:<30} ×{cnt}")
        print()


def _best_trophy_estimate(p: OpponentProfile) -> str:
    """Return the best known trophy level for an opponent."""
    candidates = [
        v for v in [
            p.api_best_trophies,
            p.api_trophies,
            p.max_trophies_as_player,
            p.max_trophies_as_opponent,
        ]
        if v and v > 0
    ]
    if candidates:
        return f"{max(candidates):,} trophies"
    return "unknown"


def _is_elite(p: OpponentProfile) -> bool:
    """Check if an opponent is elite (10K+ trophies)."""
    candidates = [
        v for v in [
            p.api_best_trophies,
            p.api_trophies,
            p.max_trophies_as_player,
            p.max_trophies_as_opponent,
        ]
        if v and v > 0
    ]
    return any(v >= 10000 for v in candidates) if candidates else False


# ---------------------------------------------------------------------------
# Cached data merge
# ---------------------------------------------------------------------------

def _merge_cached_api_data(
    exp_db_path: str, profiles: dict[str, OpponentProfile]
) -> None:
    """Merge previously fetched API data from the experiment DB."""
    try:
        conn = sqlite3.connect(exp_db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("""
            SELECT tag, api_trophies, api_best_trophies, api_wins, api_losses,
                   api_battle_count, api_three_crown_wins, api_challenge_max_wins,
                   api_clan_name
            FROM opponent_profiles
            WHERE api_trophies IS NOT NULL
        """)
        for row in cur.fetchall():
            tag = row["tag"]
            if tag in profiles:
                p = profiles[tag]
                if not p.api_trophies:  # Don't overwrite fresher data
                    p.api_trophies = row["api_trophies"]
                    p.api_best_trophies = row["api_best_trophies"]
                    p.api_wins = row["api_wins"]
                    p.api_losses = row["api_losses"]
                    p.api_battle_count = row["api_battle_count"]
                    p.api_three_crown_wins = row["api_three_crown_wins"]
                    p.api_challenge_max_wins = row["api_challenge_max_wins"]
                    p.api_clan_name = row["api_clan_name"]
        conn.close()
        logger.info("Merged cached API data from experiment DB")
    except sqlite3.OperationalError:
        pass  # Table doesn't exist yet


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Classic 1v1 (Showdown_Friendly) investigation"
    )
    parser.add_argument(
        "--main-db", required=True,
        help="Path to the main clash_royale_history.db"
    )
    parser.add_argument(
        "--experiment-db", default=None,
        help="Path for experiment database (default: data/classic_1v1_experiment.db)"
    )
    parser.add_argument(
        "--fetch-profiles", action="store_true",
        help="Fetch live opponent profiles from CR API (requires CR_API_KEY)"
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Skip API calls, use only DB cross-references"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON instead of human-readable report"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    # Resolve paths
    main_db = args.main_db
    if not os.path.exists(main_db):
        logger.error(f"Main database not found: {main_db}")
        sys.exit(1)

    exp_db_path = args.experiment_db or os.path.join(
        os.path.dirname(main_db), "classic_1v1_experiment.db"
    )

    # 1. Extract Showdown_Friendly battles
    logger.info(f"Extracting Showdown_Friendly battles from {main_db}")
    battles = extract_showdown_battles(main_db)
    if not battles:
        logger.warning("No Showdown_Friendly battles found.")
        print("No Showdown_Friendly (classic 1v1) games in the database.")
        sys.exit(0)

    logger.info(f"Found {len(battles)} Showdown_Friendly battles")

    # 2. Build opponent profiles from DB cross-references
    opponent_tags = {b.opponent_tag for b in battles}
    logger.info(f"Cross-referencing {len(opponent_tags)} unique opponents")
    profiles = cross_reference_opponents(main_db, opponent_tags)

    # Fill in names/clans/records from battles
    for b in battles:
        p = profiles[b.opponent_tag]
        p.name = b.opponent_name
        p.clan = b.opponent_clan
        p.showdown_games += 1

    for tag, p in profiles.items():
        tag_battles = [b for b in battles if b.opponent_tag == tag]
        wins = sum(1 for b in tag_battles if b.result == "win")
        losses = len(tag_battles) - wins
        scores = [f"{b.player_crowns}-{b.opponent_crowns}" for b in tag_battles]
        p.showdown_record = f"{wins}W-{losses}L ({', '.join(scores)})"

    # 3. Merge cached API data from experiment DB (if it exists)
    if os.path.exists(exp_db_path):
        _merge_cached_api_data(exp_db_path, profiles)

    # 4. Optional: fetch live profiles from CR API
    if args.fetch_profiles and not args.offline:
        logger.info("Fetching live profiles from CR API...")
        fetch_opponent_profiles(profiles)

    # 6. Write experiment database
    logger.info(f"Writing experiment database: {exp_db_path}")
    conn = init_experiment_db(exp_db_path)
    write_results(conn, battles, profiles)
    conn.close()

    # 7. Report
    if args.json:
        output = {
            "battles": [
                {
                    "battle_id": b.battle_id,
                    "battle_time": b.battle_time,
                    "result": b.result,
                    "score": f"{b.player_crowns}-{b.opponent_crowns}",
                    "opponent_name": b.opponent_name,
                    "opponent_tag": b.opponent_tag,
                    "opponent_deck": b.opponent_deck,
                    "opponent_archetype": b.opponent_archetype,
                }
                for b in battles
            ],
            "profiles": {
                tag: {
                    "name": p.name,
                    "clan": p.clan,
                    "showdown_record": p.showdown_record,
                    "best_trophy_estimate": _best_trophy_estimate(p),
                    "is_elite": _is_elite(p),
                    "is_corpus_player": p.is_corpus_player,
                    "api_trophies": p.api_trophies,
                    "api_best_trophies": p.api_best_trophies,
                    "total_db_appearances": p.total_appearances_in_db,
                }
                for tag, p in profiles.items()
            },
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(battles, profiles)


if __name__ == "__main__":
    main()
