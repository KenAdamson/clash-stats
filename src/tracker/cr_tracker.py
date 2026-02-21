#!/usr/bin/env python3
"""
Clash Royale Battle Tracker

Polls the CR API, stores match history in SQLite, and provides analytics.
Since the API only returns the last 25 battles, we poll regularly to build
a complete historical record over time.

Setup:
1. Get API key from https://developer.clashroyale.com
2. Set environment variables or pass directly:
   - CR_API_KEY: Your API token
   - CR_PLAYER_TAG: Your player tag (without #)
3. Run periodically (cron, scheduled task, etc.)

Usage:
    python cr_tracker.py --fetch          # Fetch and store new battles
    python cr_tracker.py --stats          # Show overall stats
    python cr_tracker.py --deck-stats     # Show per-deck statistics
    python cr_tracker.py --crowns         # Crown distribution analysis
    python cr_tracker.py --matchups       # Card matchup analysis
    python cr_tracker.py --recent N       # Show last N battles
    python cr_tracker.py --streaks        # Win/loss streak analysis
    python cr_tracker.py --rolling N      # Rolling window stats (last N games)
    python cr_tracker.py --trophy-history # Trophy progression over time
    python cr_tracker.py --archetypes     # Opponent archetype analysis
    python cr_tracker.py --export csv     # Export data as CSV or JSON
"""

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime
from typing import List, Tuple
import urllib.request
import urllib.error
import urllib.parse

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_API_URL = "https://api.clashroyale.com/v1"
DB_FILE = "clash_royale_history.db"

# Win-condition cards → archetype classification for opponent decks
ARCHETYPES: dict[str, list[str]] = {
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
}

# =============================================================================
# DATABASE SCHEMA
# =============================================================================

SCHEMA = """
-- Player profile snapshots (for tracking all-time stats over time)
CREATE TABLE IF NOT EXISTS player_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    player_tag TEXT NOT NULL,
    name TEXT,
    exp_level INTEGER,
    trophies INTEGER,
    best_trophies INTEGER,
    wins INTEGER,
    losses INTEGER,
    battle_count INTEGER,
    three_crown_wins INTEGER,
    challenge_cards_won INTEGER,
    challenge_max_wins INTEGER,
    tournament_battle_count INTEGER,
    tournament_cards_won INTEGER,
    war_day_wins INTEGER,
    total_donations INTEGER,
    clan_tag TEXT,
    clan_name TEXT,
    arena_name TEXT,
    raw_json TEXT
);

-- Individual battles (the gold!)
CREATE TABLE IF NOT EXISTS battles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    battle_id TEXT UNIQUE NOT NULL,  -- Hash for deduplication
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    battle_time TEXT,
    battle_type TEXT,
    arena_name TEXT,
    game_mode_name TEXT,
    is_ladder_tournament INTEGER,
    
    -- Player (you)
    player_tag TEXT NOT NULL,
    player_name TEXT,
    player_starting_trophies INTEGER,
    player_trophy_change INTEGER,
    player_crowns INTEGER,
    player_king_tower_hp INTEGER,
    player_princess_tower_hp TEXT,
    player_deck TEXT,
    player_deck_hash TEXT,
    
    -- Opponent
    opponent_tag TEXT,
    opponent_name TEXT,
    opponent_starting_trophies INTEGER,
    opponent_trophy_change INTEGER,
    opponent_crowns INTEGER,
    opponent_king_tower_hp INTEGER,
    opponent_princess_tower_hp TEXT,
    opponent_deck TEXT,
    opponent_deck_hash TEXT,
    
    -- Derived
    result TEXT,  -- 'win', 'loss', 'draw'
    crown_differential INTEGER,
    raw_json TEXT
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_battles_player_tag ON battles(player_tag);
CREATE INDEX IF NOT EXISTS idx_battles_battle_time ON battles(battle_time);
CREATE INDEX IF NOT EXISTS idx_battles_player_deck_hash ON battles(player_deck_hash);
CREATE INDEX IF NOT EXISTS idx_battles_result ON battles(result);
CREATE INDEX IF NOT EXISTS idx_battles_battle_type ON battles(battle_type);

-- Card appearances (for matchup analysis)
CREATE TABLE IF NOT EXISTS deck_cards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    battle_id TEXT NOT NULL,
    card_name TEXT NOT NULL,
    card_level INTEGER,
    card_max_level INTEGER,
    card_elixir INTEGER,
    is_player_deck INTEGER,  -- 1 = your deck, 0 = opponent
    FOREIGN KEY (battle_id) REFERENCES battles(battle_id)
);

CREATE INDEX IF NOT EXISTS idx_deck_cards_card_name ON deck_cards(card_name);
CREATE INDEX IF NOT EXISTS idx_deck_cards_battle_id ON deck_cards(battle_id);
"""

# =============================================================================
# API CLIENT
# =============================================================================

class ClashRoyaleAPI:
    """Simple CR API client using stdlib only."""
    
    def __init__(self, api_key: str, base_url: str = DEFAULT_API_URL):
        self.api_key = api_key
        self.base_url = base_url
    
    def _request(self, endpoint: str) -> dict:
        """Make authenticated API request."""
        url = f"{self.base_url}{endpoint}"
        
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {self.api_key}")
        req.add_header("Accept", "application/json")
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else ""
            raise Exception(f"API Error {e.code}: {e.reason}\n{error_body}")
        except urllib.error.URLError as e:
            raise Exception(f"Connection Error: {e.reason}")
    
    def get_player(self, player_tag: str) -> dict:
        """Get player profile."""
        encoded_tag = urllib.parse.quote(f"#{player_tag}" if not player_tag.startswith("#") else player_tag)
        return self._request(f"/players/{encoded_tag}")
    
    def get_battle_log(self, player_tag: str) -> list:
        """Get last 25 battles."""
        encoded_tag = urllib.parse.quote(f"#{player_tag}" if not player_tag.startswith("#") else player_tag)
        return self._request(f"/players/{encoded_tag}/battlelog")


# =============================================================================
# DATABASE
# =============================================================================

class BattleDatabase:
    """SQLite database for battle history."""
    
    def __init__(self, db_file: str = DB_FILE):
        self.db_file = db_file
        self.conn = sqlite3.connect(db_file)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
    
    def _init_schema(self):
        """Create tables and run any pending migrations."""
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._run_migrations()

    def _get_schema_version(self) -> int:
        """Get current schema version, creating the table if needed."""
        try:
            cursor = self.conn.execute("SELECT MAX(version) FROM schema_version")
            row = cursor.fetchone()
            return row[0] if row[0] is not None else 0
        except sqlite3.OperationalError:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
            )
            self.conn.commit()
            return 0

    def _run_migrations(self):
        """Run all pending schema migrations."""
        current = self._get_schema_version()

        if current < 1:
            self._migrate_v1()
            self.conn.execute("INSERT INTO schema_version (version) VALUES (1)")
            self.conn.commit()

    def _migrate_v1(self):
        """Add evo/star tracking, elixir leak, battle duration columns. Backfill from raw_json."""
        alter_statements = [
            "ALTER TABLE deck_cards ADD COLUMN evolution_level INTEGER DEFAULT 0",
            "ALTER TABLE deck_cards ADD COLUMN star_level INTEGER DEFAULT 0",
            "ALTER TABLE battles ADD COLUMN player_elixir_leaked REAL",
            "ALTER TABLE battles ADD COLUMN opponent_elixir_leaked REAL",
            "ALTER TABLE battles ADD COLUMN battle_duration INTEGER",
        ]
        for sql in alter_statements:
            try:
                self.conn.execute(sql)
            except sqlite3.OperationalError:
                pass  # Column already exists

        self._backfill_from_raw_json()
        self._backfill_deck_hashes()
        self.conn.commit()

    def _backfill_from_raw_json(self):
        """Backfill new columns from raw_json for existing battles."""
        cursor = self.conn.execute(
            "SELECT id, battle_id, raw_json FROM battles "
            "WHERE player_elixir_leaked IS NULL AND raw_json IS NOT NULL"
        )
        for row in cursor.fetchall():
            try:
                battle = json.loads(row["raw_json"])
                team = battle.get("team", [{}])[0]
                opponent = battle.get("opponent", [{}])[0]
                self.conn.execute(
                    "UPDATE battles SET player_elixir_leaked=?, opponent_elixir_leaked=?, "
                    "battle_duration=? WHERE id=?",
                    (
                        team.get("elixirLeaked"),
                        opponent.get("elixirLeaked"),
                        battle.get("battleDuration"),
                        row["id"],
                    ),
                )
                # Backfill deck_cards evolution_level and star_level
                for card in team.get("cards", []):
                    self.conn.execute(
                        "UPDATE deck_cards SET evolution_level=?, star_level=? "
                        "WHERE battle_id=? AND card_name=? AND is_player_deck=1",
                        (
                            card.get("evolutionLevel", 0),
                            card.get("starLevel", 0),
                            row["battle_id"],
                            card.get("name"),
                        ),
                    )
                for card in opponent.get("cards", []):
                    self.conn.execute(
                        "UPDATE deck_cards SET evolution_level=?, star_level=? "
                        "WHERE battle_id=? AND card_name=? AND is_player_deck=0",
                        (
                            card.get("evolutionLevel", 0),
                            card.get("starLevel", 0),
                            row["battle_id"],
                            card.get("name"),
                        ),
                    )
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"Warning: Could not backfill battle {row['battle_id']}: {e}")

    def _backfill_deck_hashes(self):
        """Recompute deck hashes to include evolution level."""
        cursor = self.conn.execute(
            "SELECT id, player_deck, opponent_deck FROM battles"
        )
        for row in cursor.fetchall():
            try:
                player_deck = json.loads(row["player_deck"]) if row["player_deck"] else []
                opponent_deck = json.loads(row["opponent_deck"]) if row["opponent_deck"] else []
                self.conn.execute(
                    "UPDATE battles SET player_deck_hash=?, opponent_deck_hash=? WHERE id=?",
                    (
                        self._generate_deck_hash(player_deck),
                        self._generate_deck_hash(opponent_deck),
                        row["id"],
                    ),
                )
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not recompute deck hash for battle id={row['id']}: {e}")
    
    def close(self):
        self.conn.close()
    
    def _generate_battle_id(self, battle: dict) -> str:
        """Generate unique ID for deduplication."""
        key_data = json.dumps({
            'battleTime': battle.get('battleTime'),
            'team': [(p.get('tag'), p.get('crowns')) for p in battle.get('team', [])],
            'opponent': [(p.get('tag'), p.get('crowns')) for p in battle.get('opponent', [])]
        }, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]
    
    def _generate_deck_hash(self, deck: list) -> str:
        """Generate hash for deck (ignoring levels, but including evo status)."""
        card_keys = sorted([
            f"{card.get('name', '')}:evo{card.get('evolutionLevel', 0)}"
            for card in deck
        ])
        return hashlib.md5('|'.join(card_keys).encode()).hexdigest()[:16]
    
    def battle_exists(self, battle_id: str) -> bool:
        cursor = self.conn.execute("SELECT 1 FROM battles WHERE battle_id = ?", (battle_id,))
        return cursor.fetchone() is not None
    
    def store_player_snapshot(self, player: dict):
        """Store player profile snapshot."""
        self.conn.execute("""
            INSERT INTO player_snapshots (
                player_tag, name, exp_level, trophies, best_trophies,
                wins, losses, battle_count, three_crown_wins,
                challenge_cards_won, challenge_max_wins,
                tournament_battle_count, tournament_cards_won,
                war_day_wins, total_donations, clan_tag, clan_name,
                arena_name, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            player.get('tag'),
            player.get('name'),
            player.get('expLevel'),
            player.get('trophies'),
            player.get('bestTrophies'),
            player.get('wins'),
            player.get('losses'),
            player.get('battleCount'),
            player.get('threeCrownWins'),
            player.get('challengeCardsWon'),
            player.get('challengeMaxWins'),
            player.get('tournamentBattleCount'),
            player.get('tournamentCardsWon'),
            player.get('warDayWins'),
            player.get('totalDonations'),
            player.get('clan', {}).get('tag'),
            player.get('clan', {}).get('name'),
            player.get('arena', {}).get('name'),
            json.dumps(player)
        ))
        self.conn.commit()
    
    def store_battle(self, battle: dict, player_tag: str) -> Tuple[str, bool]:
        """Store a battle. Returns (battle_id, is_new)."""
        battle_id = self._generate_battle_id(battle)
        
        if self.battle_exists(battle_id):
            return battle_id, False
        
        team = battle.get('team', [{}])[0]
        opponent = battle.get('opponent', [{}])[0]
        
        player_deck = team.get('cards', [])
        opponent_deck = opponent.get('cards', [])
        
        player_crowns = team.get('crowns', 0)
        opponent_crowns = opponent.get('crowns', 0)
        
        if player_crowns > opponent_crowns:
            result = 'win'
        elif player_crowns < opponent_crowns:
            result = 'loss'
        else:
            result = 'draw'
        
        self.conn.execute("""
            INSERT INTO battles (
                battle_id, battle_time, battle_type, arena_name, game_mode_name,
                is_ladder_tournament, player_tag, player_name, player_starting_trophies,
                player_trophy_change, player_crowns, player_king_tower_hp,
                player_princess_tower_hp, player_deck, player_deck_hash,
                opponent_tag, opponent_name, opponent_starting_trophies,
                opponent_trophy_change, opponent_crowns, opponent_king_tower_hp,
                opponent_princess_tower_hp, opponent_deck, opponent_deck_hash,
                result, crown_differential, raw_json,
                player_elixir_leaked, opponent_elixir_leaked, battle_duration
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            battle_id,
            battle.get('battleTime'),
            battle.get('type'),
            battle.get('arena', {}).get('name'),
            battle.get('gameMode', {}).get('name'),
            1 if battle.get('isLadderTournament') else 0,
            team.get('tag'),
            team.get('name'),
            team.get('startingTrophies'),
            team.get('trophyChange'),
            player_crowns,
            team.get('kingTowerHitPoints'),
            json.dumps(team.get('princessTowersHitPoints', [])),
            json.dumps(player_deck),
            self._generate_deck_hash(player_deck),
            opponent.get('tag'),
            opponent.get('name'),
            opponent.get('startingTrophies'),
            opponent.get('trophyChange'),
            opponent_crowns,
            opponent.get('kingTowerHitPoints'),
            json.dumps(opponent.get('princessTowersHitPoints', [])),
            json.dumps(opponent_deck),
            self._generate_deck_hash(opponent_deck),
            result,
            player_crowns - opponent_crowns,
            json.dumps(battle),
            team.get('elixirLeaked'),
            opponent.get('elixirLeaked'),
            battle.get('battleDuration'),
        ))
        
        # Store individual cards for matchup analysis
        for card in player_deck:
            self.conn.execute("""
                INSERT INTO deck_cards (battle_id, card_name, card_level, card_max_level,
                    card_elixir, is_player_deck, evolution_level, star_level)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?)
            """, (battle_id, card.get('name'), card.get('level'), card.get('maxLevel'),
                  card.get('elixirCost'), card.get('evolutionLevel', 0), card.get('starLevel', 0)))

        for card in opponent_deck:
            self.conn.execute("""
                INSERT INTO deck_cards (battle_id, card_name, card_level, card_max_level,
                    card_elixir, is_player_deck, evolution_level, star_level)
                VALUES (?, ?, ?, ?, ?, 0, ?, ?)
            """, (battle_id, card.get('name'), card.get('level'), card.get('maxLevel'),
                  card.get('elixirCost'), card.get('evolutionLevel', 0), card.get('starLevel', 0)))
        
        self.conn.commit()
        return battle_id, True
    
    def get_total_battles(self) -> int:
        cursor = self.conn.execute("SELECT COUNT(*) FROM battles")
        return cursor.fetchone()[0]
    
    def get_overall_stats(self) -> dict:
        cursor = self.conn.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN result = 'draw' THEN 1 ELSE 0 END) as draws,
                SUM(player_crowns) as total_crowns,
                SUM(opponent_crowns) as crowns_against,
                SUM(CASE WHEN player_crowns = 3 THEN 1 ELSE 0 END) as three_crowns,
                AVG(player_crowns) as avg_crowns,
                AVG(opponent_crowns) as avg_crowns_against,
                MIN(battle_time) as first_battle,
                MAX(battle_time) as last_battle
            FROM battles
        """)
        row = cursor.fetchone()
        return dict(row) if row else {}
    
    def get_all_time_api_stats(self) -> dict:
        """Get latest snapshot from API (all-time stats)."""
        cursor = self.conn.execute("""
            SELECT * FROM player_snapshots
            ORDER BY id DESC LIMIT 1
        """)
        row = cursor.fetchone()
        return dict(row) if row else {}
    
    def get_stats_by_battle_type(self) -> List[dict]:
        cursor = self.conn.execute("""
            SELECT 
                battle_type,
                COUNT(*) as total,
                SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) as losses,
                ROUND(100.0 * SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate
            FROM battles
            GROUP BY battle_type
            ORDER BY total DESC
        """)
        return [dict(row) for row in cursor.fetchall()]
    
    def get_deck_stats(self, min_battles: int = 3) -> List[dict]:
        cursor = self.conn.execute("""
            SELECT 
                player_deck_hash,
                player_deck,
                COUNT(*) as total,
                SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) as losses,
                ROUND(100.0 * SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate,
                SUM(CASE WHEN player_crowns = 3 THEN 1 ELSE 0 END) as three_crowns,
                ROUND(AVG(player_crowns), 2) as avg_crowns
            FROM battles
            GROUP BY player_deck_hash
            HAVING COUNT(*) >= ?
            ORDER BY total DESC
        """, (min_battles,))
        
        results = []
        for row in cursor.fetchall():
            d = dict(row)
            deck_json = json.loads(d['player_deck'])
            d['deck_cards'] = sorted([c.get('name', 'Unknown') for c in deck_json])
            results.append(d)
        return results
    
    def get_crown_distribution(self) -> dict:
        cursor = self.conn.execute("""
            SELECT 
                result,
                player_crowns,
                COUNT(*) as count
            FROM battles
            WHERE result IN ('win', 'loss')
            GROUP BY result, player_crowns
            ORDER BY result, player_crowns
        """)
        
        distribution = {'win': {}, 'loss': {}}
        for row in cursor.fetchall():
            distribution[row['result']][row['player_crowns']] = row['count']
        return distribution
    
    def get_card_matchup_stats(self, min_battles: int = 3) -> List[dict]:
        """Win rate vs opponent cards."""
        cursor = self.conn.execute("""
            SELECT 
                dc.card_name,
                COUNT(DISTINCT b.battle_id) as times_faced,
                SUM(CASE WHEN b.result = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN b.result = 'loss' THEN 1 ELSE 0 END) as losses,
                ROUND(100.0 * SUM(CASE WHEN b.result = 'win' THEN 1 ELSE 0 END) / COUNT(DISTINCT b.battle_id), 1) as win_rate
            FROM deck_cards dc
            JOIN battles b ON dc.battle_id = b.battle_id
            WHERE dc.is_player_deck = 0
            GROUP BY dc.card_name
            HAVING COUNT(DISTINCT b.battle_id) >= ?
            ORDER BY times_faced DESC
        """, (min_battles,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_recent_battles(self, limit: int = 10) -> List[dict]:
        cursor = self.conn.execute("""
            SELECT 
                battle_time, battle_type, arena_name,
                player_crowns, opponent_crowns, result,
                player_trophy_change, player_starting_trophies,
                opponent_name, player_deck
            FROM battles
            ORDER BY battle_time DESC
            LIMIT ?
        """, (limit,))
        
        results = []
        for row in cursor.fetchall():
            d = dict(row)
            deck_json = json.loads(d['player_deck'])
            d['deck_cards'] = [c.get('name', 'Unknown') for c in deck_json]
            results.append(d)
        return results
    
    def get_time_of_day_stats(self) -> List[dict]:
        """Win rate by hour of day."""
        cursor = self.conn.execute("""
            SELECT
                CAST(SUBSTR(battle_time, 10, 2) AS INTEGER) as hour,
                COUNT(*) as total,
                SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins,
                ROUND(100.0 * SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate
            FROM battles
            WHERE battle_time IS NOT NULL
            GROUP BY hour
            ORDER BY hour
        """)
        return [dict(row) for row in cursor.fetchall()]

    def get_streaks(self) -> dict:
        """Detect win/loss streaks from battle history.

        Returns:
            Dict with 'current_streak', 'longest_win_streak',
            'longest_loss_streak', and 'streaks' list.
        """
        cursor = self.conn.execute("""
            SELECT battle_time, result, player_starting_trophies, player_trophy_change
            FROM battles
            WHERE result IN ('win', 'loss')
            ORDER BY battle_time ASC
        """)
        rows = [dict(r) for r in cursor.fetchall()]

        if not rows:
            return {
                "current_streak": None,
                "longest_win_streak": None,
                "longest_loss_streak": None,
                "streaks": [],
            }

        streaks: list[dict] = []
        current_type = rows[0]["result"]
        current_start = rows[0]
        current_length = 1
        current_end = rows[0]

        def _finish_streak(stype: str, length: int, start: dict, end: dict) -> dict:
            start_trophies = start.get("player_starting_trophies") or 0
            end_trophies = (end.get("player_starting_trophies") or 0) + (end.get("player_trophy_change") or 0)
            return {
                "type": stype,
                "length": length,
                "start_trophies": start_trophies,
                "end_trophies": end_trophies,
                "start_date": (start.get("battle_time") or "")[:8],
                "end_date": (end.get("battle_time") or "")[:8],
            }

        for row in rows[1:]:
            if row["result"] == current_type:
                current_length += 1
                current_end = row
            else:
                streaks.append(_finish_streak(current_type, current_length, current_start, current_end))
                current_type = row["result"]
                current_start = row
                current_end = row
                current_length = 1

        streaks.append(_finish_streak(current_type, current_length, current_start, current_end))

        win_streaks = [s for s in streaks if s["type"] == "win"]
        loss_streaks = [s for s in streaks if s["type"] == "loss"]

        return {
            "current_streak": streaks[-1] if streaks else None,
            "longest_win_streak": max(win_streaks, key=lambda s: s["length"]) if win_streaks else None,
            "longest_loss_streak": max(loss_streaks, key=lambda s: s["length"]) if loss_streaks else None,
            "streaks": streaks,
        }

    def get_rolling_stats(self, window: int = 35) -> dict:
        """Get win rate over the last N games.

        Args:
            window: Number of recent games to analyze.

        Returns:
            Dict with total, wins, losses, draws, win_rate, three_crowns,
            avg_crowns, trophy_change, and per-game details.
        """
        cursor = self.conn.execute("""
            SELECT result, player_crowns, opponent_crowns, player_trophy_change,
                   player_starting_trophies, battle_time, battle_type
            FROM battles
            ORDER BY battle_time DESC
            LIMIT ?
        """, (window,))
        rows = [dict(r) for r in cursor.fetchall()]

        if not rows:
            return {"total": 0, "wins": 0, "losses": 0, "draws": 0,
                    "win_rate": 0.0, "three_crowns": 0, "avg_crowns": 0.0,
                    "trophy_change": 0}

        total = len(rows)
        wins = sum(1 for r in rows if r["result"] == "win")
        losses = sum(1 for r in rows if r["result"] == "loss")
        draws = sum(1 for r in rows if r["result"] == "draw")
        three_crowns = sum(1 for r in rows if r["player_crowns"] == 3)
        total_crowns = sum(r["player_crowns"] or 0 for r in rows)
        trophy_change = sum(r["player_trophy_change"] or 0 for r in rows)

        return {
            "total": total,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": round(wins / total * 100, 1) if total > 0 else 0.0,
            "three_crowns": three_crowns,
            "avg_crowns": round(total_crowns / total, 2) if total > 0 else 0.0,
            "trophy_change": trophy_change,
        }

    def get_trophy_history(self) -> List[dict]:
        """Get trophy progression over time from battle data.

        Returns:
            List of dicts with battle_time, trophies (after battle), result,
            player_trophy_change, ordered chronologically.
        """
        cursor = self.conn.execute("""
            SELECT battle_time, player_starting_trophies, player_trophy_change, result
            FROM battles
            WHERE player_starting_trophies IS NOT NULL
            ORDER BY battle_time ASC
        """)
        results = []
        for row in cursor.fetchall():
            d = dict(row)
            change = d["player_trophy_change"] or 0
            d["trophies"] = (d["player_starting_trophies"] or 0) + change
            results.append(d)
        return results

    @staticmethod
    def classify_archetype(deck: list[dict]) -> str:
        """Classify an opponent deck into an archetype based on win condition cards.

        Args:
            deck: List of card dicts from the API.

        Returns:
            Archetype name string, or "Unknown" if no match.
        """
        card_names = {card.get("name", "") for card in deck}
        for archetype, win_conditions in ARCHETYPES.items():
            if any(wc in card_names for wc in win_conditions):
                return archetype
        return "Unknown"

    def get_archetype_stats(self, min_battles: int = 3) -> List[dict]:
        """Cluster opponent decks into archetypes and show win rates.

        Args:
            min_battles: Minimum battles to include an archetype.

        Returns:
            List of dicts with archetype, total, wins, losses, win_rate.
        """
        cursor = self.conn.execute("""
            SELECT opponent_deck, result FROM battles
            WHERE opponent_deck IS NOT NULL
        """)

        archetype_data: dict[str, dict] = {}
        for row in cursor.fetchall():
            try:
                deck = json.loads(row["opponent_deck"])
            except (json.JSONDecodeError, TypeError):
                continue
            archetype = self.classify_archetype(deck)
            if archetype not in archetype_data:
                archetype_data[archetype] = {"wins": 0, "losses": 0, "draws": 0}
            entry = archetype_data[archetype]
            if row["result"] == "win":
                entry["wins"] += 1
            elif row["result"] == "loss":
                entry["losses"] += 1
            else:
                entry["draws"] += 1

        results = []
        for archetype, data in archetype_data.items():
            total = data["wins"] + data["losses"] + data["draws"]
            if total >= min_battles:
                results.append({
                    "archetype": archetype,
                    "total": total,
                    "wins": data["wins"],
                    "losses": data["losses"],
                    "win_rate": round(data["wins"] / total * 100, 1) if total > 0 else 0.0,
                })
        return sorted(results, key=lambda x: x["total"], reverse=True)

    def get_snapshot_diff(self) -> dict | None:
        """Compare the two most recent player snapshots.

        Returns:
            Dict with field-level diffs, or None if fewer than 2 snapshots.
        """
        cursor = self.conn.execute("""
            SELECT * FROM player_snapshots
            ORDER BY id DESC LIMIT 2
        """)
        rows = cursor.fetchall()
        if len(rows) < 2:
            return None

        current, previous = dict(rows[0]), dict(rows[1])
        diff_fields = [
            "trophies", "best_trophies", "wins", "losses", "battle_count",
            "three_crown_wins", "war_day_wins", "total_donations",
        ]
        diff: dict = {}
        for field in diff_fields:
            old_val = previous.get(field) or 0
            new_val = current.get(field) or 0
            diff[field] = new_val - old_val
        return diff


# =============================================================================
# REPORTING
# =============================================================================

def print_overall_stats(db: BattleDatabase):
    """Print comprehensive stats report."""
    
    # All-time from API
    api_stats = db.get_all_time_api_stats()
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
        
        wins = api_stats.get('wins', 0)
        losses = api_stats.get('losses', 0)
        battles = api_stats.get('battle_count', 0)
        three_crowns = api_stats.get('three_crown_wins', 0)
        
        win_rate = (wins / battles * 100) if battles > 0 else 0
        three_crown_rate = (three_crowns / wins * 100) if wins > 0 else 0
        
        print(f"  Battles:       {battles:,}")
        print(f"  Wins:          {wins:,} ({win_rate:.1f}%)")
        print(f"  Losses:        {losses:,}")
        print(f"  3-Crown Wins:  {three_crowns:,} ({three_crown_rate:.1f}% of wins)")
        print(f"  War Day Wins:  {api_stats.get('war_day_wins', 0):,}")
    
    # Tracked battles
    stats = db.get_overall_stats()
    total = stats.get('total', 0)
    
    print()
    print("=" * 70)
    print("TRACKED BATTLES (our SQLite database)")
    print("=" * 70)
    print()
    
    if total == 0:
        print("  No battles tracked yet. Run with --fetch to start!")
        return
    
    wins = stats.get('wins', 0)
    losses = stats.get('losses', 0)
    draws = stats.get('draws', 0)
    three_crowns = stats.get('three_crowns', 0)
    
    win_rate = (wins / total * 100) if total > 0 else 0
    three_crown_rate = (three_crowns / wins * 100) if wins > 0 else 0
    
    print(f"  Total Tracked:   {total:,}")
    print(f"  Date Range:      {stats.get('first_battle', 'N/A')[:10]} to {stats.get('last_battle', 'N/A')[:10]}")
    print()
    print(f"  Wins:            {wins:,} ({win_rate:.1f}%)")
    print(f"  Losses:          {losses:,} ({losses/total*100:.1f}%)")
    print(f"  Draws:           {draws:,} ({draws/total*100:.1f}%)")
    print()
    print(f"  3-Crown Wins:    {three_crowns:,} ({three_crown_rate:.1f}% of wins)")
    print(f"  Avg Crowns:      {stats.get('avg_crowns', 0):.2f} (opponent: {stats.get('avg_crowns_against', 0):.2f})")
    
    # By battle type
    type_stats = db.get_stats_by_battle_type()
    if type_stats:
        print()
        print("  BY MODE:")
        for ts in type_stats:
            print(f"    {ts['battle_type']:28} {ts['total']:4} games  {ts['win_rate']:5.1f}% WR")
    
    # Time of day analysis
    time_stats = db.get_time_of_day_stats()
    if time_stats and len(time_stats) >= 3:
        print()
        print("  BY TIME OF DAY (UTC):")
        sorted_times = sorted(time_stats, key=lambda x: x['win_rate'], reverse=True)
        best = sorted_times[0]
        worst = sorted_times[-1]
        print(f"    Best Hour:   {best['hour']:02d}:00 ({best['win_rate']:.1f}% WR, {best['total']} games)")
        print(f"    Worst Hour:  {worst['hour']:02d}:00 ({worst['win_rate']:.1f}% WR, {worst['total']} games)")
    print()


def print_deck_stats(db: BattleDatabase):
    deck_stats = db.get_deck_stats(min_battles=3)
    
    print()
    print("=" * 70)
    print("DECK PERFORMANCE (min 3 battles)")
    print("=" * 70)
    print()
    
    if not deck_stats:
        print("  Not enough data per deck yet. Keep playing!")
        return
    
    for i, ds in enumerate(deck_stats[:10], 1):
        cards = ", ".join(ds['deck_cards'])
        print(f"  Deck #{i}")
        print(f"    Cards:     {cards}")
        print(f"    Battles:   {ds['total']}  |  Win Rate: {ds['win_rate']}%  |  3-Crowns: {ds['three_crowns']}")
        print()


def print_crown_distribution(db: BattleDatabase):
    dist = db.get_crown_distribution()
    
    print()
    print("=" * 70)
    print("CROWN DISTRIBUTION")
    print("=" * 70)
    print()
    
    print("  WINS by crowns earned:")
    total_wins = sum(dist['win'].values()) if dist['win'] else 0
    for crowns in [1, 2, 3]:
        count = dist['win'].get(crowns, 0)
        pct = (count / total_wins * 100) if total_wins > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"    {crowns}-crown: {count:4} ({pct:5.1f}%) {bar}")
    print()
    
    print("  LOSSES by crowns earned:")
    total_losses = sum(dist['loss'].values()) if dist['loss'] else 0
    for crowns in [0, 1, 2]:
        count = dist['loss'].get(crowns, 0)
        pct = (count / total_losses * 100) if total_losses > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"    {crowns}-crown: {count:4} ({pct:5.1f}%) {bar}")
    print()


def print_matchup_stats(db: BattleDatabase):
    matchups = db.get_card_matchup_stats(min_battles=5)
    
    print()
    print("=" * 70)
    print("CARD MATCHUP ANALYSIS (min 5 games)")
    print("=" * 70)
    print()
    
    if not matchups:
        print("  Not enough data yet.")
        return
    
    problem_cards = sorted(matchups, key=lambda x: x['win_rate'])[:10]
    print("  YOUR TROUBLE CARDS (lowest WR against):")
    for m in problem_cards:
        print(f"    {m['card_name']:22} {m['times_faced']:3}x  →  {m['win_rate']:5.1f}% WR")
    print()
    
    best_matchups = sorted(matchups, key=lambda x: -x['win_rate'])[:10]
    print("  YOUR BEST MATCHUPS (highest WR against):")
    for m in best_matchups:
        print(f"    {m['card_name']:22} {m['times_faced']:3}x  →  {m['win_rate']:5.1f}% WR")
    print()


def print_recent_battles(db: BattleDatabase, limit: int = 10):
    battles = db.get_recent_battles(limit)

    print()
    print("=" * 70)
    print(f"LAST {len(battles)} BATTLES")
    print("=" * 70)
    print()

    if not battles:
        print("  No battles tracked yet.")
        return

    for b in battles:
        result_icon = "✓" if b['result'] == 'win' else ("✗" if b['result'] == 'loss' else "—")
        trophy_change = b['player_trophy_change'] or 0
        trophy_str = f"+{trophy_change}" if trophy_change >= 0 else str(trophy_change)

        print(f"  {result_icon} {b['player_crowns']}-{b['opponent_crowns']} vs {b['opponent_name']:15} | {trophy_str:>4} | {b['battle_type']}")
    print()


def print_streaks(db: BattleDatabase):
    """Print win/loss streak analysis."""
    data = db.get_streaks()

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
    print(f"  Current:       {icon} {cs['length']} {cs['type']}{'s' if cs['length'] != 1 else ''} ({cs['start_trophies']} → {cs['end_trophies']})")

    if data["longest_win_streak"]:
        ws = data["longest_win_streak"]
        print(f"  Best Win Run:  {ws['length']} wins ({ws['start_trophies']} → {ws['end_trophies']})")

    if data["longest_loss_streak"]:
        ls = data["longest_loss_streak"]
        print(f"  Worst Tilt:    {ls['length']} losses ({ls['start_trophies']} → {ls['end_trophies']})")

    # Show streak summary
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


def print_rolling_stats(db: BattleDatabase, window: int = 35):
    """Print rolling window stats for last N games."""
    stats = db.get_rolling_stats(window)

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

    # Compare to overall
    overall = db.get_overall_stats()
    overall_total = overall.get("total", 0)
    if overall_total > 0:
        overall_wins = overall.get("wins", 0)
        overall_wr = round(overall_wins / overall_total * 100, 1)
        diff = stats["win_rate"] - overall_wr
        direction = "above" if diff > 0 else "below"
        print(f"  vs Overall:    {abs(diff):.1f}pp {direction} ({overall_wr:.1f}% overall)")
    print()


def print_trophy_history(db: BattleDatabase):
    """Print trophy progression as an ASCII chart."""
    history = db.get_trophy_history()

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

    # Show start, end, and range
    print(f"  Range: {min_t:,} - {max_t:,} ({max_t - min_t:+,} spread)")
    print(f"  Games: {len(history)}")
    print()

    # If many games, sample evenly for display
    max_rows = 30
    if len(history) > max_rows:
        step = len(history) / max_rows
        indices = [int(i * step) for i in range(max_rows)]
        indices[-1] = len(history) - 1  # Always include last
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


def print_archetype_stats(db: BattleDatabase):
    """Print opponent archetype analysis."""
    stats = db.get_archetype_stats(min_battles=3)

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


def export_data(data: list[dict] | dict, fmt: str, output: str | None = None):
    """Export data as CSV or JSON.

    Args:
        data: List of dicts (or single dict) to export.
        fmt: 'csv' or 'json'.
        output: File path, or None for stdout.
    """
    if isinstance(data, dict):
        data = [data]

    if fmt == "json":
        text = json.dumps(data, indent=2, default=str)
    elif fmt == "csv":
        if not data:
            text = ""
        else:
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
            text = buf.getvalue()
    else:
        print(f"Error: Unknown export format '{fmt}'")
        return

    if output:
        with open(output, "w") as f:
            f.write(text)
        print(f"Exported {len(data)} records to {output}")
    else:
        sys.stdout.write(text)


# =============================================================================
# MAIN
# =============================================================================

def fetch_and_store(api_key: str, player_tag: str, db: BattleDatabase,
                    api_url: str = DEFAULT_API_URL):
    """Fetch latest battles and store new ones."""
    api = ClashRoyaleAPI(api_key, base_url=api_url)
    
    print(f"\n🔄 Fetching data for #{player_tag}...")
    
    try:
        player = api.get_player(player_tag)
        db.store_player_snapshot(player)
        print(f"  ✓ Profile: {player.get('name')} | {player.get('trophies'):,} trophies")
        print(f"  ✓ All-time: {player.get('wins'):,}W / {player.get('losses'):,}L / {player.get('threeCrownWins'):,} 3-crowns")

        diff = db.get_snapshot_diff()
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
            battle_id, is_new = db.store_battle(battle, player.get('tag'))
            if is_new:
                new_count += 1
        
        print(f"  ✓ Fetched {len(battles)} battles, {new_count} NEW")
        print(f"  ✓ Total tracked: {db.get_total_battles():,} battles")
    except Exception as e:
        print(f"  ✗ Error fetching battles: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Clash Royale Battle Tracker - Build your historical match database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cr_tracker.py --fetch --api-key YOUR_KEY --player-tag ABC123
  python cr_tracker.py --stats
  python cr_tracker.py --deck-stats
  python cr_tracker.py --crowns
  python cr_tracker.py --matchups
  python cr_tracker.py --recent 20

Environment variables:
  CR_API_KEY     - Your API key from developer.clashroyale.com
  CR_PLAYER_TAG  - Your player tag (without #)
  CR_API_URL     - API base URL (default: https://api.clashroyale.com/v1)
        """
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
    parser.add_argument("--api-key", type=str, help="CR API key")
    parser.add_argument("--player-tag", type=str, help="Player tag (without #)")
    parser.add_argument("--api-url", type=str, help="API base URL (default: https://api.clashroyale.com/v1)")
    parser.add_argument("--db", type=str, default=DB_FILE, help=f"Database file (default: {DB_FILE})")
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get("CR_API_KEY")
    player_tag = args.player_tag or os.environ.get("CR_PLAYER_TAG")
    api_url = args.api_url or os.environ.get("CR_API_URL", DEFAULT_API_URL)
    
    db = BattleDatabase(args.db)
    
    try:
        if args.fetch:
            if not api_key or not player_tag:
                print("Error: --api-key and --player-tag required for fetching")
                print("       Or set CR_API_KEY and CR_PLAYER_TAG environment variables")
                return 1
            fetch_and_store(api_key, player_tag.replace("#", ""), db, api_url=api_url)
        
        # Map analytics commands to (data_fn, print_fn) pairs for export support
        export_fmt = args.export
        export_out = args.output

        if args.stats:
            if export_fmt:
                data = db.get_overall_stats()
                export_data(data, export_fmt, export_out)
            else:
                print_overall_stats(db)

        if args.deck_stats:
            if export_fmt:
                export_data(db.get_deck_stats(min_battles=1), export_fmt, export_out)
            else:
                print_deck_stats(db)

        if args.crowns:
            if export_fmt:
                export_data(db.get_crown_distribution(), export_fmt, export_out)
            else:
                print_crown_distribution(db)

        if args.matchups:
            if export_fmt:
                export_data(db.get_card_matchup_stats(min_battles=1), export_fmt, export_out)
            else:
                print_matchup_stats(db)

        if args.recent:
            if export_fmt:
                export_data(db.get_recent_battles(args.recent), export_fmt, export_out)
            else:
                print_recent_battles(db, args.recent)

        if args.streaks:
            if export_fmt:
                export_data(db.get_streaks(), export_fmt, export_out)
            else:
                print_streaks(db)

        if args.rolling:
            if export_fmt:
                export_data(db.get_rolling_stats(args.rolling), export_fmt, export_out)
            else:
                print_rolling_stats(db, args.rolling)

        if args.trophy_history:
            if export_fmt:
                export_data(db.get_trophy_history(), export_fmt, export_out)
            else:
                print_trophy_history(db)

        if args.archetypes:
            if export_fmt:
                export_data(db.get_archetype_stats(min_battles=1), export_fmt, export_out)
            else:
                print_archetype_stats(db)

        # Default: show help + db status
        has_action = any([args.fetch, args.stats, args.deck_stats, args.crowns,
                         args.matchups, args.recent, args.streaks, args.rolling,
                         args.trophy_history, args.archetypes])
        if not has_action:
            parser.print_help()
            print()
            print(f"Database: {args.db}")
            print(f"Battles tracked: {db.get_total_battles():,}")
        
        return 0
    
    finally:
        db.close()


if __name__ == "__main__":
    exit(main())
