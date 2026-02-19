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
"""

import argparse
import hashlib
import json
import os
import sqlite3
from datetime import datetime
from typing import List, Tuple
import urllib.request
import urllib.error
import urllib.parse

# =============================================================================
# CONFIGURATION
# =============================================================================

# Use RoyaleAPI proxy to handle dynamic IPs
# When creating your key at developer.clashroyale.com, whitelist IP: 45.79.218.79
USE_PROXY = True
API_BASE_URL = "https://proxy.royaleapi.dev/v1" if USE_PROXY else "https://api.clashroyale.com/v1"

DB_FILE = "clash_royale_history.db"

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
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = API_BASE_URL
    
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
        self.conn.executescript(SCHEMA)
        self.conn.commit()
    
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
        """Generate hash for deck (ignoring levels)."""
        card_names = sorted([card.get('name', '') for card in deck])
        return hashlib.md5('|'.join(card_names).encode()).hexdigest()[:16]
    
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
                result, crown_differential, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            json.dumps(battle)
        ))
        
        # Store individual cards for matchup analysis
        for card in player_deck:
            self.conn.execute("""
                INSERT INTO deck_cards (battle_id, card_name, card_level, card_max_level, card_elixir, is_player_deck)
                VALUES (?, ?, ?, ?, ?, 1)
            """, (battle_id, card.get('name'), card.get('level'), card.get('maxLevel'), card.get('elixirCost')))
        
        for card in opponent_deck:
            self.conn.execute("""
                INSERT INTO deck_cards (battle_id, card_name, card_level, card_max_level, card_elixir, is_player_deck)
                VALUES (?, ?, ?, ?, ?, 0)
            """, (battle_id, card.get('name'), card.get('level'), card.get('maxLevel'), card.get('elixirCost')))
        
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


# =============================================================================
# MAIN
# =============================================================================

def fetch_and_store(api_key: str, player_tag: str, db: BattleDatabase):
    """Fetch latest battles and store new ones."""
    api = ClashRoyaleAPI(api_key)
    
    print(f"\n🔄 Fetching data for #{player_tag}...")
    
    try:
        player = api.get_player(player_tag)
        db.store_player_snapshot(player)
        print(f"  ✓ Profile: {player.get('name')} | {player.get('trophies'):,} trophies")
        print(f"  ✓ All-time: {player.get('wins'):,}W / {player.get('losses'):,}L / {player.get('threeCrownWins'):,} 3-crowns")
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
        """
    )
    parser.add_argument("--fetch", action="store_true", help="Fetch and store new battles")
    parser.add_argument("--stats", action="store_true", help="Show overall statistics")
    parser.add_argument("--deck-stats", action="store_true", help="Show per-deck statistics")
    parser.add_argument("--crowns", action="store_true", help="Show crown distribution")
    parser.add_argument("--matchups", action="store_true", help="Show card matchup analysis")
    parser.add_argument("--recent", type=int, metavar="N", help="Show last N battles")
    parser.add_argument("--api-key", type=str, help="CR API key")
    parser.add_argument("--player-tag", type=str, help="Player tag (without #)")
    parser.add_argument("--db", type=str, default=DB_FILE, help=f"Database file (default: {DB_FILE})")
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get("CR_API_KEY")
    player_tag = args.player_tag or os.environ.get("CR_PLAYER_TAG")
    
    db = BattleDatabase(args.db)
    
    try:
        if args.fetch:
            if not api_key or not player_tag:
                print("Error: --api-key and --player-tag required for fetching")
                print("       Or set CR_API_KEY and CR_PLAYER_TAG environment variables")
                return 1
            fetch_and_store(api_key, player_tag.replace("#", ""), db)
        
        if args.stats:
            print_overall_stats(db)
        
        if args.deck_stats:
            print_deck_stats(db)
        
        if args.crowns:
            print_crown_distribution(db)
        
        if args.matchups:
            print_matchup_stats(db)
        
        if args.recent:
            print_recent_battles(db, args.recent)
        
        # Default: show help + db status
        if not any([args.fetch, args.stats, args.deck_stats, args.crowns, args.matchups, args.recent]):
            parser.print_help()
            print()
            print(f"Database: {args.db}")
            print(f"Battles tracked: {db.get_total_battles():,}")
        
        return 0
    
    finally:
        db.close()


if __name__ == "__main__":
    exit(main())
