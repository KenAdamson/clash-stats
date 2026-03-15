#!/usr/bin/env python3
"""Backfill deck_cards from raw_json already in MariaDB battles table.

Reads each battle's raw_json, extracts team/opponent card data, and
inserts into deck_cards using INSERT IGNORE (safe to re-run).

Usage:
    python3 scripts/backfill_deck_cards.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pymysql

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("backfill")

MARIADB_URL = os.environ.get(
    "DATABASE_URL",
    "mysql+pymysql://clash_stats:clash_stats_pw_2026@192.168.7.59/clash_stats",
)

CHUNK = 1000   # battles to read per SELECT
SLEEP = 0.02   # seconds between flushes


def _parse_conn_str(url: str):
    # mysql+pymysql://user:pass@host[:port]/db
    url = url.replace("mysql+pymysql://", "")
    userpass, rest = url.split("@", 1)
    user, password = userpass.split(":", 1)
    hostport, db = rest.split("/", 1)
    if ":" in hostport:
        host, port = hostport.rsplit(":", 1)
        port = int(port)
    else:
        host, port = hostport, 3306
    return dict(host=host, port=port, user=user, password=password, db=db,
                connect_timeout=10, charset="utf8mb4")


def _classify_variant(card: dict) -> str:
    evo = card.get("evolutionLevel", 0)
    max_evo = card.get("maxEvolutionLevel", 0)
    if evo and evo > 0:
        return "hero" if (max_evo and max_evo > 1) else "evo"
    return "base"


def main():
    conn_args = _parse_conn_str(MARIADB_URL)
    conn = pymysql.connect(**conn_args)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM battles")
    total = cur.fetchone()[0]
    log.info("Total battles: %d", total)

    insert_sql = """
        INSERT IGNORE INTO deck_cards
            (battle_id, card_name, card_level, card_max_level, card_elixir,
             is_player_deck, evolution_level, star_level, card_variant)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    offset = 0
    total_inserted = 0
    total_battles = 0

    while True:
        cur.execute(
            "SELECT battle_id, raw_json FROM battles "
            "WHERE raw_json IS NOT NULL "
            "ORDER BY battle_time "
            f"LIMIT {CHUNK} OFFSET {offset}"
        )
        rows = cur.fetchall()
        if not rows:
            break

        batch = []
        for battle_id, raw_json in rows:
            try:
                data = json.loads(raw_json)
            except (json.JSONDecodeError, TypeError):
                continue

            for side, is_player in [("team", 1), ("opponent", 0)]:
                participants = data.get(side, [{}])
                if not participants:
                    continue
                for card in participants[0].get("cards", []):
                    name = card.get("name", "")
                    if not name:
                        continue
                    batch.append((
                        battle_id,
                        name,
                        card.get("level"),
                        card.get("maxLevel"),
                        card.get("elixirCost"),
                        is_player,
                        card.get("evolutionLevel", 0),
                        card.get("starLevel", 0),
                        _classify_variant(card),
                    ))

        if batch:
            cur.executemany(insert_sql, batch)
            conn.commit()
            total_inserted += cur.rowcount
            time.sleep(SLEEP)

        total_battles += len(rows)
        offset += len(rows)

        if total_battles % 10000 == 0 or len(rows) < CHUNK:
            log.info("  %d / %d battles | %d deck_cards inserted",
                     total_battles, total, total_inserted)

    log.info("Done. %d battles processed, %d deck_cards inserted.", total_battles, total_inserted)
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
