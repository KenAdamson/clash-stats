"""Add evolution_level, star_level, elixir leaked, and battle duration columns.

Existing databases created before the SQLAlchemy migration have the old schema
without these columns. This migration adds them and backfills from raw_json.

Revision ID: 002
Revises: 001
Create Date: 2026-02-21
"""
import json
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_exists(table: str, column: str) -> bool:
    """Check if a column already exists on a table (idempotent migrations)."""
    conn = op.get_bind()
    insp = sa.inspect(conn)
    try:
        columns = {c["name"] for c in insp.get_columns(table)}
    except Exception:
        return False
    return column in columns


def upgrade() -> None:
    # -- battles: add elixir leak + duration columns --
    if not _column_exists("battles", "player_elixir_leaked"):
        op.add_column("battles", sa.Column("player_elixir_leaked", sa.Float))
    if not _column_exists("battles", "opponent_elixir_leaked"):
        op.add_column("battles", sa.Column("opponent_elixir_leaked", sa.Float))
    if not _column_exists("battles", "battle_duration"):
        op.add_column("battles", sa.Column("battle_duration", sa.Integer))

    # -- deck_cards: add evo + star level columns --
    if not _column_exists("deck_cards", "evolution_level"):
        op.add_column("deck_cards", sa.Column("evolution_level", sa.Integer, server_default="0"))
    if not _column_exists("deck_cards", "star_level"):
        op.add_column("deck_cards", sa.Column("star_level", sa.Integer, server_default="0"))

    # -- Backfill from raw_json --
    conn = op.get_bind()

    # Backfill battles
    rows = conn.execute(sa.text(
        "SELECT id, raw_json FROM battles WHERE raw_json IS NOT NULL"
    ))
    for row in rows:
        try:
            data = json.loads(row[1])
        except (json.JSONDecodeError, TypeError):
            continue

        player_elixir = None
        opponent_elixir = None
        duration = None

        team = data.get("team", [{}])
        if team:
            player_elixir = team[0].get("elixirLeaked")
        opponent = data.get("opponent", [{}])
        if opponent:
            opponent_elixir = opponent[0].get("elixirLeaked")
        duration = data.get("battleDuration")

        if any(v is not None for v in (player_elixir, opponent_elixir, duration)):
            conn.execute(sa.text(
                "UPDATE battles SET player_elixir_leaked = :pe, "
                "opponent_elixir_leaked = :oe, battle_duration = :dur "
                "WHERE id = :id"
            ), {
                "pe": player_elixir,
                "oe": opponent_elixir,
                "dur": duration,
                "id": row[0],
            })

    # Backfill deck_cards evolution_level and star_level from battles raw_json
    battle_rows = conn.execute(sa.text(
        "SELECT battle_id, raw_json FROM battles WHERE raw_json IS NOT NULL"
    ))
    for battle_row in battle_rows:
        try:
            data = json.loads(battle_row[1])
        except (json.JSONDecodeError, TypeError):
            continue

        bid = battle_row[0]
        for side, is_player in [("team", 1), ("opponent", 0)]:
            participants = data.get(side, [{}])
            if not participants:
                continue
            cards = participants[0].get("cards", [])
            for card in cards:
                card_name = card.get("name", "")
                evo = card.get("evolutionLevel", 0)
                star = card.get("starLevel", 0)
                if evo or star:
                    conn.execute(sa.text(
                        "UPDATE deck_cards SET evolution_level = :evo, "
                        "star_level = :star "
                        "WHERE battle_id = :bid AND card_name = :name "
                        "AND is_player_deck = :is_player"
                    ), {
                        "evo": evo,
                        "star": star,
                        "bid": bid,
                        "name": card_name,
                        "is_player": is_player,
                    })


def downgrade() -> None:
    # SQLite doesn't support DROP COLUMN before 3.35.0, but we target 3.35+
    op.drop_column("deck_cards", "star_level")
    op.drop_column("deck_cards", "evolution_level")
    op.drop_column("battles", "battle_duration")
    op.drop_column("battles", "opponent_elixir_leaked")
    op.drop_column("battles", "player_elixir_leaked")
