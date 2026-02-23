"""Add card_variant column to deck_cards (base/evo/hero).

The CR API uses evolutionLevel for both Evolutions and Heroes, but
maxEvolutionLevel distinguishes them: maxEvolutionLevel > 1 means Hero.
This migration adds the column and backfills from raw_json.

Revision ID: 005
Revises: 004
Create Date: 2026-02-23
"""
import json
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_exists(table: str, column: str) -> bool:
    conn = op.get_bind()
    result = conn.execute(sa.text(f"PRAGMA table_info({table})"))
    columns = {row[1] for row in result}
    return column in columns


def _classify_variant(card: dict) -> str:
    """Determine card variant from API card data.

    - evolutionLevel > 0 and maxEvolutionLevel > 1 → hero
    - evolutionLevel > 0 and maxEvolutionLevel <= 1 → evo
    - otherwise → base
    """
    evo_level = card.get("evolutionLevel", 0)
    max_evo = card.get("maxEvolutionLevel", 0)
    if evo_level and evo_level > 0:
        if max_evo and max_evo > 1:
            return "hero"
        return "evo"
    return "base"


def upgrade() -> None:
    if not _column_exists("deck_cards", "card_variant"):
        op.add_column(
            "deck_cards",
            sa.Column("card_variant", sa.String, server_default="base"),
        )

    # Backfill from raw_json
    conn = op.get_bind()
    battle_rows = conn.execute(sa.text(
        "SELECT battle_id, raw_json FROM battles WHERE raw_json IS NOT NULL"
    ))

    batch = []
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
            for card in participants[0].get("cards", []):
                variant = _classify_variant(card)
                if variant != "base":
                    batch.append({
                        "variant": variant,
                        "bid": bid,
                        "name": card.get("name", ""),
                        "is_player": is_player,
                    })

        # Flush in batches of 500
        if len(batch) >= 500:
            conn.execute(sa.text(
                "UPDATE deck_cards SET card_variant = :variant "
                "WHERE battle_id = :bid AND card_name = :name "
                "AND is_player_deck = :is_player"
            ), batch)
            batch = []

    if batch:
        conn.execute(sa.text(
            "UPDATE deck_cards SET card_variant = :variant "
            "WHERE battle_id = :bid AND card_name = :name "
            "AND is_player_deck = :is_player"
        ), batch)


def downgrade() -> None:
    op.drop_column("deck_cards", "card_variant")
