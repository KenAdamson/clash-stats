"""Card vocabulary built dynamically from the database.

No static JSON file needed — card names and elixir costs are extracted
from the deck_cards table, which is populated by the scraper.
"""

import logging
from typing import Optional

from sqlalchemy import select, distinct
from sqlalchemy.orm import Session

from tracker.models import DeckCard

logger = logging.getLogger(__name__)

# Card type classification for one-hot encoding in TCN sequence features.
# Keys are Title Case (matching deck_cards.card_name / CardVocabulary).
# Cards not in this dict default to "troop".
CARD_TYPES: dict[str, str] = {
    # --- Spells ---
    "Arrows": "spell",
    "Barbarian Barrel": "spell",
    "Clone": "spell",
    "Earthquake": "spell",
    "Fireball": "spell",
    "Freeze": "spell",
    "Giant Snowball": "spell",
    "Goblin Curse": "spell",
    "Graveyard": "spell",
    "Heal Spirit": "spell",
    "Lightning": "spell",
    "Mirror": "spell",
    "Poison": "spell",
    "Rage": "spell",
    "Rocket": "spell",
    "Royal Delivery": "spell",
    "The Log": "spell",
    "Tornado": "spell",
    "Zap": "spell",
    "Void": "spell",
    # --- Buildings ---
    "Barbarian Hut": "building",
    "Bomb Tower": "building",
    "Cannon": "building",
    "Elixir Collector": "building",
    "Furnace": "building",
    "Goblin Cage": "building",
    "Goblin Drill": "building",
    "Goblin Hut": "building",
    "Inferno Tower": "building",
    "Mortar": "building",
    "Tesla": "building",
    "Tombstone": "building",
    "X-Bow": "building",
    "Goblin Machine": "building",
    # --- Everything else is "troop" (default) ---
}

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class CardVocabulary:
    """Maps card names to integer indices for feature vectors.

    Built from the actual cards observed in the database. Includes
    elixir cost lookup for feature engineering.

    Args:
        session: SQLAlchemy session to query deck_cards.
    """

    def __init__(self, session: Session):
        # Query all distinct card names, sorted for deterministic ordering
        rows = session.execute(
            select(DeckCard.card_name, DeckCard.card_elixir)
            .distinct(DeckCard.card_name)
            .order_by(DeckCard.card_name)
        ).all()

        # Build name→index mapping with special tokens at 0, 1
        self._card_to_idx: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self._elixir: dict[str, int] = {}

        for name, elixir in rows:
            if name not in self._card_to_idx:
                self._card_to_idx[name] = len(self._card_to_idx)
            if elixir is not None:
                self._elixir[name] = elixir

        self._idx_to_card = {v: k for k, v in self._card_to_idx.items()}
        logger.info("CardVocabulary: %d cards loaded", len(self._card_to_idx) - 2)

    @property
    def size(self) -> int:
        """Total vocabulary size including special tokens."""
        return len(self._card_to_idx)

    def encode(self, card_name: str) -> int:
        """Map a card name to its integer index."""
        return self._card_to_idx.get(card_name, self._card_to_idx[UNK_TOKEN])

    def decode(self, idx: int) -> str:
        """Map an integer index back to a card name."""
        return self._idx_to_card.get(idx, UNK_TOKEN)

    def elixir(self, card_name: str) -> Optional[int]:
        """Get the elixir cost for a card, or None if unknown."""
        return self._elixir.get(card_name)

    def card_names(self) -> list[str]:
        """All known card names (excluding special tokens)."""
        return [c for c in self._card_to_idx if c not in (PAD_TOKEN, UNK_TOKEN)]

    def card_type(self, card_name: str) -> str:
        """Get the card type ('troop', 'spell', or 'building'). Defaults to 'troop'."""
        return CARD_TYPES.get(card_name, "troop")


def kebab_to_title(name: str) -> str:
    """Convert kebab-case card name to Title Case.

    Replay events store 'baby-dragon', deck_cards stores 'Baby Dragon'.
    Special cases: 'pekka' → 'P.E.K.K.A', 'mini-pekka' → 'Mini P.E.K.K.A',
    'x-bow' → 'X-Bow'.
    """
    _SPECIAL = {
        "pekka": "P.E.K.K.A",
        "mini-pekka": "Mini P.E.K.K.A",
        "x-bow": "X-Bow",
        "the-log": "The Log",
    }
    if name in _SPECIAL:
        return _SPECIAL[name]
    return " ".join(word.capitalize() for word in name.split("-"))
