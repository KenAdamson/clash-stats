"""Card vocabulary built dynamically from the database.

No static JSON file needed — card names and elixir costs are extracted
from the deck_cards table, which is populated by the scraper.

The DB derivation is CACHED to a JSON file (default data/card_vocab.json,
TTL 24h). The answer — ~120 card names — changes only when Supercell ships
a card, but the loose index-scan below degraded from seconds to ~5.7 min/call
as deck_cards grew past 200M rows, and every 5-minute inference cron rebuilt
the vocabulary from scratch: 7,218 calls ≈ 28 CPU-days in three weeks, the
single largest CPU consumer on the box. The cache also makes the card→index
mapping stable within its TTL, which trained checkpoints implicitly depend on
(a new card sorting into the middle would shift every later index).
Set CARD_VOCAB_CACHE="" to disable, CARD_VOCAB_TTL to tune (seconds).
"""

import json
import logging
import os
import time
from typing import Optional

from sqlalchemy import select, distinct, text
from sqlalchemy.orm import Session

from tracker.models import DeckCard

logger = logging.getLogger(__name__)

_CACHE_PATH = os.environ.get("CARD_VOCAB_CACHE", "data/card_vocab.json")
_CACHE_TTL = float(os.environ.get("CARD_VOCAB_TTL", "86400"))


def _cache_load() -> Optional[dict]:
    """Return the cached vocab dict, or None if absent/stale/disabled."""
    if not _CACHE_PATH:
        return None
    try:
        with open(_CACHE_PATH) as fh:
            data = json.load(fh)
        if time.time() - data["built_at"] > _CACHE_TTL:
            return None
        return data
    except (OSError, ValueError, KeyError):
        return None


def _cache_store(rows: list, evo_cards: Optional[list] = None) -> None:
    """Atomically persist the derived vocab (and optionally the evo set)."""
    if not _CACHE_PATH:
        return
    prev = None
    try:
        with open(_CACHE_PATH) as fh:
            prev = json.load(fh)
    except (OSError, ValueError):
        pass
    data = {
        "built_at": time.time(),
        "rows": [[n, e] for n, e in rows],
        # preserve an evo set written by the other producer if we lack one
        "evo_cards": evo_cards if evo_cards is not None
                     else (prev or {}).get("evo_cards"),
    }
    tmp = f"{_CACHE_PATH}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as fh:
            json.dump(data, fh)
        os.replace(tmp, _CACHE_PATH)
    except OSError as e:
        logger.warning("card_vocab cache write failed: %s", e)
        try:
            os.unlink(tmp)
        except OSError:
            pass

# Loose index-scan (skip-scan) over deck_cards. The table is ~85M rows but holds
# only ~121 distinct cards; a plain DISTINCT/Index-Scan→Unique walks all 85M
# index entries with a per-row heap fetch for the elixir (~49 min observed). The
# recursive CTE hops directly from one distinct card_name to the next (~121
# index probes), pulling one elixir per card — seconds instead of minutes.
# PostgreSQL-only (SQLite's recursive CTE can't reference the recursive table in
# a correlated subquery); the test suite uses tiny SQLite data where the plain
# DISTINCT below is instant.
_LOOSE_SCAN_SQL = text("""
    WITH RECURSIVE names AS (
        (SELECT card_name FROM deck_cards ORDER BY card_name LIMIT 1)
        UNION ALL
        SELECT (SELECT dc.card_name FROM deck_cards dc
                WHERE dc.card_name > names.card_name
                ORDER BY dc.card_name LIMIT 1)
        FROM names WHERE names.card_name IS NOT NULL
    )
    SELECT n.card_name,
           (SELECT dc.card_elixir FROM deck_cards dc
            WHERE dc.card_name = n.card_name AND dc.card_elixir IS NOT NULL
            LIMIT 1) AS card_elixir
    FROM names n
    WHERE n.card_name IS NOT NULL
    ORDER BY n.card_name
""")

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
        # Query distinct (card_name, elixir), sorted for deterministic ordering.
        # On PostgreSQL, serve from the JSON cache when fresh (see module
        # docstring — the live derivation costs minutes against 200M+ rows and
        # used to run every 5-minute cron tick); on miss, run the loose
        # index-scan and write through. SQLite (tests) always queries live —
        # tiny data, and tests must not couple to a cache file.
        if session.bind is not None and session.bind.dialect.name == "postgresql":
            cached = _cache_load()
            if cached is not None:
                rows = [tuple(r) for r in cached["rows"]]
                logger.info("CardVocabulary: served from cache (%s)", _CACHE_PATH)
            else:
                rows = session.execute(_LOOSE_SCAN_SQL).all()
                _cache_store(rows)
        else:
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
