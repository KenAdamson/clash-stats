"""Card form classification: base / evo / hero.

The CR API encodes the FIELDED form in ``evolutionLevel`` (not a separate hero
flag): 0=base, 1=evolved, 2=hero, 3=hero+evolved. Discovered 2026-06-13 from a
known hero-wizard game; the old logic keyed off ``maxEvolutionLevel`` (hero-
CAPABLE) and mislabeled evolved hero-capable cards as 'hero'. These tests lock
in the corrected mapping.
"""

import pytest

from tracker.analytics import _card_variant
from tracker.models import DeckCard


# --- _card_variant (writer side) -------------------------------------------

@pytest.mark.parametrize("evo_level, max_evo, expected", [
    (0, 0, "base"),
    (0, 3, "base"),   # hero-capable but fielded plain
    (1, 1, "evo"),    # plain evo card
    (1, 3, "evo"),    # evolved hero-capable card — the old bug labeled this 'hero'
    (2, 3, "hero"),   # fielded as hero
    (3, 3, "hero"),   # hero + evolved
])
def test_card_variant_classification(evo_level, max_evo, expected):
    card = {"evolutionLevel": evo_level, "maxEvolutionLevel": max_evo}
    assert _card_variant(card) == expected


def test_card_variant_missing_fields_is_base():
    assert _card_variant({}) == "base"
    assert _card_variant({"evolutionLevel": None}) == "base"


# --- is_evo / is_hero hybrid (read side, Python context) -------------------

@pytest.mark.parametrize("evo_level, is_evo, is_hero", [
    (0, False, False),
    (1, True, False),
    (2, False, True),
    (3, True, True),   # hero+evo is BOTH — the booleans express what a single
                       # card_variant string cannot
])
def test_deckcard_hybrid_python(evo_level, is_evo, is_hero):
    c = DeckCard(card_name="Wizard", evolution_level=evo_level)
    assert c.is_evo is is_evo
    assert c.is_hero is is_hero


# --- is_evo / is_hero hybrid (SQL expression context) ----------------------

@pytest.mark.parametrize("level, max_level, expected", [
    (8, 14, 10),   # rare Fireball (offset +2) — ground truth
    (7, 11, 12),   # epic Bowler (offset +5) — ground truth
    (10, 16, 10),  # common Cannon (offset 0)
    (6, 16, 6),    # common Knight
    (8, 8, 16),    # maxed legendary (offset +8) -> cap
    (6, 6, 16),    # maxed champion (offset +10) -> cap
])
def test_displayed_level(level, max_level, expected):
    c = DeckCard(card_name="X", card_level=level, card_max_level=max_level)
    assert c.displayed_level == expected


def test_displayed_level_none_when_unknown():
    assert DeckCard(card_name="X", card_level=None, card_max_level=14).displayed_level is None
    assert DeckCard(card_name="X", card_level=8, card_max_level=None).displayed_level is None


def test_deckcard_hybrid_sql_expression(session):
    """The hybrid must also work as a SQL predicate, not just in Python."""
    from sqlalchemy import select, func
    session.add_all([
        DeckCard(battle_id="b", card_name="Knight", evolution_level=0),
        DeckCard(battle_id="b", card_name="Witch", evolution_level=1),
        DeckCard(battle_id="b", card_name="Wizard", evolution_level=2),
    ])
    session.commit()
    n_evo = session.execute(
        select(func.count()).select_from(DeckCard).where(DeckCard.is_evo)
    ).scalar()
    n_hero = session.execute(
        select(func.count()).select_from(DeckCard).where(DeckCard.is_hero)
    ).scalar()
    assert n_evo == 1
    assert n_hero == 1
