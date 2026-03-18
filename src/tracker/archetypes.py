"""Opponent deck archetype classification.

Tiered priority: true win conditions (build-around cards) match first,
then secondary win conditions, then support cards that only define an
archetype when nothing else matches.
"""

# Tier 1: True build-around win conditions. If present, they define the deck.
_TIER1: dict[str, list[str]] = {
    "Golem Beatdown": ["Golem"],
    "Lava Hound": ["Lava Hound"],
    "Giant Beatdown": ["Giant"],
    "Royal Giant": ["Royal Giant"],
    "Electro Giant": ["Electro Giant"],
    "Goblin Giant": ["Goblin Giant"],
    "X-Bow Siege": ["X-Bow"],
    "Mortar Siege": ["Mortar"],
    "Three Musketeers": ["Three Musketeers"],
    "Sparky": ["Sparky"],
    "Graveyard Control": ["Graveyard"],
    "Egiant": ["Elixir Golem"],
}

# Tier 1.5: Multi-card signature archetypes.
# These require 2+ specific cards to match, checked before single-card tiers.
_MULTI: list[tuple[str, set[str], int]] = [
    # (name, required_cards, min_match)
    ("Boss Bandit Ram Rider", {"Ram Rider", "Boss Bandit", "Furnace", "Executioner"}, 3),
]

# Tier 2: Strong win conditions, but can co-exist with tier 1.
_TIER2: dict[str, list[str]] = {
    "Hog Cycle": ["Hog Rider"],
    "Balloon": ["Balloon"],
    "Bridge Spam": ["Ram Rider", "Battle Ram"],
    "Miner Control": ["Miner"],
    "Goblin Barrel Bait": ["Goblin Barrel"],
    "Royal Hogs": ["Royal Hogs"],
    "Elite Barbarians": ["Elite Barbarians"],
    "Wall Breakers": ["Wall Breakers"],
}

# Tier 3: Support cards that appear in many archetypes.
# Only match when no tier 1/2 win condition is present.
_TIER3: dict[str, list[str]] = {
    "P.E.K.K.A Control": ["P.E.K.K.A"],
    "Mega Knight": ["Mega Knight"],
    "Skeleton King": ["Skeleton King"],
    "Monk": ["Monk"],
    "Archer Queen": ["Archer Queen"],
}

# Flat dict for backwards compatibility (used by similarity.py, etc.)
ARCHETYPES: dict[str, list[str]] = {
    **_TIER1,
    **{name: list(cards) for name, cards, _ in _MULTI},
    **_TIER2,
    **_TIER3,
}


def classify_archetype(deck: list[dict]) -> str:
    """Classify a deck into an archetype based on win condition cards.

    Uses tiered priority: true win conditions match before support cards.
    A deck with X-Bow + Archer Queen → "X-Bow Siege", not "Archer Queen".

    Args:
        deck: List of card dicts from the API.

    Returns:
        Archetype name string, or "Unknown" if no match.
    """
    card_names = {card.get("name", "") for card in deck}

    # Tier 1: single-card build-arounds
    for archetype, win_conditions in _TIER1.items():
        if any(wc in card_names for wc in win_conditions):
            return archetype

    # Tier 1.5: multi-card signature archetypes (before single-card tier 2)
    for archetype, sig_cards, min_match in _MULTI:
        if len(card_names & sig_cards) >= min_match:
            return archetype

    # Tier 2 & 3: single-card win conditions and support
    for tier in (_TIER2, _TIER3):
        for archetype, win_conditions in tier.items():
            if any(wc in card_names for wc in win_conditions):
                return archetype
    return "Unknown"
