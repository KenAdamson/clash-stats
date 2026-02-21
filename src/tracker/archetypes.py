"""Opponent deck archetype classification."""

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


def classify_archetype(deck: list[dict]) -> str:
    """Classify a deck into an archetype based on win condition cards.

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
