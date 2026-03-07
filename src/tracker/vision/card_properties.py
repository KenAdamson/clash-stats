"""Card visual and gameplay properties for replay-guided label generation.

Walk speeds, approximate lifespans, bounding box sizes, and spawn behavior
used to predict unit positions from replay event data.

Arena coordinate system (from API replay events):
  X: 0 (far left) to 17500 (far right)
  Y: 0 (opponent king tower) to 31500 (player king tower)
  Bridge line: ~15750 (center Y)
  Each tile: ~500 arena units

Walk speeds are in arena units per second. Values are approximate —
good enough for generating candidate bounding boxes, not physics simulation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CardVisualProps:
    """Visual and gameplay properties for a card type."""

    card_type: str  # "troop", "spell", "building"
    walk_speed: float  # arena units per second (0 for buildings/spells)
    lifespan_sec: float  # approximate max time alive
    bbox_size: tuple[float, float]  # (width, height) in normalized screen coords
    is_ranged: bool = False
    range_tiles: float = 0.0  # attack range in tiles
    spawn_units: Optional[str] = None  # e.g. "skeleton" for Witch/Graveyard
    spawn_count: int = 0
    spawn_interval_sec: float = 0.0


# Walk speed tiers (arena units/sec, ~tiles/sec * 500):
# Very Fast: 1500 (3.0 tiles/s) - Hog Rider, Lumberjack, Elite Barbs
# Fast:      1200 (2.4 tiles/s) - Mini PEKKA, Prince, Bandit
# Medium:    1000 (2.0 tiles/s) - Knight, Valkyrie, Wizard
# Slow:       750 (1.5 tiles/s) - PEKKA, Golem, Giant
# Very Slow:  500 (1.0 tiles/s) - Lava Hound

# Bbox sizes are rough normalized screen proportions for the unit sprite
TINY = (0.03, 0.02)
SMALL = (0.05, 0.04)
MEDIUM = (0.07, 0.06)
LARGE = (0.09, 0.08)
XLARGE = (0.12, 0.10)
SPELL_SMALL = (0.08, 0.06)
SPELL_MEDIUM = (0.15, 0.12)
SPELL_LARGE = (0.20, 0.16)
BUILDING = (0.08, 0.08)

CARD_PROPERTIES: dict[str, CardVisualProps] = {
    # === Player's deck (KrylarPrime) ===
    "Witch": CardVisualProps("troop", 1000, 12.0, MEDIUM, is_ranged=True, range_tiles=5.5,
                             spawn_units="skeleton", spawn_count=3, spawn_interval_sec=7.0),
    "Executioner": CardVisualProps("troop", 1000, 15.0, MEDIUM, is_ranged=True, range_tiles=4.5),
    "P.E.K.K.A": CardVisualProps("troop", 750, 20.0, XLARGE),
    "Goblin Curse": CardVisualProps("spell", 0, 8.0, SPELL_MEDIUM),
    "Bats": CardVisualProps("troop", 1500, 6.0, TINY),  # 5 bats, fast flyers
    "Graveyard": CardVisualProps("spell", 0, 10.0, SPELL_LARGE,
                                 spawn_units="skeleton", spawn_count=15, spawn_interval_sec=0.5),
    "Miner": CardVisualProps("troop", 1000, 15.0, SMALL),  # tunnels to position
    "Arrows": CardVisualProps("spell", 0, 1.5, SPELL_LARGE),

    # === Common opponent cards ===
    "Hog Rider": CardVisualProps("troop", 1500, 10.0, LARGE),
    "Mega Knight": CardVisualProps("troop", 1000, 20.0, XLARGE),
    "Golem": CardVisualProps("troop", 500, 25.0, XLARGE),
    "Giant": CardVisualProps("troop", 750, 20.0, LARGE),
    "Royal Giant": CardVisualProps("troop", 750, 18.0, LARGE, is_ranged=True, range_tiles=5.0),
    "Lava Hound": CardVisualProps("troop", 500, 20.0, LARGE),  # flying
    "Balloon": CardVisualProps("troop", 1000, 15.0, MEDIUM),  # flying
    "Sparky": CardVisualProps("troop", 750, 20.0, LARGE, is_ranged=True, range_tiles=4.5),
    "X-Bow": CardVisualProps("building", 0, 40.0, BUILDING, is_ranged=True, range_tiles=11.5),
    "Mortar": CardVisualProps("building", 0, 30.0, BUILDING, is_ranged=True, range_tiles=11.5),

    # === Support troops ===
    "Wizard": CardVisualProps("troop", 1000, 12.0, MEDIUM, is_ranged=True, range_tiles=5.5),
    "Musketeer": CardVisualProps("troop", 1000, 14.0, MEDIUM, is_ranged=True, range_tiles=6.0),
    "Ice Wizard": CardVisualProps("troop", 1000, 12.0, MEDIUM, is_ranged=True, range_tiles=5.5),
    "Electro Wizard": CardVisualProps("troop", 1000, 12.0, MEDIUM, is_ranged=True, range_tiles=5.0),
    "Baby Dragon": CardVisualProps("troop", 1000, 12.0, MEDIUM, is_ranged=True, range_tiles=3.5),
    "Inferno Dragon": CardVisualProps("troop", 1000, 12.0, MEDIUM, is_ranged=True, range_tiles=4.0),
    "Night Witch": CardVisualProps("troop", 1000, 10.0, MEDIUM,
                                   spawn_units="bat", spawn_count=2, spawn_interval_sec=5.0),
    "Mother Witch": CardVisualProps("troop", 1000, 8.0, MEDIUM, is_ranged=True, range_tiles=5.5),
    "Archer Queen": CardVisualProps("troop", 1000, 15.0, MEDIUM, is_ranged=True, range_tiles=5.0),
    "Golden Knight": CardVisualProps("troop", 1200, 15.0, MEDIUM),
    "Skeleton King": CardVisualProps("troop", 1000, 20.0, LARGE),
    "Monk": CardVisualProps("troop", 1200, 15.0, MEDIUM),

    # === Cycle / swarm ===
    "Skeletons": CardVisualProps("troop", 1200, 4.0, TINY),
    "Skeleton Army": CardVisualProps("troop", 1200, 6.0, TINY),
    "Goblins": CardVisualProps("troop", 1500, 5.0, TINY),
    "Spear Goblins": CardVisualProps("troop", 1500, 5.0, TINY, is_ranged=True, range_tiles=5.0),
    "Goblin Gang": CardVisualProps("troop", 1500, 5.0, TINY),
    "Minions": CardVisualProps("troop", 1500, 5.0, TINY),
    "Minion Horde": CardVisualProps("troop", 1500, 5.0, TINY),
    "Fire Spirit": CardVisualProps("troop", 1500, 3.0, TINY),
    "Ice Spirit": CardVisualProps("troop", 1500, 3.0, TINY),
    "Electro Spirit": CardVisualProps("troop", 1500, 3.0, TINY),
    "Heal Spirit": CardVisualProps("troop", 1500, 3.0, TINY),

    # === Tanks / mini-tanks ===
    "Knight": CardVisualProps("troop", 1000, 12.0, MEDIUM),
    "Valkyrie": CardVisualProps("troop", 1000, 14.0, MEDIUM),
    "Dark Prince": CardVisualProps("troop", 1000, 12.0, MEDIUM),
    "Prince": CardVisualProps("troop", 1200, 12.0, LARGE),
    "Mini P.E.K.K.A": CardVisualProps("troop", 1200, 12.0, MEDIUM),
    "Lumberjack": CardVisualProps("troop", 1500, 8.0, MEDIUM),
    "Elite Barbarians": CardVisualProps("troop", 1500, 10.0, MEDIUM),
    "Bandit": CardVisualProps("troop", 1200, 10.0, MEDIUM),
    "Ram Rider": CardVisualProps("troop", 1000, 14.0, LARGE),

    # === Spells ===
    "Fireball": CardVisualProps("spell", 0, 1.5, SPELL_MEDIUM),
    "Lightning": CardVisualProps("spell", 0, 1.5, SPELL_LARGE),
    "Rocket": CardVisualProps("spell", 0, 1.5, SPELL_MEDIUM),
    "Poison": CardVisualProps("spell", 0, 8.0, SPELL_LARGE),
    "Freeze": CardVisualProps("spell", 0, 4.0, SPELL_LARGE),
    "Rage": CardVisualProps("spell", 0, 6.0, SPELL_LARGE),
    "Tornado": CardVisualProps("spell", 0, 2.0, SPELL_LARGE),
    "Zap": CardVisualProps("spell", 0, 0.5, SPELL_MEDIUM),
    "The Log": CardVisualProps("spell", 0, 2.0, (0.12, 0.04)),
    "Barbarian Barrel": CardVisualProps("spell", 0, 2.0, (0.10, 0.04)),
    "Giant Snowball": CardVisualProps("spell", 0, 1.0, SPELL_SMALL),
    "Clone": CardVisualProps("spell", 0, 0.5, SPELL_LARGE),
    "Mirror": CardVisualProps("spell", 0, 0.0, (0.0, 0.0)),  # not visible
    "Earthquake": CardVisualProps("spell", 0, 3.0, SPELL_LARGE),
    "Royal Delivery": CardVisualProps("spell", 0, 3.0, SPELL_MEDIUM),
    "Void": CardVisualProps("spell", 0, 4.0, SPELL_MEDIUM),

    # === Buildings ===
    "Tombstone": CardVisualProps("building", 0, 21.0, BUILDING,
                                 spawn_units="skeleton", spawn_count=1, spawn_interval_sec=2.9),
    "Goblin Hut": CardVisualProps("building", 0, 35.0, BUILDING,
                                  spawn_units="spear-goblin", spawn_count=1, spawn_interval_sec=4.9),
    "Furnace": CardVisualProps("building", 0, 35.0, BUILDING,
                               spawn_units="fire-spirit", spawn_count=1, spawn_interval_sec=5.5),
    "Goblin Cage": CardVisualProps("building", 0, 20.0, BUILDING),
    "Goblin Drill": CardVisualProps("building", 0, 12.0, BUILDING),
    "Cannon": CardVisualProps("building", 0, 30.0, BUILDING, is_ranged=True, range_tiles=5.5),
    "Tesla": CardVisualProps("building", 0, 35.0, BUILDING, is_ranged=True, range_tiles=5.5),
    "Inferno Tower": CardVisualProps("building", 0, 35.0, BUILDING, is_ranged=True, range_tiles=6.0),
    "Bomb Tower": CardVisualProps("building", 0, 35.0, BUILDING, is_ranged=True, range_tiles=6.0),
    "Barbarian Hut": CardVisualProps("building", 0, 60.0, BUILDING),
    "Elixir Collector": CardVisualProps("building", 0, 70.0, BUILDING),
}


def get_properties(card_name: str) -> CardVisualProps:
    """Get visual properties for a card, with reasonable defaults."""
    if card_name in CARD_PROPERTIES:
        return CARD_PROPERTIES[card_name]
    # Default: medium troop
    return CardVisualProps("troop", 1000, 12.0, MEDIUM)
