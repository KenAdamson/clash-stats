"""Per-event spell-connect value (board-truth feature for the WP model).

Closes the documented spell-target-quality blind spot: the WP model reads card
PLACEMENTS only, so it can't tell a Fireball-on-Cannon (game-winning) from a
Fireball into empty grass. This computes, for each friendly spell, whether it
CONNECTED on an opponent card (target spawned before the spell's impact, inside
the blast radius) and the elixir value of what it removed.

Causality: the value is delivered at the IMPACT tick (placement + time-of-flight),
never at the throw tick — so a causal model never sees the future. The (offline)
look-ahead is paid once here, in feature construction, not in the model. This is
the cheap resolution to "look-back/state is expensive": the state lives here.
"""
import math

TILE = 1000.0
# Spell blast radius (tiles) and tap->impact time-of-flight (seconds). First-order;
# TOF is generous for lobbed Fireball/Arrows, ~0 for instant Zap (so early Zaps
# that fire before the troop spawns are correctly treated as whiffs).
SPELL_RADIUS = {"fireball":2.5,"the-log":2.0,"giant-snowball":2.0,"arrows":4.0,"poison":3.5,
 "lightning":3.5,"rocket":2.0,"earthquake":3.5,"zap":2.5,"barbarian-barrel":2.0,
 "goblin-curse":3.0,"void":3.0}
# tap->impact time-of-flight (seconds). Rocket is the slowest spell (~1.8s) — but
# it's overwhelmingly a REACTIVE value spell (rocket a troop already sitting by the
# tower), not a predictive one, so its long TOF mostly bounds the rare pre-fire; the
# connect logic captures its value via negative lead (target present before cast).
TOF = {"zap":0.15,"giant-snowball":0.5,"the-log":0.9,"arrows":1.1,"fireball":1.0,"poison":1.0,
 "lightning":0.6,"rocket":1.8,"earthquake":0.8,"barbarian-barrel":0.6,
 # Goblin Curse / Void are lingering-area DoT spells (like poison) — a predictive
 # curse on a swarm spot is chef's kiss. TOF here is the landing window; the linger
 # duration (which would catch later-spawning troops) is under-captured, same known
 # simplification as poison.
 "goblin-curse":1.0,"void":1.0}
STALE = 3.0  # opponent card older than this (before the throw) isn't a fresh clear

# A spell only produces value on targets it can actually damage/kill — a Zap near a
# P.E.K.K.A is worth nothing. Per-spell target sets gate the connect (mirrors the
# detector). Building-damaging spells also count opponent buildings.
_BUILDINGS = {"cannon","tesla","tombstone","bomb-tower","inferno-tower","goblin-cage","mortar",
 "x-bow","furnace","goblin-hut","elixir-collector","barbarian-hut","goblin-drill"}
_SWARM = {"skeletons","skeleton-army","goblins","goblin-gang","bats","spear-goblins","minions",
 "minion-horde","fire-spirit","electro-spirit","ice-spirit","heal-spirit","princess","dart-goblin",
 "wall-breakers","goblin-barrel","firecracker"}
_MEDIUM = {"barbarians","elite-barbarians","musketeer","three-musketeers","witch","wizard","archers",
 "night-witch","guards","rascals","zappies","flying-machine","mega-minion","bomber","hunter",
 "magic-archer","mother-witch","skeleton-dragons"}
# rocket one-shots most medium/heavy value targets clustered by the tower
_ROCKET = _MEDIUM | {"mini-pekka","valkyrie","prince","dark-prince","bandit","executioner","bowler",
 "cannon-cart","electro-wizard","ice-wizard","fisherman","goblin-machine"}
_SPELL_TARGETS = {
 "zap":_SWARM, "giant-snowball":_SWARM,
 "the-log":{"skeletons","skeleton-army","goblins","goblin-gang","spear-goblins","princess",
   "dart-goblin","wall-breakers","goblin-barrel","barbarians"},
 "arrows":_SWARM|{"witch"}, "fireball":_SWARM|_MEDIUM, "poison":_SWARM|_MEDIUM,
 "lightning":_MEDIUM, "rocket":_ROCKET, "goblin-curse":_SWARM|_MEDIUM, "void":_SWARM|_MEDIUM,
 "barbarian-barrel":{"skeletons","skeleton-army","goblins","goblin-gang"}}
_BUILDING_SPELLS = {"fireball","poison","lightning","rocket","earthquake","zap","giant-snowball",
 "the-log","arrows","goblin-curse","void"}

# Opponent tower target ZONES in ARENA coords (there is no tower "event" — towers
# are always present — so tower hits are detected geometrically). Boxes, not points:
# a spell targeting a tower is TAPPED at the tower's front face (to catch tower +
# defenders), so real rocket-aim clusters sit at y~6.5-8k, well forward of the visual
# body centre (vision _TOWER_POSITIONS princess screen-x 0.212/0.788 -> arena_x
# 3710/13790, body y~3.5k). The box spans body + forward tap zone, matching the
# empirical rocket hotspots (biggest single tile: x=13k y=8k, n=4156). Deep in the
# opponent half, so defensive spells in own half never match.
# (x_min, x_max, y_min, y_max)
_OPP_TOWER_BOXES = [
    (6300, 11200, 0, 5500),      # opponent king (centre, deepest)
    (1800, 6000, 2000, 9000),    # opponent princess L (body + forward tap zone)
    (11500, 15700, 2000, 9000),  # opponent princess R
]
# Per-spell tower-chip weight (0-10 scale): how much a tower hit matters. Rocket is
# the win condition for rocket/mortar/X-Bow cycle, so it's weighted highest.
TOWER_VALUE = {"rocket":9,"lightning":6,"fireball":4,"poison":3,"earthquake":4,
 "the-log":2,"arrows":2,"zap":1,"giant-snowball":1,"barbarian-barrel":1,
 "goblin-curse":2,"void":3}


def spell_connect_values(events, ticks_per_sec=20.0):
    """Per-event friendly-spell connect value, aligned to `events` input order.

    Args:
        events: list of (side, game_tick, card_name, arena_x, arena_y), any order;
            side is "team"/"opponent". card_name is the kebab replay name.
        ticks_per_sec: replay tick rate (20 Hz).

    Returns:
        list[float] aligned to `events`: at each event, the summed elixir value of
        any friendly spell whose IMPACT falls in (prev_event_tick, this_event_tick],
        i.e. the value is delivered at the first event after the spell lands.
    """
    ev = [(gt / ticks_per_sec, side, card, x, y) for (side, gt, card, x, y) in events]
    order = list(range(len(ev)))
    order.sort(key=lambda i: ev[i][0])          # chronological
    opp = [(ev[i][0], ev[i][2], ev[i][3], ev[i][4]) for i in order if ev[i][1] == "opponent"]

    # Find connecting friendly spells -> (impact_tick_sec, value_elixir)
    connects = []
    for (t, side, card, x, y) in ev:
        if side != "team" or card not in SPELL_RADIUS:
            continue
        r = SPELL_RADIUS[card] * TILE
        tof = TOF.get(card, 1.0)
        targets = set(_SPELL_TARGETS.get(card, set()))
        if card in _BUILDING_SPELLS:
            targets |= _BUILDINGS
        best = None
        for (to, oc, ox, oy) in opp:
            if oc not in targets:               # spell can't meaningfully damage it
                continue
            lead = to - t
            if not (-STALE <= lead <= tof):     # present before impact, not ancient
                continue
            if math.hypot(ox - x, oy - y) > r:
                continue
            val = _CARD_ELIXIR.get(oc, 3)       # value of what was removed
            if best is None or val > best:
                best = val
        if best is not None:
            connects.append((t + tof, float(best)))

    return _deliver(connects, ev, order)


def spell_tower_values(events, ticks_per_sec=20.0):
    """Per-event friendly-spell TOWER-chip value, aligned to `events` input order.

    A spell that lands on an opponent tower's footprint scores a weighted value
    (see TOWER_VALUE) delivered at impact — captures rocket/mortar/X-Bow cycle,
    where chipping the tower IS the win condition and there's no unit to "connect"
    on. Separate from spell_connect_values so the model can weight tower-chip (win
    progress) independently from unit-kills (tempo).
    """
    ev = [(gt / ticks_per_sec, side, card, x, y) for (side, gt, card, x, y) in events]
    order = list(range(len(ev)))
    order.sort(key=lambda i: ev[i][0])
    hits = []
    for (t, side, card, x, y) in ev:
        if side != "team" or card not in SPELL_RADIUS:
            continue
        # The spell's tap point (arena_x/y) is where it's aimed; the boxes already
        # span body + forward tap zone, so point-in-box catches real tower aims
        # without over-counting big-radius spells.
        for (xmin, xmax, ymin, ymax) in _OPP_TOWER_BOXES:
            if xmin <= x <= xmax and ymin <= y <= ymax:
                hits.append((t + TOF.get(card, 1.0), float(TOWER_VALUE.get(card, 2))))
                break                                # one tower per spell
    return _deliver(hits, ev, order)


def _deliver(connects, ev, order):
    """Deliver each (impact_tick_sec, value) at the first event at/after impact.

    A causal model never sees a spell's value before it lands. End-game spells
    whose impact falls after the final placement are delivered at the last event
    (so a game-deciding final spell isn't silently dropped).
    """
    out = [0.0] * len(ev)
    connects.sort()
    prev_t = float("-inf")
    ci = 0
    for i in order:
        this_t = ev[i][0]
        while ci < len(connects) and connects[ci][0] <= this_t:
            if connects[ci][0] > prev_t:
                out[i] += connects[ci][1]
            ci += 1
        prev_t = this_t
    if ci < len(connects) and order:
        last_i = order[-1]
        for k in range(ci, len(connects)):
            out[last_i] += connects[k][1]
    return out


# Static elixir table for common opponent targets (kebab replay names). Avoids a DB
# hit per game; covers the cards spells are thrown at. Falls back to 3 if unknown.
_CARD_ELIXIR = {
 "cannon":3,"tesla":4,"tombstone":3,"bomb-tower":4,"inferno-tower":5,"goblin-cage":4,
 "mortar":4,"x-bow":6,"furnace":4,"goblin-hut":5,"elixir-collector":6,"barbarian-hut":7,
 "goblin-drill":4,"skeletons":1,"skeleton-army":3,"goblins":2,"goblin-gang":3,"bats":2,
 "spear-goblins":2,"minions":3,"minion-horde":5,"fire-spirit":1,"electro-spirit":1,
 "ice-spirit":1,"heal-spirit":1,"princess":3,"dart-goblin":3,"wall-breakers":2,
 "goblin-barrel":3,"firecracker":3,"barbarians":5,"elite-barbarians":6,"musketeer":4,
 "three-musketeers":9,"witch":5,"wizard":5,"archers":3,"night-witch":4,"guards":3,
 "rascals":5,"zappies":4,"flying-machine":4,"mega-minion":3,"bomber":2,"hunter":4,
 "magic-archer":4,"mother-witch":4,"skeleton-dragons":4,
}
