"""Badge-array parsing: Collection Level, per-card mastery, mode progress.

The player endpoint hides real progression data inside `badges`, an array of
`{name, level, maxLevel, progress, target, iconUrls}` entries. Two things
live there that have no usable top-level equivalent:

**Collection Level** — the in-game replacement for King Level. The documented
top-level `collectionLevel` field is a dead placeholder (reads 0 for every
account tested, main and corpus alike), and `expLevel` is now a LEGACY field:
it holds a real King Level only for accounts that predate the rework, while
post-rework accounts read 1 with expPoints/totalExpPoints at 0. Reading level
off the badge is the only path that works for both cohorts, which matters
increasingly as the player base turns over.

**Per-card mastery** — `Mastery<Card>` badges whose level counts how much a
card has been USED, not how much it has been upgraded. That is a behavioural
signal distinct from card_level, and one the corpus has never captured.

Card names here are the badge's internal CamelCase (MasteryHogRider ->
"HogRider"), which does NOT match deck_cards.card_name ("Hog Rider") or the
replay kebab-case ("hog-rider"). Callers that need to join must normalize;
`mastery_key()` gives a comparable lowercase-alphanumeric form.
"""

from typing import Any, Optional

COLLECTION_BADGE = "CollectionLevel"
MASTERY_PREFIX = "Mastery"


def collection_level(player: dict) -> tuple[Optional[int], Optional[int]]:
    """(level, progress) from the CollectionLevel badge, or (None, None).

    Prefer this over player["collectionLevel"] (always 0) and over
    player["expLevel"] (legacy; 1 for post-rework accounts).
    """
    for b in player.get("badges") or ():
        if b.get("name") == COLLECTION_BADGE:
            return b.get("level"), b.get("progress")
    return None, None


def mastery_key(name: str) -> str:
    """Comparable form for joining badge card names to other card spellings."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def card_mastery(player: dict) -> dict[str, tuple[int, int]]:
    """{card_name: (mastery_level, progress)} from Mastery* badges.

    Keys are the badge's internal name with the prefix stripped
    (MasteryHogRider -> "HogRider"). See module docstring on normalization.
    """
    out: dict[str, tuple[int, int]] = {}
    for b in player.get("badges") or ():
        n = b.get("name") or ""
        if n.startswith(MASTERY_PREFIX) and len(n) > len(MASTERY_PREFIX):
            out[n[len(MASTERY_PREFIX):]] = (b.get("level") or 0, b.get("progress") or 0)
    return out


def mode_progress(player: dict) -> dict[str, dict[str, Any]]:
    """{mode_key: {arena_id, arena_name, trophies, best_trophies}}.

    From the UNDOCUMENTED `progress` dict (per-mode arenas: ChaosDraftLeague,
    AutoChess seasons, seasonal trophy road...). Undocumented means it can
    change shape or vanish without notice, so this parses defensively and
    callers should treat every field as optional.
    """
    out: dict[str, dict[str, Any]] = {}
    for key, v in (player.get("progress") or {}).items():
        if not isinstance(v, dict):
            continue
        arena = v.get("arena") or {}
        out[key or "_default"] = {
            "arena_id": arena.get("id"),
            "arena_name": arena.get("name"),
            "trophies": v.get("trophies"),
            "best_trophies": v.get("bestTrophies"),
        }
    return out
