"""PyTorch Dataset for replay event sequences.

Converts raw replay events from the database into padded tensor batches
for TCN training. Each event becomes an 18-dim feature vector (card index
is embedded to 16-dim at forward time in the model → 34 total).

Per-event features (18 hand-crafted):
  card_id (1), side (1), game_tick_norm (1), game_phase one-hot (4),
  arena_x_norm (1), arena_y_norm (1), lane one-hot (3), play_number (1),
  ability_used (1), elixir_cost (1), card_type one-hot (3), is_evo (1)
"""

import os
import sys
import logging
from collections import namedtuple
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from tracker.models import Battle, ReplayEvent, DeckCard
from tracker.ml.card_metadata import CardVocabulary, kebab_to_title
from tracker.ml.elixir_trace import per_event_elixir as _per_event_elixir
from tracker.ml.spell_value import spell_connect_values as _spell_connect_values
from tracker.ml.spell_value import spell_tower_values as _spell_tower_values

logger = logging.getLogger(__name__)

# Arena midpoints (from features.py)
ARENA_X_MID = 8750
ARENA_Y_MID = 15750

# Max arena bounds for normalization
ARENA_X_MAX = 17500
ARENA_Y_MAX = 31500

# Game tick phase boundaries
PHASE_REGULAR_END = 3360
PHASE_DOUBLE_END = 5280
PHASE_OT_END = 7920

# Cap play_number to avoid outlier influence
PLAY_NUMBER_CAP = 20

# Max game tick for normalization (OT double elixir max)
GAME_TICK_MAX = 10000

# Minimum events per game to include
MIN_EVENTS = 4

# Per-event feature vector width. BASE is the original placement/economy
# vector; EXTRA (opt-in) appends 3 exact-elixir dims (own/opp/differential)
# + 2 game-level opponent-skill dims (trophy_gap, opp efficiency)
# + 1 spell-connect-value dim (elixir removed by a friendly spell)
# + 1 spell-tower-value dim (chip on an opponent tower — rocket/mortar/X-Bow
#   cycle win condition), both delivered causally at the spell's impact tick.
#   See spell_value.py — closes the spell-target-quality blind spot.
BASE_FEATURE_DIM = 17
EXTRA_FEATURE_DIM = 7
CORPUS_ENRICHMENT_PATH = "data/corpus_enrichment.pkl"

# Deck prior (v10). Eight cards per side, own first then opponent, as vocabulary
# indices plus a base/evo/hero variant code. Both decks are known from the API
# before a game is scored, so this costs nothing at inference time.
DECK_SIZE = 8
VARIANT_IDX = {"base": 0, "evo": 1, "hero": 2}

# Card type to one-hot index
CARD_TYPE_IDX = {"troop": 0, "spell": 1, "building": 2}

# Lightweight replay-event record. Loading events as these interned namedtuples
# instead of heavy SQLAlchemy ORM objects cuts per-event memory ~2.5x, letting the
# in-memory dataset scale to the full PvP+ranked replay pool (~1.7M games).
_RE = namedtuple("_RE", ["battle_id", "side", "game_tick", "card_name",
                         "arena_x", "arena_y", "play_number", "ability_used"])


def _game_phase_onehot(tick: int) -> list[float]:
    """One-hot encode game phase from tick value."""
    if tick < PHASE_REGULAR_END:
        return [1.0, 0.0, 0.0, 0.0]
    elif tick < PHASE_DOUBLE_END:
        return [0.0, 1.0, 0.0, 0.0]
    elif tick < PHASE_OT_END:
        return [0.0, 0.0, 1.0, 0.0]
    else:
        return [0.0, 0.0, 0.0, 1.0]


def _lane_onehot(arena_x: int) -> list[float]:
    """One-hot encode lane from arena_x position."""
    margin = 2000
    if arena_x < ARENA_X_MID - margin:
        return [1.0, 0.0, 0.0]  # left
    elif arena_x > ARENA_X_MID + margin:
        return [0.0, 1.0, 0.0]  # right
    else:
        return [0.0, 0.0, 1.0]  # center


def _card_type_onehot(card_type: str) -> list[float]:
    """One-hot encode card type."""
    vec = [0.0, 0.0, 0.0]
    idx = CARD_TYPE_IDX.get(card_type, 0)
    vec[idx] = 1.0
    return vec


def select_training_battles(session: Session, max_games: int | None = None) -> list:
    """Select the full WP training battle set: (battle_id, result) rows.

    PvP + pathOfLegend win/loss games with >= MIN_EVENTS replay events, ordered
    by battle_time ascending (the time-ordered train/val split depends on this).
    Shared by SequenceDataset's default path and the shard builder so the two
    can never drift. max_games takes the most-recent N (mirrors WP_MAX_GAMES).
    """
    if max_games is None:
        max_games = int(os.environ.get("WP_MAX_GAMES", "100000000"))
    return session.execute(
        text("""
            SELECT battle_id, result FROM (
                SELECT b.battle_id, b.result, b.battle_time
                FROM battles b
                JOIN (
                    SELECT battle_id, COUNT(*) as event_count
                    FROM replay_events
                    WHERE card_name != '_invalid'
                    GROUP BY battle_id
                    HAVING COUNT(*) >= :min_events
                ) re_counts ON re_counts.battle_id = b.battle_id
                WHERE b.battle_type IN ('PvP', 'pathOfLegend')
                  AND b.result IN ('win', 'loss')
                ORDER BY b.battle_time DESC
                LIMIT :max_games
            ) sub
            ORDER BY battle_time
        """),
        {"min_events": MIN_EVENTS, "max_games": max_games},
    ).all()


class SequenceDataset(Dataset):
    """Dataset of replay event sequences for TCN training.

    Each sample is a variable-length sequence of per-event feature vectors
    plus a win/loss label.

    Args:
        session: SQLAlchemy session.
        vocab: CardVocabulary for card→index mapping.
    """

    def __init__(
        self,
        session: Session,
        vocab: CardVocabulary,
        battle_ids: list[str] | None = None,
        extra_features: bool = False,
        deck_features: bool = True,
        battle_rows: list | None = None,
        sample_sink=None,
    ):
        # battle_rows: pre-fetched (battle_id, result) rows — bypasses the battle
        #   selection query (the shard builder fetches once to preallocate, then
        #   injects here so both sides agree on N and order).
        # sample_sink: callable(battle_id, card_ids, features, label). When set,
        #   built samples are handed to the sink instead of accumulating in RAM —
        #   this is how the shard builder streams 1.7M games at constant memory.
        self.vocab = vocab
        # extra_features (opt-in, default OFF so TCN/CVAE checkpoints trained on
        # the base 17-dim vector are unaffected): append +5 board-truth dims —
        # 3 exact-elixir (own/opp/differential, analytic from placements) and 2
        # game-level opponent-skill constants (trophy_gap, opp efficiency).
        self.extra_features = extra_features
        # Deck prior is always built when decks are available — it is a game-level
        # constant, so it costs 16 int16 per game rather than anything per-tick.
        self.deck_features = deck_features
        self.feature_dim = BASE_FEATURE_DIM + (EXTRA_FEATURE_DIM if extra_features else 0)

        # Build evo set: cards that have ability_used=1 in replay_events.
        # Cached alongside the card vocabulary — the live DISTINCT walks 100M+
        # replay_events rows (~49s/call) and this constructor runs on every
        # 5-minute inference cron. The set changes only when Supercell ships an
        # evolution; a 24h-stale answer is indistinguishable in practice, and a
        # brand-new evo missing for a day merely means its ability flag is 0 in
        # features until the next refresh. SQLite (tests) always queries live.
        from tracker.ml.card_metadata import _cache_load, _cache_store
        _is_pg = session.bind is not None and session.bind.dialect.name == "postgresql"
        _cached = _cache_load() if _is_pg else None
        if _cached is not None and _cached.get("evo_cards") is not None:
            evo_cards = set(_cached["evo_cards"])
        else:
            evo_cards = set(
                session.execute(
                    text("SELECT DISTINCT card_name FROM replay_events WHERE ability_used = 1")
                ).scalars().all()
            )
            if _is_pg and _cached is not None:
                # vocab fresh but evo missing — write evo into the shared cache
                _cache_store([tuple(r) for r in _cached["rows"]], sorted(evo_cards))
        self._evo_cards = evo_cards

        # Find all PvP battles with sufficient replay events.
        # Use a pre-aggregated JOIN instead of a correlated subquery — the correlated
        # version scans 13M replay_events rows for every battle (O(n*m)), whereas
        # the JOIN aggregates once then filters (O(m) + index lookup).
        self._sample_sink = sample_sink
        if battle_rows is not None:
            pass  # injected by the shard builder — use as-is
        elif battle_ids is not None:
            # Scoped to specific battles (for incremental inference)
            battle_rows = session.execute(
                text("""
                    -- The battle filter MUST live inside the aggregate. Without
                    -- it the subquery groups all ~29M replay_events rows and the
                    -- outer WHERE then discards nearly all of it -- for a handful
                    -- of battles, every five minutes, because this is the path
                    -- incremental WP inference takes. Pushed down, it is an index
                    -- probe on idx_replay_events_battle_id instead.
                    SELECT b.battle_id, b.result
                    FROM battles b
                    JOIN (
                        SELECT battle_id, COUNT(*) as event_count
                        FROM replay_events
                        WHERE card_name != '_invalid'
                          AND battle_id IN :bids
                        GROUP BY battle_id
                        HAVING COUNT(*) >= :min_events
                    ) re_counts ON re_counts.battle_id = b.battle_id
                    WHERE b.battle_type IN ('PvP', 'pathOfLegend')
                      AND b.result IN ('win', 'loss')
                      AND b.battle_id IN :bids
                    ORDER BY b.battle_time
                """),
                {"min_events": MIN_EVENTS, "bids": tuple(battle_ids)},
            ).all()
        else:
            # Train on BOTH ladder (PvP) and ranked (pathOfLegend) replays — PoL
            # was historically excluded, discarding ~74% of the replay pool (and
            # the ranked slice where high-tier play lives). Query shared with the
            # shard builder via select_training_battles (WP_MAX_GAMES honored).
            battle_rows = select_training_battles(session)

        logger.info("Loading %d games with replay data", len(battle_rows))

        self._samples: list[tuple[np.ndarray, np.ndarray, float]] = []
        # samples: list of (card_ids, features, label)
        # battle_ids_in_order[i] is the battle_id for _samples[i] — games that
        # fail the MIN_EVENTS check at load time are skipped here too, so this
        # stays aligned with _samples. Incremental inference reads it to map
        # output embeddings back to battles without a second query.
        self.battle_ids_in_order: list[str] = []

        # Batch-load all replay events grouped by battle_id
        battle_ids = [r[0] for r in battle_rows]
        result_map = {r[0]: 1.0 if r[1] == "win" else 0.0 for r in battle_rows}

        # Opponent-skill context (only when extra_features). trophy_gap is
        # always available from battles; opponent efficiency (best/battleCount)
        # comes from the corpus enrichment cache where the opponent is known.
        self._ctx: dict[str, tuple] = {}  # battle_id -> (trophy_gap_norm, opp_eff_norm)
        if self.extra_features and battle_ids:
            import pickle
            enrich = {}
            if os.path.exists(CORPUS_ENRICHMENT_PATH):
                try:
                    with open(CORPUS_ENRICHMENT_PATH, "rb") as f:
                        enrich = pickle.load(f)
                except Exception:
                    enrich = {}
            for i in range(0, len(battle_ids), 500):
                chunk = battle_ids[i:i + 500]
                rows = session.execute(
                    text("""
                        SELECT battle_id, player_starting_trophies,
                               opponent_starting_trophies, opponent_tag
                        FROM battles WHERE battle_id IN :bids
                    """),
                    {"bids": tuple(chunk)},
                ).all()
                for bid, ptr, otr, otag in rows:
                    gap = ((otr or 0) - (ptr or 0)) / 1000.0  # ~[-2, 2]
                    v = enrich.get(otag) if otag else None
                    eff = 0.0
                    if v and v.get("bc") and v.get("best"):
                        eff = min(v["best"] / v["bc"], 6.0) / 6.0  # 0..1
                    self._ctx[bid] = (gap, eff)

        # STREAMING build: load one chunk of battles' events, build their feature
        # arrays immediately, then free the raw events before the next chunk. Each
        # chunk's events are self-contained (queried by battle_id), and chunks are
        # processed in battle_ids order, so sample order — and the time-ordered
        # train/val split — is preserved. Peak memory = built samples (compact
        # numpy) + ONE chunk of raw events, independent of total dataset size —
        # this is what lets the dataset hold the full PvP+ranked pool (1.7M games)
        # instead of OOMing on all-at-once event materialization.
        chunk_size = 500
        skipped = 0
        for i in range(0, len(battle_ids), chunk_size):
            chunk = battle_ids[i : i + chunk_size]
            # Raw column select into lightweight namedtuples (not ORM objects),
            # interning the repeated card_name/side strings.
            rows = session.execute(
                text("""
                    SELECT battle_id, side, COALESCE(game_tick,0) game_tick, card_name,
                           COALESCE(arena_x, 8750) arena_x, COALESCE(arena_y, 15750) arena_y,
                           COALESCE(play_number, 0) play_number, COALESCE(ability_used, 0) ability_used
                    FROM replay_events
                    WHERE battle_id IN :bids AND card_name != '_invalid'
                    ORDER BY battle_id, game_tick, id
                """),
                {"bids": tuple(chunk)},
            ).all()
            events_by_battle: dict[str, list] = {bid: [] for bid in chunk}
            for r in rows:
                events_by_battle[r[0]].append(_RE(
                    r[0],
                    sys.intern(r[1]) if r[1] else r[1],
                    r[2],
                    sys.intern(r[3]) if r[3] else r[3],
                    r[4], r[5], r[6], r[7],
                ))
            del rows

            decks = self._load_decks(session, chunk) if self.deck_features else {}
            _no_deck = (np.zeros((2, DECK_SIZE), dtype=np.int16),
                        np.zeros((2, DECK_SIZE), dtype=np.int8))

            for battle_id in chunk:
                evts = events_by_battle[battle_id]
                if len(evts) < MIN_EVENTS:
                    skipped += 1
                    continue
                card_ids, features = self._build_sample(battle_id, evts)
                d_ids, d_var = decks.get(battle_id, _no_deck)
                if self._sample_sink is not None:
                    self._sample_sink(battle_id, card_ids, features,
                                      result_map[battle_id], d_ids, d_var)
                else:
                    self._samples.append((card_ids, features, result_map[battle_id],
                                          d_ids, d_var))
                    self.battle_ids_in_order.append(battle_id)
            del events_by_battle

            if (i // chunk_size) % 200 == 0 and i:
                logger.info("SequenceDataset streaming build: %d/%d battles processed",
                            i, len(battle_ids))

        logger.info(
            "SequenceDataset: %d games loaded, %d skipped, avg %.1f events/game",
            len(self._samples),
            skipped,
            np.mean([s[1].shape[0] for s in self._samples]) if self._samples else 0,
        )

    def _load_decks(self, session, chunk: list) -> dict:
        """battle_id -> (ids (2,8) int16, variants (2,8) int8), own row first.

        A deck is unordered, so cards are taken in a stable name order and the
        model mean-pools them. Missing or short decks pad with 0 (the PAD index),
        which the embedding maps to a zero vector.
        """
        rows = session.execute(
            text("""
                SELECT battle_id, is_player_deck, card_name, card_variant
                FROM deck_cards WHERE battle_id IN :bids
                ORDER BY battle_id, is_player_deck DESC, card_name
            """),
            {"bids": tuple(chunk)},
        ).all()
        out: dict[str, tuple] = {}
        for bid, is_own, name, variant in rows:
            if bid not in out:
                out[bid] = (np.zeros((2, DECK_SIZE), dtype=np.int16),
                            np.zeros((2, DECK_SIZE), dtype=np.int8),
                            [0, 0])
            ids, vars_, fill = out[bid]
            r = 0 if is_own == 1 else 1
            if fill[r] >= DECK_SIZE:
                continue  # duplicate deck_cards rows exist for some battles
            ids[r, fill[r]] = self.vocab.encode(name)
            vars_[r, fill[r]] = VARIANT_IDX.get(variant or "base", 0)
            fill[r] += 1
        return {b: (i, v) for b, (i, v, _) in out.items()}

    @staticmethod
    def _is_rotated(evts: list) -> bool:
        """True when the replay's coordinate frame is 180-degrees rotated.

        RoyaleAPI renders a replay from the viewpoint of whichever player it was
        fetched for, so the arena arrives arbitrarily oriented: for ~50% of the
        corpus BOTH axes are inverted -- high y is the opponent's half and low x
        is the right lane. Side labels stay correct (verified: team plays the
        player's cards at the same rate in rotated and normal games), so the
        damage is confined to geometry.

        Left unhandled this randomises five of the 24 features -- arena_x,
        arena_y and the three-way lane one-hot -- across half the training set,
        which is why team and opponent average the SAME arena_y corpus-wide
        (15,221 vs 15,085) when the player should sit clearly higher.

        The test needs no assumption about how anyone plays: the player defends
        their own side, so their placements average further from the opponent's
        goal than the opponent's do. Confirmed against a game whose placements
        Ken described from memory.
        """
        own = [e.arena_y for e in evts if e.side == "team"]
        opp = [e.arena_y for e in evts if e.side == "opponent"]
        if len(own) < 3 or len(opp) < 3:
            return False          # too little evidence — leave it alone
        return (sum(own) / len(own)) <= (sum(opp) / len(opp))

    def _build_sample(self, battle_id: str, evts: list) -> tuple[np.ndarray, np.ndarray]:
        """Build (card_ids, features) arrays for one battle's events.

        Extracted from __init__ so the streaming loader can build each chunk's
        samples immediately and free the raw events. `evts` are _RE namedtuples
        ordered by game_tick.
        """
        card_ids = np.zeros(len(evts), dtype=np.int64)
        features = np.zeros((len(evts), self.feature_dim), dtype=np.float32)
        # Normalise the arena to a single orientation before reading any
        # geometry off it (see _is_rotated).
        rot = self._is_rotated(evts)

        for j, ev in enumerate(evts):
            title_name = kebab_to_title(ev.card_name)
            card_ids[j] = self.vocab.encode(title_name)

            # side: 1.0 for team, 0.0 for opponent
            features[j, 0] = 1.0 if ev.side == "team" else 0.0

            # game_tick normalized
            features[j, 1] = min(ev.game_tick / GAME_TICK_MAX, 1.0)

            # game_phase one-hot (4)
            features[j, 2:6] = _game_phase_onehot(ev.game_tick)

            ax = ARENA_X_MAX - ev.arena_x if rot else ev.arena_x
            ay = ARENA_Y_MAX - ev.arena_y if rot else ev.arena_y

            # arena_x normalized [-1, 1]
            features[j, 6] = (ax - ARENA_X_MID) / ARENA_X_MID

            # arena_y normalized [-1, 1]
            features[j, 7] = (ay - ARENA_Y_MID) / ARENA_Y_MID

            # lane one-hot (3)
            features[j, 8:11] = _lane_onehot(ax)

            # play_number capped and normalized
            features[j, 11] = min(ev.play_number, PLAY_NUMBER_CAP) / PLAY_NUMBER_CAP

            # ability_used
            features[j, 12] = float(ev.ability_used)

            # elixir_cost normalized (1-10 range)
            elixir = self.vocab.elixir(title_name)
            features[j, 13] = (elixir or 4) / 10.0

            # card_type one-hot (3)
            card_type = self.vocab.card_type(title_name)
            features[j, 14:17] = _card_type_onehot(card_type)

        # Extra board-truth dims (opt-in): exact-elixir per event + the
        # game-level opponent-skill constants broadcast to every tick.
        if self.extra_features:
            ev_tuples = [
                (e.side, e.game_tick, self.vocab.elixir(kebab_to_title(e.card_name)) or 4)
                for e in evts
            ]
            elx = _per_event_elixir(ev_tuples)
            gap, eff = self._ctx.get(battle_id, (0.0, 0.0))
            # spell-connect value per event (elixir removed by a friendly spell,
            # delivered at the spell's impact tick — causal, see spell_value.py)
            sev = [(e.side, e.game_tick, e.card_name, e.arena_x, e.arena_y) for e in evts]
            scv = _spell_connect_values(sev)            # unit-kill value
            stv = _spell_tower_values(sev)              # tower-chip value
            for j in range(len(evts)):
                own_b, opp_now, diff = elx[j]
                features[j, 17] = own_b / 10.0          # own elixir in hand
                features[j, 18] = opp_now / 10.0        # opp elixir in hand
                features[j, 19] = diff / 10.0           # differential [-1,1]
                features[j, 20] = gap                   # trophy_gap (norm)
                features[j, 21] = eff                   # opp efficiency (norm)
                features[j, 22] = scv[j] / 10.0         # spell-connect value (unit kills)
                features[j, 23] = stv[j] / 10.0         # spell-tower value (tower chip)

        return card_ids, features

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Return (card_ids, features, label) for a single game."""
        card_ids, features, label, d_ids, d_var = self._samples[idx]
        return (
            torch.from_numpy(card_ids),
            torch.from_numpy(features),
            label,
            torch.from_numpy(d_ids.astype(np.int64)),
            torch.from_numpy(d_var.astype(np.int64)),
        )


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, float]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad sequences to max length in batch.

    Returns:
        card_ids: (batch, max_len) int64
        features: (batch, max_len, 17) float32
        lengths: (batch,) int64 — original sequence lengths
        labels: (batch,) float32
    """
    card_ids_list, features_list, labels, deck_ids_list, deck_var_list = zip(*batch)
    lengths = torch.tensor([len(c) for c in card_ids_list], dtype=torch.int64)
    max_len = int(lengths.max())

    batch_size = len(batch)
    # Feature width is inferred from the data (base 17 or extra 22) so the same
    # collate serves both the legacy and extra_features datasets.
    feat_dim = features_list[0].shape[-1] if features_list else BASE_FEATURE_DIM
    padded_card_ids = torch.zeros(batch_size, max_len, dtype=torch.int64)
    padded_features = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)

    for i, (cids, feats) in enumerate(zip(card_ids_list, features_list)):
        seq_len = len(cids)
        padded_card_ids[i, :seq_len] = cids
        padded_features[i, :seq_len] = feats

    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    deck_ids = torch.stack(deck_ids_list)      # (batch, 2, 8)
    deck_vars = torch.stack(deck_var_list)     # (batch, 2, 8)

    return (padded_card_ids, padded_features, lengths, labels_tensor,
            deck_ids, deck_vars)
