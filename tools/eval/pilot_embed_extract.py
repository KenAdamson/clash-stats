"""Extract per-game embeddings from the WP encoder for the pilot-signal eval.

Tests the hypothesis that PLAY STYLE is an emergent property of the WP
latent space — the original goal of the embedding model (ADR-003) was to
recognize a pilot across different decks and trophy ranges. This extractor
produces the raw material; the pairing/verification analysis lives in the
companion eval (deck-disjoint positives, same-deck same-band hard negatives —
card ids are model inputs, so same-deck matches are trivial and prove
nothing).

Cohort:
  - main (#L90009GPP) and alt (#VRVR9Q2QP): same pilot, disjoint decks,
    large trophy gap — the flagship test.
  - corpus players with >=2 distinct decks at >=MIN_GAMES replay games each
    (deck-disjoint positives exist for them).
  - same-deck controls: other players on those same deck hashes (the hard
    negatives).

Three embeddings per game, all from the encoder output (before the P(win)
head), each tcn_channels[-1]-dim (512 for wp_v9):
  emb_mean : masked mean over all event ticks (whole-game character)
  emb_own  : masked mean over the pilot's OWN placements only
             (features[:,0]==1) — the style probe
  emb_last : final-tick state (how the game ended)

Resumable: already-extracted battle_ids are skipped on re-run. Output shards:
  data/pilot_embed/<model>/shard_NNNN.npz
Run with cwd=/app. Takes xpu_train.lock via the wrapper/caller, not here.
Env: PILOT_EMBED_CKPT (default: production registry model),
     PILOT_MAX_PLAYERS (default 300), PILOT_MIN_GAMES (default 20),
     PILOT_MAX_GAMES_PER_PLAYER (default 200), PILOT_CONTROLS_PER_DECK (default 8)
"""

import logging
import os
from pathlib import Path

import numpy as np
import torch
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from torch.utils.data import DataLoader

from tracker.ml.card_metadata import CardVocabulary
from tracker.ml.sequence_dataset import SequenceDataset
from tracker.ml.win_probability import WinProbabilityModel
from tracker.ml.wp_dataset import wp_collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("tracker.ml.pilot_embed")

MAIN_TAG = "#L90009GPP"
ALT_TAG = "#VRVR9Q2QP"
MAX_PLAYERS = int(os.environ.get("PILOT_MAX_PLAYERS", "300"))
MIN_GAMES = int(os.environ.get("PILOT_MIN_GAMES", "20"))
MAX_PER_PLAYER = int(os.environ.get("PILOT_MAX_GAMES_PER_PLAYER", "200"))
CONTROLS_PER_DECK = int(os.environ.get("PILOT_CONTROLS_PER_DECK", "8"))
SHARD_GAMES = 2000
SIDE_FEATURE_IDX = 0  # features[:,0]: 1.0 = the polled player's placement


def select_cohort(session: Session) -> dict[str, list[str]]:
    """Pick pilots + controls; return {player_tag: [battle_id, ...]}.

    Multi-deck pilots give deck-disjoint positives; same-deck controls give
    hard negatives. Trophy-band matching happens at eval time — extract
    generously here, pair carefully there.
    """
    # Pilots with >=2 decks at >=MIN_GAMES replay games each.
    multi = session.execute(text("""
        WITH per_deck AS (
            SELECT player_tag, player_deck_hash AS dh, count(*) AS n
            FROM battles
            WHERE replay_fetched = 1 AND result IN ('win','loss')
              AND battle_type IN ('PvP','pathOfLegend')
              AND player_deck_hash IS NOT NULL
            GROUP BY 1, 2 HAVING count(*) >= :min_games
        )
        SELECT player_tag FROM per_deck
        GROUP BY player_tag HAVING count(*) >= 2
        ORDER BY player_tag LIMIT :cap
    """), {"min_games": MIN_GAMES, "cap": MAX_PLAYERS}).scalars().all()

    pilots = set(multi) | {MAIN_TAG, ALT_TAG}

    # Controls: other players on the pilots' deck hashes (hard negatives).
    controls = session.execute(text("""
        WITH pilot_decks AS (
            SELECT DISTINCT player_deck_hash AS dh
            FROM battles
            WHERE player_tag = ANY(:pilots) AND replay_fetched = 1
              AND player_deck_hash IS NOT NULL
        ), ranked AS (
            SELECT b.player_deck_hash AS dh, b.player_tag, count(*) AS n,
                   row_number() OVER (PARTITION BY b.player_deck_hash
                                      ORDER BY count(*) DESC) AS rk
            FROM battles b JOIN pilot_decks p ON p.dh = b.player_deck_hash
            WHERE b.replay_fetched = 1 AND b.result IN ('win','loss')
              AND NOT (b.player_tag = ANY(:pilots))
            GROUP BY 1, 2 HAVING count(*) >= :min_games
        )
        SELECT DISTINCT player_tag FROM ranked WHERE rk <= :per_deck
    """), {"pilots": list(pilots), "min_games": MIN_GAMES,
           "per_deck": CONTROLS_PER_DECK}).scalars().all()

    everyone = sorted(pilots | set(controls))
    logger.info("cohort: %d pilots + %d controls = %d players",
                len(pilots), len(set(controls) - pilots), len(everyone))

    out: dict[str, list[str]] = {}
    for tag in everyone:
        ids = session.execute(text("""
            SELECT battle_id FROM battles
            WHERE player_tag = :tag AND replay_fetched = 1
              AND result IN ('win','loss')
              AND battle_type IN ('PvP','pathOfLegend')
            ORDER BY battle_time DESC LIMIT :cap
        """), {"tag": tag, "cap": MAX_PER_PLAYER}).scalars().all()
        if ids:
            out[tag] = list(ids)
    return out


def load_encoder(session: Session, device: torch.device):
    """Load the checkpoint (env override or production registry) → (model, name, feat_dim)."""
    ckpt_env = os.environ.get("PILOT_EMBED_CKPT", "").strip()
    if ckpt_env:
        path, name = Path(ckpt_env), Path(ckpt_env).stem
    else:
        row = session.execute(text(
            "SELECT filename FROM model_versions WHERE model_type='wp' "
            "AND status='production' ORDER BY version DESC LIMIT 1")).first()
        path, name = Path("data/ml_models") / row[0], Path(row[0]).stem
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = WinProbabilityModel(
        vocab_size=ckpt["vocab_size"], feature_dim=ckpt.get("feature_dim", 17),
        extra_feature_dim=ckpt.get("extra_feature_dim", 0),
        tcn_channels=ckpt.get("tcn_channels"),
        card_embed_dim=ckpt.get("card_embed_dim", 16),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    logger.info("encoder %s: %d-dim output", name, model.tcn.output_channels)
    return model, name, ckpt.get("feature_dim", 17)


@torch.no_grad()
def encode_batch(model, card_ids, features, lengths, device):
    """Replicate WinProbabilityModel.forward up to the encoder output."""
    card_ids, features, lengths = card_ids.to(device), features.to(device), lengths.to(device)
    emb = model.card_embedding(card_ids)
    base = features[:, :, :model.feature_dim]
    tcn_out = model.tcn(torch.cat([emb, base], dim=2).transpose(1, 2))  # (B, C, T)
    B, C, T = tcn_out.shape
    idx = torch.arange(T, device=device).unsqueeze(0)
    valid = (idx < lengths.unsqueeze(1)).float()                       # (B, T)
    own = valid * (features[:, :, SIDE_FEATURE_IDX] > 0.5).float()
    mean = (tcn_out * valid.unsqueeze(1)).sum(2) / valid.sum(1, keepdim=True).clamp(min=1)
    own_mean = (tcn_out * own.unsqueeze(1)).sum(2) / own.sum(1, keepdim=True).clamp(min=1)
    last = tcn_out[torch.arange(B, device=device), :, (lengths - 1).clamp(min=0).long()]
    return mean.cpu().numpy(), own_mean.cpu().numpy(), last.cpu().numpy()


def main() -> int:
    engine = create_engine(os.environ["DATABASE_URL"])
    device = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")

    with Session(engine) as session:
        model, name, feat_dim = load_encoder(session, device)
        out_dir = Path("data/pilot_embed") / name
        out_dir.mkdir(parents=True, exist_ok=True)

        done: set[str] = set()
        for shard in sorted(out_dir.glob("shard_*.npz")):
            done.update(np.load(shard, allow_pickle=False)["battle_ids"].tolist())
        logger.info("resume: %d games already extracted", len(done))

        cohort = select_cohort(session)
        todo: list[tuple[str, str]] = [
            (tag, bid) for tag, ids in cohort.items() for bid in ids if bid not in done
        ]
        logger.info("to extract: %d games from %d players", len(todo), len(cohort))
        if not todo:
            return 0

        vocab = CardVocabulary(session)
        shard_no = len(list(out_dir.glob("shard_*.npz")))
        for start in range(0, len(todo), SHARD_GAMES):
            chunk = todo[start:start + SHARD_GAMES]
            ids = [bid for _, bid in chunk]
            ds = SequenceDataset(session, vocab, battle_ids=ids,
                                 extra_features=(feat_dim > 17))
            loader = DataLoader(ds, batch_size=128, collate_fn=wp_collate_fn)
            means, owns, lasts = [], [], []
            for card_ids, features, lengths, _labels, _mask in loader:
                m, o, l = encode_batch(model, card_ids, features, lengths, device)
                means.append(m); owns.append(o); lasts.append(l)
            # battle_ids_in_order[i] aligns with sample i — the dataset SKIPS
            # low-event games, so falling back to the requested `ids` would
            # silently misalign embeddings with metadata. Hard requirement.
            kept = ds.battle_ids_in_order
            n_emb = sum(m.shape[0] for m in means)
            assert len(kept) == n_emb, (
                f"alignment: {len(kept)} kept ids vs {n_emb} embeddings")

            meta = {r[0]: r[1:] for r in session.execute(text("""
                SELECT battle_id, player_tag, player_deck_hash,
                       COALESCE(player_starting_trophies, 0), battle_time::text
                FROM battles WHERE battle_id = ANY(:ids)
            """), {"ids": list(kept)}).all()}
            rows = [(b, *meta[b]) for b in kept if b in meta]

            np.savez_compressed(
                out_dir / f"shard_{shard_no:04d}.npz",
                battle_ids=np.array([r[0] for r in rows]),
                player_tags=np.array([r[1] for r in rows]),
                deck_hashes=np.array([str(r[2]) for r in rows]),
                trophies=np.array([r[3] for r in rows], dtype=np.int32),
                battle_times=np.array([r[4] for r in rows]),
                emb_mean=np.concatenate(means)[:len(rows)],
                emb_own=np.concatenate(owns)[:len(rows)],
                emb_last=np.concatenate(lasts)[:len(rows)],
            )
            logger.info("shard_%04d: %d games", shard_no, len(rows))
            shard_no += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
