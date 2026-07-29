"""C1: deck-adversarial contrastive projection over frozen representations.

Learn a small projection g(x) whose geometry organizes by PILOT, with deck
information actively removed rather than hopefully averaged out:

  input x (per game) = [ WP-v9 own-side embedding (512) | 21 timing features ]
      -- the hybrid is C3's salvage: timing carried MORE cross-deck pilot
         signal than the embeddings, so both go in.
  contrastive head   = supervised contrastive over player identity, with
      within-batch positives restricted to SAME PLAYER on a DIFFERENT deck
      hash (PK sampling: P players x K games drawn across their decks).
  adversarial head   = 122-dim card multi-hot predictor behind a gradient
      REVERSAL layer: the projection is optimized so deck composition
      cannot be read from it. This does explicitly what C3's "no card ids"
      could not do implicitly -- C3 proved feature-level blindness is not
      statistical blindness.

Honesty constraints:
  - PILOT-LEVEL split: the projection is trained on 60% of multi-deck
    players; the Phase-0 harness runs ONLY on held-out players. Ken (MAIN,
    ALT) is excluded from training by construction, so the flagship stays
    a fair test.
  - The harness itself is unchanged (same bars: AUC_hard > 0.5 with CI
    excluding 0.5 on distance->=3 positives; flagship d=8 top decile).

Run with cwd=/app (XPU, takes xpu_train.lock via the caller):
  PYTHONPATH=/app/src python3 tools/eval/c1_adversarial_projection.py
Env: C1_EPOCHS (40), C1_ADV_LAMBDA (0.5), C1_OUT_DIM (64), C1_SEED (7)
"""

import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pilot_signal_eval import (
    MIN_GROUP, MIN_DECK_DIST, MAIN, ALT,
    load_cardsets, deck_dist, evaluate, flagship,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("tracker.ml.c1_adv")

EMB_DIR = Path("data/pilot_embed/wp_v9")
MOTOR_DIR = Path("data/pilot_embed/motor")
OUT = Path("data/pilot_embed/verdict_c1_adversarial.json")
CKPT = Path("data/pilot_embed/c1_projection.pt")

EPOCHS = int(os.environ.get("C1_EPOCHS", "40"))
ADV_LAMBDA = float(os.environ.get("C1_ADV_LAMBDA", "0.5"))
OUT_DIM = int(os.environ.get("C1_OUT_DIM", "64"))
SEED = int(os.environ.get("C1_SEED", "7"))
P_PLAYERS, K_GAMES = 32, 4
TEMP = 0.1


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lamb * g, None


class Projector(nn.Module):
    def __init__(self, in_dim, out_dim, n_cards):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, out_dim),
        )
        self.deck_head = nn.Sequential(
            nn.Linear(out_dim, 128), nn.ReLU(), nn.Linear(128, n_cards),
        )

    def forward(self, x, lamb):
        z = F.normalize(self.enc(x), dim=1)
        deck_logits = self.deck_head(GradReverse.apply(z, lamb))
        return z, deck_logits


def load_games():
    """Join embedding + motor shards on battle_id → per-game rows."""
    emb = {}
    for f in sorted(EMB_DIR.glob("shard_*.npz")):
        z = np.load(f, allow_pickle=False)
        for i, b in enumerate(z["battle_ids"].tolist()):
            emb[b] = i
        # store lazily per shard to save RAM: accumulate arrays instead
    # Simpler + RAM-fine (~500MB): concatenate everything once.
    bids, tags, decks, troph, e_own = [], [], [], [], []
    for f in sorted(EMB_DIR.glob("shard_*.npz")):
        z = np.load(f, allow_pickle=False)
        bids.append(z["battle_ids"]); tags.append(z["player_tags"])
        decks.append(z["deck_hashes"]); troph.append(z["trophies"])
        e_own.append(z["emb_own"])
    bids = np.concatenate(bids); tags = np.concatenate(tags)
    decks = np.concatenate(decks); troph = np.concatenate(troph)
    e_own = np.concatenate(e_own).astype(np.float32)

    m_feat = {}
    for f in sorted(MOTOR_DIR.glob("motor_*.npz")):
        z = np.load(f, allow_pickle=False)
        mf = z["features"]
        for i, b in enumerate(z["battle_ids"].tolist()):
            m_feat[b] = mf[i]
    keep = [i for i, b in enumerate(bids.tolist()) if b in m_feat]
    bids, tags, decks, troph = bids[keep], tags[keep], decks[keep], troph[keep]
    e_own = e_own[keep]
    timing = np.stack([m_feat[b] for b in bids.tolist()]).astype(np.float32)
    # z-score timing on the whole pool (train-only stats would be more purist;
    # these are per-feature scale constants, not label information)
    mu, sd = np.nanmean(timing, 0), np.nanstd(timing, 0)
    sd[sd == 0] = 1.0
    timing = np.nan_to_num((timing - mu) / sd, nan=0.0)
    x = np.concatenate([e_own, timing], axis=1)
    logger.info("joined games: %d, input dim %d", len(bids), x.shape[1])
    return bids, tags, decks, troph, x


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    device = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    cs = load_cardsets()
    cards = sorted({c for s in cs.values() for c in s})
    card_idx = {c: i for i, c in enumerate(cards)}
    logger.info("device=%s, card vocab=%d", device, len(cards))

    bids, tags, decks, troph_arr, x = load_games()

    # groups + multi-deck pilots (>=2 groups with a distance->=3 pair available)
    group_rows = defaultdict(list)
    for i, (t, d) in enumerate(zip(tags.tolist(), decks.tolist())):
        if d and d != "None":
            group_rows[(t, d)].append(i)
    by_player = defaultdict(list)
    for (t, d), rows in group_rows.items():
        if len(rows) >= MIN_GROUP:
            by_player[t].append(d)
    multi = [t for t, ds in by_player.items()
             if t not in (MAIN, ALT) and any(
                 (deck_dist(cs, a, b) or 0) >= MIN_DECK_DIST
                 for i, a in enumerate(ds) for b in ds[i + 1:])]
    rng.shuffle(multi)
    n_train = int(0.6 * len(multi))
    train_players = set(multi[:n_train])
    logger.info("multi-deck pilots: %d -> train %d / held-out %d (+ MAIN/ALT + controls)",
                len(multi), n_train, len(multi) - n_train)

    # training tensors: games of train players, grouped per (player, deck)
    tr_groups = {k: v for k, v in group_rows.items()
                 if k[0] in train_players and len(v) >= MIN_GROUP}
    tr_players = defaultdict(list)
    for (t, d) in tr_groups:
        tr_players[t].append(d)
    deck_hot = {}
    for d in {k[1] for k in tr_groups}:
        v = np.zeros(len(cards), dtype=np.float32)
        for c in cs.get(d, ()):
            v[card_idx[c]] = 1.0
        deck_hot[d] = v

    X = torch.from_numpy(x).to(device)
    model = Projector(x.shape[1], OUT_DIM, len(cards)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    players_list = [p for p in tr_players if len(tr_players[p]) >= 2]

    steps = max(1, len(players_list) // P_PLAYERS) * 8
    for ep in range(1, EPOCHS + 1):
        model.train()
        lamb = ADV_LAMBDA * min(1.0, ep / 10)          # adversary warmup
        tot_c = tot_a = 0.0
        for _ in range(steps):
            batch_idx, pl_ids, hots = [], [], []
            for pi, p in enumerate(rng.choice(players_list, P_PLAYERS, replace=False)):
                ds = tr_players[p]
                for d in rng.choice(ds, min(len(ds), 2), replace=False):
                    rows = tr_groups[(p, d)]
                    take = rng.choice(rows, min(K_GAMES // 2, len(rows)), replace=False)
                    for r in take:
                        batch_idx.append(r); pl_ids.append(pi)
                        hots.append(deck_hot[d])
            xb = X[batch_idx]
            z, dl = model(xb, lamb)
            pl = torch.tensor(pl_ids, device=device)
            hot = torch.from_numpy(np.stack(hots)).to(device)
            sim = z @ z.T / TEMP
            same_pl = pl.unsqueeze(0) == pl.unsqueeze(1)
            same_deck = (hot @ hot.T) >= 7.5           # >=8 shared cards ≈ same deck
            pos_mask = same_pl & ~same_deck & ~torch.eye(len(pl), dtype=torch.bool, device=device)
            logmask = ~torch.eye(len(pl), dtype=torch.bool, device=device)
            log_prob = sim - torch.logsumexp(sim.masked_fill(~logmask, -1e9), dim=1, keepdim=True)
            npos = pos_mask.sum(1).clamp(min=1)
            c_loss = -(log_prob.masked_fill(~pos_mask, 0).sum(1) / npos)[pos_mask.any(1)].mean()
            a_loss = F.binary_cross_entropy_with_logits(dl, hot)
            loss = c_loss + ADV_LAMBDA * a_loss        # GRL already scales by lamb
            opt.zero_grad(); loss.backward(); opt.step()
            tot_c += float(c_loss); tot_a += float(a_loss)
        if ep % 5 == 0 or ep == 1:
            logger.info("epoch %d/%d contrastive=%.4f adv_deck=%.4f lamb=%.2f",
                        ep, EPOCHS, tot_c / steps, tot_a / steps, lamb)

    torch.save({"state_dict": model.state_dict(), "in_dim": x.shape[1],
                "out_dim": OUT_DIM, "cards": cards, "seed": SEED}, CKPT)

    # ---- project EVERYTHING, evaluate on held-out players only ----
    model.eval()
    with torch.no_grad():
        Z = torch.cat([model(X[i:i + 8192], 0.0)[0].cpu()
                       for i in range(0, len(X), 8192)]).numpy()
    sig, troph = {}, {}
    for (t, d), rows in group_rows.items():
        if len(rows) < MIN_GROUP or t in train_players:
            continue
        c = Z[rows].mean(0)
        n = np.linalg.norm(c)
        if n > 0:
            sig[(t, d)] = c / n
            troph[(t, d)] = float(np.mean(troph_arr[rows]))
    logger.info("eval signatures (held-out only): %d", len(sig))

    fl = flagship("c1_adversarial", sig, troph, cs)
    print("== C1 KILL-TEST (flagship, Ken excluded from training) ==")
    for a in fl.get("alt_decks", []):
        print(f"  alt {a['alt_deck']} dist_to_main={a['deck_dist_to_main']} "
              f"sim={a['sim_to_main']} pct={a['percentile_vs_controls']} "
              f"(n={a['controls']} {a['pool']})")
    print()
    summary = evaluate("c1_adversarial", sig, troph, cs)
    print(json.dumps(summary, indent=1))
    OUT.write_text(json.dumps({"flagship": fl, "summary": summary,
                               "held_out_players": len(sig),
                               "epochs": EPOCHS, "adv_lambda": ADV_LAMBDA}, indent=1))


if __name__ == "__main__":
    main()
