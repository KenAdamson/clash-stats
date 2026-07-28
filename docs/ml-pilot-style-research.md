# Research Proposal: Surfacing the Pilot-Style Signal

**Status:** Proposed (parked — return when there's appetite for a deep dive)
**Date:** 2026-07-28
**Prior art in this repo:** pilot-signal verdict `2453874c` (tools/eval/pilot_signal_eval.py),
extraction infra (tools/eval/pilot_embed_extract.py, ~200K games embedded),
paused contrastive pilot-verification project (lost to hand-crafted features).

## 1. What we know

The raw WP-v9 latent space does **not** surface pilot identity under cosine
geometry: AUC 0.664 against random strangers (weak consistency exists) but
**0.428 against same-deck strangers** — a stranger on your exact deck looks
more like you than you do on your other deck. Retrieval R@1 ≈ 2%. The Ken
flagship (main↔alt, disjoint decks, 9k trophy gap) reads as noise.

This is not evidence that style doesn't exist. It is evidence that a
win-prediction objective spends its capacity on deck and game-state — as it
should — and that *"how you play" is conditional on "what you play."* Deck
choice itself is style-correlated (a zero-usage deck **is** an identity
statement), so the decomposition to target is:

```
identity  =  deck-choice prior  ×  P(action | deck, state)
                 (trivial)            (the faint signal)
```

Every method below is a different way of isolating the second factor —
conditioning away the deck instead of hoping it averages out.

## 2. Phase 0 — fix the benchmark first (prerequisite for all five)

The current benchmark has a flaw that **must** be fixed before any method is
scored: "different deck" means different deck *hash*, and hashes differ on
evo levels, so many positives are near-identical decks.

- Define deck distance = 8 − |card-set overlap|. Positives require distance
  ≥ 3 (genuinely different decks); report results stratified by distance.
- Keep the existing hard-negative construction (same deck, different pilot)
  and the flagship (main↔alt vs same-deck controls) — they were correct.
- Bar to clear, unchanged: **AUC_hard > 0.5** with CI excluding 0.5, and the
  flagship in the top decile. Every candidate is judged by this one harness.

## 3. Five candidates

### C1. Deck-adversarial contrastive projection ("strip the deck out")

Learn a small projection g(z) over **frozen** WP embeddings (pooled and/or
mid-layer states), trained with two opposing objectives:

- contrastive (InfoNCE/triplet): pull same-pilot-different-deck together,
  push same-deck-different-pilot apart — exactly our hard pairs;
- an auxiliary deck classifier behind a **gradient-reversal layer**
  (DANN-style): the projection is optimized so deck *cannot* be predicted
  from it, actively removing the dominant nuisance factor instead of
  trusting the contrastive loss to do it.

Why it can win where the raw space lost: the 0.664 easy-AUC says pilot
signal exists in the representation; it's simply dominated. A linear/MLP
re-metric with deck explicitly adversarially removed is the cheapest way to
re-weight toward the residual. Differs from the paused contrastive project
in both input (WP representations, not raw features) and the adversarial
term (it had none).
**Cost:** low (hours on XPU; embeddings already extracted).
**Kill criterion:** if AUC_hard < 0.55 after tuning, the linearly-accessible
residual is too thin — move down the list.

### C2. Behavioral stylometry via next-action surprisal (the policy view) — *strongest prior art*

Style *is* the policy π(action | state). Train one population-average
next-action model (card class, coarse position, timing bucket, given the
game prefix — deliberately the ADR-005 opponent-prediction architecture, so
this doubles as its Phase 1). A pilot's fingerprint is the **systematic
residual** between their observed actions and the population model:
per-card-class log-likelihood gaps, timing deltas, placement offsets — a
surprisal signature. Deck is conditioned away *by construction*: the model
predicts given the hand actually held.

This is authorship attribution (surprisal under a base language model) and
it is proven in a directly analogous domain: behavioral stylometry in chess
(Maia-line work) identifies individual players from a handful of games using
exactly this shape of method, across skill levels. CR's action space
(card × 18×32 tiles × timing) is richer than chess moves in some axes,
poorer in others; the transfer is plausible, not guaranteed.
**Cost:** high (train a sequence model on the 1.67M corpus; ~days of XPU),
but the artifact is dual-use (opponent prediction on the dashboard).
**Kill criterion:** surprisal signatures fail to separate the 302 multi-deck
pilots at 2× chance under the Phase-0 benchmark.

### C3. Micro-timing / motor signatures — *discard "what," keep only "when and where exactly"*

The keystroke-dynamics analog. Build features that exclude card identity
entirely: inter-placement intervals; reaction latency to opponent
placements (defense response time); elixir-hold behavior (dwell time at 10,
leak rhythm — elixir_trace.py already computes the trace); within-tile
micro-position habits (arena_x/y jitter around canonical spots at 20Hz
resolution); pre-play hesitation under pressure vs. calm (interval
distributions conditioned on tower-damage state). Per-pilot distributions
compared by Wasserstein distance, or a small classifier over distribution
moments.

Why it can win: zero deck leakage is guaranteed by the feature set, and
motor habits are the component of style most likely to survive a trophy gap
and an underleveled account — you can't buy or lose your hands. This is
also the component Ken names directly ("styles and timings").
**Cost:** low-medium (pure feature engineering over replay_events; CPU).
**Kill criterion:** timing features alone can't re-identify even the
flagship (same human, same hands, two accounts) — if THAT fails, motor
signal doesn't survive our 20Hz capture, and C3 is dead for this dataset.

### C4. Hierarchical generative model with an explicit pilot latent (the voice-print)

Extend the CVAE line (ADR-006 infra) to a structured latent:
`z = [deck (observed, conditioned) | z_state (per-game) | z_pilot (per-PLAYER,
shared across all their games)]`. Train generatively to reconstruct action
sequences; amortize z_pilot across a player's whole game set. The only way
the model can explain player-consistent cross-game variance that deck and
state don't explain is to store it in z_pilot — disentanglement by
architecture, not by hope. This is precisely how speaker embeddings
(d-vectors/x-vectors) emerge in voice: train on the task conditioned on
content, and identity concentrates in the dedicated latent.
**Cost:** highest (new model family, careful training, ~week of XPU).
**Why bother anyway:** if it works, z_pilot is a *directly usable* artifact —
opponent voice-prints for the Nemesis dashboard, alt detection, corpus
dedup of multi-account players — not just a metric.
**Kill criterion:** z_pilot collapses (posterior ≈ prior) or merely re-encodes
trophy/skill — check by predicting trophies from z_pilot; if R² is high and
pilot-ID is not, it found skill, not style.

### C5. Distributional / trajectory geometry (stop using centroids)

The verdict used the weakest possible statistic: cosine between centroids,
which averages away within-game dynamics — arguably where style lives.
Represent pilot-on-deck as the **distribution of per-tick trajectory
embeddings** and compare distributions directly:

- sliced-Wasserstein / MMD between game-trajectory point clouds;
- time-series kernels (soft-DTW, global-alignment kernels) over per-tick
  TCN states, so *when* things happen matters;
- **path signatures** (rough-path theory): principled, deck-agnostic-izable
  features of the (elixir, tempo, position) path that capture order and
  lead-lag structure — e.g., "spends before scouting" vs "banks then
  answers" produce different signature terms even with identical marginals.

**Cost:** medium (no training; heavy pairwise computation — O(n²) in games,
embarrassingly parallel on CPU).
**Why it can win:** it's the only candidate that changes the *question* the
geometry answers rather than re-learning the space; faint signals killed by
averaging are exactly what distributional distances preserve.
**Kill criterion:** no improvement over centroid cosine on AUC_hard — then
the dynamics don't carry it either, and the signal (if any) is sub-second
mechanics (→ C3) or policy-level (→ C2).

## 4. Honorable mentions (not in the five, kept for the record)

- **Per-pilot inverse RL:** recover reward weights (tempo vs value vs tower
  risk appetite); style = what you optimize. Principled, expensive, fragile.
- **Cross-account graph features:** session timing, matchmaking adjacency —
  powerful for alt-detection but it identifies the *account holder's
  schedule*, not their play. Explicitly out of scope: different question.

## 5. Recommended order and shared harness

`C3 → C1 → C5 → C2 → C4` — cheapest falsification first: C3 needs no
training and its flagship kill-test (same human, two accounts) is the
cleanest experiment in the whole program; C1 reuses existing embeddings in
hours; C5 re-asks the question with better geometry; C2 is the strongest
bet but costs a model; C4 is the moonshot with the most useful artifact.

All five report on the Phase-0 benchmark: AUC_easy / AUC_hard (deck-distance
≥3 positives), retrieval R@1/@5, flagship percentile. One harness, five
methods, no moving goalposts.
