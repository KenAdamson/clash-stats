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

**DONE 2026-07-29** (tools/eval/deck_cardsets_extract.py + pilot_signal_eval.py).
Deck distance = 8 − |card-set overlap|; positives require distance ≥ 3;
results stratified; bootstrap CIs; retrieval candidates distance-filtered;
flagship distances verified. Corrected baseline (own-side pooling):

| metric | hash-based (old) | distance-corrected | reading |
|---|---|---|---|
| AUC_easy | 0.664 | **0.617** [.605–.628] | 38% of old positives were d≤2 near-duplicates (that stratum alone: 0.747) |
| — d6-8 stratum | — | **0.594** | truly disjoint decks: barely above chance |
| AUC_hard | 0.428 | **0.374** [.363–.386] | deck beats pilot even harder with honest positives |
| retrieval R@1 | 2.0% | **0.66%** | near-duplicates were most of the "hits" |

Flagship postscript: the one alt deck that ever scored above noise (78th
pctile) turned out to have **deck distance 0 to main** — it is the alt's
copy of the main's own deck. The two genuinely disjoint alt decks (d=8) sit
at the 45th and 25th percentile. The glimmer was the leak.

Net: the raw-space negative result is *stronger* than first reported. The
bar for C1–C5 is unchanged and now trustworthy: **AUC_hard > 0.5** with CI
excluding 0.5 on distance-≥3 positives, flagship (d=8 decks only) in the
top decile.

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

**VERDICT 2026-07-29: killed by its own criterion after the tuning round**
(c1_adversarial_projection.py; hybrid 533-dim input; pilot-level 60/40
split, Ken never trained on; two runs):

| run | adversary | AUC_easy | AUC_hard | d6-8 | R@1 |
|---|---|---|---|---|---|
| λ=0.5, no same-deck pairs in batch | LOST (adv loss fell) | 0.771 | 0.176 | **0.692** | 2.4% |
| λ=4.0 + deck-matched hard negatives | STALEMATE (flat ~0.21) | 0.725 | **0.294** | 0.663 | 1.2% |

The tuned run improved the hard test by 0.12 — by *spending* style signal
(easy AUC and the d6-8 stratum both fell). The adversary never won: deck
BCE plateaued rather than climbing, meaning deck information could not be
squeezed out of the projection without bleeding pilot information with it.
**That is the entanglement hypothesis (§3.5) demonstrated adversarially:
in this feature space, deck and pilot are not separable subspaces.** The
flagship's disjoint decks stayed at noise (35/31 pctile) in both runs;
the d=0 control stayed at ~100th, as always.

Standing result across the program so far — the d6-8 (truly disjoint)
ladder: raw 0.594 → timing 0.636 → C1-weak **0.692** (peak; achieved
*without* deck-stripping) → C1-strong 0.663. The cross-deck signal is
real, learnable, transfers to unseen players, and tops out well short of
usable — in representation space. C2 (policy-level stylometry) is the
first candidate that changes the *object* being compared rather than the
geometry, and inherits 0.692 as the number to beat.

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

**VERDICT 2026-07-29: killed by its own criterion — with two salvage
findings** (tools/eval/motor_timing_extract.py + motor_signal_eval.py,
21 timing-only features, 181,827 games, unchanged Phase-0 harness):

- Kill-test: the flagship's distance-8 decks read **45.5 / 44.0 pctile** =
  noise. Timing alone cannot link Ken to Ken across disjoint decks.
  AUC_hard 0.371 [.358–.385]: cadence is *causally* deck-shaped (elixir
  costs set the rhythm), so "no card IDs in the features" was deck-blind
  at the feature level but not statistically — the harness caught it.
- **Salvage 1 — timing beats the 512-dim embeddings on every style axis:**
  AUC_easy 0.693 vs 0.617; truly-disjoint (d6-8) stratum **0.636 vs 0.594**;
  retrieval R@1 0.77% vs 0.66%. Twenty-one hand-built timing features carry
  *more* cross-deck pilot signal than the WP latent space. Style lives
  partly in tempo → C1's projection should take timing features as input
  alongside (or instead of) raw embeddings.
- **Salvage 2 — same-deck identity verification is essentially solved:**
  the distance-0 flagship deck (alt playing main's own deck) scored
  **99.7th percentile** at sim 0.944 across a 9k trophy gap and an
  underleveled account. Timing + fixed deck ≈ a hands-fingerprint. Directly
  applicable to alt/multi-account detection and corpus dedup, where the
  deck is typically shared.

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

**VERDICT 2026-07-29: killed by its own criterion — and the shape control
is the cleanest negative of the program** (c5_distributional_eval.py,
sliced-W1 quantile sketches, K=64 × L=16, full retrieval):

| variant | AUC_easy | AUC_hard | d6-8 | vs centroid cosine |
|---|---|---|---|---|
| sw_full (location+shape) | 0.582 | 0.348 | 0.556 | worse on all three |
| **sw_shape (centroid removed)** | **0.521** | 0.373 | **0.5045** | shape ≈ chance |
| sw_timing | 0.683 | 0.237 | 0.627 | ≈ timing centroid |

`sw_shape` is the decisive row: with location subtracted, the truly-disjoint
stratum sits at **0.5045** — dead chance — and retrieval R@1 is literally
zero. Within-group distribution SHAPE (consistency, variance structure,
asymmetry of good vs bad games) carries no pilot signal at this
resolution. The centroid was already the best statistic; the information
it "destroyed" was noise. Adding shape to location (sw_full) actively
dilutes the location signal.

Program conclusion after four pre-registered falsifications (raw cosine,
timing features, adversarial projection, distributional geometry):
**representation space is exhausted.** The cross-deck ladder peaked at
0.692 (C1 weak-adversary) and no statistic over per-game vectors reaches
usability. What remains is C2 (compare DECISIONS, not representations —
the chess-proven method, dual-use as ADR-005) and C4 (generative pilot
latent), or an honorable park with the same-deck hands-fingerprint as the
program's durable product.

## 3.5 The deck–tempo entanglement hypotheses (Ken, 2026-07-29 — parked for later)

C3's failure mode reframed, in Ken's words: timing tells you something about
the person, but it is too inter-related with deck to be a separable signal.
The entanglement runs BOTH directions:

- **Selection:** people who like fast play pick fast decks that suit their
  style — the deck-choice prior is itself style expression, not a nuisance.
- **Constraint:** people who play fast decks are *forced* to play fast or
  they lose; heavier decks simply can NOT be played fast — ever.
- **Elixir economy is the central driver** of both, and **macro-scale
  strategy becomes a resonant feature of an archetype's play style** — the
  deck sets a tempo band, and the pilot resonates within it.

Testable versions, for whenever this thread resumes:

- **H-t1 (residual tempo):** within a single archetype/tempo band, per-pilot
  timing deviations from the archetype's mean rhythm are stable across
  sessions (the C3 features, re-baselined per archetype instead of globally).
  C3's d0-2 stratum at 0.87 says within-deck consistency is strong; the
  question is whether the *residual* survives an archetype switch.
- **H-t2 (tempo preference as prior):** a pilot's deck HISTORY has a stable
  tempo centroid — people who switch decks switch within their tempo band
  more often than across it. Measurable from deck-hash sequences × avg
  elixir, no replays needed.
- **H-t3 (resonance):** macro-strategy features (elixir-hold discipline,
  commit-vs-bank cycles from elixir_trace) cluster by archetype, and pilot
  identity shows as a stable *phase/offset within* the archetype cluster,
  not as a separate cluster.

These fold naturally into C1 (archetype-conditioned normalization of the
timing inputs) and C5 (trajectory phase/offset is exactly what path
signatures capture).

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
