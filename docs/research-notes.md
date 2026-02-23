# Research Notes — Clash Royale Deck Archetype Detection

Running document for paper-worthy findings from the clash-stats analytics pipeline. Intended audience: writeup for publication or detailed technical blog.

## Flag 1: Unsupervised Sub-Archetype Detection via Card Co-occurrence Clustering

**Date:** 2026-02-23
**Method:** Jaccard-similarity greedy clustering on support card sets
**Corpus:** 5,503 top-ladder battles (7,000+ trophies), collected over ~1 week from 200 global leaderboard players

### The Problem

Every Clash Royale analytics platform (RoyaleAPI, StatsRoyale, Deckshop) classifies opponent decks by a single win-condition card: "Hog Rider deck", "Golem deck", "Mega Knight deck." This is a coarse taxonomy that conflates fundamentally different playstyles. A 2.6 Hog Cycle (avg 2.6 elixir, hyper-fast cycle, chip damage) plays nothing like a Hog ExeNado (Executioner + Tornado, spell-heavy control). Lumping them produces misleading matchup statistics.

The community recognizes sub-archetypes informally — competitive players know "2.6 Hog" is a different matchup than "Hog EQ" — but no analytics tool exposes this programmatically. Classification is done by hand-curated archetype lists maintained by content creators.

### The Method

1. For a given win condition (e.g., Hog Rider), collect all opponent decks containing that card from the battle corpus.
2. Extract the **support set**: the 7 non-win-condition cards in each deck.
3. Group decks by exact composition (same 8 cards = same group).
4. Greedily cluster groups by **Jaccard similarity** on support sets:
   - For each deck group (sorted by frequency, most common first):
     - Compute Jaccard similarity to each existing cluster's union support set
     - If best similarity >= threshold (0.55), merge into that cluster
     - Otherwise, start a new cluster
5. Filter clusters below a minimum size (configurable, typically 10-15 games).
6. For each cluster, identify **signature cards**: cards appearing in >60% of decks in the cluster.

**Hyperparameters:** Jaccard threshold = 0.55, minimum cluster size = 10, signature card threshold = 60%.

### Key Design Decisions

**Why Jaccard over cosine/embedding-based clustering:**
- Deck composition is a small discrete set (7 support cards out of ~100+ possible cards). Jaccard similarity on sets is the natural metric.
- No need for dimensionality reduction, embedding spaces, or learned representations. The signal is in set overlap, not continuous features.
- Interpretable: "these two decks share 5 of 7 support cards" is immediately meaningful to a player.

**Why greedy clustering over k-means/DBSCAN/hierarchical:**
- Number of clusters is unknown a priori and varies by win condition (Hog Rider has 7, PEKKA has 1).
- Deck groups have natural frequency ordering — the most common variant is the cluster seed.
- Greedy merge-or-create produces stable clusters that don't shift with random initialization.
- Deterministic: same input always produces same output.

**Why frequency-sorted seed order matters:**
- The most common deck variant defines the cluster center. Rare variants merge into the nearest common variant.
- This naturally produces clusters that align with community-recognized archetypes, because the "named" archetypes are the high-frequency ones.

### Results: Hog Rider (583 decks, 7 clusters)

| # | n | Var | WR | Avg Elixir | Signature Cards | Community Name |
|---|---|-----|-----|------------|-----------------|----------------|
| 1 | 152 | 4 | 60.5% | 3.2 | Furnace, Mighty Miner, Lightning, Cannon, Goblins | **Hog Furnace / Hog MM** |
| 2 | 140 | 4 | 61.4% | 2.6 | Musketeer, Fireball, Cannon, Skeletons, Ice Golem, Ice Spirit | **Classic 2.6 Hog Cycle** |
| 3 | 103 | 2 | 60.2% | 3.4 | Executioner, Tornado, Rocket, Valkyrie, Goblins, Ice Spirit | **Hog ExeNado** |
| 4 | 68 | 4 | 61.8% | 3.2 | Giant Skeleton, Earthquake, Cannon, Wizard, Goblins, Ice Spirit | **Hog EQ Giant Skeleton** |
| 5 | 29 | 1 | 69.0% | 4.1 | Giant Skeleton, Lightning, Firecracker, Minion Horde, Barbarians | **Hog Heavy Spam** |
| 6 | 26 | 3 | 57.7% | 4.3 | Elite Barbarians, Freeze, Boss Bandit, Executioner, Minion Horde | **Hog EBarbs Freeze** |
| 7 | 10 | 5 | 100.0% | 2.8 | Mighty Miner, Firecracker, Earthquake, Cannon, Electro Spirit | **??? (Unnamed)** |

**Validation against community taxonomy:**
- Clusters 1-3 map directly to archetypes that the competitive community already recognizes and names. The algorithm discovered these labels independently from co-occurrence patterns alone.
- Cluster 4 (Hog EQ + Giant Skeleton) is a known meta variant but not always separated from the broader Hog EQ category. The algorithm split it because Giant Skeleton changes the support card profile enough to drop below the 0.55 Jaccard threshold with Cluster 1.
- Cluster 5 (heavy spam with Barbarians + Minion Horde) is a recognizable "punish" variant — high-risk, high-reward overcommit style. Only 29 games but only 1 deck variant, suggesting a specific player or copied build.
- Cluster 6 (EBarbs + Freeze + Boss Bandit) is a hybrid that most community taxonomies would either miss entirely or label as "off-meta." The algorithm correctly separates it because its support set has almost no overlap with classic Hog builds.
- **Cluster 7 is the interesting one.** 10 games across 5 different deck variants, 100% win rate (obviously small sample). Signature: Mighty Miner + Firecracker + Earthquake. This looks like an emerging meta variant — a faster Hog Furnace that swaps Lightning for Earthquake and shifts to Firecracker for air. 5 variants in 10 games means 5 different players independently converged on this shell. **This cluster may represent a meta shift in progress that hasn't been named by the community yet.** Worth tracking as the corpus grows.

### Results: Mega Knight (205 decks, 5 clusters)

| # | n | Var | WR | Avg Elixir | Signature Cards | Community Name |
|---|---|-----|-----|------------|-----------------|----------------|
| 1 | 67 | 4 | 61.2% | 4.1 | Ram Rider, Lightning, Furnace, Giant Snowball, Royal Ghost, Golden Knight | **MK Ram Rider Bridge Spam** |
| 2 | 18 | 4 | 55.6% | 3.2 | Skeleton Barrel, Inferno Dragon, Miner, Bats, Zap | **MK Bait** |
| 3 | 16 | 2 | 62.5% | 3.6 | Balloon, Miner, Inferno Dragon, Knight, Arrows | **MK Loon** |
| 4 | 10 | 1 | 70.0% | 4.6 | Ram Rider, Wizard, Goblinstein, Lightning, Fisherman | **MK Ram Rider Heavy** |
| 5 | 10 | 3 | 80.0% | 3.6 | Ram Rider, Furnace, Mega Minion, Electro Spirit, Fireball | **MK Ram Rider Control** |

MK Bait (Cluster 2) at 55.6% is the tightest MK variant — Inferno Dragon + Skeleton Barrel creates split-lane pressure that's harder to answer than the straightforward Ram Rider bridge spam. Only 18 games; needs more data.

### Results: Full Decomposition Summary

| Win Condition | Clusters | Total Games | Notes |
|---|---|---|---|
| Miner | 9 | 295 | Highest cluster count — Miner is a versatile support card, not just a win condition |
| Hog Rider | 7 | 528 | See detailed analysis above |
| Balloon | 7 | 351 | Lava Loon vs Miner Loon vs standalone Loon — very different matchups |
| Mega Knight | 5 | 121 | Bridge spam dominant, bait is the outlier |
| Golem | 5 | 162 | Classic Golem NW, Golem Clone, Golem Lightning variants |
| Royal Giant | 5 | 249 | |
| Lava Hound | 5 | 331 | |
| Graveyard | 5 | 123 | |
| Skeleton King | 5 | 279 | |
| X-Bow | 3 | 218 | Relatively homogeneous — X-Bow decks are tightly optimized |
| P.E.K.K.A | 1 | 30 | Too few games to split, or genuinely homogeneous |
| Goblin Barrel | 1 | 74 | Classic bait is one archetype with minor card swaps |
| Archer Queen | 1 | 206 | Dominated by a single AQ shell |
| Monk | 1 | 10 | Insufficient data |

**Observation:** Miner having 9 clusters is itself a finding. Miner is used as a secondary win condition in so many different deck shells that the "Miner Control" archetype label is almost meaningless. The community recognizes this informally ("Miner is a support card") but analytics platforms still classify all Miner decks as one archetype.

### Reproducibility

- Algorithm implementation: `src/tracker/simulation/interaction_matrix.py::detect_sub_archetypes()`
- Corpus: 5,503 top-ladder battles from global leaderboard players, February 2026
- The algorithm is deterministic (no random initialization) — same corpus produces same clusters
- Threshold sensitivity: reducing Jaccard threshold from 0.55 to 0.45 merges Clusters 1 and 7 in the Hog analysis (Furnace variant absorbs the Firecracker/EQ variant). Increasing to 0.65 splits Cluster 2 into two sub-variants (with/without Ice Golem).

### Limitations

1. **Corpus bias:** Top-ladder only (7,000+ trophies). Sub-archetypes at lower trophy ranges may differ significantly. Mid-ladder MK decks are notoriously different from top-ladder MK.
2. **Temporal coverage:** ~1 week of data. Meta shifts (balance changes, new card releases) will change cluster composition. The algorithm should be re-run periodically.
3. **Cluster stability:** Small clusters (n < 20) are unstable. Cluster 7 (n=10) could be noise or could be an emerging variant. Need more data to confirm.
4. **Win rate context:** Win rates are from the corpus perspective (top-ladder players playing against each other), not from a specific player's perspective. Personal matchup posteriors are computed separately.
5. **Greedy order dependence:** The frequency-sorted seed order means the algorithm is deterministic but not optimal in any formal sense. A different seed order could produce different (equally valid) clusterings.

### Future Work

- **Temporal clustering:** Run the algorithm on rolling 2-week windows to detect meta shifts. When does a new cluster appear? When does an existing cluster die?
- **Cross-archetype signatures:** Some support cards (Goblins, The Log) appear across all clusters. These are "meta cards," not archetype-defining. Filter them dynamically (>50% prevalence across all decks) rather than using a static exclusion list.
- **Hierarchical decomposition:** First split by win condition, then by support shell, then by spell package. Three-level taxonomy instead of two.
- **Player-level archetypes:** Some players play the same deck every game. Others switch. Cluster at the player level to identify "deck loyalty" patterns.
- **Validation framework:** Compare algorithm output to hand-labeled datasets from community sources (RoyaleAPI deck archetypes, CWA deck labels, competitive tournament reports).

---

## Flag 2: Card Interaction Matrix as Matchup Prior

**Date:** 2026-02-23

### Finding

P(win | opponent has Magic Archer) = 52.9% across 391 games is the statistically strongest negative signal in the entire card interaction matrix. This is stronger than any archetype-level matchup signal.

The card-level interaction matrix provides **finer-grained matchup information than archetype-level analysis**. A player might be 60%+ against "Hog Cycle" as an archetype, but 52.9% against any deck containing Magic Archer regardless of archetype. The problem card cuts across archetype boundaries.

### Implication for Matchup Modeling

Card-level posteriors can serve as **priors for unseen matchups**. If you encounter a new deck variant you've never faced, you can estimate win probability by:

1. Computing P(win | opponent has card_X) for each of the 8 cards
2. Combining via a simple model (e.g., logistic regression on card-level posteriors, or even naive product of odds)

This gives a non-trivial matchup estimate for any deck, even one with zero historical games in the corpus. It's a natural prior for the Beta-binomial matchup model.

### Top Threat Cards (corpus-wide, n >= 50)

| Card | Faced | Win Rate | 95% CI |
|---|---|---|---|
| Magic Archer | 391 | 52.9% | [48.0, 57.8] |
| Archers | 199 | 54.8% | [47.8, 61.5] |
| X-Bow | 234 | 56.0% | [49.6, 62.2] |
| Goblin Cage | 302 | 57.3% | [51.6, 62.7] |
| Electro Giant | 227 | 57.7% | [49.6, 62.7] |
| Giant Snowball | 473 | 57.7% | [53.2, 62.2] |
| Night Witch | 64 | 57.8% | [45.6, 69.4] |
| Prince | 190 | 57.9% | [50.9, 64.7] |

---

## Flag 3: Bayesian Matchup Estimation vs Raw Win Rate

**Date:** 2026-02-23

### The Problem with Raw Win Rates

Raw win rate (wins / total) is a maximum likelihood estimate that ignores sample size uncertainty. Reporting "50% vs PEKKA Control" when n=10 gives the same confidence as "60% vs Lava Hound" when n=357. This is misleading.

### Method

Beta-binomial model: `P(win) ~ Beta(wins + alpha, losses + beta)` with uniform prior `alpha = beta = 1`.

- **Posterior mean** = (wins + 1) / (wins + losses + 2) — slightly shrunk toward 50% vs raw win/loss ratio
- **95% credible interval** from `scipy.stats.beta.ppf([0.025, 0.975], a, b)`
- **CI width** directly quantifies uncertainty. PEKKA Control: CI width = 0.53. Lava Hound: CI width = 0.10.

### Example: PEKKA Control vs Lava Hound

| Matchup | W | L | Raw WR | Posterior Mean | 95% CI | CI Width |
|---|---|---|---|---|---|---|
| PEKKA Control | 5 | 5 | 50.0% | 50.0% | [23.4, 76.6] | 0.53 |
| Lava Hound | 212 | 145 | 59.4% | 59.3% | [54.2, 64.3] | 0.10 |

The posterior mean for PEKKA is identical to the raw rate (symmetric case), but the CI tells the real story: we know almost nothing about this matchup. The Lava Hound CI is 5x tighter — that's an actionable number.

### Application

The credible interval width is the decision criterion for "do I have enough data to draw conclusions?" Any matchup with CI width > 0.20 needs more games before strategic decisions should rely on it.

---

## Methodology Notes

### Data Collection

- **Source:** Clash Royale official API (`/players/{tag}/battlelog`) via the cr-tracker pipeline
- **Corpus:** Top 200 global leaderboard players, refreshed weekly, battle logs scraped every 6 hours
- **Filtering:** Only PvP and Path of Legend battles. Minimum 7,000 starting trophies for corpus battles (top-ladder filter).
- **Deduplication:** SHA-256 hash on (battleTime, team tags, opponent tags, crowns). Same battle from two corpus players is stored once.
- **Deck representation:** Full 8-card array with card names, elixir costs, levels, and evolution status. Deck hash = MD5 of sorted (card_name, evolution_level) tuples.

### Statistical Framework

- **Bayesian estimation:** Beta-binomial conjugate model. Uniform prior Beta(1,1) for all matchups. No hierarchical shrinkage yet (future work).
- **Confidence intervals:** 95% credible intervals from the Beta posterior. These are Bayesian credible intervals, not frequentist confidence intervals — they represent the probability that the true win rate falls in the interval, given the observed data and prior.
- **Sub-archetype detection:** Greedy agglomerative clustering on Jaccard similarity of support card sets. Threshold = 0.55. Deterministic (no random initialization).

### Software

- Python 3.11, SQLAlchemy ORM, SQLite database
- `scipy.stats.beta` for posterior computation
- No ML frameworks — all analysis is classical statistics and set-theoretic clustering
- Full source: `src/tracker/simulation/` (interaction_matrix.py, matchup_model.py, runner.py)
