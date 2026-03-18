# ADR-009: Visual Game State Recognition

**Status:** In Progress (Phase 1.5 + early Phase 2)
**Date:** 2026-03-06
**Depends on:** ADR-003 (Game State Embeddings), ADR-004 (Win Probability), ADR-007 (Training Data Pipeline)

## Context

The replay scraper (ADR-007) captures card play events, timing, and positions — but only what the API exposes. The API does not provide:

- Unit positions on the arena at each tick
- Spell placement coordinates
- Troop health states during combat
- Deployment prediction reads (placing a unit *before* the opponent commits)
- Engagement outcomes (which units traded with which)
- Spatial control (lane dominance, bridge pressure, king tower defense)

Video replays contain ALL of this information at 60fps. A 5-minute game is ~18,000 frames of complete game state. The gap between what the API tells us and what the video shows is the difference between a box score and watching the game.

This ADR describes a pipeline to extract full tactical game state from replay video, bootstrapped from zero labeled training data using an LLM-in-the-loop labeling architecture.

## Decision

### Architecture Overview

A five-phase pipeline that bootstraps a real-time object detection model from unlabeled video using Claude Vision as the initial labeler, DINOv2 as the embedding backbone, and YOLO as the deployment target.

```
Phase 1: Claude Vision Bootstrap
         Frame --> Claude --> Labeled bounding boxes + metadata

Phase 2: DINOv2 Embedding Memory
         Crop --> DINOv2 --> 768-dim vector --> FAISS index

Phase 3: Semi-supervised Expansion (RLHF-like loop)
         New crop --> DINOv2 --> KNN --> auto-label / Claude verify / Claude label

Phase 4: YOLO Distillation
         Accumulated labels --> YOLOv8/v11 training --> real-time detection

Phase 5: Tactical State Reconstruction
         YOLO detections + TCN/Transformer --> full game state per tick
```

### Phase 1: Claude Vision Bootstrap

Claude labels raw frames with bounding boxes, unit types, actions, team affiliation, and confidence scores. This is the most expensive phase per-frame but produces the highest-quality labels.

**Label schema:**

```python
@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 normalized [0,1]
    unit_type: str           # canonical card name: "pekka", "hog-rider", "executioner"
    team: str                # "friendly" | "opponent"
    action: str              # "idle", "walking", "attacking", "deploying", "ability", "dying"
    ability_detail: str      # "flying" (hero wizard), "dash" (golden knight), "" if N/A
    confidence: float        # 0.0 - 1.0
    level_marker: int | None # troop level if visible in frame
    is_evo: bool             # evo visual variant
    hp_fraction: float | None # estimated HP fraction if health bar visible
```

**Frame-level metadata:**

```python
@dataclass
class FrameLabel:
    frame_number: int
    game_time: float         # seconds, derived from in-game clock OCR
    elixir_count: int | None # player elixir bar OCR
    period: str              # "regular", "double", "triple", "overtime"
    detections: list[Detection]
    spells_active: list[SpellEffect]  # fireball impact, graveyard ring, etc.
    tower_hp: dict[str, int] # {"friendly_king": 7032, "opponent_left": 0, ...}
    replay_signals: ReplayExclusiveSignals | None  # only for replay recordings
```

**Replay-exclusive signals:**

All video captures are screen recordings of in-game replays, not live gameplay. Replays expose information hidden during live play:

```python
@dataclass
class ReplayExclusiveSignals:
    opponent_elixir: int | None       # exact count (0-10) from purple bar at top-left
    opponent_elixir_confidence: float # OCR confidence
    opponent_hand: list[str]          # 4 visible cards in opponent's hand (left to right)
    opponent_selected_card: str | None  # card being hovered/about to play (highlighted)
```

These three signals are high-value detection targets:

| Signal | Screen region | Detection method | Analytical value |
|--------|--------------|------------------|------------------|
| Opponent elixir | Top-left bar (0.02-0.18, 0.02-0.06) | OCR on purple bar segments | Ground-truth elixir tracking — validates ADR-002 economy model |
| Opponent hand | Top-center (0.20-0.80, 0.01-0.08) | Card icon classification | Exact cycle position — feeds ADR-005 opponent prediction |
| Card selection intent | Same region, highlighted card | Brightness/glow detection | Pre-deployment signal — opponent "thinking" about a play |

**Card selection intent** is particularly valuable: when a player touches a card they're preparing to play, the card visually highlights before deployment. This is a ~0.5-2s leading indicator of the next play — a signal that doesn't exist in replay event data (which only records the actual deployment). Combined with ADR-005's opponent prediction model, this provides ground truth for "intent before action" training data.

**Bootstrap protocol:**

1. Sample frames at 10fps from replay videos (3,109 frames per 5-min game)
2. Send frames to Claude Vision in batches
3. Claude returns structured JSON labels per frame
4. Store labels in SQLite alongside frame paths
5. Target: 500-1000 labeled frames across 5-10 games for Phase 2 kickoff

**Expected labeling throughput:** ~20-30 frames per Claude API call (batch of related frames with temporal context improves accuracy). Claude can identify large/distinctive units (PEKKA, Mega Knight, Sparky, X-Bow) with high confidence and smaller units (skeletons, bats, spirits) with lower confidence. Spell effects and deployment circles are identifiable from visual patterns + game knowledge.

**Known challenges:**

- Overlapping units in dense engagements reduce bbox accuracy
- Small units (skeletons, bats, spirits) may be sub-10px and hard to bound
- Spell visual effects (Fireball explosion, Lightning strike) occlude units
- Death animations can be confused with attack animations
- Evo variants have subtle visual differences from base cards

### Phase 1.5: Replay-Guided Label Generation (Implemented)

**Key insight:** We already have ground truth from the replay event data. For any game with replay events scraped, we know:

- The exact 8 cards in each player's deck (from `deck_cards` table)
- Each card's level, evo status, and elixir cost
- The exact game tick when each card was played (from `replay_events`)
- The arena coordinates (x, y) where each card was placed
- Whether a hero ability was activated

This transforms Phase 1 from open-vocabulary detection ("what's in this frame?") to **constrained verification** ("the replay says a PEKKA should be *here* — confirm and refine the bbox").

**How it works:**

For a given frame at game_time T:

1. Query all replay events for the battle up to tick T x 20
2. For each past event, estimate if the unit is still alive (confidence decays with time)
3. Estimate current position: play_position + walk_direction x walk_speed x elapsed_time
4. Map arena coordinates (0-17500 x 0-31500) to normalized screen coordinates
5. Generate predicted bounding boxes with card name, team, level, evo, action state

**Constraint power:**

| Without replay data | With replay data |
|---------------------|------------------|
| 115+ possible card types | Exactly 16 cards (8 per player) |
| Unknown card levels | Exact levels known |
| Unknown evo/hero status | Exact variant known |
| No timing information | Exact play tick for every card |
| No position prior | Arena (x, y) at deployment |
| Team identification by color | Team identification by level badge + deck membership |

**Module:** `src/tracker/vision/replay_guided_labels.py`

Generates `PredictedFrameLabel` objects containing:
- All units predicted to be alive with position estimates and confidence scores
- Both player decks for classifier vocabulary constraint
- Game period (regular/double/triple/overtime)
- YOLO-format export (`to_yolo_lines()`) for direct training data generation

**Claude's role changes:** Instead of labeling from scratch, Claude receives the frame + predicted labels and:
- Confirms or rejects each prediction ("Is this PEKKA at (0.45, 0.52)?")
- Refines bbox coordinates (the position estimate is ~20-30px off due to walk speed approximation)
- Adds spawned sub-units not in replay data (Witch skeletons, Graveyard skeletons, Tombstone skeletons)
- Identifies spell visual effects and engagement states
- **Reads replay-exclusive signals:** opponent elixir count (OCR from top-left bar), opponent's visible hand (4 card icons at top-center), and card selection intent (highlighted card the opponent is about to play)

**Video capture storage:**

The `battles` table includes a `video_path` column (migration 009) pointing to video files on `/mnt/media/clash-videos/`. Frame extraction uses VAAPI hardware acceleration on the Arc A770 (HEVC decode + MJPEG encode at 538fps, 10fps output).

**Scope:** Currently targets personal games with video captures. Expandable to any game with replay data — just needs a video file linked via `video_path`.

### Deployment: Spool-Based Docker Container

The vision pipeline runs as a Docker container alongside the existing tracker stack. Job submission uses a file-based spool directory — no external queue infrastructure needed.

```
/var/spool/vision-queue/
  pending/     <-- drop a job file here
  processing/  <-- container moves it here while working
  complete/    <-- results written here
  failed/      <-- errors land here
```

**Job file format** (JSON):
```json
{
  "battle_id": "fdfd30356b255900f20520f28ebeba18",
  "video_path": "/mnt/media/clash-videos/SK-game/SK-game.mp4",
  "fps": 10,
  "video_start_offset": 0.0,
  "phases": ["extract", "replay_labels", "dinov2", "claude_verify"]
}
```

The container polls `pending/`, processes jobs sequentially, and writes results (labels, embeddings, YOLO annotations) to `complete/`. Same pattern as the existing `transcode-worker.sh`.

### Phase 2: DINOv2 Embedding Memory

DINOv2 (ViT-B/14) produces 768-dimensional embeddings from image crops that cluster semantically without any task-specific training. A cropped PEKKA looks like other PEKKAs in embedding space. This gives us visual memory — a growing database of "what things look like."

**Pipeline:**

```
Frame --> YOLO-prior or sliding window --> candidate crops
     --> resize to 224x224
     --> DINOv2 forward pass --> 768-dim embedding
     --> FAISS index insert (with label metadata)
```

**Infrastructure:**

- **Model:** `dinov2-base` (ViT-B/14, 86M params, ~330MB)
- **Runtime:** Intel Extension for PyTorch (IPEX) or ONNX Runtime with OpenVINO EP on Arc A770
- **Vector store:** FAISS `IndexIVFFlat` with `nprobe=16`. Start with `IndexFlatIP` (brute-force cosine) until >100K vectors, then switch to IVF for speed.
- **Storage:** Embeddings + metadata in SQLite (`game_visual_embeddings` table) with FAISS index as a sidecar file. Same pattern as ADR-003's `game_embeddings` table.

**Embedding granularity:**

Two embedding levels serve different purposes:

1. **Crop embeddings** — one per detected unit per frame. Used for unit identification (KNN lookup: "this crop looks like a Hog Rider"). ~768 dims.
2. **Scene embeddings** — one per frame. Full-frame DINOv2 [CLS] token. Used for game state similarity ("this frame looks like other frames where the opponent is about to push left lane"). ~768 dims.

**Expected cluster structure:**

DINOv2 should naturally separate:
- Unit type (PEKKA cluster vs Hog cluster vs Wizard cluster)
- Team color (blue-tinted vs red-tinted units)
- Action state (walking animation vs attack animation vs death)
- Scale/perspective (units near bridge vs units near king tower)

### Phase 3: Semi-supervised Expansion (The RLHF-like Loop)

This is where the pipeline becomes self-improving. New frames are processed through the embedding pipeline, and KNN similarity to existing labeled crops determines the labeling path.

**Confidence cascade:**

```python
def label_crop(crop_embedding, faiss_index, metadata_db):
    distances, indices = faiss_index.search(crop_embedding, k=5)
    neighbors = [metadata_db[i] for i in indices[0]]

    # Consensus among k-nearest neighbors
    label_votes = Counter(n.unit_type for n in neighbors)
    top_label, top_count = label_votes.most_common(1)[0]
    consensus = top_count / len(neighbors)
    avg_distance = distances[0][:top_count].mean()

    if consensus >= 0.8 and avg_distance < tau_auto:
        # High confidence: auto-label, no human needed
        return AutoLabel(top_label, confidence=consensus)

    elif consensus >= 0.6 and avg_distance < tau_verify:
        # Medium confidence: queue for Claude verification
        # Send crop + top-3 KNN matches to Claude: "Is this a {label}?"
        return QueueForVerification(top_label, neighbors[:3])

    else:
        # Low confidence: novel unit/state, Claude labels from scratch
        return QueueForLabeling(crop, neighbors[:3])
```

**Threshold tuning:**

- `tau_auto` and `tau_verify` are tuned per unit class. PEKKA (large, distinctive) gets aggressive auto-labeling quickly. Skeletons (small, numerous, similar to other small units) stay in the verification loop longer.
- Track precision per class: if auto-labeled PEKKAs are later corrected by Claude more than 5% of the time, tighten `tau_auto` for that class.
- Thresholds converge as the vector DB grows. After ~5000 labeled crops per class, auto-labeling should handle >90% of detections for common units.

**Active learning:** Prioritize sending Claude the *hardest* cases — crops near the decision boundary between two classes, novel action states, occluded units. This maximizes information gain per Claude API call.

**Verification batch format:**

```
"I found 12 crops I'm uncertain about. For each, here's the crop and
the 3 most similar labeled examples from my database. Please confirm
or correct the label."

[crop_1.jpg] Best match: Executioner (d=0.12) -- Correct? Y/N, actual: ___
[crop_2.jpg] Best match: Witch (d=0.18) -- Correct? Y/N, actual: ___
...
```

This is efficient — Claude sees the crop AND the reference matches, making verification fast and accurate.

### Phase 4: YOLO Distillation

Once the labeled dataset reaches sufficient size, distill the knowledge into a YOLO model for real-time inference.

**Training data targets:**

| Milestone | Labeled frames | Labeled crops | Expected mAP | Capability |
|-----------|---------------|---------------|--------------|------------|
| MVP       | 1,000         | ~15,000       | 0.4-0.5      | Large units only (PEKKA, MK, Golem, Giant) |
| Useful    | 5,000         | ~75,000       | 0.6-0.7      | All troops + buildings + large spells |
| Production| 20,000        | ~300,000      | 0.8+         | Full unit vocabulary including small units |

**Model selection:**

- **YOLOv8m or YOLOv11m** — medium variant balances speed and accuracy
- **Input resolution:** 1280x1280 (native aspect close to phone screen 1284x2778, padded)
- **Classes:** ~120 unit types x team (friendly/opponent) = ~240 classes, or ~120 classes with team as a separate classifier head

**Training infrastructure:**

- Arc A770 via Intel Extension for PyTorch (IPEX) or export to OpenVINO IR for training
- Alternatively: train on CPU with Ultralytics (slower but guaranteed to work), deploy with OpenVINO on A770
- Data augmentation: brightness/contrast jitter (simulates different arena skins), small rotation, mosaic augmentation (standard YOLO)
- Validation split: hold out full games (not random frames) to test temporal generalization

**Deployment inference:**

- OpenVINO runtime on Arc A770 for real-time detection
- Target: >60 fps inference at 1280x1280 input (well within A770 capability)
- Export pipeline: PyTorch -> ONNX -> OpenVINO IR -> Arc A770

### Phase 5: Tactical State Reconstruction

YOLO gives us unit positions every frame. Combined with the temporal models from ADR-003/004/005, this enables full game state reconstruction.

**Per-frame state vector:**

```python
@dataclass
class ArenaState:
    timestamp: float                    # game time in seconds
    units: list[TrackedUnit]           # position, type, team, HP, action
    towers: dict[str, TowerState]      # HP, targeting
    elixir: tuple[float, float]        # estimated for both players
    active_spells: list[SpellZone]     # AoE regions currently active
```

**Unit tracking across frames:**

Raw YOLO detections are per-frame. Multi-object tracking (MOT) links detections across frames into continuous trajectories:

- **Tracker:** ByteTrack or BoT-SORT (both work with any detector)
- **Re-ID:** DINOv2 crop embeddings as appearance features for the tracker's re-identification module
- This gives each unit a persistent ID across its lifetime: spawn -> walk -> engage -> die

**Derived analytics (the payoff):**

| Metric | Source | Value |
|--------|--------|-------|
| Placement heatmaps | YOLO positions + team labels | Where does the opponent place Hog? Always left lane? |
| Engagement matrices | Tracked unit interactions | PEKKA vs MK: who wins the 1v1 at each HP state? |
| Reaction time | Deployment detection + opponent unit tracking | How fast does opponent respond to my push? |
| Elixir estimation | Unit spawn tracking + known costs | Opponent's approximate elixir at each moment |
| Spatial control | Unit position density by team | Who controls bridge? Who's overcommitted? |
| Predictive placements | Unit deployed before opponent commits | The "miner in front of king tower" play — quantified |
| Card cycle tracking | Unit spawn sequence | Opponent's exact hand, predicted next card |
| Opponent elixir curve | Replay elixir bar OCR | Ground-truth elixir count at every tick — validates economy models |
| Deployment intent | Card selection highlighting | ~0.5-2s advance warning of opponent's next play |
| Win probability curve | ArenaState sequence -> ADR-004 model | P(win) at every tick, WPA per card placement |

**Integration with existing models:**

- ArenaState feeds directly into ADR-003's feature extraction (replacing replay-event-derived features with ground-truth visual features)
- ADR-004 (Win Probability) gets per-tick inputs instead of per-event inputs — smooth WPA curves
- ADR-005 (Opponent Prediction) gets spatial context — not just *which* card but *where* they'll play it
- ADR-006 (Counterfactual) can simulate "what if I placed PEKKA 2 tiles to the left?"

## Implementation Plan

### Immediate (Week 1)

1. **Define label schema** — finalize the Detection/FrameLabel dataclasses, create SQLite tables
2. **Build frame sampling pipeline** — 10fps extraction with VAAPI (done), frame selection heuristics (skip static screens, focus on action)
3. **Claude Vision bootstrap** — label 50 frames from SK-game, establish JSON format, measure labeling quality
4. **Validate approach** — can Claude reliably identify units and place bounding boxes? What's the precision on large vs small units?

### Short-term (Weeks 2-4)

5. **DINOv2 embedding pipeline** — IPEX or OpenVINO runtime on A770, crop -> embed -> FAISS store
6. **Verification loop** — build the KNN confidence cascade, queue system for Claude verification batches
7. **Label 5+ games** — target 500 labeled frames, ~7,500 crop embeddings in FAISS
8. **Cluster analysis** — validate that DINOv2 embeddings cluster by unit type, visualize with UMAP (we already have the UMAP pipeline from ADR-003)

### Medium-term (Weeks 4-8)

9. **YOLO MVP training** — large units only, validate on held-out games
10. **ByteTrack integration** — multi-object tracking across frames
11. **Semi-supervised scaling** — process replay backlog through the confidence cascade
12. **ArenaState reconstruction** — first full game state timeline from video

### Long-term (Weeks 8+)

13. **Production YOLO** — full unit vocabulary, >0.8 mAP
14. **TCN/Transformer integration** — ArenaState sequences as input to ADR-003/004/005 models
15. **Real-time analysis mode** — live video feed -> YOLO -> ArenaState -> tactical overlay
16. **Automated game review** — "here are the 3 plays that lost you this game, and what you should have done"

## Hardware Requirements

All inference and training runs on the existing server:

| Component | Role | Notes |
|-----------|------|-------|
| Arc A770 (16GB VRAM) | DINOv2 inference, YOLO training + inference, VAAPI video decode/encode | Primary compute |
| 16GB RAM (32GB incoming) | FAISS index, frame loading, training batch assembly | 32GB eliminates swap pressure |
| NVMe (455GB) | Frame storage, FAISS index, model checkpoints | Extracted frames: ~500MB per game at 10fps |
| NAS (12TB) | Raw video archive, trained model versioning | Cold storage only — never train from NAS |

## Dependencies

```
# Vision + detection
torch                    # PyTorch (with IPEX for Arc A770)
intel-extension-for-pytorch  # IPEX — Arc GPU support
ultralytics              # YOLOv8/v11
transformers             # DINOv2 model loading
faiss-cpu                # Vector similarity (faiss-gpu not needed — IPEX handles it)

# Tracking
supervision              # Roboflow's CV utilities (ByteTrack integration, bbox viz)

# Existing
numpy, pandas, scikit-learn, umap-learn  # Already installed (ADR-003)
```

## Cost Analysis

**Claude API costs (Phase 1 + 3):**

- Bootstrap: ~50 frames/game x 10 games = 500 frames. At ~1 image token per frame + structured output, roughly $5-10 total.
- Verification loop: decreasing over time as auto-labeling takes over. ~$20-50 for the first 5,000 frames, near-zero after 20,000.
- Total estimated API cost to reach production YOLO: **<$100**

Compare to: hiring annotators on Scale AI or Labelbox at $0.05-0.10 per bbox x 300,000 boxes = $15,000-30,000.

The LLM-in-the-loop approach is ~100-300x cheaper.

## Risks

1. **Claude bbox accuracy** — LLM vision is not pixel-precise. Initial bboxes may be rough. Mitigation: use Claude labels as noisy supervision, DINOv2 embeddings provide the precision through clustering.
2. **Small unit detection** — skeletons, bats, spirits are <10px. May require higher-resolution input or a separate small-object detector. Mitigation: start with large units, add small-unit classes incrementally.
3. **IPEX maturity on Arc** — Intel's PyTorch extension for Arc GPUs is improving but may have rough edges for training. Mitigation: fallback to CPU training (slower) or export to OpenVINO for inference-only.
4. **Arena skin variants** — different arena backgrounds at different trophy ranges change the visual context. Mitigation: background subtraction or arena-agnostic augmentation during training.
5. **Game updates** — Supercell periodically updates unit visuals (new evolutions, balance changes with visual tweaks). Mitigation: the semi-supervised loop naturally adapts — novel visuals enter the Claude labeling path and propagate to the YOLO model on retrain.

## Success Criteria

| Metric | Target | Validation |
|--------|--------|------------|
| Large unit detection mAP | >0.85 | Hold-out games from different trophy ranges |
| All unit detection mAP | >0.70 | Hold-out games with diverse decks |
| Unit tracking MOTA | >0.60 | Manual annotation of 5 full games |
| Card cycle reconstruction accuracy | >90% | Compare to API battlelog ground truth |
| Elixir estimation error | <1.5 elixir | Compare to API elixir leak data |
| Per-frame inference latency | <16ms (60fps) | Arc A770, OpenVINO runtime |

## References

- DINOv2: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision" (2023)
- YOLOv8: Ultralytics (2023), https://docs.ultralytics.com
- ByteTrack: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box" (2022)
- BoT-SORT: Aharon et al., "BoT-SORT: Robust Associations Multi-Pedestrian Tracking" (2022)
- FAISS: Johnson et al., "Billion-scale similarity search with GPUs" (2019)
