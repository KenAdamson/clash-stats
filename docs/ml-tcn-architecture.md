# Temporal Convolutional Network — Architecture & Training

**Version:** 1.0
**Date:** 2026-03-01
**Implements:** ADR-003 Phase 1

## 1. Architectural Motivation

The Phase 0 UMAP pipeline operates on aggregated tabular features — 50-dim vectors that summarize an entire game into fixed statistics. This discards temporal ordering: "Pekka first, then Miner" produces the same feature vector as "Miner first, then Pekka." The TCN addresses this by operating on the raw event *sequence*, learning to extract temporal patterns that tabular aggregation cannot capture.

**Why TCN over LSTM/GRU:**
- **Parallelizable:** Convolutions process the entire sequence in parallel; RNNs are inherently sequential.
- **Stable gradients:** No vanishing/exploding gradient problem — dilated convolutions maintain consistent gradient flow.
- **Controllable receptive field:** The receptive field is exactly $\text{kernel\_size} \times \sum_{i=0}^{L-1} 2^i$ ticks, computable a priori from architecture choices.
- **Efficient at inference:** A trained TCN processes all events simultaneously, making it suitable for batch embedding of thousands of games.

**Why TCN over Transformer (for Phase 1):**
- **Lower parameter count:** ~2M vs ~5-10M for a transformer with comparable capacity.
- **Data efficiency:** Trains effectively on 500+ games; transformers need 2,000+ for the self-attention mechanism to learn meaningful patterns.
- **Causal by construction:** Dilated causal convolutions enforce that output at time $t$ depends only on inputs at times $\leq t$. Transformers require an explicit causal mask.
- The transformer is planned for Phase 2 when corpus scale supports it.

## 2. Model Architecture

**Implementation:** `src/tracker/ml/tcn.py::GameEmbeddingModel`

```
Input:
  card_ids:  (batch, seq_len)       int64
  features:  (batch, seq_len, 17)   float32
  lengths:   (batch,)               int64

┌─────────────────────────────────────────────────────────┐
│  Card Embedding                                         │
│                                                         │
│  nn.Embedding(vocab_size, 16, padding_idx=0)            │
│  card_ids → (batch, seq_len, 16)                        │
│                                                         │
│  Concatenate with features:                             │
│  (batch, seq_len, 16+17) = (batch, seq_len, 33)         │
│                                                         │
│  Transpose to channels-first:                           │
│  (batch, 33, seq_len) — required by Conv1d              │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  TCN Encoder: 6 TemporalBlocks                          │
│                                                         │
│  Block 0: Conv1d( 33→ 64, k=3, d=1 ) + residual        │
│  Block 1: Conv1d( 64→ 64, k=3, d=2 ) + residual        │
│  Block 2: Conv1d( 64→128, k=3, d=4 ) + residual        │
│  Block 3: Conv1d(128→128, k=3, d=8 ) + residual        │
│  Block 4: Conv1d(128→256, k=3, d=16) + residual        │
│  Block 5: Conv1d(256→256, k=3, d=32) + residual        │
│                                                         │
│  Output: (batch, 256, seq_len)                          │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  Masked Global Pooling                                  │
│                                                         │
│  mean_pool:    masked mean over time → (batch, 256)     │
│  max_pool:     masked max over time  → (batch, 256)     │
│  last_hidden:  hidden state at t=lengths[i]-1           │
│                                      → (batch, 256)     │
│                                                         │
│  Concatenate: (batch, 768)                              │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  Projection Head                                        │
│                                                         │
│  Linear(768, 256) → ReLU → Linear(256, 128)             │
│                                                         │
│  Output: (batch, 128) — game embeddings                 │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  Classification Head                                    │
│                                                         │
│  Linear(128, 1) — raw logit                             │
│  Sigmoid at inference → P(win)                          │
│                                                         │
│  Training loss: BCEWithLogitsLoss(logit, label)         │
└─────────────────────────────────────────────────────────┘
```

### 2.1 TemporalBlock

Each block performs two convolution-batchnorm-ReLU-dropout stages with a residual skip connection:

```
Input x: (batch, in_channels, seq_len)
  │
  ├──────────────────────────────────────────┐
  │                                          │
  ▼                                          │
  WeightNorm(Conv1d(in_ch, out_ch, k, d=d)) │
  Trim right padding                        │
  BatchNorm1d(out_ch)                       │
  ReLU                                      │ Residual:
  Dropout(p)                                │   Conv1d(in_ch, out_ch, 1)
  │                                          │   if in_ch ≠ out_ch
  ▼                                          │   else Identity
  WeightNorm(Conv1d(out_ch, out_ch, k, d=d))│
  Trim right padding                        │
  BatchNorm1d(out_ch)                       │
  ReLU                                      │
  Dropout(p)                                │
  │                                          │
  ▼                                          ▼
  ReLU( conv_output + residual(x) )

Output: (batch, out_channels, seq_len)
```

**Causal padding:** Each `Conv1d` uses left-padding of size `(kernel_size - 1) * dilation`, which is then trimmed from the right after convolution. This ensures that the output at position $t$ depends only on inputs at positions $\leq t$ — the causal constraint.

**Weight normalization:** Applied to convolutional layers via `torch.nn.utils.parametrizations.weight_norm`. Reparameterizes the weight matrix as $\mathbf{w} = g \cdot \frac{\mathbf{v}}{||\mathbf{v}||}$, decoupling magnitude ($g$) from direction ($\mathbf{v}/||\mathbf{v}||$). This stabilizes optimization dynamics relative to bare weights.

### 2.2 Receptive Field Analysis

With kernel size $k=3$ and 6 blocks with dilation factors $d = [1, 2, 4, 8, 16, 32]$, each block's receptive field expands by $(k-1) \times d$ positions. Two convolutions per block double this.

**Per-block receptive field contribution:**

| Block | Dilation | Per-conv expansion | Block expansion |
|-------|----------|-------------------|-----------------|
| 0 | 1 | $(3-1) \times 1 = 2$ | $2 \times 2 = 4$ |
| 1 | 2 | $(3-1) \times 2 = 4$ | $2 \times 4 = 8$ |
| 2 | 4 | $(3-1) \times 4 = 8$ | $2 \times 8 = 16$ |
| 3 | 8 | $(3-1) \times 8 = 16$ | $2 \times 16 = 32$ |
| 4 | 16 | $(3-1) \times 16 = 32$ | $2 \times 32 = 64$ |
| 5 | 32 | $(3-1) \times 32 = 64$ | $2 \times 64 = 128$ |

**Total receptive field:** $1 + \sum = 1 + 4 + 8 + 16 + 32 + 64 + 128 = 253$ time steps.

At an average of 25-40 events per game, this means the deepest layers have a receptive field that covers the entire game sequence with substantial margin. The model can learn dependencies between the opening play and the game-ending sequence.

### 2.3 Masked Global Pooling

Variable-length sequences are padded to the maximum batch length. The pooling layer uses a binary mask to exclude padding positions:

```python
# mask: (batch, 1, seq_len) where mask[i, 0, t] = 1 if t < lengths[i]
mask = (arange(seq_len) < lengths.unsqueeze(1)).unsqueeze(1).float()

# Masked mean: sum over valid positions / count of valid positions
mean_pool = (tcn_out * mask).sum(dim=2) / lengths.float().clamp(min=1)

# Masked max: set padding to -inf, then take max
max_pool = tcn_out.masked_fill(mask == 0, -inf).max(dim=2).values

# Last valid hidden: index by lengths[i] - 1
last_hidden = tcn_out[batch_idx, :, lengths - 1]
```

The concatenation of three pooling strategies captures complementary information:
- **Mean pool:** Overall trend across the game
- **Max pool:** Most activated feature at any point (peak signals)
- **Last hidden:** Final game state (most relevant to outcome)

### 2.4 Parameter Count

```
Card Embedding:    vocab_size × 16     ≈  2,000 × 16  =   32,000
TCN Block 0:       33 × 64 × 3 × 2 + bias + BN        ≈   13,000
TCN Block 1:       64 × 64 × 3 × 2                     ≈   24,000
TCN Block 2:       64 × 128 × 3 × 2 + residual 1x1     ≈   58,000
TCN Block 3:       128 × 128 × 3 × 2                    ≈   99,000
TCN Block 4:       128 × 256 × 3 × 2 + residual 1x1     ≈  230,000
TCN Block 5:       256 × 256 × 3 × 2                    ≈  394,000
Projection:        768 × 256 + 256 × 128                ≈  230,000
Classifier:        128 × 1 + 1                           ≈      129
                                                   Total ≈ 1.08M
```

Actual count varies with vocabulary size. With ~120 unique cards: ~1.1M parameters.

## 3. Training Pipeline

**Implementation:** `src/tracker/ml/training.py`

### 3.1 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch size | 64 | Balances gradient noise and memory |
| Learning rate | 1e-3 | Standard for AdamW on small models |
| Weight decay | 1e-4 | L2 regularization to prevent overfitting |
| Epochs (max) | 50 | Upper bound; early stopping typically triggers at 15-25 |
| Early stopping patience | 10 | Epochs without improvement before halt |
| Dropout | 0.2 | Applied within each TemporalBlock |
| Embedding dim | 128 | Output embedding dimension |
| Validation fraction | 0.2 | Last 20% of games by time |

### 3.2 Optimizer: AdamW

$$\theta_{t+1} = \theta_t - \alpha \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

AdamW applies weight decay directly to parameters (decoupled from the adaptive learning rate), which provides more consistent regularization than L2 penalty in the loss. Default betas: $\beta_1 = 0.9$, $\beta_2 = 0.999$.

### 3.3 Learning Rate Schedule: Cosine Annealing

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}} \pi\right)\right)$$

The learning rate follows a cosine curve from `LEARNING_RATE` (1e-3) down to 0 over `EPOCHS` (50) steps. This provides aggressive exploration early and fine-grained convergence late.

### 3.4 Loss Function: BCEWithLogitsLoss

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\sigma(z_i)) + (1-y_i) \log(1-\sigma(z_i)) \right]$$

where $z_i$ is the raw logit from the classification head and $y_i \in \{0, 1\}$ is the win/loss label. `BCEWithLogitsLoss` combines sigmoid and cross-entropy in a single numerically stable operation.

### 3.5 Train/Validation Split

The split is **temporal, not random**: the first 80% of games (ordered by `battle_time`) form the training set, and the last 20% form the validation set. This simulates the real-world scenario where the model must predict outcomes for games it hasn't seen, using patterns from earlier games.

**Why not random split:** Random splitting would allow the model to "see" games from the same time period during training and validation, which inflates validation accuracy. The temporal split is strictly harder and more realistic — the model must generalize to new games against potentially new archetypes.

### 3.6 Early Stopping

After each epoch, validation loss is computed. If it does not improve for `EARLY_STOPPING_PATIENCE` (10) consecutive epochs, training halts. The best model (lowest validation loss) is saved to `data/ml_models/tcn_v1.pt`.

**Checkpoint format:**
```python
{
    "model_state_dict": model.state_dict(),
    "vocab_size": model.card_embedding.num_embeddings,
    "epoch": best_epoch,
    "val_loss": best_val_loss,
    "val_acc": best_val_acc,
}
```

### 3.7 Device Detection

```python
def _detect_device() -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")     # Intel Arc A770 via oneAPI
    if torch.cuda.is_available():
        return torch.device("cuda")     # NVIDIA GPU
    return torch.device("cpu")
```

The Intel XPU path supports the host machine's Arc A770 GPU via PyTorch's oneAPI/IPEX integration. The model's ~1M parameters and 64-sample batches fit comfortably in the A770's 16GB VRAM.

## 4. Embedding Extraction

**Implementation:** `src/tracker/ml/training.py::TCNTrainer.extract_embeddings()`

After training, the model runs inference on all games in the dataset. The `forward()` method returns both embeddings and logits; only the 128-dim embeddings are retained for storage.

```python
@torch.no_grad()
def extract_embeddings(self) -> np.ndarray:
    self.model.eval()
    all_embeddings = []
    for card_ids, features, lengths, labels in self.full_loader:
        embeddings, _ = self.model(card_ids, features, lengths)
        all_embeddings.append(embeddings.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)
```

**Post-extraction pipeline:**
1. UMAP(128→3) for 3D visualization coordinates
2. HDBSCAN clustering on the full 128-dim space
3. Store as `GameEmbedding` rows: `embedding_15d` (reused column, stores 128-dim), `embedding_3d`, `cluster_id`

## 5. Incremental Inference

**Implementation:** `src/tracker/ml/training.py::embed_new()`

When new games arrive after model training, they are embedded without retraining:

1. **Identify unembedded games:** Query battles with replay data but no `game_embeddings` row for `model_version = "tcn-v1"`.
2. **Build full dataset:** Reconstruct the `SequenceDataset` (needed for card vocabulary and feature encoding).
3. **Load saved model:** `tcn_v1.pt` checkpoint with `dropout=0.0` (inference mode).
4. **Forward pass:** Extract 128-dim embeddings for new games only.
5. **UMAP transform:** Apply the saved `umap_3d_standalone.pkl` reducer via `transform()` (not `fit_transform()`) — projects into the existing 3D manifold without distorting it.
6. **Store:** New `GameEmbedding` rows with `cluster_id = None` (would require full re-clustering to assign).

**Vocabulary growth handling:** If new cards have been seen since training (e.g., new card release), they map to index 0 (`<PAD>`). Their embeddings are effectively random until the model is retrained.

## 6. Full Training Pipeline: `train_tcn()`

The end-to-end pipeline orchestrated by `train_tcn()`:

```
1.  Build CardVocabulary from deck_cards table
2.  Create SequenceDataset (loads all PvP games with ≥4 valid replay events)
3.  Guard: require ≥50 games (insufficient signal below this)
4.  Initialize GameEmbeddingModel(vocab_size, dropout=0.2, embedding_dim=128)
5.  Create TCNTrainer with train/val split
6.  Train with early stopping → save best checkpoint
7.  Load best checkpoint for inference
8.  Extract 128-dim embeddings for all games
9.  Map dataset indices to battle_ids (handling filtered games)
10. UMAP 128d → 3d for visualization
11. HDBSCAN clustering on 128-dim space
12. Store all embeddings + cluster assignments in game_embeddings table
```

**Battle ID mapping challenge (step 9):** The `SequenceDataset` may filter games during construction (e.g., if `_invalid` events reduce the valid count below `MIN_EVENTS`). The pipeline re-derives the kept battle IDs by cross-referencing the dataset length with the event count query, ensuring correct battle_id → embedding alignment.
