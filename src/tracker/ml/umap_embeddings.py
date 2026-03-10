"""Two-stage UMAP embedding pipeline.

Stage 1: 50-dim features → 15-dim analytical embedding (for clustering/similarity)
Stage 2: 15-dim → 3-dim visualization embedding (for 3D scatter plots)

Supports supervised UMAP using win/loss labels for better separation.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from sqlalchemy import select
from sqlalchemy.orm import Session

from tracker.models import Battle
from tracker.ml.storage import GameEmbedding, to_blob

logger = logging.getLogger(__name__)

# Default UMAP hyperparameters (ADR-003)
UMAP_15D_PARAMS = dict(
    n_components=15,
    n_neighbors=30,
    min_dist=0.1,
    metric="euclidean",
    random_state=42,
)
UMAP_3D_PARAMS = dict(
    n_components=3,
    n_neighbors=30,
    min_dist=0.3,
    spread=1.5,
    metric="euclidean",
    random_state=42,
)

MODEL_VERSION = "umap-v2"


class EmbeddingPipeline:
    """Fits and transforms game features into UMAP embeddings.

    Args:
        model_dir: Directory to save/load fitted models.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or Path("data/ml_models")
        self.scaler = StandardScaler()
        self.reducer_15d = UMAP(**UMAP_15D_PARAMS)
        self.reducer_3d = UMAP(**UMAP_3D_PARAMS)
        self._fitted = False

    def fit_transform(
        self,
        battle_ids: list[str],
        features: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit UMAP on feature matrix and return embeddings.

        Args:
            battle_ids: Battle IDs corresponding to rows.
            features: Shape (n_games, n_features).
            labels: Optional win/loss labels (1/0) for supervised UMAP.

        Returns:
            Tuple of (embeddings_15d, embeddings_3d).
        """
        logger.info("Fitting UMAP on %d games, %d features", len(battle_ids), features.shape[1])

        # Scale features
        scaled = self.scaler.fit_transform(features)

        # Stage 1: high-dimensional analytical embedding
        if labels is not None:
            logger.info("Using supervised UMAP with win/loss labels")
            self.reducer_15d.set_params(target_metric="categorical")
            embeddings_15d = self.reducer_15d.fit_transform(scaled, y=labels)
        else:
            embeddings_15d = self.reducer_15d.fit_transform(scaled)

        # Stage 2: 3D visualization embedding
        embeddings_3d = self.reducer_3d.fit_transform(embeddings_15d)

        self._fitted = True
        self._save_models()

        logger.info(
            "UMAP complete: %d → %d → %d dimensions",
            features.shape[1], embeddings_15d.shape[1], embeddings_3d.shape[1],
        )

        return embeddings_15d, embeddings_3d

    def transform(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Transform new features using fitted UMAP (for incremental updates).

        Args:
            features: Shape (n_new_games, n_features).

        Returns:
            Tuple of (embeddings_15d, embeddings_3d).
        """
        if not self._fitted:
            self._load_models()

        scaled = self.scaler.transform(features)
        embeddings_15d = self.reducer_15d.transform(scaled)
        embeddings_3d = self.reducer_3d.transform(embeddings_15d)
        return embeddings_15d, embeddings_3d

    def _save_models(self) -> None:
        """Persist fitted models to disk."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        path = self.model_dir / "umap_pipeline.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "reducer_15d": self.reducer_15d,
                "reducer_3d": self.reducer_3d,
            }, f)
        logger.info("Saved UMAP models to %s", path)

    def _load_models(self) -> None:
        """Load fitted models from disk."""
        path = self.model_dir / "umap_pipeline.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No fitted models at {path}. Run --train-embeddings first.")
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.scaler = state["scaler"]
        self.reducer_15d = state["reducer_15d"]
        self.reducer_3d = state.get("reducer_3d") or state.get("reducer_2d")
        self._fitted = True
        logger.info("Loaded UMAP models from %s", path)


    def reduce_to_3d(self, embeddings_high: np.ndarray) -> np.ndarray:
        """Reduce arbitrary high-dim embeddings to 3D for visualization.

        Fits a new UMAP(n_components=3) on the input. Used to project
        TCN 128-dim embeddings to 3D for the Plotly scatter plot.

        Args:
            embeddings_high: Shape (n_games, high_dim).

        Returns:
            Shape (n_games, 3) coordinates.
        """
        reducer = UMAP(**UMAP_3D_PARAMS)
        embeddings_3d = reducer.fit_transform(embeddings_high)

        # Save the reducer for this dimensionality
        self.model_dir.mkdir(parents=True, exist_ok=True)
        path = self.model_dir / "umap_3d_standalone.pkl"
        with open(path, "wb") as f:
            pickle.dump(reducer, f)
        logger.info("Saved standalone 3D UMAP reducer to %s", path)

        return embeddings_3d


def train_embeddings(
    session: Session,
    battle_ids: list[str],
    features: np.ndarray,
    supervised: bool = True,
    model_dir: Optional[Path] = None,
) -> None:
    """Full training pipeline: fit UMAP, cluster, store embeddings.

    Args:
        session: DB session.
        battle_ids: Battle IDs for feature rows.
        features: Shape (n_games, n_features).
        supervised: Use win/loss labels for supervised UMAP.
        model_dir: Directory for model persistence.
    """
    from tracker.ml.clustering import label_clusters

    # Get win/loss labels if supervised
    labels = None
    if supervised:
        results = session.execute(
            select(Battle.battle_id, Battle.result)
            .where(Battle.battle_id.in_(battle_ids))
        ).all()
        result_map = {r[0]: r[1] for r in results}
        labels = np.array([
            1.0 if result_map.get(bid) == "win" else 0.0
            for bid in battle_ids
        ])

    # Fit UMAP
    pipeline = EmbeddingPipeline(model_dir=model_dir)
    embeddings_15d, embeddings_3d = pipeline.fit_transform(battle_ids, features, labels)

    # Cluster on 15-dim space
    cluster_ids = label_clusters(embeddings_15d)

    # Store embeddings in DB
    logger.info("Storing %d embeddings", len(battle_ids))
    for i, battle_id in enumerate(battle_ids):
        session.merge(GameEmbedding(
            battle_id=battle_id,
            embedding_15d=to_blob(embeddings_15d[i]),
            embedding_3d=to_blob(embeddings_3d[i]),
            cluster_id=int(cluster_ids[i]) if cluster_ids[i] >= 0 else None,
            model_version=MODEL_VERSION,
        ))

        if (i + 1) % 500 == 0:
            session.flush()

    session.commit()
    logger.info("Embeddings stored successfully")
