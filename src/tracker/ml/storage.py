"""Storage helpers for ML feature vectors and embeddings.

Provides numpy array ↔ SQLite BLOB serialization and ORM models
for the game_features and game_embeddings tables.
"""

import numpy as np
from datetime import datetime
from typing import Optional

from sqlalchemy import ForeignKey, String, LargeBinary, func
from sqlalchemy.orm import Mapped, mapped_column

from tracker.models import Base


class GameFeature(Base):
    """Per-game feature vector extracted from replay data."""

    __tablename__ = "game_features"

    battle_id: Mapped[str] = mapped_column(
        ForeignKey("battles.battle_id"), primary_key=True
    )
    feature_vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    feature_version: Mapped[str] = mapped_column(String, default="v1")
    created_at: Mapped[Optional[datetime]] = mapped_column(default=func.now())


class GameEmbedding(Base):
    """UMAP embedding for a game, stored at two resolutions (15-dim analytical, 3-dim visualization)."""

    __tablename__ = "game_embeddings"

    battle_id: Mapped[str] = mapped_column(
        ForeignKey("battles.battle_id"), primary_key=True
    )
    embedding_15d: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    embedding_3d: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    cluster_id: Mapped[Optional[int]]
    model_version: Mapped[str] = mapped_column(String, default="umap-v1")
    created_at: Mapped[Optional[datetime]] = mapped_column(default=func.now())


def to_blob(arr: np.ndarray) -> bytes:
    """Serialize a numpy float32 array to bytes for SQLite BLOB storage."""
    return arr.astype(np.float32).tobytes()


def from_blob(data: bytes, dim: int) -> np.ndarray:
    """Deserialize a BLOB back to a numpy float32 array."""
    return np.frombuffer(data, dtype=np.float32).reshape(dim)
