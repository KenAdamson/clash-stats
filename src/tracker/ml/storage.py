"""Storage helpers for ML feature vectors and embeddings.

Provides ORM models for game_features and game_embeddings tables.
Supports both legacy BLOB columns and native pgvector columns
during the VectorChord migration.
"""

import numpy as np
from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
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
    feature_vec: Mapped[Optional[list]] = mapped_column(Vector(51), nullable=True)
    feature_version: Mapped[str] = mapped_column(String, default="v1")
    created_at: Mapped[Optional[datetime]] = mapped_column(default=func.now())


class GameEmbedding(Base):
    """Game embedding at multiple resolutions."""

    __tablename__ = "game_embeddings"

    battle_id: Mapped[str] = mapped_column(
        ForeignKey("battles.battle_id"), primary_key=True
    )
    # Legacy BLOB columns (kept during migration)
    embedding_15d: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    embedding_3d: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    # Native vector columns (VectorChord/pgvector)
    embedding_tcn_128d: Mapped[Optional[list]] = mapped_column(Vector(128), nullable=True)
    embedding_vec_3d: Mapped[Optional[list]] = mapped_column(Vector(3), nullable=True)

    cluster_id: Mapped[Optional[int]]
    model_version: Mapped[str] = mapped_column(String, default="umap-v1")
    created_at: Mapped[Optional[datetime]] = mapped_column(default=func.now())


def to_blob(arr: np.ndarray) -> bytes:
    """Serialize a numpy float32 array to bytes for BLOB storage."""
    return arr.astype(np.float32).tobytes()


def from_blob(data: bytes, dim: int) -> np.ndarray:
    """Deserialize a BLOB back to a numpy float32 array."""
    return np.frombuffer(data, dtype=np.float32).reshape(dim)
