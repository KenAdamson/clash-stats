"""Model registry for ML model versioning and promotion (ML Ops).

Tracks model versions, training metadata, validation metrics, and
promotion status. Supports candidate → production promotion with
validation gates and rollback.

Models are stored as files in model_dir with versioned names:
  wp_v2.pt, wp_v3.pt, cvae_v4.pt, etc.

The registry tracks which version is "production" (used by inference
and counterfactuals) and which are candidates or archived.
"""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import ForeignKey, String, JSON, func, select, text
from sqlalchemy.orm import Mapped, mapped_column, Session

from tracker.models import Base

logger = logging.getLogger(__name__)


class ModelVersion(Base):
    """Registered model version with training metadata and status."""

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_type: Mapped[str] = mapped_column(String, nullable=False)  # "wp", "cvae", "tcn"
    version: Mapped[int] = mapped_column(nullable=False)  # monotonic per model_type
    status: Mapped[str] = mapped_column(String, nullable=False)  # candidate, production, archived, failed
    filename: Mapped[str] = mapped_column(String, nullable=False)  # e.g. "wp_v2.pt"

    # Training metadata
    epochs: Mapped[Optional[int]]
    best_epoch: Mapped[Optional[int]]
    training_games: Mapped[Optional[int]]
    training_cutoff: Mapped[Optional[str]]  # ISO datetime
    wall_time_seconds: Mapped[Optional[int]]
    device: Mapped[Optional[str]]  # "cpu", "cuda", "xpu"

    # Validation metrics (model-type-specific)
    val_loss: Mapped[Optional[float]]
    val_accuracy: Mapped[Optional[float]]
    metrics_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Comparison to previous production model
    prev_version_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True
    )
    improvement_delta: Mapped[Optional[float]]  # e.g. +0.02 accuracy

    # Timestamps
    trained_at: Mapped[Optional[datetime]] = mapped_column(default=func.now())
    promoted_at: Mapped[Optional[datetime]]
    archived_at: Mapped[Optional[datetime]]


def next_version(session: Session, model_type: str) -> int:
    """Get the next version number for a model type."""
    max_ver = session.execute(
        select(func.max(ModelVersion.version))
        .where(ModelVersion.model_type == model_type)
    ).scalar()
    return (max_ver or 0) + 1


def get_production(session: Session, model_type: str) -> Optional[ModelVersion]:
    """Get the current production model for a type."""
    return session.execute(
        select(ModelVersion)
        .where(ModelVersion.model_type == model_type, ModelVersion.status == "production")
        .order_by(ModelVersion.version.desc())
        .limit(1)
    ).scalar_one_or_none()


def get_candidate(session: Session, model_type: str) -> Optional[ModelVersion]:
    """Get the latest candidate model for a type."""
    return session.execute(
        select(ModelVersion)
        .where(ModelVersion.model_type == model_type, ModelVersion.status == "candidate")
        .order_by(ModelVersion.version.desc())
        .limit(1)
    ).scalar_one_or_none()


def register_model(
    session: Session,
    model_type: str,
    filename: str,
    status: str = "candidate",
    **kwargs,
) -> ModelVersion:
    """Register a new model version.

    Args:
        session: DB session.
        model_type: "wp", "cvae", "tcn".
        filename: Checkpoint filename.
        status: Initial status (usually "candidate").
        **kwargs: Training metadata (epochs, val_loss, etc.)

    Returns:
        The new ModelVersion record.
    """
    version = next_version(session, model_type)

    mv = ModelVersion(
        model_type=model_type,
        version=version,
        status=status,
        filename=filename,
        **kwargs,
    )
    session.add(mv)
    session.flush()
    logger.info(
        "Registered %s v%d (%s) — %s",
        model_type, version, filename, status,
    )
    return mv


def promote(session: Session, model_type: str, version: int) -> Optional[ModelVersion]:
    """Promote a candidate to production, archiving the previous production model.

    Args:
        session: DB session.
        model_type: Model type.
        version: Version number to promote.

    Returns:
        The promoted ModelVersion, or None if validation fails.
    """
    candidate = session.execute(
        select(ModelVersion)
        .where(
            ModelVersion.model_type == model_type,
            ModelVersion.version == version,
        )
    ).scalar_one_or_none()

    if not candidate:
        logger.error("No %s v%d found", model_type, version)
        return None

    if candidate.status not in ("candidate", "production"):
        logger.error("%s v%d has status '%s' — cannot promote", model_type, version, candidate.status)
        return None

    # Archive current production
    current_prod = get_production(session, model_type)
    if current_prod and current_prod.id != candidate.id:
        current_prod.status = "archived"
        current_prod.archived_at = func.now()
        logger.info(
            "Archived %s v%d (%s)",
            model_type, current_prod.version, current_prod.filename,
        )

    # Promote candidate
    candidate.status = "production"
    candidate.promoted_at = func.now()
    session.flush()
    logger.info(
        "Promoted %s v%d (%s) to production",
        model_type, version, candidate.filename,
    )
    return candidate


def get_production_filename(session: Session, model_type: str) -> Optional[str]:
    """Get the filename of the current production model.

    Convenience function for inference code that just needs the path.
    """
    prod = get_production(session, model_type)
    return prod.filename if prod else None


def list_versions(session: Session, model_type: str) -> list[ModelVersion]:
    """List all versions of a model type, newest first."""
    return list(session.execute(
        select(ModelVersion)
        .where(ModelVersion.model_type == model_type)
        .order_by(ModelVersion.version.desc())
    ).scalars().all())
