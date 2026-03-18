"""Activity-aware corpus scheduling via ML prioritization.

Trains a GradientBoostingClassifier to predict P(has_new_battles | player, current_time),
then scores corpus players so the scrape queue processes likely-active players first.
"""

import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from tracker.models import Battle, PlayerCorpus

logger = logging.getLogger(__name__)

MODEL_FILENAME = "activity_model.pkl"


def _cyclical_encode(value: float, period: float) -> tuple[float, float]:
    """Encode a cyclical feature as (sin, cos)."""
    angle = 2 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


def _build_player_profiles(session: Session) -> dict[str, dict]:
    """Build per-player activity profiles from battle history.

    Returns:
        Dict mapping player_tag -> {
            'hourly_counts': dict[int, int],  # hour_utc -> battle count
            'dow_counts': dict[int, int],      # day_of_week -> battle count
            'total_battles': int,
            'last_battle_time': datetime | None,
            'last_scraped': datetime | None,
            'trophy_mid': float | None,
            'first_battle_time': datetime | None,
        }
    """
    # Get per-player hourly + dow histograms from battle_time.
    # battle_time is a DATETIME column — HOUR() and DAYOFWEEK() are native and fast.
    rows = session.execute(
        text("""
            SELECT
                player_tag,
                HOUR(battle_time) AS hour_utc,
                DAYOFWEEK(battle_time) AS dow,
                COUNT(*) AS cnt
            FROM battles
            WHERE corpus IS NOT NULL
              AND battle_time IS NOT NULL
            GROUP BY player_tag, hour_utc, dow
        """)
    ).all()

    profiles: dict[str, dict] = {}
    for row in rows:
        tag = row[0]
        hour = row[1]
        dow = row[2]
        cnt = row[3]

        if hour is None or dow is None:
            continue

        if tag not in profiles:
            profiles[tag] = {
                'hourly_counts': {},
                'dow_counts': {},
                'total_battles': 0,
            }

        p = profiles[tag]
        p['hourly_counts'][int(hour)] = p['hourly_counts'].get(int(hour), 0) + cnt
        p['total_battles'] += cnt

        # dow from MySQL: 1=Sunday, 2=Monday, ..., 7=Saturday
        # Convert to 0=Monday, ..., 6=Sunday
        dow_py = (dow - 2) % 7
        p['dow_counts'][dow_py] = p['dow_counts'].get(dow_py, 0) + cnt

    # Get last/first battle time per player — native DATETIME, so
    # MAX/MIN return datetime objects directly.
    for tag, p in profiles.items():
        p['last_battle_time'] = None
        p['first_battle_time'] = None

    time_rows = session.execute(
        text("""
            SELECT player_tag,
                   MAX(battle_time) AS last_bt,
                   MIN(battle_time) AS first_bt
            FROM battles
            WHERE corpus IS NOT NULL
              AND battle_time IS NOT NULL
            GROUP BY player_tag
        """)
    ).all()

    for row in time_rows:
        tag = row[0]
        if tag in profiles:
            last_bt = row[1]
            first_bt = row[2]
            if last_bt:
                if last_bt.tzinfo is None:
                    last_bt = last_bt.replace(tzinfo=timezone.utc)
                profiles[tag]['last_battle_time'] = last_bt
            if first_bt:
                if first_bt.tzinfo is None:
                    first_bt = first_bt.replace(tzinfo=timezone.utc)
                profiles[tag]['first_battle_time'] = first_bt

    # Get last_scraped and trophy info from player_corpus
    corpus_rows = session.execute(
        select(
            PlayerCorpus.player_tag,
            PlayerCorpus.last_scraped,
            PlayerCorpus.trophy_range_low,
            PlayerCorpus.trophy_range_high,
        ).where(PlayerCorpus.active == 1)
    ).all()

    for row in corpus_rows:
        tag = row[0]
        if tag in profiles:
            profiles[tag]['last_scraped'] = row[1]
            low = row[2] or 0
            high = row[3] or 0
            profiles[tag]['trophy_mid'] = (low + high) / 2.0 if (low or high) else None

    return profiles


def _build_feature_vector(
    profile: dict,
    hour_utc: int,
    day_of_week: int,
    now: datetime,
) -> np.ndarray:
    """Build a 10-feature vector for a (player, time) pair.

    Features:
        0-1: hour_utc sin/cos (cyclical)
        2-3: day_of_week sin/cos (cyclical)
        4:   player_hour_weight — fraction of player's battles at this hour
        5:   player_total_battles (log-scaled)
        6:   hours_since_last_battle
        7:   hours_since_last_scraped
        8:   player_avg_battles_per_day
        9:   trophy_range_mid (normalized to ~0-1 scale)
    """
    features = np.zeros(10, dtype=np.float32)

    # Cyclical time encoding
    h_sin, h_cos = _cyclical_encode(hour_utc, 24.0)
    d_sin, d_cos = _cyclical_encode(day_of_week, 7.0)
    features[0] = h_sin
    features[1] = h_cos
    features[2] = d_sin
    features[3] = d_cos

    # Player hour weight
    total = profile['total_battles']
    if total > 0:
        hour_count = profile['hourly_counts'].get(hour_utc, 0)
        features[4] = hour_count / total
    else:
        features[4] = 0.0

    # Log-scaled total battles
    features[5] = math.log1p(total)

    # Hours since last battle
    last_bt = profile.get('last_battle_time')
    if last_bt:
        delta = (now - last_bt).total_seconds() / 3600.0
        features[6] = min(delta, 720.0)  # Cap at 30 days
    else:
        features[6] = 720.0

    # Hours since last scraped
    last_sc = profile.get('last_scraped')
    if last_sc:
        if last_sc.tzinfo is None:
            last_sc = last_sc.replace(tzinfo=timezone.utc)
        delta = (now - last_sc).total_seconds() / 3600.0
        features[7] = min(delta, 720.0)
    else:
        features[7] = 720.0

    # Avg battles per day
    first_bt = profile.get('first_battle_time')
    if first_bt and last_bt and total > 1:
        span_days = max((last_bt - first_bt).total_seconds() / 86400.0, 1.0)
        features[8] = total / span_days
    else:
        features[8] = 0.0

    # Trophy midpoint (normalized: 7000-15000 range → 0-1)
    trophy_mid = profile.get('trophy_mid')
    if trophy_mid and trophy_mid > 0:
        features[9] = (trophy_mid - 7000.0) / 8000.0
    else:
        features[9] = 0.5  # Unknown → middle

    return features


def _build_training_data(
    profiles: dict[str, dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Build training feature matrix and labels from player profiles.

    Positive examples: (player, hour, dow) combos where battles occurred.
    Negative examples: (player, hour, dow) combos with no battles (sampled).

    Returns:
        (X, y) — feature matrix and binary labels.
    """
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    # Use a fixed reference time for training — doesn't matter for
    # hour/dow features, and hours_since_* will be recomputed at inference
    ref_time = datetime.now(timezone.utc)

    for tag, profile in profiles.items():
        total = profile['total_battles']
        if total < 5:
            continue  # Skip players with too little data

        # Positive examples: hours where this player was active
        for hour, count in profile['hourly_counts'].items():
            # Weight: approximate day_of_week from dow_counts
            for dow, dow_count in profile['dow_counts'].items():
                # Only create examples for (hour, dow) combos that are plausible
                # Use a probability-weighted approach to avoid N*24*7 explosion
                pass

        # Simpler approach: create examples at the hourly level
        # Positive: hours with battles (1 example per active hour)
        active_hours = set(profile['hourly_counts'].keys())
        for hour in range(24):
            # For each hour, create one example with average dow features
            avg_dow = sum(
                dow * cnt for dow, cnt in profile['dow_counts'].items()
            ) / total if total > 0 else 3.0

            vec = _build_feature_vector(profile, hour, int(avg_dow), ref_time)

            if hour in active_hours:
                # Weight by how many battles at this hour
                weight = profile['hourly_counts'][hour]
                # Add proportional positives (capped at 5 to avoid imbalance)
                n_pos = min(max(1, weight // 10), 5)
                for _ in range(n_pos):
                    X_list.append(vec)
                    y_list.append(1)
            else:
                # Negative: no battles at this hour
                X_list.append(vec)
                y_list.append(0)

        # Also create (hour, dow) cross examples for peak times
        top_hours = sorted(
            profile['hourly_counts'].items(), key=lambda x: x[1], reverse=True
        )[:3]
        for hour, _ in top_hours:
            for dow in range(7):
                vec = _build_feature_vector(profile, hour, dow, ref_time)
                dow_count = profile['dow_counts'].get(dow, 0)
                if dow_count > 0:
                    X_list.append(vec)
                    y_list.append(1)
                else:
                    X_list.append(vec)
                    y_list.append(0)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    return X, y


def train_activity_model(
    session: Session,
    model_dir: str | Path = "data/ml_models",
) -> Optional[dict]:
    """Train activity prediction model from battle history.

    Args:
        session: SQLAlchemy session.
        model_dir: Directory to save the trained model.

    Returns:
        Dict with training metrics, or None if insufficient data.
    """
    import joblib
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score

    logger.info("Building player activity profiles...")
    profiles = _build_player_profiles(session)
    logger.info("Built profiles for %d players.", len(profiles))

    if len(profiles) < 10:
        logger.warning("Too few players with battle data (%d). Need at least 10.", len(profiles))
        return None

    logger.info("Building training data...")
    X, y = _build_training_data(profiles)
    logger.info(
        "Training data: %d examples (%d positive, %d negative).",
        len(y), int(y.sum()), int(len(y) - y.sum()),
    )

    if len(y) < 100:
        logger.warning("Too few training examples (%d). Need at least 100.", len(y))
        return None

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    # Train classifier
    logger.info("Training GradientBoostingClassifier...")
    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:, 1]
    accuracy = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)

    # Feature importance
    feature_names = [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "player_hour_weight", "log_total_battles",
        "hours_since_last_battle", "hours_since_last_scraped",
        "avg_battles_per_day", "trophy_mid",
    ]
    importances = clf.feature_importances_
    importance_str = ", ".join(
        f"{name}={imp:.3f}"
        for name, imp in sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
    )

    logger.info(
        "Activity model trained: accuracy=%.3f, AUC=%.3f, "
        "top features: %s",
        accuracy, auc, importance_str,
    )

    # Save model
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    save_path = model_path / MODEL_FILENAME
    joblib.dump(clf, save_path)
    logger.info("Model saved to %s", save_path)

    metrics = {
        "accuracy": round(accuracy, 4),
        "auc": round(auc, 4),
        "train_examples": len(y_train),
        "val_examples": len(y_val),
        "positive_rate": round(float(y.sum()) / len(y), 4),
        "players_profiled": len(profiles),
        "feature_importances": dict(zip(feature_names, importances.tolist())),
    }
    return metrics


def _load_model(model_dir: str | Path = "data/ml_models"):
    """Load trained activity model from disk.

    Returns:
        Classifier or None if not found.
    """
    import joblib

    model_path = Path(model_dir) / MODEL_FILENAME
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def score_corpus_players(
    session: Session,
    model_dir: str | Path = "data/ml_models",
) -> Optional[list[tuple[str, float]]]:
    """Score all active corpus players by P(has_new_battles) at current time.

    Args:
        session: SQLAlchemy session.
        model_dir: Directory containing the trained model.

    Returns:
        List of (player_tag, probability) sorted descending,
        or None if model not found.
    """
    clf = _load_model(model_dir)
    if clf is None:
        logger.info("Activity model: no trained model found, skipping scoring.")
        return None

    logger.info("Activity model: building player profiles from battle history...")
    profiles = _build_player_profiles(session)
    if not profiles:
        logger.info("Activity model: no player profiles found.")
        return None

    now = datetime.now(timezone.utc)
    hour_utc = now.hour
    day_of_week = now.weekday()  # 0=Monday, 6=Sunday

    logger.info(
        "Activity model: scoring %d players at UTC hour %d (dow %d)...",
        len(profiles), hour_utc, day_of_week,
    )

    # Build feature vectors for all profiled players
    tags = []
    X_list = []
    for tag, profile in profiles.items():
        vec = _build_feature_vector(profile, hour_utc, day_of_week, now)
        tags.append(tag)
        X_list.append(vec)

    if not tags:
        return []

    X = np.array(X_list, dtype=np.float32)
    probs = clf.predict_proba(X)[:, 1]

    # Sort by probability descending
    scored = list(zip(tags, probs.tolist()))
    scored.sort(key=lambda x: x[1], reverse=True)

    logger.info("Activity model: scoring complete.")
    return scored
