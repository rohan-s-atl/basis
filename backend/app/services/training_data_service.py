from typing import Any

from sqlalchemy.orm import Session

from app.db.models import FeatureSnapshot, Outcome, Prediction
from app.services.feature_service import flatten_feature_snapshot


def export_training_dataset(db: Session, *, limit: int = 10_000) -> list[dict[str, Any]]:
    rows = (
        db.query(FeatureSnapshot, Outcome)
        .join(Prediction, Prediction.id == FeatureSnapshot.prediction_id)
        .join(Outcome, Outcome.prediction_id == Prediction.id)
        .order_by(Outcome.computed_at.desc())
        .limit(limit)
        .all()
    )

    dataset: list[dict[str, Any]] = []
    for snapshot, outcome in rows:
        dataset.append(
            {
                "features": flatten_feature_snapshot(
                    snapshot.event_features,
                    snapshot.market_features,
                    snapshot.derived_features,
                ),
                "label": outcome.label,
            }
        )
    return dataset
