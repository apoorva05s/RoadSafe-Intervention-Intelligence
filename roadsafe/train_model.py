"""Train and persist the crash-level fatality model.

Run from the project directory with: python train_model.py
"""
import json
from datetime import datetime, timezone

import joblib

from ml_model import DATA, PROCESSED, ROOT, build_model, load_crashes, make_features, metrics_for, temporal_split


def train():
    frame = load_crashes()
    train_rows, test_rows, cutoff = temporal_split(frame, test_years=2)
    all_features, enriched = make_features(frame)
    train_mask = enriched["year"] < cutoff
    test_mask = ~train_mask
    train_rows = enriched.loc[train_mask].copy()
    test_rows = enriched.loc[test_mask].copy()
    train_features = all_features.loc[train_mask]
    test_features = all_features.loc[test_mask]
    model = build_model()
    model.fit(train_features, train_rows["fatal"])
    metrics = metrics_for(model, test_features, test_rows["fatal"])

    enriched["ml_fatality_probability"] = model.predict_proba(all_features)[:, 1]
    PROCESSED.mkdir(exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "feature_columns": train_features.columns.tolist()}, ROOT / "models" / "fatality_risk_model.pkl")
    metadata = {
        "model_type": type(model).__name__, "training_period": [int(train_rows.year.min()), int(train_rows.year.max())],
        "test_period": [int(test_rows.year.min()), int(test_rows.year.max())], "feature_names": train_features.columns.tolist(),
        "metrics": metrics, "target": "fatal = severity == Fatal", "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "data/ksp_points.csv", "data_version": f"{len(frame)} crash rows",
    }
    (ROOT / "models" / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    enriched.to_csv(PROCESSED / "crash_ml_predictions.csv", index=False)
    station = enriched.groupby("station").agg(
        mean_predicted_fatality_probability=("ml_fatality_probability", "mean"),
        median_predicted_fatality_probability=("ml_fatality_probability", "median"),
        high_risk_crash_share=("ml_fatality_probability", lambda values: (values >= 0.5).mean()),
        crash_count=("crash_id", "count"),
    ).reset_index()
    station.to_csv(PROCESSED / "station_ml_risk.csv", index=False)
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    train()