"""Crash-level fatality model with temporal evaluation and safe feature construction."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
PROCESSED = DATA / "processed"

BASE_CATEGORICAL = ["station", "collision", "road_type", "zone", "subdivision"]
BASE_NUMERIC = ["year", "lat", "lng", "historical_station_fatal_rate", "historical_station_crashes"]


def load_crashes(path=None):
    frame = pd.read_csv(path or DATA / "ksp_points.csv")
    frame = frame.copy()
    frame["fatal"] = frame["severity"].astype(str).str.strip().str.casefold().eq("fatal").astype(int)
    frame["crash_id"] = np.arange(len(frame))
    station_master = pd.read_csv(PROCESSED / "station_master.csv", usecols=["station", "zone", "subdivision"])
    frame = frame.merge(station_master.drop_duplicates("station"), on="station", how="left")
    frame["zone"] = frame["zone"].fillna("Unknown")
    frame["subdivision"] = frame["subdivision"].fillna("Unknown")
    return frame


def add_prior_history(frame):
    """Add leave-current-year-out historical station features; no future labels enter a row."""
    frame = frame.sort_values(["station", "year", "crash_id"]).copy()
    grouped = frame.groupby("station", sort=False)
    prior_crashes = grouped.cumcount()
    prior_fatal = grouped["fatal"].cumsum() - frame["fatal"]
    frame["historical_station_crashes"] = prior_crashes.astype(float)
    frame["historical_station_fatal_rate"] = np.where(prior_crashes > 0, prior_fatal / prior_crashes, 0.0)
    return frame


def make_features(frame, feature_columns=None):
    frame = add_prior_history(frame)
    numeric = frame[BASE_NUMERIC].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    categorical = pd.get_dummies(frame[BASE_CATEGORICAL].fillna("Unknown").astype(str), prefix=BASE_CATEGORICAL)
    features = pd.concat([numeric, categorical], axis=1).astype(float)
    if feature_columns is not None:
        features = features.reindex(columns=feature_columns, fill_value=0.0)
    return features, frame


def temporal_split(frame, test_years=2):
    cutoff = int(frame["year"].max()) - test_years + 1
    return frame[frame["year"] < cutoff].copy(), frame[frame["year"] >= cutoff].copy(), cutoff


def build_model(random_state=42):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=random_state, n_jobs=-1)
    return XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.85,
        colsample_bytree=0.85, objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=random_state, n_jobs=2,
    )


def metrics_for(model, X, y):
    from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    probability = model.predict_proba(X)[:, 1]
    predicted = (probability >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, predicted).tolist(),
    }


def intervention_relevance(profile, intervention_id, mean_probability):
    """Transparent ML-informed relevance adjustment, not an effect estimate."""
    profile = profile or {}
    ped = float(profile.get("fatal_ped_share", 0.27) or 0.27)
    highway = float(profile.get("fatal_highway_share", 0.24) or 0.24)
    junction = float(profile.get("fatal_junction_share", 0.23) or 0.23)
    hitrun = float(profile.get("fatal_hitrun_share", 0.35) or 0.35)
    severity = float(np.clip(mean_probability, 0, 1))
    signal = {
        "ped": ped,
        "helmet": 1 - ped,
        "speed": highway,
        "blackspot": junction,
        "hmv": highway,
        "hitrun": hitrun,
        "night": severity,
        "dui": severity,
    }.get(intervention_id, severity)
    return float(1 + np.clip(0.20 * (signal - 0.25) + 0.10 * (severity - 0.20), -0.12, 0.15))