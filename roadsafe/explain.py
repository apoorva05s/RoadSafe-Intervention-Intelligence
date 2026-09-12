"""SHAP explanations for the persisted fatality model."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_bundle(root=None):
    root = Path(root or Path(__file__).parent)
    model_path = root / "models" / "fatality_risk_model.pkl"
    metadata_path = root / "models" / "model_metadata.json"
    if not model_path.exists():
        return None, None
    try:
        import joblib
    except ImportError:
        # Keep the dashboard usable when the optional persisted-model dependency
        # is missing from the interpreter that launches Streamlit.
        return None, {"load_error": "joblib is not installed; statistical intelligence remains available."}
    try:
        bundle = joblib.load(model_path)
    except Exception as exc:
        return None, {"load_error": f"saved ML model could not be loaded: {exc}"}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return bundle, metadata


def shap_values(bundle, features):
    """Return SHAP values, with model-native importance as a safe fallback."""
    model = bundle["model"]
    names = list(features.columns)
    try:
        import shap
        values = shap.TreeExplainer(model)(features).values
        if values.ndim == 3:
            values = values[:, :, 1]
        return np.asarray(values), "SHAP"
    except Exception:
        values = np.tile(getattr(model, "feature_importances_", np.zeros(len(names))), (len(features), 1))
        return values, "Model feature importance fallback"


def global_importance(bundle, features):
    values, method = shap_values(bundle, features)
    return pd.DataFrame({"feature": features.columns, "importance": np.abs(values).mean(axis=0), "explanation_method": method}).sort_values("importance", ascending=False)


def local_explanation(bundle, features, row_index=0, top_n=8):
    values, method = shap_values(bundle, features.iloc[[row_index]])
    output = pd.DataFrame({"feature": features.columns, "contribution": values[0], "value": features.iloc[row_index].values})
    return output.reindex(output.contribution.abs().sort_values(ascending=False).index).head(top_n), method