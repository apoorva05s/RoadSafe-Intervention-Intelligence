# RoadSafe Fatality Model Performance

## Model purpose

The crash-level model estimates the probability that a recorded crash is fatal from observed crash characteristics. It is a severity model, not an accident-occurrence model and not a causal model.

## Data and target

- Source: `data/ksp_points.csv`
- Rows: 10,535 crash records
- Target: `fatal = 1` when `severity == "Fatal"`; otherwise `0`
- Fatal rows: 1,827
- Features: year, latitude, longitude, station, zone, subdivision, collision, road type, and prior station history
- Historical features use prior rows only and do not use the current crash target

## Temporal evaluation

The model is trained on 2016-2021 and evaluated on the later 2022-2023 period. This is the primary evaluation because it better represents deployment on future crash records than a random split.

Model: XGBoost classifier (`XGBClassifier`)

| Metric | Test result |
|---|---:|
| ROC-AUC | 0.604 |
| PR-AUC / Average Precision | 0.252 |
| Precision | 0.230 |
| Recall | 0.103 |
| F1 | 0.142 |

Confusion matrix at probability threshold 0.50:

| | Predicted non-fatal | Predicted fatal |
|---|---:|---:|
| Actual non-fatal | 2,451 | 231 |
| Actual fatal | 603 | 69 |

The test prevalence is approximately 19.2%, so PR-AUC is especially relevant. The model shows signal above a prevalence-only baseline, but recall at the default threshold is limited. It should support prioritisation and investigation, not be used as an automated enforcement decision.

## Explainability

SHAP global importance and local explanations are available in the **ML & Explainability** page. SHAP describes which features move a model prediction; it does not prove that a road type, collision type, or station causes fatality.

## Saved artifacts

- `models/fatality_risk_model.pkl`
- `models/model_metadata.json`
- `data/processed/crash_ml_predictions.csv`
- `data/processed/station_ml_risk.csv`

Retrain with:

```bash
python train_model.py
```

## Limitations and next evaluation steps

The crash records are observational, station boundaries are administrative, exposure denominators are unavailable, and the KSP dataset covers a historical period. Future model reviews should add calibration curves, threshold selection by operational cost, time-based rolling validation, external-city validation, and monitoring for data drift. Missing `joblib`, XGBoost, or SHAP dependencies do not stop the dashboard: it falls back to statistical risk intelligence and clearly labels the ML page unavailable.