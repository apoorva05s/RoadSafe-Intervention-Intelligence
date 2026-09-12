# RoadSafe Bengaluru

RoadSafe is an intervention intelligence platform for road safety. It combines historical crash intelligence, explainable risk prioritisation, local structural-break evidence, counterfactual intervention analysis, and budget optimisation to help authorities decide not only **where** risk is highest, but **what to do**, **what could happen**, and **how to spend limited safety budgets**.

For the complete implementation guide, see [SYSTEM_DOCUMENTATION.md](SYSTEM_DOCUMENTATION.md). For measured ML results, see [MODEL_PERFORMANCE.md](MODEL_PERFORMANCE.md).

## Run

```bash
pip install -r requirements.txt
python clean.py
streamlit run app.py
```

The application works without an LLM or external API key. The policy brief uses a deterministic template generated from verified structured values.

## Workflow

```text
Raw BTP / KSP / MoRTH data
        |
Data quality and station harmonisation
        |
Risk intelligence and crash anatomy
        |
Structural-break and matched-control signals
        |
Counterfactual scenarios
        |
Exact budget-constrained portfolio
        |
Action brief and measurement plan
        |
Future learning loop
```

## Application pages

- **Command Center:** city KPIs, risk map, priorities, and selected-budget summary.
- **Risk Intelligence:** explainable 0-100 score with burden, severity, and trend components.
- **ML & Explainability:** temporal XGBoost fatality-risk model, ROC-AUC/PR-AUC diagnostics, global SHAP importance, local crash explanations, and station-level ML severity risk.
- **What Changed:** interpretable mean-shift signals, matched pre-period controls, and trend compatibility.
- **What Works:** intervention knowledge base with literature evidence labels.
- **Counterfactual Lab:** compare multiple interventions for a station.
- **Budget Optimizer:** exact 0/1 dynamic programming and efficient frontier for Rs 5-100 Cr budgets.
- **Action Plans:** deterministic station brief and portfolio actions.
- **Method & Limits:** provenance, assumptions, and limitations shown in the product.

## Methodology

The risk score is an explainable weighted min-max index:

```text
score = 100 * (w1 * burden_z + w2 * severity_z + w3 * trend_z)
```

The default weights are 0.40, 0.30, and 0.30. Fatality values for 2024 and 2025 are estimated as fatal crashes multiplied by the observed 2021-2023 city killed/fatal ratio and remain flagged.

Forecasts use a simple OLS trend with COVID years excluded when enough observations remain. They are modeled baselines, not validated predictions.

Structural breaks are screened using pre/post mean shifts. Comparable controls are chosen using pre-break fatality level and trend only. A difference-in-differences-like differential is shown only as a quasi-experimental signal when pre-trends are compatible. No intervention date exists in the supplied data, so the product never claims that a break was caused by an intervention.

Scenario impact is conservative arithmetic:

```text
potential reduction = baseline * attributable share * literature effect * rollout ramp
combined effect = 1 - product(1 - individual effect)
```

Intervention effects and costs are literature-derived or modeled assumptions. The optimizer treats each station-intervention pair as a discrete package and uses exact dynamic programming at 0.1 Cr cost resolution. Its objective is modeled potential impact, not observed lives saved.

## Data and provenance

- BTP station-wise crash and fatality files for 2018-2025.
- BTP city historical totals for 2007-2025.
- BTP Road Safety Report profile tables and black spots.
- KSP crash-level records, station profiles, collision types, road types, and geocoded points.
- MoRTH city road-user, violation, and totals tables.
- Station coordinates are approximate centroids, not road-segment geometry.

`data/processed/` contains the canonical station master, station-year panel, city master, intervention candidates, data dictionary, and validation summary. The pipeline preserves source fields and explicitly records estimated killed values.

The crash ML pipeline is reproducible with `python train_model.py`. It creates `data/processed/crash_ml_dataset.csv`, `data/processed/crash_ml_predictions.csv`, `data/processed/station_ml_risk.csv`, `models/fatality_risk_model.pkl`, and `models/model_metadata.json`. The target is `fatal = severity == Fatal`; training uses 2016-2021 and testing uses 2022-2023. Historical station features use prior rows only. XGBoost is preferred, with a balanced random-forest fallback when XGBoost is unavailable.

The ML model predicts crash fatality probability from crash characteristics. SHAP explains model behaviour; it does not establish physical causality. ML severity informs intervention relevance through a transparent profile mapping, while intervention effect sizes remain evidence- or literature-based.

## Limitations

There is no station exposure denominator, validated intervention log, intervention timing, or complete geocoding. Stations are administrative jurisdictions. City-wide time patterns are not station-specific. Observational data and structural breaks cannot guarantee causal effects. Add treatment dates, comparison locations, implementation status, and quarterly outcomes before promoting signals to stronger evidence.

## 3-5 minute demo

1. Start on Command Center and set the budget to Rs 25 Cr.
2. Select a high-risk station in Risk Intelligence and show statistical, ML severity, and composite risk.
3. Open ML & Explainability to show temporal test metrics, global SHAP, and a local crash explanation.
4. Open What Changed to show a structural-break signal and pre-trend compatibility.
5. Compare three interventions in Counterfactual Lab and point out the modeled/literature badges.
6. Open Budget Optimizer and explain the exact discrete portfolio and efficient frontier.
7. Open Action Plans to generate the deterministic brief and monitoring metric.
8. Close on Method & Limits: RoadSafe is designed to learn once implementation and outcome data arrive.

## Hackathon presentation outline

1. Problem: crash data tells authorities where, but not what to fund next.
2. Product: the RoadSafe intervention intelligence workflow.
3. AI: temporal XGBoost learns crash-level severity patterns; SHAP explains them.
4. Evidence: explainable risk, crash anatomy, and cautious quasi-experimental signals.
5. Decision engine: counterfactual comparisons and budget-constrained portfolio selection.
6. Trust and scale: evidence labels remain visible, with Bengaluru as the first reusable city deployment.

Long-term architecture: local city evidence feeds a shared intervention knowledge base, while each deployment retains its local station, road, and implementation context. The future learning loop is **identify -> diagnose -> intervene -> measure -> evaluate -> learn**.
