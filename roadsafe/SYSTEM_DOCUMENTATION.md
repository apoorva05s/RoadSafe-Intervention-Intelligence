# RoadSafe Bengaluru System Documentation

## 1. What the system is

RoadSafe is a road-safety intervention intelligence platform. It is designed to answer:

1. Where is fatality burden concentrated?
2. Why is a station high risk?
3. What crash characteristics are associated with fatal outcomes?
4. What changed in the historical trajectory?
5. Which interventions are relevant to the observed crash profile?
6. What is the modeled outcome under alternative interventions?
7. Which discrete intervention packages fit a fixed budget?

It is decision-support software. It is not an accident-occurrence predictor, a traffic simulator, a digital twin, or a causal proof system.

The central separation is:

```text
Historical station data -> statistical risk and forecasts
Crash-level data       -> ML fatality severity probability and SHAP explanations
Historical trajectories -> structural-break and quasi-experimental signals
Evidence knowledge base -> intervention effect assumptions
All of the above       -> counterfactual scenarios and budget portfolio
```

## 2. Application entry point

`app.py` is the Streamlit entry point. It imports `decision_app.py`, which builds the complete interface. The application can run without an ML model or optional model dependencies; in that situation it keeps the statistical workflow available and displays an ML-unavailable message.

Run it from the directory containing `app.py`:

```powershell
cd C:\Users\Apoorva\Downloads\roadsafe_bengaluru(1)\roadsafe
streamlit run app.py
```

The equivalent explicit-interpreter command is:

```powershell
C:\Users\Apoorva\anaconda3\envs\cnn_env\python.exe -m streamlit run app.py
```

## 3. Source data

### BTP station data

The raw station files provide station-wise totals, fatal crashes, killed people, and injuries for 2018-2023. The 2024 and 2025 files provide fatal and total crashes but do not provide killed people.

### BTP city history

The city file provides Bengaluru annual totals from 2007-2025 and is used for city-level historical context and forecasting.

### KSP crash records

`data/ksp_points.csv` contains 10,535 crash-level records from 2016-2023 with:

- station
- year
- severity
- collision
- road type
- latitude
- longitude

These records are used for crash anatomy, fatal crash mapping, ML training, and station-level aggregation of predicted severity.

### Other sources

- `city_profile.csv`: city-level road-user, time, day, violation, and crash-type shares.
- `ksp_station_profile.csv`: station-level KSP crash anatomy summaries.
- `ksp_fatal_by_collision.csv`: station fatal collision distributions.
- `ksp_fatal_by_roadtype.csv`: station fatal road-type distributions.
- `blackspots.csv`: BTP-listed high-risk locations and coordinates.
- `stations_latlng.csv`: approximate station centroid coordinates.
- `interventions.csv`: intervention triggers, costs, effect assumptions, horizons, and sources.

The source data does not contain a complete validated intervention log. Therefore the application does not invent intervention dates, treatment status, traffic exposure, near misses, or observed intervention outcomes.

## 4. Data preparation pipeline

The pipeline is implemented in `clean.py`.

### 4.1 Station name harmonisation

`clean.py` uses an explicit `ALIASES` dictionary to map known historical station names to canonical names. For example, older names such as `Ulsoor` and `Yelahanka` are mapped to the canonical station labels used by later BTP files. Rows containing total-summary labels are removed. Ambiguous names are not automatically guessed.

### 4.2 Long station-year table

The station files are converted into a common schema:

```text
station, year, fatal, killed, injured, total, killed_est
```

Rows are grouped by station and year after alias mapping. This makes duplicate station-year detection explicit.

### 4.3 Estimated 2024-2025 killed values

For 2024 and 2025, killed people are estimated because only fatal-crash counts are available:

```text
estimated killed = fatal crashes * city killed/fatal ratio for 2021-2023
```

The calculated ratio is approximately `1.036`. The result is rounded to one decimal place and marked with `killed_est = True`. This value is never treated as an observed count in the documentation or interface.

### 4.4 Processed outputs

Running `python clean.py` creates or refreshes:

```text
data/processed/station_master.csv
 data/processed/station_year_panel.csv
 data/processed/city_master.csv
 data/processed/crash_ml_dataset.csv
 data/processed/intervention_candidates.csv
 data/processed/data_dictionary.csv
 data/processed/validation_summary.txt
```

The pipeline reports annual reconciliation against city totals, station counts, missing metadata, estimated rows, and duplicate station-year rows.

## 5. Statistical risk engine

The statistical engine is implemented in `risk.py`.

For each station, the engine calculates the recent average over 2023-2025:

```text
killed_recent = mean(killed_2023, killed_2024, killed_2025)
crashes_recent = mean(total_2023, total_2024, total_2025)
fatal_share = killed_recent / sum(all station killed_recent)
fatal_rate = killed_recent / crashes_recent
```

The trend is an ordinary least-squares slope of annual killed values, expressed as a percentage of the station mean per year. At least four non-null years are required. Stations with insufficient history receive the median clipped trend so that missing history is not silently interpreted as either safe or dangerous.

Each component is min-max scaled across stations:

```text
z(x) = (x - minimum) / (maximum - minimum)
```

The default score is:

```text
statistical risk = 100 * (
    0.40 * z(fatal_share)
  + 0.30 * z(fatal_rate)
  + 0.30 * z(trend)
)
```

The sidebar allows the user to change these three weights. They are normalised to sum to one. The resulting score is a relative composite decision index, not a probability of a crash.

Bands are:

```text
0-25       Low
25-50      Medium
50-75      High
75-100     Critical
```

The WHY engine retains the score decomposition and explains burden, severity, and trend in plain language.

## 6. Forecast engine

`forecast.py` produces a modeled baseline for 2026-2028.

The default process:

1. Read annual killed values.
2. Exclude 2020 and 2021 when enough other observations remain.
3. Fit an OLS line to the remaining annual values.
4. Project the line into 2026, 2027, and 2028.
5. Clamp projected counts below zero to zero.
6. Calculate a residual-based uncertainty band.

This is a simple baseline for scenario arithmetic. It is not presented as a validated prediction and does not claim forecast certainty.

## 7. Crash anatomy and WHEN evidence

KSP station profiles are used where station-specific data exists. The application can show pedestrian, hit-and-run, highway, junction, uncontrolled-location, collision-type, and road-type patterns.

When station-specific temporal data is unavailable, the application uses city-level BTP profile shares. These are explicitly described as city-wide and are not relabeled as station observations.

## 8. Machine-learning model

### 8.1 Purpose

The ML model answers:

> Given the characteristics of a recorded crash, how likely is its severity label to be fatal?

It does not predict whether a crash will occur, and it does not estimate the causal effect of an intervention.

### 8.2 Target

The target is generated from the actual KSP severity column:

```text
fatal = 1 if severity, after trimming and case-folding, equals "fatal"
fatal = 0 otherwise
```

The data contains 1,827 fatal records out of 10,535 total records.

### 8.3 Features

Features are built in `ml_model.py` from fields that exist in the data:

- year
- latitude
- longitude
- station
- zone
- subdivision
- collision
- road type
- prior station crash count
- prior station fatality rate

Categorical fields are one-hot encoded. Prior station features are calculated using only rows earlier in the ordered station history, so the current crash target is excluded from its own historical rate.

### 8.4 Temporal evaluation

The data is split by time rather than randomly:

```text
Training: 2016-2021
Testing:  2022-2023
```

This represents the intended deployment direction: learn from earlier crashes and evaluate on later crashes.

### 8.5 Model selection and persistence

`train_model.py` prefers `XGBClassifier`. If XGBoost cannot be imported, the code falls back to a balanced `RandomForestClassifier`.

Running:

```powershell
python train_model.py
```

creates:

```text
models/fatality_risk_model.pkl
models/model_metadata.json
data/processed/crash_ml_predictions.csv
data/processed/station_ml_risk.csv
```

The Streamlit app loads the saved bundle. It does not retrain on every page refresh.

### 8.6 Current test metrics

The saved model was trained on 2016-2021 and tested on 2022-2023:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.604 |
| PR-AUC / Average Precision | 0.252 |
| Precision at 0.50 | 0.230 |
| Recall at 0.50 | 0.103 |
| F1 at 0.50 | 0.142 |

The confusion matrix at threshold 0.50 is:

```text
                  Predicted non-fatal   Predicted fatal
Actual non-fatal          2451                 231
Actual fatal               603                  69
```

These metrics indicate useful but limited predictive signal. The model should support prioritisation and investigation. It should not be used as an automatic enforcement or dispatch decision.

## 9. SHAP explainability

`explain.py` uses SHAP TreeExplainer for the persisted tree model.

### Global explanation

The ML page calculates mean absolute SHAP values across a sample of crash records and ranks features by their average contribution magnitude. This describes which features most influence model output across the sample.

### Local explanation

For a selected station, the page chooses a crash record and displays:

- predicted fatality probability
- strongest positive or negative feature contributions
- encoded feature values

The wording is deliberately associative:

> The model associates this feature with higher or lower predicted severity for this observation.

SHAP does not prove that a collision type, road type, coordinate, or station causes fatality.

If SHAP fails to initialise, the application uses model-native feature importance as a labeled fallback. If the saved model or `joblib` is unavailable, the statistical risk workflow remains usable.

## 10. ML station risk and composite decision score

Crash-level predicted probabilities are aggregated by station:

```text
mean_predicted_fatality_probability
median_predicted_fatality_probability
high_risk_crash_share = share with probability >= 0.50
```

The mean probability is min-max scaled across stations into an ML Severity Risk Index from 0 to 100. It is a relative severity index, not a station accident probability.

The application displays two separate signals:

```text
Statistical Risk       historical burden, severity, and trend
ML Severity Risk       crash-level modeled severity pattern
```

When ML artifacts are available, the composite decision score is:

```text
composite decision risk = 0.50 * statistical risk + 0.50 * ML severity risk
```

The composite is a prioritisation index. It is not a calibrated probability.

## 11. ML-informed intervention relevance

The ML model does not invent intervention effects. Its role is to adjust the relevance of an existing intervention candidate based on observed crash anatomy and station-level predicted severity.

The transparent mapping in `intervention_relevance()` uses signals such as:

| Signal | More relevant intervention themes |
|---|---|
| pedestrian fatal share | pedestrian infrastructure |
| non-pedestrian / two-wheeler proxy | helmet compliance |
| highway fatal share | speed management and heavy-vehicle management |
| junction fatal share | black-spot or junction redesign |
| hit-and-run fatal share | hit-and-run deterrence |
| high modeled severity | reinforces attention to severity-oriented actions |

The adjustment is bounded to a small relevance multiplier. It changes ranking/potential prioritisation only. It does not change the literature effect size into a causal estimate.

## 12. Causal and quasi-experimental evidence

`causal.py` is intentionally cautious because no validated intervention log is present.

### Structural-break screen

For each station and each allowable break year, the engine calculates:

```text
pre-break mean
post-break mean
pre-break slope
post-break slope
change percentage
```

The candidate with the largest absolute mean change is retained. It is classified as:

```text
No strong break       absolute change < 20%
Possible break        20% to < 35%
Strong structural break >= 35%
```

This is a trajectory signal, not proof of a treatment effect.

### Matched-control signal

For a candidate break, comparable stations are selected using pre-break information:

- historical pre-break mean level
- pre-break slope

The post-period change for the target is compared with the average change among up to three controls:

```text
quasi-experimental differential = target change - average control change
```

A pre-trend compatibility label is assigned from the gap between target and control pre-period slopes:

```text
Good      slope gap <= 0.5
Moderate  slope gap <= 1.5
Poor      slope gap > 1.5
```

Only Good or Moderate compatibility receives the local quasi-experimental signal label. Poor compatibility remains a structural-break signal. Neither label proves that an intervention caused the change.

### Evidence tiers

```text
Tier 1: verified intervention date, treated location, credible comparison
Tier 2: local quasi-experimental signal
Tier 3: structural-break signal only
Tier 4: external or literature evidence
```

Because there is no intervention log, current intervention effect candidates are Tier 4 literature-derived estimates unless future data promotes them.

## 13. Counterfactual engine

`counterfactual.py` compares intervention options for a station against its modeled baseline.

For each candidate:

```text
baseline = sum of 2026-2028 forecast values
potential reduction = baseline
                     * attributable share
                     * intervention effect estimate
                     * rollout ramp
counterfactual fatalities = max(baseline - potential reduction, 0)
percentage reduction = potential reduction / baseline
cost per potential life = cost / potential reduction
```

The rollout ramp is:

```text
(0.5 + 1 + 1) / 3 = 0.8333
```

It represents a partial first year and full second and third years. The result is labeled potential/modelled impact and is not an observed life-saving count.

For combined interventions, the documented conservative overlap formula is:

```text
combined effect = 1 - product(1 - individual effect)
```

This prevents simple addition from exceeding a 100% reduction and acknowledges overlapping victims or mechanisms.

## 14. Budget optimizer

`optimize.py` treats each station-intervention pair as a discrete 0/1 package.

Objective:

```text
maximise total modeled potential reduction
```

Constraint:

```text
total cost <= selected budget
```

Costs are in crore and discretised to 0.1 crore. Dynamic programming evaluates feasible package combinations exactly at that resolution. It is not a greedy heuristic.

The application supports budgets of:

```text
Rs 5 Cr, Rs 10 Cr, Rs 25 Cr, Rs 50 Cr, Rs 100 Cr
```

The current implementation does not add a separate maximum-interventions-per-station constraint. Each station-intervention row is treated as a selectable package, so users should interpret a multi-action portfolio as a modeled package list and review implementation dependencies before funding. The selected portfolio's total cost is always checked against the budget.

The efficient frontier reruns the optimizer at each supported budget and plots spend against modeled potential impact. Diminishing returns are interpreted from the resulting curve, not asserted as an observed economic law.

## 15. Map behavior

The Command Center map uses a coordinate-based Plotly scatter plot rather than requiring remote map tiles. This avoids the blank grey map failure mode when a basemap cannot load.

Layers:

- station points sized by recent fatality burden
- station color by composite decision risk when ML is available, otherwise statistical risk
- BTP black spots as black `x` markers
- sampled KSP fatal crash points as dark red markers

Station coordinates are approximate centroids. KSP points are shown only where valid coordinates exist. The map is a screening view, not road-segment geometry.

## 16. Streamlit pages

### Command Center

Shows city deaths, high/critical station count, modeled 2026-2028 baseline, selected budget, and modeled portfolio potential impact. It also shows priority intervention rows and the coordinate-based map.

### Risk Intelligence

Shows station statistical risk, recent deaths, fatality rate, composite risk when available, historical trajectory, and score components. The existing WHY explanation is retained.

### ML & Explainability

Shows model type, test period, ROC-AUC, PR-AUC, global SHAP importance, local crash probability, local feature contributions, station ML Severity Risk, and composite risk.

### What Changed

Shows the historical trajectory, detected break, pre/post change, matched controls, trend compatibility, evidence tier, and the warning that no intervention causality is established.

### What Works

Shows intervention candidates, modeled potential impact, cost, source, and evidence tier. Current candidates rely primarily on literature evidence because local intervention outcomes are unavailable.

### Counterfactual Lab

Allows a station and several intervention IDs to be selected. It compares baseline modeled fatalities with counterfactual modeled fatalities and displays cost, potential reduction, percentage reduction, evidence status, and confidence language.

### Budget Optimizer

Shows the selected portfolio, spend, potential impact, and efficient frontier across supported budgets.

### Action Plans

Generates a deterministic station brief from computed values and shows portfolio actions with phase, location, action, owner suggestion, cost, evidence, and monitoring metric. No LLM is required.

### Method & Limits

Explains observed, estimated, modeled, quasi-experimental, and literature-derived values, plus the learning-loop architecture.

## 17. Provenance labels

The product uses these meanings:

| Label | Meaning in RoadSafe |
|---|---|
| OBSERVED | Directly present in a source dataset |
| ESTIMATED | Derived from an explicit documented ratio or imputation |
| MODELED | Produced by forecast, ML, counterfactual, or optimisation logic |
| QUASI-EXPERIMENTAL | A cautious local comparison signal with pre-trend checks |
| LITERATURE | External effect or evidence assumption |

The label belongs next to the claim it qualifies. A modeled potential reduction is not described as guaranteed lives saved.

## 18. Generated artifacts and reproducibility

The main commands are:

```powershell
python clean.py
python train_model.py
python -m unittest discover -s tests -v
streamlit run app.py
```

The model metadata records training period, test period, model type, feature names, metrics, timestamp, data source, and row-count version.

The tests cover:

- statistical score range
- forecast year output
- non-negative counterfactual impact
- conservative combined effects
- optimizer budget constraint
- duplicate station-year detection
- temporal ML split
- ML probability range
- SHAP feature-name alignment
- non-causal evidence wording

## 19. Current limitations

1. No traffic exposure denominator is available by station.
2. Station is an administrative police jurisdiction, not a road segment.
3. Station coordinates are approximate centroids.
4. KSP coordinate coverage and geocoding are incomplete.
5. 2024 and 2025 killed values are estimated.
6. The ML model has limited recall at the default threshold and requires further calibration and rolling validation.
7. The available intervention table contains effect assumptions and costs, not observed local outcomes.
8. No intervention dates or treated/control implementation log is available.
9. Structural breaks and SHAP associations do not establish causality.
10. The optimizer maximises modeled potential impact, not observed fatalities prevented.
11. City-level time shares should not be interpreted as station-specific time patterns.

## 20. Future learning loop

To become a closed-loop intervention learning system, collect for each intervention:

```text
intervention_id
station or road segment
design and implementation date
implementation status
baseline measurement window
post-intervention measurement windows
comparison locations
exposure measures
fatal and severe crash outcomes
relevant enforcement or engineering indicators
```

Then the platform can estimate before/after and matched comparison effects, monitor drift in the ML model, update intervention evidence tiers, and learn which actions work in which local contexts.
