import pathlib
import sys
import unittest

import pandas as pd

ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from causal import build_evidence
from counterfactual import combined_effect, scenarios
from forecast import fit_project
from explain import global_importance, load_bundle
from ml_model import load_crashes, make_features, temporal_split
from optimize import optimize_portfolio
from risk import compute


class CoreTests(unittest.TestCase):
    def test_risk_scores_are_bounded(self):
        scores = compute()[0]["score"]
        self.assertTrue(((scores >= 0) & (scores <= 100)).all())

    def test_forecast_years(self):
        forecast = fit_project(pd.Series({2018: 4, 2019: 5, 2020: 3, 2021: 5, 2022: 6, 2023: 7, 2024: 8, 2025: 9}))
        self.assertEqual(forecast.year.tolist(), [2026, 2027, 2028])

    def test_counterfactual_is_non_negative(self):
        recs = pd.DataFrame({"station": ["A"], "baseline_3yr": [30], "attributable_share": [0.4], "effect_size": [0.2], "cost_cr": [2], "intervention": ["Test"], "id": ["test"]})
        result = scenarios("A", 30, recs)
        self.assertGreaterEqual(result.potential_reduction.iloc[0], 0)
        self.assertAlmostEqual(combined_effect([0.2, 0.3]), 0.44)

    def test_optimizer_respects_budget(self):
        candidates = pd.DataFrame({"station": ["A", "B"], "intervention": ["X", "Y"], "cost_cr": [2, 3], "potential_reduction": [1, 2]})
        portfolio, summary = optimize_portfolio(candidates, 3)
        self.assertLessEqual(summary["cost_cr"], 3)
        self.assertEqual(portfolio.station.tolist(), ["B"])

    def test_station_year_data_has_no_duplicates(self):
        panel = pd.read_csv(ROOT / "data" / "processed" / "station_year_panel.csv")
        self.assertEqual(panel.duplicated(["station", "year"]).sum(), 0)

    def test_evidence_is_explicitly_non_causal(self):
        killed = pd.DataFrame([[1, 2, 3, 8, 9, 10], [1, 2, 3, 3, 3, 3]], index=["A", "B"], columns=range(2018, 2024))
        _, evidence = build_evidence(killed)
        self.assertTrue(evidence.interpretation.str.contains("not proof").all())

    def test_ml_temporal_split_and_probability_range(self):
        bundle, metadata = load_bundle(ROOT)
        self.assertIsNotNone(bundle)
        crashes = load_crashes()
        train, test, cutoff = temporal_split(crashes)
        self.assertLess(train.year.max(), test.year.min())
        features, _ = make_features(crashes, bundle["feature_columns"])
        probabilities = bundle["model"].predict_proba(features)[:, 1]
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())

    def test_shap_feature_names_match(self):
        bundle, _ = load_bundle(ROOT)
        crashes = load_crashes()
        features, _ = make_features(crashes, bundle["feature_columns"])
        importance = global_importance(bundle, features.iloc[:20])
        self.assertEqual(set(importance.feature), set(features.columns))


if __name__ == "__main__":
    unittest.main()
