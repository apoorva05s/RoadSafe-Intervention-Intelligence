"""Conservative scenario arithmetic for intervention comparisons."""
import numpy as np
import pandas as pd


def scenarios(station, baseline, recommendations, overlap=0.85):
    rows = recommendations[recommendations.station == station].copy()
    rows["potential_reduction"] = (rows["baseline_3yr"] * rows["attributable_share"] *
                                   rows["effect_size"] * (0.5 + 1 + 1) / 3)
    rows["counterfactual_fatalities"] = np.maximum(baseline - rows["potential_reduction"], 0)
    rows["percentage_reduction"] = np.where(baseline > 0, rows["potential_reduction"] / baseline, 0)
    rows["cost_per_potential_life"] = np.where(rows["potential_reduction"] > 0,
                                                rows["cost_cr"] / rows["potential_reduction"], np.nan)
    rows["evidence_status"] = "Literature-derived estimate"
    rows["evidence_tier"] = 4
    rows["confidence"] = "Modeled potential; local outcome data unavailable"
    return rows.sort_values("potential_reduction", ascending=False)


def combined_effect(effects):
    """Conservative non-additive overlap model: 1 - product(1 - effect)."""
    effects = np.clip(np.asarray(list(effects), dtype=float), 0, 1)
    return float(1 - np.prod(1 - effects)) if len(effects) else 0.0